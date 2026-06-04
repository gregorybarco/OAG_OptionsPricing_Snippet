# market_data.py
# Schwab API market data fetch functions.
# Returns clean, typed Python structures. No hardcoded tickers or parameters.
#
# NOTE: This file has been scrubbed for public release as a code sample
# submitted in support of OAG application RAD_NYC_DAT_6444. Internal
# project paths, credential references, and pipeline wiring have been
# removed. The ETL logic, API parsing, and bucket-merge architecture
# are intact and representative of the production code.

import datetime
import time
import httpx
import pandas as pd


# SECTION 1 -- Price data

def get_daily_ohlcv(client, ticker):
    """
    Returns daily OHLCV history as a DatetimeIndex DataFrame.
    Columns: open, high, low, close, volume

    Uses the Schwab client returned by auth.get_client().
    """
    response = client.get_price_history_every_day(
        ticker, need_extended_hours_data=False)
    assert response.status_code == 200, \
        '[market_data] get_daily_ohlcv: API error %d for %s' % (response.status_code, ticker)

    candles = response.json()['candles']
    df = pd.DataFrame(candles)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df = df.set_index('datetime')
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print('[market_data] [%s] Daily OHLCV: %s to %s  (%d rows)'
          % (ticker,
             str(df.index[0].date()),
             str(df.index[-1].date()),
             len(df)))
    return df


def get_spot_price(client, ticker):
    """Returns the last traded price of ticker as a float."""
    response = client.get_quote(ticker)
    assert response.status_code == 200, \
        '[market_data] get_spot_price: API error %d for %s' % (response.status_code, ticker)
    return float(response.json()[ticker]['quote']['lastPrice'])


# SECTION 2 -- Option chain

def get_parsed_option_chain(client, ticker, settings):
    """
    Pulls and parses the live Schwab option chain for ticker.

    settings keys used:
        dte_min : int  minimum days to expiration for the API pull
        dte_max : int  maximum days to expiration for the API pull

    Retries up to 3 times on non-200 responses.

    Returns dict:
        'spot'         : float  underlying last price
        'q_div_yld'    : float  dividend yield in decimal (e.g. 0.0116)
        'interest_rate': float  benchmark interest rate in decimal
        'calls'        : {expiry_str: {strike_float: {field: value, ...}}}
        'puts'         : {expiry_str: {strike_float: {field: value, ...}}}

    The calls/puts dicts are keyed by expiry date string ('YYYY-MM-DD')
    then by strike as a float. Each contract dict contains the fields
    listed in _parse_exp_date_map.
    """
    from_date = datetime.date.today() + datetime.timedelta(days=settings['dte_min'])
    to_date   = datetime.date.today() + datetime.timedelta(days=settings['dte_max'])

    for attempt in range(3):
        try:
            response = client.get_option_chain(
                ticker,
                from_date=from_date,
                to_date=to_date,
            )
            if response.status_code == 200:
                break
            print('[market_data] [%s] Chain attempt %d failed (HTTP %d) -- retrying.'
                  % (ticker, attempt + 1, response.status_code))
            time.sleep(2)
        except httpx.ReadTimeout:
            print('[market_data] [%s] Chain attempt %d read timeout -- retrying.'
                  % (ticker, attempt + 1))
            time.sleep(3)
            if attempt == 2:
                raise

    assert response.status_code == 200, \
        '[market_data] get_parsed_option_chain: API error %d for %s' \
        % (response.status_code, ticker)

    raw = response.json()

    # Schwab places underlying metadata in raw['underlying'], not top-level.
    # Top-level mirrors a few fields but NOT dividendYield (only in underlying).
    underlying = raw.get('underlying', {}) or {}

    spot = (underlying.get('last')
            or underlying.get('mark')
            or raw.get('underlyingPrice', 0.0))

    interest_rate = (underlying.get('interestRate')
                     if 'interestRate' in underlying
                     else raw.get('interestRate', 0.0))

    # Prefer dividendYield (percent) over dividendAmount (annual cash $).
    q_div_raw = underlying.get('dividendAmount') or 0.0
    if 'dividendYield' in underlying:
        q_div_raw = underlying['dividendYield']

    # Schwab returns these in percent on some endpoints; convert to decimal.
    q_div_yld     = q_div_raw    / 100.0 if q_div_raw    > 1.0 else float(q_div_raw)
    interest_rate = interest_rate / 100.0 if interest_rate > 1.0 else float(interest_rate)

    print('[market_data] [%s] spot=%.4f  q_div=%.4f  r=%.4f  DTE window: %d-%d'
          % (ticker, spot, q_div_yld, interest_rate,
             settings['dte_min'], settings['dte_max']))

    calls = _parse_exp_date_map(raw.get('callExpDateMap', {}))
    puts  = _parse_exp_date_map(raw.get('putExpDateMap', {}))

    return {
        'spot'         : float(spot),
        'q_div_yld'    : float(q_div_yld),
        'interest_rate': float(interest_rate),
        'calls'        : calls,
        'puts'         : puts,
    }


def get_parsed_option_chain_by_buckets(client, ticker, contract_selection_settings):
    """
    Pulls the option chain one DTE bucket at a time and merges the results.

    Schwab's API returns HTTP 502 Bad Gateway when the full DTE window for
    large-chain tickers produces a payload too large for the server to return
    in one response. Splitting into per-bucket calls keeps each individual
    response within the server's size limit.

    For each bucket in contract_selection_settings['dte_buckets'], one API
    call is made for that bucket's dte_min to dte_max window only. Results
    are merged into one chain dict identical in structure to the dict returned
    by get_parsed_option_chain -- all downstream code is unchanged.

    spot, q_div_yld, and interest_rate are taken from the first bucket
    (they are the same underlying value regardless of expiry window queried).
    calls and puts are merged across all buckets -- no key collisions since
    each bucket covers a distinct date range.

    Returns same structure as get_parsed_option_chain:
        'spot'          : float
        'q_div_yld'     : float
        'interest_rate' : float
        'calls'         : {expiry_str: {strike_float: {field: value}}}
        'puts'          : {expiry_str: {strike_float: {field: value}}}
    """
    buckets = contract_selection_settings['dte_buckets']
    n       = len(buckets)

    merged = {
        'spot'          : None,
        'q_div_yld'     : None,
        'interest_rate' : None,
        'calls'         : {},
        'puts'          : {},
    }

    for i, bucket in enumerate(buckets):
        bucket_settings = {
            'dte_min': bucket['dte_min'],
            'dte_max': bucket['dte_max'],
        }
        bucket_chain = get_parsed_option_chain(client, ticker, bucket_settings)

        # Underlying metadata from the first successful bucket only.
        if merged['spot'] is None:
            merged['spot']          = bucket_chain['spot']
            merged['q_div_yld']     = bucket_chain['q_div_yld']
            merged['interest_rate'] = bucket_chain['interest_rate']

        merged['calls'].update(bucket_chain['calls'])
        merged['puts'].update(bucket_chain['puts'])

        print('[market_data] [%s] Bucket %d/%d merged: DTE %d-%d  (%d call expiries  %d put expiries)'
              % (ticker, i + 1, n,
                 bucket['dte_min'], bucket['dte_max'],
                 len(bucket_chain['calls']),
                 len(bucket_chain['puts'])))

        # Brief pause between bucket calls to stay within Schwab API rate limits.
        if i < n - 1:
            time.sleep(0.5)

    print('[market_data] [%s] Chain merge complete. %d call expiries  %d put expiries total.'
          % (ticker, len(merged['calls']), len(merged['puts'])))
    return merged


def _parse_exp_date_map(exp_date_map):
    """
    Parses a Schwab callExpDateMap or putExpDateMap into:
        {expiry_date_str: {strike_float: {field: value, ...}}}

    expiry_date_str is the date portion of the Schwab key ('YYYY-MM-DD').
    strike_float    is the strike as a Python float.

    Fields per contract:
        bid           ask           mark          last
        iv            (percent, or -999 sentinel if not computed)
        delta         gamma         theta         vega
        dte           open_interest volume
    """
    parsed = {}
    for expiry_key, strikes in exp_date_map.items():
        expiry_date = expiry_key.split(':')[0]
        parsed[expiry_date] = {}
        for strike_key, contracts in strikes.items():
            c      = contracts[0]
            strike = float(strike_key)
            parsed[expiry_date][strike] = {
                'bid'          : c.get('bid',              None),
                'ask'          : c.get('ask',              None),
                'mark'         : c.get('mark',             None),
                'last'         : c.get('last',             None),
                'iv'           : c.get('volatility',       None),
                'delta'        : c.get('delta',            None),
                'gamma'        : c.get('gamma',            None),
                'theta'        : c.get('theta',            None),
                'vega'         : c.get('vega',             None),
                'dte'          : c.get('daysToExpiration', None),
                'open_interest': c.get('openInterest',     None),
                'volume'       : c.get('totalVolume',      None),
            }
    return parsed
