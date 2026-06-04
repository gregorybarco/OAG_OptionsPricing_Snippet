# contract_selector.py
# Deterministic contract selection from a parsed Schwab option chain.
# Bucket-based term structure selection with per-bucket moneyness grids.
#
# NOTE: This file has been scrubbed for public release as a code sample
# submitted in support of OAG application RAD_NYC_DAT_6444. Internal
# project paths, credential references, and pipeline wiring have been
# removed. The selection logic, bucket architecture, and moneyness
# grid mapping are intact and representative of the production code.
#
# DESIGN:
#   settings['dte_buckets'] defines N DTE windows, each with its own
#   moneyness_grid. For each bucket, one expiry is selected: the available
#   expiry whose DTE is closest to the bucket midpoint. Within that expiry,
#   moneyness targets map to the nearest available common call+put strike.
#
#   Short-dated buckets use a tight grid around ATM where options have
#   meaningful prices and vega. Long-dated buckets use a wide grid to
#   capture term structure dynamics in the wings.
#
#   Selection is deterministic and reproducible: same chain, same settings,
#   same result every time.
#
# INPUT:  parsed chain dict from market_data.get_parsed_option_chain
# OUTPUT: list of per-expiry dicts consumed by the calibration engine


def select_contracts(chain, settings):
    """
    Selects one contract set per DTE bucket from a parsed option chain.

    chain    : dict from market_data.get_parsed_option_chain
               keys: 'spot', 'q_div_yld', 'interest_rate', 'calls', 'puts'
    settings : contract selection settings dict
               keys:
                 dte_buckets : list of dicts, each with:
                     dte_min        : int
                     dte_max        : int
                     moneyness_grid : list of floats (fractions of spot)
                     n_paths        : int, passed through to contract_data

    For each bucket, the expiry with DTE closest to the bucket midpoint
    is selected. Buckets with no available expiry are skipped with a print.

    Returns list of dicts, one per selected expiry, in ascending DTE order:
        {
            'expiry'       : str ('YYYY-MM-DD'),
            'dte'          : int,
            'contract_data': {
                's0'              : float,
                'd_to_expiry'     : float,
                'q_div_yld'       : float,
                'r_free_benchmark': float,
                'initial_state'   : None,
                'n_paths'         : int,
            },
            'market_data'  : [
                {
                    'strike'    : float,
                    'call_price': float,
                    'put_price' : float,
                    'iv_call'   : float or None,
                    'iv_put'    : float or None,
                },
                ...
            ],
        }
    """
    spot          = chain['spot']
    q_div_yld     = chain['q_div_yld']
    interest_rate = chain['interest_rate']
    calls         = chain['calls']
    puts          = chain['puts']
    dte_buckets   = settings['dte_buckets']

    # Build full list of (dte, expiry) pairs present in both call and put sides
    common_expiries = sorted(set(calls.keys()) & set(puts.keys()))
    all_with_dte = []
    for expiry in common_expiries:
        call_strikes = calls[expiry]
        if not call_strikes:
            continue
        sample = next(iter(call_strikes))
        dte = call_strikes[sample].get('dte')
        if dte is None:
            continue
        all_with_dte.append((dte, expiry))
    all_with_dte.sort()

    print('[contract_selector] %d buckets. Selecting one expiry per bucket.' % len(dte_buckets))

    result = []
    for bucket in dte_buckets:
        bmin  = bucket['dte_min']
        bmax  = bucket['dte_max']
        bgrid = bucket['moneyness_grid']
        bmid  = (bmin + bmax) / 2.0

        # Candidates within this bucket's DTE range
        candidates = [(dte, exp) for dte, exp in all_with_dte if bmin <= dte <= bmax]

        if not candidates:
            print('[contract_selector]   Bucket [%d-%d DTE]: no expiries available -- skip.'
                  % (bmin, bmax))
            continue

        # Pick the expiry with DTE closest to the bucket midpoint
        dte, expiry = min(candidates, key=lambda x: abs(x[0] - bmid))

        # Strikes present on both call and put sides for this expiry
        common_strikes = sorted(
            set(calls[expiry].keys()) & set(puts[expiry].keys())
        )
        if not common_strikes:
            print('[contract_selector]   Bucket [%d-%d DTE] -> %s (DTE=%d): '
                  'no common call+put strikes -- skip.' % (bmin, bmax, expiry, dte))
            continue

        selected_strikes = _nearest_strikes(common_strikes, spot, bgrid)

        market_data = []
        for strike in selected_strikes:
            call_row  = calls[expiry][strike]
            put_row   = puts[expiry][strike]
            call_mark = call_row.get('mark')
            put_mark  = put_row.get('mark')

            if call_mark is None or put_mark is None:
                continue

            market_data.append({
                'strike'    : float(strike),
                'call_price': float(call_mark),
                'put_price' : float(put_mark),
                'iv_call'   : _clean_iv(call_row.get('iv')),
                'iv_put'    : _clean_iv(put_row.get('iv')),
            })

        if not market_data:
            print('[contract_selector]   Bucket [%d-%d DTE] -> %s (DTE=%d): '
                  'all contracts filtered -- skip.' % (bmin, bmax, expiry, dte))
            continue

        strikes_str = ', '.join('%.2f' % d['strike'] for d in market_data)
        print('[contract_selector]   Bucket [%d-%d DTE] -> %s (DTE=%d): '
              '%d contracts  strikes: [%s]'
              % (bmin, bmax, expiry, dte, len(market_data), strikes_str))

        # Per-contract detail table.
        # Bid/ask are live market quotes. Mark = (bid+ask)/2, used by the
        # calibration engine. IV is the Black-Scholes implied vol in percent.
        hdr = ('%-10s  %-6s  %-8s %-8s %-8s  %-8s %-8s %-8s  %-7s %-7s'
               % ('Strike', 'Mness',
                  'C-Bid', 'C-Ask', 'C-Mark',
                  'P-Bid', 'P-Ask', 'P-Mark',
                  'C-IV%', 'P-IV%'))
        print('[contract_selector]     %s' % hdr)
        for d in market_data:
            k        = d['strike']
            call_row = calls[expiry][k]
            put_row  = puts[expiry][k]
            c_bid    = call_row.get('bid')  or 0.0
            c_ask    = call_row.get('ask')  or 0.0
            p_bid    = put_row.get('bid')   or 0.0
            p_ask    = put_row.get('ask')   or 0.0
            c_iv     = d['iv_call']
            p_iv     = d['iv_put']
            c_iv_str = ('%.2f' % (c_iv * 100.0)) if c_iv else '  --  '
            p_iv_str = ('%.2f' % (p_iv * 100.0)) if p_iv else '  --  '
            print('[contract_selector]     %-10.2f  %-6.3f  %-8.2f %-8.2f %-8.2f  %-8.2f %-8.2f %-8.2f  %-7s %-7s'
                  % (k, k / spot,
                     c_bid, c_ask, d['call_price'],
                     p_bid, p_ask, d['put_price'],
                     c_iv_str, p_iv_str))

        result.append({
            'expiry': expiry,
            'dte'   : dte,
            'contract_data': {
                's0'              : float(spot),
                'd_to_expiry'     : float(dte),
                'q_div_yld'       : float(q_div_yld),
                'r_free_benchmark': float(interest_rate),
                'initial_state'   : None,
                'n_paths'         : bucket.get('n_paths', 1_000_000),
            },
            'market_data': market_data,
        })

    result.sort(key=lambda x: x['dte'])

    print('[contract_selector] %d expiries selected across %d buckets.'
          % (len(result), len(dte_buckets)))
    return result


# Internal helpers

def _nearest_strikes(available_strikes, spot, moneyness_grid):
    """
    Maps each moneyness target (fraction of spot) to the nearest available strike.
    Deduplicates and returns a sorted list.
    """
    targets  = [spot * m for m in moneyness_grid]
    selected = set()
    for target in targets:
        nearest = min(available_strikes, key=lambda k: abs(k - target))
        selected.add(nearest)
    return sorted(selected)


def _clean_iv(iv_raw):
    """
    Converts a Schwab IV field to a decimal float, or None if invalid.
    Schwab returns IV in percent. Sentinel values: -999, 0, None.
    """
    if iv_raw is None:
        return None
    iv = float(iv_raw)
    if iv <= 0.0 or abs(iv - (-999.0)) < 1.0:
        return None
    return iv / 100.0
