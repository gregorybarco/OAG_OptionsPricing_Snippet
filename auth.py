# auth.py
# Schwab API authentication wrapper.
#
# NOTE: This file has been scrubbed for public release as a code sample
# submitted in support of OAG application RAD_NYC_DAT_6444. Internal
# project paths, credential references, and pipeline wiring have been
# removed. The authentication logic, token lifecycle management, and
# timeout handling are intact and representative of the production code.

import os
import time
import json
import httpx
from pathlib import Path
import schwab

# Credentials are loaded from environment variables.
# SCHWAB_APP_KEY, SCHWAB_APP_SECRET, and SCHWAB_CALLBACK_URL
# must be set in the environment before calling get_client().
_TOKEN_PATH = os.getenv("SCHWAB_TOKEN_PATH", "tokens/schwab_token.json")

# Re-authenticate when the master token is older than 6.5 days.
_RE_AUTH_THRESHOLD = 6.5 * 24 * 60 * 60   # seconds

# Large option chains (broad index tickers) can return several MB of JSON.
# The default httpx read timeout (5s) fires before the response arrives.
# 120s is sufficient for any supported ticker.
_CHAIN_READ_TIMEOUT_S = 120.0


def _read_token_age_seconds(token_path):
    """
    Reads creation_timestamp from the schwab-py token JSON.
    Returns true token age in seconds, or None if unreadable.
    schwab-py writes creation_timestamp (epoch float) at the top level.
    Falls back to file mtime if the field is absent.
    """
    try:
        with open(token_path, 'r') as f:
            data = json.load(f)
        ts = data.get('creation_timestamp')
        if ts is not None:
            return time.time() - float(ts)
    except Exception:
        pass
    return None


def _apply_api_timeout(client):
    """
    Sets a longer read timeout on the httpx session backing the Schwab client.
    Required for large option chains where the JSON payload can be several MB.
    Accesses client.session (httpx.Client) directly. AttributeError is caught
    if the attribute name changes in a future schwab-py version.
    """
    try:
        client.session.timeout = httpx.Timeout(timeout=30.0,
                                                read=_CHAIN_READ_TIMEOUT_S)
    except AttributeError:
        pass
    return client


def get_client():
    """
    Guarded Schwab authentication wrapper.

    Token lifecycle:
      - Silent load and 30-min access token refresh when master token is alive.
      - Server validation ping to confirm session is live.
      - If ping fails within 6.5 days: treat as transient network error,
        preserve token and return client anyway.
      - If ping fails after 6.5 days: token is expired, purge and re-auth
        via browser (schwab-py easy_client opens a browser window).

    Returns an authenticated schwab client object with extended read timeout.
    """
    app_key      = os.getenv("SCHWAB_APP_KEY")
    app_secret   = os.getenv("SCHWAB_APP_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL")

    if os.path.exists(_TOKEN_PATH):
        file_age = time.time() - os.path.getmtime(_TOKEN_PATH)
        true_age = _read_token_age_seconds(_TOKEN_PATH)
        if true_age is None:
            true_age = file_age
        days_old = true_age / (24 * 3600)

        try:
            client = schwab.auth.client_from_token_file(
                _TOKEN_PATH, app_key, app_secret)
            resp   = client.get_quote('SPY')   # lightweight validation ping

            if resp.status_code == 200:
                print('[auth] Token verified. Age: %.2f days.' % days_old)
                return _apply_api_timeout(client)

            raise ValueError('Token rejected by server (status %d).'
                             % resp.status_code)

        except Exception:
            if true_age < _RE_AUTH_THRESHOLD:
                print('[auth] Ping failed but token is only %.2f days old.' % days_old)
                print('[auth] Preserving token -- treating as network/API issue.')
                return _apply_api_timeout(
                    schwab.auth.client_from_token_file(
                        _TOKEN_PATH, app_key, app_secret))

            print('[auth] Token expired (%.2f days). Purging for re-auth.' % days_old)
            time.sleep(1)
            try:
                os.remove(_TOKEN_PATH)
                print('[auth] Stale token purged.')
            except OSError:
                pass

    print('[auth] Launching browser for weekly authentication.')
    client = schwab.auth.easy_client(
        api_key=app_key,
        app_secret=app_secret,
        callback_url=callback_url,
        token_path=_TOKEN_PATH,
    )
    print('[auth] New 7-day token secured.')
    return _apply_api_timeout(client)
