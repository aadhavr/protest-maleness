"""Shared setup for collectors: import path, polite HTTP with retries."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests  # noqa: E402

UA = {"User-Agent": "protest-maleness research pipeline (contact: set PM_CONTACT_EMAIL)"}
if os.environ.get("PM_CONTACT_EMAIL"):
    UA["User-Agent"] = f"protest-maleness research pipeline ({os.environ['PM_CONTACT_EMAIL']})"


def get(url, params=None, headers=None, tries=5, wait=6.0, timeout=60, verify=True):
    """GET with retry on 429/5xx and on network errors; respects Retry-After."""
    h = {**UA, **(headers or {})}
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout, verify=verify)
        except requests.exceptions.SSLError as ex:
            raise RuntimeError(f"SSL error (will not retry): {str(ex)[:160]}")
        except requests.RequestException as ex:
            print(f"  network error ({ex}); retry {i + 1}/{tries}")
            time.sleep(wait * (i + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            ra = r.headers.get("Retry-After")
            time.sleep(float(ra) if ra and ra.isdigit() else wait * (i + 1))
            continue
        return r
    raise RuntimeError(f"GET failed after {tries} tries: {url}")


def need_env(*names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"set environment variables: {', '.join(missing)}")
    return [os.environ[n] for n in names]
