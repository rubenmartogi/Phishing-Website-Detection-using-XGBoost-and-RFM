# ══════════════════════════════════════════════════════════════════════
# ██  BAGIAN 4 — FUNGSI UTILITAS UMUM
# ══════════════════════════════════════════════════════════════════════

import ipaddress
import math
import re
from urllib.parse import urljoin, urlparse

import numpy as np
import tldextract

_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)


def clamp01(v):
    return float(min(max(v, 0.0), 1.0))


def safe_avg(*probs):
    """Rata-rata probabilitas yang aman terhadap None."""
    vals = [p for p in probs if p is not None]
    return clamp01(sum(vals) / len(vals)) if vals else 0.5


def to_float(v, d=0.0):
    try:
        x = float(v)
        return d if (np.isnan(x) or np.isinf(x)) else x
    except Exception:
        return d


def entropy(s):
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))


def is_ip(h):
    try:
        ipaddress.ip_address(h)
        return 1
    except Exception:
        return 0


def parse_url(url):
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    p = urlparse(u)
    return u, p, (p.hostname or "").lower(), p.path or ""


def _extract_parts(hostname):
    e = _TLD_EXTRACTOR(hostname or "")
    return (e.subdomain or "").lower(), (e.domain or "").lower(), (e.suffix or "").lower()


def _word_stats(text):
    words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    if not words:
        return 0, 0, 0, 0.0
    lens = [len(w) for w in words]
    return len(words), min(lens), max(lens), float(sum(lens)) / len(lens)


def _same_or_subdomain(h, base):
    h, base = (h or "").lower(), (base or "").lower()
    return h == base or h.endswith("." + base) or base.endswith("." + h)


def _is_unsafe_anchor(href):
    h = (href or "").strip().lower()
    return h in ("", "#") or h.startswith(("javascript:", "mailto:"))


def _is_external_href(href, base_url, base_host):
    h = (href or "").strip()
    if not h or _is_unsafe_anchor(h):
        return False
    host = (urlparse(urljoin(base_url, h)).hostname or "").lower()
    return bool(host) and not _same_or_subdomain(host, base_host)


def is_valid_url(url):
    try:
        _, _, host, _ = parse_url(url)
        if not host:
            return False
        if "." in host:
            return True
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False


def to_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(val) if isinstance(val, (int, float)) else default
