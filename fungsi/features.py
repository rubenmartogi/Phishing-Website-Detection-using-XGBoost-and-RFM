# ══════════════════════════════════════════════════════════════════════
# ██  BAGIAN 5 & 6 — EKSTRAKSI FITUR URL (37) & KONTEN WEB (44)
# ══════════════════════════════════════════════════════════════════════

import re
from functools import lru_cache
from urllib.parse import urlparse

import requests

from fungsi.config import (
    BRANDS, PHISH_HINTS, SHORTENERS, STANDARD_PORTS, SUSPICIOUS_TLD,
    USER_AGENT, WEB_FETCH_TIMEOUT,
)
from fungsi.utils import (
    _extract_parts, _is_external_href, _is_unsafe_anchor, _same_or_subdomain,
    _word_stats, entropy, is_ip, parse_url,
)

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from external_features import enrich_external_features
except ImportError:
    def enrich_external_features(out, base_host):
        pass


# ── BAGIAN 5 — FITUR URL (37 Fitur) ──────────────────────────────────

@lru_cache(maxsize=128)
def extract_url_features(url):
    full, parsed, hostname, path = parse_url(url)
    subdomain, domain, tld = _extract_parts(hostname)
    digits_url  = sum(c.isdigit() for c in full)
    digits_host = sum(c.isdigit() for c in hostname)
    words_raw, shortest_raw, longest_raw, avg_raw = _word_stats(full)
    _, shortest_host, longest_host, avg_host = _word_stats(hostname)
    _, shortest_path, longest_path, avg_path = _word_stats(path)
    random_domain       = 1 if entropy(domain) > 3.5 else 0
    shortening_service  = 1 if any(s in hostname for s in SHORTENERS) else 0
    prefix_suffix       = 1 if "-" in domain else 0
    path_extension      = 1 if "." in path.split("/")[-1] else 0
    nb_redirection      = max(full.lower().count("http") - 1, 0)
    try:
        parsed_port = parsed.port
    except ValueError:
        parsed_port = None
    port_flag           = 1 if (parsed_port is not None and parsed_port not in STANDARD_PORTS) else 0
    tld_last            = tld.split(".")[-1] if tld else ""
    suspicious_tld_flag = 1 if (tld in SUSPICIOUS_TLD or tld_last in SUSPICIOUS_TLD) else 0
    domain_in_brand     = 1 if any(b in domain     for b in BRANDS) else 0
    brand_in_subdomain  = 1 if any(b in subdomain  for b in BRANDS) else 0
    brand_in_path       = 1 if any(b in path.lower() for b in BRANDS) else 0
    statistical_report  = 1 if (suspicious_tld_flag or is_ip(hostname) or full.count("@") >= 1 or random_domain) else 0

    return {
        "length_url":         len(full),
        "length_hostname":    len(hostname),
        "ip":                 is_ip(hostname),
        "nb_dots":            full.count("."),
        "nb_hyphens":         full.count("-"),
        "nb_at":              full.count("@"),
        "nb_qm":              full.count("?"),
        "nb_and":             full.count("&"),
        "nb_eq":              full.count("="),
        "nb_underscore":      full.count("_"),
        "nb_tilde":           full.count("~"),
        "nb_percent":         full.count("%"),
        "nb_slash":           full.count("/"),
        "nb_star":            full.count("*"),
        "nb_colon":           full.count(":"),
        "nb_comma":           full.count(","),
        "nb_semicolumn":      full.count(";"),
        "nb_dollar":          full.count("$"),
        "nb_space":           full.count(" "),
        "nb_www":             1 if "www" in hostname else 0,
        "nb_com":             full.count(".com"),
        "nb_dslash":          full.count("//"),
        "http_in_path":       1 if "http" in path else 0,
        "https_token":        1 if "https" in full.replace("https://", "") else 0,
        "ratio_digits_url":   digits_url / max(len(full), 1),
        "ratio_digits_host":  digits_host / max(len(hostname), 1),
        "punycode":           1 if "xn--" in hostname else 0,
        "port":               port_flag,
        "tld_in_path":        1 if tld and tld in path else 0,
        "tld_in_subdomain":   1 if tld and tld in subdomain else 0,
        "abnormal_subdomain": 1 if ("http" in subdomain or "https" in subdomain) else 0,
        "nb_subdomains":      len(subdomain.split(".")) if subdomain else 0,
        "prefix_suffix":      prefix_suffix,
        "random_domain":      random_domain,
        "shortening_service": shortening_service,
        "path_extension":     path_extension,
        "nb_redirection":     nb_redirection,
        # ─── BATAS 37 FITUR URL — jangan tambah fitur baru di sini ───
    }


# ── BAGIAN 6 — FITUR KONTEN WEB (44 Fitur Tambahan) ─────────────────

@lru_cache(maxsize=128)
def extract_web_content_features(url):
    out = {}
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=WEB_FETCH_TIMEOUT,
            allow_redirects=True,
        )
        html, final_url = resp.text or "", resp.url or url
    except Exception:
        return out

    base_host = (urlparse(final_url).hostname or "").lower()
    out["nb_external_redirection"] = sum(
        1 for r in (resp.history or [])
        if not _same_or_subdomain((urlparse(r.url).hostname or "").lower(), base_host)
    )

    if BeautifulSoup is None:
        return out
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return out

    anchors = [a.get("href", "") for a in soup.find_all("a")]
    total_a = len(anchors)
    ext_a   = sum(1 for h in anchors if _is_external_href(h, final_url, base_host))
    int_a   = max(total_a - ext_a, 0)
    out["nb_hyperlinks"]       = total_a
    out["ratio_intHyperlinks"] = (int_a / total_a) if total_a else 0.0
    out["ratio_extHyperlinks"] = (ext_a / total_a) if total_a else 0.0

    ext_redir = sum(
        1 for h in anchors
        if _is_external_href(h, final_url, base_host)
        and any(k in h.lower() for k in ["redirect=", "redir=", "url=", "next="])
    )
    out["ratio_extRedirection"] = (ext_redir / ext_a) if ext_a else 0.0
    out["ratio_extErrors"]      = 0.0

    css_links = [
        l.get("href", "") for l in soup.find_all("link")
        if "stylesheet" in " ".join((l.get("rel") or [])).lower()
    ]
    out["nb_extCSS"] = sum(1 for h in css_links if _is_external_href(h, final_url, base_host))

    login_form = 0
    for f in soup.find_all("form"):
        has_pw = bool(f.find("input", {"type": re.compile("password", re.I)}))
        has_ext_action = (
            _is_external_href(f.get("action", ""), final_url, base_host) if f.get("action") else False
        )
        if has_pw or has_ext_action:
            login_form = 1
            break
    out["login_form"] = login_form

    favicon_external = 0
    for l in soup.find_all("link"):
        if "icon" in " ".join((l.get("rel") or [])).lower() and _is_external_href(l.get("href", ""), final_url, base_host):
            favicon_external = 1
            break
    out["external_favicon"] = favicon_external

    tag_urls = []
    for t in soup.find_all(["link", "script", "meta"]):
        u = (
            t.get("href", "") if t.name == "link"
            else t.get("src", "") if t.name == "script"
            else ""
        )
        if t.name == "meta":
            m = re.search(r"url=([^;]+)$", (t.get("content", "") or "").lower())
            u = m.group(1).strip() if m else ""
        if u:
            tag_urls.append(u)
    out["links_in_tags"] = (
        (sum(1 for u in tag_urls if _is_external_href(u, final_url, base_host)) / len(tag_urls)) * 100.0
        if tag_urls else 0.0
    )

    media_urls = [
        t.get("src", "") or t.get("data-src", "")
        for t in soup.find_all(["img", "audio", "embed", "source", "video", "track"])
        if t.get("src", "") or t.get("data-src", "")
    ]
    if media_urls:
        ext_m = sum(1 for u in media_urls if _is_external_href(u, final_url, base_host))
        out["ratio_intMedia"] = ((len(media_urls) - ext_m) / len(media_urls)) * 100.0
        out["ratio_extMedia"] = (ext_m / len(media_urls)) * 100.0
    else:
        out["ratio_intMedia"] = out["ratio_extMedia"] = 0.0

    html_lower = html.lower()
    out["iframe"]        = 1 if soup.find("iframe") else 0
    out["popup_window"]  = 1 if "window.open(" in html_lower else 0
    out["onmouseover"]   = 1 if "onmouseover" in html_lower else 0
    out["right_clic"]    = 1 if ("contextmenu" in html_lower or "event.button==2" in html_lower) else 0
    out["safe_anchor"]   = (sum(1 for h in anchors if _is_unsafe_anchor(h)) / total_a) * 100.0 if total_a else 0.0

    title = (soup.title.string.strip().lower() if soup.title and soup.title.string else "")
    out["empty_title"] = 1 if not title else 0
    _, domain, _ = _extract_parts(base_host)
    out["domain_in_title"] = 1 if (domain and title and domain in title) else 0

    page_text = soup.get_text(" ", strip=True).lower()
    out["domain_with_copyright"] = 1 if (
        ("|" in page_text or "copyright" in page_text) and domain and domain in page_text
    ) else 0

    enrich_external_features(out, base_host)
    return out


def _compute_url_extended_features(url):
    """18 fitur URL tambahan — hanya aktif di mode 81, tidak masuk model 37."""
    full, parsed, hostname, path = parse_url(url)
    subdomain, domain, tld = _extract_parts(hostname)
    words_raw, shortest_raw, longest_raw, avg_raw = _word_stats(full)
    _, shortest_host, longest_host, avg_host = _word_stats(hostname)
    _, shortest_path, longest_path, avg_path = _word_stats(path)
    tld_last            = tld.split(".")[-1] if tld else ""
    suspicious_tld_flag = 1 if (tld in SUSPICIOUS_TLD or tld_last in SUSPICIOUS_TLD) else 0
    domain_in_brand     = 1 if any(b in domain     for b in BRANDS) else 0
    brand_in_subdomain  = 1 if any(b in subdomain  for b in BRANDS) else 0
    brand_in_path       = 1 if any(b in path.lower() for b in BRANDS) else 0
    random_domain       = 1 if entropy(domain) > 3.5 else 0
    statistical_report  = 1 if (suspicious_tld_flag or is_ip(hostname) or full.count("@") >= 1 or random_domain) else 0

    return {
        "nb_external_redirection": 0,  # diisi nilai real oleh extract_web_content_features()
        "length_words_raw":   words_raw,
        "char_repeat":        sum(1 for i in range(1, len(full)) if full[i] == full[i - 1]),
        "shortest_words_raw": shortest_raw,
        "shortest_word_host": shortest_host,
        "shortest_word_path": shortest_path,
        "longest_words_raw":  longest_raw,
        "longest_word_host":  longest_host,
        "longest_word_path":  longest_path,
        "avg_words_raw":      avg_raw,
        "avg_word_host":      avg_host,
        "avg_word_path":      avg_path,
        "phish_hints":        sum(1 for k in PHISH_HINTS if k in full.lower()),
        "domain_in_brand":    domain_in_brand,
        "brand_in_subdomain": brand_in_subdomain,
        "brand_in_path":      brand_in_path,
        "suspicious_tld":     suspicious_tld_flag,
        "statistical_report": statistical_report,
    }


def extract_features(url, include_web_content):
    """Dipertahankan untuk kompatibilitas — logika utama sudah dipindah ke model.py."""
    feats = extract_url_features(url)
    if include_web_content:
        feats.update(_compute_url_extended_features(url))
        feats.update(extract_web_content_features(url))
    return feats
