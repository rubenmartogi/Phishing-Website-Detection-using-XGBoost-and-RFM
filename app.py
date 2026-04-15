import math
import ipaddress
import pickle
import re
from urllib.parse import urlparse, urljoin

import numpy as np
import pandas as pd
import requests
import tldextract
from flask import Flask, jsonify, request, send_from_directory

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# KONFIGURASI
FEATURE_COLUMNS_PATH = "feature_columns.txt"
FEATURE_MEDIANS_PATH = "feature_medians.pkl"
RF_MODEL_PATH = "random_forest_model.pkl"
XGB_MODEL_PATH = "xgboost_model.pkl"
META_MODEL_PATH = "rule_lr.pkl"

PHISHING_CLASS_VALUE = 1

# Default mode prediksi
DEFAULT_PREDICT_MODE = "url37"
DEFAULT_USE_PREFILTER = True
DEFAULT_DECISION_MODE = "hybrid_prefilter"

# Prefilter thresholds
PREFILTER_HARD_PHISHING_SCORE = 7
PREFILTER_BLOCK_ON_VI_HIT = True
PREFILTER_PHISHING_MIN_CONF = 0.95

WEB_FETCH_TIMEOUT = 6
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

SUSPICIOUS_TLD = [
    "zip", "xyz", "top", "tk", "ga", "ml", "gq", "cf", "pw", "cc", "club",
    "ws", "biz", "online", "site", "live", "work", "icu", "info",
    "cn", "ru", "loan", "download", "click"
]
STANDARD_PORTS = {21, 22, 23, 80, 443, 445, 1433, 1521, 3306, 3389}
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly"
}
PHISH_HINTS = [
    "login", "verify", "update", "secure", "account", "bank",
    "paypal", "apple", "microsoft", "confirm", "signin", "password"
]
BRANDS = [
    "google", "facebook", "apple", "microsoft", "amazon", "paypal",
    "instagram", "whatsapp", "telegram", "netflix", "github", "linkedin"
]

URL_FEATURES_37 = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and",
    "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon",
    "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash",
    "http_in_path", "https_token", "ratio_digits_url", "ratio_digits_host", "punycode", "port",
    "tld_in_path", "tld_in_subdomain", "abnormal_subdomain", "nb_subdomains", "prefix_suffix",
    "random_domain", "shortening_service", "path_extension", "nb_redirection"
]

WEB_CONTENT_KEYS = {
    "nb_hyperlinks", "ratio_intHyperlinks", "ratio_extHyperlinks", "nb_extCSS",
    "ratio_extRedirection", "ratio_extErrors", "login_form", "external_favicon",
    "links_in_tags", "ratio_intMedia", "ratio_extMedia", "iframe", "popup_window",
    "safe_anchor", "onmouseover", "right_clic", "empty_title", "domain_in_title",
    "domain_with_copyright", "whois_registered_domain", "domain_registration_length",
    "domain_age", "web_traffic", "dns_record", "google_index", "page_rank",
    "statistical_report", "nb_external_redirection"
}

_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)
app = Flask(__name__, static_folder="static")

_RF_MODEL = None
_XGB_MODEL = None
_META_MODEL = None

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def clamp01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def to_float(v, default=0.0) -> float:
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return float(default)
        return x
    except Exception:
        return float(default)


def to_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def normalize_decision_mode(v: str) -> str:
    x = str(v or "").strip().lower()
    aliases = {
        "rf": "rf_only",
        "rf_only": "rf_only",
        "random_forest": "rf_only",
        "xgb": "xgb_only",
        "xgboost": "xgb_only",
        "xgb_only": "xgb_only",
        "ml": "ml_stacking_only",
        "stacking": "ml_stacking_only",
        "ml_stacking_only": "ml_stacking_only",
        "hybrid": "hybrid_prefilter",
        "hybrid_prefilter": "hybrid_prefilter",
        "rule_ml_stacking": "hybrid_prefilter",
    }
    return aliases.get(x, DEFAULT_DECISION_MODE)


def get_feature_columns():
    try:
        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []


def get_feature_medians():
    try:
        with open(FEATURE_MEDIANS_PATH, "rb") as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def parse_url(url: str):
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    parsed = urlparse(u)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    return u, parsed, hostname, path


def is_ip(hostname: str) -> int:
    try:
        ipaddress.ip_address(hostname)
        return 1
    except Exception:
        return 0


def _extract_parts(hostname: str):
    ext = _TLD_EXTRACTOR(hostname or "")
    subdomain = (ext.subdomain or "").lower()
    domain = (ext.domain or "").lower()
    suffix = (ext.suffix or "").lower()
    return subdomain, domain, suffix


def _word_stats(text: str):
    words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    if not words:
        return 0, 0, 0, 0.0
    lens = [len(w) for w in words]
    return len(words), min(lens), max(lens), float(sum(lens)) / len(lens)


def _same_or_subdomain(host: str, base_host: str) -> bool:
    host = (host or "").lower()
    base_host = (base_host or "").lower()
    if not host or not base_host:
        return False
    return host == base_host or host.endswith("." + base_host) or base_host.endswith("." + host)


def _is_unsafe_anchor(href: str) -> bool:
    h = (href or "").strip().lower()
    return (h == "" or h == "#" or h.startswith("javascript:") or h.startswith("mailto:"))


def _is_external_href(href: str, base_url: str, base_host: str) -> bool:
    h = (href or "").strip()
    if not h or _is_unsafe_anchor(h):
        return False
    absolute = urljoin(base_url, h)
    host = (urlparse(absolute).hostname or "").lower()
    if not host:
        return False
    return not _same_or_subdomain(host, base_host)


def is_valid_url(url: str) -> bool:
    try:
        full, _, host, _ = parse_url(url)
        if not host:
            return False
        if "." in host:
            return True
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False


def _class_index(model, class_value, fallback_idx=1):
    classes = list(getattr(model, "classes_", []))
    if class_value in classes:
        return classes.index(class_value)
    if len(classes) == 2:
        return 1 if fallback_idx >= 1 else 0
    return 0


def _proba_for_class(model, X, class_value):
    try:
        probs = model.predict_proba(X)[0]
        idx = _class_index(model, class_value, fallback_idx=1)
        if idx >= len(probs):
            idx = len(probs) - 1
        return float(probs[idx])
    except Exception:
        pred = int(model.predict(X)[0])
        return 1.0 if pred == class_value else 0.0


def _label_to_phishing_flag(raw_label: int) -> int:
    return 1 if int(raw_label) == PHISHING_CLASS_VALUE else 0


def _resolve_include_web_content(cols: list) -> bool:
    return any(c in WEB_CONTENT_KEYS for c in cols)


def _jsonable_classes(model):
    out = []
    if model is None:
        return out
    for c in list(getattr(model, "classes_", [])):
        try:
            out.append(int(c))
        except Exception:
            out.append(str(c))
    return out


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_url_features(url: str) -> dict:
    full, parsed, hostname, path = parse_url(url)
    subdomain, domain, tld = _extract_parts(hostname)

    digits_url = sum(c.isdigit() for c in full)
    digits_host = sum(c.isdigit() for c in hostname)

    words_raw, shortest_raw, longest_raw, avg_raw = _word_stats(full)
    _, shortest_host, longest_host, avg_host = _word_stats(hostname)
    _, shortest_path, longest_path, avg_path = _word_stats(path)

    random_domain = 1 if entropy(domain) > 3.5 else 0
    shortening_service = 1 if any(s in hostname for s in SHORTENERS) else 0
    prefix_suffix = 1 if "-" in domain else 0
    path_extension = 1 if "." in path.split("/")[-1] else 0
    nb_redirection = max(full.lower().count("http") - 1, 0)

    try:
        parsed_port = parsed.port
    except ValueError:
        parsed_port = None
    port_flag = 1 if (parsed_port is not None and parsed_port not in STANDARD_PORTS) else 0

    tld_last_label = tld.split(".")[-1] if tld else ""
    suspicious_tld_flag = 1 if (tld in SUSPICIOUS_TLD or tld_last_label in SUSPICIOUS_TLD) else 0

    domain_in_brand = 1 if any(b in domain for b in BRANDS) else 0
    brand_in_subdomain = 1 if any(b in subdomain for b in BRANDS) else 0
    brand_in_path = 1 if any(b in path.lower() for b in BRANDS) else 0

    statistical_report = 1 if (
        suspicious_tld_flag == 1
        or is_ip(hostname) == 1
        or full.count("@") >= 1
        or random_domain == 1
    ) else 0

    feats = {
        "length_url": len(full),
        "length_hostname": len(hostname),
        "ip": is_ip(hostname),
        "nb_dots": full.count("."),
        "nb_hyphens": full.count("-"),
        "nb_at": full.count("@"),
        "nb_qm": full.count("?"),
        "nb_and": full.count("&"),
        "nb_eq": full.count("="),
        "nb_underscore": full.count("_"),
        "nb_tilde": full.count("~"),
        "nb_percent": full.count("%"),
        "nb_slash": full.count("/"),
        "nb_star": full.count("*"),
        "nb_colon": full.count(":"),
        "nb_comma": full.count(","),
        "nb_semicolumn": full.count(";"),
        "nb_dollar": full.count("$"),
        "nb_space": full.count(" "),
        "nb_www": 1 if "www" in hostname else 0,
        "nb_com": full.count(".com"),
        "nb_dslash": full.count("//"),
        "http_in_path": 1 if "http" in path else 0,
        "https_token": 1 if "https" in full.replace("https://", "") else 0,
        "ratio_digits_url": digits_url / max(len(full), 1),
        "ratio_digits_host": digits_host / max(len(hostname), 1),
        "punycode": 1 if "xn--" in hostname else 0,
        "port": port_flag,
        "tld_in_path": 1 if tld and (tld in path) else 0,
        "tld_in_subdomain": 1 if tld and (tld in subdomain) else 0,
        "abnormal_subdomain": 1 if ("http" in subdomain or "https" in subdomain) else 0,
        "nb_subdomains": len(subdomain.split(".")) if subdomain else 0,
        "prefix_suffix": prefix_suffix,
        "random_domain": random_domain,
        "shortening_service": shortening_service,
        "path_extension": path_extension,
        "nb_redirection": nb_redirection,
        "nb_external_redirection": 0,
        "length_words_raw": words_raw,
        "char_repeat": sum(1 for i in range(1, len(full)) if full[i] == full[i - 1]),
        "shortest_words_raw": shortest_raw,
        "shortest_word_host": shortest_host,
        "shortest_word_path": shortest_path,
        "longest_words_raw": longest_raw,
        "longest_word_host": longest_host,
        "longest_word_path": longest_path,
        "avg_words_raw": avg_raw,
        "avg_word_host": avg_host,
        "avg_word_path": avg_path,
        "phish_hints": sum(1 for k in PHISH_HINTS if k in full.lower()),
        "domain_in_brand": domain_in_brand,
        "brand_in_subdomain": brand_in_subdomain,
        "brand_in_path": brand_in_path,
        "suspicious_tld": suspicious_tld_flag,
        "statistical_report": statistical_report,
    }
    return feats


def extract_web_content_features(url: str) -> dict:
    out = {}
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=WEB_FETCH_TIMEOUT,
            allow_redirects=True
        )
        html = resp.text or ""
        final_url = resp.url or url
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
    ext_a = sum(1 for h in anchors if _is_external_href(h, final_url, base_host))
    int_a = max(total_a - ext_a, 0)

    out["nb_hyperlinks"] = total_a
    out["ratio_intHyperlinks"] = (int_a / total_a) if total_a else 0.0
    out["ratio_extHyperlinks"] = (ext_a / total_a) if total_a else 0.0

    ext_redir = 0
    for h in anchors:
        if not _is_external_href(h, final_url, base_host):
            continue
        hl = h.lower()
        if any(k in hl for k in ["redirect=", "redir=", "url=", "next="]):
            ext_redir += 1
    out["ratio_extRedirection"] = (ext_redir / ext_a) if ext_a else 0.0
    out["ratio_extErrors"] = 0.0

    css_links = [
        l.get("href", "")
        for l in soup.find_all("link")
        if "stylesheet" in " ".join((l.get("rel") or [])).lower()
    ]
    out["nb_extCSS"] = sum(1 for h in css_links if _is_external_href(h, final_url, base_host))

    login_form = 0
    for f in soup.find_all("form"):
        has_pwd = bool(f.find("input", {"type": re.compile("password", re.I)}))
        action = f.get("action", "")
        external_action = _is_external_href(action, final_url, base_host) if action else False
        if has_pwd or external_action:
            login_form = 1
            break
    out["login_form"] = login_form

    favicon_external = 0
    for l in soup.find_all("link"):
        rel = " ".join((l.get("rel") or [])).lower()
        if "icon" in rel:
            href = l.get("href", "")
            if _is_external_href(href, final_url, base_host):
                favicon_external = 1
                break
    out["external_favicon"] = favicon_external

    tag_urls = []
    for t in soup.find_all(["link", "script", "meta"]):
        if t.name == "link":
            u = t.get("href", "")
        elif t.name == "script":
            u = t.get("src", "")
        else:
            content = (t.get("content", "") or "").lower()
            m = re.search(r"url=([^;]+)$", content)
            u = m.group(1).strip() if m else ""
        if u:
            tag_urls.append(u)

    if tag_urls:
        ext_tag = sum(1 for u in tag_urls if _is_external_href(u, final_url, base_host))
        out["links_in_tags"] = (ext_tag / len(tag_urls)) * 100.0
    else:
        out["links_in_tags"] = 0.0

    media_urls = []
    for t in soup.find_all(["img", "audio", "embed", "source", "video", "track"]):
        u = t.get("src", "") or t.get("data-src", "")
        if u:
            media_urls.append(u)

    if media_urls:
        ext_m = sum(1 for u in media_urls if _is_external_href(u, final_url, base_host))
        int_m = len(media_urls) - ext_m
        out["ratio_intMedia"] = (int_m / len(media_urls)) * 100.0
        out["ratio_extMedia"] = (ext_m / len(media_urls)) * 100.0
    else:
        out["ratio_intMedia"] = 0.0
        out["ratio_extMedia"] = 0.0

    html_lower = html.lower()
    out["iframe"] = 1 if soup.find("iframe") else 0
    out["popup_window"] = 1 if "window.open(" in html_lower else 0
    out["onmouseover"] = 1 if "onmouseover" in html_lower else 0
    out["right_clic"] = 1 if ("contextmenu" in html_lower or "event.button==2" in html_lower) else 0

    unsafe_anchor = sum(1 for h in anchors if _is_unsafe_anchor(h))
    out["safe_anchor"] = (unsafe_anchor / total_a) * 100.0 if total_a else 0.0

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip().lower()
    out["empty_title"] = 1 if not title else 0

    _, domain, _ = _extract_parts(base_host)
    out["domain_in_title"] = 1 if (domain and title and domain in title) else 0

    page_text = soup.get_text(" ", strip=True).lower()
    has_copyright = ("©" in page_text) or ("copyright" in page_text)
    out["domain_with_copyright"] = 1 if (has_copyright and domain and domain in page_text) else 0

    return out


def extract_features(url: str, include_web_content: bool) -> dict:
    full_url, _, _, _ = parse_url(url)
    feats = extract_url_features(full_url)
    if include_web_content:
        feats.update(extract_web_content_features(full_url))
    return feats


# =========================================================
# RULE-BASED EVALUATION
# =========================================================
def rule_based_eval(url: str, return_detail: bool = False):
    feats = extract_url_features(url)

    very_important = {
        "suspicious_tld": (feats.get("suspicious_tld", 0) == 1),
        "nb_at": (feats.get("nb_at", 0) >= 1),
        "ip": (feats.get("ip", 0) == 1),
        "nb_underscore": (feats.get("nb_underscore", 0) > 3),
    }
    important = {
        "ratio_digits_url": (feats.get("ratio_digits_url", 0) > 0.3),
        "nb_subdomains": (feats.get("nb_subdomains", 0) > 3),
        "nb_percent": (feats.get("nb_percent", 0) > 5),
        "nb_tilde": (feats.get("nb_tilde", 0) >= 1),
        "nb_semicolumn": (feats.get("nb_semicolumn", 0) >= 1),
        "nb_star": (feats.get("nb_star", 0) >= 1),
        "nb_comma": (feats.get("nb_comma", 0) >= 1),
        "random_domain": (feats.get("random_domain", 0) == 1),
    }
    less_important = {
        "length_hostname": (feats.get("length_hostname", 0) > 30),
        "nb_dollar": (feats.get("nb_dollar", 0) >= 1),
        "nb_qm": (feats.get("nb_qm", 0) > 2),
        "nb_colon": (feats.get("nb_colon", 0) > 1),
        "nb_eq": (feats.get("nb_eq", 0) > 8),
        "nb_dots": (feats.get("nb_dots", 0) > 4),
        "nb_slash": (feats.get("nb_slash", 0) > 7),
        "nb_and": (feats.get("nb_and", 0) > 3),
        "nb_hyphens": (feats.get("nb_hyphens", 0) > 3),
        "http_in_path": (feats.get("http_in_path", 0) == 1),
        "https_token": (feats.get("https_token", 0) == 1),
        "port": (feats.get("port", 0) == 1),
        "shortening_service": (feats.get("shortening_service", 0) == 1),
    }

    vi_hits = [k for k, v in very_important.items() if v]
    imp_hits = [k for k, v in important.items() if v]
    less_hits = [k for k, v in less_important.items() if v]

    vi_count = len(vi_hits)
    imp_count = len(imp_hits)
    less_count = len(less_hits)

    risk_score = (3 * vi_count) + (2 * imp_count) + less_count

    # ========================================
    # 3 KATEGORI: Phishing | Suspicious | Benign
    # ========================================
    # Phishing (score >= 7 OR VI >= 1)      → rule_flag = 1 (skip ML)
    # Suspicious (1 <= score < 7)          → rule_flag = 0 (verify with ML)
    # Benign (score = 0)                   → rule_flag = 0 (verify with ML)
    
    if (vi_count >= 1) or (risk_score >= PREFILTER_HARD_PHISHING_SCORE):
        category = "Phishing"
        rule_flag = 1  
    elif risk_score > 0:
        category = "Suspicious"
        rule_flag = 0  
    else:
        category = "Benign"
        rule_flag = 0 

    if return_detail:
        return risk_score, category, rule_flag, {
            "vi_count": vi_count,
            "imp_count": imp_count,
            "less_count": less_count,
            "vi_hits": vi_hits,
            "imp_hits": imp_hits,
            "less_hits": less_hits,
        }

    return risk_score, category, rule_flag


# =========================================================
# MODEL LOADING & PREDICTION
# =========================================================
def load_models():
    global _RF_MODEL, _XGB_MODEL, _META_MODEL

    if _RF_MODEL is None:
        with open(RF_MODEL_PATH, "rb") as f:
            _RF_MODEL = pickle.load(f)

    if _XGB_MODEL is None:
        with open(XGB_MODEL_PATH, "rb") as f:
            _XGB_MODEL = pickle.load(f)

    if _META_MODEL is None:
        try:
            with open(META_MODEL_PATH, "rb") as f:
                _META_MODEL = pickle.load(f)
        except Exception:
            _META_MODEL = None

    return _RF_MODEL, _XGB_MODEL, _META_MODEL


def build_ml_vector(url: str, cols: list, medians: dict, include_web_content: bool):
    feats = extract_features(url, include_web_content=include_web_content)

    # Alias untuk typo lama
    if "suspecious_tld" in cols and "suspecious_tld" not in feats:
        feats["suspecious_tld"] = feats.get("suspicious_tld", 0)
    if "suspicious_tld" in cols and "suspicious_tld" not in feats:
        feats["suspicious_tld"] = feats.get("suspecious_tld", 0)

    vector = []
    for c in cols:
        default_v = medians.get(c, 0.0)
        v = feats.get(c, default_v)
        vector.append(to_float(v, default_v))

    return np.array([vector], dtype=float)


def predict_models(
    url: str,
    cols: list,
    medians: dict,
    include_web_content: bool,
    precomputed_rule_score: float = None,
    precomputed_rule_flag: int = None
):
    rf, xgb, meta = load_models()
    X = build_ml_vector(url, cols, medians, include_web_content)
    features_df = pd.DataFrame(X, columns=cols)

    rf_raw = int(rf.predict(features_df)[0])
    xgb_raw = int(xgb.predict(features_df)[0])

    rf_pred = _label_to_phishing_flag(rf_raw)
    xgb_pred = _label_to_phishing_flag(xgb_raw)

    rf_prob = _proba_for_class(rf, features_df, PHISHING_CLASS_VALUE)
    xgb_prob = _proba_for_class(xgb, features_df, PHISHING_CLASS_VALUE)

    stack_pred, stack_prob, meta_n_in = None, None, None
    if meta is not None:
        try:
            meta_n_in = int(getattr(meta, "n_features_in_", 2))
        except Exception:
            meta_n_in = 2

        rule_score = precomputed_rule_score
        rule_flag = precomputed_rule_flag
        if rule_score is None or rule_flag is None:
            rule_score, _, rule_flag = rule_based_eval(url)

        meta_feats = [rf_prob, xgb_prob]
        if meta_n_in >= 3:
            meta_feats = [rf_prob, xgb_prob, rule_flag]
            if meta_n_in >= 4:
                meta_feats = [rf_prob, xgb_prob, rule_flag, rule_score]

        meta_X = np.array([meta_feats[:meta_n_in]], dtype=float)

        try:
            stack_raw = int(meta.predict(meta_X)[0])
            stack_pred = _label_to_phishing_flag(stack_raw)
            stack_prob = _proba_for_class(meta, meta_X, PHISHING_CLASS_VALUE)
        except Exception:
            stack_pred, stack_prob = None, None

    return {
        "rf_raw": rf_raw,
        "xgb_raw": xgb_raw,
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "rf_prob": rf_prob,
        "xgb_prob": xgb_prob,
        "stack_pred": stack_pred,
        "stack_prob": stack_prob,
        "meta_n_in": meta_n_in,
        "rf_classes": _jsonable_classes(rf),
        "xgb_classes": _jsonable_classes(xgb),
        "meta_classes": _jsonable_classes(meta),
    }


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def index():
    return send_from_directory("Phishing_detection_app", "advanced_hybrid_detector.html")


@app.get("/health")
def health():
    cols = get_feature_columns()
    return jsonify({
        "status": "ok",
        "feature_count": len(cols),
        "pipeline": "single_pipeline(rule_prefilter_phishing_only -> ml_for_suspicious)",
        "supported_decision_modes": [
            "rf_only",
            "xgb_only",
            "ml_stacking_only",
            "hybrid_prefilter"
        ],
        "default_decision_mode": DEFAULT_DECISION_MODE,
        "web_content_auto": _resolve_include_web_content(cols),
        "phishing_class_value": PHISHING_CLASS_VALUE,
        "default_use_prefilter": DEFAULT_USE_PREFILTER,
        "prefilter_hard_phishing_score": PREFILTER_HARD_PHISHING_SCORE,
        "prefilter_block_on_vi_hit": PREFILTER_BLOCK_ON_VI_HIT
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    debug = to_bool(data.get("debug", False), False)
    use_prefilter = to_bool(data.get("use_prefilter", DEFAULT_USE_PREFILTER), DEFAULT_USE_PREFILTER)
    decision_mode = normalize_decision_mode(data.get("decision_mode") or data.get("mode"))

    if not url:
        return jsonify({"error": "URL kosong"}), 400
    if not is_valid_url(url):
        return jsonify({"error": "URL tidak valid. Contoh: https://example.com"}), 400

    cols = get_feature_columns()
    if not cols:
        return jsonify({"error": "feature_columns.txt tidak ditemukan atau kosong"}), 500

    medians = get_feature_medians()
    include_web_content = _resolve_include_web_content(cols)

    # Rule-based dihitung sekali
    risk_score, risk_category, rule_flag, rule_detail = rule_based_eval(url, return_detail=True)
    rule_prob = clamp01(risk_score / 10.0)

    def build_response(
        final_phishing_prob: float,
        decision_source: str,
        model_name: str,
        rf_prob=None,
        xgb_prob=None,
        stack_prob=None,
        web_used=None,
    ):
        p_phish = clamp01(final_phishing_prob)
        p_safe = clamp01(1.0 - p_phish)

        # Keputusan akhir berbasis probabilitas murni dengan cut-off 0.6
        final_label = 1 if p_phish >= 0.6 else 0
        category = "phishing" if final_label == 1 else "benign"
        confidence = p_phish if final_label == 1 else p_safe

        resp = {
            "final_label": final_label,
            "final_phishing_prob": p_phish,
            "final_safe_prob": p_safe,
            "final_threshold": 0.6,
            "decision_mode": decision_mode,
            "decision_source": decision_source,
            "model_name": model_name,
            "rule_flag": rule_flag,
            "model_feature_count": len(cols),
            "web_content_used": include_web_content if web_used is None else bool(web_used),
            "mode": "single_pipeline",
            "rf_prob": rf_prob,
            "xgb_prob": xgb_prob,
            "stack_prob": stack_prob,
            "label": category,
            "category": category,
            "confidence": confidence,
        }

        if debug:
            resp["debug"] = {
                "prefilter_enabled": use_prefilter,
                "decision_mode": decision_mode,
                "phishing_class_value": PHISHING_CLASS_VALUE,
                "rule_prob": rule_prob,
                "risk_score": risk_score,
                "risk_category": risk_category,
                "rule_detail": rule_detail,
            }

        return jsonify(resp)

    # MODE 1: RANDOM FOREST ONLY
    if decision_mode == "rf_only":
        preds = predict_models(
            url=url,
            cols=cols,
            medians=medians,
            include_web_content=include_web_content,
            precomputed_rule_score=risk_score,
            precomputed_rule_flag=rule_flag
        )
        p = clamp01(float(preds.get("rf_prob") or 0.0))
        return build_response(
            final_phishing_prob=p,
            decision_source="rf_only",
            model_name="Random Forest Only",
            rf_prob=preds.get("rf_prob"),
            xgb_prob=None,
            stack_prob=None
        )

    # MODE 2: XGB ONLY
    if decision_mode == "xgb_only":
        preds = predict_models(
            url=url,
            cols=cols,
            medians=medians,
            include_web_content=include_web_content,
            precomputed_rule_score=risk_score,
            precomputed_rule_flag=rule_flag
        )
        p = clamp01(float(preds.get("xgb_prob") or 0.0))
        return build_response(
            final_phishing_prob=p,
            decision_source="xgb_only",
            model_name="XGBoost Only",
            rf_prob=None,
            xgb_prob=preds.get("xgb_prob"),
            stack_prob=None
        )

    # MODE 3: ML + STACKING ONLY (tanpa prefilter)
    if decision_mode == "ml_stacking_only":
        preds = predict_models(
            url=url,
            cols=cols,
            medians=medians,
            include_web_content=include_web_content,
            precomputed_rule_score=risk_score,
            precomputed_rule_flag=rule_flag
        )

        if preds.get("stack_prob") is not None:
            p = clamp01(float(preds["stack_prob"]))
            source = "ml_stacking_only"
            model_name = "RF + XGB + Logistic Regression Stacking"
        else:
            rf_p = float(preds.get("rf_prob") or 0.0)
            xgb_p = float(preds.get("xgb_prob") or 0.0)
            p = clamp01(0.5 * rf_p + 0.5 * xgb_p)
            source = "ml_rf_xgb_average_only"
            model_name = "RF + XGB Average (Meta model unavailable)"

        return build_response(
            final_phishing_prob=p,
            decision_source=source,
            model_name=model_name,
            rf_prob=preds.get("rf_prob"),
            xgb_prob=preds.get("xgb_prob"),
            stack_prob=preds.get("stack_prob")
        )

    # MODE 4: HYBRID PREFILTER
    # Aturan final:
    # - Rule category = Phishing => hard block (langsung final phishing)
    # - Rule category = Suspicious => lanjut ML
    if use_prefilter and rule_flag == 1:
        p = clamp01(max(PREFILTER_PHISHING_MIN_CONF, rule_prob))
        return build_response(
            final_phishing_prob=p,
            decision_source="rule_based_prefilter_phishing",
            model_name="Rule-Based Prefilter",
            web_used=False
        )

    # Suspicious => lanjut ke ML
    preds = predict_models(
        url=url,
        cols=cols,
        medians=medians,
        include_web_content=include_web_content,
        precomputed_rule_score=risk_score,
        precomputed_rule_flag=rule_flag
    )

    if preds.get("stack_prob") is not None:
        p = clamp01(float(preds["stack_prob"]))
        source = "ml_stacking_after_rule_suspicious"
        model_name = "RF + XGB + Logistic Regression Stacking"
    else:
        rf_p = float(preds.get("rf_prob") or 0.0)
        xgb_p = float(preds.get("xgb_prob") or 0.0)
        p = clamp01(0.5 * rf_p + 0.5 * xgb_p)
        source = "ml_rf_xgb_after_rule_suspicious"
        model_name = "RF + XGB Average (Meta model unavailable)"

    return build_response(
        final_phishing_prob=p,
        decision_source=source,
        model_name=model_name,
        rf_prob=preds.get("rf_prob"),
        xgb_prob=preds.get("xgb_prob"),
        stack_prob=preds.get("stack_prob")
    )


# Alias endpoint lama
@app.post("/predict_url")
@app.post("/analyze")
def predict_alias():
    return predict()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)