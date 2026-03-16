import math
import ipaddress
import pickle
import re
import pandas as pd
from urllib.parse import urlparse

import numpy as np
import tldextract
from flask import Flask, jsonify, request, send_from_directory

# Konfigurasi
FEATURE_COLUMNS_PATH = "feature_columns.txt"
RF_MODEL_PATH = "random_forest_model.pkl"
XGB_MODEL_PATH = "xgboost_model.pkl"
META_MODEL_PATH = "rule_lr.pkl"

SUSPICIOUS_TLD = [
    "zip", "xyz", "top", "tk", "ga", "ml", "gq", "cf", "pw", "cc", "club", "ws", "biz", "online", "site", "live", "work", "icu", "info",
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

# tldextract: pakai daftar suffix bawaan paket (tanpa fetch internet saat runtime)
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)

app = Flask(__name__, static_folder="static")


# Utils
def entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def clamp01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def get_feature_columns():
    try:
        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            feature_columns = [line.strip() for line in f if line.strip()]
        return feature_columns
    except FileNotFoundError:
        return []


def parse_url(url: str):
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    return url, parsed, hostname, path


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
    suffix = (ext.suffix or "").lower()  # contoh: "com", "co.uk"
    return subdomain, domain, suffix


def tld_of(hostname: str) -> str:
    _, _, suffix = _extract_parts(hostname)
    return suffix


def subdomain_of(hostname: str) -> str:
    subdomain, _, _ = _extract_parts(hostname)
    return subdomain


def abnormal_subdomain(subdomain: str) -> int:
    if not subdomain:
        return 0
    return 1 if ("http" in subdomain or "https" in subdomain) else 0


def _word_stats(text: str):
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    if not words:
        return 0, 0, 0, 0.0
    lengths = [len(w) for w in words]
    return len(words), min(lengths), max(lengths), float(sum(lengths)) / len(lengths)


# Feature Extraction
def extract_features(url: str) -> dict:
    full, parsed, hostname, path = parse_url(url)
    subdomain, domain, tld = _extract_parts(hostname)

    digits_url = sum(c.isdigit() for c in full)
    digits_host = sum(c.isdigit() for c in hostname)

    nb_at = full.count("@")

    # word stats
    words_raw, shortest_raw, longest_raw, avg_raw = _word_stats(full)
    words_host, shortest_host, longest_host, avg_host = _word_stats(hostname)
    words_path, shortest_path, longest_path, avg_path = _word_stats(path)

    # simple heuristics
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

    # dukung suffix multi-part: "co.uk" -> cek juga label terakhir "uk"
    tld_last_label = tld.split(".")[-1] if tld else ""
    suspicious_tld_flag = 1 if (tld in SUSPICIOUS_TLD or tld_last_label in SUSPICIOUS_TLD) else 0

    feats = {
        # === URL‑based features (notebook) ===
        "length_url": len(full),
        "length_hostname": len(hostname),
        "ip": is_ip(hostname),
        "nb_dots": full.count("."),
        "nb_hyphens": full.count("-"),
        "nb_at": nb_at,
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
        "abnormal_subdomain": abnormal_subdomain(subdomain),
        "nb_subdomains": len(subdomain.split(".")) if subdomain else 0,
        "prefix_suffix": prefix_suffix,
        "random_domain": random_domain,
        "shortening_service": shortening_service,
        "path_extension": path_extension,
        "nb_redirection": nb_redirection,

        # === word stats (notebook) ===
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
        "suspicious_tld": suspicious_tld_flag,

        # === content/externals (placeholder sesuai notebook) ===
        "statistical_report": 0,
        "nb_hyperlinks": 0,
        "ratio_intHyperlinks": 0.0,
        "ratio_extHyperlinks": 0.0,
        "nb_extCSS": 0,
        "ratio_extRedirection": 0.0,
        "ratio_intErrors": 0.0,
        "login_form": 0,
        "external_favicon": 0,
        "links_in_tags": 0,
        "ratio_intMedia": 0.0,
        "ratio_extMedia": 0.0,
        "iframe": 0,
        "popup_window": 0,
        "safe_anchor": 0,
        "onmouseover": 0,
        "right_clic": 0,
        "empty_title": 0,
        "domain_in_title": 0,
        "domain_with_copyright": 0,
        "whois_registered_domain": 0,
        "domain_registration_length": -1,
        "domain_age": -1,
        "web_traffic": 0,
        "dns_record": 0,
        "google_index": 0,
        "page_rank": 0,

        # === brand features (placeholder) ===
        "domain_in_brand": 0,
        "brand_in_subdomain": 0,
        "brand_in_path": 0,

        # === external redirection (placeholder) ===
        "nb_external_redirection": 0,
    }
    return feats


# Rule-based
def rule_based_eval(url: str):
    feats = extract_features(url)
    _, _, hostname, _ = parse_url(url)
    subdomain, domain, tld = _extract_parts(hostname)

    ratio_digits_url = feats["ratio_digits_url"]
    nb_subdomains = len(subdomain.split(".")) if subdomain else 0
    random_domain = 1 if entropy(domain) > 3.5 else 0
    shortening_service = 1 if any(s in hostname for s in SHORTENERS) else 0
    suspicious_tld = 1 if (tld in SUSPICIOUS_TLD or (tld.split(".")[-1] if tld else "") in SUSPICIOUS_TLD) else 0

    very_important = {
        "suspicious_tld": (suspicious_tld == 1),
        "nb_at": (feats["nb_at"] >= 1),
        "ip": (feats["ip"] == 1),
        "nb_underscore": (feats["nb_underscore"] > 3),
    }
    important = {
        "ratio_digits_url": (ratio_digits_url > 0.3),
        "nb_subdomains": (nb_subdomains > 3),
        "nb_percent": (feats["nb_percent"] > 5),
        "nb_tilde": (feats["nb_tilde"] >= 1),
        "nb_semicolumn": (feats["nb_semicolumn"] >= 1),
        "nb_star": (feats["nb_star"] >= 1),
        "nb_comma": (feats["nb_comma"] >= 1),
        "random_domain": (random_domain == 1),
    }
    less_important = {
        "length_hostname": (feats["length_hostname"] > 30),
        "nb_dollar": (feats["nb_dollar"] >= 1),
        "nb_qm": (feats["nb_qm"] > 2),
        "nb_colon": (feats["nb_colon"] > 1),
        "nb_eq": (feats["nb_eq"] > 8),
        "nb_dots": (feats["nb_dots"] > 4),
        "nb_slash": (feats["nb_slash"] > 7),
        "nb_and": (feats["nb_and"] > 3),
        "nb_hyphens": (feats["nb_hyphens"] > 3),
        "http_in_path": (feats["http_in_path"] == 1),
        "https_token": (feats["https_token"] == 1),
        "port": (feats["port"] == 1),
        "shortening_service": (shortening_service == 1),
    }

    vi_count = sum(int(v) for v in very_important.values())
    imp_count = sum(int(v) for v in important.values())
    less_count = sum(int(v) for v in less_important.values())

    rule1_flag = (vi_count >= 1)
    risk_score = (3 * vi_count) + (2 * imp_count) + (1 * less_count)
    rule_flag = int(rule1_flag or (risk_score >= 7))

    if rule_flag == 1:
        category = "Phishing"
    elif 4 <= risk_score <= 6:
        category = "Suspicious"
    elif 1 <= risk_score <= 3:
        category = "Caution"
    else:
        category = "Benign"

    return risk_score, category, rule_flag


# Model Loading
def load_models():
    rf = pickle.load(open(RF_MODEL_PATH, "rb"))
    xgb = pickle.load(open(XGB_MODEL_PATH, "rb"))
    meta = None
    try:
        meta = pickle.load(open(META_MODEL_PATH, "rb"))
    except Exception:
        meta = None
    return rf, xgb, meta


def build_ml_vector(url: str):
    feats = extract_features(url)
    cols = get_feature_columns()
    if not cols:
        return None, []
    vector = [feats.get(c, 0) for c in cols]
    return np.array([vector]), cols


def predict_models(url: str):
    rf, xgb, meta = load_models()
    X, cols = build_ml_vector(url)
    if X is None:
        return None

    features_df = pd.DataFrame(X, columns=cols)

    rf_pred = int(rf.predict(features_df)[0])
    xgb_pred = int(xgb.predict(features_df)[0])

    try:
        rf_prob = float(rf.predict_proba(features_df)[0][1])
    except Exception:
        rf_prob = float(rf_pred)

    try:
        xgb_prob = float(xgb.predict_proba(features_df)[0][1])
    except Exception:
        xgb_prob = float(xgb_pred)

    stack_pred = None
    stack_prob = None
    if meta is not None:
        try:
            n_in = getattr(meta, "n_features_in_", 2)
        except Exception:
            n_in = 2

        meta_feats = [rf_prob, xgb_prob]
        if n_in >= 3:
            risk_score, _, rule_flag = rule_based_eval(url)
            meta_feats = [rf_prob, xgb_prob, rule_flag]
            if n_in >= 4:
                meta_feats = [rf_prob, xgb_prob, rule_flag, risk_score]

        meta_X = np.array([meta_feats[:n_in]])
        try:
            stack_pred = int(meta.predict(meta_X)[0])
            stack_prob = float(meta.predict_proba(meta_X)[0][1])
        except Exception:
            stack_pred = None
            stack_prob = None

    return {
        "rf_pred": rf_pred,
        "rf_prob": rf_prob,
        "xgb_pred": xgb_pred,
        "xgb_prob": xgb_prob,
        "stack_pred": stack_pred,
        "stack_prob": stack_prob,
    }


# Routes
@app.get("/")
def index():
    return send_from_directory("Phishing_detection_app", "advanced_hybrid_detector.html")


def is_valid_url(url: str) -> bool:
    url = url.strip()
    # Regex: support userinfo (user@), domain/IP, optional port, path
    pattern = r"^(https?://)?([a-zA-Z0-9\-._~%!$&'()*+,;=:]+@)?((([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,}|(\d{1,3}(\.\d{1,3}){3})))(:\d+)?(/.*)?$"
    return re.match(pattern, url) is not None


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL kosong"}), 400

    if not is_valid_url(url):
        return jsonify({"error": "URL tidak valid. Masukkan URL yang benar (misal: https://example.com)"}), 400

    risk_score, category, rule_flag = rule_based_eval(url)
    rule_prob = clamp01(risk_score / 10.0)

    # TRUE PRE-FILTER: jika rule sudah jelas phishing, stop di sini (tidak jalankan ML)
    if rule_flag == 1:
        final_phishing_prob = rule_prob
        final_safe_prob = clamp01(1.0 - final_phishing_prob)
        return jsonify({
            "final_label": 1,
            "final_phishing_prob": final_phishing_prob,
            "final_safe_prob": final_safe_prob,
            "decision_source": "rule",
            "risk_score": risk_score,
            "risk_category": category,
            "rule_flag": rule_flag
        })

    # Kasus ambigu -> lanjut ML
    preds = predict_models(url)
    if preds is None:
        return jsonify({"error": "feature_columns.txt tidak ditemukan atau kosong"}), 500

    stack_prob = preds.get("stack_prob")
    stack_pred = preds.get("stack_pred")
    xgb_prob = preds.get("xgb_prob")
    rf_prob = preds.get("rf_prob")

    ml_prob = next((p for p in [stack_prob, xgb_prob, rf_prob] if p is not None), 0.0)
    final_phishing_prob = clamp01(float(ml_prob))
    final_safe_prob = clamp01(1.0 - final_phishing_prob)

    if stack_pred is not None:
        final_label = int(stack_pred)
        decision_source = "stacking"
    else:
        final_label = 1 if final_phishing_prob >= 0.5 else 0
        decision_source = "xgb" if xgb_prob is not None else "rf"

    return jsonify({
        "final_label": final_label,
        "final_phishing_prob": final_phishing_prob,
        "final_safe_prob": final_safe_prob,
        "decision_source": decision_source,
        "risk_score": risk_score,
        "risk_category": category,
        "rule_flag": rule_flag
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)