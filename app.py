import ipaddress
import json
import math
import os
import pickle   
import re
from urllib.parse import urljoin, urlparse
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import shap
import tldextract
import platform
from dotenv import load_dotenv
load_dotenv()
from external_features import enrich_external_features
from flask import Flask, jsonify, request, send_from_directory

from llm_utils import get_llm_reasoning

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# ── Konstanta ──────────────────────────────────────────────────────────────────
PHISHING_CLASS_VALUE    = 1
FINAL_THRESHOLD         = 0.6
DEFAULT_USE_PREFILTER   = True
DEFAULT_DECISION_MODE   = "hybrid_prefilter"
PREFILTER_PHISHING_MIN_CONF = 0.95
WEB_FETCH_TIMEOUT       = 6
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

SUSPICIOUS_TLD = ["zip","xyz","top","tk","ga","ml","gq","cf","pw","cc","club","ws","biz","online","site","live","work","icu","info","cn","ru","loan","download","click"]
STANDARD_PORTS = {21,22,23,80,443,445,1433,1521,3306,3389}
SHORTENERS = {"bit.ly","goo.gl","tinyurl.com","ow.ly","t.co","is.gd","buff.ly","adf.ly","bit.do","cutt.ly"}
PHISH_HINTS = ["login","verify","update","secure","account","bank","paypal","apple","microsoft","confirm","signin","password"]
BRANDS = ["google","facebook","apple","microsoft","amazon","paypal","instagram","whatsapp","telegram","netflix","github","linkedin"]

WEB_CONTENT_KEYS = {
    "nb_hyperlinks","ratio_intHyperlinks","ratio_extHyperlinks","nb_extCSS",
    "ratio_extRedirection","ratio_extErrors","login_form","external_favicon",
    "links_in_tags","ratio_intMedia","ratio_extMedia","iframe","popup_window",
    "safe_anchor","onmouseover","right_clic","empty_title","domain_in_title",
    "domain_with_copyright","whois_registered_domain","domain_registration_length",
    "domain_age","web_traffic","dns_record","google_index","page_rank","nb_external_redirection",
}

_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))

# Menggunakan ekstensi CSV murni untuk stabilitas penuh saat dibaca Microsoft Excel
LOG_PATH = os.path.join(BASE_DIR, "log_feature_extraction.csv")

def check_log_locked():
    if not os.path.exists(LOG_PATH):
        return False, ""

    try:
        fh = open(LOG_PATH, "a", newline="", encoding="utf-8-sig")
    except (PermissionError, OSError):
        return True, (
            "File log (log_feature_extraction.csv) sedang terbuka di Excel atau program lain. "
            "Tutup file tersebut terlebih dahulu, lalu ulangi deteksi."
        )
    except Exception:
        return False, ""

    try:
        if platform.system() == "Windows":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            try:
                import fcntl  # type: ignore[import]
            except ImportError:
                return False, ""
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)
        return False, ""
    except (OSError, PermissionError):
        return True, (
            "File log (log_feature_extraction.csv) sedang terbuka di Excel atau program lain. "
            "Tutup file tersebut terlebih dahulu, lalu ulangi deteksi."
        )
    finally:
        fh.close()

import threading as _threading
_log_lock = _threading.Lock()

executor = ThreadPoolExecutor(max_workers=4)
GLOBAL_MODELS = {}

def init_startup_cache():
    print("[STARTUP] Inisialisasi Cache Model...")
    for suffix in ["_37", "_81"]:
        try:
            GLOBAL_MODELS[f"rf{suffix}"] = pickle.load(open(os.path.join(BASE_DIR, f"random_forest_model{suffix}.pkl"), "rb"))
            GLOBAL_MODELS[f"xgb{suffix}"] = pickle.load(open(os.path.join(BASE_DIR, f"xgboost_model{suffix}.pkl"), "rb"))
            meta_path = os.path.join(BASE_DIR, f"rule_lr{suffix}.pkl")
            GLOBAL_MODELS[f"meta{suffix}"] = pickle.load(open(meta_path, "rb")) if os.path.exists(meta_path) else None
            
            col_path = os.path.join(BASE_DIR, f"feature_columns{suffix}.txt")
            if os.path.exists(col_path):
                GLOBAL_MODELS[f"cols{suffix}"] = [l.strip() for l in open(col_path, encoding="utf-8") if l.strip()]
            else:
                GLOBAL_MODELS[f"cols{suffix}"] = []
            print(f"[STARTUP] Cache model {suffix} berhasil dimuat.")
        except Exception as e:
            print(f"[STARTUP WARNING] Gagal memuat cache model {suffix}: {e}")

init_startup_cache()

try:
    FEATURE_COLUMNS_81 = [
        l.strip() for l in open(os.path.join(BASE_DIR, "feature_columns_81.txt"), encoding="utf-8") if l.strip()
    ]
except Exception:
    FEATURE_COLUMNS_81 = []

def app_path(*parts): return os.path.join(BASE_DIR, *parts)
def clamp01(v): return float(min(max(v, 0.0), 1.0))
def safe_avg(*probs):
    """Rata-rata probabilitas yang aman terhadap None."""
    vals = [p for p in probs if p is not None]
    return clamp01(sum(vals) / len(vals)) if vals else 0.5
def to_float(v, d=0.0):
    try:
        x = float(v)
        return d if (np.isnan(x) or np.isinf(x)) else x
    except Exception: return d
def entropy(s):
    if not s: return 0.0
    return -sum((s.count(c)/len(s))*math.log2(s.count(c)/len(s)) for c in set(s))
def is_ip(h):
    try: ipaddress.ip_address(h); return 1
    except: return 0

def parse_url(url):
    u = url.strip()
    if not u.startswith(("http://","https://")): u = "http://" + u
    p = urlparse(u)
    return u, p, (p.hostname or "").lower(), p.path or ""

def _extract_parts(hostname):
    e = _TLD_EXTRACTOR(hostname or "")
    return (e.subdomain or "").lower(), (e.domain or "").lower(), (e.suffix or "").lower()

def _word_stats(text):
    words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    if not words: return 0,0,0,0.0
    lens = [len(w) for w in words]
    return len(words), min(lens), max(lens), float(sum(lens))/len(lens)

def _same_or_subdomain(h, base):
    h, base = (h or "").lower(), (base or "").lower()
    return h==base or h.endswith("."+base) or base.endswith("."+h)

def _is_unsafe_anchor(href):
    h = (href or "").strip().lower()
    return h in ("","#") or h.startswith(("javascript:","mailto:"))

def _is_external_href(href, base_url, base_host):
    h = (href or "").strip()
    if not h or _is_unsafe_anchor(h): return False
    host = (urlparse(urljoin(base_url, h)).hostname or "").lower()
    return bool(host) and not _same_or_subdomain(host, base_host)

def is_valid_url(url):
    try:
        _, _, host, _ = parse_url(url)
        if not host: return False
        if "." in host: return True
        ipaddress.ip_address(host); return True
    except: return False

@lru_cache(maxsize=128)
def extract_url_features(url):
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
    try: parsed_port = parsed.port
    except ValueError: parsed_port = None
    port_flag = 1 if (parsed_port is not None and parsed_port not in STANDARD_PORTS) else 0
    tld_last = tld.split(".")[-1] if tld else ""
    suspicious_tld_flag = 1 if (tld in SUSPICIOUS_TLD or tld_last in SUSPICIOUS_TLD) else 0
    domain_in_brand = 1 if any(b in domain for b in BRANDS) else 0
    brand_in_subdomain = 1 if any(b in subdomain for b in BRANDS) else 0
    brand_in_path = 1 if any(b in path.lower() for b in BRANDS) else 0
    statistical_report = 1 if (suspicious_tld_flag or is_ip(hostname) or full.count("@")>=1 or random_domain) else 0
    return {
        "length_url": len(full), "length_hostname": len(hostname), "ip": is_ip(hostname),
        "nb_dots": full.count("."), "nb_hyphens": full.count("-"), "nb_at": full.count("@"),
        "nb_qm": full.count("?"), "nb_and": full.count("&"), "nb_eq": full.count("="),
        "nb_underscore": full.count("_"), "nb_tilde": full.count("~"), "nb_percent": full.count("%"),
        "nb_slash": full.count("/"), "nb_star": full.count("*"), "nb_colon": full.count(":"),
        "nb_comma": full.count(","), "nb_semicolumn": full.count(";"), "nb_dollar": full.count("$"),
        "nb_space": full.count(" "), "nb_www": 1 if "www" in hostname else 0,
        "nb_com": full.count(".com"), "nb_dslash": full.count("//"),
        "http_in_path": 1 if "http" in path else 0,
        "https_token": 1 if "https" in full.replace("https://","") else 0,
        "ratio_digits_url": digits_url/max(len(full),1),
        "ratio_digits_host": digits_host/max(len(hostname),1),
        "punycode": 1 if "xn--" in hostname else 0, "port": port_flag,
        "tld_in_path": 1 if tld and tld in path else 0,
        "tld_in_subdomain": 1 if tld and tld in subdomain else 0,
        "abnormal_subdomain": 1 if ("http" in subdomain or "https" in subdomain) else 0,
        "nb_subdomains": len(subdomain.split(".")) if subdomain else 0,
        "prefix_suffix": prefix_suffix, "random_domain": random_domain,
        "shortening_service": shortening_service, "path_extension": path_extension,
        "nb_redirection": nb_redirection, "nb_external_redirection": 0,
        "length_words_raw": words_raw,
        "char_repeat": sum(1 for i in range(1,len(full)) if full[i]==full[i-1]),
        "shortest_words_raw": shortest_raw, "shortest_word_host": shortest_host,
        "shortest_word_path": shortest_path, "longest_words_raw": longest_raw,
        "longest_word_host": longest_host, "longest_word_path": longest_path,
        "avg_words_raw": avg_raw, "avg_word_host": avg_host, "avg_word_path": avg_path,
        "phish_hints": sum(1 for k in PHISH_HINTS if k in full.lower()),
        "domain_in_brand": domain_in_brand, "brand_in_subdomain": brand_in_subdomain,
        "brand_in_path": brand_in_path, "suspicious_tld": suspicious_tld_flag,
        "statistical_report": statistical_report,
    }

@lru_cache(maxsize=128)
def extract_web_content_features(url):
    out = {}
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=WEB_FETCH_TIMEOUT, allow_redirects=True)
        html, final_url = resp.text or "", resp.url or url
    except Exception: return out
    base_host = (urlparse(final_url).hostname or "").lower()
    out["nb_external_redirection"] = sum(
        1 for r in (resp.history or [])
        if not _same_or_subdomain((urlparse(r.url).hostname or "").lower(), base_host)
    )
    if BeautifulSoup is None: return out
    try: soup = BeautifulSoup(html, "html.parser")
    except: return out
    anchors = [a.get("href","") for a in soup.find_all("a")]
    total_a = len(anchors)
    ext_a = sum(1 for h in anchors if _is_external_href(h, final_url, base_host))
    int_a = max(total_a - ext_a, 0)
    out["nb_hyperlinks"] = total_a
    out["ratio_intHyperlinks"] = (int_a/total_a) if total_a else 0.0
    out["ratio_extHyperlinks"] = (ext_a/total_a) if total_a else 0.0
    ext_redir = sum(1 for h in anchors if _is_external_href(h,final_url,base_host) and any(k in h.lower() for k in ["redirect=","redir=","url=","next="]))
    out["ratio_extRedirection"] = (ext_redir/ext_a) if ext_a else 0.0
    out["ratio_extErrors"] = 0.0
    css_links = [l.get("href","") for l in soup.find_all("link") if "stylesheet" in " ".join((l.get("rel") or [])).lower()]
    out["nb_extCSS"] = sum(1 for h in css_links if _is_external_href(h, final_url, base_host))
    login_form = 0
    for f in soup.find_all("form"):
        if bool(f.find("input",{"type":re.compile("password",re.I)})) or (_is_external_href(f.get("action",""), final_url, base_host) if f.get("action") else False):
            login_form = 1; break
    out["login_form"] = login_form
    favicon_external = 0
    for l in soup.find_all("link"):
        if "icon" in " ".join((l.get("rel") or [])).lower() and _is_external_href(l.get("href",""), final_url, base_host):
            favicon_external = 1; break
    out["external_favicon"] = favicon_external
    tag_urls = []
    for t in soup.find_all(["link","script","meta"]):
        u = t.get("href","") if t.name=="link" else t.get("src","") if t.name=="script" else ""
        if t.name=="meta":
            m = re.search(r"url=([^;]+)$",(t.get("content","") or "").lower()); u = m.group(1).strip() if m else ""
        if u: tag_urls.append(u)
    out["links_in_tags"] = (sum(1 for u in tag_urls if _is_external_href(u,final_url,base_host))/len(tag_urls))*100.0 if tag_urls else 0.0
    media_urls = [t.get("src","") or t.get("data-src","") for t in soup.find_all(["img","audio","embed","source","video","track"]) if t.get("src","") or t.get("data-src","")]
    if media_urls:
        ext_m = sum(1 for u in media_urls if _is_external_href(u,final_url,base_host))
        out["ratio_intMedia"] = ((len(media_urls)-ext_m)/len(media_urls))*100.0
        out["ratio_extMedia"] = (ext_m/len(media_urls))*100.0
    else: out["ratio_intMedia"] = out["ratio_extMedia"] = 0.0
    html_lower = html.lower()
    out["iframe"] = 1 if soup.find("iframe") else 0
    out["popup_window"] = 1 if "window.open(" in html_lower else 0
    out["onmouseover"] = 1 if "onmouseover" in html_lower else 0
    out["right_clic"] = 1 if ("contextmenu" in html_lower or "event.button==2" in html_lower) else 0
    out["safe_anchor"] = (sum(1 for h in anchors if _is_unsafe_anchor(h))/total_a)*100.0 if total_a else 0.0
    title = (soup.title.string.strip().lower() if soup.title and soup.title.string else "")
    out["empty_title"] = 1 if not title else 0
    _, domain, _ = _extract_parts(base_host)
    out["domain_in_title"] = 1 if (domain and title and domain in title) else 0
    page_text = soup.get_text(" ", strip=True).lower()
    out["domain_with_copyright"] = 1 if ((("©" in page_text) or ("copyright" in page_text)) and domain and domain in page_text) else 0

    enrich_external_features(out, base_host)
    return out

def extract_features(url, include_web_content):
    feats = extract_url_features(url)
    if include_web_content:
        feats.update(extract_web_content_features(url))
    return feats

def rule_based_eval(url, return_detail=False):
    feats = extract_url_features(url)
    # Sangat Penting: satu fitur saja → langsung Phishing (sesuai dokumen TASI Bab 3.4)
    very_important = {
        "suspicious_tld":     feats.get("suspicious_tld",0)==1,
        "ip":                 feats.get("ip",0)==1,
        "random_domain":      feats.get("random_domain",0)==1,
        "shortening_service": feats.get("shortening_service",0)==1,
        "http_in_path":       feats.get("http_in_path",0)==1,
        "https_token":        feats.get("https_token",0)==1,
    }
    # Penting: bobot ×2 (sesuai dokumen TASI Bab 3.4)
    important = {
        "nb_at":          feats.get("nb_at",0)>=1,
        "nb_subdomains":  feats.get("nb_subdomains",0)>3,
        "nb_dots":        feats.get("nb_dots",0)>4,
        "nb_slash":       feats.get("nb_slash",0)>7,
        "length_hostname":feats.get("length_hostname",0)>30,
        "nb_percent":     feats.get("nb_percent",0)>5,
        "nb_tilde":       feats.get("nb_tilde",0)>=1,
        "nb_semicolumn":  feats.get("nb_semicolumn",0)>=1,
        "nb_star":        feats.get("nb_star",0)>=1,
        "nb_comma":       feats.get("nb_comma",0)>=1,
        "nb_dollar":      feats.get("nb_dollar",0)>=1,
        "nb_qm":          feats.get("nb_qm",0)>2,
        "nb_colon":       feats.get("nb_colon",0)>1,
        "nb_eq":          feats.get("nb_eq",0)>8,
        "nb_and":         feats.get("nb_and",0)>3,
        "nb_hyphens":     feats.get("nb_hyphens",0)>3,
        "nb_underscore":  feats.get("nb_underscore",0)>3,
    }
    # Cukup Penting: bobot ×1 (sesuai dokumen TASI Bab 3.4)
    less_important = {
        "ratio_digits_url": feats.get("ratio_digits_url",0)>0.3,
        "port":             feats.get("port",0)==1,
    }
    vi_hits = [k for k,v in very_important.items() if v]
    imp_hits = [k for k,v in important.items() if v]
    less_hits = [k for k,v in less_important.items() if v]
    risk_score = 2*len(imp_hits) + len(less_hits)
    if vi_hits or risk_score >= 5:
        category, rule_flag = "Phishing", 1
    elif risk_score > 0:
        category, rule_flag = "Suspicious", 0
    else:
        category, rule_flag = "Benign", 0
    if return_detail:
        return risk_score, category, rule_flag, {
            "vi_count": len(vi_hits), "imp_count": len(imp_hits), "less_count": len(less_hits),
            "vi_hits": vi_hits, "imp_hits": imp_hits, "less_hits": less_hits,
        }
    return risk_score, category, rule_flag

def get_model_and_features(url):
    feats = extract_features(url, include_web_content=True)
    web_content_ok = any(feats.get(k) not in (None, 0) for k in WEB_CONTENT_KEYS)
    suffix = "_81" if web_content_ok else "_37"

    rf   = GLOBAL_MODELS.get(f"rf{suffix}")
    xgb  = GLOBAL_MODELS.get(f"xgb{suffix}")
    meta = GLOBAL_MODELS.get(f"meta{suffix}")
    cols = GLOBAL_MODELS.get(f"cols{suffix}", [])

    if not cols:
        raise RuntimeError(f"Fitur kolom kosong untuk konfigurasi suffix {suffix}")
    return rf, xgb, meta, cols, web_content_ok, feats

def build_ml_vector(feats, cols):
    vector = [to_float(feats.get(c, 0.0)) for c in cols]
    return np.array([vector], dtype=float), cols

def _class_index(model, class_value, fallback_idx=1):
    classes = list(getattr(model, "classes_", []))
    if class_value in classes: return classes.index(class_value)
    return 1 if len(classes)==2 and fallback_idx>=1 else 0

def _proba_for_class(model, X, class_value):
    try:
        probs = model.predict_proba(X)[0]
        idx = _class_index(model, class_value, 1)
        return float(probs[min(idx, len(probs)-1)])
    except:
        return 1.0 if int(model.predict(X)[0])==class_value else 0.0

def _label_to_phishing_flag(raw): return 1 if int(raw)==PHISHING_CLASS_VALUE else 0

def predict_models(feats, cols, precomputed_rule_score=None, precomputed_rule_flag=None, rf=None, xgb=None, meta=None):
    X, used_cols = build_ml_vector(feats, cols)
    df = pd.DataFrame(X, columns=used_cols)
    rf_prob  = _proba_for_class(rf, df, PHISHING_CLASS_VALUE)
    xgb_prob = _proba_for_class(xgb, df, PHISHING_CLASS_VALUE)
    stack_pred = stack_prob = meta_n_in = None

    if meta is not None:
        try:
            meta_n_in = int(getattr(meta, "n_features_in_", 2))
        except:
            meta_n_in = 2

        rs = precomputed_rule_score if precomputed_rule_score is not None else 0.0
        rf_ = precomputed_rule_flag if precomputed_rule_flag is not None else 0

        meta_feats = [rf_prob, xgb_prob]
        if meta_n_in >= 3:
            meta_feats = [rf_prob, xgb_prob, rf_]
        if meta_n_in >= 4:
            meta_feats = [rf_prob, xgb_prob, rf_, rs]

        meta_X = np.array([meta_feats[:meta_n_in]], dtype=float)
        try:
            stack_pred = _label_to_phishing_flag(int(meta.predict(meta_X)[0]))
            stack_prob = _proba_for_class(meta, meta_X, PHISHING_CLASS_VALUE)
        except Exception as e:
            print(f"[STACK WARNING] Meta-learner gagal: {e}")
            stack_pred = None
            stack_prob = None

    return {
        "rf_prob": rf_prob, "xgb_prob": xgb_prob,
        "rf_pred": _label_to_phishing_flag(int(rf.predict(df)[0])),
        "xgb_pred": _label_to_phishing_flag(int(xgb.predict(df)[0])),
        "stack_pred": stack_pred, "stack_prob": stack_prob, "meta_n_in": meta_n_in,
    }

def _shap_vector(shap_values, class_index=1):
    if isinstance(shap_values, list):
        vals = shap_values[class_index] if len(shap_values) > class_index else shap_values[0]
        return np.array(vals).flatten()
    v = np.array(shap_values)
    if v.ndim == 2: return v[0]
    if v.ndim == 3:
        return v[0,:,class_index] if v.shape[-1] > class_index else v[0,:,0]
    return v.flatten()

def get_shap_top(model, X_arr, feature_names, top_n=5):
    fn = list(feature_names)
    try:
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(X_arr)
        vals = _shap_vector(sv, 1)
    except Exception:
        try:
            bg = np.zeros((5, X_arr.shape[1]))
            exp = shap.KernelExplainer(model.predict_proba, bg)
            sv  = exp.shap_values(X_arr, nsamples=100)
            vals = _shap_vector(sv, 1)
        except Exception:
            return []

    vals = np.array(vals, dtype=float).flatten()[:len(fn)]
    top_idx = np.argsort(np.abs(vals))[::-1][:top_n]
    result = []
    for i in top_idx:
        s = float(vals[i])
        if abs(s) < 1e-4:       # skip fitur yang benar-benar nol kontribusinya
            continue
        result.append({
            "name": fn[i], "shap_signed": s, "abs_shap": abs(s),
            "direction": "PHISHING" if s > 0 else ("BENIGN" if s < 0 else "NEUTRAL"),
        })
    return result

def get_phishing_risk_direction(feature_name, feature_value):
    feature_rules = {
        "ip":                    (lambda v: v==1,  "Menggunakan IP address langsung sebagai domain (bukan nama domain)",
                                  lambda v: v==0,  "Menggunakan nama domain, bukan IP address"),
        "nb_at":                 (lambda v: v>=1,  "Ada simbol @ dalam URL — umum dipakai untuk menyamarkan domain asli",
                                  lambda v: v==0,  "Tidak ada simbol @ yang mencurigakan"),
        "nb_underscore":         (lambda v: v>3,   "Terlalu banyak underscore, tidak wajar untuk domain/path normal",
                                  lambda v: v==0,  "Tidak ada underscore mencurigakan dalam URL"),
        "nb_percent":            (lambda v: v>5,   "Banyak karakter ter-encode (%xx) — tanda URL obfuskasi",
                                  lambda v: v==0,  "Tidak ada encoding mencurigakan dalam URL"),
        "nb_tilde":              (lambda v: v>=1,  "Karakter ~ dalam URL — jarang dipakai situs benign",
                                  lambda v: v==0,  "Tidak ada karakter tilde mencurigakan"),
        "nb_semicolumn":         (lambda v: v>=1,  "Karakter ; dalam URL — bisa dipakai untuk menyisipkan parameter palsu",
                                  lambda v: v==0,  "Tidak ada semicolons mencurigakan"),
        "nb_star":               (lambda v: v>=1,  "Karakter * dalam URL — tidak lazim untuk situs normal",
                                  lambda v: v==0,  "Tidak ada karakter bintang mencurigakan"),
        "nb_comma":              (lambda v: v>=1,  "Karakter koma dalam URL — tidak standar",
                                  lambda v: v==0,  "Tidak ada koma mencurigakan dalam URL"),
        "nb_dollar":             (lambda v: v>=1,  "Karakter $ dalam URL — jarang di situs resmi",
                                  lambda v: v==0,  "Tidak ada karakter dollar mencurigakan"),
        "nb_hyphens":            (lambda v: v>3,   "Terlalu banyak tanda hubung — domain phishing sering pakai banyak hyphen",
                                  lambda v: v<=1,  "Jumlah tanda hubung dalam batas wajar"),
        "nb_dots":               (lambda v: v>4,   "Terlalu banyak titik — kemungkinan subdomain berlapis untuk menyamarkan domain asli",
                                  lambda v: v<=2,  "Jumlah titik dalam URL normal"),
        "nb_slash":              (lambda v: v>7,   "Path URL sangat panjang dan kompleks",
                                  lambda v: v<=4,  "Struktur path URL sederhana dan wajar"),
        "nb_qm":                 (lambda v: v>2,   "Banyak tanda tanya — query parameter berlebihan",
                                  lambda v: v==0,  "Tidak ada query parameter mencurigakan"),
        "nb_and":                (lambda v: v>3,   "Banyak parameter & dalam query string",
                                  lambda v: v==0,  "Tidak ada parameter & berlebihan"),
        "nb_colon":              (lambda v: v>1,   "Banyak titik dua — bisa menyembunyikan port atau parameter palsu",
                                  lambda v: v==1,  "Jumlah titik dua normal (hanya dari protokol)"),
        "length_url":            (lambda v: v>75,  "URL sangat panjang — umum dipakai untuk menyembunyikan tujuan asli",
                                  lambda v: v<50,  "Panjang URL singkat dan wajar"),
        "length_hostname":       (lambda v: v>30,  "Hostname sangat panjang — domain phishing sering panjang untuk meniru brand",
                                  lambda v: v<15,  "Hostname singkat dan simpel, ciri domain resmi"),
        "ratio_digits_url":      (lambda v: v>0.3, "Proporsi angka terlalu tinggi dalam URL",
                                  lambda v: v<0.05,"Hampir tidak ada angka acak dalam URL"),
        "nb_subdomains":         (lambda v: v>3,   "Subdomain berlapis-lapis — taktik phishing untuk meniru struktur domain resmi",
                                  lambda v: v<=1,  "Jumlah subdomain normal"),
        "random_domain":         (lambda v: v==1,  "Nama domain terlihat acak/gibberish — bukan nama yang mudah diingat",
                                  lambda v: v==0,  "Nama domain terlihat natural, bukan acak"),
        "prefix_suffix":         (lambda v: v==1,  "Ada tanda hubung (-) dalam nama domain — pola umum domain phishing seperti 'paypal-secure.com'",
                                  lambda v: v==0,  "Tidak ada tanda hubung di nama domain"),
        "shortening_service":    (lambda v: v==1,  "Menggunakan layanan pemendek URL (bit.ly, dll) — menyembunyikan tujuan asli",
                                  lambda v: v==0,  "Tidak menggunakan URL shortener"),
        "suspicious_tld":        (lambda v: v==1,  "TLD (ekstensi domain) termasuk kategori mencurigakan seperti .xyz, .tk, .top",
                                  lambda v: v==0,  "Ekstensi domain normal dan terpercaya"),
        "http_in_path":          (lambda v: v==1,  "Teks 'http' muncul di dalam path URL — indikasi URL-in-URL untuk pengalihan berbahaya",
                                  lambda v: v==0,  "Tidak ada protokol tersembunyi di path URL"),
        "https_token":           (lambda v: v==1,  "Kata 'https' muncul di path URL (bukan di protokol) — trik psikologis agar terlihat aman",
                                  lambda v: v==0,  "Tidak ada kata 'https' yang disalahgunakan di path"),
        "port":                  (lambda v: v==1,  "Menggunakan port tidak standar — situs resmi jarang mengekspos port selain 80/443",
                                  lambda v: v==0,  "Menggunakan port standar (80/443)"),
        "nb_redirection":        (lambda v: v>0,   "URL mengandung redirect berantai — taktik untuk menyembunyikan tujuan akhir",
                                  lambda v: v==0,  "Tidak ada redirect tersembunyi dalam URL"),
        "punycode":              (lambda v: v==1,  "Domain menggunakan Punycode (xn--) — taktik homograph attack untuk meniru domain asli",
                                  lambda v: v==0,  "Tidak ada Punycode — domain menggunakan karakter standar"),
        "tld_in_path":           (lambda v: v==1,  "Ekstensi domain (.com, .net) muncul di path URL — tanda struktur URL tidak wajar",
                                  lambda v: v==0,  "Struktur URL normal, TLD hanya di domain"),
        "tld_in_subdomain":      (lambda v: v==1,  "Ekstensi domain muncul di subdomain — trik phishing agar terlihat seperti domain lain",
                                  lambda v: v==0,  "Subdomain normal, tidak mengandung TLD palsu"),
        "abnormal_subdomain":    (lambda v: v==1,  "Subdomain mengandung 'http/https' — struktur sangat mencurigakan",
                                  lambda v: v==0,  "Subdomain normal"),
        "path_extension":        (lambda v: v==1,  "File di path punya ekstensi tertentu — bisa mengunduh file berbahaya",
                                  lambda v: v==0,  "Tidak ada ekstensi file mencurigakan di path"),
        "phish_hints":           (lambda v: v>0,   "URL mengandung kata kunci phishing seperti 'login', 'verify', 'secure', 'account'",
                                  lambda v: v==0,  "Tidak ada kata kunci phishing dalam URL"),
        "domain_in_brand":       (lambda v: v==0,  "Domain tidak dikenali sebagai brand terkenal",
                                  lambda v: v==1,  "Domain dikenali sebagai brand besar yang terpercaya"),
        "brand_in_subdomain":    (lambda v: v==1,  "Nama brand besar (Google, PayPal, dll) muncul di subdomain — taktik phishing umum",
                                  lambda v: v==0,  "Tidak ada penyalahgunaan nama brand di subdomain"),
        "brand_in_path":         (lambda v: v==1,  "Nama brand besar muncul di path URL — bisa jadi upaya meniru halaman brand tersebut",
                                  lambda v: v==0,  "Tidak ada nama brand di path"),
        "nb_www":                (lambda v: v==0,  "Tidak menggunakan sub-domain www standar",
                                  lambda v: v==1,  "Menggunakan sub-domain www standar"),
        "login_form":            (lambda v: v==1,  "Halaman memiliki form login — terutama mencurigakan bila domain tidak dikenal",
                                  lambda v: v==0,  "Tidak ada form login mencurigakan di halaman"),
        "external_favicon":      (lambda v: v==1,  "Favicon (ikon tab) dimuat dari domain lain — tanda situs meniru tampilan brand lain",
                                  lambda v: v==0,  "Favicon dihosting di domain yang sama"),
        "iframe":                (lambda v: v==1,  "Halaman menggunakan iframe tersembunyi — umum di serangan phishing dan clickjacking",
                                  lambda v: v==0,  "Tidak ada iframe mencurigakan"),
        "popup_window":          (lambda v: v==1,  "Halaman membuka popup secara otomatis — taktik umum untuk mengelabui pengguna",
                                  lambda v: v==0,  "Tidak ada popup otomatis"),
        "onmouseover":           (lambda v: v==1,  "Halaman memanipulasi URL saat mouse hover — menyembunyikan tujuan link asli",
                                  lambda v: v==0,  "Tidak ada manipulasi link saat hover"),
        "right_clic":            (lambda v: v==1,  "Klik kanan dinonaktifkan — taktik menyembunyikan source code dari pengguna",
                                  lambda v: v==0,  "Klik kanan berfungsi normal"),
        "empty_title":           (lambda v: v==1,  "Halaman tidak memiliki judul (title kosong) — indikasi halaman dibuat terburu-buru",
                                  lambda v: v==0,  "Halaman memiliki judul yang normal"),
        "domain_in_title":       (lambda v: v==0,  "Nama domain tidak tercantum di judul halaman",
                                  lambda v: v==1,  "Judul halaman mencantumkan nama domain — tanda situs yang konsisten"),
        "domain_with_copyright": (lambda v: v==0,  "Tidak ada pernyataan hak cipta domain resmi",
                                  lambda v: v==1,  "Ada pernyataan copyright dengan nama domain — tanda situs resmi"),
        "ratio_extHyperlinks":   (lambda v: v>0.5, "Lebih dari setengah link mengarah ke domain lain — konten hampir semua dari luar",
                                  lambda v: v<0.2, "Sebagian besar link mengarah ke domain sendiri"),
        "ratio_intHyperlinks":   (lambda v: v<0.1, "Hampir tidak ada link internal — situs tidak punya konten navigasi sendiri",
                                  lambda v: v>0.5, "Banyak link internal — menandakan situs punya struktur konten yang wajar"),
        "nb_external_redirection":(lambda v: v>2,  "Banyak redirect ke domain eksternal berbeda — pola phishing multi-hop",
                                   lambda v: v==0, "Tidak ada redirect ke domain luar"),
        "ratio_extRedirection":  (lambda v: v>0.3, "Proporsi link redirect eksternal tinggi",
                                  lambda v: v==0,  "Tidak ada link redirect eksternal mencurigakan"),
        "links_in_tags":         (lambda v: v>50,  "Lebih dari setengah resource (script, CSS) dimuat dari domain luar",
                                  lambda v: v<20,  "Sebagian besar resource dimuat dari domain sendiri"),
        "nb_extCSS":             (lambda v: v>3,   "Banyak file CSS dari domain eksternal — bisa dipakai untuk menyembunyikan konten",
                                  lambda v: v==0,  "Tidak ada CSS dari domain luar"),
        "ratio_extMedia":        (lambda v: v>50,  "Lebih dari setengah media (gambar, video) dari domain lain",
                                  lambda v: v<20,  "Sebagian besar media dihosting di domain sendiri"),
        "page_rank":             (lambda v: v<1.0, "PageRank sangat rendah — situs belum dikenal mesin pencari",
                                  lambda v: v>=4.0,"PageRank tinggi — situs sudah dikenal dan terpercaya"),
        "google_index":          (lambda v: v==0,  "Situs tidak terindeks Google — baru dibuat atau sengaja disembunyikan",
                                  lambda v: v==1,  "Situs sudah terindeks Google — menandakan keberadaan yang benign"),
        "web_traffic":           (lambda v: 0 < v < 100, "Traffic sangat rendah berdasarkan data API — situs hampir tidak dikenal",
                                  lambda v: v>10000, "Traffic tinggi — situs populer dan sudah dikenal luas"),
        "dns_record":            (lambda v: v==0,  "Tidak punya DNS record valid — sangat mencurigakan",
                                  lambda v: v==1,  "DNS record valid"),
        "domain_age":            (lambda v: 0 < v < 30,  "Domain sangat baru (< 30 hari) — situs phishing sering pakai domain baru",
                                  lambda v: v>365, "Domain sudah lama terdaftar (> 1 tahun) — indikasi situs terpercaya"),
        "domain_registration_length": (lambda v: 0 < v < 180, "Masa registrasi domain sangat singkat — situs phishing jarang registrasi jangka panjang",
                                        lambda v: v>720, "Domain diregistrasi untuk jangka panjang — menandakan komitmen situs resmi"),
        "whois_registered_domain": (lambda v: v==0, "Data registrasi domain WHOIS tidak ditemukan",
                                     lambda v: v==1, "Data WHOIS tersedia — domain terdaftar secara resmi"),
        "statistical_report":    (lambda v: v==1,  "Terdeteksi oleh laporan statistik keamanan — pola URL mencurigakan secara kumulatif",
                                  lambda v: v==0,  "Tidak memenuhi threshold laporan statistik keamanan"),
    }

    if feature_name in feature_rules:
        phish_fn, phish_reason, benign_fn, benign_reason = feature_rules[feature_name]
        if phish_fn(feature_value) and phish_reason:
            return "PHISHING", phish_reason
        if benign_fn(feature_value) and benign_reason:
            return "BENIGN", benign_reason

    return "NEUTRAL", "Nilai fitur tidak memenuhi kondisi ekstrem phishing maupun benign secara definitif"

# ── Perbaikan Prompt Builder (Strict Context & Anti Malu-maluin) ──────────────────
def build_llm_prompt(url, category, p_phish, model_main, top_features_with_reasons, rekomendasi_tetap):
    lines = []
    for i, f in enumerate(top_features_with_reasons, 1):
        nilai_str = f"Nilai: {f['value']} | " if "value" in f else ""
        lines.append(f"  {i}. Fitur: {f['display_name']} | {nilai_str}Status: {f['impact']} | Hasil Ekstraksi: {f['reason']}")
    
    features_block = "\n".join(lines) if lines else "  (Data fitur tidak tersedia)"

    return (
        f"Kamu adalah analis keamanan siber profesional. Tugasmu adalah menulis penjelasan ringkas (3-4 kalimat) "
        f"dalam Bahasa Indonesia mengenai hasil deteksi sistem terhadap URL berikut.\n\n"
        f"HASIL DETEKSI SISTEM (WAJIB DIIKUTI):\n"
        f"- URL yang diperiksa: {url}\n"
        f"- Kesimpulan Akhir: {category.upper()}\n"
        f"- Probabilitas Phishing: {p_phish:.2f} (Threshold Bahaya >= {FINAL_THRESHOLD})\n"
        f"- Model Utama: {model_main}\n\n"
        f"DATA INTEGRITAS FITUR (JANGAN DIUBAH ATAU DIPUTARBALIKKAN):\n"
        f"{features_block}\n\n"
        f"PETUNJUK PENULISAN PENJELASAN:\n"
        f"1. Jelaskan secara logis mengapa skor bisa bernilai {p_phish:.2f} berdasarkan data fitur di atas.\n"
        f"2. Fokuskan penjelasan pada dukungan untuk Kesimpulan Akhir: jika hasilnya BENIGN, jelaskan bahwa fitur utama mendukung benign; "
        f"jika hasilnya PHISHING, jelaskan bahwa fitur utama mendukung phishing.\n"
        f"3. Jangan menggunakan kalimat yang meragukan kesimpulan akhir. Tulis dengan tegas sesuai hasil deteksi.\n"
        f"4. JANGAN PERNAH mengada-ada atau membawa nama fitur yang tidak tertulis pada data di atas!\n"
        f"5. Kamu HARUS mengakhiri kalimat penjelasanmu tepat dengan teks instruksi ini tanpa diubah: {rekomendasi_tetap}"
    )

def fetch_llm_reasoning_safe(prompt):
    try:
        return get_llm_reasoning(prompt)
    except Exception as e:
        return f"Penjelasan otomatis tertunda karena interupsi jaringan API: {str(e)}"

def generate_explanation(url, feats_full, cols, rf_prob, xgb_prob, stack_prob,
                         decision_source, rule_detail, risk_score, rule_flag,
                         rf_model, xgb_model, final_label):
    if decision_source == "rule_based_prefilter_phishing":
        p_phish = clamp01(max(PREFILTER_PHISHING_MIN_CONF, risk_score/10.0))
    else:
        p_phish = stack_prob if stack_prob is not None else (
            0.5*rf_prob + 0.5*xgb_prob if rf_prob is not None and xgb_prob is not None
            else rf_prob or xgb_prob or 0.0
        )
    category = "phishing" if final_label == 1 else "benign"

    if decision_source == "rf_only": active_model, model_main = rf_model, "Random Forest"
    elif decision_source == "xgb_only": active_model, model_main = xgb_model, "XGBoost"
    elif decision_source == "rule_based_prefilter_phishing": active_model, model_main = None, "Rule-Based Prefilter"
    else:
        rf_p  = rf_prob  if rf_prob  is not None else -1.0
        xgb_p = xgb_prob if xgb_prob is not None else -1.0
        if rf_model is not None and xgb_model is not None:
            active_model, model_main = (rf_model, "Random Forest") if rf_p >= xgb_p else (xgb_model, "XGBoost")
        elif rf_model is not None: active_model, model_main = rf_model, "Random Forest"
        elif xgb_model is not None: active_model, model_main = xgb_model, "XGBoost"
        else: active_model, model_main = None, "Unknown"
    if stack_prob is not None and decision_source not in ("rf_only","xgb_only"):
        model_main = "RF + XGB + Stacking"

    feat_array = np.array([[feats_full.get(c,0.0) for c in cols]], dtype=float)
    shap_items = []
    if active_model is not None:
        shap_items = get_shap_top(active_model, feat_array, cols, top_n=len(cols)) 
        if not shap_items:
            other = xgb_model if active_model is rf_model else rf_model
            if other is not None:
                shap_items = get_shap_top(other, feat_array, cols, top_n=len(cols))

    shap_items = [
        {**f, "value": feats_full.get(f["name"], 0)}
        for f in shap_items
    ]

    FEATURE_LABEL_MAP = {
        # Fitur URL struktural
        "length_url":           "Panjang URL",
        "length_hostname":      "Panjang Hostname",
        "ip":                   "Penggunaan IP Address",
        "nb_dots":              "Jumlah Titik (.) di URL",
        "nb_hyphens":           "Jumlah Tanda Hubung (-)",
        "nb_at":                "Jumlah Simbol @",
        "nb_qm":                "Jumlah Tanda Tanya (?)",
        "nb_and":               "Jumlah Simbol &",
        "nb_eq":                "Jumlah Simbol =",
        "nb_underscore":        "Jumlah Garis Bawah (_)",
        "nb_tilde":             "Jumlah Simbol Tilde (~)",
        "nb_percent":           "Jumlah Karakter Persen (%)",
        "nb_slash":             "Jumlah Garis Miring (/)",
        "nb_star":              "Jumlah Simbol Bintang (*)",
        "nb_colon":             "Jumlah Titik Dua (:)",
        "nb_comma":             "Jumlah Koma (,)",
        "nb_semicolumn":        "Jumlah Titik Koma (;)",
        "nb_dollar":            "Jumlah Simbol Dollar ($)",
        "nb_space":             "Jumlah Spasi di URL",
        "nb_www":               "Subdomain WWW",
        "nb_com":               "Kemunculan .com di URL",
        "nb_dslash":            "Jumlah Double Slash (//)",
        "http_in_path":         "HTTP di Path URL",
        "https_token":          "Token HTTPS Palsu di Path",
        "ratio_digits_url":     "Rasio Angka dalam URL",
        "ratio_digits_host":    "Rasio Angka dalam Hostname",
        "punycode":             "Penggunaan Punycode (xn--)",
        "port":                 "Port Tidak Standar",
        "tld_in_path":          "TLD Muncul di Path",
        "tld_in_subdomain":     "TLD Muncul di Subdomain",
        "abnormal_subdomain":   "Subdomain Tidak Normal",
        "nb_subdomains":        "Jumlah Subdomain",
        "prefix_suffix":        "Tanda Hubung di Nama Domain",
        "random_domain":        "Domain Acak/Tidak Bermakna",
        "shortening_service":   "Layanan Pemendek URL",
        "path_extension":       "Ekstensi File di Path",
        "nb_redirection":       "Redirect Tersembunyi di URL",
        "length_words_raw":     "Jumlah Kata dalam URL",
        "char_repeat":          "Pengulangan Karakter",
        "shortest_words_raw":   "Kata Terpendek di URL",
        "shortest_word_host":   "Kata Terpendek di Hostname",
        "shortest_word_path":   "Kata Terpendek di Path",
        "longest_words_raw":    "Kata Terpanjang di URL",
        "longest_word_host":    "Kata Terpanjang di Hostname",
        "longest_word_path":    "Kata Terpanjang di Path",
        "avg_words_raw":        "Rata-rata Panjang Kata di URL",
        "avg_word_host":        "Rata-rata Panjang Kata di Hostname",
        "avg_word_path":        "Rata-rata Panjang Kata di Path",
        "phish_hints":          "Kata Kunci Phishing di URL",
        "domain_in_brand":      "Domain Cocok dengan Brand Besar",
        "brand_in_subdomain":   "Nama Brand di Subdomain",
        "brand_in_path":        "Nama Brand di Path URL",
        "suspicious_tld":       "TLD Mencurigakan",
        "statistical_report":   "Laporan Statistik Keamanan",
        # Fitur konten halaman web
        "nb_hyperlinks":        "Jumlah Tautan di Halaman",
        "ratio_intHyperlinks":  "Rasio Tautan Internal",
        "ratio_extHyperlinks":  "Rasio Tautan Eksternal",
        "nb_extCSS":            "Jumlah CSS Eksternal",
        "ratio_extRedirection": "Rasio Redirect Eksternal",
        "ratio_extErrors":      "Rasio Error Eksternal",
        "login_form":           "Formulir Login di Halaman",
        "external_favicon":     "Favicon dari Domain Lain",
        "links_in_tags":        "Resource Eksternal di Tag HTML",
        "ratio_intMedia":       "Rasio Media Internal",
        "ratio_extMedia":       "Rasio Media Eksternal",
        "iframe":               "Penggunaan Iframe",
        "popup_window":         "Popup Otomatis",
        "safe_anchor":          "Anchor Tidak Aman",
        "onmouseover":          "Manipulasi Hover Mouse",
        "right_clic":           "Klik Kanan Dinonaktifkan",
        "empty_title":          "Judul Halaman Kosong",
        "domain_in_title":      "Domain di Judul Halaman",
        "domain_with_copyright":"Hak Cipta Domain di Halaman",
        "nb_external_redirection": "Redirect ke Domain Luar",
        # Fitur domain & reputasi
        "whois_registered_domain":    "Status WHOIS Domain",
        "domain_registration_length": "Durasi Registrasi Domain",
        "domain_age":           "Usia Domain",
        "web_traffic":          "Trafik Web",
        "dns_record":           "DNS Record",
        "google_index":         "Indeks Google",
        "page_rank":            "PageRank Mesin Pencari",
    }

    # Fitur web-content yang nilainya 0 karena API gagal bukan berarti kondisi nyata 0.
    # SHAP sudah memperhitungkan nilai ini dari konteks training — percayai SHAP.
    WEB_FETCH_DEPENDENT = {
        "google_index", "page_rank", "web_traffic", "dns_record",
        "domain_age", "domain_registration_length", "whois_registered_domain",
        "nb_hyperlinks", "ratio_intHyperlinks", "ratio_extHyperlinks",
        "nb_extCSS", "ratio_extRedirection", "login_form", "external_favicon",
        "links_in_tags", "ratio_intMedia", "ratio_extMedia",
    }

    rekomendasi_tetap = "REKOMENDASI: BLOKIR." if final_label == 1 else "REKOMENDASI: IZINKAN."

    top_features_reasons = []
    top_features = []
    for f in shap_items:
        feat_val = feats_full.get(f["name"], 0)
        rule_dir, reason = get_phishing_risk_direction(f["name"], feat_val)
        shap_dir = f["direction"]

        # Prioritas: SHAP (kebenaran matematis model)
        # Rule hanya tiebreaker jika SHAP nyaris nol
        if abs(f["shap_signed"]) < 1e-6:
            display_impact = rule_dir if rule_dir != "NEUTRAL" else shap_dir
        elif f["name"] in WEB_FETCH_DEPENDENT and feat_val == 0:
            # Nilai 0 karena API gagal → percayai arah SHAP
            display_impact = shap_dir
            if shap_dir == "BENIGN":
                reason = "Berdasarkan analisis model, fitur ini berkontribusi pada keamanan URL"
            elif shap_dir == "PHISHING":
                reason = "Berdasarkan analisis model, fitur ini menambah risiko phishing"
            else:
                reason = "Tidak ada dampak signifikan dari fitur ini"
        elif rule_dir == shap_dir or (rule_dir != "NEUTRAL" and shap_dir == "NEUTRAL"):
            display_impact = rule_dir  # sepakat atau rule punya pendapat, SHAP netral
        else:
            # SHAP dan rule tidak sepakat → percayai SHAP, perbarui reason
            display_impact = shap_dir
            if shap_dir == "BENIGN":
                reason = "Analisis model: fitur ini mendukung keamanan URL"
            elif shap_dir == "PHISHING":
                reason = "Analisis model: fitur ini menambah indikasi phishing"

        top_features_reasons.append({
            "display_name": FEATURE_LABEL_MAP.get(f["name"], f["name"].replace("_", " ")),
            "value": feat_val,
            "impact": display_impact,
            "reason": reason
        })
        top_features.append({
            "name": f["name"], "value": feat_val, "impact": display_impact, "reason": reason,
            "shap_value": f["shap_signed"], "shap_value_signed": f["shap_signed"],
            "abs_shap": f["abs_shap"], "shap_direction": shap_dir,
        })

    prompt = build_llm_prompt(url, category, p_phish, model_main, top_features_reasons, rekomendasi_tetap)
    future = executor.submit(fetch_llm_reasoning_safe, prompt)
    
    try:
        llm_reasoning = future.result(timeout=15)
    except Exception:
        llm_reasoning = f"Situs secara dominan terdeteksi sebagai {category.upper()} oleh {model_main} dengan keyakinan {p_phish*100:.1f}%."

    return {
        "top_influential_features": top_features, "main_contributing_model": model_main,
        "phishing_probability": round(p_phish, 4), "llm_reasoning": llm_reasoning, "shap_items": shap_items,
    }

def log_feature_extraction(url, mode, feats, feature_columns_81, status, decision_source):
    with _log_lock:
        return _log_feature_extraction_impl(url, mode, feats, feature_columns_81, status, decision_source)

def _log_feature_extraction_impl(url, mode, feats, feature_columns_81, status, decision_source):
    csv_path = LOG_PATH
    write_header = not os.path.exists(csv_path)
    
    # 1. Hitung Nomor Urut (NO) Secara Otomatis Berdasarkan Baris Terakhir
    no = 1
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                lines = f.readlines()
                # Baris 1: sep=;, Baris 2: Header, Baris 3: Data Pertama
                if len(lines) >= 3: 
                    last_line = lines[-1].strip().split(";")
                    if last_line[0].isdigit():
                        no = int(last_line[0]) + 1
        except:
            no = 1

    # 2. Definisikan Urutan Kolom Secara Baku (TOP_FEATURE Tanpa S Sesuai Excel)
    headers = ["NO", "URL", "MODE", "DECISION_SOURCE", "STATUS", "PHISHING_PROB"] + feature_columns_81 + ["TOP_FEATURE", "LLM_REASONING"]

    # 3. Susun Data ke Dalam Dictionary
    row_dict = {
        "NO": no,
        "URL": url,
        "MODE": mode,
        "DECISION_SOURCE": decision_source,
        "STATUS": status,
    }
    for feat in feature_columns_81:
        val = feats.get(feat, 0.0)
        if isinstance(val, (list, dict, tuple)):
            row_dict[feat] = str(val)
        elif val is None:
            row_dict[feat] = 0.0
        else:
            row_dict[feat] = val

    MAX_CELL = 32000
    row_dict["PHISHING_PROB"] = feats.get("_phishing_prob", "")
    row_dict["TOP_FEATURE"]   = str(feats.get("_top_features","")).replace("\n", "  ||  ").replace("\r", " ")[:MAX_CELL]
    row_dict["LLM_REASONING"] = str(feats.get("_llm_reasoning","")).replace("\n", " ").replace("\r", " ")[:MAX_CELL]

    # Tulis ke CSV — deteksi jika file sedang terkunci (dibuka di Excel)
    import csv

    def _try_open_exclusive(path, mode):
        """Buka file secara eksklusif. Raise PermissionError jika terkunci."""
        import sys
        fh = open(path, mode, newline="", encoding="utf-8-sig")
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, PermissionError):
            fh.close()
            raise PermissionError("FILE_LOCKED")
        return fh

    try:
        if write_header:
            fh = _try_open_exclusive(csv_path, "w")
            fh.write("sep=;\n")
            writer = csv.DictWriter(fh, fieldnames=headers, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerow(row_dict)
            fh.close()
        else:
            fh = _try_open_exclusive(csv_path, "a")
            writer = csv.DictWriter(fh, fieldnames=headers, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row_dict)
            fh.close()
        print(f"[LOG SUCCESS] Baris #{no} berhasil dicatat.")
        return None  # sukses
    except PermissionError:
        msg = "Log CSV sedang terbuka di Excel/program lain. Tutup file log terlebih dahulu, lalu coba lagi."
        print(f"[LOG LOCKED] {msg}")
        return msg  # kembalikan pesan error ke caller
    except Exception as e:
        print(f"[LOG ERROR] {e}")
        return str(e)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return send_from_directory("Phishing_detection_app", "advanced_hybrid_detector.html")

@app.get("/health")
def health(): return jsonify({"status": "ok"})

def to_bool(val, default=False):
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() in ("1","true","yes","y","on")
    return bool(val) if isinstance(val, (int,float)) else default

def normalize_decision_mode(mode):
    if not mode: return DEFAULT_DECISION_MODE
    m = str(mode).strip().lower()
    if m in ("rf","rf_only"): return "rf_only"
    if m in ("xgb","xgb_only"): return "xgb_only"
    if m in ("stack","stacking","ml_stacking_only"): return "ml_stacking_only"
    return "hybrid_prefilter"

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    debug = to_bool(data.get("debug", False))
    use_prefilter = to_bool(data.get("use_prefilter", DEFAULT_USE_PREFILTER), DEFAULT_USE_PREFILTER)
    decision_mode = normalize_decision_mode(data.get("decision_mode") or data.get("mode"))

    if not url: return jsonify({"error": "URL kosong"}), 400
    if not is_valid_url(url): return jsonify({"error": "URL tidak valid"}), 400

    # Cek file log sebelum mulai proses apapun.
    # Jika terkunci → tolak request sekarang juga, jangan buang waktu proses.
    log_locked, lock_msg = check_log_locked()
    if log_locked:
        return jsonify({"error": lock_msg}), 503

    rf, xgb, meta, cols, web_content_ok, feats_full = get_model_and_features(url)
    include_web_content = web_content_ok
    mode = "81" if web_content_ok else "37"
    risk_score, risk_category, rule_flag, rule_detail = rule_based_eval(url, return_detail=True)

    def build_response(final_phishing_prob, decision_source, model_name,
                       rf_prob=None, xgb_prob=None, stack_prob=None,
                       web_used=None, rf_model=None, xgb_model=None):
        p_phish = clamp01(final_phishing_prob)
        p_safe  = clamp01(1.0 - p_phish)
        final_label = 1 if p_phish >= FINAL_THRESHOLD else 0
        category = "phishing" if final_label == 1 else "benign"
        confidence = round(p_phish if final_label == 1 else p_safe, 4)

        exp = generate_explanation(
            url, feats_full, cols, rf_prob, xgb_prob, stack_prob,
            decision_source, rule_detail, risk_score, rule_flag,
            rf_model, xgb_model, final_label
        )

        resp = {
            "final_label": final_label, "final_phishing_prob": round(p_phish, 4), "final_safe_prob": round(p_safe, 4),
            "final_threshold": FINAL_THRESHOLD, "decision_mode": decision_mode, "decision_source": decision_source,
            "model_name": model_name, "rule_flag": rule_flag, "model_feature_count": len(cols),
            "web_content_used": include_web_content if web_used is None else bool(web_used),
            "label": category, "category": category, "confidence": confidence,
            "rf_prob": round(rf_prob, 4) if rf_prob is not None else None,
            "xgb_prob": round(xgb_prob, 4) if xgb_prob is not None else None,
            "stack_prob": round(stack_prob, 4) if stack_prob is not None else None,
            "llm_reasoning": exp["llm_reasoning"],
            "explanation_detailed": {
                "final_prediction": "PHISHING" if final_label == 1 else "BENIGN",
                "confidence_score": confidence, "phishing_probability": round(p_phish, 4), "benign_probability": round(p_safe, 4),
                "main_contributing_model": exp["main_contributing_model"], "top_influential_features": exp["top_influential_features"],
                "model_contribution_probability": {
                    k: round(v,4) for k,v in {"Random Forest": rf_prob, "XGBoost": xgb_prob, "Logistic Regression (Stacking)": stack_prob}.items() if v is not None
                },
            },
        }

        if debug:
            resp["debug"] = {
                "prefilter_enabled": use_prefilter, "decision_mode": decision_mode,
                "risk_score": risk_score, "risk_category": risk_category, "rule_detail": rule_detail, "features_analyzed": len(cols),
            }

        log_top_lines = []
        all_shap = exp.get("shap_items", [])

        # Catat semua fitur SHAP dengan abs > 1e-6 (skip hanya jika benar-benar nol)
        if all_shap:
            for f in all_shap:
                shap_score = f["shap_signed"]
                feat_val = feats_full.get(f["name"], None)

                if abs(shap_score) < 1e-6 or feat_val is None:
                    continue

                arah = "PHISHING" if shap_score > 0 else ("BENIGN" if shap_score < 0 else "NETRAL")
                log_top_lines.append(
                    f"Fitur: {f['name']} | Nilai: {feat_val} | SHAP: {shap_score:+.4f} | Dampak: [{arah}]"
                )
        else:
            for f in exp.get("top_influential_features", []):
                shap_val = f.get("shap_value_signed", f.get("shap_value", 0))
                arah = f.get("shap_direction", f.get("impact", "?"))
                log_top_lines.append(
                    f"Fitur: {f['name']} | Nilai: {f.get('value',0)} | SHAP: {shap_val:+.4f} | Dampak: [{arah}]"
                )

        log_top = "\n".join(log_top_lines)
        feats_for_log = dict(feats_full)
        feats_for_log["_phishing_prob"] = round(p_phish, 4)
        feats_for_log["_top_features"]  = log_top
        feats_for_log["_llm_reasoning"] = exp.get("llm_reasoning", "")
        
        log_err = log_feature_extraction(url, mode, feats_for_log, FEATURE_COLUMNS_81, category, decision_source)
        if log_err:
            resp["log_warning"] = log_err

        return jsonify(resp)

    if decision_mode == "rf_only":
        preds = predict_models(feats_full, cols, risk_score, rule_flag, rf, xgb, meta)
        return build_response(clamp01(preds["rf_prob"]), "rf_only", "Random Forest", rf_prob=preds["rf_prob"], rf_model=rf)

    if decision_mode == "xgb_only":
        preds = predict_models(feats_full, cols, risk_score, rule_flag, rf, xgb, meta)
        return build_response(clamp01(preds["xgb_prob"]), "xgb_only", "XGBoost", xgb_prob=preds["xgb_prob"], xgb_model=xgb)

    if decision_mode == "ml_stacking_only":
        preds = predict_models(feats_full, cols, risk_score, rule_flag, rf, xgb, meta)
        if preds["stack_prob"] is not None:
            return build_response(clamp01(preds["stack_prob"]), "ml_stacking_only", "RF + XGB + Logistic Regression Stacking", preds["rf_prob"], preds["xgb_prob"], preds["stack_prob"], rf_model=rf, xgb_model=xgb)
        p = safe_avg(preds["rf_prob"], preds["xgb_prob"])
        return build_response(p, "ml_rf_xgb_average_only", "RF + XGB Average", preds["rf_prob"], preds["xgb_prob"], rf_model=rf, xgb_model=xgb)

    if use_prefilter and rule_flag == 1:
        p = clamp01(max(PREFILTER_PHISHING_MIN_CONF, risk_score/10.0))
        return build_response(p, "rule_based_prefilter_phishing", "Rule-Based Prefilter", web_used=False)

    preds = predict_models(feats_full, cols, risk_score, rule_flag, rf, xgb, meta)
    if preds["stack_prob"] is not None:
        return build_response(clamp01(preds["stack_prob"]), "ml_stacking_only", "RF + XGB + Logistic Regression Stacking", preds["rf_prob"], preds["xgb_prob"], preds["stack_prob"], rf_model=rf, xgb_model=xgb)
    p = safe_avg(preds["rf_prob"], preds["xgb_prob"])
    return build_response(p, "ml_rf_xgb_average_only", "RF + XGB Average", preds["rf_prob"], preds["xgb_prob"], rf_model=rf, xgb_model=xgb)

@app.post("/predict_url")
@app.post("/analyze")
def predict_alias(): return predict()

if __name__ == "__main__":
    init_startup_cache()
    app.run(debug=True, host="0.0.0.0", port=5000)