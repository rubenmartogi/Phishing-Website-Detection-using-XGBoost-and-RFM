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
LOG_PATH = os.path.join(BASE_DIR, "log_feature_extraction.xlsx")

# ThreadPoolExecutor untuk menangani pemanggilan LLM non-blocking
executor = ThreadPoolExecutor(max_workers=4)

# ── GLOBAL MODEL CACHE IN STARTUP (Saran Review) ──────────────────────────────
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
# Memuat daftar kolom 81 untuk log Excel secara aman
try:
    FEATURE_COLUMNS_81 = [
        l.strip() for l in open(os.path.join(BASE_DIR, "feature_columns_81.txt"), encoding="utf-8") if l.strip()
    ]
except Exception:
    FEATURE_COLUMNS_81 = []

# ── Helper umum ────────────────────────────────────────────────────────────────
def app_path(*parts): return os.path.join(BASE_DIR, *parts)
def clamp01(v): return float(min(max(v, 0.0), 1.0))
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

# ── URL parsing & feature extraction ──────────────────────────────────────────
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

# [SARAN REVIEW] Dibungkus dengan cache agar pemanggilan beruntun tidak mengulang komputasi regex yang mahal
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

# [SARAN REVIEW] Caching web content features per request agar tidak hit requests HTTP berkali-kali
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

# [SARAN REVIEW] Satu pipa ekstraksi gabungan untuk menghindari double parsing URL
def extract_features(url, include_web_content):
    feats = extract_url_features(url)
    if include_web_content:
        feats.update(extract_web_content_features(url))
    return feats

# ── Rule-based evaluation ──────────────────────────────────────────────────────
def rule_based_eval(url, return_detail=False):
    feats = extract_url_features(url)
    very_important = {
        "suspicious_tld": feats.get("suspicious_tld",0)==1,
        "nb_at": feats.get("nb_at",0)>=1,
        "ip": feats.get("ip",0)==1,
        "nb_underscore": feats.get("nb_underscore",0)>3,
    }
    important = {
        "ratio_digits_url": feats.get("ratio_digits_url",0)>0.3,
        "nb_subdomains": feats.get("nb_subdomains",0)>3,
        "nb_percent": feats.get("nb_percent",0)>5,
        "nb_tilde": feats.get("nb_tilde",0)>=1,
        "nb_semicolumn": feats.get("nb_semicolumn",0)>=1,
        "nb_star": feats.get("nb_star",0)>=1,
        "nb_comma": feats.get("nb_comma",0)>=1,
        "random_domain": feats.get("random_domain",0)==1,
    }
    less_important = {
        "length_hostname": feats.get("length_hostname",0)>30,
        "nb_dollar": feats.get("nb_dollar",0)>=1,
        "nb_qm": feats.get("nb_qm",0)>2,
        "nb_colon": feats.get("nb_colon",0)>1,
        "nb_eq": feats.get("nb_eq",0)>8,
        "nb_dots": feats.get("nb_dots",0)>4,
        "nb_slash": feats.get("nb_slash",0)>7,
        "nb_and": feats.get("nb_and",0)>3,
        "nb_hyphens": feats.get("nb_hyphens",0)>3,
        "http_in_path": feats.get("http_in_path",0)==1,
        "https_token": feats.get("https_token",0)==1,
        "port": feats.get("port",0)==1,
        "shortening_service": feats.get("shortening_service",0)==1,
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

# ── Memilih model dari cache startup global ──────────────────────────────────
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
        except:
            pass

    return {
        "rf_prob": rf_prob, "xgb_prob": xgb_prob,
        "rf_pred": _label_to_phishing_flag(int(rf.predict(df)[0])),
        "xgb_pred": _label_to_phishing_flag(int(xgb.predict(df)[0])),
        "stack_pred": stack_pred, "stack_prob": stack_prob, "meta_n_in": meta_n_in,
    }

# ── SHAP ───────────────────────────────────────────────────────────────────────
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
        result.append({
            "name": fn[i], "shap_signed": s, "abs_shap": abs(s),
            "direction": "PHISHING" if s > 0 else ("BENIGN" if s < 0 else "NEUTRAL"),
        })
    return result

def get_phishing_risk_direction(feature_name, feature_value):
    # Setiap fitur punya dua sisi: kondisi phishing dan kondisi benign
    # Format: (cek_phishing, alasan_phishing, cek_benign, alasan_benign)
    feature_rules = {
        # ── URL structure ──────────────────────────────────────────────────────
        "ip":                    (lambda v: v==1,  "Menggunakan IP address langsung sebagai domain (bukan nama domain)",
                                  lambda v: v==0,  "Menggunakan nama domain, bukan IP address"),
        "nb_at":                 (lambda v: v>=1,  "Ada simbol @ dalam URL — umum dipakai untuk menyamarkan domain asli",
                                  lambda v: v==0,  "Tidak ada simbol @ yang mencurigakan"),
        "nb_underscore":         (lambda v: v>3,   "Terlalu banyak underscore, tidak wajar untuk domain/path normal",
                                  lambda v: v==0,  "Tidak ada underscore mencurigakan dalam URL"),
        "nb_percent":            (lambda v: v>5,   "Banyak karakter ter-encode (%xx) — tanda URL obfuskasi",
                                  lambda v: v==0,  "Tidak ada encoding mencurigakan dalam URL"),
        "nb_tilde":              (lambda v: v>=1,  "Karakter ~ dalam URL — jarang dipakai situs legitimate",
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
        "phish_hints":           (lambda v: v>0,   f"URL mengandung kata kunci phishing seperti 'login', 'verify', 'secure', 'account' (ditemukan {int(feature_value)} kata)",
                                  lambda v: v==0,  "Tidak ada kata kunci phishing dalam URL"),
        # ── Brand/identity ────────────────────────────────────────────────────
        "domain_in_brand":       (lambda v: v==0,  "Domain tidak dikenali sebagai brand terkenal",
                                  lambda v: v==1,  "Domain dikenali sebagai brand besar yang terpercaya"),
        "brand_in_subdomain":    (lambda v: v==1,  "Nama brand besar (Google, PayPal, dll) muncul di subdomain — taktik phishing umum",
                                  lambda v: v==0,  "Tidak ada penyalahgunaan nama brand di subdomain"),
        "brand_in_path":         (lambda v: v==1,  "Nama brand besar muncul di path URL — bisa jadi upaya meniru halaman brand tersebut",
                                  lambda v: v==0,  "Tidak ada nama brand di path"),
        # nb_www: nilai 1 hanya berarti ada string "www" di hostname.
        # Untuk domain phishing seperti linkedin-notification-center.com, ini tidak relevan sbg benign.
        # Dampaknya sepenuhnya tergantung konteks SHAP.
        "nb_www":                (lambda v: False, "",
                                  lambda v: False, ""),
        # ── Web content features ──────────────────────────────────────────────
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
        "domain_in_title":       (lambda v: v==0,  "Judul halaman tidak menyebut domain situs ini",
                                  lambda v: v==1,  "Judul halaman mencantumkan nama domain — tanda situs yang konsisten"),
        "domain_with_copyright": (lambda v: v==0,  "Tidak ada klaim copyright yang menyebut domain ini",
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
        # ── External signals ──────────────────────────────────────────────────
        "page_rank":             (lambda v: v<1.0, "PageRank sangat rendah — situs belum dikenal mesin pencari",
                                  lambda v: v>=4.0,"PageRank tinggi — situs sudah dikenal dan terpercaya"),
        "google_index":          (lambda v: v==0,  "Situs tidak terindeks Google — baru dibuat atau sengaja disembunyikan",
                                  lambda v: v==1,  "Situs sudah terindeks Google — menandakan keberadaan yang legitimate"),
        "web_traffic":           (lambda v: v<100, "Traffic sangat rendah — situs hampir tidak dikenal",
                                  lambda v: v>10000,"Traffic tinggi — situs populer dan sudah dikenal luas"),
        "dns_record":            (lambda v: v==0,  "Tidak punya DNS record valid — sangat mencurigakan",
                                  lambda v: v==1,  "DNS record valid"),
        "domain_age":            (lambda v: v<30,  "Domain sangat baru (< 30 hari) — situs phishing sering pakai domain baru",
                                  lambda v: v>365, "Domain sudah lama terdaftar (> 1 tahun) — indikasi situs terpercaya"),
        "domain_registration_length": (lambda v: v<180, "Masa registrasi domain sangat singkat — situs phishing jarang registrasi jangka panjang",
                                        lambda v: v>720, "Domain diregistrasi untuk jangka panjang — menandakan komitmen situs resmi"),
        "whois_registered_domain": (lambda v: v==0, "Data WHOIS tidak tersedia atau domain tidak terdaftar resmi",
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

    # Fallback untuk fitur binary: nilai 0 = absen indikator buruk
    zero_is_benign = {"ip","nb_at","suspicious_tld","port","shortening_service","random_domain",
                      "nb_tilde","nb_semicolumn","nb_star","nb_comma","nb_dollar","http_in_path",
                      "punycode","abnormal_subdomain","iframe","popup_window","onmouseover",
                      "right_clic","empty_title","login_form","external_favicon"}
    if feature_value == 0 and feature_name in zero_is_benign:
        return "BENIGN", f"Tidak ada indikator buruk ({feature_name.replace('_',' ')} = 0)"

    # Fallback terakhir: coba tebak dari nilai dan nama fitur
    if isinstance(feature_value, (int, float)):
        if feature_name.startswith("nb_") and feature_value == 0:
            return "BENIGN", f"Tidak ada {feature_name.replace('nb_','').replace('_',' ')} dalam URL"
        if feature_name.startswith("ratio_") and feature_value == 0.0:
            return "BENIGN", f"Rasio {feature_name.replace('ratio_','').replace('_',' ')} = 0"

    return "NEUTRAL", "Nilai fitur tidak memenuhi kondisi phishing maupun benign secara definitif"

# ── LLM prompt builder ─────────────────────────────────────────────────────────
def build_llm_prompt(url, category, p_phish, model_main, shap_items):
    FEATURE_LABEL = {
        "page_rank":                  "reputasi domain di mesin pencari (PageRank)",
        "google_index":               "apakah situs terindeks oleh Google",
        "web_traffic":                "jumlah perkiraan pengunjung situs",
        "dns_record":                 "keberadaan DNS record yang valid",
        "domain_age":                 "usia domain sejak pertama kali didaftarkan (hari)",
        "domain_registration_length": "masa registrasi domain (hari)",
        "whois_registered_domain":    "status registrasi domain di WHOIS",
        "nb_hyperlinks":              "jumlah total link di halaman",
        "ratio_intHyperlinks":        "proporsi link yang mengarah ke domain sendiri",
        "ratio_extHyperlinks":        "proporsi link yang mengarah ke domain lain",
        "nb_extCSS":                  "jumlah file CSS dari domain eksternal",
        "ratio_extRedirection":       "proporsi link dengan redirect ke luar",
        "login_form":                 "keberadaan form login di halaman",
        "external_favicon":           "apakah ikon tab dimuat dari domain lain (meniru brand)",
        "links_in_tags":              "proporsi resource (script/CSS) dari domain luar",
        "ratio_intMedia":             "proporsi gambar/media dari domain sendiri",
        "ratio_extMedia":             "proporsi gambar/media dari domain luar",
        "iframe":                     "keberadaan iframe tersembunyi di halaman",
        "popup_window":               "apakah halaman membuka popup otomatis",
        "safe_anchor":                "proporsi link yang tidak mengarah ke mana-mana (#, javascript:)",
        "onmouseover":                "manipulasi URL saat kursor hover di atas link",
        "right_clic":                 "apakah klik kanan dinonaktifkan",
        "empty_title":                "apakah halaman tidak memiliki judul (title kosong)",
        "domain_in_title":            "apakah judul halaman menyebutkan nama domain",
        "domain_with_copyright":      "apakah ada klaim copyright dengan nama domain",
        "nb_www":                     "apakah hostname mengandung string 'www'",
        "domain_in_brand":            "apakah domain dikenali sebagai brand besar terpercaya",
        "brand_in_subdomain":         "apakah nama brand besar muncul di subdomain (taktik peniruan)",
        "brand_in_path":              "apakah nama brand besar muncul di path URL",
        "suspicious_tld":             "apakah ekstensi domain termasuk TLD berisiko tinggi (.xyz, .tk, dll)",
        "ip":                         "apakah domain menggunakan alamat IP langsung",
        "phish_hints":                "jumlah kata kunci phishing dalam URL (login, verify, secure, dll)",
        "nb_at":                      "jumlah simbol @ dalam URL",
        "random_domain":              "apakah nama domain terlihat acak/tidak bermakna",
        "shortening_service":         "apakah URL menggunakan layanan pemendek URL",
        "length_url":                 "total panjang URL (karakter)",
        "length_hostname":            "panjang hostname dalam URL",
        "ratio_digits_url":           "proporsi karakter angka dalam URL",
        "nb_subdomains":              "jumlah tingkat subdomain",
        "prefix_suffix":              "apakah nama domain mengandung tanda hubung (-)",
        "nb_redirection":             "jumlah redirect berantai dalam URL",
        "nb_hyphens":                 "jumlah tanda hubung (-) di seluruh URL",
        "nb_dots":                    "jumlah titik (.) di URL",
        "http_in_path":               "apakah teks 'http' muncul di path URL (URL-in-URL)",
        "https_token":                "apakah kata 'https' muncul di path (bukan di protokol)",
        "port":                       "apakah URL menggunakan port tidak standar",
        "punycode":                   "apakah domain menggunakan encoding punycode (xn--)",
        "statistical_report":         "apakah URL memenuhi pola statistik domain mencurigakan",
        "nb_external_redirection":    "jumlah redirect ke domain eksternal berbeda",
    }

    # Tentukan nilai kontekstual nb_www
    nb_www_val = next((f.get("value") for f in shap_items if f["name"] == "nb_www"), None)
    if nb_www_val is not None:
        # Cek apakah ada indikator lain yang mencurigakan dari URL
        phishing_features = {f["name"] for f in shap_items if f.get("shap_signed", 0) > 0}
        if nb_www_val == 1 and len(phishing_features) > 1:
            FEATURE_LABEL["nb_www"] = "kehadiran string 'www' di hostname (dalam konteks ini tidak cukup menjadi bukti keamanan karena banyak indikator phishing lain yang kuat)"

    lines = []
    for i, f in enumerate(shap_items, 1):
        label = FEATURE_LABEL.get(f["name"], f["name"].replace("_"," "))
        value_str = f", nilai={f.get('value', '?')}"
        direction = "mendorong ke PHISHING" if f["shap_signed"] > 0 else "mendorong ke BENIGN"
        lines.append(f"  {i}. {label}{value_str} | SHAP={f['shap_signed']:+.4f} → {direction}")
    shap_block = "\n".join(lines) if lines else "  (data SHAP tidak tersedia)"

    return (
        f"Kamu adalah analis keamanan siber. Tulis penjelasan 3-4 kalimat "
        f"dalam bahasa Indonesia mengapa URL berikut diklasifikasikan sebagai {category.upper()} "
        f"(probabilitas phishing = {p_phish:.2f}, threshold = {FINAL_THRESHOLD}).\n\n"
        f"URL: {url}\n"
        f"Model: {model_main}\n\n"
        f"Data SHAP (fitur paling berpengaruh):\n{shap_block}\n\n"
        f"Petunjuk penting:\n"
        f"- Fokus pada fitur dengan SHAP absolut terbesar sebagai alasan utama\n"
        f"- Gunakan bahasa natural, jangan sebut nama variabel teknis (misal: jangan bilang 'nb_www', tapi jelaskan maknanya)\n"
        f"- Jika fitur 'www' muncul di tengah banyak indikator phishing lain, jangan jadikan itu alasan utama keamanan\n"
        f"- Jelaskan secara jujur mengapa kombinasi fitur ini mendukung kesimpulan {category.upper()}\n"
        f"- Akhiri dengan: REKOMENDASI: BLOKIR atau REKOMENDASI: IZINKAN."
    )

#  Safe LLM Non-blocking Worker (Saran Review) 
def fetch_llm_reasoning_safe(prompt):
    try:
        return get_llm_reasoning(prompt)
    except Exception as e:
        return f"Penjelasan otomatis tertunda karena interupsi jaringan API: {str(e)}"

def generate_explanation(url, feats_full, cols, rf_prob, xgb_prob, stack_prob,
                         decision_source, rule_detail, risk_score, rule_flag,
                         rf_model, xgb_model, final_label):
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
        shap_items = get_shap_top(active_model, feat_array, cols, top_n=5)
        if not shap_items:
            other = xgb_model if active_model is rf_model else rf_model
            if other is not None:
                shap_items = get_shap_top(other, feat_array, cols, top_n=5)

    shap_items = [
        {**f, "value": feats_full.get(f["name"], 0)}
        for f in shap_items
    ]

    prompt = build_llm_prompt(url, category, p_phish, model_main, shap_items)

    top_features = []
    for f in shap_items:
        feat_val = feats_full.get(f["name"], 0)
        rule_dir, reason = get_phishing_risk_direction(f["name"], feat_val)
        display_impact = rule_dir if rule_dir != "NEUTRAL" else f["direction"]
        top_features.append({
            "name": f["name"], "value": feat_val, "impact": display_impact, "reason": reason,
            "shap_value": f["shap_signed"], "shap_value_signed": f["shap_signed"],
            "abs_shap": f["abs_shap"], "shap_direction": f["direction"],
        })

    # [SARAN REVIEW] Mengirim prompt ke thread pool executor agar non-blocking
    prompt = build_llm_prompt(url, category, p_phish, model_main, shap_items)
    future = executor.submit(fetch_llm_reasoning_safe, prompt)
    
    try:
        llm_reasoning = future.result(timeout=20)
    except Exception:
        llm_reasoning = f"Situs secara dominan terdeteksi sebagai {category.upper()} oleh {model_main} dengan keyakinan {p_phish*100:.1f}%."

    return {
        "top_influential_features": top_features, "main_contributing_model": model_main,
        "phishing_probability": round(p_phish, 4), "llm_reasoning": llm_reasoning, "shap_items": shap_items,
    }

# Logging
def log_feature_extraction(url, mode, feats, feature_columns_81, status):
    try:
        df_old = pd.read_excel(LOG_PATH, engine="openpyxl") if os.path.exists(LOG_PATH) else None
        no = int(df_old["NO"].max()) + 1 if df_old is not None else 1
    except: df_old = None; no = 1
    row = {"NO": no, "URL": url, "MODE": mode, "STATUS": status}
    for feat in feature_columns_81: row[feat] = feats.get(feat, 0.0)
    row["EXPLANATION"] = feats.get("_explanation","")
    row["TOP_FEATURES"] = feats.get("_top_features","")
    row["LLM_REASONING"] = feats.get("_llm_reasoning","")
    row_df = pd.DataFrame([row])
    if df_old is None:
        row_df.to_excel(LOG_PATH, index=False)
    else:
        with pd.ExcelWriter(LOG_PATH, mode="a", engine="openpyxl", if_sheet_exists="overlay") as w:
            row_df.to_excel(w, index=False, header=False, startrow=len(df_old)+1)

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

        # Simpan ke log Excel — catat SEMUA fitur yang punya dampak signifikan
        log_top_lines = []
        all_shap = exp.get("shap_items", [])
        if all_shap:
            log_top_lines.append("=== TOP SHAP FEATURES ===")
            for i, f in enumerate(all_shap):
                feat_val = feats_full.get(f["name"], 0)
                direction = "→ PHISHING" if f["shap_signed"] > 0 else "→ BENIGN"
                _, reason = get_phishing_risk_direction(f["name"], feat_val)
                log_top_lines.append(
                    f"{i+1}. {f['name']} | nilai={feat_val} | SHAP={f['shap_signed']:+.4f} {direction} | {reason}"
                )
        log_top_lines.append("=== SEMUA FITUR BERDAMPAK SIGNIFIKAN ===")
        phishing_contrib = []
        benign_contrib = []
        for feat_name, feat_val in feats_full.items():
            if feat_name.startswith("_"): continue
            dir_, reason_ = get_phishing_risk_direction(feat_name, feat_val)
            if dir_ == "PHISHING":
                phishing_contrib.append(f"  [PHISHING] {feat_name}={feat_val} | {reason_}")
            elif dir_ == "BENIGN":
                benign_contrib.append(f"  [BENIGN]   {feat_name}={feat_val} | {reason_}")
        if phishing_contrib:
            log_top_lines.append("-- Mendorong ke PHISHING:")
            log_top_lines.extend(phishing_contrib)
        if benign_contrib:
            log_top_lines.append("-- Mendorong ke BENIGN:")
            log_top_lines.extend(benign_contrib)
        log_top = "\n".join(log_top_lines)
        feats_for_log = dict(feats_full)
        feats_for_log["_explanation"] = exp.get("llm_reasoning","")
        feats_for_log["_top_features"] = log_top
        feats_for_log["_llm_reasoning"] = exp.get("llm_reasoning","")
        log_feature_extraction(url, mode, feats_for_log, FEATURE_COLUMNS_81, category)

        return jsonify(resp)

    # ── Routing per decision mode ─────────────────────────────────────────────
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
        p = clamp01(0.5*(preds["rf_prob"] or 0.0) + 0.5*(preds["xgb_prob"] or 0.0))
        return build_response(p, "ml_rf_xgb_average_only", "RF + XGB Average", preds["rf_prob"], preds["xgb_prob"], rf_model=rf, xgb_model=xgb)

    # hybrid_prefilter
    if use_prefilter and rule_flag == 1:
        p = clamp01(max(PREFILTER_PHISHING_MIN_CONF, risk_score/10.0))
        return build_response(p, "rule_based_prefilter_phishing", "Rule-Based Prefilter", web_used=False)

    preds = predict_models(feats_full, cols, risk_score, rule_flag, rf, xgb, meta)
    if preds["stack_prob"] is not None:
        return build_response(clamp01(preds["stack_prob"]), "ml_stacking_only", "RF + XGB + Logistic Regression Stacking", preds["rf_prob"], preds["xgb_prob"], preds["stack_prob"], rf_model=rf, xgb_model=xgb)
    p = clamp01(0.5*(preds["rf_prob"] or 0.0) + 0.5*(preds["xgb_prob"] or 0.0))
    return build_response(p, "ml_rf_xgb_average_only", "RF + XGB Average", preds["rf_prob"], preds["xgb_prob"], rf_model=rf, xgb_model=xgb)

@app.post("/predict_url")
@app.post("/analyze")
def predict_alias(): return predict()

if __name__ == "__main__":
    # Memanggil cache memori global sekali saat web backend menyala
    init_startup_cache()
    app.run(debug=True, host="0.0.0.0", port=5000)