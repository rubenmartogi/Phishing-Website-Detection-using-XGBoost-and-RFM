"""
get_prob_untuk_skripsi.py
=========================


Cara pakai:
    python get_prob_untuk_skripsi.py

Output: rf_prob, xgb_prob, dan p_akhir untuk dua URL sampel.
"""
import os
import pickle
import math

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Load model ────────────────────────────────────────────
rf_81  = pickle.load(open(os.path.join(MODELS_DIR, "random_forest_model_81.pkl"), "rb"))
xgb_81 = pickle.load(open(os.path.join(MODELS_DIR, "xgboost_model_81.pkl"),      "rb"))
lr_81  = pickle.load(open(os.path.join(MODELS_DIR, "rule_lr_81.pkl"),             "rb"))

rf_37  = pickle.load(open(os.path.join(MODELS_DIR, "random_forest_model_37.pkl"), "rb"))
xgb_37 = pickle.load(open(os.path.join(MODELS_DIR, "xgboost_model_37.pkl"),      "rb"))
lr_37  = pickle.load(open(os.path.join(MODELS_DIR, "rule_lr_37.pkl"),             "rb"))

cols_81 = [l.strip() for l in open(os.path.join(MODELS_DIR, "feature_columns_81.txt")) if l.strip()]
cols_37 = [l.strip() for l in open(os.path.join(MODELS_DIR, "feature_columns_37.txt")) if l.strip()]

import pandas as pd
import numpy as np


fitur_benign_raw = {
    "length_url": 21, "length_hostname": 12, "ip": 0, "nb_dots": 2,
    "nb_hyphens": 0, "nb_at": 0, "nb_qm": 0, "nb_and": 0, "nb_eq": 0,
    "nb_underscore": 0, "nb_tilde": 0, "nb_percent": 0, "nb_slash": 3,
    "nb_star": 0, "nb_colon": 1, "nb_comma": 0, "nb_semicolumn": 0,
    "nb_dollar": 0, "nb_space": 0, "nb_www": 1, "nb_com": 0, "nb_dslash": 1,
    "http_in_path": 0, "https_token": 0, "ratio_digits_url": 0.0,
    "ratio_digits_host": 0.0, "punycode": 0, "port": 0, "tld_in_path": 0,
    "tld_in_subdomain": 0, "abnormal_subdomain": 0, "nb_subdomains": 1,
    "prefix_suffix": 0, "random_domain": 0, "shortening_service": 0,
    "path_extension": 0, "nb_redirection": 0,
    # fitur extended 18
    "nb_external_redirection": 0, "length_words_raw": 4, "char_repeat": 4,
    "shortest_words_raw": 3, "shortest_word_host": 3, "shortest_word_path": 0,
    "longest_words_raw": 5, "longest_word_host": 4, "longest_word_path": 0,
    "avg_words_raw": 3.75, "avg_word_host": 3.3333, "avg_word_path": 0.0,
    "phish_hints": 0, "domain_in_brand": 0, "brand_in_subdomain": 0,
    "brand_in_path": 0, "suspicious_tld": 0, "statistical_report": 0,
    # fitur web content
    "nb_hyperlinks": 161, "ratio_intHyperlinks": 0.8509, "ratio_extHyperlinks": 0.1491,
    "nb_extCSS": 1, "ratio_extRedirection": 0.0, "ratio_extErrors": 0.0,
    "login_form": 1, "external_favicon": 0, "links_in_tags": 33.333,
    "ratio_intMedia": 75.0, "ratio_extMedia": 25.0, "iframe": 0, "popup_window": 0,
    "safe_anchor": 0.0, "onmouseover": 0, "right_clic": 0, "empty_title": 0,
    "domain_in_title": 1, "domain_with_copyright": 1,
    "whois_registered_domain": 0, "domain_registration_length": 0,
    "domain_age": 0, "web_traffic": 0.0, "dns_record": 1,
    "google_index": 1, "page_rank": 4.78,
}

fitur_phishing_raw = {
    "length_url": 62, "length_hostname": 23, "ip": 0, "nb_dots": 2,
    "nb_hyphens": 0, "nb_at": 0, "nb_qm": 1, "nb_and": 1, "nb_eq": 2,
    "nb_underscore": 0, "nb_tilde": 0, "nb_percent": 0, "nb_slash": 4,
    "nb_star": 0, "nb_colon": 1, "nb_comma": 0, "nb_semicolumn": 0,
    "nb_dollar": 0, "nb_space": 0, "nb_www": 0, "nb_com": 0, "nb_dslash": 1,
    "http_in_path": 0, "https_token": 0, "ratio_digits_url": 0.2581,
    "ratio_digits_host": 0.0435, "punycode": 0, "port": 0, "tld_in_path": 0,
    "tld_in_subdomain": 0, "abnormal_subdomain": 0, "nb_subdomains": 1,
    "prefix_suffix": 0, "random_domain": 0, "shortening_service": 0,
    "path_extension": 0, "nb_redirection": 0,
}

def prediksi(rf, xgb, lr, cols, fitur_raw, label):
    row = [float(fitur_raw.get(c, 0.0)) for c in cols]
    X   = pd.DataFrame([row], columns=cols)

    rf_prob  = float(rf.predict_proba(X)[0, 1])
    xgb_prob = float(xgb.predict_proba(X)[0, 1])

    meta_X   = np.array([[rf_prob, xgb_prob]])
    p_akhir  = float(lr.predict_proba(meta_X)[0, 1])

    print(f"\n{'='*55}")
    print(f"URL: {label}")
    print(f"{'='*55}")
    print(f"  rf_prob   = {rf_prob:.4f}")
    print(f"  xgb_prob  = {xgb_prob:.4f}")
    print(f"  p_akhir (Meta-LR) = {p_akhir:.4f}")
    print()
    print(f"  Rumus Meta-LR:")
    print(f"  p_akhir = LR([{rf_prob:.4f}, {xgb_prob:.4f}]) = {p_akhir:.4f}")


    try:
        coef      = lr.coef_[0]
        intercept = lr.intercept_[0]
        logodds   = intercept + coef[0]*rf_prob + coef[1]*xgb_prob
        prob_manual = 1 / (1 + math.exp(-logodds))
        print()
        print(f"  Formula LR lengkap:")
        print(f"  intercept = {intercept:.4f}")
        print(f"  coef_rf   = {coef[0]:.4f}")
        print(f"  coef_xgb  = {coef[1]:.4f}")
        print()
        print(f"  log-odds = {intercept:.4f} + {coef[0]:.4f}×{rf_prob:.4f} + {coef[1]:.4f}×{xgb_prob:.4f}")
        print(f"           = {intercept:.4f} + {coef[0]*rf_prob:.4f} + {coef[1]*xgb_prob:.4f}")
        print(f"           = {logodds:.4f}")
        print(f"  sigmoid({logodds:.4f}) = {prob_manual:.4f}")
    except Exception as e:
        print(f"  (Formula LR tidak tersedia: {e})")

prediksi(rf_81, xgb_81, lr_81, cols_81, fitur_benign_raw,
         "https://www.ietf.org/ (benign, 81 fitur)")

prediksi(rf_37, xgb_37, lr_37, cols_37, fitur_phishing_raw,
         "http://nmhokdypni.sum4iit.club/ (phishing, 37 fitur)")

print("\n" + "="*55)
print("Simpan output ini untuk dimasukkan ke skripsi.")
print("="*55)
