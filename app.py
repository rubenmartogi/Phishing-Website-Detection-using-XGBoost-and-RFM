import ipaddress
import json
import math
import os
import pickle
import re
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import shap
import tldextract
from flask import Flask, jsonify, request, send_from_directory

from llm_utils import get_llm_reasoning

"""
================================================================================
PHISHING WEBSITE DETECTION - AI MODEL EXPLANATION SYSTEM
================================================================================
"""


def _extract_phishing_shap_vector(shap_values, class_index=1):
    """
    Mengambil SHAP values untuk kelas phishing dari berbagai format output SHAP.
    Output akhir selalu berbentuk vector 1D sepanjang jumlah fitur.
    """
    import numpy as np

    # Format lama: list berisi nilai SHAP per kelas [class_0, class_1]
    if isinstance(shap_values, list):
        if len(shap_values) > class_index:
            vals = shap_values[class_index]
        else:
            vals = shap_values[0]
        return np.array(vals).reshape(np.array(vals).shape[0], -1)[0]

    vals = np.array(shap_values)

    # Format umum: (n_samples, n_features)
    if vals.ndim == 2:
        return vals[0]

    # Format baru multi-output: (n_samples, n_features, n_classes)
    if vals.ndim == 3:
        if vals.shape[-1] > class_index:
            return vals[0, :, class_index]
        return vals[0, :, 0]

    # Fallback terakhir
    return vals.flatten()


def _make_shap_top_items(shap_vals, feature_names, top_n=5):
    """
    Membuat daftar fitur teratas berdasarkan |SHAP|,
    tetapi tetap menyimpan signed SHAP agar arah kontribusi tidak hilang.
    """
    import numpy as np

    shap_vals = np.array(shap_vals, dtype=float).flatten()
    n = min(len(shap_vals), len(feature_names))
    shap_vals = shap_vals[:n]
    feature_names = list(feature_names)[:n]

    top_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    top_items = []

    for i in top_idx:
        signed_value = float(shap_vals[i])
        top_items.append(
            {
                "name": feature_names[i],
                "shap_value_signed": signed_value,
                "abs_shap": float(abs(signed_value)),
                "shap_direction": (
                    "PHISHING"
                    if signed_value > 0
                    else "BENIGN"
                    if signed_value < 0
                    else "NEUTRAL"
                ),
            }
        )
    return top_items


def _human_label(feat_name):
    mapping = {
        "nb_hyperlinks": "Jumlah tautan di halaman",
        "ratio_intHyperlinks": "Persentase tautan internal",
        "ratio_extHyperlinks": "Persentase tautan eksternal",
        "nb_at": "Tanda @ pada URL",
        "random_domain": "Nama domain terlihat acak",
        "shortening_service": "Menggunakan layanan pemendek URL",
        "length_url": "Panjang URL",
        "nb_subdomains": "Banyak subdomain",
        "suspicious_tld": "TLD mencurigakan",
        "domain_in_brand": "Domain termasuk brand terkenal",
        "page_rank": "Reputasi halaman",
        "google_index": "Status terindeks Google",
        "web_traffic": "Traffic website",
        "nb_www": "Penggunaan awalan www",
        # tambahkan mapping lain sesuai kebutuhan
    }
    return mapping.get(feat_name, feat_name.replace("_", " "))


def build_clean_top_lists(top_features, max_items=3):
    increases, decreases = [], []
    # Urutkan berdasarkan kontribusi absolut jika ada
    for f in sorted(top_features, key=lambda x: x.get("abs_shap", 0), reverse=True):
        label = _human_label(f.get("name", ""))
        rule_reason = (f.get("reason") or "").strip()
        shap_abs = float(f.get("abs_shap", 0) or 0)
        shap_dir = f.get("shap_direction") or f.get("impact") or "NEUTRAL"

        # Tentukan level pengaruh dari nilai absolut SHAP (tanpa menampilkan angka):
        if shap_abs >= 0.1:
            level = "kuat"
        elif shap_abs >= 0.03:
            level = "sedang"
        elif shap_abs > 0:
            level = "kecil"
        else:
            level = None

        # Jika rule-based tidak memberikan alasan bermakna, buat alasan ringkas dari SHAP
        if not rule_reason or rule_reason.lower().startswith("tidak ada dampak"):
            if level and shap_dir in ("PHISHING", "BENIGN"):
                reason = f"Memberi sinyal {level} ke {shap_dir}"
            else:
                reason = "Tidak ada indikasi pengaruh yang jelas"
        else:
            reason = rule_reason

        item = {"label": label, "reason": reason}
        if shap_dir == "PHISHING":
            if len(increases) < max_items:
                increases.append(item)
        elif shap_dir == "BENIGN":
            if len(decreases) < max_items:
                decreases.append(item)
        # jika sudah cukup fitur di kedua sisi, keluar
        if len(increases) >= max_items and len(decreases) >= max_items:
            break
    return increases, decreases


def explain_prediction(model, X_row, feature_names, top_n=3, background=None):
    import numpy as np
    import shap

    arr = X_row.values if hasattr(X_row, "values") else np.array(X_row)
    _fn_list = list(feature_names)

    try:
        # Cocok untuk Random Forest dan XGBoost karena keduanya tree-based model.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(arr)
        shap_vals = _extract_phishing_shap_vector(shap_values, class_index=1)
        # Hitung semua SHAP sekaligus — top 5 untuk tampilan, semua untuk log Excel
        all_items = _make_shap_top_items(shap_vals, _fn_list, top_n=len(_fn_list))
        top = all_items[:top_n]

        explanation = ", ".join(
            f"{item['name']} ({item['shap_value_signed']:.4f}; arah={item['shap_direction']})"
            for item in top
        )
        return top, explanation, all_items

    except Exception as e_tree:
        try:
            arr = X_row.values if hasattr(X_row, "values") else np.array(X_row)
            num_features = arr.shape[1]

            # Background manual ini hanya fallback agar KernelExplainer tetap berjalan.
            # Untuk analisis final, background lebih baik diambil dari data training/validasi.
            if background is None:
                if num_features == 2:
                    background = np.array(
                        [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.1, 0.1], [0.9, 0.9]]
                    )
                elif num_features == 3:
                    background = np.array(
                        [
                            [0.0, 0.0, 0],
                            [1.0, 1.0, 1],
                            [0.5, 0.5, 0],
                            [0.7, 0.8, 1],
                            [0.2, 0.1, 0],
                        ]
                    )
                else:
                    background = np.array(
                        [
                            [0.0, 0.0, 0, 0.0],
                            [1.0, 1.0, 1, 8.0],
                            [0.5, 0.5, 0, 2.0],
                            [0.8, 0.7, 1, 6.0],
                            [0.2, 0.3, 0, 1.0],
                        ]
                    )
                    if background.shape[1] < num_features:
                        extra_cols = num_features - background.shape[1]
                        extra = np.full((background.shape[0], extra_cols), 0.5)
                        background = np.hstack([background, extra])
                    elif background.shape[1] > num_features:
                        background = background[:, :num_features]

            # KernelExplainer adalah fallback model-agnostic.
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(arr, nsamples=100)
            shap_vals = _extract_phishing_shap_vector(shap_values, class_index=1)
            all_items_k = _make_shap_top_items(shap_vals, _fn_list, top_n=len(_fn_list))
            top = all_items_k[:top_n]

            explanation = ", ".join(
                f"{item['name']} ({item['shap_value_signed']:.4f}; arah={item['shap_direction']})"
                for item in top
            )
            return top, f"(KernelExplainer) {explanation}", all_items_k

        except Exception as e_kernel:
            return [], f"Penjelasan otomatis gagal: {str(e_kernel)}", []


def get_phishing_risk_direction(feature_name, feature_value):
    """
    Tentukan apakah fitur mengarah ke PHISHING atau BENIGN berdasarkan karakteristik URL
    """
    phishing_indicators = {
        "ip": (lambda v: v == 1, "PHISHING", "IP address digunakan di URL"),
        "nb_at": (lambda v: v >= 1, "PHISHING", "@ symbol dalam URL"),
        "nb_underscore": (lambda v: v > 3, "PHISHING", "Banyak underscore dalam URL"),
        "nb_percent": (lambda v: v > 5, "PHISHING", "Banyak % symbol dalam URL"),
        "nb_tilde": (lambda v: v >= 1, "PHISHING", "~ symbol dalam URL"),
        "nb_semicolumn": (lambda v: v >= 1, "PHISHING", "; symbol dalam URL"),
        "nb_star": (lambda v: v >= 1, "PHISHING", "* symbol dalam URL"),
        "nb_comma": (lambda v: v >= 1, "PHISHING", ", symbol dalam URL"),
        "ratio_digits_url": (lambda v: v > 0.3, "PHISHING", "Banyak angka dalam URL"),
        "nb_subdomains": (lambda v: v > 3, "PHISHING", "Banyak subdomain dalam URL"),
        "random_domain": (
            lambda v: v == 1,
            "PHISHING",
            "Domain terlihat random/tidak punya pola",
        ),
        "prefix_suffix": (lambda v: v == 1, "PHISHING", "Hyphen dalam nama domain"),
        "shortening_service": (
            lambda v: v == 1,
            "PHISHING",
            "Menggunakan URL shortener",
        ),
        "suspicious_tld": (lambda v: v == 1, "PHISHING", "TLD mencurigakan"),
        "length_hostname": (lambda v: v > 30, "PHISHING", "Hostname terlalu panjang"),
        "nb_dots": (lambda v: v > 4, "PHISHING", "Banyak dot dalam URL"),
        "nb_slash": (lambda v: v > 7, "PHISHING", "Banyak slash dalam URL"),
        "nb_qm": (lambda v: v > 2, "PHISHING", "Banyak query parameter"),
        "nb_and": (lambda v: v > 3, "PHISHING", "Banyak & dalam URL"),
        "nb_hyphens": (lambda v: v > 3, "PHISHING", "Banyak hyphen dalam URL"),
        "http_in_path": (lambda v: v == 1, "PHISHING", "HTTP protocol dalam path"),
        "port": (lambda v: v == 1, "PHISHING", "Port tidak standard"),
        "nb_dollar": (lambda v: v >= 1, "PHISHING", "$ symbol dalam URL"),
        "nb_colon": (lambda v: v > 1, "PHISHING", "Banyak colon dalam URL"),
        "nb_redirection": (
            lambda v: v > 0,
            "PHISHING",
            "Multiple redirection dalam URL",
        ),
        "length_url": (lambda v: v > 75, "PHISHING", "URL terlalu panjang"),
        "phish_hints": (
            lambda v: v > 0,
            "PHISHING",
            "Mengandung hint phishing (login, verify, etc)",
        ),
        "domain_in_brand": (
            lambda v: v == 1,
            "BENIGN",
            "Domain termasuk brand terkenal",
        ),
        "brand_in_subdomain": (
            lambda v: v == 1,
            "BENIGN",
            "Brand terkenal di subdomain",
        ),
        "https_token": (
            lambda v: v == 0,
            "BENIGN",
            "Tidak ada HTTPS token tersembunyi",
        ),
        "nb_www": (lambda v: v == 1, "BENIGN", "WWW prefix dalam domain"),
    }

    benign_indicators = {
        "https_token": (lambda v: v == 0, "BENIGN", "HTTPS tidak ada di path"),
        "nb_www": (lambda v: v == 1, "BENIGN", "Memiliki WWW prefix"),
        "length_url": (lambda v: v < 50, "BENIGN", "URL length normal"),
    }

    # SOLUSI: Periksa indikator phishing terlebih dahulu
    if feature_name in phishing_indicators:
        check_fn, direction, reason = phishing_indicators[feature_name]
        if check_fn(feature_value):
            return direction, reason

    # SOLUSI: Jalankan evaluasi untuk benign_indicators jika tidak memicu phishing
    if feature_name in benign_indicators:
        check_fn, direction, reason = benign_indicators[feature_name]
        if check_fn(feature_value):
            return direction, reason

    # Default jika fitur bernilai 0 (kondisi aman/absennya indikator buruk)
    if feature_value == 0:
        if feature_name in [
            "ip",
            "nb_at",
            "suspicious_tld",
            "port",
            "shortening_service",
            "random_domain",
        ]:
            return "BENIGN", f"{feature_name} = 0 (tidak ada indikator phishing)"

    return "NEUTRAL", "Tidak ada dampak signifikan"


def generate_comprehensive_explanation(
    url,
    feature_values,
    feature_names,
    rf_prob,
    xgb_prob,
    stack_prob,
    decision_source,
    rule_detail=None,
    risk_score=None,
    rule_flag=None,
    rf_model=None,
    xgb_model=None,
    final_label=None,
):
    """
    Menghasilkan penjelasan komprehensif yang menggabungkan output model,
    nilai SHAP (jika tersedia), dan rule-based indicators.
    Perubahan utama:
    - Pilih model yang paling 'bertanggung jawab' (active_model) untuk dijelaskan oleh SHAP.
    - Jika SHAP gagal, lakukan fallback yang lebih toleran sehingga tidak menghasilkan top_features kosong.
    """
    # Normalisasi input fitur
    if isinstance(feature_values, list):
        feat_dict = {name: val for name, val in zip(feature_names, feature_values)}
        feat_array = np.array([feature_values], dtype=float)
    else:
        feat_dict = feature_values
        feat_array = np.array(
            [[feature_values.get(n, 0) for n in feature_names]], dtype=float
        )

    # Tentukan probabilitas akhir (mengikuti logika di predict())
    final_phishing_prob = (
        stack_prob
        if stack_prob is not None
        else (
            0.5 * rf_prob + 0.5 * xgb_prob
            if (rf_prob is not None and xgb_prob is not None)
            else (rf_prob or xgb_prob or 0)
        )
    )

    # Jika final_label diberikan (build_response), sesuaikan juga untuk kasus rule-based prefilter
    if final_label is not None:
        final_prediction = "PHISHING" if final_label == 1 else "BENIGN"
        if (
            final_label == 1
            and decision_source == "rule_based_prefilter_phishing"
            and risk_score is not None
        ):
            final_phishing_prob = clamp01(risk_score / 10.0)
    else:
        final_prediction = (
            "BENIGN" if final_phishing_prob < FINAL_THRESHOLD else "PHISHING"
        )

    confidence = (
        final_phishing_prob
        if final_prediction == "PHISHING"
        else (1.0 - final_phishing_prob)
    )
    confidence = max(0.0, min(1.0, float(confidence)))

    # Probabilitas per-model (dipakai untuk memilih penyebab dominan)
    probs = {}
    if rf_prob is not None:
        probs["Random Forest"] = rf_prob
    if xgb_prob is not None:
        probs["XGBoost"] = xgb_prob
    if stack_prob is not None:
        probs["Logistic Regression (Stacking)"] = stack_prob

    # Tentukan model yang dominan untuk ditampilkan
    if decision_source == "rule_based_prefilter_phishing":
        dominant_model = "Rule-Based Detection"
    else:
        # Jika stacking tersedia, tampilkan stacking sebagai model dominan
        if stack_prob is not None:
            dominant_model = "Logistic Regression (Stacking)"
        else:
            dominant_model = max(probs, key=probs.get) if probs else "Unknown"

    # Pilih active_model untuk dijelaskan via SHAP
    active_model = None
    active_model_name = None
    # Prioritaskan explicit decision_mode
    if decision_source == "rf_only":
        active_model = rf_model
        active_model_name = "Random Forest"
    elif decision_source == "xgb_only":
        active_model = xgb_model
        active_model_name = "XGBoost"
    elif decision_source == "rule_based_prefilter_phishing":
        active_model = None
        active_model_name = "Rule-Based Detection"
    else:
        # Untuk stacking/average pilih model dengan probabilitas tertinggi yang tersedia
        try:
            rf_p = float(rf_prob) if rf_prob is not None else -1.0
        except Exception:
            rf_p = -1.0
        try:
            xgb_p = float(xgb_prob) if xgb_prob is not None else -1.0
        except Exception:
            xgb_p = -1.0

        if rf_model is not None and xgb_model is not None:
            if rf_p >= xgb_p:
                active_model = rf_model
                active_model_name = "Random Forest"
            else:
                active_model = xgb_model
                active_model_name = "XGBoost"
        elif rf_model is not None:
            active_model = rf_model
            active_model_name = "Random Forest"
        elif xgb_model is not None:
            active_model = xgb_model
            active_model_name = "XGBoost"
        else:
            active_model = None
            active_model_name = "Unknown"

    top_features = []
    shap_explanation = ""
    all_shap_items = []  # Semua fitur dengan SHAP (untuk log Excel)

    # Coba jelaskan menggunakan SHAP pada active_model terlebih dahulu, jika tersedia
    if active_model is not None:
        try:
            shap_top, shap_explanation, all_shap_items = explain_prediction(
                active_model, feat_array, feature_names, top_n=5
            )
            # Bentuk struktur top_features dari hasil SHAP.
            # impact di-reconcile antara SHAP direction dan rule-based direction.
            for item in shap_top:
                feat_name = item["name"]
                # Beberapa fitur model seperti page_rank/google_index/web_traffic bisa tidak
                # berhasil diekstrak saat runtime dan masuk ke model sebagai default 0.
                # Tetap tampilkan fitur SHAP tersebut agar daftar dampak fitur konsisten 5 item.
                feat_value = feat_dict.get(feat_name, 0)
                rule_direction, reason = get_phishing_risk_direction(
                    feat_name, feat_value
                )
                shap_direction = item.get("shap_direction", "NEUTRAL")
                # Reconcile: rule lebih reliable secara semantik untuk ditampilkan ke user.
                # Jika rule punya pendapat (bukan NEUTRAL), pakai rule sebagai impact display.
                # Ini mencegah fitur seperti nb_www=1 tampil sebagai PHISHING hanya karena
                # signed SHAP kebetulan negatif kecil (noise).
                if rule_direction != "NEUTRAL":
                    display_impact = rule_direction
                else:
                    display_impact = shap_direction
                top_features.append(
                    {
                        "name": feat_name,
                        "value": feat_value,
                        "impact": display_impact,
                        "rule_impact": rule_direction,
                        "reason": reason,
                        "shap_value": float(item["shap_value_signed"]),
                        "shap_value_signed": float(item["shap_value_signed"]),
                        "abs_shap": float(item["abs_shap"]),
                        "shap_direction": shap_direction,
                    }
                )
        except Exception as e:
            # Log debug, tapi jangan hentikan alur
            print(f"[DEBUG] SHAP explanation for {active_model_name} failed: {e}")
            shap_explanation = ""

    # Jika SHAP di active_model gagal atau tidak tersedia, coba model alternatif
    if not top_features and active_model is not None:
        other_model = xgb_model if active_model is rf_model else rf_model
        other_name = "XGBoost" if active_model is rf_model else "Random Forest"
        if other_model is not None:
            try:
                shap_top, shap_explanation, all_shap_items_alt = explain_prediction(
                    other_model, feat_array, feature_names, top_n=5
                )
                for item in shap_top:
                    feat_name = item["name"]
                    feat_value = feat_dict.get(feat_name, 0)
                    rule_direction, reason = get_phishing_risk_direction(
                        feat_name, feat_value
                    )
                    shap_direction = item.get("shap_direction", "NEUTRAL")
                    if rule_direction != "NEUTRAL":
                        display_impact = rule_direction
                    else:
                        display_impact = shap_direction
                    top_features.append(
                        {
                            "name": feat_name,
                            "value": feat_value,
                            "impact": display_impact,
                            "rule_impact": rule_direction,
                            "reason": reason,
                            "shap_value": float(item["shap_value_signed"]),
                            "shap_value_signed": float(item["shap_value_signed"]),
                            "abs_shap": float(item["abs_shap"]),
                            "shap_direction": shap_direction,
                        }
                    )
                # Jika berhasil, ubah nama model dominan yang dijelaskan
                if top_features:
                    active_model_name = other_name
                    all_shap_items = all_shap_items_alt
            except Exception as e:
                print(
                    f"[DEBUG] SHAP alternative explanation ({other_name}) failed: {e}"
                )
                shap_explanation = ""

    # Fallback: jika SHAP tidak menghasilkan fitur apapun, gunakan rule-based heuristic
    if not top_features:
        non_neutral = []
        for feat_name, feat_value in feat_dict.items():
            direction, reason = get_phishing_risk_direction(feat_name, feat_value)
            if direction != "NEUTRAL":
                non_neutral.append(
                    {
                        "name": feat_name,
                        "value": feat_value,
                        "impact": direction,
                        "reason": reason,
                        "shap_value": None,
                    }
                )
        # Jika ada, ambil hingga 5 dengan prioritas PHISHING
        if non_neutral:
            non_neutral = sorted(
                non_neutral, key=lambda x: 0 if x["impact"] == "PHISHING" else 1
            )
            top_features = non_neutral[:5]
        else:
            # Jika tidak ada indikator non-neutral, pilih fitur berdasar magnitudo nilai (non-zero dulu)
            scored = []
            for feat_name, feat_value in feat_dict.items():
                try:
                    score = abs(float(feat_value))
                except Exception:
                    score = 0.0
                if score > 0:
                    scored.append((feat_name, score))
            scored = sorted(scored, key=lambda x: x[1], reverse=True)
            chosen = [n for n, s in scored][:5]
            if chosen:
                for feat_name in chosen:
                    direction, reason = get_phishing_risk_direction(
                        feat_name, feat_dict.get(feat_name)
                    )
                    top_features.append(
                        {
                            "name": feat_name,
                            "value": feat_dict.get(feat_name),
                            "impact": direction,
                            "reason": reason,
                            "shap_value": None,
                        }
                    )
            else:
                # Sebagai cadangan terakhir, ambil beberapa fitur default bila tersedia
                defaults = [
                    "length_url",
                    "length_hostname",
                    "nb_subdomains",
                    "ratio_digits_url",
                    "suspicious_tld",
                ]
                for d in defaults:
                    if d in feat_dict:
                        direction, reason = get_phishing_risk_direction(
                            d, feat_dict.get(d)
                        )
                        top_features.append(
                            {
                                "name": d,
                                "value": feat_dict.get(d),
                                "impact": direction,
                                "reason": reason,
                                "shap_value": None,
                            }
                        )
                top_features = top_features[:5]

    phishing_count = sum(1 for f in top_features if f.get("impact") == "PHISHING")
    benign_count = sum(1 for f in top_features if f.get("impact") == "BENIGN")

    # SHAP adalah kontribusi berbobot, bukan voting.
    # Hitung total bobot (abs SHAP) per arah dari top-5 agar bisa dijelaskan
    # mengapa jumlah fitur lebih banyak ke PHISHING belum tentu menaikkan skor.
    top5_phishing_shap = round(
        sum(
            float(f.get("abs_shap") or 0)
            for f in top_features
            if f.get("shap_direction") == "PHISHING"
        ),
        4,
    )
    top5_benign_shap = round(
        sum(
            float(f.get("abs_shap") or 0)
            for f in top_features
            if f.get("shap_direction") == "BENIGN"
        ),
        4,
    )
    total_feature_count = len(feature_names) if feature_names else 0

    phishing_prob_display = round(final_phishing_prob, 4)
    benign_prob_display = round(1.0 - final_phishing_prob, 4)

    # Susun reasoning textual yang informatif
    if final_prediction == "BENIGN":
        if final_phishing_prob < FINAL_THRESHOLD:
            reasoning = f"🟢 URL terdeteksi sebagai BENIGN dengan phishing probability {phishing_prob_display:.2%}. "
            if shap_explanation:
                reasoning += f"(SHAP) Top features: {shap_explanation}. "
            # Jelaskan paradoks jumlah vs bobot agar tidak membingungkan
            reasoning += (
                f"Dari {total_feature_count} fitur yang dianalisis, "
                f"{phishing_count} dari 5 fitur terkuat mengarah ke PHISHING "
                f"(total bobot={top5_phishing_shap:.4f}) dan "
                f"{benign_count} mengarah ke BENIGN "
                f"(total bobot={top5_benign_shap:.4f}). "
                f"SHAP bukan sistem voting — selisih bobot top-5 sangat kecil, "
                f"dan sisa {total_feature_count - 5} fitur lainnya secara kumulatif mendukung BENIGN "
                f"sehingga skor akhir tetap rendah. "
            )
            reasoning += "Website ini AMAN untuk dikunjungi."
        else:
            reasoning = f"URL dikategorikan sebagai BENIGN (phishing probability: {phishing_prob_display:.2%})."
    else:
        reasoning = f"🔴 URL terdeteksi sebagai PHISHING dengan phishing probability {phishing_prob_display:.2%}. "
        if shap_explanation:
            reasoning += f"(SHAP) Top features: {shap_explanation}. "
        if phishing_count > 0:
            reasoning += (
                f"{phishing_count} dari 5 fitur terkuat menunjukkan indikator phishing "
                f"(total bobot SHAP={top5_phishing_shap:.4f}). "
            )
        if decision_source == "rule_based_prefilter_phishing" and rule_detail:
            reasoning += f"Keputusan rule-based dipicu oleh: {rule_detail}. "
        reasoning += "Website ini TIDAK AMAN untuk dikunjungi."

    explanation = {
        "final_prediction": final_prediction,
        "confidence_score": confidence,
        "phishing_probability": final_phishing_prob,
        "benign_probability": 1.0 - final_phishing_prob,
        "main_contributing_model": (
            "Rule-Based Detection"
            if decision_source == "rule_based_prefilter_phishing"
            else (
                "Logistic Regression (Stacking)"
                if stack_prob is not None
                else active_model_name
            )
        ),
        "top_influential_features": top_features,
        "all_shap_features": all_shap_items,
        "model_contribution_probability": probs,
        "phishing_indicators_count": phishing_count,
        "benign_indicators_count": benign_count,
        "top5_phishing_shap_sum": top5_phishing_shap,
        "top5_benign_shap_sum": top5_benign_shap,
        "total_feature_count": total_feature_count,
        "ai_reasoning": reasoning,
        "shap_explanation": shap_explanation,
        "detailed_explanation": format_explanation_text(
            final_prediction,
            confidence,
            (
                "Rule-Based Detection"
                if decision_source == "rule_based_prefilter_phishing"
                else (
                    "Logistic Regression (Stacking)"
                    if stack_prob is not None
                    else active_model_name
                )
            ),
            top_features,
            probs,
            reasoning,
        ),
    }
    # Prepare structured input for LLM: tiga lapis (asal skor, menaikkan, menurunkan)
    try:
        increases, decreases = build_clean_top_lists(
            explanation.get("top_influential_features", []), max_items=5
        )
        # Perbarui top_influential_features agar alasan tidak lagi "Tidak ada dampak signifikan"
        tf = explanation.get("top_influential_features", []) or []
        updated_tf = []
        for f in tf:
            rule_reason = (f.get("reason") or "").strip()
            shap_abs = float(f.get("abs_shap", 0) or 0)
            shap_dir = f.get("shap_direction") or f.get("impact") or "NEUTRAL"
            if not rule_reason or rule_reason.lower().startswith("tidak ada dampak"):
                if shap_abs >= 0.1:
                    level = "kuat"
                elif shap_abs >= 0.03:
                    level = "sedang"
                elif shap_abs > 0:
                    level = "kecil"
                else:
                    level = None
                if level and shap_dir in ("PHISHING", "BENIGN"):
                    reason = f"Memberi sinyal {level} ke {shap_dir}"
                else:
                    reason = "Tidak ada indikasi pengaruh yang jelas"
                f["reason"] = reason
            updated_tf.append(f)
        explanation["top_influential_features"] = updated_tf

        explanation_payload = {
            "score_origin": {
                "model": explanation.get("main_contributing_model"),
                "phishing_probability": round(
                    explanation.get("phishing_probability", 0.0), 4
                ),
                "total_features_analyzed": explanation.get("total_feature_count", 0),
            },
            "shap_balance_top5": {
                "phishing_direction_weight": explanation.get(
                    "top5_phishing_shap_sum", 0
                ),
                "benign_direction_weight": explanation.get("top5_benign_shap_sum", 0),
                "note": (
                    "Ini adalah bobot kontribusi model (bukan voting). "
                    "Jumlah fitur mengarah PHISHING bisa lebih banyak tetapi jika bobotnya kecil "
                    "dan sisa fitur mendukung BENIGN, skor akhir tetap rendah."
                ),
            },
            "increases": increases,
            "decreases": decreases,
        }
        explanation["llm_input_structured"] = explanation_payload

        prompt_llm = f"""
Anda asisten ringkas. Tuliskan penjelasan singkat dalam bahasa Indonesia (bahasa awam) dengan tepat 3 lapis:
1) Asal skor akhir: 1 kalimat yang jelaskan model dominan dan bahwa skor adalah probabilitas.
2) Fitur yang paling MENAIKKAN risiko: sebut hingga 5 fitur teratas dengan label singkat + 1 kalimat alasan tiap fitur. Jangan tampilkan angka atau istilah teknis.
3) Fitur yang paling MENURUNKAN risiko: sebut hingga 5 fitur (jika ada) serupa formatnya.
Akhiri dengan 1 kalimat rekomendasi: "Aman untuk dikunjungi" atau "Tidak aman — hindari".
PENTING:
- Jangan gunakan istilah 'SHAP', 'koefisien', atau nilai teknis.
- Tidak boleh menambahkan alasan di luar daftar fitur yang diberikan.
- Jika jumlah fitur mengarah PHISHING lebih banyak tapi skor akhir tetap rendah/BENIGN,
  jelaskan dengan bahasa awam bahwa model memakai bobot kontribusi (bukan voting)
  dan fitur-fitur lainnya di luar top-5 secara keseluruhan mendukung BENIGN.
Gunakan input terstruktur berikut (format JSON) untuk menghasilkan teks:
{json.dumps(explanation_payload, ensure_ascii=False, indent=2)}
Buat keseluruhan output sekitar 5–8 kalimat agar alasannya cukup jelas dan tidak membingungkan.
"""

        try:
            llm_out = get_llm_reasoning(prompt_llm)
        except Exception:
            llm_out = None
        explanation["llm_reasoning"] = llm_out
        explanation["_llm_prompt"] = prompt_llm
    except Exception:
        # jika ada kegagalan pembuatan payload/LLM, jangan ganggu alur utama
        pass

    return explanation


def format_explanation_text(
    final_prediction, confidence, main_model, top_features, probabilities, reasoning
):
    text = f"""
═══════════════════════════════════════════════════════════════
📋 AI MODEL EXPLANATION - PHISHING WEBSITE DETECTION (BENIGN FOCUS)
═══════════════════════════════════════════════════════════════

🎯 FINAL PREDICTION
   Status: {"🟢 BENIGN (SAFE)" if final_prediction == "BENIGN" else "🔴 PHISHING (DANGEROUS)"}
   Classification Confidence: {confidence:.2%}
   Prediction Range: {f"0.00-{FINAL_THRESHOLD:.2f} (BENIGN)" if final_prediction == "BENIGN" else f">={FINAL_THRESHOLD:.2f} (PHISHING)"}

🔍 DECISION ANALYSIS
   Main Contributing Model: {main_model}

⭐ TOP INFLUENTIAL FEATURES (Based on SHAP)
"""
    for i, feat in enumerate(top_features, 1):
        direction_emoji = "🔴" if feat["impact"] == "PHISHING" else "🟢"
        shap_val = feat.get("shap_value", None)
        shap_info = f" (SHAP: {shap_val:.4f})" if shap_val is not None else ""

        text += f"\n   {i}. {feat['name']} = {feat['value']}{shap_info}"
        text += f"\n      {direction_emoji} Impact: {feat['impact']}"
        text += f"\n      📌 Reason: {feat['reason']}"

    text += "\n\n📊 MODEL CONTRIBUTION PROBABILITY\n"
    for model_name, prob in probabilities.items():
        text += f"   • {model_name}: {prob:.2%} (Phishing)\n"

    text += f"\n💡 AI REASONING\n   {reasoning}\n"
    text += "\n📌 EXPLANATION FOCUS: BENIGN Range (0.00-0.60)\n"
    text += "   URLs with phishing probability < 0.60 are classified as BENIGN (SAFE)\n"
    text += "   This explanation emphasizes features supporting the BENIGN classification.\n"
    text += "\n═══════════════════════════════════════════════════════════════\n"

    return text


# --- (Sisanya adalah fungsi ekstraksi fitur bawaanmu yang tidak bermasalah) ---
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

PHISHING_CLASS_VALUE = 1
FINAL_THRESHOLD = 0.6
DEFAULT_PREDICT_MODE = "url37"
DEFAULT_USE_PREFILTER = True
DEFAULT_DECISION_MODE = "hybrid_prefilter"
PREFILTER_HARD_PHISHING_SCORE = 5
PREFILTER_BLOCK_ON_VI_HIT = True
PREFILTER_PHISHING_MIN_CONF = 0.95
WEB_FETCH_TIMEOUT = 6
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

SUSPICIOUS_TLD = [
    "zip",
    "xyz",
    "top",
    "tk",
    "ga",
    "ml",
    "gq",
    "cf",
    "pw",
    "cc",
    "club",
    "ws",
    "biz",
    "online",
    "site",
    "live",
    "work",
    "icu",
    "info",
    "cn",
    "ru",
    "loan",
    "download",
    "click",
]
STANDARD_PORTS = {21, 22, 23, 80, 443, 445, 1433, 1521, 3306, 3389}
SHORTENERS = {
    "bit.ly",
    "goo.gl",
    "tinyurl.com",
    "ow.ly",
    "t.co",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "bit.do",
    "cutt.ly",
}
PHISH_HINTS = [
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "bank",
    "paypal",
    "apple",
    "microsoft",
    "confirm",
    "signin",
    "password",
]
BRANDS = [
    "google",
    "facebook",
    "apple",
    "microsoft",
    "amazon",
    "paypal",
    "instagram",
    "whatsapp",
    "telegram",
    "netflix",
    "github",
    "linkedin",
]

WEB_CONTENT_KEYS = {
    "nb_hyperlinks",
    "ratio_intHyperlinks",
    "ratio_extHyperlinks",
    "nb_extCSS",
    "ratio_extRedirection",
    "ratio_extErrors",
    "login_form",
    "external_favicon",
    "links_in_tags",
    "ratio_intMedia",
    "ratio_extMedia",
    "iframe",
    "popup_window",
    "safe_anchor",
    "onmouseover",
    "right_clic",
    "empty_title",
    "domain_in_title",
    "domain_with_copyright",
    "whois_registered_domain",
    "domain_registration_length",
    "domain_age",
    "web_traffic",
    "dns_record",
    "google_index",
    "page_rank",
    "statistical_report",
    "nb_external_redirection",
}

_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def app_path(*parts):
    return os.path.join(BASE_DIR, *parts)


app = Flask(__name__, static_folder=app_path("static"))
LOG_PATH = app_path("log_feature_extraction.xlsx")


def log_feature_extraction(url, mode, feats, feature_columns, status):
    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_excel(LOG_PATH)
            no = int(df["NO"].max()) + 1
        except Exception:
            df = None
            no = 1
    else:
        df = None
        no = 1
    row = {"NO": no, "URL": url, "MODE": mode, "STATUS": status}
    for feat in feature_columns:
        row[feat] = feats.get(feat, 0.0)
    row["EXPLANATION"] = feats.get("_explanation", "")
    row["TOP_FEATURES"] = feats.get("_top_features", "")
    row["LLM_REASONING"] = feats.get("_llm_reasoning", "")
    row_df = pd.DataFrame([row])
    if df is None:
        row_df.to_excel(LOG_PATH, index=False)
    else:
        with pd.ExcelWriter(
            LOG_PATH, mode="a", engine="openpyxl", if_sheet_exists="overlay"
        ) as writer:
            row_df.to_excel(writer, index=False, header=False, startrow=len(df) + 1)


try:
    FEATURE_COLUMNS_HYBRID81 = [
        line.strip()
        for line in open(app_path("feature_columns_hybrid81.txt"), encoding="utf-8")
        if line.strip()
    ]
except Exception:
    FEATURE_COLUMNS_HYBRID81 = []

# Alias sementara supaya kode lama yang masih refer ke FEATURE_COLUMNS_81 tidak error.
FEATURE_COLUMNS_81 = FEATURE_COLUMNS_HYBRID81


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_columns(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []


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


def is_ip(hostname: str) -> int:
    try:
        ipaddress.ip_address(hostname)
        return 1
    except Exception:
        return 0


def _extract_parts(hostname: str):
    ext = _TLD_EXTRACTOR(hostname or "")
    return (
        (ext.subdomain or "").lower(),
        (ext.domain or "").lower(),
        (ext.suffix or "").lower(),
    )


def _word_stats(text: str):
    words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    if not words:
        return 0, 0, 0, 0.0
    lens = [len(w) for w in words]
    return len(words), min(lens), max(lens), float(sum(lens)) / len(lens)


def _same_or_subdomain(host: str, base_host: str) -> bool:
    host, base_host = (host or "").lower(), (base_host or "").lower()
    if not host or not base_host:
        return False
    return (
        host == base_host
        or host.endswith("." + base_host)
        or base_host.endswith("." + host)
    )


def _is_unsafe_anchor(href: str) -> bool:
    h = (href or "").strip().lower()
    return h == "" or h == "#" or h.startswith("javascript:") or h.startswith("mailto:")


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
        _, _, host, _ = parse_url(url)
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
    return 1 if len(classes) == 2 and fallback_idx >= 1 else 0


def _proba_for_class(model, X, class_value):
    try:
        probs = model.predict_proba(X)[0]
        idx = _class_index(model, class_value, fallback_idx=1)
        if idx >= len(probs):
            idx = len(probs) - 1
        return float(probs[idx])
    except Exception:
        return 1.0 if int(model.predict(X)[0]) == class_value else 0.0


def _label_to_phishing_flag(raw_label: int) -> int:
    return 1 if int(raw_label) == PHISHING_CLASS_VALUE else 0


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


def parse_url(url: str):
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    parsed = urlparse(u)
    return u, parsed, (parsed.hostname or "").lower(), parsed.path or ""


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
    port_flag = (
        1 if (parsed_port is not None and parsed_port not in STANDARD_PORTS) else 0
    )
    tld_last_label = tld.split(".")[-1] if tld else ""
    suspicious_tld_flag = (
        1 if (tld in SUSPICIOUS_TLD or tld_last_label in SUSPICIOUS_TLD) else 0
    )

    domain_in_brand = 1 if any(b in domain for b in BRANDS) else 0
    brand_in_subdomain = 1 if any(b in subdomain for b in BRANDS) else 0
    brand_in_path = 1 if any(b in path.lower() for b in BRANDS) else 0
    statistical_report = (
        1
        if (
            suspicious_tld_flag == 1
            or is_ip(hostname) == 1
            or full.count("@") >= 1
            or random_domain == 1
        )
        else 0
    )

    return {
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


def extract_web_content_features(url: str) -> dict:
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
        1
        for r in (resp.history or [])
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
        if any(k in h.lower() for k in ["redirect=", "redir=", "url=", "next="]):
            ext_redir += 1
    out["ratio_extRedirection"] = (ext_redir / ext_a) if ext_a else 0.0
    out["ratio_extErrors"] = 0.0

    css_links = [
        l.get("href", "")
        for l in soup.find_all("link")
        if "stylesheet" in " ".join((l.get("rel") or [])).lower()
    ]
    out["nb_extCSS"] = sum(
        1 for h in css_links if _is_external_href(h, final_url, base_host)
    )

    login_form = 0
    for f in soup.find_all("form"):
        has_pwd = bool(f.find("input", {"type": re.compile("password", re.I)}))
        action = f.get("action", "")
        if has_pwd or (
            _is_external_href(action, final_url, base_host) if action else False
        ):
            login_form = 1
            break
    out["login_form"] = login_form

    favicon_external = 0
    for l in soup.find_all("link"):
        if "icon" in " ".join((l.get("rel") or [])).lower() and _is_external_href(
            l.get("href", ""), final_url, base_host
        ):
            favicon_external = 1
            break
    out["external_favicon"] = favicon_external

    tag_urls = []
    for t in soup.find_all(["link", "script", "meta"]):
        u = (
            t.get("href", "")
            if t.name == "link"
            else t.get("src", "")
            if t.name == "script"
            else ""
        )
        if t.name == "meta":
            m = re.search(r"url=([^;]+)$", (t.get("content", "") or "").lower())
            u = m.group(1).strip() if m else ""
        if u:
            tag_urls.append(u)
    out["links_in_tags"] = (
        (
            sum(1 for u in tag_urls if _is_external_href(u, final_url, base_host))
            / len(tag_urls)
        )
        * 100.0
        if tag_urls
        else 0.0
    )

    media_urls = [
        t.get("src", "") or t.get("data-src", "")
        for t in soup.find_all(["img", "audio", "embed", "source", "video", "track"])
    ]
    media_urls = [u for u in media_urls if u]
    if media_urls:
        ext_m = sum(1 for u in media_urls if _is_external_href(u, final_url, base_host))
        out["ratio_intMedia"] = ((len(media_urls) - ext_m) / len(media_urls)) * 100.0
        out["ratio_extMedia"] = (ext_m / len(media_urls)) * 100.0
    else:
        out["ratio_intMedia"] = out["ratio_extMedia"] = 0.0

    html_lower = html.lower()
    out["iframe"] = 1 if soup.find("iframe") else 0
    out["popup_window"] = 1 if "window.open(" in html_lower else 0
    out["onmouseover"] = 1 if "onmouseover" in html_lower else 0
    out["right_clic"] = (
        1 if ("contextmenu" in html_lower or "event.button==2" in html_lower) else 0
    )
    out["safe_anchor"] = (
        (sum(1 for h in anchors if _is_unsafe_anchor(h)) / total_a) * 100.0
        if total_a
        else 0.0
    )

    title = (
        soup.title.string.strip().lower() if soup.title and soup.title.string else ""
    )
    out["empty_title"] = 1 if not title else 0
    _, domain, _ = _extract_parts(base_host)
    out["domain_in_title"] = 1 if (domain and title and domain in title) else 0
    page_text = soup.get_text(" ", strip=True).lower()
    out["domain_with_copyright"] = (
        1
        if (
            (("©" in page_text) or ("copyright" in page_text))
            and domain
            and domain in page_text
        )
        else 0
    )
    return out


def extract_features(url: str, include_web_content: bool) -> dict:
    full_url, _, _, _ = parse_url(url)
    feats = extract_url_features(full_url)
    if include_web_content:
        feats.update(extract_web_content_features(full_url))
    return feats


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
    moderate_rules = {
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
    less_hits = [k for k, v in moderate_rules.items() if v]
    risk_score = (2 * len(imp_hits)) + len(less_hits)

    if len(vi_hits) >= 1 or risk_score >= 5:
        category, rule_flag = "Phishing", 1
    elif risk_score > 0:
        category, rule_flag = "Suspicious", 0
    else:
        category, rule_flag = "Benign", 0

    if return_detail:
        return (
            risk_score,
            category,
            rule_flag,
            {
                "vi_count": len(vi_hits),
                "imp_count": len(imp_hits),
                "less_count": len(less_hits),
                "vi_hits": vi_hits,
                "imp_hits": imp_hits,
                "less_hits": less_hits,
            },
        )
    return risk_score, category, rule_flag


def get_model_and_features(url):
    full_url, _, _, _ = parse_url(url)

    # Cek web-content secara terpisah.
    # Nilai 0 pada fitur web-content tetap valid, jadi jangan pakai logika
    # "ada nilai non-zero" untuk memilih model.
    web_feats = extract_web_content_features(full_url)
    web_content_ok = bool(web_feats)

    suffix = "_hybrid81" if web_content_ok else "_url37"

    rf = load_model(app_path(f"random_forest_model{suffix}.pkl"))
    xgb = load_model(app_path(f"xgboost_model{suffix}.pkl"))

    try:
        meta = load_model(app_path(f"rule_lr{suffix}.pkl"))
    except Exception:
        meta = None

    cols = load_feature_columns(app_path(f"feature_columns{suffix}.txt"))

    if not cols:
        raise RuntimeError(
            f"Feature columns kosong/tidak ditemukan untuk suffix {suffix}"
        )

    return rf, xgb, meta, cols, web_content_ok


def build_ml_vector(url: str, cols: list, include_web_content: bool):
    feats = extract_features(url, include_web_content=include_web_content)
    vector = [to_float(feats.get(c, 0.0), 0.0) for c in cols]
    return np.array([vector], dtype=float), cols


def predict_models(
    url: str,
    cols: list,
    include_web_content: bool,
    precomputed_rule_score: float = None,
    precomputed_rule_flag: int = None,
    rf=None,
    xgb=None,
    meta=None,
):
    X, used_cols = build_ml_vector(url, cols, include_web_content)
    features_df = pd.DataFrame(X, columns=used_cols)
    rf_raw, xgb_raw = int(rf.predict(features_df)[0]), int(xgb.predict(features_df)[0])
    rf_prob = _proba_for_class(rf, features_df, PHISHING_CLASS_VALUE)
    xgb_prob = _proba_for_class(xgb, features_df, PHISHING_CLASS_VALUE)
    stack_pred, stack_prob, meta_n_in = None, None, None
    if meta is not None:
        try:
            meta_n_in = int(getattr(meta, "n_features_in_", 2))
        except Exception:
            meta_n_in = 2
        rule_score = (
            precomputed_rule_score
            if precomputed_rule_score is not None
            else rule_based_eval(url)[0]
        )
        rule_flag = (
            precomputed_rule_flag
            if precomputed_rule_flag is not None
            else rule_based_eval(url)[2]
        )
        meta_feats = [rf_prob, xgb_prob]
        if meta_n_in >= 3:
            meta_feats = [rf_prob, xgb_prob, rule_flag]
        if meta_n_in >= 4:
            meta_feats = [rf_prob, xgb_prob, rule_flag, rule_score]
        meta_X = np.array([meta_feats[:meta_n_in]], dtype=float)
        try:
            stack_pred = _label_to_phishing_flag(int(meta.predict(meta_X)[0]))
            stack_prob = _proba_for_class(meta, meta_X, PHISHING_CLASS_VALUE)
        except Exception:
            pass
    return {
        "rf_raw": rf_raw,
        "xgb_raw": xgb_raw,
        "rf_pred": _label_to_phishing_flag(rf_raw),
        "xgb_pred": _label_to_phishing_flag(xgb_raw),
        "rf_prob": rf_prob,
        "xgb_prob": xgb_prob,
        "stack_pred": stack_pred,
        "stack_prob": stack_prob,
        "meta_n_in": meta_n_in,
        "rf_classes": _jsonable_classes(rf),
        "xgb_classes": _jsonable_classes(xgb),
        "meta_classes": _jsonable_classes(meta),
    }


@app.get("/")
def index():
    return send_from_directory(
        "Phishing_detection_app", "advanced_hybrid_detector.html"
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def to_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(val, (int, float)):
        return bool(val)
    return default


def normalize_decision_mode(mode):
    if not mode:
        return DEFAULT_DECISION_MODE
    m = str(mode).strip().lower()
    if m in ("rf", "rf_only"):
        return "rf_only"
    if m in ("xgb", "xgb_only"):
        return "xgb_only"
    if m in ("stack", "stacking", "ml_stacking_only"):
        return "ml_stacking_only"
    if m in ("hybrid", "hybrid_prefilter"):
        return "hybrid_prefilter"
    return m


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    debug = to_bool(data.get("debug", False), False)
    use_prefilter = to_bool(
        data.get("use_prefilter", DEFAULT_USE_PREFILTER), DEFAULT_USE_PREFILTER
    )
    decision_mode = normalize_decision_mode(
        data.get("decision_mode") or data.get("mode")
    )

    if not url:
        return jsonify({"error": "URL kosong"}), 400
    if not is_valid_url(url):
        return jsonify({"error": "URL tidak valid. Contoh: https://example.com"}), 400

    rf, xgb, meta, cols, web_content_ok = get_model_and_features(url)
    include_web_content = web_content_ok
    feats_full = extract_features(url, include_web_content=True)
    mode = "hybrid81" if web_content_ok else "url37"
    risk_score, risk_category, rule_flag, rule_detail = rule_based_eval(
        url, return_detail=True
    )

    def build_response(
        final_phishing_prob,
        decision_source,
        model_name,
        rf_prob=None,
        xgb_prob=None,
        stack_prob=None,
        web_used=None,
        rf_model=None,
        xgb_model=None,
    ):
        p_phish = clamp01(final_phishing_prob)
        p_safe = clamp01(1.0 - p_phish)
        final_label = 1 if p_phish >= FINAL_THRESHOLD else 0
        category = "phishing" if final_label == 1 else "benign"
        confidence = round(p_phish if final_label == 1 else p_safe, 4)

        try:
            comprehensive_exp = generate_comprehensive_explanation(
                url=url,
                feature_values=feats_full,
                feature_names=cols,
                rf_prob=rf_prob,
                xgb_prob=xgb_prob,
                stack_prob=stack_prob,
                decision_source=decision_source,
                rule_detail=rule_detail,
                risk_score=risk_score,
                rule_flag=rule_flag,
                rf_model=rf_model,
                xgb_model=xgb_model,
                final_label=final_label,
            )
            explanation_dict = {
                "final_prediction": comprehensive_exp["final_prediction"],
                "confidence_score": round(comprehensive_exp["confidence_score"], 4),
                "phishing_probability": round(
                    comprehensive_exp["phishing_probability"], 4
                ),
                "benign_probability": round(comprehensive_exp["benign_probability"], 4),
                "main_contributing_model": comprehensive_exp["main_contributing_model"],
                "top_influential_features": [
                    {
                        "name": f["name"],
                        "value": f["value"],
                        "impact": f["impact"],
                        "reason": f["reason"],
                        "shap_value": f.get("shap_value", None),
                        "shap_value_signed": f.get(
                            "shap_value_signed", f.get("shap_value", None)
                        ),
                        "abs_shap": f.get("abs_shap", None),
                        "shap_direction": f.get(
                            "shap_direction", f.get("impact", None)
                        ),
                        "rule_impact": f.get("rule_impact", None),
                    }
                    for f in comprehensive_exp["top_influential_features"]
                ],
                "model_contribution_probability": {
                    str(k): round(v, 4)
                    for k, v in comprehensive_exp[
                        "model_contribution_probability"
                    ].items()
                },
                "phishing_indicators_count": comprehensive_exp[
                    "phishing_indicators_count"
                ],
                "benign_indicators_count": comprehensive_exp["benign_indicators_count"],
                "ai_reasoning": comprehensive_exp["ai_reasoning"],
                "shap_explanation": comprehensive_exp.get("shap_explanation", ""),
                "detailed_explanation": comprehensive_exp["detailed_explanation"],
                # Teruskan semua SHAP values untuk log Excel
                "all_shap_features": comprehensive_exp.get("all_shap_features", []),
            }
        except Exception as e:
            explanation_dict = {
                "final_prediction": "PHISHING" if final_label == 1 else "BENIGN",
                "confidence_score": confidence,
                "phishing_probability": round(p_phish, 4),
                "benign_probability": round(p_safe, 4),
                "main_contributing_model": model_name,
                "top_influential_features": [],
                "model_contribution_probability": {
                    "Random Forest": round(rf_prob, 4) if rf_prob else 0.0,
                    "XGBoost": round(xgb_prob, 4) if xgb_prob else 0.0,
                    "Logistic Regression (Stacking)": round(stack_prob, 4)
                    if stack_prob
                    else 0.0,
                },
                "ai_reasoning": "Penjelasan detail tidak tersedia.",
                "shap_explanation": "",
                "detailed_explanation": "",
            }

        resp = {
            "final_label": final_label,
            "final_phishing_prob": round(p_phish, 4),
            "final_safe_prob": round(p_safe, 4),
            "final_threshold": FINAL_THRESHOLD,
            "decision_mode": decision_mode,
            "decision_source": decision_source,
            "model_name": model_name,
            "rule_flag": rule_flag,
            "model_feature_count": len(cols),
            "web_content_used": include_web_content
            if web_used is None
            else bool(web_used),
            "mode": "single_pipeline",
            "rf_prob": round(rf_prob, 4) if rf_prob else None,
            "xgb_prob": round(xgb_prob, 4) if xgb_prob else None,
            "stack_prob": round(stack_prob, 4) if stack_prob else None,
            "label": category,
            "category": category,
            "confidence": confidence,
        }

        # FIX: Masukkan explanation_dict ke json response agar tidak hilang
        resp["explanation_detailed"] = explanation_dict
        resp["explanation"] = explanation_dict["ai_reasoning"]

        # ── AI Reasoning: narasi alami ────────────────────────────
        model_main = explanation_dict.get("main_contributing_model", model_name)
        total_f_count = len(cols)
        all_shap_all = explanation_dict.get("all_shap_features", [])
        top5_names = [
            f.get("name", "")
            for f in explanation_dict.get("top_influential_features", [])
        ]

        # Pisahkan top-5 per arah untuk narasi
        p_top5 = [
            f
            for f in explanation_dict.get("top_influential_features", [])
            if f.get("shap_direction") == "PHISHING"
        ]
        b_top5 = [
            f
            for f in explanation_dict.get("top_influential_features", [])
            if f.get("shap_direction") == "BENIGN"
        ]
        p_names_str = ", ".join(f.get("name", "") for f in p_top5)
        b_names_str = ", ".join(f.get("name", "") for f in b_top5)
        other_count = total_f_count - 5
        result_word = "BENIGN" if category == "benign" else "PHISHING"
        result_icon = "\u2705" if category == "benign" else "\u26a0\ufe0f"

        # Bangun narasi alami tanpa menyebut angka teknis
        if category == "benign":
            if p_names_str and b_names_str:
                body = (
                    f"Beberapa karakteristik URL seperti **{p_names_str}** "
                    f"menunjukkan indikator yang perlu diperhatikan, "
                    f"namun hal ini diimbangi oleh indikator positif yang lebih dominan "
                    f"dari **{b_names_str}** yang mendukung keamanan situs."
                )
            elif b_names_str:
                body = (
                    f"Karakteristik URL seperti **{b_names_str}** "
                    f"memberikan sinyal keamanan yang kuat."
                )
            else:
                body = (
                    "Karakteristik URL secara keseluruhan menunjukkan pola yang aman."
                )
            overall = (
                "Berbagai karakteristik lain dari URL ini juga turut dianalisis "
                "dan secara keseluruhan mengarah pada kesimpulan bahwa situs ini aman untuk diakses."
            )
            conclusion = f"{result_icon} **Kesimpulan: URL ini AMAN untuk dikunjungi.**"
        else:
            if p_names_str and b_names_str:
                body = (
                    f"Karakteristik URL seperti **{p_names_str}** "
                    f"menunjukkan indikator phishing yang kuat. "
                    f"Meskipun ada beberapa sinyal aman dari **{b_names_str}**, "
                    f"hal tersebut tidak cukup untuk mengimbangi sinyal bahaya yang ada."
                )
            elif p_names_str:
                body = (
                    f"Karakteristik URL seperti **{p_names_str}** "
                    f"menunjukkan indikator phishing yang jelas dan berbahaya."
                )
            else:
                body = "Karakteristik URL secara keseluruhan menunjukkan pola phishing yang mencurigakan."
            overall = (
                "Berbagai karakteristik lain dari URL ini juga turut dianalisis "
                "dan memperkuat penilaian bahwa situs ini berbahaya."
            )
            conclusion = (
                f"{result_icon} **Kesimpulan: URL ini TIDAK AMAN — hindari situs ini.**"
            )

        resp["llm_reasoning"] = (
            f"URL ini dinilai **{result_word}** dengan skor phishing **{p_phish:.2f}** "
            f"oleh model {model_main}.\n\n"
            f"{body} {overall}\n\n"
            f"{conclusion}"
        )

        if debug:
            resp["debug"] = {
                "prefilter_enabled": use_prefilter,
                "decision_mode": decision_mode,
                "phishing_class_value": PHISHING_CLASS_VALUE,
                "rule_prob": clamp01(risk_score / 10.0),
                "risk_score": risk_score,
                "risk_category": risk_category,
                "rule_detail": rule_detail,
                "features_analyzed": len(cols),
            }

        log_explanation = explanation_dict.get("ai_reasoning", "")

        # Log ke Excel:
        # Sertakan fitur yang punya SHAP nyata (abs_shap > 0) ATAU nilai != 0.
        # Exclude hanya fitur yang benar-benar null: nilai=0 DAN abs_shap≈0.
        all_shap_for_log = explanation_dict.get("all_shap_features", [])
        all_shap_lookup = {item["name"]: item for item in all_shap_for_log}

        def _sl(v):
            if v >= 0.1:
                return "kuat"
            if v >= 0.03:
                return "sedang"
            return "kecil"

        # Kumpulkan kandidat fitur
        candidates = {}  # name -> shap_item
        for f in all_shap_for_log:
            if float(f.get("abs_shap") or 0) > 1e-9:  # punya SHAP
                candidates[f["name"]] = f
        for feat_name in cols:
            if feat_name in candidates:
                continue
            try:
                fval = float(feats_full.get(feat_name, 0))
            except Exception:
                fval = 0.0
            if abs(fval) > 1e-9:  # nilai non-zero
                # pakai SHAP dari lookup jika ada; kalau tidak, buat entri minimal
                candidates[feat_name] = all_shap_lookup.get(
                    feat_name,
                    {
                        "name": feat_name,
                        "shap_direction": "NEUTRAL",
                        "abs_shap": 0.0,
                        "shap_value_signed": 0.0,
                    },
                )

        # Urutkan: SHAP terbesar dulu, lalu fitur non-zero tanpa SHAP
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: float(x.get("abs_shap") or 0),
            reverse=True,
        )

        log_feature_lines = []
        for i, f in enumerate(sorted_candidates):
            feat_name = f.get("name", "")
            val = feats_full.get(feat_name, 0)
            shap_dir = f.get("shap_direction", "NEUTRAL")
            signed = float(f.get("shap_value_signed") or 0)
            abs_v = float(f.get("abs_shap") or 0)
            if abs_v > 1e-9:
                level = _sl(abs_v)
                sign_str = f"+{abs_v:.4f}" if signed >= 0 else f"-{abs_v:.4f}"
                log_feature_lines.append(
                    f"{i + 1}. {feat_name} | nilai: {val} | arah: {shap_dir} "
                    f"| Memberi sinyal {level} ke {shap_dir} dengan skor {sign_str}"
                )
            else:
                # Nilai non-zero tapi SHAP tidak signifikan
                log_feature_lines.append(
                    f"{i + 1}. {feat_name} | nilai: {val} | arah: - | tidak berkontribusi signifikan"
                )
        if not log_feature_lines:
            # Fallback jika SHAP tidak tersedia sama sekali
            log_feature_lines = [
                f"{i + 1}. {f['name']} | nilai: {feats_full.get(f['name'], 0)} "
                f"| arah: {f.get('impact', '')} | {f.get('reason', '')}"
                for i, f in enumerate(
                    explanation_dict.get("top_influential_features", [])
                )
            ]
        log_top_features = "\n".join(log_feature_lines)
        feats_for_log = dict(feats_full)
        feats_for_log["_explanation"] = log_explanation
        feats_for_log["_top_features"] = log_top_features
        feats_for_log["_llm_reasoning"] = resp.get("llm_reasoning", "")
        log_feature_extraction(
            url, mode, feats_for_log, FEATURE_COLUMNS_HYBRID81, category
        )

        return jsonify(resp)

    if decision_mode == "rf_only":
        preds = predict_models(
            url, cols, include_web_content, risk_score, rule_flag, rf, xgb, meta
        )
        return build_response(
            clamp01(float(preds.get("rf_prob") or 0.0)),
            "rf_only",
            "Random Forest",
            preds.get("rf_prob"),
            None,
            None,
            rf_model=rf,
        )

    if decision_mode == "xgb_only":
        preds = predict_models(
            url, cols, include_web_content, risk_score, rule_flag, rf, xgb, meta
        )
        return build_response(
            clamp01(float(preds.get("xgb_prob") or 0.0)),
            "xgb_only",
            "XGBoost",
            None,
            preds.get("xgb_prob"),
            None,
            xgb_model=xgb,
        )

    if decision_mode == "ml_stacking_only":
        preds = predict_models(
            url, cols, include_web_content, risk_score, rule_flag, rf, xgb, meta
        )
        p, src, name = (
            (
                clamp01(float(preds["stack_prob"])),
                "ml_stacking_only",
                "RF + XGB + Logistic Regression Stacking",
            )
            if preds.get("stack_prob") is not None
            else (
                clamp01(
                    0.5 * float(preds.get("rf_prob") or 0.0)
                    + 0.5 * float(preds.get("xgb_prob") or 0.0)
                ),
                "ml_rf_xgb_average_only",
                "RF + XGB Average (Meta model unavailable)",
            )
        )
        return build_response(
            p,
            src,
            name,
            preds.get("rf_prob"),
            preds.get("xgb_prob"),
            preds.get("stack_prob"),
            rf_model=rf,
            xgb_model=xgb,
        )

    if use_prefilter and rule_flag == 1:
        return build_response(
            clamp01(max(PREFILTER_PHISHING_MIN_CONF, risk_score / 10.0)),
            "rule_based_prefilter_phishing",
            "Rule-Based Prefilter",
            web_used=False,
        )

    preds = predict_models(
        url, cols, include_web_content, risk_score, rule_flag, rf, xgb, meta
    )
    p, src, name = (
        (
            clamp01(float(preds["stack_prob"])),
            "ml_stacking_only",
            "RF + XGB + Logistic Regression Stacking",
        )
        if preds.get("stack_prob") is not None
        else (
            clamp01(
                0.5 * float(preds.get("rf_prob") or 0.0)
                + 0.5 * float(preds.get("xgb_prob") or 0.0)
            ),
            "ml_rf_xgb_average_only",
            "RF + XGB Average (Meta model unavailable)",
        )
    )
    return build_response(
        p,
        src,
        name,
        preds.get("rf_prob"),
        preds.get("xgb_prob"),
        preds.get("stack_prob"),
        rf_model=rf,
        xgb_model=xgb,
    )


@app.post("/predict_url")
@app.post("/analyze")
def predict_alias():
    return predict()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
