# Phishing Website Detection (Hybrid: Rule-Based Prefilter + ML Stacking)

Sistem deteksi website phishing hybrid tiga lapis yang menggabungkan rule-based prefilter, stacking ensemble (Random Forest + XGBoost + Logistic Regression), SHAP Explainability, dan LLM Reasoning.

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Arsitektur &amp; Alur Pipeline](#arsitektur--alur-pipeline)
- [Mode Fitur](#mode-fitur)
- [Rule-Based Prefilter](#rule-based-prefilter)
- [Model Machine Learning](#model-machine-learning)
- [SHAP &amp; LLM Reasoning](#shap--llm-reasoning)
- [Instalasi](#instalasi)
- [Menjalankan](#menjalankan)
- [API Endpoint](#api-endpoint)
- [Decision Mode](#decision-mode)
- [Logging](#logging)
- [Hasil Model](#hasil-model)
- [Struktur File](#struktur-file)

---

## Gambaran Umum

Sistem deteksi phishing berbasis hybrid tiga lapis:

1. **Rule-Based Prefilter** — 24 rule deterministik pada struktur URL; URL phishing jelas langsung divonis tanpa perlu ML
2. **ML Stacking Ensemble** — Random Forest + XGBoost sebagai base learner, Logistic Regression sebagai meta-learner
3. **Explainability** — SHAP (nilai kontribusi fitur) + LLM Reasoning (narasi Bahasa Indonesia otomatis)

```
URL → Validasi → Ekstraksi Fitur (37/81)
          │
          ├── Rule-Based Prefilter (24 rule)
          │       └── PHISHING langsung jika rule terpicu (bypass ML)
          │
          └── ML Stacking (jika lolos prefilter)
                  ├── Random Forest  → rf_prob
                  ├── XGBoost        → xgb_prob
                  └── LR Meta-learner → stack_prob
                          │
                          └── Threshold 0.6 → PHISHING / BENIGN
                                  │
                                  ├── SHAP (top influential features)
                                  ├── LLM Reasoning (narasi otomatis)
                                  └── JSON Response + Log CSV
```

---

## Arsitektur & Alur Pipeline

| Tahap | Komponen               | Keterangan                                                                                                        |
| ----- | ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1     | Validasi URL           | `is_valid_url()` — cek format domain/IP                                                                        |
| 2a    | URL Feature Extraction | `extract_url_features()` — 37 fitur, no network, cached LRU                                                    |
| 2b    | Web Content Extraction | `extract_web_content_          features()` + `enrich_external_features()` — 44 fitur tambahan via HTTP fetch |
| 3     | Pemilihan Mode Model   | `get_model_and_features()` — mode `_37` jika web fetch gagal, mode `_81` jika berhasil                     |
| 4     | Rule-Based Prefilter   | `rule_based_eval()` — 24 rule, 3 tingkat kepentingan                                                           |
| 5     | ML Stacking            | `predict_models()` — RF → XGB → LR meta-learner                                                              |
| 6     | Keputusan Akhir        | `build_response()` — threshold `FINAL_THRESHOLD = 0.6`                                                       |
| 7     | SHAP                   | `get_shap_top()` — TreeExplainer / KernelExplainer fallback                                                    |
| 8     | LLM Reasoning          | `build_llm_prompt()` + `get_llm_reasoning()` — async, timeout 15s                                            |
| 9     | Logging                | `log_feature_extraction()` — CSV thread-safe, 81 fitur + SHAP + LLM                                            |

**Artifact Model:**

- `models/random_forest_model_37.pkl`, `models/xgboost_model_37.pkl`, `models/rule_lr_37.pkl`
- `models/random_forest_model_81.pkl`, `models/xgboost_model_81.pkl`, `models/rule_lr_81.pkl`
- `models/feature_columns_37.txt`, `models/feature_columns_81.txt`
- `models/ga_tuning_report_37.json`, `models/ga_tuning_report_81.json`

---

## Mode Fitur

Sistem secara otomatis memilih mode berdasarkan keberhasilan web fetch:

| Mode     | Fitur    | File Model   | Kondisi                   |
| -------- | -------- | ------------ | ------------------------- |
| URL-Only | 37 fitur | `*_37.pkl` | Web fetch gagal / timeout |
| Hybrid   | 81 fitur | `*_81.pkl` | Web fetch berhasil        |

**37 Fitur URL-based:** panjang URL/hostname, karakter khusus (`@`, `?`, `%`, `-`, `_`, `~`, `;`, `*`, `,`, `$`), struktur domain (`nb_subdomains`, `tld_in_path`), indikator anomali (`ip`, `random_domain`, `shortening_service`, `http_in_path`, `https_token`, `punycode`), statistik kata, brand hints.

**44 Fitur tambahan (Hybrid):** hyperlink ratio, CSS eksternal, iframe, login form, favicon eksternal, media ratio, redirect chain, WHOIS, domain age, DNS record, Google index, PageRank, web traffic.

---

## Rule-Based Prefilter

24 rule deterministik dibagi 3 tingkat kepentingan:

### Tingkat Sangat Penting (4 rule) — `vi_hits`

**Satu rule saja sudah cukup → URL langsung PHISHING (bypass ML)**

| Rule               | Kondisi                                                                              |
| ------------------ | ------------------------------------------------------------------------------------ |
| `suspicious_tld` | TLD termasuk daftar 24 TLD berbahaya (`.xyz`, `.top`, `.tk`, `.click`, dll.) |
| `random_domain`  | Entropi nama domain > 3.5 (domain acak/DGA-like)                                     |
| `ip`             | URL menggunakan IP address langsung sebagai host                                     |
| `http_in_path`   | Kata `http` muncul di dalam path URL (URL-in-URL)                                  |

### Tingkat Penting (17 rule) — `imp_hits`, bobot ×2

`nb_at`, `nb_subdomains>3`, `nb_dots>4`, `nb_slash>7`, `length_hostname>30`, `nb_percent>5`, `nb_tilde≥1`, `nb_semicolumn≥1`, `nb_star≥1`, `nb_comma≥1`, `nb_dollar≥1`, `nb_qm>2`, `nb_colon>1`, `nb_eq>8`, `nb_and>3`, `nb_hyphens>3`, `nb_underscore>3`

### Tingkat Cukup Penting (3 rule) — `less_hits`, bobot ×1

`ratio_digits_url>0.3`, `port` (non-standard), `shortening_service`

**Formula skor risiko:** `risk_score = 2×len(imp_hits) + 1×len(less_hits)`

**Keputusan Rule:**

- `vi_hits ≥ 1` ATAU `risk_score ≥ 5` → **PHISHING** (confidence minimal 0.95)
- `risk_score > 0` → **Suspicious** (lanjut ke ML)
- `risk_score = 0` → **Benign** (lanjut ke ML)

---

## Model Machine Learning

### Training

Model dilatih dengan dataset `data_cleaning.csv` (Kaggle), split 80:20, random state=12.

- **Label training:** `1 = phishing`, `0 = benign`
- **Hyperparameter tuning:** Genetic Algorithm (GA) — populasi 16, generasi 8, 5-Fold Stratified CV, metrik F1

### Stacking Ensemble

```
RF Base Learner  → rf_prob  ─┐
XGB Base Learner → xgb_prob  ├─→ [rf_prob, xgb_prob, (rule_flag?), (risk_score?)] → LR meta-learner → stack_prob
                              ┘
```

Input meta-learner LR bergantung pada `n_features_in_` model `.pkl` (2–4 fitur).

### Threshold Keputusan

`FINAL_THRESHOLD = 0.6` — jika `p_phish ≥ 0.6` → label **phishing**

---

## SHAP & LLM Reasoning

### SHAP (Shapley Additive exPlanations)

- **TreeExplainer** untuk RF dan XGBoost (cepat)
- **KernelExplainer** untuk Stacking/fallback (model-agnostic)
- Top fitur dipilih berdasarkan `|SHAP value|` terbesar
- Arah: `SHAP > 0` → mendorong phishing, `SHAP < 0` → mendorong benign

### LLM Reasoning

- Prompt terstruktur Bahasa Indonesia dikirim ke LLM via `llm_utils.get_llm_reasoning()`
- Prompt memuat: URL, kesimpulan akhir, probabilitas, sumber keputusan, top fitur + nilai + status
- Jika sumber keputusan = `rule_based_prefilter_phishing`: LLM wajib menyebut rule yang terpicu, **dilarang** menyebut SHAP/RF/XGB
- Output 3–4 kalimat diakhiri `REKOMENDASI: BLOKIR.` atau `REKOMENDASI: IZINKAN.`
- Async via `ThreadPoolExecutor`, timeout 15 detik

---

## Instalasi

Gunakan Python 3.11 dan jalankan perintah dari direktori utama proyek, yaitu direktori yang berisi `app.py` dan `requirements.txt`. Dependensi proyek sudah dicantumkan dan versinya dikunci di `requirements.txt`.

```bash
# 1. Virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Konfigurasi Lokal Opsional

Untuk mengaktifkan penjelasan LLM atau pengayaan reputasi domain, buat `.env` dari file contoh. Jika `.env` sudah ada, jangan menimpanya karena mungkin berisi konfigurasi lokal.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

`OPENAI_API_KEY` dipakai untuk penjelasan LLM. `OPR_API_KEY`, `SERP_API_KEY`, dan `SEMRUSH_KEY` dipakai oleh pengayaan fitur eksternal. Semua API key bersifat opsional; jangan membagikan file `.env` yang berisi key. Tanpa key LLM, deteksi tetap berjalan tanpa penjelasan LLM.

Di Windows, jika PowerShell memblokir aktivasi virtual environment, izinkan aktivasi hanya untuk sesi terminal saat ini, lalu aktifkan kembali:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Dependencies utama:**
`flask`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `tldextract`, `requests`, `beautifulsoup4`, `python-dotenv`

---

## Menjalankan

### Flask API

```bash
python app.py
```

Buka antarmuka lokal di `http://127.0.0.1:5000`. Periksa status aplikasi melalui `http://127.0.0.1:5000/health`.

Artifact model harus tersedia di folder `models/` sebelum aplikasi digunakan. Jika artifact belum ada, pastikan `DataFiles/data_cleaning.csv` tersedia lalu jalankan `python train_model.py`; proses Genetic Algorithm dapat memerlukan waktu cukup lama.

### Notebook Training & Evaluasi

```bash
python -m pip install notebook
jupyter notebook "Phishing Website Detection_Models & Training.ipynb"
```

Jalankan cell secara berurutan. Notebook ini membaca artifact dan laporan GA dari folder `models/`; sesuaikan path pemuatan notebook ke folder tersebut sebelum menjalankan jika masih menunjuk ke direktori utama.

---

## API Endpoint

| Method | Endpoint         | Keterangan                                        |
| ------ | ---------------- | ------------------------------------------------- |
| GET    | `/`            | Halaman utama (`advanced_hybrid_detector.html`) |
| GET    | `/health`      | Status pipeline — cek model loaded               |
| POST   | `/predict`     | Prediksi URL (endpoint utama)                     |
| POST   | `/predict_url` | Alias `/predict`                                |
| POST   | `/analyze`     | Alias `/predict`                                |

### Contoh Request

```bash
# cURL
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url":"http://secure-paypal-login.xyz/index.php"}'

# PowerShell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
  -Method POST -ContentType "application/json" `
  -Body '{"url":"http://secure-paypal-login.xyz/index.php"}'
```

### Parameter Request (JSON)

| Parameter         | Tipe   | Default              | Keterangan                            |
| ----------------- | ------ | -------------------- | ------------------------------------- |
| `url`           | string | —                   | URL yang akan diperiksa (wajib)       |
| `decision_mode` | string | `hybrid_prefilter` | Mode keputusan (lihat tabel di bawah) |
| `use_prefilter` | bool   | `true`             | Aktifkan rule-based prefilter         |
| `debug`         | bool   | `false`            | Tambahkan info debug ke response      |

### Response JSON

```json
{
  "label": "phishing",
  "category": "phishing",
  "final_label": 1,
  "final_phishing_prob": 0.9800,
  "final_safe_prob": 0.0200,
  "final_threshold": 0.6,
  "confidence": 0.9800,
  "decision_mode": "hybrid_prefilter",
  "decision_source": "rule_based_prefilter_phishing",
  "model_name": "Rule-Based Prefilter",
  "model_feature_count": 37,
  "web_content_used": false,
  "rf_prob": null,
  "xgb_prob": null,
  "stack_prob": null,
  "llm_reasoning": "URL ini terdeteksi PHISHING karena ...",
  "explanation_detailed": {
    "final_prediction": "PHISHING",
    "confidence_score": 0.9800,
    "phishing_probability": 0.9800,
    "benign_probability": 0.0200,
    "main_contributing_model": "Rule-Based Prefilter",
    "top_influential_features": [
      {
        "name": "suspicious_tld",
        "value": 1,
        "impact": "PHISHING",
        "reason": "TLD domain termasuk daftar TLD mencurigakan...",
        "shap_value": 3.0,
        "rule_level": "Sangat Penting"
      }
    ],
    "model_contribution_probability": {}
  }
}
```

---

## Decision Mode

| Mode                 | Parameter                    | Keterangan                                        |
| -------------------- | ---------------------------- | ------------------------------------------------- |
| `hybrid_prefilter` | default                      | Rule-Based Prefilter → jika lolos → ML Stacking |
| `rf_only`          | `"decision_mode": "rf"`    | Hanya Random Forest                               |
| `xgb_only`         | `"decision_mode": "xgb"`   | Hanya XGBoost                                     |
| `ml_stacking_only` | `"decision_mode": "stack"` | ML Stacking tanpa prefilter                       |

**`decision_source`** di response menjelaskan dari mana keputusan akhir berasal:

| Nilai                             | Artinya                                           |
| --------------------------------- | ------------------------------------------------- |
| `rule_based_prefilter_phishing` | Divonis phishing oleh rule (ML tidak dipakai)     |
| `ml_stacking_only`              | Keputusan dari stack_prob LR                      |
| `ml_rf_xgb_average_only`        | Rata-rata rf_prob + xgb_prob (meta-learner gagal) |
| `rf_only`                       | Hanya RF yang dipakai                             |
| `xgb_only`                      | Hanya XGBoost yang dipakai                        |

---

## Logging

Setiap prediksi dicatat ke `log_feature_extraction.csv`:

- **Format:** CSV separator `;`, encoding `utf-8-sig` (kompatibel Excel)
- **Kolom:** `NO`, `URL`, `MODE`, `DECISION_SOURCE`, `STATUS`, `PHISHING_PROB`, [81 fitur], `TOP_FEATURE`, `LLM_REASONING`
- **Thread-safe:** menggunakan `threading.Lock()`
- **Lock detection:** jika file sedang dibuka di Excel, request ditolak dengan HTTP 503
- **Dipakai notebook:** Cell 15 membaca log ini untuk SHAP lokal waterfall (4 URL terakhir unik)

---

## Hasil Model

Evaluasi membandingkan Random Forest, XGBoost, dan stacking (Random Forest + XGBoost dengan Logistic Regression sebagai meta-learner) pada dua mode fitur: URL-only (37 fitur) dan hybrid (81 fitur). Pengujian menggunakan 20% data sebagai data uji.

Metrik yang dilaporkan meliputi accuracy, precision, recall, F1-score, dan ROC-AUC. Threshold 0.6 digunakan untuk menentukan label prediksi dan menghitung metrik yang bergantung pada threshold; ROC-AUC dihitung dari probabilitas prediksi sehingga tidak bergantung pada threshold.

Jalankan notebook evaluasi setelah artifact model tersedia di folder `models/` untuk menghasilkan nilai metrik aktual.

---

## Catatan Penting

⚠️ **Label Dataset**

- Training `data_cleaning.csv`: `1 = phishing`, `0 = benign`
- Konsisten di `app.py` (`PHISHING_CLASS_VALUE = 1`) dan notebook (Cell 2, `TARGET_COL = 'label'`)

⚠️ **Dua Set Model PKL**

- Sistem menggunakan **dua set model**: `*_37.pkl` dan `*_81.pkl`
- Pemilihan otomatis berdasarkan keberhasilan web fetch

⚠️ **Rule-Based vs ML**

- Rule-based: tidak dilatih, deterministik berbasis threshold manual
- ML: RF, XGBoost, dan LR meta-learner dilatih dengan GA tuning

⚠️ **SHAP di app.py vs Notebook**

- `app.py`: SHAP dihitung real-time per URL menggunakan TreeExplainer/KernelExplainer
- Notebook Cell 12–14: SHAP global (batch, seluruh test set) → plot beeswarm/bar
- Notebook Cell 15: SHAP lokal waterfall dari log CSV (4 URL terakhir unik)
- Notebook Cell 15 menggunakan `KernelExplainer` eksplisit untuk Stacking agar tidak salah-route ke LinearExplainer

---

## Struktur File

```
├── app.py                                      ← Flask backend + full pipeline
├── train_model.py                              ← Training pipeline (GA tuning)
├── requirements.txt                            ← Dependensi Python
├── .env.example                                ← Contoh konfigurasi opsional
├── external_features.py                        ← Pengayaan WHOIS, DNS, PageRank, dll.
├── llm_utils.py                                ← Wrapper panggilan LLM API
├── fungsi/                                     ← Logika fitur, model, rule, SHAP, dan logging
├── models/                                     ← Model PKL, daftar fitur, dan laporan GA
├── DataFiles/
│   └── data_cleaning.csv                       ← Dataset training/evaluasi
├── Phishing Website Detection_Models & Training.ipynb ← Evaluasi & SHAP
├── preprocessing.ipynb                         ← Eksplorasi dan preprocessing data
├── Phishing_detection_app/                     ← UI web
├── static/                                     ← Aset frontend
├── log_feature_extraction.csv                  ← Log prediksi aplikasi
└── results/                                    ← Grafik dan keluaran analisis notebook
```

---

## Lisensi

Untuk keperluan akademik — Tugas Akhir Institut Teknologi Del 2026.
