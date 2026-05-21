# Phishing Website Detection (Hybrid: Rule-Based + ML Stacking)

Sistem deteksi website phishing hybrid yang menggabungkan rule-based filtering, Random Forest, XGBoost, dan Logistic Regression stacking untuk prediksi real-time.

## Daftar Isi
- [Gambaran Umum](#gambaran-umum)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Instalasi](#instalasi)
- [Menjalankan](#menjalankan)
- [API Endpoint](#api-endpoint)
- [Evaluasi Batch](#evaluasi-batch)
- [Hasil Model](#hasil-model)

## Gambaran Umum

Sistem hybrid tiga tingkat:
1. **Rule-based filtering** — deteksi cepat pola URL anomali
2. **Base Models** — Random Forest & XGBoost
3. **Stacking Ensemble** — Logistic Regression meta-learner

URL → Ekstrak Fitur → Rule Engine → Base Models → Stacking → Prediksi (Phishing/Benign)

## Arsitektur Sistem

| Komponen | Deskripsi |
|----------|-----------|
| `app.py` | Flask backend + ekstraksi fitur real-time |
| `train_model.py` | Training pipeline dengan GA tuning |
| `proses.ipynb` | Evaluasi batch pada dataset besar |
| Fitur | 37 fitur: URL-based |
| Label | Training: 1=phishing/0=benign; Test: 0=phishing/1=benign |

**Artifact Model:**
- `random_forest_model.pkl`, `xgboost_model.pkl`, `rule_lr.pkl`
- `feature_columns.txt`, `feature_medians.pkl`
- `ga_tuning_report.json`

**Fitur mencakup:**
- Panjang URL/hostname, karakter khusus (`@`, `?`, `%`, `-`, `_`)
- Token protokol (`http_in_path`, `https_token`)
- Struktur domain (`nb_subdomains`, `tld_in_path`, dll.)
- Indikator anomali (`ip`, `random_domain`, `shortening_service`)

## Instalasi

```bash
# 1. Virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

**Dependencies utama:**
- flask, numpy, pandas, scikit-learn, xgboost, tldextract, requests

## Menjalankan

### Training
```bash
python train_model.py
```

### Flask API
```bash
python app.py
# Akses: http://127.0.0.1:5000
```

### Evaluasi Batch
Buka dan jalankan `proses.ipynb` untuk mengevaluasi pada dataset test besar.

## API Endpoint

**GET** `/` — Halaman utama  
**GET** `/health` — Status pipeline  
**POST** `/predict` — Prediksi URL  

### Contoh Request
```bash
# PowerShell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
  -Method POST -ContentType "application/json" `
  -Body '{"url":"http://secure-paypal-login.xyz/index.php"}'

# cURL
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url":"http://secure-paypal-login.xyz/index.php"}'
```

### Response
```json
{
  "label": "phishing",
  "probability": 0.95,
  "confidence": "high"
}
```

## Evaluasi Batch

Notebook `proses.ipynb` menyediakan:
- Evaluasi pada dataset_test.csv
- Remapping label test → label model
- Progress report & ETA
- Distribusi probabilitas per bucket
- Perbandingan timing pipeline

## Hasil Model

| Model | Accuracy | AUC |
|-------|----------|-----|
| Random Forest | 97.20% | 99.54% |
| XGBoost | 97.73% | 99.59% |
| Stacking (LR) | 97.42% | 99.62% |

## Catatan Penting

⚠️ **Label Dataset**
- Training `data_cleaning.csv`: 1=phishing, 0=benign
- Test `dataset_test.csv`: 0=phishing, 1=benign
- Notebook proses.ipynb otomatis melakukan remapping

⚠️ **Rule-based vs ML**
- Rule-based: tidak dilatih, hanya threshold manual
- ML: Random Forest, XGBoost, dan meta-learner LR dilatih dengan GA

⚠️ **Implementasi**
- Risk formula: `3*vi_count + 2*imp_count + 1*less_count`
- Very-important threshold: `vi_count >= 1`
- Suspicious TLD list: 24 TLD
- Hyperparameter RF: n_estimators=200, max_depth=5
- Hyperparameter XGB: n_estimators=200, max_depth=4

## Struktur File
```
├─ app.py (Flask backend)
├─ train_model.py (Training pipeline)
├─ proses.ipynb (Evaluasi batch)
├─ requirements.txt
├─ feature_columns.txt
├─ feature_medians.pkl
├─ {random_forest, xgboost, rule_lr}_model.pkl
├─ ga_tuning_report.json
├─ DataFiles/
│  ├─ data_cleaning.csv
│  ├─ dataset_test.csv
│  └─ [lainnya]
├─ static/ (Frontend assets)
└─ Phishing_detection_app/ (Dart UI)
```

## Lisensi
Untuk keperluan akademik dan penelitian keamanan siber.

<div class="mini-card"><span>Decision Mode</span><b>${escapeHtml(result.decisionMode)}</b></div>
                <div class="mini-card"><span>Decision Source</span><b>${escapeHtml(result.decisionSource)}</b></div>
                <div class="mini-card"><span>Rule</span><b>score &lt; ${threshold.toFixed(2)} = BENIGN, score ≥ ${threshold.toFixed(2)} = PHISHING</b></div>