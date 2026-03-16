# Phishing Website Detection (Hybrid: Rule-Based + RF + XGBoost + LR Stacking)

Proyek ini mendeteksi URL phishing menggunakan pendekatan **hybrid**:

1. **Rule-based filtering** (deteksi cepat berbasis pola URL),
2. **Machine learning base models**: Random Forest + XGBoost,
3. **Meta-learner**: Logistic Regression (stacking layer).

---

## Arsitektur Sistem

Alur prediksi:

1. URL masuk
2. Ekstraksi fitur URL
3. Rule-based menghitung:
   - `vi_count`, `imp_count`, `less_count`
   - `risk_score = 3*vi_count + 2*imp_count + 1*less_count`
   - `rule_flag`
4. Jika `rule_flag == 1` → gunakan skor rule sebagai confidence
5. Jika `rule_flag == 0` → gunakan ML (RF + XGB + LR meta)

> Catatan implementasi saat ini: `rule_lr.pkl` berisi **LogisticRegression meta-learner**, sehingga inference ML menggunakan probabilitas RF & XGB sebagai input ke LR.

---

## Fitur yang Digunakan

Model dilatih dengan **37 fitur URL-based** (lihat `feature_columns.txt`), termasuk:

- panjang URL/hostname
- jumlah karakter khusus (`@`, `?`, `%`, `-`, `_`, dll.)
- token protokol (`http_in_path`, `https_token`)
- struktur domain (`nb_subdomains`, `tld_in_path`, dll.)
- indikator anomali (`ip`, `random_domain`, `shortening_service`, dst.)

---

## Konfigurasi Utama (Sesuai Implementasi)

- **Very-important threshold**: `vi_count >= 1`
- **Port feature**: `port = 1` jika **non-standar**
- **Suspicious TLD list**: 24 TLD
- **Risk formula**: `3*vi + 2*imp + 1*less`

---

## Model Training

Model yang dilatih dan disimpan:

- `random_forest_model.pkl`
- `xgboost_model.pkl`
- `rule_lr.pkl` (meta Logistic Regression)
- `feature_columns.txt`

Hyperparameter:
- **Random Forest**: `n_estimators=200`, `max_depth=5`, `random_state=12`
- **XGBoost**: `n_estimators=200`, `max_depth=4`, regularisasi L1/L2, `random_state=12`
- **Stacking Meta**: `LogisticRegression(max_iter=1000, class_weight='balanced')`

---

## Struktur File Penting

```text
Phishing-Website-Detection-using-XGBoost-and-RFM/
├─ app.py
├─ train_model.py
├─ requirements.txt
├─ feature_columns.txt
├─ random_forest_model.pkl
├─ xgboost_model.pkl
├─ rule_lr.pkl
├─ static/
│  ├─ advanced_hybrid_detector.css
│  └─ advanced_hybrid_detector.js
└─ Phishing_detection_app/
   ├─ main.dart
   ├─ phishing.dart
   └─ phishing_detect.dart
```

---

## Instalasi

Gunakan virtual environment (disarankan), lalu install dependency:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
- flask
- numpy
- pandas
- scikit-learn
- xgboost
- tldextract
- requests

---

## Menjalankan Training

```bash
python train_model.py
```

Output training akan menghasilkan file `.pkl` dan `feature_columns.txt`.

---

## Menjalankan API Flask

```bash
python app.py
```

Default API:
- `GET /` : health/index
- `POST /predict` : prediksi phishing dari URL

Contoh request (PowerShell):

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"url":"http://secure-paypal-login-verify.xyz/index.php?user=abc"}'
```

---

## Catatan Sinkronisasi Dokumen

Dokumen dan implementasi harus konsisten pada poin berikut:

- Rule-based **tidak dilatih** (rule engine/manual threshold)
- Yang dilatih: RF, XGB, dan LR (meta-learner stacking)
- Label klasifikasi: `1 = phishing`, `0 = benign`
- Urutan fitur inference harus sama dengan `feature_columns.txt`

---

## Hasil Singkat (contoh run terbaru)

- Random Forest: Accuracy ~80.84%
- XGBoost: Accuracy ~87.75%
- Stacking: Accuracy ~87.84%

Stacking memberi peningkatan kecil dibanding base model tunggal.

---

## Lisensi

Gunakan untuk keperluan akademik dan pengembangan riset keamanan siber.