# ══════════════════════════════════════════════════════════════════════
# ██  BAGIAN 11 & 12 — PETA RISIKO FITUR & LLM EXPLANATION
# ══════════════════════════════════════════════════════════════════════

import numpy as np

from fungsi.config import FINAL_THRESHOLD, PREFILTER_PHISHING_MIN_CONF, RULE_REASON_MAP
from fungsi.model import executor
from fungsi.shap_explain import get_shap_top
from fungsi.utils import clamp01

try:
    from llm_utils import get_llm_reasoning
except ImportError:
    def get_llm_reasoning(prompt):
        return "LLM tidak tersedia."


# ── BAGIAN 11 — PETA RISIKO FITUR ────────────────────────────────────

def get_phishing_risk_direction(feature_name, feature_value):
    """Untuk tiap fitur: kondisi → PHISHING atau → BENIGN + alasannya."""
    feature_rules = {
        "ip":                    (lambda v: v == 1, "Menggunakan IP address langsung sebagai domain (bukan nama domain)",
                                  lambda v: v == 0, "Menggunakan nama domain, bukan IP address"),
        "nb_at":                 (lambda v: v >= 1, "Ada simbol @ dalam URL — umum dipakai untuk menyamarkan domain asli",
                                  lambda v: v == 0, "Tidak ada simbol @ yang mencurigakan"),
        "nb_underscore":         (lambda v: v > 3,  "Terlalu banyak underscore, tidak wajar untuk domain/path normal",
                                  lambda v: v == 0, "Tidak ada underscore mencurigakan dalam URL"),
        "nb_percent":            (lambda v: v > 5,  "Banyak karakter ter-encode (%xx) — tanda URL obfuskasi",
                                  lambda v: v == 0, "Tidak ada encoding mencurigakan dalam URL"),
        "nb_tilde":              (lambda v: v >= 1, "Karakter ~ dalam URL — jarang dipakai situs benign",
                                  lambda v: v == 0, "Tidak ada karakter tilde mencurigakan"),
        "nb_semicolumn":         (lambda v: v >= 1, "Karakter ; dalam URL — bisa dipakai untuk menyisipkan parameter palsu",
                                  lambda v: v == 0, "Tidak ada semicolons mencurigakan"),
        "nb_star":               (lambda v: v >= 1, "Karakter * dalam URL — tidak lazim untuk situs normal",
                                  lambda v: v == 0, "Tidak ada karakter bintang mencurigakan"),
        "nb_comma":              (lambda v: v >= 1, "Karakter koma dalam URL — tidak standar",
                                  lambda v: v == 0, "Tidak ada koma mencurigakan dalam URL"),
        "nb_dollar":             (lambda v: v >= 1, "Karakter $ dalam URL — jarang di situs resmi",
                                  lambda v: v == 0, "Tidak ada karakter dollar mencurigakan"),
        "nb_hyphens":            (lambda v: v > 3,  "Terlalu banyak tanda hubung — domain phishing sering pakai banyak hyphen",
                                  lambda v: v <= 1, "Jumlah tanda hubung dalam batas wajar"),
        "nb_dots":               (lambda v: v > 4,  "Terlalu banyak titik — kemungkinan subdomain berlapis untuk menyamarkan domain asli",
                                  lambda v: v <= 2, "Jumlah titik dalam URL normal"),
        "nb_slash":              (lambda v: v > 7,  "Path URL sangat panjang dan kompleks",
                                  lambda v: v <= 4, "Struktur path URL sederhana dan wajar"),
        "nb_qm":                 (lambda v: v > 2,  "Banyak tanda tanya — query parameter berlebihan",
                                  lambda v: v == 0, "Tidak ada query parameter mencurigakan"),
        "nb_and":                (lambda v: v > 3,  "Banyak parameter & dalam query string",
                                  lambda v: v == 0, "Tidak ada parameter & berlebihan"),
        "nb_colon":              (lambda v: v > 1,  "Banyak titik dua — bisa menyembunyikan port atau parameter palsu",
                                  lambda v: v == 1, "Jumlah titik dua normal (hanya dari protokol)"),
        "length_url":            (lambda v: v > 75, "URL sangat panjang — umum dipakai untuk menyembunyikan tujuan asli",
                                  lambda v: v < 50, "Panjang URL singkat dan wajar"),
        "length_hostname":       (lambda v: v > 30, "Hostname sangat panjang — domain phishing sering panjang untuk meniru brand",
                                  lambda v: v < 15, "Hostname singkat dan simpel, ciri domain resmi"),
        "ratio_digits_url":      (lambda v: v > 0.3,  "Proporsi angka terlalu tinggi dalam URL",
                                  lambda v: v < 0.05, "Hampir tidak ada angka acak dalam URL"),
        "nb_subdomains":         (lambda v: v > 3,  "Subdomain berlapis-lapis — taktik phishing untuk meniru struktur domain resmi",
                                  lambda v: v <= 1, "Jumlah subdomain normal"),
        "random_domain":         (lambda v: v == 1, "Nama domain terlihat acak/gibberish — bukan nama yang mudah diingat",
                                  lambda v: v == 0, "Nama domain terlihat natural, bukan acak"),
        "prefix_suffix":         (lambda v: v == 1, "Ada tanda hubung (-) dalam nama domain — pola umum domain phishing seperti 'paypal-secure.com'",
                                  lambda v: v == 0, "Tidak ada tanda hubung di nama domain"),
        "shortening_service":    (lambda v: v == 1, "Menggunakan layanan pemendek URL (bit.ly, dll) — menyembunyikan tujuan asli",
                                  lambda v: v == 0, "Tidak menggunakan URL shortener"),
        "suspicious_tld":        (lambda v: v == 1, "TLD (ekstensi domain) termasuk kategori mencurigakan seperti .xyz, .tk, .top",
                                  lambda v: v == 0, "Ekstensi domain normal dan terpercaya"),
        "http_in_path":          (lambda v: v == 1, "Teks 'http' muncul di dalam path URL — indikasi URL-in-URL untuk pengalihan berbahaya",
                                  lambda v: v == 0, "Tidak ada protokol tersembunyi di path URL"),
        "https_token":           (lambda v: v == 1, "Kata 'https' muncul di path URL (bukan di protokol) — trik psikologis agar terlihat aman",
                                  lambda v: v == 0, "Tidak ada kata 'https' yang disalahgunakan di path"),
        "port":                  (lambda v: v == 1, "Menggunakan port tidak standar — situs resmi jarang mengekspos port selain 80/443",
                                  lambda v: v == 0, "Menggunakan port standar (80/443)"),
        "nb_redirection":        (lambda v: v > 0,  "URL mengandung redirect berantai — taktik untuk menyembunyikan tujuan akhir",
                                  lambda v: v == 0, "Tidak ada redirect tersembunyi dalam URL"),
        "punycode":              (lambda v: v == 1, "Domain menggunakan Punycode (xn--) — taktik homograph attack untuk meniru domain asli",
                                  lambda v: v == 0, "Tidak ada Punycode — domain menggunakan karakter standar"),
        "tld_in_path":           (lambda v: v == 1, "Ekstensi domain (.com, .net) muncul di path URL — tanda struktur URL tidak wajar",
                                  lambda v: v == 0, "Struktur URL normal, TLD hanya di domain"),
        "tld_in_subdomain":      (lambda v: v == 1, "Ekstensi domain muncul di subdomain — trik phishing agar terlihat seperti domain lain",
                                  lambda v: v == 0, "Subdomain normal, tidak mengandung TLD palsu"),
        "abnormal_subdomain":    (lambda v: v == 1, "Subdomain mengandung 'http/https' — struktur sangat mencurigakan",
                                  lambda v: v == 0, "Subdomain normal"),
        "path_extension":        (lambda v: v == 1, "File di path punya ekstensi tertentu — bisa mengunduh file berbahaya",
                                  lambda v: v == 0, "Tidak ada ekstensi file mencurigakan di path"),
        "phish_hints":           (lambda v: v > 0,  "URL mengandung kata kunci phishing seperti 'login', 'verify', 'secure', 'account'",
                                  lambda v: v == 0, "Tidak ada kata kunci phishing dalam URL"),
        "domain_in_brand":       (lambda v: v == 0, "Domain tidak dikenali sebagai brand terkenal",
                                  lambda v: v == 1, "Domain dikenali sebagai brand besar yang terpercaya"),
        "brand_in_subdomain":    (lambda v: v == 1, "Nama brand besar (Google, PayPal, dll) muncul di subdomain — taktik phishing umum",
                                  lambda v: v == 0, "Tidak ada penyalahgunaan nama brand di subdomain"),
        "brand_in_path":         (lambda v: v == 1, "Nama brand besar muncul di path URL — bisa jadi upaya meniru halaman brand tersebut",
                                  lambda v: v == 0, "Tidak ada nama brand di path"),
        "nb_www":                (lambda v: v == 0, "Tidak menggunakan sub-domain www standar",
                                  lambda v: v == 1, "Menggunakan sub-domain www standar"),
        "login_form":            (lambda v: v == 1, "Halaman memiliki form login — terutama mencurigakan bila domain tidak dikenal",
                                  lambda v: v == 0, "Tidak ada form login mencurigakan di halaman"),
        "external_favicon":      (lambda v: v == 1, "Favicon (ikon tab) dimuat dari domain lain — tanda situs meniru tampilan brand lain",
                                  lambda v: v == 0, "Favicon dihosting di domain yang sama"),
        "iframe":                (lambda v: v == 1, "Halaman menggunakan iframe tersembunyi — umum di serangan phishing dan clickjacking",
                                  lambda v: v == 0, "Tidak ada iframe mencurigakan"),
        "popup_window":          (lambda v: v == 1, "Halaman membuka popup secara otomatis — taktik umum untuk mengelabui pengguna",
                                  lambda v: v == 0, "Tidak ada popup otomatis"),
        "onmouseover":           (lambda v: v == 1, "Halaman memanipulasi URL saat mouse hover — menyembunyikan tujuan link asli",
                                  lambda v: v == 0, "Tidak ada manipulasi link saat hover"),
        "right_clic":            (lambda v: v == 1, "Klik kanan dinonaktifkan — taktik menyembunyikan source code dari pengguna",
                                  lambda v: v == 0, "Klik kanan berfungsi normal"),
        "empty_title":           (lambda v: v == 1, "Halaman tidak memiliki judul (title kosong) — indikasi halaman dibuat terburu-buru",
                                  lambda v: v == 0, "Halaman memiliki judul yang normal"),
        "domain_in_title":       (lambda v: v == 0, "Nama domain tidak tercantum di judul halaman",
                                  lambda v: v == 1, "Judul halaman mencantumkan nama domain — tanda situs yang konsisten"),
        "domain_with_copyright": (lambda v: v == 0, "Tidak ada pernyataan hak cipta domain resmi",
                                  lambda v: v == 1, "Ada pernyataan copyright dengan nama domain — tanda situs resmi"),
        "ratio_extHyperlinks":   (lambda v: v > 0.5,  "Lebih dari setengah link mengarah ke domain lain — konten hampir semua dari luar",
                                  lambda v: v < 0.2,  "Sebagian besar link mengarah ke domain sendiri"),
        "ratio_intHyperlinks":   (lambda v: v < 0.1,  "Hampir tidak ada link internal — situs tidak punya konten navigasi sendiri",
                                  lambda v: v > 0.5,  "Banyak link internal — menandakan situs punya struktur konten yang wajar"),
        "nb_external_redirection":(lambda v: v > 2,   "Banyak redirect ke domain eksternal berbeda — pola phishing multi-hop",
                                   lambda v: v == 0,  "Tidak ada redirect ke domain luar"),
        "ratio_extRedirection":  (lambda v: v > 0.3,  "Proporsi link redirect eksternal tinggi",
                                  lambda v: v == 0,   "Tidak ada link redirect eksternal mencurigakan"),
        "links_in_tags":         (lambda v: v > 50,   "Lebih dari setengah resource (script, CSS) dimuat dari domain luar",
                                  lambda v: v < 20,   "Sebagian besar resource dimuat dari domain sendiri"),
        "nb_extCSS":             (lambda v: v > 3,    "Banyak file CSS dari domain eksternal — bisa dipakai untuk menyembunyikan konten",
                                  lambda v: v == 0,   "Tidak ada CSS dari domain luar"),
        "ratio_extMedia":        (lambda v: v > 50,   "Lebih dari setengah media (gambar, video) dari domain lain",
                                  lambda v: v < 20,   "Sebagian besar media dihosting di domain sendiri"),
        "page_rank":             (lambda v: v < 1.0,  "PageRank sangat rendah — situs belum dikenal mesin pencari",
                                  lambda v: v >= 4.0, "PageRank tinggi — situs sudah dikenal dan terpercaya"),
        "google_index":          (lambda v: v == 0,   "Situs tidak terindeks Google — baru dibuat atau sengaja disembunyikan",
                                  lambda v: v == 1,   "Situs sudah terindeks Google — menandakan keberadaan yang benign"),
        "web_traffic":           (lambda v: 0 < v < 100,  "Traffic sangat rendah berdasarkan data API — situs hampir tidak dikenal",
                                  lambda v: v > 10000,    "Traffic tinggi — situs populer dan sudah dikenal luas"),
        "dns_record":            (lambda v: v == 0,   "Tidak punya DNS record valid — sangat mencurigakan",
                                  lambda v: v == 1,   "DNS record valid"),
        "domain_age":            (lambda v: 0 < v < 30,  "Domain sangat baru (< 30 hari) — situs phishing sering pakai domain baru",
                                  lambda v: v > 365,     "Domain sudah lama terdaftar (> 1 tahun) — indikasi situs terpercaya"),
        "domain_registration_length": (lambda v: 0 < v < 180, "Masa registrasi domain sangat singkat — situs phishing jarang registrasi jangka panjang",
                                        lambda v: v > 720, "Domain diregistrasi untuk jangka panjang — menandakan komitmen situs resmi"),
        "whois_registered_domain": (lambda v: v == 0, "Data registrasi domain WHOIS tidak ditemukan",
                                     lambda v: v == 1, "Data WHOIS tersedia — domain terdaftar secara resmi"),
        "statistical_report":    (lambda v: v == 1,  "Terdeteksi oleh laporan statistik keamanan — pola URL mencurigakan secara kumulatif",
                                  lambda v: v == 0,  "Tidak memenuhi threshold laporan statistik keamanan"),
    }

    if feature_name in feature_rules:
        phish_fn, phish_reason, benign_fn, benign_reason = feature_rules[feature_name]
        if phish_fn(feature_value) and phish_reason:
            return "PHISHING", phish_reason
        if benign_fn(feature_value) and benign_reason:
            return "BENIGN", benign_reason

    return "NEUTRAL", "Nilai fitur tidak memenuhi kondisi ekstrem phishing maupun benign secara definitif"


# ── BAGIAN 12 — PROMPT LLM & GENERATE EXPLANATION ────────────────────

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

# Fitur dengan nilai 0 karena API gagal (bukan kondisi nyata 0) — percayai arah SHAP
WEB_FETCH_DEPENDENT = {
    "google_index", "page_rank", "web_traffic", "dns_record",
    "domain_age", "domain_registration_length", "whois_registered_domain",
    "nb_hyperlinks", "ratio_intHyperlinks", "ratio_extHyperlinks",
    "nb_extCSS", "ratio_extRedirection", "login_form", "external_favicon",
    "links_in_tags", "ratio_intMedia", "ratio_extMedia",
}


def build_llm_prompt(url, category, p_phish, model_main, top_features_with_reasons,
                     rekomendasi_tetap, decision_source=None):
    lines = []
    for i, f in enumerate(top_features_with_reasons, 1):
        nilai_str = f"Nilai: {f['value']} | " if "value" in f else ""
        lines.append(
            f"  {i}. Fitur: {f['display_name']} | {nilai_str}Status: {f['impact']} | Hasil Ekstraksi: {f['reason']}"
        )

    features_block = "\n".join(lines) if lines else "  (Data fitur tidak tersedia)"
    is_rule_based  = (decision_source == "rule_based_prefilter_phishing")

    konteks_model = (
        "- Sumber Keputusan: Rule-Based Prefilter (BUKAN model machine learning). URL ini langsung divonis "
        "PHISHING karena melanggar satu atau lebih rule keamanan URL yang tercantum pada DATA INTEGRITAS FITUR "
        "di bawah, sebelum sempat dievaluasi oleh model machine learning.\n"
        if is_rule_based else ""
    )

    instruksi = [
        f"Jelaskan secara logis mengapa skor bisa bernilai {p_phish:.2f} berdasarkan data fitur di atas.",
        "Fokuskan penjelasan pada dukungan untuk Kesimpulan Akhir: jika hasilnya BENIGN, jelaskan bahwa fitur "
        "utama mendukung benign; jika hasilnya PHISHING, jelaskan bahwa fitur utama mendukung phishing.",
        "Jangan menggunakan kalimat yang meragukan kesimpulan akhir. Tulis dengan tegas sesuai hasil deteksi.",
        "JANGAN PERNAH mengada-ada atau membawa nama fitur yang tidak tertulis pada data di atas!",
    ]

    if is_rule_based:
        instruksi.append(
            "Karena Sumber Keputusan adalah Rule-Based Prefilter, kamu WAJIB menyebutkan secara EKSPLISIT "
            "nama rule/fitur pada DATA INTEGRITAS FITUR yang terpicu (gunakan nama pada kolom 'Fitur') beserta "
            "alasannya (kolom 'Hasil Ekstraksi'), karena rule-rule itulah satu-satunya alasan URL ini "
            "dikategorikan PHISHING."
        )
        instruksi.append(
            "JANGAN menyebut SHAP, Random Forest, XGBoost, atau analisis model machine learning lain sebagai "
            "dasar keputusan — keputusan ini sepenuhnya berasal dari rule-based prefilter di atas."
        )

    instruksi.append(
        f"Kamu HARUS mengakhiri kalimat penjelasanmu tepat dengan teks instruksi ini tanpa diubah: {rekomendasi_tetap}"
    )

    instruksi_block = "\n".join(f"{i}. {teks}" for i, teks in enumerate(instruksi, 1))

    return (
        f"Kamu adalah analis keamanan siber profesional. Tugasmu adalah menulis penjelasan ringkas (3-4 kalimat) "
        f"dalam Bahasa Indonesia mengenai hasil deteksi sistem terhadap URL berikut.\n\n"
        f"HASIL DETEKSI SISTEM (WAJIB DIIKUTI):\n"
        f"- URL yang diperiksa: {url}\n"
        f"- Kesimpulan Akhir: {category.upper()}\n"
        f"- Probabilitas Phishing: {p_phish:.2f} (Threshold Bahaya >= {FINAL_THRESHOLD})\n"
        f"- Model Utama: {model_main}\n"
        f"{konteks_model}\n"
        f"DATA INTEGRITAS FITUR (JANGAN DIUBAH ATAU DIPUTARBALIKKAN):\n"
        f"{features_block}\n\n"
        f"PETUNJUK PENULISAN PENJELASAN:\n"
        f"{instruksi_block}"
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
        p_phish = clamp01(max(PREFILTER_PHISHING_MIN_CONF, risk_score / 10.0))
    else:
        p_phish = stack_prob if stack_prob is not None else (
            0.5 * rf_prob + 0.5 * xgb_prob if rf_prob is not None and xgb_prob is not None
            else rf_prob or xgb_prob or 0.0
        )
    category = "phishing" if final_label == 1 else "benign"

    if decision_source == "rf_only":
        active_model, model_main = rf_model, "Random Forest"
    elif decision_source == "xgb_only":
        active_model, model_main = xgb_model, "XGBoost"
    elif decision_source == "rule_based_prefilter_phishing":
        active_model, model_main = None, "Rule-Based Prefilter"
    else:
        rf_p  = rf_prob  if rf_prob  is not None else -1.0
        xgb_p = xgb_prob if xgb_prob is not None else -1.0
        if rf_model is not None and xgb_model is not None:
            active_model, model_main = (rf_model, "Random Forest") if rf_p >= xgb_p else (xgb_model, "XGBoost")
        elif rf_model is not None:
            active_model, model_main = rf_model, "Random Forest"
        elif xgb_model is not None:
            active_model, model_main = xgb_model, "XGBoost"
        else:
            active_model, model_main = None, "Unknown"
    if stack_prob is not None and decision_source not in ("rf_only", "xgb_only"):
        model_main = "RF + XGB + Stacking"

    feat_array = np.array([[feats_full.get(c, 0.0) for c in cols]], dtype=float)
    shap_items = []
    if active_model is not None:
        shap_items = get_shap_top(active_model, feat_array, cols, top_n=len(cols))
        if not shap_items:
            other = xgb_model if active_model is rf_model else rf_model
            if other is not None:
                shap_items = get_shap_top(other, feat_array, cols, top_n=len(cols))

    shap_items = [{**f, "value": feats_full.get(f["name"], 0)} for f in shap_items]

    rekomendasi_tetap = "REKOMENDASI: BLOKIR." if final_label == 1 else "REKOMENDASI: IZINKAN."
    top_features_reasons = []
    top_features = []

    if decision_source == "rule_based_prefilter_phishing":
        for level_key, level_label, level_weight in (
            ("vi_hits",   "Sangat Penting", 3.0),
            ("imp_hits",  "Penting",         2.0),
            ("less_hits", "Cukup Penting",   1.0),
        ):
            for fname in (rule_detail or {}).get(level_key, []):
                feat_val = feats_full.get(fname, 0)
                reason   = RULE_REASON_MAP.get(fname, "Rule ini terpicu dan menjadi salah satu indikator phishing.")
                top_features_reasons.append({
                    "display_name": FEATURE_LABEL_MAP.get(fname, fname.replace("_", " ")),
                    "value":  feat_val,
                    "impact": "PHISHING",
                    "reason": f"[{level_label}] {reason}",
                })
                top_features.append({
                    "name": fname, "value": feat_val, "impact": "PHISHING", "reason": reason,
                    "shap_value": level_weight, "shap_value_signed": level_weight,
                    "abs_shap": level_weight, "shap_direction": "PHISHING", "rule_level": level_label,
                })
    else:
        for f in shap_items:
            feat_val = feats_full.get(f["name"], 0)
            rule_dir, reason = get_phishing_risk_direction(f["name"], feat_val)
            shap_dir = f["direction"]

            if abs(f["shap_signed"]) < 1e-6:
                display_impact = rule_dir if rule_dir != "NEUTRAL" else shap_dir
            elif f["name"] in WEB_FETCH_DEPENDENT and feat_val == 0:
                display_impact = shap_dir
                if shap_dir == "BENIGN":
                    reason = "Berdasarkan analisis model, fitur ini berkontribusi pada keamanan URL"
                elif shap_dir == "PHISHING":
                    reason = "Berdasarkan analisis model, fitur ini menambah risiko phishing"
                else:
                    reason = "Tidak ada dampak signifikan dari fitur ini"
            elif rule_dir == shap_dir or (rule_dir != "NEUTRAL" and shap_dir == "NEUTRAL"):
                display_impact = rule_dir
            else:
                display_impact = shap_dir
                if shap_dir == "BENIGN":
                    reason = "Analisis model: fitur ini mendukung keamanan URL"
                elif shap_dir == "PHISHING":
                    reason = "Analisis model: fitur ini menambah indikasi phishing"

            top_features_reasons.append({
                "display_name": FEATURE_LABEL_MAP.get(f["name"], f["name"].replace("_", " ")),
                "value": feat_val, "impact": display_impact, "reason": reason,
            })
            top_features.append({
                "name": f["name"], "value": feat_val, "impact": display_impact, "reason": reason,
                "shap_value": f["shap_signed"], "shap_value_signed": f["shap_signed"],
                "abs_shap": f["abs_shap"], "shap_direction": shap_dir,
            })

    prompt = build_llm_prompt(url, category, p_phish, model_main, top_features_reasons,
                              rekomendasi_tetap, decision_source)
    future = executor.submit(fetch_llm_reasoning_safe, prompt)
    try:
        llm_reasoning = future.result(timeout=15)
    except Exception:
        llm_reasoning = (
            f"Situs secara dominan terdeteksi sebagai {category.upper()} "
            f"oleh {model_main} dengan keyakinan {p_phish * 100:.1f}%."
        )

    return {
        "top_influential_features": top_features,
        "main_contributing_model":  model_main,
        "phishing_probability":     round(p_phish, 4),
        "llm_reasoning":            llm_reasoning,
        "shap_items":               shap_items,
    }
