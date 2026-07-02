# ══════════════════════════════════════════════════════════════════════
# ██  BAGIAN 1 — KONSTANTA & KONFIGURASI GLOBAL
# ══════════════════════════════════════════════════════════════════════

PHISHING_CLASS_VALUE        = 1
FINAL_THRESHOLD             = 0.6
DEFAULT_USE_PREFILTER       = True
DEFAULT_DECISION_MODE       = "hybrid_prefilter"
PREFILTER_PHISHING_MIN_CONF = 0.95
WEB_FETCH_TIMEOUT           = 6
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

SUSPICIOUS_TLD = [
    "zip","xyz","top","tk","ga","ml","gq","cf","pw","cc","club","ws","biz",
    "online","site","live","work","icu","info","cn","ru","loan","download","click",
]
STANDARD_PORTS = {21, 22, 23, 80, 443, 445, 1433, 1521, 3306, 3389}
SHORTENERS = {
    "bit.ly","goo.gl","tinyurl.com","ow.ly","t.co","is.gd",
    "buff.ly","adf.ly","bit.do","cutt.ly",
}
PHISH_HINTS = [
    "login","verify","update","secure","account","bank","paypal","apple",
    "microsoft","confirm","signin","password",
]
BRANDS = [
    "google","facebook","apple","microsoft","amazon","paypal",
    "instagram","whatsapp","telegram","netflix","github","linkedin",
]

# Teks alasan tiap rule — dikirim ke LLM agar penjelasan berbasis rule yang benar-benar terpicu
RULE_REASON_MAP = {
    # Sangat Penting
    "suspicious_tld":     "TLD domain termasuk dalam daftar TLD yang sering disalahgunakan untuk phishing, spam, dan malware (misalnya .xyz, .top, .click, .icu, .pw, dll), sehingga menjadi indikator langsung domain mencurigakan.",
    "random_domain":      "Nama domain terdeteksi acak/tidak bermakna (mirip hasil generate otomatis/DGA), yang umum dipakai pada domain phishing berumur pendek dan bereputasi rendah.",
    "ip":                 "URL menggunakan alamat IP secara langsung sebagai host (bukan nama domain), taktik umum untuk menyamarkan identitas domain asli pada phishing.",
    "http_in_path":       "Kata 'http' muncul pada path URL, mengindikasikan upaya penyamaran (URL-in-URL) agar URL terlihat seperti tautan yang sah.",
    # Penting (bobot x2)
    "nb_at":              "Terdapat simbol '@' pada URL, yang dapat menyembunyikan alamat tujuan sebenarnya karena bagian sebelum '@' diabaikan browser.",
    "nb_subdomains":      "Jumlah subdomain lebih dari 3, membuat URL terlihat kompleks dan menyerupai struktur domain resmi untuk mengelabui pengguna.",
    "nb_dots":            "Jumlah titik (.) pada URL lebih dari 4, menunjukkan struktur domain yang kompleks dan sering dipakai untuk spoofing domain.",
    "nb_slash":           "Jumlah garis miring (/) pada URL lebih dari 7, membuat struktur path lebih panjang dan membingungkan pengguna.",
    "length_hostname":    "Panjang hostname lebih dari 30 karakter, sering dipakai untuk menyembunyikan domain utama melalui obfuscation.",
    "nb_percent":         "Jumlah karakter persen (%) pada URL lebih dari 5, mengindikasikan encoding karakter berlebihan untuk menyamarkan isi URL.",
    "nb_tilde":           "Terdapat karakter tilde (~) pada URL, jarang muncul pada URL normal dan dapat menunjukkan pola manipulatif.",
    "nb_semicolumn":      "Terdapat karakter titik koma (;) pada URL, jarang dipakai pada URL normal dan dapat dimanfaatkan untuk memanipulasi query/path.",
    "nb_star":            "Terdapat karakter bintang (*) pada URL, tidak umum dipakai pada URL normal dan dapat menunjukkan pola manipulasi.",
    "nb_comma":           "Terdapat karakter koma (,) pada URL, yang dapat dipakai untuk memanipulasi parameter dan struktur URL.",
    "nb_dollar":          "Terdapat simbol dolar ($) pada URL, jarang muncul pada URL situs resmi dan bisa menjadi indikator manipulasi.",
    "nb_qm":              "Jumlah tanda tanya (?) pada URL lebih dari 2, query string berlebihan sering dipakai untuk menyembunyikan parameter berbahaya.",
    "nb_colon":           "Jumlah titik dua (:) pada URL lebih dari 1 (di luar protokol), dapat dipakai untuk memanipulasi protokol atau port URL.",
    "nb_eq":              "Jumlah simbol sama dengan (=) pada URL lebih dari 8, menunjukkan parameter URL yang kompleks dan manipulatif.",
    "nb_and":             "Jumlah simbol '&' pada URL lebih dari 3, banyak parameter yang dipisahkan '&' sering dipakai untuk menyamarkan URL phishing.",
    "nb_hyphens":         "Jumlah tanda hubung (-) pada URL lebih dari 3, sering dipakai membuat domain phishing menyerupai nama website yang resmi.",
    "nb_underscore":      "Jumlah garis bawah (_) pada URL lebih dari 3, dipakai untuk memodifikasi struktur URL agar terlihat berbeda dari domain aslinya.",
    # Cukup Penting (bobot x1)
    "ratio_digits_url":   "Lebih dari 30% karakter pada hostname berupa angka, sering menunjukkan domain acak atau hasil generate otomatis.",
    "port":               "URL menggunakan port di luar port standar (selain 21, 22, 23, 80, 443, 445, 1433, 1521, 3306, 3389), menjadi indikator tambahan phishing meski tidak kuat jika berdiri sendiri.",
    "shortening_service": "URL menggunakan layanan pemendek tautan (misalnya bit.ly, tinyurl), yang sering dimanfaatkan untuk menyembunyikan tujuan asli URL.",
}

# Label tingkat bobot rule — dipakai untuk konteks LLM reasoning
RULE_LEVEL_LABEL = {"vi_hits": "Sangat Penting", "imp_hits": "Penting", "less_hits": "Cukup Penting"}

# Fitur yang HANYA ada jika web fetch berhasil — penentu mode 81 vs 37
WEB_CONTENT_KEYS = {
    "nb_hyperlinks","ratio_intHyperlinks","ratio_extHyperlinks","nb_extCSS",
    "ratio_extRedirection","ratio_extErrors","login_form","external_favicon",
    "links_in_tags","ratio_intMedia","ratio_extMedia","iframe","popup_window",
    "safe_anchor","onmouseover","right_clic","empty_title","domain_in_title",
    "domain_with_copyright","whois_registered_domain","domain_registration_length",
    "domain_age","web_traffic","dns_record","google_index","page_rank","nb_external_redirection",
}
