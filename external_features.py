"""
external_features.py
====================
Modul pengayaan fitur eksternal untuk sistem deteksi phishing.

Mengisi 7 fitur WEB_CONTENT_KEYS yang sebelumnya selalu 0:
  - dns_record
  - domain_age
  - domain_registration_length
  - whois_registered_domain
  - page_rank
  - google_index
  - web_traffic

Cara pakai di app.py:
    from external_features import enrich_external_features

    # Di dalam extract_web_content_features(), sebelum return out:
    enrich_external_features(out, hostname)

Dependensi:
    pip install python-whois dnspython requests

API key yang diperlukan (tambahkan ke environment variable atau .env):
    OPR_API_KEY   = key dari https://www.domcop.com/openpagerank/
    SERP_API_KEY  = key dari https://serpapi.com/  (opsional, untuk google_index)
    SEMRUSH_KEY   = key dari https://www.semrush.com/api-analytics/  (opsional, untuk web_traffic)
"""

import os
import socket
import datetime
import logging
from functools import lru_cache
from typing import Dict, Any

import requests

log = logging.getLogger(__name__)

# ── Konfigurasi API Key (baca dari env agar tidak hardcode di kode) ────────────
OPR_API_KEY   = os.environ.get("OPR_API_KEY", "")      # OpenPageRank - GRATIS
SERP_API_KEY  = os.environ.get("SERP_API_KEY", "")     # SerpAPI - opsional
SEMRUSH_KEY   = os.environ.get("SEMRUSH_KEY", "")      # Semrush - opsional

# Timeout per request API eksternal (detik)
API_TIMEOUT = 5

# ── Cache TTL: gunakan lru_cache untuk sesi (restart = reset cache) ────────────
# Untuk produksi skala besar, ganti dengan Redis + TTL per domain.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. DNS RECORD  (tidak butuh API eksternal)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@lru_cache(maxsize=512)
def get_dns_record(hostname: str) -> int:
    """
    Kembalikan 1 jika hostname punya DNS A-record, 0 jika tidak.
    Tidak butuh API key. Pakai socket stdlib Python.
    """
    try:
        result = socket.getaddrinfo(hostname, None, socket.AF_INET)
        return 1 if result else 0
    except (socket.gaierror, OSError):
        return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. WHOIS: domain_age, domain_registration_length, whois_registered_domain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@lru_cache(maxsize=256)
def get_whois_features(hostname: str) -> Dict[str, float]:
    """
    Kembalikan dict berisi:
      - whois_registered_domain : 1 jika WHOIS valid, 0 jika tidak
      - domain_age              : usia domain dalam hari (sejak creation_date)
      - domain_registration_length : lama masa registrasi dalam hari
                                     (expiration_date - creation_date)

    Menggunakan library python-whois. Fallback semua = 0 jika gagal.

    pip install python-whois
    """
    defaults = {
        "whois_registered_domain": 0,
        "domain_age": 0,
        "domain_registration_length": 0,
    }
    try:
        import whois  # python-whois
    except ImportError:
        log.warning("python-whois tidak terinstall. `pip install python-whois`")
        return defaults

    try:
        w = whois.whois(hostname)
        domain_name = w.domain_name

        # Normalisasi: kadang jadi list, kadang string
        if isinstance(domain_name, list):
            domain_name = domain_name[0] if domain_name else None

        if not domain_name:
            return defaults

        creation = w.creation_date
        expiry   = w.expiration_date

        if isinstance(creation, list): creation = creation[0]
        if isinstance(expiry, list):   expiry   = expiry[0]

        now = datetime.datetime.utcnow()

        domain_age = 0
        if isinstance(creation, datetime.datetime):
            domain_age = max(int((now - creation).days), 0)

        reg_length = 0
        if isinstance(creation, datetime.datetime) and isinstance(expiry, datetime.datetime):
            reg_length = max(int((expiry - creation).days), 0)

        return {
            "whois_registered_domain": 1,
            "domain_age": domain_age,
            "domain_registration_length": reg_length,
        }

    except Exception as e:
        log.debug("WHOIS gagal untuk %s: %s", hostname, e)
        return defaults


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. PAGE RANK  — OpenPageRank API (GRATIS: 10.000 req/hari)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Daftar gratis di: https://www.domcop.com/openpagerank/
# Lalu set env:  OPR_API_KEY=xxxxxxxx
#
# Skala output: 0.0 – 10.0  (sama dengan rentang dataset training Anda)

@lru_cache(maxsize=256)
def get_page_rank(hostname: str) -> float:
    """
    Kembalikan PageRank (0.0–10.0) via OpenPageRank API.
    Fallback = 0.0 jika API key tidak ada atau request gagal.
    """
    if not OPR_API_KEY:
        return 0.0

    # Ambil root domain saja (agar cache lebih efisien)
    domain = _root_domain(hostname)

    try:
        resp = requests.get(
            "https://openpagerank.com/api/v1.0/getPageRank",
            params={"domains[]": domain},
            headers={"API-OPR": OPR_API_KEY},
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        rank = data["response"][0].get("page_rank_decimal", 0.0)
        return float(rank) if rank is not None else 0.0
    except Exception as e:
        log.debug("OpenPageRank gagal untuk %s: %s", hostname, e)
        return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. GOOGLE INDEX  — SerpAPI (opsional) atau fallback ringan
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Opsi A (akurat, berbayar): SerpAPI — https://serpapi.com/
#   Set env: SERP_API_KEY=xxxxxxxx
#   Paket gratis: 100 search/bulan
#
# Opsi B (fallback ringan, gratis): cek robots.txt + DNS
#   Ini bukan 100% akurat tapi tidak butuh API key.

@lru_cache(maxsize=256)
def get_google_index(hostname: str) -> int:
    """
    Kembalikan 1 jika situs kemungkinan diindeks Google, 0 jika tidak.

    Urutan prioritas:
      1. SerpAPI site: query  (jika SERP_API_KEY tersedia)
      2. Fallback: cek keberadaan sitemap.xml / robots.txt
         Situs phishing baru jarang punya ini → proxy kasar untuk indexability.
    """
    # ── Opsi A: SerpAPI ──────────────────────────────────────────────────────
    if SERP_API_KEY:
        return _google_index_serpapi(hostname)

    # ── Opsi B: Fallback ringan (tanpa API) ──────────────────────────────────
    return _google_index_fallback(hostname)


def _google_index_serpapi(hostname: str) -> int:
    domain = _root_domain(hostname)
    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={"engine": "google", "q": f"site:{domain}", "api_key": SERP_API_KEY, "num": 1},
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("organic_results", [])
        return 1 if results else 0
    except Exception as e:
        log.debug("SerpAPI gagal untuk %s: %s", hostname, e)
        return 0


def _google_index_fallback(hostname: str) -> int:
    """
    Heuristik ringan: coba fetch robots.txt.
    Situs yang punya robots.txt cenderung sengaja mendaftarkan dirinya ke search engine.
    Ini BUKAN pengganti API yang sempurna — tapi jauh lebih baik dari hardcode 0.
    """
    try:
        url = f"http://{hostname}/robots.txt"
        r = requests.get(url, timeout=API_TIMEOUT, allow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.text) > 20:
            return 1
        return 0
    except Exception:
        return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. WEB TRAFFIC  — Semrush API (opsional) atau fallback = 0
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Semrush API: https://developer.semrush.com/api/v3/analytics/
#   Set env: SEMRUSH_KEY=xxxxxxxx
#   Paket gratis terbatas. Cek "Domain Overview" endpoint.
#
# Catatan: web_traffic dalam dataset training menggunakan skala Alexa rank
#   (semakin kecil = semakin populer, atau sebaliknya tergantung encoding).
#   Sesuaikan normalisasi dengan dataset Anda.

@lru_cache(maxsize=256)
def get_web_traffic(hostname: str) -> float:
    """
    Kembalikan estimasi traffic. Default 0 jika API tidak tersedia.
    Semrush mengembalikan organic traffic bulanan (angka absolut).
    """
    if not SEMRUSH_KEY:
        # Tidak ada API: kembalikan 0 (sama dengan default sebelumnya)
        # Model sudah terlatih dengan banyak 0 di training set untuk domain phishing.
        return 0.0

    domain = _root_domain(hostname)
    try:
        resp = requests.get(
            "https://api.semrush.com/",
            params={
                "type": "domain_organic",
                "key": SEMRUSH_KEY,
                "domain": domain,
                "database": "us",
                "display_limit": 1,
                "export_columns": "Or",  # Organic traffic estimate
            },
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        if len(lines) >= 2:
            traffic = float(lines[1].split(";")[0])
            return traffic
        return 0.0
    except Exception as e:
        log.debug("Semrush gagal untuk %s: %s", hostname, e)
        return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNGSI UTAMA: enrich_external_features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def enrich_external_features(feats: Dict[str, Any], hostname: str) -> None:
    """
    Isi feats dict secara in-place dengan 7 fitur eksternal.

    Panggil di akhir extract_web_content_features() sebelum return out:

        enrich_external_features(out, base_host)

    Semua lookup menggunakan lru_cache sehingga domain yang sama
    tidak dihit API dua kali dalam satu sesi server.
    """
    if not hostname:
        return

    # 1. DNS record (tanpa API, cepat)
    feats["dns_record"] = get_dns_record(hostname)

    # 2. WHOIS features (python-whois, tanpa API key)
    whois_feats = get_whois_features(hostname)
    feats.update(whois_feats)

    # 3. Page rank (OpenPageRank API, gratis 10k/hari)
    feats["page_rank"] = get_page_rank(hostname)

    # 4. Google index (SerpAPI jika ada key, atau fallback robots.txt)
    feats["google_index"] = get_google_index(hostname)

    # 5. Web traffic (Semrush API jika ada key, atau 0)
    feats["web_traffic"] = get_web_traffic(hostname)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper internal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _root_domain(hostname: str) -> str:
    """
    Ambil 2 level terakhir dari hostname.
    'www.sub.google.com' → 'google.com'
    Berguna agar cache lebih efisien dan menghindari duplikasi API call
    untuk subdomain yang sama.
    """
    parts = hostname.lower().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname
