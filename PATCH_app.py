"""
PATCH app.py — 3 perubahan minimal
====================================

Tambahkan file external_features.py di direktori yang sama dengan app.py,
lalu terapkan 3 perubahan berikut.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERUBAHAN 1 — Import modul baru
Tambahkan baris ini di bagian import atas app.py (setelah `import tldextract`):

    from external_features import enrich_external_features

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERUBAHAN 2 — Tambahkan pemanggilan enrich di extract_web_content_features()
Ubah fungsi di app.py baris ~199–261 menjadi seperti ini
(perhatikan 2 baris tambahan di akhir sebelum return):

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

    # ── TAMBAHAN: isi 7 fitur eksternal yang sebelumnya selalu 0 ──────────────
    enrich_external_features(out, base_host)   # <── BARIS BARU INI SAJA

    return out

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERUBAHAN 3 — Variabel environment (buat file .env atau set di OS)
Taruh di direktori root proyek (sejajar dengan app.py):

    # .env
    OPR_API_KEY=isi_dengan_key_dari_openpagerank
    SERP_API_KEY=isi_jika_punya_serpapi   (opsional)
    SEMRUSH_KEY=isi_jika_punya_semrush    (opsional)

Lalu di atas app.py (sebelum import lain), tambahkan:

    from dotenv import load_dotenv
    load_dotenv()

    pip install python-dotenv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Itu saja. Tidak ada perubahan lain di app.py.
"""
