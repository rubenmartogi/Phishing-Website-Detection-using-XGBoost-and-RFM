# ══════════════════════════════════════════════════════════════════════
# ██  BAGIAN 2 & 13 — CEK LOCK FILE LOG & LOGGING KE CSV
# ══════════════════════════════════════════════════════════════════════
# Tulis hasil deteksi ke log_feature_extraction.csv setelah SHAP & LLM selesai.
# Dilindungi threading lock — aman untuk request concurrent.

import csv
import os
import platform
import sys
import threading

_log_lock = threading.Lock()


def check_log_locked(log_path):
    """
    Cek apakah CSV log sedang dibuka Excel/program lain.
    Return: (is_locked: bool, pesan_error: str)
    """
    if not os.path.exists(log_path):
        return False, ""

    try:
        fh = open(log_path, "a", newline="", encoding="utf-8-sig")
    except (PermissionError, OSError):
        return True, (
            "File log (log_feature_extraction.csv) sedang terbuka di Excel atau program lain. "
            "Tutup file tersebut terlebih dahulu, lalu ulangi deteksi."
        )
    except Exception:
        return False, ""

    try:
        if platform.system() == "Windows":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            try:
                import fcntl  # type: ignore[import]
            except ImportError:
                return False, ""
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)
        return False, ""
    except (OSError, PermissionError):
        return True, (
            "File log (log_feature_extraction.csv) sedang terbuka di Excel atau program lain. "
            "Tutup file tersebut terlebih dahulu, lalu ulangi deteksi."
        )
    finally:
        fh.close()


def log_feature_extraction(url, mode, feats, feature_columns_81, status, decision_source, log_path):
    """Thread-safe wrapper untuk _log_feature_extraction_impl."""
    with _log_lock:
        return _log_feature_extraction_impl(url, mode, feats, feature_columns_81, status, decision_source, log_path)


def _try_open_exclusive(path, mode):
    """Buka file secara eksklusif. Raise PermissionError jika terkunci."""
    fh = open(path, mode, newline="", encoding="utf-8-sig")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, PermissionError):
        fh.close()
        raise PermissionError("FILE_LOCKED")
    return fh


def _log_feature_extraction_impl(url, mode, feats, feature_columns_81, status, decision_source, log_path):
    csv_path    = log_path
    write_header = not os.path.exists(csv_path)

    no = 1
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    last_line = lines[-1].strip().split(";")
                    if last_line[0].isdigit():
                        no = int(last_line[0]) + 1
        except Exception:
            no = 1

    headers  = ["NO", "URL", "MODE", "DECISION_SOURCE", "STATUS", "PHISHING_PROB"] + feature_columns_81 + ["TOP_FEATURE", "LLM_REASONING"]

    row_dict = {
        "NO": no, "URL": url, "MODE": mode,
        "DECISION_SOURCE": decision_source, "STATUS": status,
    }

    _EXTERNAL_API_FEATURES = {
        "domain_age", "domain_registration_length", "web_traffic",
        "whois_registered_domain", "google_index", "page_rank", "dns_record",
    }
    _SENTINEL = object()

    for feat in feature_columns_81:
        raw = feats.get(feat, _SENTINEL)
        if raw is _SENTINEL or raw is None:
            row_dict[feat] = "NaN"
        elif feat in _EXTERNAL_API_FEATURES and raw == 0:
            row_dict[feat] = "NaN"
        elif isinstance(raw, (list, dict, tuple)):
            row_dict[feat] = str(raw)
        elif isinstance(raw, float):
            row_dict[feat] = round(raw, 4)
        elif isinstance(raw, int):
            row_dict[feat] = raw
        else:
            row_dict[feat] = raw

    MAX_CELL = 32000
    row_dict["PHISHING_PROB"] = feats.get("_phishing_prob", "")
    row_dict["TOP_FEATURE"]   = str(feats.get("_top_features", "")).replace("\n", "  ||  ").replace("\r", " ")[:MAX_CELL]
    row_dict["LLM_REASONING"] = str(feats.get("_llm_reasoning", "")).replace("\n", " ").replace("\r", " ")[:MAX_CELL]

    try:
        if write_header:
            fh = _try_open_exclusive(csv_path, "w")
            fh.write("sep=;\n")
            writer = csv.DictWriter(fh, fieldnames=headers, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerow(row_dict)
            fh.close()
        else:
            fh = _try_open_exclusive(csv_path, "a")
            writer = csv.DictWriter(fh, fieldnames=headers, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row_dict)
            fh.close()
        print(f"[LOG SUCCESS] Baris #{no} berhasil dicatat.")
        return None
    except PermissionError:
        msg = "Log CSV sedang terbuka di Excel/program lain. Tutup file log terlebih dahulu, lalu coba lagi."
        print(f"[LOG LOCKED] {msg}")
        return msg
    except Exception as e:
        print(f"[LOG ERROR] {e}")
        return str(e)
