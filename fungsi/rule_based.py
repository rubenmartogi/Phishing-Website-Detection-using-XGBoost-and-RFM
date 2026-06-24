# ══════════════════════════════════════════════════════════════════════
# ██  BAGIAN 7 — RULE-BASED PREFILTER (24 RULE)
# ══════════════════════════════════════════════════════════════════════
# Filter sebelum ML. 24 rule, 3 tingkat:
#   Sangat Penting (4)  → satu saja terpicu → langsung PHISHING
#   Penting (17, ×2)    → masuk risk_score
#   Cukup Penting (3, ×1) → masuk risk_score
# Vonis PHISHING jika: ada vi_hits ATAU risk_score >= 5

from fungsi.features import extract_url_features


def rule_based_eval(url, return_detail=False):
    feats = extract_url_features(url)

    very_important = {
        "suspicious_tld":  feats.get("suspicious_tld", 0) == 1,
        "random_domain":   feats.get("random_domain", 0) == 1,
        "ip":              feats.get("ip", 0) == 1,
        "http_in_path":    feats.get("http_in_path", 0) == 1,
    }
    important = {
        "nb_at":           feats.get("nb_at", 0) >= 1,
        "nb_subdomains":   feats.get("nb_subdomains", 0) > 3,
        "nb_dots":         feats.get("nb_dots", 0) > 4,
        "nb_slash":        feats.get("nb_slash", 0) > 7,
        "length_hostname": feats.get("length_hostname", 0) > 30,
        "nb_percent":      feats.get("nb_percent", 0) > 5,
        "nb_tilde":        feats.get("nb_tilde", 0) >= 1,
        "nb_semicolumn":   feats.get("nb_semicolumn", 0) >= 1,
        "nb_star":         feats.get("nb_star", 0) >= 1,
        "nb_comma":        feats.get("nb_comma", 0) >= 1,
        "nb_dollar":       feats.get("nb_dollar", 0) >= 1,
        "nb_qm":           feats.get("nb_qm", 0) > 2,
        "nb_colon":        feats.get("nb_colon", 0) > 1,
        "nb_eq":           feats.get("nb_eq", 0) > 8,
        "nb_and":          feats.get("nb_and", 0) > 3,
        "nb_hyphens":      feats.get("nb_hyphens", 0) > 3,
        "nb_underscore":   feats.get("nb_underscore", 0) > 3,
    }
    less_important = {
        "ratio_digits_url":   feats.get("ratio_digits_url", 0) > 0.3,
        "port":               feats.get("port", 0) == 1,
        "shortening_service": feats.get("shortening_service", 0) == 1,
    }

    vi_hits   = [k for k, v in very_important.items() if v]
    imp_hits  = [k for k, v in important.items() if v]
    less_hits = [k for k, v in less_important.items() if v]

    # risk_score = (Penting × 2) + (Cukup Penting × 1)
    risk_score = 2 * len(imp_hits) + len(less_hits)

    if vi_hits or risk_score >= 5:
        category, rule_flag = "Phishing", 1
    elif risk_score > 0:
        category, rule_flag = "Suspicious", 0
    else:
        category, rule_flag = "Benign", 0

    if return_detail:
        return risk_score, category, rule_flag, {
            "vi_count":   len(vi_hits),
            "imp_count":  len(imp_hits),
            "less_count": len(less_hits),
            "vi_hits":    vi_hits,
            "imp_hits":   imp_hits,
            "less_hits":  less_hits,
        }
    return risk_score, category, rule_flag
