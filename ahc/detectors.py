import ast, re, math
from collections import Counter

def _ast_sig(code: str) -> Counter:
    try:
        tree = ast.parse(code)
    except Exception:
        return Counter({"parse_error": 1})
    return Counter(type(n).__name__ for n in ast.walk(tree))

def _tok(code: str):
    return re.findall(r"\w+|[^\w\s]", code)

def _ngram(code: str, n=5) -> Counter:
    toks = _tok(code)
    return Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))

def _cos(a: Counter, b: Counter) -> float:
    inter = set(a) & set(b)
    num = sum(a[x]*b[x] for x in inter)
    den = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
    return num/den if den else 0.0

def _stylometry(code: str) -> dict:
    ids = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
    comments = re.findall(r"#[^\n]*", code)
    snakes = sum(1 for s in ids if "_" in s)
    camels = sum(1 for s in ids if re.search(r"[a-z][A-Z]", s))
    return {
        "ident_avg_len": (sum(map(len, ids))/max(1, len(ids))),
        "snake_ratio": snakes/max(1, len(ids)),
        "camel_ratio": camels/max(1, len(ids)),
        "comment_density": len(comments)/max(1, code.count("\n")+1)
    }

def ai_copy_score(student_code: str, peer_codes: list[str], baseline_ai: list[str]) -> dict:
    ast_s = _ast_sig(student_code)
    ng_s = _ngram(student_code, 5)
    sty  = _stylometry(student_code)

    sim_ai = max((_cos(ast_s, _ast_sig(ai))*0.5 + _cos(ng_s, _ngram(ai, 5))*0.5) for ai in baseline_ai) if baseline_ai else 0.0
    sim_peers = max((_cos(ng_s, _ngram(p, 5)) for p in peer_codes), default=0.0)
    footprints = int(bool(re.search(r"As an AI|I cannot|I am just an AI", student_code)))

    score = min(1.0, 0.6*sim_ai + 0.3*sim_peers + 0.1*footprints + 0.1*(sty["camel_ratio"] > 0.2))
    reasons = []
    if sim_ai > 0.6: reasons.append("דמיון גבוה לקורפוס פתרונות LLM")
    if sim_peers > 0.65: reasons.append("דמיון גבוה לפתרון עמית")
    if footprints: reasons.append("טביעות אצבע טקסטואליות")

    return {"score": round(score, 3), "reasons": reasons, "features": {"sim_ai": sim_ai, "sim_peers": sim_peers, **sty}}
