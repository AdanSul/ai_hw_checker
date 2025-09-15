from collections import Counter, defaultdict
from math import sqrt, log
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import re
import difflib
import ast
from functools import lru_cache

# -----------------------------
# helpers:
# -----------------------------
def _clip01(x: float) -> float:
    """
    Clamp value into [0,1].
    """
    return max(0.0, min(1.0, float(x)))


def _canonicalize_tokens(tokens: List[str]) -> List[str]:
    """Identifiers -> ID, numbers -> NUM; keep operators/punctuation."""
    out = []
    for t in tokens:
        if re.fullmatch(r"\d+(\.\d+)?", t):
            out.append("NUM")
        elif re.fullmatch(r"[A-Za-z_]\w*", t):
            out.append("ID")
        else:
            out.append(t)
    return out


# ---------- safer, string-aware line-comment stripping ----------
def _strip_line_comments_preserving_strings(s: str, marker: str) -> str:
    """
    Remove single-line comments that start with 'marker' (e.g., '#', '//')
    only when NOT inside string literals (handles ', ", ''' ''', and \"\"\").
    """
    assert marker in ("#", "//")
    i = 0
    n = len(s)
    out = []
    in_str: Optional[str] = None  # "'", '"', "'''", '"""', or None
    while i < n:
        # Enter string?
        if in_str is None:
            # triple-quoted first
            if s.startswith("'''", i):
                in_str = "'''"
                out.append("'''")
                i += 3
                continue
            if s.startswith('"""', i):
                in_str = '"""'
                out.append('"""')
                i += 3
                continue
            # single quotes
            if s[i] == "'" or s[i] == '"':
                in_str = s[i]
                out.append(s[i])
                i += 1
                continue

            # comment marker?
            if marker == "#" and s[i] == "#":
                # skip until newline
                while i < n and s[i] != "\n":
                    i += 1
                # keep newline if present
                if i < n:
                    out.append("\n")
                    i += 1
                continue

            if marker == "//" and s.startswith("//", i):
                # skip until newline
                while i < n and s[i] != "\n":
                    i += 1
                if i < n:
                    out.append("\n")
                    i += 1
                continue

            # normal char
            out.append(s[i])
            i += 1
        else:
            # inside string
            if in_str in ("'''", '"""'):
                if s.startswith(in_str, i):
                    out.append(in_str)
                    i += 3
                    in_str = None
                else:
                    out.append(s[i])
                    i += 1
            else:
                # single/double quoted string with escapes
                if s[i] == "\\" and i + 1 < n:
                    out.append(s[i:i+2])
                    i += 2
                elif s[i] == in_str:
                    out.append(s[i])
                    i += 1
                    in_str = None
                else:
                    out.append(s[i])
                    i += 1
    return "".join(out)


def _normalize_text_general(s: str, *, strip_cpp_slashes: bool = False) -> str:
    """
    Light, content-agnostic normalization for code/math/text:
    - remove block comments (/* ... */)  [regex-based; may remove inside strings]
    - remove single-line comments (#) while preserving strings
    - optionally remove C/JS '//' comments while preserving strings
    - lowercase
    - collapse whitespace
    - strip leading/trailing spaces
    """
    if not s:
        return ""
    # Remove block comments (may affect strings; acceptable for general text)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)

    # Remove # line comments (string-aware)
    s = _strip_line_comments_preserving_strings(s, "#")

    # Optionally remove // comments (string-aware)
    if strip_cpp_slashes:
        s = _strip_line_comments_preserving_strings(s, "//")

    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _tokens_general(s: str) -> List[str]:
    """
    Unified tokenizer for code + math + text.
    Splits into alphanumeric words and punctuation separately.
    Example: "a = b+1" -> ["a", "=", "b", "+", "1"]
    """
    if not s:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", s)


def _char_ngrams(s: str, n_min: int = 3, n_max: int = 6) -> Counter:
    s = s.replace("\n", " ")
    ngrams = Counter()
    L = len(s)
    for n in range(n_min, n_max + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            ngrams[s[i:i+n]] += 1
    return ngrams


def _jaccard_shingles(tokens: List[str], w: int = 3) -> set:
    if w <= 0 or len(tokens) < w:
        return set()
    return {" ".join(tokens[i:i+w]) for i in range(len(tokens) - w + 1)}


def _cosine_from_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    if len(a) < len(b):
        for k, v in a.items():
            if k in b:
                dot += v * b[k]
    else:
        for k, v in b.items():
            if k in a:
                dot += v * a[k]
    na = sqrt(sum(v*v for v in a.values()))
    nb = sqrt(sum(v*v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def _topk_mean(xs: List[float], k: int) -> float:
    if not xs:
        return 0.0
    k = max(1, min(k, len(xs)))
    return mean(sorted(xs, reverse=True)[:k])


def _border_ratio(a: List[str], b: List[str]) -> float:
    """
    Fraction of identical prefix+suffix tokens (no trimming), capped.
    Penalizes big shared headers/footers.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    j = 0
    while j < n - i and a[-1-j] == b[-1-j]:
        j += 1
    r = (i + j) / n
    return min(0.6, r)


# ---------- AST canonicalization on RAW (un-normalized) Python source ----------
@lru_cache(maxsize=512)
def _ast_normalize_python(src: str) -> Optional[str]:
    """
    Parse Python RAW source and canonicalize names/consts so renames don't matter.
    Returns a stable string signature (or None if not Python / parse fails).
    """
    try:
        tree = ast.parse(src)
    except Exception:
        return None

    class Canon(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = "FUNC"
            self.generic_visit(node)
            return node
        def visit_AsyncFunctionDef(self, node):
            node.name = "FUNC"
            self.generic_visit(node)
            return node
        def visit_ClassDef(self, node):
            node.name = "CLASS"
            self.generic_visit(node)
            return node
        def visit_Name(self, node):
            # preserve context type but canonicalize id
            return ast.copy_location(ast.Name(id="ID", ctx=type(node.ctx)()), node)
        def visit_arg(self, node):
            node.arg = "ID"
            return node
        def visit_alias(self, node):
            node.name = "ID"
            node.asname = None
            return node
        def visit_Attribute(self, node):
            self.generic_visit(node)
            node.attr = "ATTR"
            return node
        def visit_Constant(self, node):
            v = node.value
            if isinstance(v, (int, float, complex)):
                node.value = 0
            elif isinstance(v, str):
                node.value = "STR"
            else:
                node.value = None
            return node

    tree = Canon().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def compare_with_peers(
    student_submission: str,
    peer_submissions: List[str],
    *,
    shingle_w: int = 3,
    ngram_min: int = 3,          # avoid noisy bigrams
    ngram_max: int = 6,
    weight_token: float = 0.05,  # small weight
    weight_char:  float = 0.20,
    weight_seq:   float = 0.30,  # rename-robust sequence on canonical tokens
    weight_ast:   float = 0.40,  # AST structure (rename-invariant)
    agg_topk: int = 1,
    max_workers: int = 8,
) -> float:
    """
    Peer similarity in [0,1], robust to renames:
      - Weighted token shingles (low weight)
      - Char n-gram TF-IDF cosine
      - Canonicalized token sequence similarity (IDs->ID, nums->NUM)
      - Python AST-normalized similarity (if both parse, using RAW code)
    Penalize large shared borders (headers/footers).
    """
    if not peer_submissions:
        return 0.0

    # Normalize strings for token & char models (do NOT use for AST)
    stu_norm = _normalize_text_general(student_submission)
    peers_norm = [_normalize_text_general(p or "") for p in peer_submissions]

    # Tokenization
    stu_tokens = _tokens_general(stu_norm)
    peers_tokens = [_tokens_general(p) for p in peers_norm]
    stu_can = _canonicalize_tokens(stu_tokens)
    peers_can = [_canonicalize_tokens(toks) for toks in peers_tokens]

    # Shingles + IDF for weighted Jaccard
    stu_sh = _jaccard_shingles(stu_tokens, w=shingle_w)
    peers_sh = [_jaccard_shingles(toks, w=shingle_w) for toks in peers_tokens]
    all_docs_sh = [stu_sh] + peers_sh
    df_sh = defaultdict(int)
    for sset in all_docs_sh:
        for s in sset:
            df_sh[s] += 1
    N_docs = len(all_docs_sh)
    idf_sh = {s: (log((N_docs + 1) / (df_sh[s] + 1)) + 1.0) for s in df_sh}
    def weighted_jaccard(A: set, B: set) -> float:
        if not A and not B:
            return 0.0
        inter = sum(idf_sh[s] for s in (A & B))
        uni   = sum(idf_sh[s] for s in (A | B))
        return (inter / uni) if uni > 0 else 0.0

    # Char n-gram TF-IDF
    docs_ngrams = [_char_ngrams(stu_norm, n_min=ngram_min, n_max=ngram_max)]
    docs_ngrams += [_char_ngrams(t, n_min=ngram_min, n_max=ngram_max) for t in peers_norm]
    N_docs_ng = len(docs_ngrams)
    df_ng = defaultdict(int)
    for c in docs_ngrams:
        for k in c.keys():
            df_ng[k] += 1
    idf_ng = {k: (log((N_docs_ng + 1) / (df_ng[k] + 1)) + 1.0) for k in df_ng}
    def tfidf(counts: Counter) -> Counter:
        if not counts:
            return Counter()
        return Counter({k: counts[k] * idf_ng.get(k, 1.0) for k in counts})
    stu_tfidf = tfidf(docs_ngrams[0])
    peers_tfidf = [tfidf(c) for c in docs_ngrams[1:]]

    # Precompute AST signature of student from RAW source
    stu_ast_sig = _ast_normalize_python(student_submission)

    def score_one(i: int) -> float:
        # Border (header/footer) penalty driver (on normalized tokens)
        border = _border_ratio(stu_tokens, peers_tokens[i])

        # 1) Token shingles (weighted)
        s_tok = weighted_jaccard(stu_sh, peers_sh[i])

        # 2) Char n-gram cosine
        s_char = _cosine_from_counts(stu_tfidf, peers_tfidf[i])

        # 3) Canonical token sequence (rename-invariant), use token lists
        seq_canon = difflib.SequenceMatcher(
            None, stu_can, peers_can[i], autojunk=False
        ).ratio()

        # 4) AST similarity (if both parse) using RAW sources
        peer_ast_sig = _ast_normalize_python(peer_submissions[i])
        if stu_ast_sig is not None and peer_ast_sig is not None:
            s_ast = difflib.SequenceMatcher(None, stu_ast_sig, peer_ast_sig, autojunk=False).ratio()
        else:
            s_ast = 0.0

        # Combine
        combined = (
            weight_seq  * seq_canon +
            weight_ast  * s_ast +
            weight_char * s_char +
            weight_token* s_tok
        )

        # Penalize big borders unless the canonical sequence already matches very well
        penalty = 1.0 - 0.5 * border * (1.0 - seq_canon)  # cap 50% when seq is low
        return _clip01(combined * penalty)

    scores = []
    if len(peer_submissions) > 8 and max_workers > 1:
        # Note: threads give limited benefit for CPU-bound Python (GIL),
        # kept for API parity; consider ProcessPoolExecutor if needed.
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(score_one, i) for i in range(len(peer_submissions))]
            for f in as_completed(futs):
                scores.append(f.result())
    else:
        for i in range(len(peer_submissions)):
            scores.append(score_one(i))

    if not scores:
        return 0.0

    m = max(scores)
    tk = _topk_mean(scores, k=min(agg_topk, len(scores)))  # = m for agg_topk=1
    final = _clip01(0.95 * m + 0.05 * tk)
    return final


def ai_copy_score(
    student_submission: str,
    peer_submissions: List[str],
    baseline_ai: Optional[List[str]] = None
) -> float:
    # Compare with other students
    peer_similarity_score = compare_with_peers(student_submission, peer_submissions)
    # Placeholder if you later add AI-baseline similarity:
    # ai_similarity_score = compare_with_ai_baseline(student_submission, baseline_ai or [])
    # final_score = (peer_similarity_score * 0.4) + (ai_similarity_score * 0.6)
    final_score = peer_similarity_score
    return final_score


# # -----------------------------
# # tests/test_compare_with_peers.py
# # -----------------------------

# def _approx_between(x, lo, hi, eps=1e-6):
#     assert lo - eps <= x <= hi + eps, f"value {x:.4f} not in [{lo}, {hi}]"


# def test_helpers_basic():
#     # _clip01
#     assert _clip01(-1.2) == 0.0
#     assert _clip01(0.0) == 0.0
#     assert _clip01(0.5) == 0.5
#     assert _clip01(1.0) == 1.0
#     assert _clip01(4.2) == 1.0

#     # _normalize_text_general removes comments & squashes spaces (with // opt-in)
#     s = """
#         // a comment
#         # another comment
#         int main() { /* block
#                         comment */ return 0; }
#     """
#     n = _normalize_text_general(s, strip_cpp_slashes=True)
#     assert "comment" not in n
#     assert "  " not in n  # collapsed whitespace
#     assert n.startswith("int main()")

#     # _tokens_general splits words and punctuation
#     toks = _tokens_general("a = b+1; // c")
#     assert toks == ["a", "=", "b", "+", "1", ";", "/", "/", "c"] or toks == ["a","=","b","+","1",";"]


# def test_compare_with_peers_identical():
#     student = "def f(x):\n    return x*x + 1\n"
#     peers = [
#         "def g(y):\n    return y + 2\n",
#         "def f(x):\n    return x*x + 1\n",  # identical
#         "print('hello world')\n"
#     ]
#     score = compare_with_peers(student, peers)
#     # identical should be very high (near 1.0)
#     _approx_between(score, 0.90, 1.00)


# def test_compare_with_peers_light_rename():
#     student = "def square(x):\n    res = x * x + 1\n    return res\n"
#     peers = [
#         "def square(val):\n    r = val*val + 1\n    return r\n",  # light rename/format
#         "def unrelated(a):\n    return a + 2\n"
#     ]
#     score = compare_with_peers(student, peers)
#     # light rename should still be high
#     _approx_between(score, 0.75, 1.00)


# def test_compare_with_peers_different():
#     student = "Natural language paragraph about trees and forests."
#     peers = [
#         "Sorting algorithm in Python using quicksort.",
#         "Matrix multiplication implementation with Numpy.",
#         "Network server in Go handling TCP sockets."
#     ]
#     score = compare_with_peers(student, peers)
#     # very different content should be low
#     _approx_between(score, 0.00, 0.35)


# def test_compare_with_peers_boilerplate_overlap():
#     boilerplate = (
#         "/* Course: CS101, Starter template */\n"
#         "// Do not modify the header below\n"
#         "def main():\n    pass\n"
#     )
#     student = boilerplate + "\n# Student solution\n" \
#               "def solve(a,b):\n    return a*a + b*b\n"
#     peer = boilerplate + "\n# Peer solution\n" \
#            "def solve(a,b):\n    return a*b + a + b\n"

#     score = compare_with_peers(student, [peer])
#     # Expect mid-range due to shared header but different bodies
#     _approx_between(score, 0.20, 0.60)


# # ---- Extra regression tests for normalization & AST-on-raw issues ----
# def test_python_floor_division_survives():
#     s = "def f(a,b):\n    return a // b\n"
#     peers = ["def g(x,y):\n    return x // y\n"]
#     score = compare_with_peers(s, peers)
#     # Using RAW AST, floor-division should not get eaten as a comment.
#     _approx_between(score, 0.50, 1.00)


# def test_hash_in_string_not_comment():
#     s = 's = "not # a comment"  # real comment here\nprint(s)\n'
#     n = _normalize_text_general(s)  # '#' in string should remain
#     assert "comment" in n  # the word inside the string literal should still be present


# # Allow running this file directly without pytest
# if __name__ == "__main__":
#     passed = 0
#     failed = 0
#     for fn in [
#         test_helpers_basic,
#         test_compare_with_peers_identical,
#         test_compare_with_peers_light_rename,
#         test_compare_with_peers_different,
#         test_compare_with_peers_boilerplate_overlap,
#         test_python_floor_division_survives,
#         test_hash_in_string_not_comment,
#     ]:
#         try:
#             fn()
#             print(f"[OK] {fn.__name__}")
#             passed += 1
#         except AssertionError as e:
#             print(f"[FAIL] {fn.__name__}: {e}")
#             failed += 1
#         except Exception as e:
#             print(f"[ERROR] {fn.__name__}: {e}")
#             failed += 1
#     print(f"\nSummary: {passed} passed, {failed} failed")
#     if failed:
#         raise SystemExit(1)