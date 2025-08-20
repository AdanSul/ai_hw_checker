import os, re, json, time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
from json import JSONDecodeError

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from ahc.validators import validate_tasks

load_dotenv()

DEFAULT_MODEL = os.getenv("AHC_PARSE_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("AHC_PARSE_TEMPERATURE", "0.0"))

_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BRACES     = re.compile(r"(\{.*\})", re.DOTALL)

def _extract_json_block(text: str) -> str | None:
    m = _JSON_FENCE.search(text)
    if m: return m.group(1).strip()
    m = _BRACES.search(text)
    return m.group(1).strip() if m else None

# ---- debug dump + sanitizer ----
def _dump_debug(name: str, text: str):
    logdir = Path("outputs/logs"); logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"{name}_{int(time.time())}.txt").write_text(text, encoding="utf-8")

# escape any backslash that isn't followed by a valid JSON escape char
_BAD_BSLASH = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')
def _sanitize_json_text(s: str) -> str:
    # 1) fix bad backslashes like C:\Users -> C:\\Users
    s = _BAD_BSLASH.sub(r"\\\\", s)
    # 2) normalize smart quotes if present
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return s

def llm_call_json(system_prompt: str, user_prompt: str,
                  model: str | None = None, temperature: float | None = None) -> Dict[str, Any]:
    """Call OpenAI in JSON mode, then safely load JSON (with sanitize fallback)."""
    chat = ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        max_retries=1,
        # Force JSON response when supported (gpt-4o, gpt-4o-mini, etc.)
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    resp = chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content

    block = _extract_json_block(resp) or resp
    try:
        return json.loads(block)
    except JSONDecodeError:
        # Try sanitized version
        fixed = _sanitize_json_text(block)
        try:
            return json.loads(fixed)
        except Exception as e2:
            # dump for debugging and re-raise
            _dump_debug("parse_raw", resp)
            _dump_debug("parse_block", block)
            _dump_debug("parse_fixed", fixed)
            raise ValueError(f"Failed to parse JSON from model output even after sanitize: {e2}")

def _to_int(v, default=10):
    try:
        return int(v)
    except Exception:
        return default

def _normalize_task(t: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonicalize fields, coerce types, and set sensible defaults.
    """
    if "task_id" not in t:
        if "task_number" in t:
            t["task_id"] = t["task_number"]
        elif "id" in t:
            t["task_id"] = t["id"]
        else:
            t["task_id"] = "task1"
    tid = str(t["task_id"]).strip()
    if re.fullmatch(r"\d+", tid):
        tid = f"task{tid}"
    t["task_id"] = tid

    t.setdefault("description", "")
    t.setdefault("examples", "")
    t.setdefault("type", "code")
    t.setdefault("lang", "python")

    fname = t.get("expected_filename")
    if not isinstance(fname, str) or not fname.strip():
        t["expected_filename"] = f"{t['task_id']}.py"
    else:
        t["expected_filename"] = fname.strip()
    t["points"] = _to_int(t.get("points", 10), default=10)
    for k in ("task_id", "description", "expected_filename"):
        if isinstance(t.get(k), str):
            t[k] = t[k].strip()
    return t


def _merge_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    variants: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for parsed in trials:
        for t in parsed.get("tasks", []):
            try:
                t = _normalize_task(dict(t))
                if "task_id" in t and "description" in t:
                    variants[t["task_id"]].append(t)
            except Exception:
                continue
    merged: list[Dict[str, Any]] = []
    for tid, items in variants.items():
        key_counts = Counter(
            (it["description"], it.get("expected_filename",""), it.get("type","code"), it.get("lang","python"))
            for it in items
        )
        if not key_counts:
            continue
        best_key, _ = max(key_counts.items(), key=lambda kv: (kv[1], len(kv[0][0])))
        desc, fname, typ, lang = best_key
        pts_counts = Counter(int(it.get("points", 10)) for it in items)
        points = pts_counts.most_common(1)[0][0] if pts_counts else 10
        merged.append({
            "task_id": tid,
            "description": desc,
            "expected_filename": fname or f"{tid}.py",
            "type": typ,
            "lang": lang,
            "points": points,
            "examples": next((it.get("examples","") for it in items if it.get("examples")), "")
        })
    def _tid_key(task_id: str):
        s = str(task_id) 
        m = re.search(r"(\d+)", s)
        return (int(m.group(1)) if m else 1_000_000, s)

    merged.sort(key=lambda t: _tid_key(t["task_id"]))
    return {"tasks": merged}

def parse_assignment(assignment_path: str, model: str = DEFAULT_MODEL,
                     temperature: float = DEFAULT_TEMPERATURE, n_trials: int = 1) -> Dict[str, Any]:
    # n_trials=1 for speed & easier debugging; increase later if needed
    spec = Path(assignment_path).read_text(encoding="utf-8")
    system = (
        "You extract a STRICT JSON schema of tasks from an academic assignment. "
        "Return ONLY JSON. No explanations. Ensure valid UTF-8 and correct escaping."
    )
    user = (
        "Extract tasks from the assignment spec below.\n"
        "Return JSON with a single key 'tasks': a list of objects.\n"
        "Each task MUST include: task_id, description, expected_filename (or infer), "
        "type ('code'|'short_answer'|'math_proof'), lang (when type='code'), points (int), examples (optional).\n\n"
        "Assignment spec:\n-----\n"
        f"{spec}\n-----\n"
        "Return ONLY JSON."
    )
    trials: List[Dict[str, Any]] = []
    for _ in range(max(1, n_trials)):
        data = llm_call_json(system, user, model=model, temperature=temperature)
        tasks = [_normalize_task(t) for t in data.get("tasks", []) if isinstance(t, dict)]
        trials.append({"tasks": tasks})
    merged = _merge_trials(trials)
    return validate_tasks(merged)
