import os
import re
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from ahc.cache_utils import get_cached_feedback, store_feedback  

load_dotenv()

DEFAULT_MODEL = os.getenv("AHC_EVAL_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("AHC_EVAL_TEMPERATURE", "0.0"))

# Disk cache for per-task evaluations to deduplicate across students
EVAL_CACHE_DIR = os.getenv("AHC_EVAL_CACHE_DIR", "outputs/cache/evals")
BATCH_ENABLED = os.getenv("AHC_EVAL_BATCH", "0") == "1"  # batch-per-student


# ------------------------------------------------------------------ #
#                          LLM helpers                               #
# ------------------------------------------------------------------ #

def _llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        max_retries=1,
    )

def llm_call_text(system_prompt: str, user_prompt: str,
                  model: str | None = None, temperature: float | None = None) -> str:
    chat = _llm(model=model, temperature=temperature)
    return chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content.strip()

# ------------------------------------------------------------------ #
#                         Cache utilities                            #
# ------------------------------------------------------------------ #

def _code_hash(code: str) -> str:
    # Raw hash to maximize exact-match reuse; adjust if you want whitespace-insensitive
    h = hashlib.sha256()
    h.update(code.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def _cache_path(task_id: str, code_sha: str) -> Path:
    d = Path(EVAL_CACHE_DIR) / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{code_sha}.json"

def _cache_get(task_id: str, code_sha: str) -> Dict[str, Any] | None:
    p = _cache_path(task_id, code_sha)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # expected shape: {"score": int, "feedback": str, "_meta": {...}}
        if isinstance(data, dict) and "score" in data and "feedback" in data:
            return {"score": int(data["score"]), "feedback": str(data["feedback"])}
    except Exception:
        return None
    return None

def _cache_set(task_id: str, code_sha: str, score: int, feedback: str, meta: Dict[str, Any] | None = None) -> None:
    p = _cache_path(task_id, code_sha)
    payload = {"score": int(score), "feedback": feedback, "_meta": meta or {}}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------------------------------------------------ #
#                        File/code helpers                           #
# ------------------------------------------------------------------ #

def _read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

_SCORE_RE = re.compile(r"Score:\s*(\d{1,3})\b", re.IGNORECASE)

def _extract_score_and_feedback(response: str, max_points: int = 100) -> Tuple[int, str]:
    m = re.search(r"Feedback:\s*(.*?)\s*Score:\s*(\d{1,3})\b", response, re.DOTALL | re.IGNORECASE)
    if m:
        fb = m.group(1).strip()
        sc = int(m.group(2))
        return min(sc, max_points), fb
    m = _SCORE_RE.search(response)
    if m:
        sc = int(m.group(1))
        fb = response.split("Score:")[0].strip()
        if fb.lower().startswith("feedback:"):
            fb = fb[len("feedback:"):].strip()
        return min(sc, max_points), (fb or "No feedback provided.")
    return 0, response.strip()

# ------------------------------------------------------------------ #
#                       Per-task evaluation                          #
# ------------------------------------------------------------------ #

def _eval_single_task(task: Dict[str, Any], student_code: str,
                      model: str, temperature: float) -> Tuple[int, str]:
    """
    One LLM call (fallback path). Returns (score, feedback).
    """
    desc    = task["description"]
    system_msg = (
        "You are a kind and constructive TA grading Python homework.\n"
        "Be encouraging, specific, and fair. Focus on correctness and clarity.\n"
        "Respond in ONE LINE only, exactly in the format:\n"
        "Feedback: <short, friendly sentence> Score: <number>\n"
        "No newlines, no code blocks, and no extra text."
    )
    user_msg = (
        f"Task (grade 0..100):\n{desc}\n\n"
        f"Student Code:\n```python\n{student_code}\n```\n\n"
        "Evaluate correctness and completeness against the task. "
        "Return only: Feedback: ... Score: <0..100>."
    )
    response = llm_call_text(system_msg, user_msg, model=model, temperature=temperature)
    return _extract_score_and_feedback(response, max_points=100)

# ------------------------------------------------------------------ #
#                      Batch evaluation (optional)                   #
# ------------------------------------------------------------------ #

def _eval_batch(tasks_with_code: List[Dict[str, Any]], model: str, temperature: float) -> List[Dict[str, Any]]:
    """
    Single LLM call that grades ALL provided tasks for a student.
    Returns a list of {"task": task_id, "score": int, "feedback": str}.
    """
    print("im hereeeeeeeee")
    chat = ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        max_retries=1,
        # Force JSON response
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    system_msg = SystemMessage(content=(
        "You are a kind and constructive TA grading multiple Python tasks for one student.\n"
        "Return ONLY JSON with this exact schema:\n"
        "{ \"results\": [ {\"task\": \"task1\", \"score\": 0..100, \"feedback\": \"...\"}, ... ] }\n"
        "Be concise but friendly in feedback."
    ))

    # Build a compact but informative user prompt
    blocks = []
    for t in tasks_with_code:
        blocks.append(
            f"### {t['task_id']}\n"
            f"Task:\n{t['description']}\n\n"
            f"Student Code ({t.get('expected_filename','')}):\n```python\n{t['code']}\n```"
        )
    user_msg = HumanMessage(content="\n\n".join(blocks) + "\n\nReturn ONLY JSON as specified.")

    raw = chat.invoke([system_msg, user_msg]).content
    # Best effort parse
    try:
        data = json.loads(raw)
    except Exception:
        # try to extract a json-looking block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("Batch grading: failed to parse JSON response.")
        data = json.loads(m.group(0))

    if not isinstance(data, dict) or "results" not in data or not isinstance(data["results"], list):
        raise ValueError("Batch grading: invalid JSON shape.")

    out: List[Dict[str, Any]] = []
    for item in data["results"]:
        try:
            out.append({
                "task": str(item["task"]),
                "score": int(item["score"]),
                "feedback": str(item.get("feedback", "")).strip(),
            })
        except Exception:
            # skip malformed items
            continue
    return out

# ------------------------------------------------------------------ #
#                  Public API: evaluate one student                  #
# ------------------------------------------------------------------ #

def evaluate_single_student(
    student_id: str,
    submission_dir: str,
    tasks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    enable_batch: bool | None = None,  # override env if desired
) -> Dict[str, Any]:
    """
    Evaluate all tasks for a single student.
    - Uses disk cache per (task_id, code_hash) to deduplicate across students.
    - performs one batch LLM call to grade all tasks at once (AHC_EVAL_BATCH=1).
    - Falls back to per-task calls for any missing/failed items.
    """
    if enable_batch is None:
        enable_batch = BATCH_ENABLED

    # Read all codes upfront and compute hashes
    enriched: List[Dict[str, Any]] = []
    for t in tasks:
        tid     = t["task_id"]
        fname   = t.get("expected_filename") or f"{tid}.py"
        code    = _read_file_safe(Path(submission_dir) / fname)
        code_sha = _code_hash(code) if code else ""
        enriched.append({**t, "code": code, "code_sha": code_sha})

    results: List[Dict[str, Any]] = []
    codes_concat: str = "\n".join(e["code"] for e in enriched if e["code"])

    # First, fill from cache (disk cache + optional legacy cache)
    missing_for_batch: List[Dict[str, Any]] = []
    for e in enriched:
        tid, code, code_sha = e["task_id"], e["code"], e["code_sha"]

        if not code:
            results.append({"task": tid, "score": 0, "feedback": "File missing or unreadable."})
            continue

        # disk cache
        cached = _cache_get(tid, code_sha)
        if cached:
            results.append({"task": tid, "score": cached["score"], "feedback": cached["feedback"]})
            continue

        # optional legacy text cache (pair of desc+code)
        legacy = get_cached_feedback(e["description"], code)
        if legacy:
            sc, fb = _extract_score_and_feedback(legacy, max_points=100)
            _cache_set(tid, code_sha, sc, fb, {"source": "legacy_cache"})
            results.append({"task": tid, "score": sc, "feedback": fb})
            continue

        # not cached → candidate for batch/per-task LLM
        missing_for_batch.append(e)

    # Batch call over all missing items
    if enable_batch and missing_for_batch:
        try:
            batch_output = _eval_batch(missing_for_batch, model=model, temperature=temperature)
            # index by task_id
            by_tid = {it["task"]: it for it in batch_output}
            for e in list(missing_for_batch):
                tid, code_sha = e["task_id"], e["code_sha"]
                if tid in by_tid:
                    sc, fb = by_tid[tid]["score"], by_tid[tid]["feedback"]
                    _cache_set(tid, code_sha, sc, fb, {"source": "batch"})
                    results.append({"task": tid, "score": sc, "feedback": fb})
                    # remove from missing list
                    missing_for_batch.remove(e)
        except Exception:
            # fall back silently to per-task path
            pass

    # Per-task fallback for any remaining tasks
    for e in missing_for_batch:
        tid, code, code_sha = e["task_id"], e["code"], e["code_sha"]
        try:
            sc, fb = _eval_single_task(e, code, model=model, temperature=temperature)
            _cache_set(tid, code_sha, sc, fb, {"source": "single"})
            # also store legacy cache for backward reuse
            try:
                store_feedback(e["description"], code, f"Feedback: {fb} Score: {sc}")
            except Exception:
                pass
        except Exception as ex:
            sc, fb = 0, f"GPT evaluation failed: {ex}"
        results.append({"task": tid, "score": sc, "feedback": fb})

    # Final score: average of per-task 0..100 scores
    scores = [it["score"] for it in results if isinstance(it.get("score"), (int, float))]
    final_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    return {
        "student_id": student_id,
        "results": results,
        "final_score": final_score,   
        "codes_concat": codes_concat,
    }
