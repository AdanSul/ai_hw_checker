import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

DEFAULT_MODEL = os.getenv("AHC_EVAL_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("AHC_EVAL_TEMPERATURE", "0.0"))
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
    desc = task.get("description", "")
    examples = task.get("examples", "")
    task_type = task.get("type", "code")
    task_lang = task.get("lang", "python")

    # Dynamically build system message based on task type
    if task_type == "code":
        system_intro = f"You are a fair, supportive TA grading a SINGLE {task_lang} programming task."
    elif task_type == "mathematics":
        system_intro = "You are a fair, supportive TA grading a SINGLE mathematics task, such as a proof or problem solution."
    else:
        system_intro = "You are a fair, supportive TA grading a SINGLE written response or short answer task."
    
    system_msg = (
        f"{system_intro}\n"
        "Grade correctness and completeness against the task instructions and examples (if any).\n"
        "Ignore style/formatting/naming unless they break correctness. Small cosmetic issues do not cause point deductions.\n"
        "If the code is empty/missing/irrelevant, it can receive 0–49.\n"
        "Be concise, kind, and actionable.\n"
        "\n"
        "OUTPUT FORMAT (must be exactly one line, no newlines, no code blocks):\n"
        "Feedback: <friendly, specific sentence, <= 25 words> Score: <integer 0..100>\n"
        "Never include extra text. Never say 'As an AI'."
    )

    # Dynamically format the student's submission based on task type
    if task_type == "code":
        submission_block = f"Student Code:\n```{task_lang}\n{student_code}\n```"
    else:
        submission_block = f"Student Submission:\n{student_code}"

    user_msg = (
        "Task (grade on a 0..100 scale):\n"
        f"{desc}\n\n"
        + (f"Examples (optional hints/tests):\n{examples}\n\n" if examples else "")
        + f"{submission_block}\n\n"
        "Return exactly: Feedback: ... Score: <0..100>. No newlines, no extra words."
    )
    
    response = llm_call_text(system_msg, user_msg, model=model, temperature=temperature)
    return _extract_score_and_feedback(response, max_points=100)

# ------------------------------------------------------------------ #
#                      Batch evaluation                              #
# ------------------------------------------------------------------ #

def _eval_batch(tasks_with_code: List[Dict[str, Any]], model: str, temperature: float) -> List[Dict[str, Any]]:
    """
    Single LLM call that grades ALL provided tasks for one student.
    """
    chat = ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        max_retries=1,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    system_msg = SystemMessage(content=(
        "You are a fair, supportive TA grading MULTIPLE tasks for a single student.\n"
        "Judge by reasoning only. Do not execute any provided mathematical statements.\n"
        "For EACH task, grade correctness and completeness against its instructions and examples (if any).\n"
        "Ignore style/formatting/naming unless they break correctness. Minor cosmetics do not reduce the score.\n"
        "If a task's submission is empty/missing/irrelevant, you may assign 0–49.\n"
        "Keep feedback short, kind, and actionable.\n"
        "\n"
        "STRICT OUTPUT: return ONLY a JSON object with this exact shape:\n"
        "{\n"
        '  "results": [\n'
        '    {"task": "task1", "score": 0, "feedback": "specific sentence (<= 25 words)"},\n'
        '    {"task": "task2", "score": 0, "feedback": "specific sentence (<= 25 words)"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Include exactly one object per provided task, preserving the given task IDs.\n"
        "- score is an integer 0..100.\n"
        "- feedback is a single specific sentence (<= 25 words). No line breaks, no code blocks.\n"
        "- Do NOT include any keys other than 'results'.\n"
        "- Never say 'As an AI'."
    ))
    blocks = []
    for t in tasks_with_code:
        tid = t.get("task_id", "")
        desc = t.get("description", "")
        examples = t.get("examples", "")
        task_type = t.get("type", "code")
        task_lang = t.get("lang", "python")
        student_code = t.get("code", "")
        
        if task_type == "code":
            submission_block = f"Student Code ({task_lang}):\n```{task_lang}\n{student_code}\n```"
        else:
            submission_block = f"Student Submission ({task_type}):\n{student_code}"

        block = [
            f"### Task: {tid} ({task_type})",
            f"Instructions:\n{desc}",
        ]
        if examples:
            block.append(f"Examples:\n{examples}")
        
        block.append(submission_block)
        blocks.append("\n".join(block))

    user_msg = HumanMessage(content="\n\n".join(blocks) + "\n\nReturn ONLY the JSON object as specified.")

    raw = chat.invoke([system_msg, user_msg]).content

    try:
        data = json.loads(raw)
    except Exception:
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
    - Optionally performs one batch LLM call to grade all tasks at once (AHC_EVAL_BATCH=1).
    - Falls back to per-task calls for any items missing after batch.
    """
    if enable_batch is None:
        enable_batch = BATCH_ENABLED

    # Enrich each task dict to also 
    # hold task information, with an additional key: "code",
    # which stores the content of the student's submitted file for that task.
    enriched: List[Dict[str, Any]] = []
    for t in tasks:
        tid     = t["task_id"]
        fname   = t.get("expected_filename") or f"{tid}.py"
        code    = _read_file_safe(Path(submission_dir) / fname)
        enriched.append({**t, "code": code})


    results: List[Dict[str, Any]] = []
    # This string is a compilation of all the student's code from all tasks, joined by newlines.
    codes_concat: str = "\n".join(e["code"] for e in enriched if e["code"])

    # Prepare list for evaluation
    pending: List[Dict[str, Any]] = []
    for e in enriched:
        if not e["code"]:
            results.append({"task": e["task_id"], "score": 0, "feedback": "File missing."})
        else:
            pending.append(e)

    # Batch call
    if enable_batch and pending:
        try:
            batch_output = _eval_batch(pending, model=model, temperature=temperature)
            by_tid = {it["task"]: it for it in batch_output}
            # consume whatever returned in batch
            done_tids = set()
            for e in list(pending):
                tid = e["task_id"]
                if tid in by_tid:
                    sc, fb = by_tid[tid]["score"], by_tid[tid]["feedback"]
                    results.append({"task": tid, "score": sc, "feedback": fb})
                    done_tids.add(tid)
            # keep only those not returned by batch
            pending = [e for e in pending if e["task_id"] not in done_tids]
        except Exception:
            # fall back silently to per-task path
            pass

    # Per-task for any leftovers
    for e in pending:
        try:
            sc, fb = _eval_single_task(e, e["code"], model=model, temperature=temperature)
        except Exception as ex:
            sc, fb = 0, f"GPT evaluation failed: {ex}"
        results.append({"task": e["task_id"], "score": sc, "feedback": fb})

    # Final score: average of per-task 0..100 scores
    scores = [it["score"] for it in results if isinstance(it.get("score"), (int, float))]
    final_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    return {
        "student_id": student_id,
        "results": results,
        "final_score": final_score,   # 0..100 average
        "codes_concat": codes_concat,
    }
