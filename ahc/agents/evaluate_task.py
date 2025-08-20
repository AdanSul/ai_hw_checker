import os
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from ahc.cache_utils import get_cached_feedback, store_feedback

load_dotenv()

DEFAULT_MODEL = os.getenv("AHC_EVAL_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("AHC_EVAL_TEMPERATURE", "0.0"))

def _llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        max_retries=1,
    )

def llm_call_text(system_prompt: str, user_prompt: str, model: str | None = None, temperature: float | None = None) -> str:
    chat = _llm(model=model, temperature=temperature)
    return chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content.strip()

def _read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

_SCORE_RE = re.compile(r"Score:\s*(\d{1,3})\b")

def _extract_score_and_feedback(response: str, max_points: int = 100) -> tuple[int, str]:
    m = re.search(r"Feedback:\s*(.*?)\s*Score:\s*(\d{1,3})\b", response, re.DOTALL)
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

def evaluate_single_student(
    student_id: str,
    submission_dir: str,
    tasks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    codes_concat: str = ""

    for t in tasks:
        task_id = t["task_id"]
        desc    = t["description"]
        # Always grade per-task on a 0..100 scale
        points_for_prompt = 100
        fname   = t.get("expected_filename") or f"{task_id}.py"

        code_path = Path(submission_dir) / fname
        student_code = _read_file_safe(code_path)
        codes_concat += (student_code + "\n")

        # cache
        cached = get_cached_feedback(desc, student_code)
        if cached:
            score, feedback = _extract_score_and_feedback(cached, max_points=points_for_prompt)
            results.append({"task": task_id, "score": score, "feedback": feedback})
            continue

        # kinder, supportive tone; still single-line, machine-parsable
        system_msg = (
            "You are a kind and constructive TA grading Python homework.\n"
            "Be encouraging, specific, and fair. Focus on correctness and clarity.\n"
            "Respond in ONE LINE only, exactly in the format:\n"
            "Feedback: <short, friendly sentence> Score: <number>\n"
            "No newlines, no code blocks, no extra text."
        )
        user_msg = (
            f"Task (grade 0..{points_for_prompt}):\n{desc}\n\n"
            f"Student Code:\n```python\n{student_code}\n```\n\n"
            "Evaluate correctness and completeness against the task. "
            "Be supportive and concise. Return only: Feedback: ... Score: <0..100>."
        )

        try:
            response = llm_call_text(system_msg, user_msg, model=model, temperature=temperature)
            store_feedback(desc, student_code, response)
        except Exception as e:
            results.append({"task": task_id, "score": 0, "feedback": f"GPT evaluation failed: {e}"})
            continue

        score, feedback = _extract_score_and_feedback(response, max_points=points_for_prompt)
        results.append({"task": task_id, "score": score, "feedback": feedback})

    # final score = average of per-task 0..100 scores
    scores = [item["score"] for item in results if item.get("score") is not None]
    final_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    return {
        "student_id": student_id,
        "results": results,
        "final_score": final_score,   # 0..100 average
        "codes_concat": codes_concat,
    }
