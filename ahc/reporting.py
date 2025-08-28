from __future__ import annotations
import re
import pandas as pd

_TASK_SCORE_RE   = re.compile(r"^task(\d+)_score$")
_TASK_FEED_RE    = re.compile(r"^task(\d+)_feedback$")
_TASK_AI_RE      = re.compile(r"^task(\d+)_ai(?:_score)?$")
_BASE_ORDER      = ["student_id", "final_score", "ai_score"]

def _task_sort_key(col: str) -> tuple[int, int, str]:
    m = _TASK_SCORE_RE.match(col)
    if m: return (int(m.group(1)), 0, col)
    m = _TASK_FEED_RE.match(col)
    if m: return (int(m.group(1)), 1, col)
    return (10**9, 99, col)

def _order_task_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    base = [c for c in _BASE_ORDER if c in cols]
    rest = [c for c in cols if c not in base]
    rest_sorted = sorted(rest, key=_task_sort_key)
    return df.reindex(columns=base + rest_sorted)

def _add_ai_average_and_drop_per_task(df: pd.DataFrame) -> pd.DataFrame:
    ai_cols = [c for c in df.columns if _TASK_AI_RE.match(c)]
    if ai_cols:
        df["ai_score"] = df[ai_cols].mean(axis=1).round(3)
        df = df.drop(columns=ai_cols)
    return df

def build_dataframe(results: list[dict], include_feedback: bool = True) -> pd.DataFrame:
    """
    Build flat table:
      - columns: student_id, final_score, ai_score (if exists),
                 then task1_score, [task1_feedback], task2_score, ...
    """
    rows: list[dict] = []
    for r in results:
        row: dict = {}
        sid = str(r.get("student_id", "")).strip()
        m = re.search(r"(\d+)$", sid)  # keep trailing digits if "hw2-2001"
        row["student_id"] = m.group(1) if m else sid
        row["final_score"] = r.get("final_score", 0)

        for t in r.get("results", []):
            tid_raw = str(t.get("task") or t.get("task_id") or "")
            m = re.search(r"(\d+)", tid_raw)
            tid = f"task{m.group(1)}" if m else tid_raw or "task1"

            row[f"{tid}_score"] = t.get("score", 0)
            if include_feedback:
                row[f"{tid}_feedback"] = t.get("feedback", "")

            # if "ai" in t: row[f"{tid}_ai"] = t["ai"]

        if "ai_score" in r:
            row["ai_score"] = r["ai_score"]

        rows.append(row)

    df = pd.DataFrame(rows)
    df = _add_ai_average_and_drop_per_task(df)
    df = _order_task_columns(df)
    df = df.fillna("")
    return df
