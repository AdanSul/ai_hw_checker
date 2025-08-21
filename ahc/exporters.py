import re
import io
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd

_TASK_SCORE_RE = re.compile(r"^task(\d+)_score$")
_TASK_FEED_RE  = re.compile(r"^task(\d+)_feedback$")
_TASK_AI_RE    = re.compile(r"^task(\d+)_ai(?:_score)?$")  # if such cols exist
_BASE_ORDER    = ["student_id", "final_score", "ai_score"]

def _task_sort_key(col: str) -> tuple[int, int, str]:
    m = _TASK_SCORE_RE.match(col)
    if m: return (int(m.group(1)), 0, col)  # score before feedback
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

def build_dataframe(results: List[Dict], include_feedback: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    for r in results:
        row: dict = {}
        sid = str(r.get("student_id", "")).strip()
        m = re.search(r"(\d+)$", sid)          # keep only trailing digits (e.g., hw2-2001 -> 2001)
        row["student_id"] = m.group(1) if m else sid
        row["final_score"] = r.get("final_score", 0)

        for t in r.get("results", []):
            tid_raw = str(t.get("task") or t.get("task_id") or "")
            m = re.search(r"(\d+)", tid_raw)
            tid = f"task{m.group(1)}" if m else (tid_raw or "task1")
            row[f"{tid}_score"] = t.get("score", 0)
            if include_feedback:
                row[f"{tid}_feedback"] = t.get("feedback", "")

        if "ai_score" in r:
            row["ai_score"] = r["ai_score"]

        rows.append(row)

    df = pd.DataFrame(rows)
    df = _add_ai_average_and_drop_per_task(df)
    df = _order_task_columns(df)
    df = df.fillna("")
    return df

def export_csv(results: List[Dict], out_path: Path,
               include_feedback: bool = True,
               add_class_avg: bool = True) -> Path:
    df = build_dataframe(results, include_feedback=include_feedback)

    if add_class_avg and not df.empty:
        avg_row = {"student_id": "CLASS_AVG"}
        # numeric averages
        for c in df.columns:
            if c == "student_id": continue
            if df[c].dtype.kind in "if":  # int/float
                avg_row[c] = round(float(pd.to_numeric(df[c], errors="coerce").mean()), 3)
            else:
                avg_row[c] = ""
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def export_jsonl(results: List[Dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path

def bytes_of_csv_jsonl(results: List[Dict], include_feedback: bool = True) -> tuple[bytes, bytes]:
    df = build_dataframe(results, include_feedback=include_feedback)
    csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    jsonl_buf = io.StringIO()
    for r in results:
        jsonl_buf.write(json.dumps(r, ensure_ascii=False) + "\n")
    jsonl_bytes = jsonl_buf.getvalue().encode("utf-8")

    return csv_bytes, jsonl_bytes
