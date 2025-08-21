from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path
import zipfile
import tempfile
import re

from ahc.exporters import export_csv, export_jsonl

import re
import json, io
import pandas as pd

_TASK_SCORE_RE   = re.compile(r"^task(\d+)_score$")
_TASK_FEED_RE    = re.compile(r"^task(\d+)_feedback$")
_TASK_AI_RE      = re.compile(r"^task(\d+)_ai(?:_score)?$") 
_BASE_ORDER      = ["student_id", "final_score", "ai_score"]  # will keep this order first

def _task_sort_key(col: str) -> tuple[int, int, str]:
    """
    Sort key that puts task1_score, task1_feedback, task2_score, task2_feedback, ...
    Non-task columns go to the end but we also keep BASE_ORDER first.
    """
    m = _TASK_SCORE_RE.match(col)
    if m:
        return (int(m.group(1)), 0, col)  # score before feedback
    m = _TASK_FEED_RE.match(col)
    if m:
        return (int(m.group(1)), 1, col)
    return (10**9, 99, col)  # everything else at the end

def _order_task_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)

    # ensure base columns appear first in the declared order (if present)
    base = [c for c in _BASE_ORDER if c in cols]

    # everything else (except base) sorted by task number, score then feedback
    rest = [c for c in cols if c not in base]
    rest_sorted = sorted(rest, key=_task_sort_key)

    ordered = base + rest_sorted
    return df.reindex(columns=ordered)

def _add_ai_average_and_drop_per_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    If there are per-task AI columns like task1_ai or task1_ai_score, compute row-wise mean,
    store it in 'ai_score', and drop the per-task AI columns.
    If 'ai_score' already exists and no per-task AI columns exist, we keep it as is.
    """
    ai_cols = [c for c in df.columns if _TASK_AI_RE.match(c)]
    if ai_cols:
        df["ai_score"] = df[ai_cols].mean(axis=1).round(3)
        df = df.drop(columns=ai_cols)
    return df

# ---------- Helpers (pure) ----------

def student_num(s: str):
    """Return only the trailing number after '-' (e.g. 'hw2-2001' -> '2001')."""
    if not isinstance(s, str) or s == "CLASS_AVG":
        return s
    after = s.rsplit('-', 1)[-1]
    m = re.search(r'(\d+)$', after)
    return m.group(1) if m else after

def build_dataframe(results: list[dict], include_feedback: bool = True) -> pd.DataFrame:
    """
    Builds a flat table for UI/CSV.
    - student_id, final_score (0..100)
    - optional ai_score (averaged if per-task AI cols exist)
    - taskN_score and (optionally) taskN_feedback, ordered task1..taskN
    """
    rows: list[dict] = []
    for r in results:
        row: dict = {}
        # student id -> keep only trailing digits if it looks like "hw2-2001"
        sid = str(r.get("student_id", "")).strip()
        m = re.search(r"(\d+)$", sid)
        row["student_id"] = m.group(1) if m else sid

        # final score already normalized 0..100 in the pipeline
        row["final_score"] = r.get("final_score", 0)

        # per-task fields
        for t in r.get("results", []):
            tid_raw = str(t.get("task") or t.get("task_id") or "")
            m = re.search(r"(\d+)", tid_raw)
            tid = f"task{m.group(1)}" if m else tid_raw or "task1"

            row[f"{tid}_score"] = t.get("score", 0)
            if include_feedback:
                row[f"{tid}_feedback"] = t.get("feedback", "")

            # If your detector adds per-task AI columns, you can set them here, e.g.:
            # if "ai" in t: row[f"{tid}_ai"] = t["ai"]

        # if the pipeline already computed a single ai_score, copy it in
        if "ai_score" in r:
            row["ai_score"] = r["ai_score"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # If there are any per-task AI columns, average them into 'ai_score' and drop the per-task cols
    df = _add_ai_average_and_drop_per_task(df)

    # Order columns: student_id, final_score, ai_score, then task1_score, task1_feedback, task2_...
    df = _order_task_columns(df)

    # Fill NaNs (e.g., missing feedback) so CSVs are clean
    df = df.fillna("")
    return df


def extract_zip_to_temp(uploaded_file) -> Path:
    """Save uploaded ZIP to a temp dir and extract. Returns submissions root Path."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ahc_subs_"))
    zpath = temp_dir / "subs.zip"
    zpath.write_bytes(uploaded_file.getvalue())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(temp_dir / "subs")
    return temp_dir / "subs"

def bytes_of_csv_jsonl(results: list[dict], include_feedback: bool = True) -> tuple[bytes, bytes]:
    df = build_dataframe(results, include_feedback=include_feedback)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    jsonl_buf = io.StringIO()
    for r in results:
        jsonl_buf.write(json.dumps(r, ensure_ascii=False) + "\n")
    jsonl_bytes = jsonl_buf.getvalue().encode("utf-8")

    return csv_bytes, jsonl_bytes


# ---------- UI pieces ----------

def render_results_table(df: pd.DataFrame, ai_threshold: float = 0.25):
    """Styled dataframe: highlight rows by ai_score >= threshold."""
    def _row_style(row):
        score = row.get("ai_score", 0) or 0.0
        return [
            "background-color: #fff0f0" if (isinstance(score, (int, float)) and score >= ai_threshold) else ""
            for _ in row
        ]
    st.dataframe(df.style.apply(_row_style, axis=1), use_container_width=True)

def render_summary(df: pd.DataFrame):
    """Class averages and quick stats."""
    try:
        class_avg = round(float(df["final_score"].mean()), 2)
        ai_avg    = round(float((df["ai_score"].fillna(0)).mean()), 3)
        st.metric("Class average (0..100)", class_avg)
        st.metric("Mean AI score", ai_avg)
    except Exception:
        st.write("—")

def render_downloads(csv_bytes: bytes, jsonl_bytes: bytes):
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="grades.csv", mime="text/csv", use_container_width=True)
    st.download_button("⬇️ Download JSONL", data=jsonl_bytes, file_name="grades.jsonl", mime="application/json", use_container_width=True)

def _copy_button(text: str, key: str):
    """Small HTML/JS snippet to copy text to clipboard (works in Streamlit)."""
    display = text if len(text) < 400 else (text[:400] + " …")
    html = f"""
    <div style="border:1px solid #eee;padding:8px;border-radius:8px;">
      <div style="white-space:pre-wrap;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;font-size:0.9rem;">
        {display.replace('<','&lt;').replace('>','&gt;')}
      </div>
      <button onclick="navigator.clipboard.writeText(`{text.replace('`','\\`')}`)" style="margin-top:6px;">Copy</button>
    </div>
    """
    st.components.v1.html(html, height=120)

def render_student_view(results: list[dict]):
    """Student selector + per-task expanders with feedback + copy button."""
    student_ids = [r["student_id"] for r in results]
    sid = st.selectbox("Select student", options=student_ids, index=0)
    rec = next(x for x in results if x["student_id"] == sid)

    st.markdown(f"**Final score:** {round(float(rec.get('final_score', 0.0)), 2)} / 100")
    ai = (rec.get("ai_suspicion", {}) or {}).get("score", 0.0)
    reasons = ", ".join((rec.get("ai_suspicion", {}) or {}).get("reasons", [])) or "—"
    st.markdown(f"**AI score:** {ai:.3f}  \n*Reasons:* {reasons}")

    for it in rec.get("results", []):
        with st.expander(f"{it['task']} — Score: {it.get('score', 0)}"):
            st.markdown("**Feedback:**")
            _copy_button(it.get("feedback", ""), key=f"copy_{sid}_{it['task']}")
