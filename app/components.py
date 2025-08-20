from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path
import zipfile
import tempfile
import re

from ahc.exporters import export_csv, export_jsonl

# ---------- Helpers (pure) ----------

def student_num(s: str):
    """Return only the trailing number after '-' (e.g. 'hw2-2001' -> '2001')."""
    if not isinstance(s, str) or s == "CLASS_AVG":
        return s
    after = s.rsplit('-', 1)[-1]
    m = re.search(r'(\d+)$', after)
    return m.group(1) if m else after

def build_dataframe(results: list[dict], include_feedback: bool) -> pd.DataFrame:
    """Flatten pipeline results -> wide DataFrame (pairs: taskX_score, taskX_feedback)."""
    if not results:
        return pd.DataFrame()
    task_ids = [it["task"] for it in results[0]["results"]]
    rows = []
    for r in results:
        row = {
            "student_id": student_num(r.get("student_id")),
            "final_score": round(float(r.get("final_score", 0.0)), 3),  # already 0..100
            "ai_score": (r.get("ai_suspicion", {}) or {}).get("score"),
        }
        by_task = {it["task"]: it for it in r.get("results", [])}
        for t in task_ids:
            row[f"{t}_score"] = by_task.get(t, {}).get("score")
            if include_feedback:
                row[f"{t}_feedback"] = (by_task.get(t, {}) or {}).get("feedback", "")
        rows.append(row)
    return pd.DataFrame(rows)

def extract_zip_to_temp(uploaded_file) -> Path:
    """Save uploaded ZIP to a temp dir and extract. Returns submissions root Path."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ahc_subs_"))
    zpath = temp_dir / "subs.zip"
    zpath.write_bytes(uploaded_file.getvalue())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(temp_dir / "subs")
    return temp_dir / "subs"

def bytes_of_csv_jsonl(results: list[dict], include_feedback: bool):
    """Export to bytes (no permanent disk writes) for Streamlit download buttons."""
    tmp = Path(tempfile.mkdtemp(prefix="ahc_dl_"))
    csv_p = tmp / "grades.csv"
    jsonl_p = tmp / "grades.jsonl"
    export_csv(results, csv_p, include_feedback=include_feedback, add_class_avg=True)
    export_jsonl(results, jsonl_p)
    return csv_p.read_bytes(), jsonl_p.read_bytes()

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
