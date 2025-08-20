# ------------------------------------
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------------


import streamlit as st
from ahc.pipeline import run_full_pipeline
from ahc.exporters import export_csv, export_jsonl
import zipfile, tempfile, os
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="AI Homework Checker", layout="wide")
st.title("AI Homework Checker")

with st.sidebar:
    assignment = st.file_uploader("📄 הוראות תרגיל (MD)", type=["md"])
    subs_zip  = st.file_uploader("🗂️ הגשות (ZIP של תיקיות)", type=["zip"])
    model = st.selectbox("מודל", ["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    go = st.button("🚀 הרצה")

tabs = st.tabs(["תוצאות", "פירוט סטודנט"])

if go and assignment and subs_zip:
    with tempfile.TemporaryDirectory() as td:
        a_path = Path(td) / "assignment.md"
        a_path.write_text(assignment.getvalue().decode("utf-8"), encoding="utf-8")
        subs_dir = Path(td) / "subs"
        subs_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(subs_zip) as z:
            z.extractall(subs_dir)

        results = run_full_pipeline(str(a_path), str(subs_dir), model, temperature)

        # הורדות
        out_dir = Path(td)
        csv_p = out_dir / "grades.csv"
        jsonl_p = out_dir / "grades.jsonl"
        export_csv(results, csv_p)
        export_jsonl(results, jsonl_p)

        st.download_button("⬇️ הורדת CSV", data=csv_p.read_bytes(), file_name="grades.csv", mime="text/csv")
        st.download_button("⬇️ הורדת JSONL", data=jsonl_p.read_bytes(), file_name="grades.jsonl", mime="application/json")

        rows=[]
        for r in results:
            row={"student_id": r["student_id"], "final_score": r["final_score"], "ai_score": r["ai_suspicion"]["score"]}
            for it in r["results"]:
                row[f"{it['task']}_score"] = it["score"]
            rows.append(row)
        df = pd.DataFrame(rows)
        with tabs[0]:
            st.dataframe(df, use_container_width=True)
        with tabs[1]:
            sid = st.selectbox("בחרי סטודנט", [r["student_id"] for r in results])
            rec = next(x for x in results if x["student_id"]==sid)
            st.markdown(f"**ציון סופי:** {rec['final_score']}")
            st.markdown(f"**AI חשד:** {rec['ai_suspicion']['score']}  \n*סיבות:* {', '.join(rec['ai_suspicion']['reasons']) or '—'}")
            for it in rec["results"]:
                st.markdown(f"### {it['task']}")
                st.markdown(f"**ציון:** {it['score']}")
                st.markdown("**פידבק:**")
                st.code(it.get("feedback",""), language="markdown")
