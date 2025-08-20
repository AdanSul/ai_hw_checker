# ----------------------------------------------------
import sys, os
THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in (THIS_DIR, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
# --------------------------------------------------------

import streamlit as st
from pathlib import Path
import tempfile
import traceback

from ahc.pipeline import run_full_pipeline
# NOTE: local imports (no "app." prefix)
from components import (
    build_dataframe,
    extract_zip_to_temp,
    bytes_of_csv_jsonl,
    render_results_table,
    render_summary,
    render_downloads,
    render_student_view,
)
from state import (
    ensure_session_state,
    set_assignment_preview,
    set_results_df,
    get_results,
    get_df,
    set_error,
    get_error,
    get_assignment_preview,
)

# ---------- Streamlit base config ----------
st.set_page_config(page_title="AI Homework Checker", layout="wide")
st.title("AI Homework Checker")

# ---------- Init state ----------
ensure_session_state()

# ---------- Sidebar (inputs) ----------
with st.sidebar:
    st.header("Run settings")
    assignment_file = st.file_uploader("📄 Assignment file (Markdown)", type=["md", "txt"])
    subs_zip       = st.file_uploader("🗂️ Submissions ZIP", type=["zip"])
    model          = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo-0125"])
    temperature    = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    show_feedback  = st.checkbox("Show feedback columns in table", value=True)
    ai_threshold   = st.slider("AI score highlight threshold", 0.0, 1.0, 0.25, 0.01)
    run_btn        = st.button("🚀 Run")

# ---------- Tabs ----------
tab_results, tab_student, tab_logs = st.tabs(["Results", "Student", "Logs/Preview"])

# ---------- Run pipeline on click ----------
if run_btn:
    set_error("")
    if not assignment_file or not subs_zip:
        st.warning("Please upload both an assignment file and a submissions ZIP.", icon="⚠️")
    else:
        try:
            with st.spinner("Parsing → evaluating → aggregating..."):
                tmp_dir = Path(tempfile.mkdtemp(prefix="ahc_run_"))
                a_path = tmp_dir / "assignment.md"
                try:
                    a_text = assignment_file.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    a_text = assignment_file.getvalue().decode("latin-1")
                a_path.write_text(a_text, encoding="utf-8")
                set_assignment_preview(a_text)

                subs_dir = extract_zip_to_temp(subs_zip)

                results = run_full_pipeline(
                    assignment_path=str(a_path),
                    submissions_dir=str(subs_dir),
                    model=model,
                    temperature=temperature,
                )
                df = build_dataframe(results, include_feedback=show_feedback)
                set_results_df(results, df)

            st.success("Run completed successfully ✅", icon="✅")

        except Exception as e:
            set_error(f"{type(e).__name__}: {e}")
            st.error("An error occurred during the run.", icon="❌")
            with tab_logs:
                st.code(traceback.format_exc(), language="text")

# ---------- Results tab ----------
with tab_results:
    df = get_df()
    results = get_results()

    if df.empty:
        st.info("No results to display yet. Upload files and click Run.", icon="ℹ️")
    else:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Results table")
            render_results_table(df, ai_threshold)
        with right:
            st.subheader("Summary")
            render_summary(df)
            if results:
                csv_bytes, jsonl_bytes = bytes_of_csv_jsonl(results, include_feedback=show_feedback)
                render_downloads(csv_bytes, jsonl_bytes)

# ---------- Student tab ----------
with tab_student:
    results = get_results()
    if not results:
        st.info("No data yet. Run the checker, then select a student.", icon="ℹ️")
    else:
        render_student_view(results)

# ---------- Logs tab ----------
with tab_logs:
    st.subheader("Assignment preview (Markdown)")
    preview = get_assignment_preview()
    if preview:
        st.code(preview, language="markdown")
    else:
        st.write("—")

    err = get_error()
    if err:
        st.subheader("Last error")
        st.code(err, language="text")
