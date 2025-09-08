from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
import tempfile, io, json, os, uuid
import pandas as pd

from ahc.pipeline import run_full_pipeline
from ahc.reporting import build_dataframe
from ahc.io_utils import extract_zip_to_dir

app = FastAPI(title="AI Homework Checker API")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUN_STORE: dict[str, dict] = {}  

@app.get("/health")
def health():
    return {"ok": True}

def _run_pipeline_flexible(assignment_path: Path, zip_path: Path, model: str, temperature: float):
    try:
        return run_full_pipeline(
            assignment_path=str(assignment_path),
            submissions_zip=str(zip_path),
            model=model,
            temperature=temperature,
        )
    except TypeError:
        subs_root = extract_zip_to_dir(zip_path)
        return run_full_pipeline(
            assignment_path=str(assignment_path),
            submissions_dir=str(subs_root),
            model=model,
            temperature=temperature,
        )

@app.post("/run")
async def run(
    assignment: UploadFile = File(...),
    submissions: UploadFile = File(...),
    model: str = Form("gpt-4o-mini"),
    # temperature: float = Form(0.1),
    show_feedback: bool = Form(True),
    batch_per_student: bool = Form(True),
):
    os.environ["AHC_EVAL_BATCH"] = "1" if str(batch_per_student).lower() in ("1", "true", "yes", "on") else "0"

    # Save uploads to a temp dir
    tmp = Path(tempfile.mkdtemp(prefix="ahc_api_"))
    a_path = tmp / (assignment.filename or "assignment.md")
    a_path.write_bytes(await assignment.read())
    z_path = tmp / (submissions.filename or "submissions.zip")
    z_path.write_bytes(await submissions.read())

    # Run pipeline
    results = _run_pipeline_flexible(a_path, z_path, model=model, temperature=0.1)

    # Build DataFrame from results
    df = build_dataframe(results, include_feedback=bool(show_feedback))

    # Compute summary stats
    class_avg = 0.0
    ai_avg = 0.0
    if "final_score" in df.columns:
        class_avg = float(pd.to_numeric(df["final_score"], errors="coerce").mean())
    if "ai_score" in df.columns:
        ai_avg = float(pd.to_numeric(df.get("ai_score", 0), errors="coerce").fillna(0).mean())

    summary = {
        "class_avg": class_avg,
        "ai_avg": ai_avg,
    }

    # ---------- CSV ----------
    summary_row = {col: "" for col in df.columns}
    label_col = next((c for c in df.columns if c.lower() in ("student_id", "submission_id", "name", "id", "student", "submission")), df.columns[0])
    summary_row[label_col] = "CLASS_AVG"
    if "final_score" in df.columns:
        summary_row["final_score"] = summary["class_avg"]
    if "ai_score" in df.columns:
        summary_row["ai_score"] = summary["ai_avg"]

    df_with_summary = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    csv_buf = io.StringIO()
    df_with_summary.to_csv(csv_buf, index=False)
    # -----------------------------------------

    # ---------- JSONL ----------
    jsonl_buf = io.StringIO()
    for r in results:
        jsonl_buf.write(json.dumps(r, ensure_ascii=False) + "\n")
    jsonl_buf.write(json.dumps({"CLASS_AVG": summary}, ensure_ascii=False) + "\n")
    # ---------------------------------------------

    # Store in memory under a fresh run_id
    run_id = uuid.uuid4().hex
    RUN_STORE[run_id] = {
        "results": results,
        "csv": csv_buf.getvalue(),
        "jsonl": jsonl_buf.getvalue(),
    }

    # Respond with links that embed this run_id
    return JSONResponse({
        "run_id": run_id,
        "results": results,
        "summary": summary,
        "download": {
            "csv": f"/download/{run_id}/csv",
            "jsonl": f"/download/{run_id}/jsonl",
        }
    })


@app.get("/download/{run_id}/csv")
def download_csv(run_id: str):
    data = RUN_STORE.get(run_id)
    if not data:
        raise HTTPException(status_code=404, detail="run_id not found")
    buf = io.BytesIO(data["csv"].encode("utf-8"))
    return StreamingResponse(buf, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="grades_{run_id}.csv"'
    })

@app.get("/download/{run_id}/jsonl")
def download_jsonl(run_id: str):
    data = RUN_STORE.get(run_id)
    if not data:
        raise HTTPException(status_code=404, detail="run_id not found")
    buf = io.BytesIO(data["jsonl"].encode("utf-8"))
    return StreamingResponse(buf, media_type="application/json", headers={
        "Content-Disposition": f'attachment; filename="grades_{run_id}.jsonl"'
    })
