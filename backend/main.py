from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
import tempfile, io, json, os, uuid
import pandas as pd

# project imports
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

RUN_STORE: dict[str, dict] = {}  # run_id -> {"results":..., "csv": str, "jsonl": str}

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
    temperature: float = Form(0.1),
    show_feedback: bool = Form(True),
    batch_per_student: bool = Form(False),
):
    os.environ["AHC_EVAL_BATCH"] = "1" if str(batch_per_student).lower() in ("1","true","yes","on") else "0"

    tmp = Path(tempfile.mkdtemp(prefix="ahc_api_"))
    a_path = tmp / (assignment.filename or "assignment.md")
    a_path.write_bytes(await assignment.read())
    z_path = tmp / (submissions.filename or "submissions.zip")
    z_path.write_bytes(await submissions.read())

    results = _run_pipeline_flexible(a_path, z_path, model=model, temperature=float(temperature))
    df = build_dataframe(results, include_feedback=bool(show_feedback))

    csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
    jsonl_buf = io.StringIO(); [jsonl_buf.write(json.dumps(r, ensure_ascii=False) + "\n") for r in results]

    run_id = uuid.uuid4().hex
    RUN_STORE[run_id] = {
        "results": results,
        "csv": csv_buf.getvalue(),
        "jsonl": jsonl_buf.getvalue(),
    }

    summary = {
        "class_avg": float(pd.to_numeric(df["final_score"], errors="coerce").mean()) if "final_score" in df.columns else 0.0,
        "ai_avg": float(pd.to_numeric(df.get("ai_score", 0), errors="coerce").fillna(0).mean()) if "ai_score" in df.columns else 0.0,
    }

    return JSONResponse({
        "run_id": run_id,
        "results": results,     # עדיין מחזירים כדי לבנות טבלה מיד
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
