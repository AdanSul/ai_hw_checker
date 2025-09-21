import time
from ahc.config import load_config
from ahc.pipeline import run_full_pipeline
from ahc.exporters import export_csv, export_jsonl
from pathlib import Path

def main():
    print(">> Loading config...", flush=True)
    cfg = load_config("config.json")

    start_time = time.time() 
    print(">> Running pipeline (parse → evaluate → export)...", flush=True)
    results = run_full_pipeline(
        assignment_path=cfg["assignment_path"],
        submissions_dir=cfg["submissions_dir"],
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.1),
    )
    elapsed = time.time() - start_time  
    print(f">> Got {len(results)} results back from pipeline in {elapsed:.2f} seconds.", flush=True)

    out_dir = Path(cfg["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">> Exporting to {out_dir} ...", flush=True)
    export_csv(results, out_dir / "grades1.csv", include_feedback=True, add_class_avg=True)
    export_jsonl(results, out_dir / "grades1.jsonl")

    print(f">> Done. Wrote CSV+JSONL under: {out_dir.resolve()}", flush=True)

if __name__ == "__main__":
    main()
