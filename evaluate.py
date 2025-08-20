from ahc.config import load_config
from ahc.pipeline import run_full_pipeline
from ahc.exporters import export_csv, export_jsonl
from pathlib import Path

def main():
    print(">> Loading config...", flush=True)
    cfg = load_config("config.json")

    print(">> Running pipeline (parse → evaluate → export)...", flush=True)
    results = run_full_pipeline(
        assignment_path=cfg["assignment_path"],
        submissions_dir=cfg["submissions_dir"],
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.1),
    )

    out_dir = Path(cfg["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">> Exporting to {out_dir} ...", flush=True)
    export_csv(results, out_dir / "grades.csv", include_feedback=True, add_class_avg=True)
    export_jsonl(results, out_dir / "grades.jsonl")

    try:
        from statistics import mean
        totals = []
        for r in results:
            total_max = sum(it.get("max_points", 100) for it in r.get("results", [])) or 1
            totals.append((r.get("final_score", 0.0) / total_max) * 100.0)
        class_avg = round(mean(totals), 3) if totals else 0.0
        print(f">> Class average (normalized to 100): {class_avg}", flush=True)
    except Exception:
        pass

    print(f">> Done. Wrote CSV+JSONL under: {out_dir.resolve()}", flush=True)

if __name__ == "__main__":
    main()
