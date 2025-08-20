import csv, json, re
from statistics import mean

def _student_num(s):
    if not isinstance(s, str) or s == "CLASS_AVG":
        return s
    after = s.rsplit('-', 1)[-1]
    m = re.search(r'(\d+)$', after)
    return m.group(1) if m else after

def export_csv(
    results: list[dict],
    path,
    include_feedback: bool = True,
    add_class_avg: bool = True,
):
    """
    Wide CSV with per-task pairs: taskX_score, taskX_feedback.
    final_score is already 0..100 (average of tasks) as computed by evaluate_task.py.
    """
    if not results:
        open(path, "w", encoding="utf-8").close()
        return

    # task order from first record
    task_ids = [it["task"] for it in results[0]["results"]]

    # headers
    fieldnames = ["student_id", "final_score", "ai_score"]
    for t in task_ids:
        fieldnames.append(f"{t}_score")
        if include_feedback:
            fieldnames.append(f"{t}_feedback")

    # rows
    rows = []
    for r in results:
        row = {
            "student_id": _student_num(r.get("student_id")),
            "final_score": round(float(r.get("final_score", 0.0)), 3),  # already 0..100
            "ai_score": (r.get("ai_suspicion", {}) or {}).get("score"),
        }
        by_task = {it["task"]: it for it in r.get("results", [])}
        for t in task_ids:
            row[f"{t}_score"] = by_task.get(t, {}).get("score")
            if include_feedback:
                row[f"{t}_feedback"] = (by_task.get(t, {}) or {}).get("feedback", "")
        rows.append(row)

    # write
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

        if add_class_avg and rows:
            avg_row = {k: "" for k in fieldnames}
            avg_row["student_id"] = "CLASS_AVG"
            avg_row["final_score"] = round(mean(r["final_score"] for r in rows), 3)
            avg_row["ai_score"] = round(mean((r["ai_score"] or 0.0) for r in rows), 3)

            # per-task averages (0..100)
            for t in task_ids:
                vals = [r.get(f"{t}_score") for r in rows if r.get(f"{t}_score") is not None]
                if vals:
                    avg_row[f"{t}_score"] = round(mean(vals), 3)
            w.writerow(avg_row)

def export_jsonl(results: list[dict], path):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
