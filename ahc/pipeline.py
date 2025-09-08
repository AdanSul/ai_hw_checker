from pathlib import Path
import glob
from ahc.agents.parse_assignment import parse_assignment
from ahc.agents.evaluate_task import evaluate_single_student
from ahc.detectors import ai_copy_score
from ahc.validators import validate_student_result

def run_full_pipeline(assignment_path: str, submissions_dir: str, model: str, temperature: float = 0.1) -> list[dict]:
    parsed = parse_assignment(assignment_path, model=model, temperature=0.0)
    tasks = parsed["tasks"]

    # איסוף סטודנטים
    student_dirs = [p for p in glob.glob(str(Path(submissions_dir) / "*")) if Path(p).is_dir()]
    # הערכה ראשונית + איסוף טקסט קוד לכל סטודנט
    prelim = []
    for sd in student_dirs:
        sid = Path(sd).name
        res = evaluate_single_student(sid, sd, tasks, model=model, temperature=temperature)
        prelim.append(res)

    peer_codes = [r["codes_concat"] for r in prelim]
    
    #---------- TO DO -------------
    baseline_ai = []  # TODO

    results = []
    for i, r in enumerate(prelim):
        ai_sig = ai_copy_score(r["codes_concat"], peer_codes[:i] + peer_codes[i+1:], baseline_ai=baseline_ai)
        out = {k: v for k, v in r.items() if k != "codes_concat"}
        out["ai_suspicion"] = ai_sig
        results.append(validate_student_result(out))

    return results
