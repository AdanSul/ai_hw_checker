def validate_tasks(parsed: dict) -> dict:
    assert isinstance(parsed, dict), "parsed must be dict"
    assert "tasks" in parsed and isinstance(parsed["tasks"], list), "parsed['tasks'] must list"
    for t in parsed["tasks"]:
        for key in ("task_id", "description"):
            assert key in t and isinstance(t[key], str), f"task missing {key}"
        t.setdefault("type", "code")            
        t.setdefault("lang", "python")            
        t.setdefault("points", 10)
        t.setdefault("expected_filename", f"{t['task_id']}.py")
    return parsed

def validate_student_result(sr: dict) -> dict:
    assert "student_id" in sr and "results" in sr
    return sr
