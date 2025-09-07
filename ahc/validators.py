from typing import Dict, List, Any

def validate_tasks(parsed: dict) -> dict:
    """
    Validates the structure of the parsed tasks and sets sensible defaults.
    """
    # Assert top-level structure
    assert isinstance(parsed, dict), "parsed must be a dictionary."
    assert "tasks" in parsed and isinstance(parsed["tasks"], list), "parsed['tasks'] must be a list."

    for t in parsed["tasks"]:
        # Assert required keys are present and are strings
        for key in ("task_id", "description"):
            assert key in t and isinstance(t[key], str), f"Task missing required key: {key}."

        # Set defaults for all tasks
        t.setdefault("type", "code")
        t.setdefault("points", 10)
        t.setdefault("examples", "")
        
        # Conditionally set defaults for 'code' tasks only
        if t["type"].lower() == "code":
            t.setdefault("lang", "python")
            t.setdefault("expected_filename", f"{t['task_id']}.py")
        else:
            # For non-code tasks, ensure these keys do not exist
            t.pop("lang", None)
            t.pop("expected_filename", None)

    return parsed

def validate_student_result(sr: dict) -> dict:
    assert "student_id" in sr and "results" in sr
    return sr
