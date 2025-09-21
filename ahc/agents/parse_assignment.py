import os, re, json, time, hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
from json import JSONDecodeError
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from ahc.validators import validate_tasks
import pypdf
import docx

load_dotenv()

DEFAULT_MODEL = os.getenv("AHC_PARSE_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("AHC_PARSE_TEMPERATURE", "0.0"))

# store cached parses
CACHE_DIR_DEFAULT = os.getenv("AHC_PARSE_CACHE_DIR", "outputs/cache")
SCHEMA_VERSION = "tasks_v1" 

_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BRACES     = re.compile(r"(\{.*\})", re.DOTALL)


def _read_file_as_text(file_path: str) -> str:
    """Reads a file of various types and returns its text content."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if not path.is_file():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if extension == ".pdf":
        print("Reading PDF file:", file_path)
        try:
            reader = pypdf.PdfReader(path)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() or ""
            return text_content
        except Exception as e:
            raise IOError(f"Failed to read PDF file {file_path}: {e}")

    elif extension == ".docx":
        print("Reading DOCX file:", file_path)
        try:
            doc = docx.Document(path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise IOError(f"Failed to read DOCX file {file_path}: {e}")

    # Default for text-based files (.md, .txt, etc.)
    else:
        # print("Reading text file:", file_path)
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            raise IOError(f"Failed to read text file {file_path}: {e}")

def _extract_json_block(text: str) -> str | None:
    m = _JSON_FENCE.search(text)
    if m: return m.group(1).strip()
    m = _BRACES.search(text)
    return m.group(1).strip() if m else None

# ---- debug dump + sanitizer ----
def _dump_debug(name: str, text: str):
    logdir = Path("outputs/logs"); logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"{name}_{int(time.time())}.txt").write_text(text, encoding="utf-8")

# escape any backslash that isn't followed by a valid JSON escape char
_BAD_BSLASH = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')
def _sanitize_json_text(s: str) -> str:
    # 1) fix bad backslashes like C:\Users -> C:\\Users
    s = _BAD_BSLASH.sub(r"\\\\", s)
    # 2) normalize smart quotes if present
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return s

def llm_call_json(system_prompt: str, user_prompt: str,
                  model: str | None = None, temperature: float | None = None) -> Dict[str, Any]:
    """Call OpenAI in JSON mode."""
    chat = ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        max_retries=1,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    resp = chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content

    block = _extract_json_block(resp) or resp
    try:
        return json.loads(block)
    except JSONDecodeError:
        # Try sanitized version
        fixed = _sanitize_json_text(block)
        try:
            return json.loads(fixed)
        except Exception as e2:
            # dump for debugging and re-raise
            _dump_debug("parse_raw", resp)
            _dump_debug("parse_block", block)
            _dump_debug("parse_fixed", fixed)
            raise ValueError(f"Failed to parse JSON from model output even after sanitize: {e2}")

def _to_int(v, default=10):
    try:
        return int(v)
    except Exception:
        return default

def _normalize_task(t: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonicalize fields, coerce types, and set sensible defaults.
    """
    # Standardize the task ID
    if "task_id" not in t:
        if "task_number" in t:
            t["task_id"] = t["task_number"]
        elif "id" in t:
            t["task_id"] = t["id"]
        else:
            t["task_id"] = "task1"
    
    tid = str(t["task_id"]).strip()
    if re.fullmatch(r"\d+", tid):
        tid = f"task{tid}"
    t["task_id"] = tid

    # Set default values for description and examples
    t.setdefault("description", "")
    t.setdefault("examples", "")
    
    # Standardize the task type
    t.setdefault("type", "code")
    
    # Ensure type is a string and handle case variations
    t["type"] = str(t["type"]).strip().lower()

    # Handle language and filename based on task type
    if t["type"] == "code":
        t.setdefault("lang", "python")
        fname = t.get("expected_filename")
        if not isinstance(fname, str) or not fname.strip():
            t["expected_filename"] = f"{t['task_id']}.py"
        else:
            t["expected_filename"] = fname.strip()
    else:
        # Remove keys not relevant for non-code tasks
        t.pop("expected_filename", None)
        t.pop("lang", None)

    # Coerce points to an integer with a default of 10
    t["points"] = _to_int(t.get("points", 10), default=10)
    
    # Strip leading/trailing whitespace from string fields
    for k in ("task_id", "description"):
        if isinstance(t.get(k), str):
            t[k] = t[k].strip() 
    return t

def _merge_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    variants: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for parsed in trials:
        for t in parsed.get("tasks", []):
            try:
                # Normalize each task first
                t = _normalize_task(dict(t))
                if "task_id" in t and "description" in t:
                    variants[t["task_id"]].append(t)
            except Exception:
                continue

    merged: list[Dict[str, Any]] = []
    for tid, items in variants.items():
        if not items:
            continue

        # Find the most common variant for description, type, and points
        desc_counts = Counter(it["description"] for it in items)
        best_desc = desc_counts.most_common(1)[0][0]

        type_counts = Counter(it["type"] for it in items)
        best_type = type_counts.most_common(1)[0][0]

        pts_counts = Counter(int(it.get("points", 10)) for it in items)
        points = pts_counts.most_common(1)[0][0] if pts_counts else 10

        # Build the new task dictionary
        new_task = {
            "task_id": tid,
            "description": best_desc,
            "type": best_type,
            "points": points,
            "examples": next((it.get("examples", "") for it in items if it.get("examples")), "")
        }

        if best_type == "code":
            lang_counts = Counter(it.get("lang", "python") for it in items)
            best_lang = lang_counts.most_common(1)[0][0]
            
            filename_counts = Counter(it.get("expected_filename", f"{tid}.py") for it in items)
            best_filename = filename_counts.most_common(1)[0][0]

            new_task["lang"] = best_lang
            new_task["expected_filename"] = best_filename

        merged.append(new_task)

    def _tid_key(task_id: str):
        s = str(task_id)
        m = re.search(r"(\d+)", s)
        return (int(m.group(1)) if m else 1_000_000, s)

    merged.sort(key=lambda t: _tid_key(t["task_id"]))
    return {"tasks": merged}

# -------------------- DISK CACHE HELPERS --------------------
def _cache_key(spec_text: str, model: str) -> str:
    """Stable key for the assignment spec + model + schema version."""
    h = hashlib.sha256()
    h.update(spec_text.encode("utf-8"))
    h.update(b"|model=" + (model or "").encode("utf-8"))
    h.update(b"|schema=" + SCHEMA_VERSION.encode("utf-8"))
    return h.hexdigest()

def _cache_file_for(key: str, cache_dir: str) -> Path:
    p = Path(cache_dir); p.mkdir(parents=True, exist_ok=True)
    return p / f"{key}.json"

def _save_cache(cache_path: Path, tasks_obj: Dict[str, Any], meta: Dict[str, Any]):
    record = {"_meta": meta, "tasks": tasks_obj.get("tasks", [])}
    cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_cache(cache_path: Path) -> Dict[str, Any] | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            return {"tasks": data["tasks"]}  # return only the shape downstream expects
    except Exception:
        return None
    return None

# -------------------- PUBLIC API --------------------
def parse_assignment(assignment_path: str,
                     model: str = DEFAULT_MODEL,
                     temperature: float = DEFAULT_TEMPERATURE,
                     n_trials: int = 1,
                     use_cache: bool = True,
                     cache_dir: str = CACHE_DIR_DEFAULT,
                     force_refresh: bool = False) -> Dict[str, Any]:
    """
    Parse an assignment file into tasks. Supports disk cache so repeated runs
    won't call the LLM again for the same spec+model unless force_refresh=True.
    """
    # 1) read spec
    spec = _read_file_as_text(assignment_path)

    # 2) cache check
    key = _cache_key(spec, model)
    cache_path = _cache_file_for(key, cache_dir)
    if use_cache and not force_refresh and cache_path.exists():
        cached = _load_cache(cache_path)
        if cached:
            # return validated cached result
            return validate_tasks(cached)

    # 3) build prompts
    # system = (
    #     "You extract a STRICT JSON schema of tasks from an academic assignment. "
    #     "Return ONLY JSON. No explanations. Ensure valid UTF-8 and correct escaping."
    # )
    system = (
        "You are a specialized text-to-JSON extractor."
        "Your sole function is to parse an academic assignment specification and output a strict JSON array of tasks."
        "You must never generate any prose, explanations, or conversational text."
        "Your output must be a single, valid JSON object, formatted with UTF-8 encoding and correctly escaped."
    )
    # user = (
    #     "Extract tasks from the assignment spec below.\n"
    #     "Return JSON with a single key 'tasks': a list of objects.\n"
    #     "Each task MUST include: task_id, description, expected_filename (or infer), "
    #     "type ('code'|'short_answer'|'math_proof'), lang (when type='code'), points (int), examples (optional).\n\n"
    #     "Assignment spec:\n-----\n"
    #     f"{spec}\n-----\n"
    #     "Return ONLY JSON."
    # )
    user = (
        "Extract tasks from the assignment specification provided below.\n"
        "Your response must be a single JSON object with a key 'tasks', which is an array of task objects.\n"
        "Each task object MUST contain the following fields:\n"
        "- task_id: A unique identifier for the task (string).\n"
        "- description: A detailed description of the task (string).\n"
        "- expected_filename: The expected filename for submissions (string). If not specified, infer it as '<task_id>.py'.\n"
        "- type: The type of task, which can be 'code', or 'mathematics' (string).\n"
        "- lang: The programming language for code tasks (string). Required if type is 'code'.\n"
        "- points: The maximum points achievable for the task (integer).\n"
        "- examples: Optional examples or additional information about the task (string, optional).\n\n"
        "Assignment specification:\n-----\n"
        f"{spec}\n-----\n"
        "Remember, your output must be valid JSON without any additional text or explanations."
    )

    # 4) LLM calls (one or more trials), normalization, merge
    trials: List[Dict[str, Any]] = []
    for _ in range(max(1, n_trials)):
        data = llm_call_json(system, user, model=model, temperature=temperature)
        tasks = [_normalize_task(t) for t in data.get("tasks", []) if isinstance(t, dict)]
        trials.append({"tasks": tasks})

    merged = _merge_trials(trials)
    validated = validate_tasks(merged)

    # 5) save to cache (metadata only in file, not returned to pipeline)
    if use_cache:
        meta = {
            "created_at": int(time.time()),
            "model": model,
            "temperature": temperature,
            "schema_version": SCHEMA_VERSION,
            "assignment_sha256": key,
            "source_file": str(Path(assignment_path).resolve()),
        }
        _save_cache(cache_path, validated, meta)

    return validated
