import hashlib
import json
import os

CACHE_PATH = "gpt_feedback_cache.json"

# Load existing cache or create new
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        cache = json.load(f)
else:
    cache = {}

def _make_key(task_desc: str, code: str) -> str:
    return hashlib.sha256((task_desc + code).encode()).hexdigest()

def get_cached_feedback(task_desc: str, code: str) -> str | None:
    return cache.get(_make_key(task_desc, code))

def store_feedback(task_desc: str, code: str, feedback: str):
    key = _make_key(task_desc, code)
    cache[key] = feedback
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
