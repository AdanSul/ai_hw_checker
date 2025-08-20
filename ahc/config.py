import os, json
from dotenv import load_dotenv
load_dotenv()

def load_config(path: str = "config.json") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    return cfg
