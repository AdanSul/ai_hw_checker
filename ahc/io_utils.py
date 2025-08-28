from __future__ import annotations
from pathlib import Path
import tempfile, zipfile

def extract_zip_to_dir(zip_path: Path) -> Path:
    """
    Extract a submissions ZIP into a temp directory and return the root path.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="ahc_subs_"))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(temp_dir / "subs")
    return temp_dir / "subs"
