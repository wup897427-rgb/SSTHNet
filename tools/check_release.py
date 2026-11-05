"""Check the allowlisted snapshot without needing numerical dependencies."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".csv", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".vital",
                      ".pkl", ".pickle", ".parquet", ".pem", ".key", ".zip", ".log"}


def check():
    errors = []
    entries = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())["files"]
    for item in entries:
        path = ROOT / item["file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["release_sha256"]:
            errors.append("Source hash mismatch: " + item["file"])
    count = 0
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts or "__pycache__" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append("Excluded asset present: " + relative)
        if path.stat().st_size > 2_000_000:
            errors.append("Unexpected large file: " + relative)
        if path.suffix in (".py", ".md", ".json", ".txt"):
            text = path.read_text(encoding="utf-8-sig")
            if path.suffix == ".py":
                ast.parse(text, filename=relative)
                count += 1
            for pattern in (r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
                            r"gh[pousr]_[A-Za-z0-9]{20,}",
                            r"github_pat_[A-Za-z0-9_]{20,}",
                            r"https://[^\s/@:]+:[^\s/@]+@"):
                if re.search(pattern, text):
                    errors.append("Possible credential: " + relative)
            if re.search(r"/home/pjc\d+|/var/tmp/pjc\d+|C:\\Users\\32116", text):
                errors.append("Private workstation path: " + relative)
    print(json.dumps(dict(curated_sources=len(entries), parsed_python_files=count,
                          errors=errors, status="PASS" if not errors else "FAIL"), indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(check())
