#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path

def compute_payload_core_sha256(obj: dict) -> str:
    core = {k: v for k, v in obj.items() if k != "integrity"}
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

def verify_file(path_str: str):
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            obj = json.loads(line.strip())
            integrity = obj.get("integrity", {})
            computed = compute_payload_core_sha256(obj)
            ok = (integrity.get("value") == computed)
            print(f"[Line {idx}] integrity_ok={ok}")

if __name__ == "__main__":
    verify_file(sys.argv[1])
