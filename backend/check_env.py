"""
Check that the current Python has every requirement at the minimum version.

Exit code 0 when everything is satisfied, 1 otherwise (missing packages and
outdated versions are printed). The launchers use this to decide whether the
system Python can run the app or a virtual environment must be prepared.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

REQ = Path(__file__).resolve().parent.parent / "requirements.txt"


def _version_tuple(v: str):
    return tuple(int(p) for p in re.findall(r"\d+", v)[:4])


def problems() -> list[str]:
    out = []
    if sys.version_info < (3, 10):
        out.append(f"Python {sys.version.split()[0]} is too old; 3.10 or newer is required")
    for line in REQ.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(>=\s*([0-9][^,;\s]*))?", line)
        if not m:
            continue
        name, minimum = m.group(1), m.group(4)
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            out.append(f"{name} is not installed")
            continue
        if minimum and _version_tuple(installed) < _version_tuple(minimum):
            out.append(f"{name} {installed} is older than the required {minimum}")
    return out


if __name__ == "__main__":
    issues = problems()
    for i in issues:
        print(f"  - {i}")
    sys.exit(1 if issues else 0)
