"""Reads local settings from a .env file.

Keys come from the process environment or from a gitignored .env in the repo
root. The process environment always wins. This module only handles names: it
never logs, prints or returns a value, so a key cannot leak through it.

QANUN_ENV_FILE points at another file; the desktop app uses it to look next to
the exe, since its repo root is a temporary folder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping

from qanun.config import ROOT

LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = LINE.match(line)
        if not match:
            continue
        name, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[name] = value
    return values


def env_path() -> Path:
    return Path(os.environ.get("QANUN_ENV_FILE") or ROOT / ".env")


def load_env_file(
    path: Path | None = None, environ: MutableMapping[str, str] | None = None
) -> list[str]:
    """Add variables from the file that are not already set, and return their names."""
    environ = os.environ if environ is None else environ
    path = path or env_path()
    try:
        # utf-8-sig: Notepad can save a byte order mark, which would corrupt the first name.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return []
    added = []
    for name, value in parse_env(text).items():
        if value and not environ.get(name):
            environ[name] = value
            added.append(name)
    return added
