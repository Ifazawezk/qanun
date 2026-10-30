"""Windows desktop app: runs the API on a private local port and shows it in a window.

Built with desktop/build.ps1. It uses keyword search only, so it needs no model
download and stays small enough to share as a single file.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Windowed builds have no console, and uvicorn's logging expects real streams.
for name in ("stdout", "stderr"):
    if getattr(sys, name) is None:
        setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
os.environ["QANUN_ROOT"] = str(BASE)
os.environ.setdefault("QANUN_EMBEDDING_MODEL", "none")

import uvicorn  # noqa: E402

from qanun.api.app import create_app  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until_ready(url: str, timeout: float = 60) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            time.sleep(0.1)
    return False


def show(url: str) -> None:
    """Open a native window, or the default browser if that isn't possible."""
    try:
        import webview

        webview.create_window("qanun", url, width=1180, height=820, min_size=(420, 560))
        webview.start()
    except Exception:
        webbrowser.open(url)
        threading.Event().wait()


def main() -> None:
    port = free_port()
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_config=None)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

    url = f"http://127.0.0.1:{port}"
    if wait_until_ready(f"{url}/health"):
        show(url)
    os._exit(0)


if __name__ == "__main__":
    main()
