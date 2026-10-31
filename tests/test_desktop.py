"""The desktop build has no PyTorch, and an LLM is optional there."""

import subprocess
import sys

from qanun.app_factory import embedding_model, reranker_model
from qanun.config import ROOT
from qanun.env import env_path

BLOCK_TORCH = (
    "import sys\n"
    "for name in ('torch', 'sentence_transformers', 'transformers'):\n"
    "    sys.modules[name] = None\n"
)


def run_without_torch(code: str, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "QANUN_ENV_FILE": str(ROOT / "no-such-file"), **env_extra}
    for name in ("QANUN_LLM_API_KEY", "QANUN_LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if name not in env_extra:
            env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-c", BLOCK_TORCH + code], capture_output=True, text=True, env=env, cwd=ROOT
    )


def test_desktop_settings_turn_off_embeddings_and_the_reranker(monkeypatch):
    monkeypatch.setenv("QANUN_EMBEDDING_MODEL", "none")
    assert embedding_model() is None
    assert reranker_model(None, translating=True, settings={}) is None


def test_every_module_imports_without_pytorch():
    """PyInstaller collects all qanun submodules, so none may need torch to import."""
    code = (
        "import pkgutil, importlib, qanun\n"
        "for m in pkgutil.walk_packages(qanun.__path__, 'qanun.'):\n"
        "    if m.name != 'qanun.api.app':\n"
        "        importlib.import_module(m.name)\n"
    )
    result = run_without_torch(code, {"QANUN_EMBEDDING_MODEL": "none"})
    assert result.returncode == 0, result.stderr


def test_no_key_desktop_setup_answers_extractively():
    code = (
        "from qanun.app_factory import build_assistant\n"
        "a = build_assistant()\n"
        "assert a.engine.dense is None and a.generator is None and a.translator is None\n"
        "r = a.ask('How is end of service gratuity calculated?')\n"
        "assert r.mode == 'extractive' and r.citations\n"
        "print('ok')\n"
    )
    result = run_without_torch(code, {"QANUN_EMBEDDING_MODEL": "none"})
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_desktop_setup_with_a_key_builds_without_pytorch():
    code = (
        "from qanun.app_factory import build_assistant\n"
        "a = build_assistant()\n"
        "assert a.engine.dense is None and a.engine.reranker is None\n"
        "assert a.generator is not None and a.translator is not None\n"
    )
    result = run_without_torch(code, {"QANUN_EMBEDDING_MODEL": "none", "QANUN_LLM_API_KEY": "placeholder"})
    assert result.returncode == 0, result.stderr


def test_env_file_can_be_pointed_elsewhere(monkeypatch, tmp_path):
    monkeypatch.setenv("QANUN_ENV_FILE", str(tmp_path / ".env"))
    assert env_path() == tmp_path / ".env"


def test_launcher_keeps_keys_out_of_the_build():
    build = (ROOT / "desktop" / "build.ps1").read_text(encoding="utf-8")
    assert ".env" not in build
    assert "--exclude-module torch" in build
    assert "QANUN_ENV_FILE" in (ROOT / "desktop" / "launcher.py").read_text(encoding="utf-8")
