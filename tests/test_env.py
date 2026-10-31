import subprocess
from pathlib import Path

from qanun.config import ROOT
from qanun.env import load_env_file, parse_env


def test_parse_env_formats():
    text = (
        "# a comment\n"
        "PLAIN=one\n"
        "export EXPORTED=two\n"
        'QUOTED="three # kept"\n'
        "SINGLE='four'\n"
        "COMMENTED=five # dropped\n"
        "EMPTY=\n"
        "not a setting\n"
    )
    assert parse_env(text) == {
        "PLAIN": "one",
        "EXPORTED": "two",
        "QUOTED": "three # kept",
        "SINGLE": "four",
        "COMMENTED": "five",
        "EMPTY": "",
    }


def test_process_environment_wins_over_the_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text("A=from-file\nB=from-file\nC=\n", encoding="utf-8")
    env = {"A": "from-process"}
    assert load_env_file(path, env) == ["B"]
    assert env == {"A": "from-process", "B": "from-file"}


def test_windows_byte_order_mark_and_line_endings(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"\xef\xbb\xbfQANUN_LLM_API_KEY=abc\r\nOTHER=def\r\n")
    env = {}
    load_env_file(path, env)
    assert env == {"QANUN_LLM_API_KEY": "abc", "OTHER": "def"}


def test_missing_file_is_fine(tmp_path):
    assert load_env_file(tmp_path / "nope", {}) == []


def test_loader_returns_names_and_never_values(tmp_path):
    path = tmp_path / ".env"
    path.write_text("QANUN_LLM_API_KEY=value-that-must-stay-private\n", encoding="utf-8")
    assert "value-that-must-stay-private" not in repr(load_env_file(path, {}))


def test_example_file_holds_no_values_for_secrets():
    values = parse_env((ROOT / ".env.example").read_text(encoding="utf-8"))
    assert "QANUN_LLM_API_KEY" in values
    secrets = {k: v for k, v in values.items() if "KEY" in k or "TOKEN" in k or "SECRET" in k}
    assert all(not v for v in secrets.values())


def test_env_file_is_ignored_by_git_and_example_is_not():
    def ignored(name: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "-q", name], cwd=Path(ROOT), capture_output=True
        )
        return result.returncode == 0

    assert ignored(".env")
    assert ignored(".env.local")
    assert not ignored(".env.example")
