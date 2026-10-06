"""Contract for the existing CLI surface (unchanged by this test work)."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr

import pytest

from crdt_sync import __version__
from crdt_sync.__main__ import USAGE, main


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def test_version_prints_package_version():
    code, out, err = run_cli(["version"])
    assert code == 0
    assert out.strip() == __version__
    assert err == ""


@pytest.mark.parametrize("argv", [["help"], ["-h"], ["--help"], []])
def test_help_variants_print_usage(argv):
    code, out, err = run_cli(argv)
    assert code == 0
    assert out == USAGE
    assert "version" in out and "help" in out
    assert err == ""


def test_unknown_command_returns_2_and_writes_stderr():
    code, out, err = run_cli(["bogus"])
    assert code == 2
    assert "unknown command: bogus" in err
    assert USAGE in err
    assert out == ""


def test_module_entry_point_matches_in_process(capsys, monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["python3 -m crdt_sync"])
    sys.modules.pop("crdt_sync.__main__", None)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("crdt_sync", run_name="__main__")
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "version" in captured.out
