"""Ugovor konzolne aplikacije `neural` (faza 4).

NAPOMENA: tekst ugovora nije bio u repozitorijumu kad je ovaj fajl napisan; sadržaj je
izveden iz opisa zadatka (PRESETS, preset_argv, forward_command, main, bez sporednih
efekata pri importu). Ako se pojavi zvanični ugovor, uskladiti ovaj fajl sa njim.
"""
from __future__ import annotations

import importlib
import io
import sys
from contextlib import redirect_stdout

import pytest

import neural.cli as cli


def test_presets_exact_argv():
    assert cli.PRESETS["quick"] == ["--dims", "2", "--seeds", "2", "--duration-ms", "5000"]
    assert cli.PRESETS["phase1"] == ["--dims", "2", "--seeds", "10", "--duration-ms", "60000"]
    assert cli.PRESETS["dims3"] == ["--dims", "3", "--seeds", "10", "--duration-ms", "60000"]
    assert cli.PRESETS["roll"] == ["--dims", "3", "--seeds", "10", "--duration-ms", "60000",
                                   "--roll-gain", "0.0125"]
    assert cli.PRESETS["phase4"] == ["--dims", "2", "--seeds", "10", "--duration-ms", "60000",
                                     "--escape-sign", "1", "--side-wall-mm", "0.5"]
    assert set(cli.PRESET_HELP) == set(cli.PRESETS)


def test_preset_argv_returns_copy_and_raises_for_unknown():
    a = cli.preset_argv("quick")
    a.append("--mutated")
    assert cli.PRESETS["quick"][-1] == "5000"
    with pytest.raises(KeyError):
        cli.preset_argv("nepostojeci")


def test_forward_command_format():
    assert cli.forward_command(["--dims", "2"]) == "python -m experiments.dimension_escape --dims 2"


def test_import_has_no_side_effects(capsys):
    importlib.reload(cli)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_main_help_returns_0():
    with redirect_stdout(io.StringIO()):
        assert cli.main(["--help"]) == 0


def test_main_bad_argument_returns_2(capsys):
    assert cli.main(["nepoznata-komanda"]) == 2


def test_main_empty_argv_calls_interactive(monkeypatch):
    calls = []

    def fake_interactive(*args, **kwargs):
        calls.append(args)
        return 0

    monkeypatch.setattr(cli, "interactive", fake_interactive)
    assert cli.main([]) == 0
    assert calls == [()]


def test_run_dry_run_prints_command_and_returns_0(capsys):
    assert cli.main(["run", "--preset", "quick", "--dry-run"]) == 0
    assert capsys.readouterr().out.strip() == cli.forward_command(cli.preset_argv("quick"))


def test_run_unknown_preset_returns_2():
    assert cli.main(["run", "--preset", "nepostojeci"]) == 2


def test_run_forwards_to_harness_main_at_call_time(monkeypatch, tmp_path, capsys):
    seen = {}

    class FakeHarness:
        @staticmethod
        def main(argv):
            seen["argv"] = list(argv)
            return 0

    monkeypatch.setitem(sys.modules, "experiments.dimension_escape.__main__", FakeHarness)
    rc = cli.main(["run", "--preset", "quick", "--workers", "1", "--out", str(tmp_path)])
    assert rc == 0
    # korisnički argumenti idu posle presets-a, pa poslednji pobeđuje
    assert seen["argv"] == cli.preset_argv("quick") + ["--workers", "1", "--out", str(tmp_path)]


def test_run_phase4_record_forwarded(monkeypatch):
    seen = {}

    class FakeHarness:
        @staticmethod
        def main(argv):
            seen["argv"] = list(argv)
            return 0

    monkeypatch.setitem(sys.modules, "experiments.dimension_escape.__main__", FakeHarness)
    assert cli.main(["run", "phase4", "--record"]) == 0
    assert seen["argv"] == cli.preset_argv("phase4") + ["--record"]


def test_run_converts_systemexit_from_harness_to_int(monkeypatch):
    class FakeHarness:
        @staticmethod
        def main(argv):
            raise SystemExit(2)

    monkeypatch.setitem(sys.modules, "experiments.dimension_escape.__main__", FakeHarness)
    assert cli.main(["run", "--dims", "2", "--roll-gain", "0.1"]) == 2


def test_list_results_contract_keys(tmp_path):
    d = tmp_path / "r1"
    d.mkdir()
    (d / "report.json").write_text('{"dims": 2, "seeds": [0, 1], "duration_ms": 5000.0}', encoding="utf-8")
    rows = cli.list_results(tmp_path)
    assert rows == [{"name": "r1", "path": str(d), "dims": 2, "seeds": 2, "duration_ms": 5000.0,
                     "has_roll": False}]


def test_info_keys():
    d = cli.info()
    assert {"version", "git_hash", "config_path", "mechanisms"} <= set(d)
    assert "da_stdp" in d["mechanisms"] and "homeostasis" in d["mechanisms"]
    assert d["homeostasis_mechanism_confidence"] == "incomplete"
