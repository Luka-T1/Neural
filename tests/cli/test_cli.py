"""Dodatni testovi CLI-ja: parsiranje subkomandi, show, list, meni. Ugovor je u test_cli_contract.py."""
from __future__ import annotations

import builtins
import json
import sys

import pytest

import neural.cli as cli


class _Fake:
    """Zamena za experiments.dimension_escape.__main__ koja beleži argv."""

    calls: list = []
    rc = 0

    @classmethod
    def main(cls, argv):
        cls.calls.append(list(argv))
        return cls.rc


@pytest.fixture
def fake_harness(monkeypatch):
    _Fake.calls = []
    _Fake.rc = 0
    monkeypatch.setitem(sys.modules, "experiments.dimension_escape.__main__", _Fake)
    return _Fake


def _ns(argv):
    return cli.build_parser().parse_args(argv)


# ------------------------------------------------------------- parsiranje
def test_parse_list_default_and_custom_dir():
    assert _ns(["list"]).results_dir == str(cli.DEFAULT_RESULTS_DIR)
    assert _ns(["list", "--results-dir", "x/y"]).results_dir == "x/y"


def test_parse_show_with_and_without_json():
    ns = _ns(["show", "some/dir"])
    assert ns.cmd == "show" and ns.dir == "some/dir" and ns.json is False
    assert _ns(["show", "some/dir", "--json"]).json is True


def test_parse_presets_and_info():
    assert _ns(["presets"]).cmd == "presets"
    assert _ns(["info"]).cmd == "info"


def test_parse_run_is_handled_before_top_level_parser(fake_harness, tmp_path):
    rc = cli.main(["run", "--preset", "dims3", "--workers", "1", "--out", str(tmp_path)])
    assert rc == 0
    assert fake_harness.calls[-1][-4:] == ["--workers", "1", "--out", str(tmp_path)]
    assert fake_harness.calls[-1][:2] == ["--dims", "3"]


def test_run_without_preset_forwards_only_extras(fake_harness, tmp_path):
    cli.main(["run", "--seeds", "1", "--workers", "1", "--out", str(tmp_path)])
    assert fake_harness.calls[-1] == ["--seeds", "1", "--workers", "1", "--out", str(tmp_path)]


def test_run_roll_preset_dry_run_shows_roll_gain(capsys):
    assert cli.main(["run", "--preset", "roll", "--dry-run"]) == 0
    assert "--roll-gain 0.0125" in capsys.readouterr().out


def test_run_harness_return_code_is_propagated(fake_harness, tmp_path):
    fake_harness.rc = 0
    assert cli.main(["run", "--preset", "quick", "--workers", "1", "--out", str(tmp_path)]) == 0


def test_bad_argument_for_list_returns_2(capsys):
    assert cli.main(["list", "--nepoznato"]) == 2


# ------------------------------------------------------------- show
def test_show_without_report_md_returns_2(tmp_path, capsys):
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    assert cli.show(tmp_path) == 2
    assert "report.md" in capsys.readouterr().err


def test_show_missing_dir_returns_2(tmp_path, capsys):
    assert cli.show(tmp_path / "nema") == 2
    assert capsys.readouterr().err


def test_show_prints_md_and_json(tmp_path, capsys):
    (tmp_path / "report.md").write_text("# naslov\n", encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps({"dims": 2}), encoding="utf-8")
    assert cli.show(tmp_path, as_json=True) == 0
    out = capsys.readouterr().out
    assert "# naslov" in out and '"dims": 2' in out


def test_show_json_without_report_json_returns_2(tmp_path):
    (tmp_path / "report.md").write_text("x", encoding="utf-8")
    assert cli.show(tmp_path, as_json=True) == 2


# ------------------------------------------------------------- list
def test_list_nonexistent_dir_returns_0_and_empty(tmp_path, capsys):
    missing = tmp_path / "nema"
    assert cli.main(["list", "--results-dir", str(missing)]) == 0
    assert "ne postoji" in capsys.readouterr().out
    assert cli.list_results(missing) == []


def test_list_skips_dirs_without_report_json(tmp_path):
    (tmp_path / "bez").mkdir()
    (tmp_path / "bez" / "report.md").write_text("x", encoding="utf-8")
    good = tmp_path / "b_dobar"
    good.mkdir()
    (good / "report.json").write_text(json.dumps({"dims": 3, "seeds": [0], "duration_ms": 1.0,
                                                  "roll": {}}), encoding="utf-8")
    rows = cli.list_results(tmp_path)
    assert [r["name"] for r in rows] == ["b_dobar"]
    assert rows[0]["has_roll"] is True


def test_list_prints_table(tmp_path, capsys):
    d = tmp_path / "a"
    d.mkdir()
    (d / "report.json").write_text(json.dumps({"dims": 2, "seeds": [0, 1], "duration_ms": 60000.0}),
                                   encoding="utf-8")
    assert cli.main(["list", "--results-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "IME" in out and "a" in out and "60000" in out


# ------------------------------------------------------------- meni
def _feed(answers):
    it = iter(answers)

    def fake_input(prompt=""):
        return next(it)

    return fake_input


def test_menu_quit_returns_0():
    assert cli.interactive(_feed(["q"])) == 0


def test_menu_eof_and_keyboard_interrupt_return_0():
    def eof(prompt=""):
        raise EOFError

    def kbd(prompt=""):
        raise KeyboardInterrupt

    assert cli.interactive(eof) == 0
    assert cli.interactive(kbd) == 0


def test_menu_unknown_choice_continues(capsys):
    assert cli.interactive(_feed(["9", "q"])) == 0
    assert "Nepoznat izbor" in capsys.readouterr().out


def test_menu_presets_and_info(capsys):
    assert cli.interactive(_feed(["3", "5", "q"])) == 0
    out = capsys.readouterr().out
    assert "quick" in out and "mechanisms:" in out


def test_menu_run_requires_d_to_start(fake_harness, capsys):
    assert cli.interactive(_feed(["4", "quick", "n", "q"])) == 0
    assert fake_harness.calls == []
    assert "python -m experiments.dimension_escape" in capsys.readouterr().out


def test_menu_run_starts_on_d(fake_harness):
    assert cli.interactive(_feed(["4", "quick", "d", "q"])) == 0
    assert fake_harness.calls == [cli.preset_argv("quick")]


def test_menu_unknown_preset_does_not_run(fake_harness, capsys):
    assert cli.interactive(_feed(["4", "nema", "q"])) == 0
    assert fake_harness.calls == []
    assert "Nepoznat preset" in capsys.readouterr().out


def test_menu_default_input_is_builtins_input_at_call_time(monkeypatch):
    monkeypatch.setattr(builtins, "input", _feed(["q"]))
    assert cli.interactive() == 0


def test_run_positional_preset_dry_run(capsys):
    assert cli.main(["run", "phase4", "--seeds", "2", "--dry-run"]) == 0
    expected = cli.forward_command(cli.preset_argv("phase4") + ["--seeds", "2"])
    assert capsys.readouterr().out.strip() == expected


def test_run_positional_and_flag_preset_returns_2():
    assert cli.main(["run", "phase4", "--preset", "quick", "--dry-run"]) == 2
