"""Ugovor konfiguracije: _mini_yaml i PyYAML daju iste vrednosti i tipove."""
from __future__ import annotations

import os
import re
import sys

import pytest

from neural.core.config import DEFAULT_CONFIG_PATH, _mini_yaml, load_config


def _yaml():
    if os.environ.get("NEURAL_REQUIRE_YAML") == "1":
        return __import__("yaml")  # nedostatak PyYAML-a obara test
    return pytest.importorskip("yaml")


def test_parity_with_pyyaml_on_default_config():
    yaml = _yaml()
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    mini, ref = _mini_yaml(text), yaml.safe_load(text)
    assert mini == ref
    assert {k: type(v) for k, v in mini.items()} == {k: type(v) for k, v in ref.items()}


def test_load_config_same_with_and_without_pyyaml(monkeypatch):
    _yaml()
    with_yaml = load_config().to_dict()
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert load_config().to_dict() == with_yaml


def test_require_yaml_flag_is_honoured():
    if os.environ.get("NEURAL_REQUIRE_YAML") == "1":
        import yaml  # noqa: F401  (mora postojati)
    else:
        pytest.importorskip("yaml")


@pytest.mark.parametrize("text", ["a:\n  b: 1\n", "x: 1\n\n  y: 2\n", "- x\n", "ok: 1\nno colon\n", "k:\n"])
def test_mini_yaml_errors_carry_line_number(text):
    with pytest.raises(ValueError, match=r"line \d+"):
        _mini_yaml(text)


def test_mini_yaml_error_line_is_correct():
    with pytest.raises(ValueError) as e:
        _mini_yaml("a: 1\nb: 2\nbroken\n")
    assert re.search(r"line 3\b", str(e.value))
