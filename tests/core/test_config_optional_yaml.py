"""load_config works without PyYAML (optional dependency)."""
import os
import sys

import pytest

from neural.core.config import DEFAULT_CONFIG_PATH, Config, _mini_yaml, load_config


def test_load_default_without_yaml(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)  # `import yaml` -> ImportError
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.seed == 12345 and cfg.dt_ms == 0.025 and cfg.default_method == "rk4"
    assert cfg.mechanisms == {}


def test_overrides_without_yaml(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    cfg = load_config(overrides={"seed": 7, "mechanisms": {"a": 1}})
    assert cfg.seed == 7 and cfg.mechanisms == {"a": 1}


def test_mini_yaml_scalars():
    d = _mini_yaml("a: 1\nb: 2.5\nc: hello  # comment\n\nd: {}\n")
    assert d == {"a": 1, "b": 2.5, "c": "hello", "d": {}}


@pytest.mark.parametrize("text", [
    "a:\n  b: 1\n",       # nesting
    "  a: 1\n",           # indentation
    "\ta: 1\n",           # tab
    "- x\n",              # list
    "just text\n",        # no colon
    "key:\n",             # key without value
    "key:   # c\n",       # key without value after comment
])
def test_mini_yaml_rejects_unsupported(text):
    with pytest.raises(ValueError):
        _mini_yaml(text)


def test_parity_with_pyyaml(monkeypatch):
    yaml = __import__("yaml") if os.environ.get("NEURAL_REQUIRE_YAML") == "1" else pytest.importorskip("yaml")
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    mini, ref = _mini_yaml(text), yaml.safe_load(text)
    assert mini == ref
    assert {k: type(v) for k, v in mini.items()} == {k: type(v) for k, v in ref.items()}
