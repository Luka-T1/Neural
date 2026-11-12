"""Contract: the organism is isolated from world/body/probe and has a fixed (9,) -> (4,) interface."""
import re
from pathlib import Path

import numpy as np
import pytest

from neural.organism.organism import Organism

ROOT = Path(__file__).resolve().parents[2] / "neural" / "organism"
FORBIDDEN = ["probe", "world", "body", "env", "x_perp", "escaped", "ChannelWorld"]
PATTERN = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(FORBIDDEN) + r")(?![A-Za-z0-9])", re.I)


def test_sources_found():
    assert list(ROOT.rglob("*.py"))


def test_no_forbidden_words_in_organism_source():
    hits = []
    for f in ROOT.rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = PATTERN.search(line)
            if m:
                hits.append(f"{f.name}:{i}: {m.group(1)}")
    assert not hits, hits


def test_step_shapes():
    org = Organism(seed=0)
    out = org.step(np.zeros(9), 0.1)
    assert isinstance(out, np.ndarray) and out.shape == (4,)


@pytest.mark.parametrize("n", [0, 1, 8, 10])
def test_step_rejects_wrong_shape(n):
    org = Organism(seed=0)
    with pytest.raises(ValueError):
        org.step(np.zeros(n), 0.1)
