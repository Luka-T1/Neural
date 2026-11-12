"""Ugovor: faza 2 (DA-STDP test) je samo specifikacija; VALIDATION.md ne tvrdi validaciju."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
DESIGN = (ROOT / "docs" / "experiment_design.md").read_text(encoding="utf-8")
VALID = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")


def _section_j():
    i = DESIGN.index("## (j) Faza 2 kandidat")
    return DESIGN[i:]


def test_section_j_present_and_marked_unimplemented():
    j = _section_j()
    assert "nije implementirano i nije pokrenuto" in j
    for sub in ("j.1", "j.2", "j.3", "j.4", "j.5", "j.6", "j.7"):
        assert f"**{sub} " in j, sub


def test_section_j_locked_numbers_and_conditions():
    j = _section_j()
    for token in ("20 epizoda", "T_ep = 30 s", "1000–1009", "N = 20", "N = 48", "d_z = 0.8",
                  "d_z = 0.5", "0.955", "|ρ| ≥ 0.1", "T_ep + 1 s", "Holm korekcija za 3 poređenja",
                  "da_shuffled", "no_da", "frozen", "random_motor", "obrtanje s u epizodi 11"):
        assert token in j, token


def test_validation_does_not_claim_da_stdp_and_mentions_candidate():
    row = next(l for l in VALID.splitlines() if "Doprinos DA-STDP" in l)
    assert "NE — nije validirano" in row
    assert "docs/experiment_design.md` (j); nije pokrenut" in VALID


def test_results_folder_has_only_validated_runs_and_no_timing_artifact():
    res = ROOT / "experiments" / "dimension_escape" / "results"
    names = sorted(p.name for p in res.iterdir() if p.is_dir() and p.name != "__pycache__")
    assert names == ["deo2_dims2_seeds10", "deo2_dims3_seeds10", "deo2_phase4_learn_seeds10", "deo2_phase4_seeds10", "deo2_roll_seeds10", "dims2_seeds10", "dims3_short"]
    assert not re.search(r"_timing", VALID)
