"""Primer: eksperiment refleksa istezanja.

Pokretanje (iz korena repozitorijuma):
    python examples/stretch_reflex/run.py --seed 0

Ispisuje mere za eksperiment, kontrolu bez refleksa i baznu liniju bez istezanja,
i čuva grafikon u examples/stretch_reflex/out/ (ako je matplotlib dostupan).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from neural.systems.stretch_reflex import run_stretch_experiment  # noqa: E402


def _fmt(res: dict) -> str:
    return (f"latency_ms={res['latency_ms']:.6f} emg_peak={res['emg_peak']:.6f} "
            f"force_peak_N={res['force_peak_N']:.6f} angle_peak_rad={res['angle_peak_rad']:.6f}")


def _plot(seed: int, on: dict, off: dict, base: dict, out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nije dostupan; grafikon nije sačuvan")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
    for res, label in ((on, "refleks"), (off, "bez refleksa"), (base, "bez istezanja")):
        axes[0].plot(res["t"], res["angle"], label=label)
        axes[1].plot(res["t"], res["emg"], label=label)
        axes[2].plot(res["t"], res["force"], label=label)
    axes[0].set_ylabel("ugao [rad]")
    axes[1].set_ylabel("EMG [a.u.]")
    axes[2].set_ylabel("sila ekstenzora [N]")
    axes[2].set_xlabel("vreme [s]")
    axes[0].legend()
    fig.suptitle(f"Refleks istezanja, seed={seed}")
    fig.tight_layout()
    path = out_dir / f"stretch_reflex_seed{seed}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()

    on = run_stretch_experiment(seed=args.seed, reflex_enabled=True)
    off = run_stretch_experiment(seed=args.seed, reflex_enabled=False)
    base = run_stretch_experiment(seed=args.seed, stretch=False)

    print(f"seed={args.seed}")
    print(f"refleks:        {_fmt(on)}")
    print(f"bez refleksa:   {_fmt(off)}")
    print(f"bez istezanja:  {_fmt(base)}")
    path = _plot(args.seed, on, off, base, args.out)
    if path is not None:
        print(f"grafikon: {path}")


if __name__ == "__main__":
    main()
