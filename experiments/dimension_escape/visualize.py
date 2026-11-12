"""Grafik vremenske linije (opciono; zahteva matplotlib)."""
from __future__ import annotations


def plot_timeline(tl: dict, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nije instaliran; grafik preskočen")
        return False
    runs = tl.get("runs")
    if runs is None:
        runs = [tl]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    for r in runs:
        lab = f"{r['cond']} s{r['seed']}"
        axes[0].plot(r["t_ms"], r["x_perp"], label=lab, lw=0.8)
        axes[1].plot(r["t_ms"], r["q_cpg"], lw=0.8)
        for e in r.get("da_events_ms", []):
            axes[1].axvline(e, color="g", alpha=0.2, lw=0.5)
        if r.get("t_escape_ms") is not None:
            for ax in axes:
                ax.axvline(r["t_escape_ms"], color="r", alpha=0.5, lw=0.8)
    axes[0].set_ylabel("‖x⊥‖")
    axes[1].set_ylabel("q_cpg")
    axes[1].set_xlabel("t [ms]")
    if len(runs) <= 8:
        axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
