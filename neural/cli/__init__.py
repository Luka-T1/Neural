"""Konzolna aplikacija `neural`.

Tanak omotač oko harness-a `experiments.dimension_escape`: ne sadrži logiku eksperimenta.
Bez sporednih efekata pri importu (na Windows-u ProcessPoolExecutor koristi spawn i ponovo
uvozi module). Harness se uvozi tek u trenutku poziva `run`, pa testovi mogu da ga zamene.

Subkomande: run, list, show, presets, info. Bez argumenata pokreće se meni.
Izlazni kodovi: 0 uspeh, 2 loši argumenti ili nepostojeći direktorijum/izveštaj.
"""
from __future__ import annotations

import argparse
import builtins
import importlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = Path("experiments") / "dimension_escape" / "results"
HARNESS_MODULE = "experiments.dimension_escape.__main__"

PRESETS: dict[str, list[str]] = {
    "quick": ["--dims", "2", "--seeds", "2", "--duration-ms", "5000"],
    "phase1": ["--dims", "2", "--seeds", "10", "--duration-ms", "60000"],
    "dims3": ["--dims", "3", "--seeds", "10", "--duration-ms", "60000"],
    "roll": ["--dims", "3", "--seeds", "10", "--duration-ms", "60000", "--roll-gain", "0.0125"],
    "phase4": ["--dims", "2", "--seeds", "10", "--duration-ms", "60000", "--escape-sign", "1",
               "--side-wall-mm", "0.5"],
}

PRESET_HELP: dict[str, str] = {
    "quick": "probni rad: 2 dims, 2 seeda, 5 s (nekoliko minuta)",
    "phase1": "faza 1: 2 dims, 10 seedova, 60 s",
    "dims3": "3 dimenzije bez roll-a: 3 dims, 10 seedova, 60 s",
    "roll": "faza 3: 3 dims sa roll-gain 0.0125 rad/ms, 10 seedova, 60 s",
    "phase4": "faza 4: jednostrani zid, 2 dims, escape-sign +1, zid 0.5 mm, 10 seedova, 60 s",
}

RUN_EPILOG = (
    "Ostali argumenti se prosleđuju harness-u (experiments.dimension_escape): "
    "--seeds, --dims, --condition, --duration-ms, --dt-ms, --roll-gain, --escape-sign, --side-wall-mm, --workers, --out, "
    "--trace-every-ms, --record. Poslednja vrednost pobeđuje, pa korisnički argumenti zamenjuju preset. "
    "Preset ide odmah posle `run` (pozicioni) ili preko --preset, ne oba."
)


def preset_argv(name: str) -> list[str]:
    """Kopija argumenata presets-a; KeyError za nepoznat preset."""
    return list(PRESETS[name])


def forward_command(argv: list[str]) -> str:
    return "python -m experiments.dimension_escape " + shlex.join(argv)


# ----------------------------------------------------------------- run
def _exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _execute(forwarded: list[str]) -> int:
    """Poziva harness u trenutku poziva; SystemExit iz harness-a pretvara u int."""
    harness = importlib.import_module(HARNESS_MODULE)
    try:
        rc = harness.main(forwarded)
    except SystemExit as e:
        return _exit_code(e.code)
    return 0 if rc is None else int(rc)


def _cmd_run(rest: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="neural run", description="Pokreni eksperiment (harness).",
                                 epilog=RUN_EPILOG, allow_abbrev=False)
    ap.add_argument("--preset", default=None,
                    help="ime presets-a (ili pozicioni: `run <preset>` odmah posle run): " + ", ".join(PRESETS))
    ap.add_argument("--dry-run", action="store_true", help="samo ispiši komandu, ne pokreći")
    # pozicioni preset samo odmah posle `run`; inače bi vrednost nepoznate opcije (--workers 1) bila pogrešno pročitana
    preset_pos = None
    if rest and not rest[0].startswith("-"):
        preset_pos, rest = rest[0], rest[1:]
    ns, extras = ap.parse_known_args(rest)
    ns.preset_pos = preset_pos
    if ns.preset_pos is not None and ns.preset is not None:
        print("Preset je zadat dvaput (pozicioni i --preset); zadaj samo jedan.", file=sys.stderr)
        return 2
    if ns.preset_pos is not None:
        ns.preset = ns.preset_pos
    if ns.preset is not None and ns.preset not in PRESETS:
        print(f"Nepoznat preset: {ns.preset!r}. Dostupni: {', '.join(PRESETS)}", file=sys.stderr)
        return 2
    forwarded = (preset_argv(ns.preset) if ns.preset is not None else []) + list(extras)
    print(forward_command(forwarded), flush=True)
    if ns.dry_run:
        return 0
    return _execute(forwarded)


# ----------------------------------------------------------------- list / show
def list_results(dir) -> list[dict]:
    """Poddirektorijumi sa report.json, sortirani po imenu."""
    root = Path(dir)
    if not root.is_dir():
        return []
    rows = []
    for sub in sorted(root.iterdir(), key=lambda p: p.name):
        rep_path = sub / "report.json"
        if not sub.is_dir() or not rep_path.is_file():
            continue
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        rows.append({"name": sub.name, "path": str(sub), "dims": rep.get("dims"),
                     "seeds": len(rep.get("seeds", [])), "duration_ms": rep.get("duration_ms"),
                     "has_roll": "roll" in rep})
    return rows


def _cmd_list(results_dir) -> int:
    root = Path(results_dir)
    if not root.is_dir():
        print(f"Direktorijum ne postoji: {root}")
        return 0
    rows = list_results(root)
    if not rows:
        print(f"Nema rezultata (report.json) u {root}")
        return 0
    print(f"{'IME':<28} {'DIMS':>4} {'SEEDS':>5} {'TRAJANJE_MS':>12} ROLL")
    for r in rows:
        dur = "-" if r["duration_ms"] is None else f"{r['duration_ms']:.0f}"
        print(f"{r['name']:<28} {str(r['dims']):>4} {r['seeds']:>5} {dur:>12} "
              f"{'da' if r['has_roll'] else 'ne'}")
    return 0


def show(path, as_json: bool = False) -> int:
    d = Path(path)
    if d.is_file():
        d = d.parent
    md = d / "report.md"
    if not md.is_file():
        print(f"Nema report.md u: {d}", file=sys.stderr)
        return 2
    print(md.read_text(encoding="utf-8"))
    if as_json:
        js = d / "report.json"
        if not js.is_file():
            print(f"Nema report.json u: {d}", file=sys.stderr)
            return 2
        print(js.read_text(encoding="utf-8"))
    return 0


# ----------------------------------------------------------------- presets / info
def _cmd_presets() -> int:
    for name, argv in PRESETS.items():
        print(f"{name}: {PRESET_HELP.get(name, '')}")
        print(f"  {forward_command(argv)}")
    return 0


def _version() -> str:
    try:
        return importlib.metadata.version("neural")
    except importlib.metadata.PackageNotFoundError:
        import tomllib
        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            return str(tomllib.load(f)["project"]["version"])


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def info() -> dict:
    # neural/mechanisms/__init__.py je prazan: mehanizme eksplicitno uvozimo da se registruju.
    from neural.core.config import DEFAULT_CONFIG_PATH
    from neural.core.registry import REGISTRY
    from neural.mechanisms import cpg, da_stdp, homeostasis, hh, stdp, synapses  # noqa: F401
    return {
        "version": _version(),
        "git_hash": _git_hash(),
        "config_path": str(DEFAULT_CONFIG_PATH),
        "mechanisms": {name: REGISTRY.get(name).confidence for name in REGISTRY.names()},
        "homeostasis_mechanism_confidence": getattr(homeostasis, "MECHANISM_CONFIDENCE", None),
    }


def _cmd_info() -> int:
    d = info()
    print(f"version: {d['version']}")
    print(f"git_hash: {d['git_hash']}")
    print(f"config_path: {d['config_path']}")
    print("mechanisms:")
    for name, conf in d["mechanisms"].items():
        print(f"  {name}: {conf}")
    if d["homeostasis_mechanism_confidence"] is not None:
        print(f"homeostasis (MECHANISM_CONFIDENCE): {d['homeostasis_mechanism_confidence']}")
    return 0


# ----------------------------------------------------------------- parser / main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neural", description="Konzola za eksperimente (dimension_escape).")
    sub = p.add_subparsers(dest="cmd", metavar="KOMANDA")
    sub.add_parser("run", help="pokreni eksperiment (ostali argumenti idu harness-u; vidi `run --help`)")
    pl = sub.add_parser("list", help="lista rezultata")
    pl.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ps = sub.add_parser("show", help="prikaži report.md (a uz --json i report.json)")
    ps.add_argument("dir")
    ps.add_argument("--json", action="store_true")
    sub.add_parser("presets", help="lista presets-a")
    sub.add_parser("info", help="verzija, git hash, konfiguracija i mehanizmi")
    return p


def _dispatch(ns) -> int:
    if ns.cmd == "list":
        return _cmd_list(ns.results_dir)
    if ns.cmd == "show":
        return show(ns.dir, ns.json)
    if ns.cmd == "presets":
        return _cmd_presets()
    if ns.cmd == "info":
        return _cmd_info()
    return 2


def interactive(input_fn=None) -> int:
    ask = input_fn or builtins.input
    menu = ("\n1) lista rezultata\n2) prikaži rezultat\n3) presets\n4) pokreni preset\n"
            "5) info\nq) izlaz")
    try:
        while True:
            print(menu)
            choice = ask("> ").strip().lower()
            if choice == "q":
                return 0
            if choice == "1":
                _cmd_list(DEFAULT_RESULTS_DIR)
            elif choice == "2":
                d = ask("Direktorijum: ").strip()
                if d:
                    show(d)
                else:
                    print("Prazan direktorijum.")
            elif choice == "3":
                _cmd_presets()
            elif choice == "4":
                name = ask("Ime presets-a: ").strip()
                if name not in PRESETS:
                    print(f"Nepoznat preset: {name!r}")
                    continue
                forwarded = preset_argv(name)
                print(forward_command(forwarded))
                if ask("Pokreni? [d/N] ").strip().lower() == "d":
                    _execute(forwarded)
            elif choice == "5":
                _cmd_info()
            else:
                print("Nepoznat izbor.")
    except (EOFError, KeyboardInterrupt):
        print()
        return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        return interactive()
    try:
        if argv[0] == "run":
            return _cmd_run(argv[1:])
        parser = build_parser()
        ns = parser.parse_args(argv)
        if ns.cmd is None:
            parser.print_help()
            return 2
        return _dispatch(ns)
    except SystemExit as e:
        return _exit_code(e.code)
