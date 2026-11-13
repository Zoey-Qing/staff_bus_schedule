"""Run the surrogate on every cluster that Benders has finished, with the same start schedule and the time Benders used before
re-timing, five jobs at a time. A run with a result file is skipped.

Usage: python run_equal_time.py <benders_dir> <out_dir> --seeds 1,2,3 [--ablation static|lns]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_file(stem):
    for folder in (ROOT / "staff_scheduling", ROOT / "integrated" / "data_gen"):
        if (folder / f"{stem}.xlsx").exists():
            return folder / f"{stem}.xlsx"
    raise FileNotFoundError(stem)


def job(spec):
    src, out, seed, threads, ablation, noise, factor = spec
    with open(src) as f:
        seconds = factor * json.load(f)["benders_seconds"]
    stem, cluster = src.stem.rsplit("_", 1)
    cmd = [sys.executable, "-u", str(ROOT / "surrogate" / "staff_surrogate.py"), "--cluster", cluster, "--data", str(data_file(stem)),
           "--schedule", str(src.with_name(f"{src.stem}_x0.csv")), "--time-limit", f"{seconds:.0f}", "--seed", str(seed),
           "--threads", str(threads), "--ablation", ablation, "--out", str(out.with_suffix(".json"))] \
        + (["--noise", noise] if noise is not None else [])
    with open(out.with_suffix(".log"), "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"finished {out.name} exit {code}", flush=True)
    return code


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("benders_dir")
    p.add_argument("out_dir")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--ablation", default="none")
    p.add_argument("--noise", default=None, help="passed to staff_surrogate.py; its default when omitted")
    p.add_argument("--time-factor", type=float, default=1.0, help="time limit as a multiple of the Benders time (convergence runs)")
    a = p.parse_args()
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [(f, out_dir / f"{f.stem}_s{s}", s, a.threads, a.ablation, a.noise, a.time_factor) for s in a.seeds.split(",") for f in sorted(Path(a.benders_dir).resolve().glob("*.json"))]
    specs = [sp for sp in specs if not sp[1].with_suffix(".log").exists() or "DONE" not in sp[1].with_suffix(".log").read_text()]
    print(f"{len(specs)} runs to do", flush=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        codes = list(pool.map(job, specs))
    failed = sum(c != 0 for c in codes)
    print(f"ALL DONE, failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)
