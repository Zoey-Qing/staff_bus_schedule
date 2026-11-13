"""Run the Benders-seeded SG-LNS (benders_sglns.py) on every cluster with a finished Benders run, five jobs at a time.
The search gets the time Benders used before re-timing, as the plain SG-LNS runs did; the whole pipeline therefore costs
the Benders run plus that time. A run whose log ends with DONE is skipped.

Usage: python run_benders_sglns.py <benders_dir> <out_dir> --seeds 1,2,3 [--start benders|x0] [--prices benders|direct]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_equal_time import ROOT, data_file


def job(spec):
    src, out, seed, a = spec
    with open(src) as f:
        seconds = json.load(f)["benders_seconds"]
    stem, cluster = src.stem.rsplit("_", 1)
    cmd = [sys.executable, "-u", str(ROOT / "surrogate" / "benders_sglns.py"), "--cluster", cluster, "--data", str(data_file(stem)),
           "--benders-dir", str(src.parent), "--time-limit", f"{seconds:.0f}", "--seed", str(seed), "--threads", str(a.threads),
           "--start", a.start, "--prices", a.prices, "--out", str(out.with_suffix(".json"))]
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
    p.add_argument("--start", default="benders")
    p.add_argument("--prices", default="benders")
    a = p.parse_args()
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [(f, out_dir / f"{f.stem}_s{s}", s, a) for s in a.seeds.split(",") for f in sorted(Path(a.benders_dir).resolve().glob("*.json"))]
    specs = [sp for sp in specs if not sp[1].with_suffix(".log").exists() or "DONE" not in sp[1].with_suffix(".log").read_text()]
    print(f"{len(specs)} runs to do", flush=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        codes = list(pool.map(job, specs))
    failed = sum(c != 0 for c in codes)
    print(f"ALL DONE, failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)
