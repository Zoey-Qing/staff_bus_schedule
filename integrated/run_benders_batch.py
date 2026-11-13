"""Run net_benders on every cluster of the given data sets, a few at a time. A cluster with a result file is skipped.

Usage: python run_benders_batch.py <out_dir> <data.xlsx> [<data.xlsx> ...] --jobs 5 --threads 3
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bus_scheduling_routing"))
from bus_riders import branch_cluster

HERE = Path(__file__).resolve().parent


def tasks(out_dir, files):
    for f in files:
        for c in sorted(set(branch_cluster(f).values())):
            prefix = out_dir / f"{Path(f).stem}_{c[:8]}"
            if not Path(f"{prefix}.json").exists():
                yield f, c[:8], prefix


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("data", nargs="+")
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--master-times", default=None, help="passed to net_benders.py; its default when omitted")
    p.add_argument("--bound-time", default=None, help="passed to net_benders.py; its default when omitted")
    p.add_argument("--closed-price", default=None, help="passed to net_benders.py; its default when omitted")
    a = p.parse_args()
    extra = (["--master-times", a.master_times] if a.master_times else []) + (["--bound-time", a.bound_time] if a.bound_time else []) \
        + (["--closed-price", a.closed_price] if a.closed_price else [])
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = [str(Path(f).resolve()) for f in a.data]
    todo, running, failed = list(tasks(out_dir, data)), [], []
    print(f"{len(todo)} clusters to run", flush=True)
    while todo or running:
        while todo and len(running) < a.jobs:
            f, c, prefix = todo.pop(0)
            log = open(f"{prefix}.log", "a")
            proc = subprocess.Popen([sys.executable, "-u", str(HERE / "net_benders.py"), f, str(prefix), "--cluster", c,
                                     "--threads", str(a.threads)] + extra, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, prefix.name, log))
            print(f"started {prefix.name}", flush=True)
        time.sleep(30)
        for item in list(running):
            proc, name, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print(f"finished {name} exit {proc.returncode}", flush=True)
                if proc.returncode:
                    failed.append(name)
    print(f"ALL DONE, failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)
