"""E4 (penalties in the objective), E5 (scaled penalties) and E6 (sequential with other W_bus), five jobs at a time.

Usage: python run_e4_e6.py <e1_dir> <out_dir> [e4,e5,e6]. A job with a result file is skipped.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = str((HERE.parent / "staff_scheduling" / "Berkeley_Mock_Data_25_Stores.xlsx").resolve())
STEM = Path(DATA).stem
CLUSTERS = ["1aea822f", "662b3495", "a1e8b5a7", "ba5b9796", "df75e3dc"]


def jobs(e1, out, which):
    py = [sys.executable, "-u"]
    for c in CLUSTERS if "e4" in which else []:
        yield out / f"e4_{c}", py + [str(HERE / "run_penalty_objective.py"), DATA, str(e1 / f"{STEM}_{c}"), str(out / f"e4_{c}"), "--cluster", c]
    for c in ("a1e8b5a7", "1aea822f") if "e5" in which else []:
        for scale in ("0.1", "10"):
            prefix = out / f"e5_m{scale}_{c}"
            yield prefix, py + [str(HERE / "net_benders.py"), DATA, str(prefix), "--cluster", c, "--m-scale", scale]
    for c in CLUSTERS if "e6" in which else []:
        for w in ("0", "10"):
            prefix = out / f"e6_w{w}_{c}"
            yield prefix, py + [str(HERE / "net_benders.py"), DATA, str(prefix), "--cluster", c, "--w-bus", w, "--seq-only"]


if __name__ == "__main__":
    e1, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["e4", "e5", "e6"]
    todo = [j for j in jobs(e1, out, which) if not Path(f"{j[0]}.json").exists()]
    running, failed = [], []
    print(f"{len(todo)} jobs to run", flush=True)
    while todo or running:
        while todo and len(running) < 5:
            prefix, cmd = todo.pop(0)
            log = open(f"{prefix}.log", "a")
            running.append((subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT), prefix.name, log))
            print(f"started {prefix.name}", flush=True)
        time.sleep(20)
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
