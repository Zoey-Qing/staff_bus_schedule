"""Run the surrogate on the five base clusters, from the code check start schedules, once E4-E6 have finished.

Each cluster runs twice: with service limits (compare with E1) and with the penalty objective (compare with E4).
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "surrogate" / "results" / "compare"
E1 = ROOT / "revision" / "results_new" / "e1_e3"
CLUSTERS = ["1aea822f", "662b3495", "a1e8b5a7", "ba5b9796", "df75e3dc"]
WORKERS = 5


def benders_running():
    cmd = "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout
    return "run_e4_e6" in out or "net_benders.py" in out


def job(spec):
    cluster, name, extra = spec
    out = OUT / f"{name}_{cluster}"
    cmd = [sys.executable, "-u", str(ROOT / "surrogate" / "staff_surrogate.py"), "--cluster", cluster,
           "--schedule", str(E1 / f"Berkeley_Mock_Data_25_Stores_{cluster}_x0.csv"), "--radius", "40", "--nonneg", "--no-fixed",
           "--threads", "3", "--out", str(out.with_suffix(".json"))] + extra
    with open(out.with_suffix(".log"), "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"finished {name}_{cluster} exit {code}", flush=True)
    return code


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    while benders_running():
        time.sleep(60)
    print("E4-E6 finished, starting", flush=True)
    specs = [(c, n, e) for c in CLUSTERS for n, e in (("limits", ["--service-limits"]), ("penalty", []))]
    with ThreadPoolExecutor(WORKERS) as pool:
        codes = list(pool.map(job, specs))
    print("failures", sum(c != 0 for c in codes), flush=True)
