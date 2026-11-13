"""Collect run_<variant>.json and the CPLEX logs of master_study.py into master_study_results.csv."""
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ["nocut", "A", "B", "C", "D", "E", "E2", "theta"]
NAMES = dict(nocut="no cuts (theta = 0), aggregated link", A="5 cuts, aggregated link (current)",
             B="A + disaggregated y >= x", C="B without y <= load", D="A + disaggregated y >= x as user cuts",
             E="A + one row per stop and rider y >= sum_s x[i,s]", E2="E with y continuous in [0,1]",
             theta="A with objective theta only (diagnostic)")


def root_bounds(log):
    """(root LP after presolve, root bound after the cut rounds) from the node-0 lines of the CPLEX log.
    On a 'Cuts:' line the bound is the node objective (first number); otherwise it is the Best Bound column."""
    first = bound = None
    for line in log.splitlines():
        if re.match(r"^\s*\*?\s+0\+?\s+\d+\s", line):
            nums = re.findall(r"-?\d+\.\d+", line)
            if not nums or "infeasible" in line or "cutoff" in line:
                continue
            if "Cuts:" in line or len(nums) < 3:
                val = float(nums[0])
            else:
                val = float(nums[-2])
            if first is None and not line.lstrip().startswith("*"):
                first = val
            bound = val
    return first, bound


rows = []
for v in ORDER:
    f = HERE / f"run_{v}.json"
    if not f.exists():
        continue
    r = json.load(open(f))
    log = (HERE / f"run_{v}_cplex.log").read_text(encoding="utf-8")
    root_lp, root_bound = root_bounds(log)
    rows.append(dict(variant=v, description=NAMES[v], cuts=r["cuts"], disagg_added=r["added"], ylink_removed=r["removed"],
                     constraints=r["constraints"], lp=r["lp"], lp_wages=r["lp_wages"], lp_overtime=r["lp_overtime"],
                     lp_theta=r["lp_theta"], root_lp=root_lp, root_bound=root_bound, best_bound=r["best_bound"], incumbent=r.get("incumbent"),
                     gap=r.get("gap"), nodes=r["nodes"], mip_seconds=r["mip_seconds"], status=r["mip_status"],
                     inc_wages=r.get("inc_wages"), inc_overtime=r.get("inc_overtime"), inc_theta=r.get("inc_theta"),
                     inc_is_x0=r.get("inc_same_as_x0"),
                     inc_max_events=r.get("inc_max_events_per_rider_hour")))
df = pd.DataFrame(rows)
df.to_csv(HERE / "master_study_results.csv", index=False)
pd.set_option("display.width", 250)
print(df.drop(columns="description").to_string(index=False, float_format=lambda z: f"{z:,.1f}"))
