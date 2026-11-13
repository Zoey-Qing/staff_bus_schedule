"""Service and rest checks of the sequential plan x0 and of our plan (2026-11-11, reviewer requests).

1. Slack families: the violation of every soft rule family, not only the sum per priority level (StaffModel of
   net_benders.py, unchanged; the slacks take their smallest values for the fixed schedule).
2. Days off: staff with a shift start on six days may still work on all seven calendar days (night shifts). We count,
   per plan, the staff whose shifts touch all seven calendar days, and the staff whose longest work-free interval in
   the week (cyclic) is shorter than 24 hours.
Output: service_check.csv
"""
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "E:/schedule/integrated")
sys.path.insert(0, "E:/schedule/surrogate")
import net_benders as B  # noqa: E402
from staff_surrogate import branch_cluster, staff_args  # noqa: E402
from net_benders import _solve  # noqa: E402

R = Path("E:/schedule/revision/results_new/original")
D = "E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx"
S = "Berkeley_Mock_Data_25_Stores_original_unbalanced"


def families(sm, x_on):
    s, d = sm.solver, sm.solver.data if hasattr(sm.solver, "data") else None
    for key, var in s.x.items():
        var.lb = var.ub = 1 if key in x_on else 0
    sol, _ = _solve(sm.mdl, 600, 1e-9, None, 3)
    g = lambda vs: sum(sol.get_value(v) for v in vs)
    out = {
        "L1 demand shortage": g(s.sl[k] for k in s.active_bt),
        "L1 staff without a working day": g(s.v_idle.values()),
        "L2 experienced staff on site": g(s.v_min_onsite_not_new[k] for k in s.active_bt),
        "L3 manager on site": g(s.v_min_onsite_mgr[k] for k in s.active_bt),
        "L3 two staff on site": g(s.v_min_onsite_all[k] for k in s.active_bt),
        "L3 manager at first/last hour": g(list(s.sl_mgr_first.values()) + list(s.sl_mgr_last.values())),
        "L3 six working days": g(s.v_min_work_days.values()),
        "L3 shift length": g(s.v_max_shift.values()),
        "L4 two non-managers on site": g(s.v_min_onsite_not_mgr[k] for k in s.active_bt),
        "L2+L3 daily hours": g(s.v_max_daily.values()),
        "levels": {p: sol.get_value(e) for p, e in sm.levels.items()},
    }
    for var in s.x.values():
        var.lb, var.ub = 0, 1
    return out


def rest(x_on, shift):
    seven, short = 0, 0
    by = {}
    for i, sid in x_on:
        by.setdefault(i, set()).update(range(int(shift[sid]["start"]), int(shift[sid]["end"])))
    for _i, hours in by.items():
        days = {h // 24 for h in hours}
        seven += len(days) == 7
        free, best = 0, 0
        for h in list(range(168)) * 2:  # cyclic week
            free = 0 if h in hours else free + 1
            best = max(best, free)
        short += min(best, 168) < 24
    return seven, short, len(by)


def main():
    rows = []
    for c in ["1aea822f", "a1e8b5a7", "ba5b9796", "df75e3dc"]:
        cluster = next(k for k in set(branch_cluster(D).values()) if k.startswith(c))
        prob = B.ClusterProblem(cluster, D, staff_args(), 3)
        sm = B.StaffModel(prob.args, prob.data)
        a4 = R / "ablation_m5eq" / c / "A4_no_smooth"
        for name, f in (("x0", a4 / "input" / f"{S}_{c}_x0.csv"), ("ours", a4 / f"{S}_{c}_s1.schedule.csv")):
            x = B.load_schedule(f, prob.data.staff_ids)
            fam = families(sm, x)
            seven, short, n = rest(x, prob.shift)
            lev = fam.pop("levels")
            rows.append(dict(cluster=c, plan=name, **{k: round(v, 3) for k, v in fam.items()},
                             **{f"level {p}": round(v, 3) for p, v in lev.items()},
                             staff_working=n, staff_all_7_days=seven, staff_no_24h_rest=short))
            print(rows[-1], flush=True)
    with open(Path(__file__).with_name("service_check.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
