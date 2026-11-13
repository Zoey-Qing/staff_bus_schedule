"""Two checks of the bus plans of x0 and of our plan (2026-11-11, reviewer requests).

1. Lost work: the staffing rules count the nominal shift, but a rider dropped off after the shift start or picked up
   before the shift end loses those minutes. Sum them per plan (staff-minutes per week) and compare with the
   scheduled staff-minutes of the riders.
2. Fleet: no plan uses empty runs, so each bus stays at its accommodation. Count, per bus type and accommodation, the
   largest number of trips on the road at the same minute of the (cyclic) week, and compare the total with the fleet
   n_k of the plan. Equality means the fleet is exactly what the trips need.
Uses the verified routing code unchanged. Output: timing_fleet_check.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "E:/schedule/integrated")
sys.path.insert(0, "E:/schedule/surrogate")
import net_benders as B  # noqa: E402
import net_routing as N  # noqa: E402
from staff_surrogate import branch_cluster, staff_args  # noqa: E402

R = Path("E:/schedule/revision/results_new/original")
D = "E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx"
S = "Berkeley_Mock_Data_25_Stores_original_unbalanced"


def check(prob, x):
    net = prob.net
    ev = prob.events(x)
    q, p, _e = net.demand(ev)
    plan = N.route_schedule(net, ev, 3, time_limit=120)
    lost = 0.0
    load = defaultdict(lambda: np.zeros(N.WEEK, dtype=int))
    for h in plan.trips:
        tb = net.table(h.acc, h.k)
        tau = tb.taus[tb.seqs.index(h.seq)]
        for b, tj in zip(h.seq, tau):
            u = h.delta + tj
            lost += q.get((b, h.acc, h.t), 0) * max(0, u) + p.get((b, h.acc, h.t), 0) * max(0, -u)
        mins = (np.arange(h.dep, h.dep + h.dur)) % N.WEEK
        load[h.k, h.acc][mins] += 1
    need = defaultdict(int)
    for (k, a), arr in load.items():
        need[net.buses[k].key] += int(arr.max())
    rider_min = sum(60 * prob.shift[s]["duration"] for i, s in x if i in net.home)
    return lost, rider_min, dict(need), dict(plan.buses), len(plan.runs)


def main():
    rows = []
    for c in ["1aea822f", "a1e8b5a7", "ba5b9796", "df75e3dc"]:
        cluster = next(k for k in set(branch_cluster(D).values()) if k.startswith(c))
        prob = B.ClusterProblem(cluster, D, staff_args(), 3)
        a4 = R / "ablation_m5eq" / c / "A4_no_smooth"
        for name, f in (("x0", a4 / "input" / f"{S}_{c}_x0.csv"), ("ours", a4 / f"{S}_{c}_s1.schedule.csv")):
            x = B.load_schedule(f, prob.data.staff_ids)
            lost, rm, need, fleet, runs = check(prob, x)
            rows.append(dict(cluster=c, plan=name, lost_staff_minutes=round(lost), rider_staff_minutes=rm,
                             lost_pct=round(100 * lost / rm, 3), fleet=fleet, max_on_road=need, empty_runs=runs,
                             fleet_equals_need=sum(fleet.values()) == sum(need.values())))
            print(rows[-1], flush=True)
    with open(Path(__file__).with_name("timing_fleet_check.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
