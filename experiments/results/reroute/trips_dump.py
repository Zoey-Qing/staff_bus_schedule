"""Dump the trips of the bus plans of x0 and of our plan (2026-11-12, for the trip-timing figure).
Uses the verified routing code unchanged, as timing_fleet_check.py. Output: trips_<cluster>.csv with one row per trip:
plan, accommodation, hour of the week, stops, departure minute offset delta, duration, drop-offs, pick-ups.
Usage: python trips_dump.py a1e8b5a7 [...]
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, "E:/schedule/integrated")
sys.path.insert(0, "E:/schedule/surrogate")
import net_benders as B  # noqa: E402
import net_routing as N  # noqa: E402
from staff_surrogate import branch_cluster, staff_args  # noqa: E402

R = Path("E:/schedule/revision/results_new/original")
D = "E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx"
S = "Berkeley_Mock_Data_25_Stores_original_unbalanced"


def main(c):
    cluster = next(k for k in set(branch_cluster(D).values()) if k.startswith(c))
    prob = B.ClusterProblem(cluster, D, staff_args(), 3)
    net = prob.net
    a4 = R / "ablation_m5eq" / c / "A4_no_smooth"
    rows = []
    for name, f in (("x0", a4 / "input" / f"{S}_{c}_x0.csv"), ("ours", a4 / f"{S}_{c}_s1.schedule.csv")):
        x = B.load_schedule(f, prob.data.staff_ids)
        ev = prob.events(x)
        q, p, _e = net.demand(ev)
        plan = N.route_schedule(net, ev, 3, time_limit=120)
        for h in plan.trips:
            rows.append(dict(plan=name, acc=h.acc, t=h.t, stops=len(h.seq), delta=h.delta, dur=h.dur,
                             drop=sum(q.get((b, h.acc, h.t), 0) for b in h.seq),
                             pick=sum(p.get((b, h.acc, h.t), 0) for b in h.seq)))
        print(name, len(plan.trips), "trips", flush=True)
    with open(Path(__file__).with_name(f"trips_{c}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        main(c)
