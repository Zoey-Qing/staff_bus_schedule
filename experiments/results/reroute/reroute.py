"""Re-route the final schedule of every method with a longer integer-routing time limit (2026-11-11).

Reviewer request: a stronger common routing evaluation of every final schedule, to separate schedule quality from the
accuracy of the heuristic routing solver. Uses the verified code unchanged: net_routing.route_schedule with
time_limit=1200 s (the experiments used 120 s), and the staff cost of net_benders.StaffModel.
Also writes the parts of the transport cost (driving, punctuality, empty runs, fleet) of every plan.
Output: reroute.csv (one row per cluster and method).
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, "E:/schedule/integrated")
sys.path.insert(0, "E:/schedule/surrogate")
import net_benders as B  # noqa: E402
import net_routing as N  # noqa: E402
from staff_surrogate import branch_cluster, staff_args  # noqa: E402

R = Path("E:/schedule/revision/results_new/original")
D = "E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx"
S = "Berkeley_Mock_Data_25_Stores_original_unbalanced"
LIMIT = 1200


def files(c):
    a4 = R / "ablation_m5eq" / c / "A4_no_smooth"
    return [("sequential x0", a4 / "input" / f"{S}_{c}_x0.csv"),
            ("ours", a4 / f"{S}_{c}_s1.schedule.csv"),
            ("L1 transport-aware staff model", R / "compare_all" / c / "L1" / f"{S}_{c}_ormeci_schedule.csv"),
            ("L2 classical Benders", R / "benders_nolift" / f"{S}_{c}_schedule.csv"),
            ("L3 staff-roster CG", R / "compare_all" / c / "L4" / f"{S}_{c}_nolift_schedule.csv"),
            ("decomposition without lifting", R / "compare_all" / c / "L3" / f"{S}_{c}_nolift_schedule.csv")]


def main():
    out = Path(__file__).with_name("reroute.csv")
    clusters = sys.argv[1:] or ["1aea822f", "a1e8b5a7", "ba5b9796", "df75e3dc"]
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["cluster", "method", "labor", "transport", "z", "lp", "route_gap", "driving", "punctuality",
                        "empty", "fleet", "buses", "trips", "seconds"])
        for c in clusters:
            cluster = next(k for k in set(branch_cluster(D).values()) if k.startswith(c))
            prob = B.ClusterProblem(cluster, D, staff_args(), 3)
            ev = B.StaffModel(prob.args, prob.data)
            for name, f in files(c):
                if not f.exists():
                    print("missing", c, name, f, flush=True)
                    continue
                x = B.load_schedule(f, prob.data.staff_ids)
                t0 = time.time()
                vals = ev.value_of(x, 3)
                plan = N.route_schedule(prob.net, prob.events(x), 3, time_limit=LIMIT)
                p = B._parts(plan)
                labor = vals["j_reg"] + vals["j_over"]
                w.writerow([c, name, round(labor, 2), round(p["cost"], 2), round(labor + p["cost"], 2), round(p["lp"], 2),
                            p["route_gap"], round(p["driving"], 2), round(p["punctuality"], 2), round(p["empty"], 2),
                            round(p["fleet"], 2), p["buses"], p["trips"], round(time.time() - t0, 1)])
                fh.flush()
                print(c, name, round(labor + p["cost"], 2), "route gap", p["route_gap"], flush=True)


if __name__ == "__main__":
    main()
