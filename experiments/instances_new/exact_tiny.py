"""Exact integrated optimum of a tiny instance with one branch and one accommodation (2026-11-11, check C3).

With one branch, every trip is a direct round trip to the branch. If a round trip takes at most 60 minutes, trips of
different hours never overlap, so one bus serves all of them and every trip can arrive exactly at the shift change
(u = 0, no punctuality cost). The exact transport cost is then
    T(x) = min over bus types k that fit all riders of [ f_k + sum over open stops of c_rt,k ]
    (f_k: weekly fixed cost of type k, c_rt,k: its round-trip driving cost; a second bus never helps because trips
    do not overlap), which is linear in the stop variables for each k.
v2 (2026-11-12): v1 used only the type with the lowest fixed cost, which is not the cheapest on T2; it now solves one
MIP per type and takes the minimum. v1 output is kept in exact_v1_single_type/. The integrated model becomes one MIP: the staff model of the branch with the
service limits of x0, the stop variables of the paper, and this transport cost. Solved to optimality by CPLEX.
The script checks the assumptions, solves the MIP, routes the optimal schedule with the routing solver to confirm
T(x*), and compares with the lower bound and plan of our method (runs/<name>).
Usage: python exact_tiny.py T1 [T2 ...]
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "E:/schedule/integrated")
sys.path.insert(0, "E:/schedule/surrogate")
import net_benders as B  # noqa: E402
import net_routing as N  # noqa: E402
from net_benders import _solve  # noqa: E402
from staff_surrogate import branch_cluster, staff_args  # noqa: E402

ROOT = Path("E:/schedule/revision/instances_new")


def main(name):
    data = str(ROOT / f"{name}.xlsx")
    cluster = list(set(branch_cluster(data).values()))[0]
    prob = B.ClusterProblem(cluster, data, staff_args(), 3)
    net = prob.net
    (b, a), riders = next(iter(net.riders.items()))
    assert len(net.riders) == 1, "one branch and one accommodation expected"
    tau, km = net.tau[a, b], net.km[a, b]
    assert 2 * tau <= 60, "a round trip must fit in one hour"
    run = ROOT / "runs" / name
    stem = name
    bres = json.load(open(run / "bd" / f"{stem}_{cluster[:8]}.json"))
    limits = {int(k): v for k, v in bres["limits"].items()}

    best = None
    for bus in (bt for bt in net.buses if bt.capacity >= len(riders)):
        c_rt = bus.per_km * 2 * km + bus.per_min * 2 * tau
        sm = B.StaffModel(prob.args, prob.data)  # w_bus_trip is not part of StaffModel's cost terms
        m = sm.mdl
        for p, expr in sm.levels.items():
            m.add_constraint(expr <= limits[p] + 1e-6)
        y = prob.add_stops(sm)
        m.minimize(sm.j_reg + sm.j_over + m.sum(c_rt * v for v in y.values()) + bus.fixed)
        sol, info = _solve(m, 7200, 1e-9, None, 3)
        print(name, bus.key, "z", sol.objective_value, info.get("status"), flush=True)
        if best is None or sol.objective_value < best[0]:
            best = (sol.objective_value, bus, sol, info, sm, y)
    z_mip, bus, sol, info, sm, y = best
    x = {key for key, var in sm.solver.x.items() if sol.get_value(var) > 0.5}
    vals, plan, z_routed = prob.plan_cost(B.StaffModel(prob.args, prob.data), x)

    res = dict(name=name, cluster=cluster[:8], riders=len(riders), round_trip_min=2 * tau, bus=bus.key.split('|')[-1], bus_capacity=bus.capacity,
               z_star=z_mip, mip_status=info.get("status"), mip_gap=info.get("mip_gap"), best_bound=info.get("best_bound"),
               z_star_routed=z_routed, open_stops=sum(1 for v in y.values() if sol.get_value(v) > 0.5))
    js = glob.glob(str(run / f"{stem}_{cluster[:8]}_s1.json"))
    s1 = glob.glob(str(run / "stage1" / "*_stage1_s1.json"))
    if js and s1:
        ours = json.load(open(js[0]))
        st = json.load(open(s1[0]))
        res.update(z0=ours["z_seq"], lb=st["lb"], z_ours=ours["z"],
                   lb_le_zstar=st["lb"] <= z_mip + 1e-6, zstar_le_ours=z_mip <= ours["z"] + 1e-6,
                   gap_ours_pct=100 * (ours["z"] - st["lb"]) / ours["z"], ours_above_opt_pct=100 * (ours["z"] - z_mip) / z_mip)
    print(res, flush=True)
    json.dump(res, open(ROOT / f"exact_{name}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
