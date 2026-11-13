"""Cost structure of one cluster: where the money is and how much any method could save.

For x0 (sequential), M2 and M5-eq (compare_all/) this computes wages, overtime, transport and the service levels, and the
big-M penalty sum_l M_l V_l that the sequential objective carried (M = 1000, 100, 10, 1; not part of Z). It also solves
the staff model alone under the service limits, min J_reg + J_over (no buses), and records its value and best bound:
the bound is a floor on the labour cost of every feasible schedule.

Usage: python cost_structure.py <cluster8> <out.json> [--time 900] [--threads 3]
"""
import argparse, glob, json, sys
from pathlib import Path
sys.path.insert(0, "E:/schedule/integrated")
import net_benders as B
from model import _solve

R = Path("E:/schedule/revision/results_new/original")
D = "E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx"
S = "Berkeley_Mock_Data_25_Stores_original_unbalanced"
M = {1: 1000, 2: 100, 3: 10, 4: 1}

p = argparse.ArgumentParser(); p.add_argument("cluster"); p.add_argument("out")
p.add_argument("--time", type=float, default=900); p.add_argument("--threads", type=int, default=3)
a = p.parse_args()
c = a.cluster
prob = B.ClusterProblem(next(k for k in set(B.branch_cluster(D).values()) if k.startswith(c)), D, B.staff_args(), a.threads)
ev = B.StaffModel(prob.args, prob.data)
res = dict(cluster=c, staff=len(prob.data.staff_ids), branches=len(prob.data.branch_ids))
scheds = {"x0": R / "benders" / f"{S}_{c}_x0.csv",
          "M2": next(iter(glob.glob(str(R / "compare_all" / c / "M2" / "*_schedule.csv")))),
          "M5eq": R / "compare_all" / c / f"m5eq_{S}_{c}_s1.schedule.csv"}
for name, f in scheds.items():
    x = B.load_schedule(f, prob.data.staff_ids)
    vals, plan, z = prob.plan_cost(ev, x)
    res[name] = dict(z=z, j_reg=vals["j_reg"], j_over=vals["j_over"], transport=plan.cost, levels=vals["levels"],
                     big_m_penalty=sum(M[l] * v for l, v in vals["levels"].items()), shifts=len(x),
                     paid_hours=sum(prob.shift[s]["duration"] for _i, s in x))
limits = res["x0"]["levels"]
prob.args.w_bus_trip = 0.0
sm = B.StaffModel(prob.args, prob.data)
for l, e in sm.levels.items():
    sm.mdl.add_constraint(e <= limits[l] + 1e-6)
sm.mdl.minimize(sm.j_reg + sm.j_over)
st = sm.mdl.new_solution()
for key in B.load_schedule(scheds["x0"], prob.data.staff_ids):
    st.add_var_value(sm.solver.x[key], 1)
sm.mdl.add_mip_start(st, write_level=1)
sol, info = _solve(sm.mdl, a.time, 1e-6, None, a.threads)
res["labour_min"] = dict(value=sol.objective_value if sol else None, bound=info["best_bound"], status=info["status"],
                         gap=info["mip_gap"])
json.dump(res, open(a.out, "w"), indent=1, default=float)
print("DONE", json.dumps(res, default=float))
