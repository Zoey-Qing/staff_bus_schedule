"""Describe each cluster as a data set: workload, unevenness of demand over time and branches, distance from housing."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bus_scheduling_routing"))
import instance as I
import net_benders as B
import net_routing as N
from bus_riders import branch_cluster


def describe(data_file, args):
    rows = []
    groups = {}
    for b, c in branch_cluster(data_file).items():
        groups.setdefault(c, []).append(b)
    for c, bs in sorted(groups.items()):
        args.data_file = str(data_file)
        d = I.BranchSubsetLoader(args, sorted(bs)).load_and_preprocess()
        net = N.Network(c, data_file)
        by_hour = np.array([sum(d.branch_demand.get((b, t), 0) for b in bs) for t in range(168)], dtype=float)
        by_branch = np.array([sum(d.branch_demand.get((b, t), 0) for t in range(168)) for b in bs], dtype=float)
        direct = [net.tau[a, b] for a, bb in net.branches_of.items() for b in bb]
        rows.append(dict(data=Path(data_file).stem, cluster=c[:8], branches=len(bs), staff=len(d.staff_ids), riders=len(net.home),
                         accommodations=len(net.accs), demand_hours=by_hour.sum(), hours_per_staff=by_hour.sum() / len(d.staff_ids),
                         peak_to_mean=by_hour.max() / by_hour[by_hour > 0].mean(), branch_cv=by_branch.std() / by_branch.mean(),
                         mean_direct_minutes=float(np.mean(direct))))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("data", nargs="+")
    a = p.parse_args()
    args = B.staff_args()
    pd.DataFrame([r for f in a.data for r in describe(f, args)]).to_csv(a.out, index=False)
