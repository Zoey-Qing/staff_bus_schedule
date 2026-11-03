"""Route a fixed schedule with the time-space model, one cluster at a time. Writes one row per cluster.

Usage: python run_net_routing.py <data.xlsx> <schedule.csv> <out.csv>
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bus_scheduling_routing"))
import net_routing as N
from bus_riders import branch_cluster


def run(data, schedule, out, threads=0, log=False):
    sched = pd.read_csv(schedule)
    events = [(r.Staff, int(r.Start), int(r.End)) for r in sched.itertuples()]
    rows = []
    for c in sorted(set(branch_cluster(data).values())):
        t0 = time.time()
        net = N.Network(c, data)
        plan = N.route_schedule(net, events, threads, log=print if log else None)
        stops = [len(h.seq) for h in plan.trips]
        rows.append(dict(cluster=c[:8], accommodations=len(net.accs), cost=plan.cost, lp=plan.lp,
                         gap=(plan.cost - plan.lp) / plan.cost if plan.cost else 0.0, driving=plan.driving,
                         punctuality=plan.punctuality, empty=plan.empty, fleet=plan.fleet,
                         buses=sum(plan.buses.values()), trips=len(stops),
                         stops_per_trip=sum(stops) / max(1, len(stops)), max_stops=max(stops, default=0),
                         trips_at_5=sum(s == 5 for s in stops), empty_runs=sum(v for _g, v in plan.runs),
                         iterations=plan.iterations, seconds=time.time() - t0))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("schedule")
    p.add_argument("out")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--log", action="store_true")
    a = p.parse_args()
    run(a.data, a.schedule, a.out, a.threads, a.log)
