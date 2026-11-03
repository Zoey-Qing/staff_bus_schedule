"""E7-E11: route the sequential schedules of a batch under each changed routing assumption. One row per setting and cluster.

Usage: python run_routing_switches.py <result_dir> <out.csv> <data.xlsx> [<data.xlsx> ...]
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

SETTINGS = {
    "base": {},
    "window15": {"NET_WINDOW": "15"},
    "window60": {"NET_WINDOW": "60"},
    "speed30": {"NET_SPEED": "30"},
    "speed50": {"NET_SPEED": "50"},
    "fixed_x4": {"NET_FIXED_SCALE": "4"},
    "theta_off": {"NET_THETA": "0"},
    "step5": {"NET_STEP": "5"},
    "step10": {"NET_STEP": "10"},
    "stationed": {"NET_EMPTY": "0"},
}


def child(result_dir, data, out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bus_scheduling_routing"))
    import net_routing as N
    from bus_riders import branch_cluster

    rows = []
    for c in sorted(set(branch_cluster(data).values())):
        sched = pd.read_csv(Path(result_dir) / f"{Path(data).stem}_{c[:8]}_x0.csv")
        net = N.Network(c, data)
        events = [(r.Staff, int(r.Start), int(r.End)) for r in sched.itertuples()]
        t0 = time.time()
        plan = N.route_schedule(net, events, 3)
        stops = [len(h.seq) for h in plan.trips]
        rows.append(dict(data=Path(data).stem, cluster=c[:8], accommodations=len(net.accs), cost=plan.cost, lp=plan.lp, driving=plan.driving,
                         punctuality=plan.punctuality, empty=plan.empty, fleet=plan.fleet, buses=sum(plan.buses.values()), trips=len(stops),
                         stops_per_trip=sum(stops) / max(1, len(stops)), seconds=time.time() - t0))
    with open(out, "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(*sys.argv[2:])
        sys.exit(0)
    result_dir, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    all_rows, running = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            for name, env in SETTINGS.items():
                path = str(Path(tmp) / f"{Path(f).stem}_{name}.json")
                while len(running) >= 5:
                    for item in list(running):
                        if item[0].poll() is not None:
                            running.remove(item)
                    time.sleep(1)
                proc = subprocess.Popen([sys.executable, __file__, "--child", result_dir, str(Path(f).resolve()), path], env={**os.environ, **env})
                running.append((proc, name, path))
                all_rows.append((proc, name, path))
        for proc, name, path in all_rows:
            assert proc.wait() == 0, (name, path)
        rows = []
        for proc, name, path in all_rows:
            rows += [dict(setting=name, **r) for r in json.load(open(path))]
    d = pd.DataFrame(rows)
    d = d[(d.setting != "stationed") | (d.accommodations > 1)]
    d.to_csv(out, index=False)
    print("done", len(d))
