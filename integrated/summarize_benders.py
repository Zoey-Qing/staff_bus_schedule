"""One row per cluster from the json files of net_benders.

Usage: python summarize_benders.py <result_dir> <out.csv>
"""
import json
import sys
from pathlib import Path

import pandas as pd


def row(f):
    r = json.load(open(f))
    data = f.stem.rsplit("_", 1)[0]
    plan, seq = r["plan"], r["seq"]
    gap_at_last = r["history"][-1] if r["history"] else {}
    return dict(data=data, cluster=r["cluster"], z_seq=r["z_seq"], z_unc=r["z_unc"], z=r["z"], z_benders=r["z_benders"], lb=r["lb"], gap=r["gap"],
                saving=r["saving"], saving_vs_best_baseline=r["saving_vs_best_baseline"],
                saving_amount=r["z_seq"] - r["z"], wages_change=r["wages"] - r["seq_wages"], overtime_change=r["overtime"] - r["seq_overtime"],
                transport_seq=seq["cost"], transport=plan["cost"], transport_change=plan["cost"] - seq["cost"],
                transport_share_seq=seq["cost"] / r["z_seq"], trips_seq=seq["trips"], trips=plan["trips"], stops_per_trip=plan["stops_per_trip"],
                buses_seq=seq["buses"], buses=plan["buses"], empty=plan["empty"], route_gap=plan["route_gap"],
                levels_seq=str(r["seq_levels"]), levels=str(r["levels"]), rounds=r["rounds"], best_round=r["best_round"],
                master_gap_last=gap_at_last.get("master_gap"), status_last=gap_at_last.get("master_status"),
                cut_violation=r["cut_violation"], z_check=r["z_check"], hours=r["seconds"] / 3600)


if __name__ == "__main__":
    rows = [row(f) for f in sorted(Path(sys.argv[1]).glob("*.json"))]
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
