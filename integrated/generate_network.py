"""Generate larger synthetic networks in the same Excel format as the base dataset.

Each generated branch copies a real branch of the base data as its template: the branch's weekly demand
profile, opening hours, staff (same roles, nationalities, skills and wages, new ids), and that staff's
shift pattern from an optimised base schedule. Geography follows the base network: accommodations are
placed in the base area, and each accommodation's branches are scattered around it at the base
network's branch-to-accommodation distances. Every branch's riders live at the branch's accommodation.

Parameters (documented for the paper's data section):
  --branches N            number of branches
  --per-accommodation K   branches served by one accommodation
  --acc-per-cluster A     accommodations per cluster (a cluster owns its buses)
  --buses-per-5 B         buses per 5 branches in a cluster (base data: 2 = one small + one large)
  --distance-scale F      multiply branch-to-accommodation distances by F (default 1)

Writes <out>.xlsx (all sheets the code reads) and <out>_schedule.csv (the cloned staff schedule).

Usage:
  python generate_network.py --branches 100 --per-accommodation 20 --seed 1 --out data_gen/net100_k20_s1
"""
import argparse
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from instance import DATA_FILE, haversine_km

ROOT = Path(__file__).resolve().parents[1]
BASE_SCHEDULE = ROOT / "revision" / "rerun_20261024" / "schedule_result.csv"


def _uid(rng):
    return str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))))


def generate(n_branches, per_acc, acc_per_cluster, buses_per_5, seed, base_xlsx=DATA_FILE, base_schedule=BASE_SCHEDULE,
             distance_scale=1.0):
    rng = np.random.default_rng(seed)
    x = pd.ExcelFile(base_xlsx)
    br = x.parse("branches")
    br.columns = [c.strip() for c in br.columns]
    lon = "longitude" if "longitude" in br.columns else "longitute"
    acc = x.parse("accommodation")
    staff = x.parse("staff")
    demand = x.parse("staff_demand")
    ops = x.parse("operation_hours", header=1)
    bus_info = x.parse("bus_info")
    vehicles = x.parse("vehicles")
    sched = pd.read_csv(base_schedule)

    # Base geography: area and branch-to-own-accommodation distances.
    acc_pos = {str(a.id): (a.latitude, a.longitude) for a in acc.itertuples()}
    home = staff.groupby("branch_id")["accommodation_id"].agg(lambda s: s.mode().iat[0]).to_dict()
    d_home = [haversine_km(r.latitude, getattr(r, lon), *acc_pos[str(home[r.branch_id])]) for r in br.itertuples()]
    lat_lo, lat_hi = br.latitude.min(), br.latitude.max()
    lon_lo, lon_hi = br[lon].min(), br[lon].max()
    km_per_deg_lat, km_per_deg_lon = 111.0, 111.0 * math.cos(math.radians((lat_lo + lat_hi) / 2))

    n_acc = math.ceil(n_branches / per_acc)
    n_cluster = math.ceil(n_acc / acc_per_cluster)
    clusters = [_uid(rng) for _ in range(n_cluster)]
    accs = []
    for j in range(n_acc):
        accs.append({"id": _uid(rng), "name": f"Staff Accommodation G{j + 1}", "address": f"Generated housing {j + 1}, Riyadh",
                     "latitude": round(rng.uniform(lat_lo, lat_hi), 6), "longitude": round(rng.uniform(lon_lo, lon_hi), 6),
                     "capacity": 0, "cluster_id": clusters[j // acc_per_cluster], "active": True})

    templates = br.branch_id.tolist()
    rows_br, rows_staff, rows_dem, rows_ops, rows_sched = [], [], [], [], []
    next_staff = 1
    for k in range(n_branches):
        a = accs[k // per_acc]
        bid = f"_gen_{k + 1:04d}"
        dist = float(rng.choice(d_home)) * rng.uniform(0.8, 1.2) * distance_scale
        ang = rng.uniform(0, 2 * math.pi)
        rows_br.append({"branch_id": bid,
                        "latitude": round(a["latitude"] + dist * math.sin(ang) / km_per_deg_lat, 6),
                        "longitude": round(a["longitude"] + dist * math.cos(ang) / km_per_deg_lon, 6),
                        "cluster_id": a["cluster_id"], "template_branch": None})
        tpl = templates[int(rng.integers(len(templates)))]
        rows_br[-1]["template_branch"] = tpl
        for r in demand[demand.branch_id == tpl].itertuples(index=False):
            rows_dem.append({"staff_demand": r.staff_demand, "branch_id": bid, "day": r.day, "hour": r.hour})
        for r in ops[ops.branch_id == tpl].itertuples(index=False):
            rows_ops.append({"branch_id": bid, "week_day": r.week_day, "start_time": r.start_time, "end_time": r.end_time})
        id_map = {}
        for r in staff[staff.branch_id == tpl].to_dict("records"):
            new = dict(r)
            id_map[r["staff_id"]] = next_staff
            new.update(staff_id=next_staff, branch_id=bid, name=f"Staff {next_staff}", accommodation_id=a["id"])
            rows_staff.append(new)
            next_staff += 1
        for r in sched[sched.Staff.isin(id_map)].to_dict("records"):
            rows_sched.append({**r, "Staff": id_map[r["Staff"]]})

    branches_df = pd.DataFrame(rows_br)
    staff_df = pd.DataFrame(rows_staff)
    riders_per_acc = staff_df[~staff_df.Nationality.astype(str).str.lower().isin(["saudi", "saudi arabia"])] \
        .groupby("accommodation_id").size()
    for a in accs:
        a["capacity"] = int(riders_per_acc.get(a["id"], 0))

    bus_rows, veh_rows = [], []
    by_kind = {k: bus_info[vehicles.vehicle_type == k] for k in ("Small Bus", "Large Bus")}
    model_of = dict(zip(vehicles.vehicle_type, vehicles.cost_model_id))
    for c in clusters:
        n_br = int((branches_df.cluster_id == c).sum())
        n_bus = max(2, round(buses_per_5 * n_br / 5))
        for j in range(n_bus):
            kind = "Small Bus" if j % 2 == 0 else "Large Bus"
            ref = by_kind[kind].iloc[int(rng.integers(len(by_kind[kind])))]
            bid = f"_genbus_{len(bus_rows) + 1:04d}"
            bus_rows.append({"id": bid, "capacity": int(ref.capacity), "cluster_id": c, "fixed_cost": float(ref.fixed_cost),
                             "cost_per_min": float(ref.cost_per_min), "cost_per_km": float(ref.cost_per_km)})
            veh_rows.append({"id": bid, "license_plate": f"GEN{len(bus_rows):04d}", "capacity": int(ref.capacity),
                             "vehicle_type": kind, "cost_model_id": model_of[kind], "status": "available",
                             "cluster_id": c, "branch_id": None, "active": True})

    sheets = {
        "branches": branches_df.drop(columns=["template_branch"]),
        "accommodation": pd.DataFrame(accs),
        "clusters": pd.DataFrame([{"id": c, "name": f"Cluster G{i + 1}",
                                   "accommodation_id": next(a["id"] for a in accs if a["cluster_id"] == c)}
                                  for i, c in enumerate(clusters)]),
        "constraints": x.parse("constraints"),
        "bus_schedules": pd.DataFrame(columns=x.parse("bus_schedules").columns),
        "bus_info": pd.DataFrame(bus_rows),
        "staff": staff_df,
        "staff_demand": pd.DataFrame(rows_dem),
        "vehicle_cost_model": x.parse("vehicle_cost_model"),
        "vehicles": pd.DataFrame(veh_rows),
    }
    return sheets, pd.DataFrame(rows_ops), pd.DataFrame(rows_sched), branches_df[["branch_id", "template_branch"]]


def write(out, sheets, ops, schedule, templates):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out.with_suffix(".xlsx"), engine="openpyxl") as w:
        # operation_hours has a note in its first row; the staff loader reads it with header=1.
        pd.DataFrame([["Week day goes from 1 (Sunday) to 7 (Saturday)", None, None, None]]) \
            .to_excel(w, sheet_name="operation_hours", index=False, header=False)
        ops.to_excel(w, sheet_name="operation_hours", index=False, startrow=1)
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    schedule.to_csv(f"{out}_schedule.csv", index=False)
    templates.to_csv(f"{out}_templates.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--branches", type=int, required=True)
    p.add_argument("--per-accommodation", type=int, default=20)
    p.add_argument("--acc-per-cluster", type=int, default=1)
    p.add_argument("--buses-per-5", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--distance-scale", type=float, default=1.0,
                   help="multiply branch-to-accommodation distances (scenario: branches far from housing)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sheets, ops, schedule, templates = generate(a.branches, a.per_accommodation, a.acc_per_cluster, a.buses_per_5, a.seed,
                                                distance_scale=a.distance_scale)
    write(a.out, sheets, ops, schedule, templates)
    print(f"{a.branches} branches, {len(sheets['accommodation'])} accommodations, {len(sheets['clusters'])} clusters, "
          f"{len(sheets['staff'])} staff, {len(sheets['vehicles'])} buses, {len(schedule)} shifts -> {a.out}.xlsx")
