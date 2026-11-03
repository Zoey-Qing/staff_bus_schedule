"""Instances for the integrated staff-scheduling and commuter-bus model.

An instance is a set of branches over the full one-week horizon. The staff side is loaded with the
same DataLoader the paper's staff model uses (staff_scheduling/data_loader.py) and filtered to the
chosen branches, so the staff constraints are exactly those of the paper. The transport side holds
the riders' accommodations, distances, vehicle types, available buses, and the complete set of
candidate routes (every ordering of every subset of an accommodation's branches, up to max_stops).
"""
import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "staff_scheduling"))
sys.path.insert(0, str(ROOT / "bus_scheduling_routing"))

from data_loader import DataLoader  # noqa: E402  (staff_scheduling)
from bus_riders import accommodation_cluster, allowed_accommodations, branch_cluster  # noqa: E402  (bus_scheduling_routing)

DATA_FILE = ROOT / "staff_scheduling" / "Berkeley_Mock_Data_25_Stores.xlsx"
TRAVEL_SPEED_KMH = 40.0  # same as the routing notebook
FIXED_COST_PERIOD_FRACTION = 1.0 / 4.0  # monthly fixed cost -> one week
import os

# extra buses beyond the available fleet (decision D5); BUS_RENTAL_FACTOR=1 gives the benchmarks' unlimited-fleet rule
RENTAL_FACTOR = float(os.environ.get("BUS_RENTAL_FACTOR", "1.5"))
WEEK_HOURS = 168


class BranchSubsetLoader(DataLoader):
    """The paper's staff DataLoader, restricted to a chosen set of branches."""

    def __init__(self, args, branches):
        super().__init__(args, target_cluster=None)
        self.target_branches = set(branches)

    def load_and_preprocess(self):
        super().load_and_preprocess()
        keep = self.target_branches
        missing = keep - set(self.branch_ids)
        if missing:
            raise ValueError(f"Unknown branches: {sorted(missing)}")
        staff = [i for i in self.staff_ids if self.staff_branch_map.get(i) in keep]
        staff_set = set(staff)
        self.staff_ids = staff
        self.manager_ids = [i for i in self.manager_ids if i in staff_set]
        self.non_new_staff_ids = [i for i in self.non_new_staff_ids if i in staff_set]
        self.non_saudi_staff_ids = [i for i in self.non_saudi_staff_ids if i in staff_set]
        self.staff_branch_map = {i: b for i, b in self.staff_branch_map.items() if i in staff_set}
        self.staff_cluster_map = {i: c for i, c in self.staff_cluster_map.items() if i in staff_set}
        self.branch_ids = sorted(keep)
        self.branch_cluster_map = {b: c for b, c in self.branch_cluster_map.items() if b in keep}
        self.cluster_ids = sorted(set(self.branch_cluster_map.values()))
        self.branch_demand = {(b, t): d for (b, t), d in self.branch_demand.items() if b in keep}
        self.demand = {t: sum(self.branch_demand.get((b, t), 0) for b in keep) for t in self.time_slots}
        self.operating_hours = [t for t, d in self.demand.items() if d > 0]
        self.branch_operation_limits = {b: v for b, v in self.branch_operation_limits.items() if b in keep}
        return self


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class Route:
    rid: str
    acc: str
    stops: tuple
    distance_km: float
    travel_min: float


@dataclass
class VehicleType:
    key: str  # (cluster, kind) as "cluster|kind"
    cluster: str
    kind: str
    capacity: int
    monthly_fixed: float
    per_km: float
    per_min: float
    available: int

    @property
    def weekly_fixed(self):
        return self.monthly_fixed * FIXED_COST_PERIOD_FRACTION


@dataclass
class Instance:
    name: str
    staff: BranchSubsetLoader
    branches: list
    acc_of_branch: dict  # branch -> accommodations a route serving it may start at (bus_riders rule)
    acc_cluster: dict  # accommodation -> cluster owning the buses based there
    riders_by_branch: dict  # branch -> non-Saudi staff ids
    vehicles: dict  # key -> VehicleType
    routes: list = field(default_factory=list)
    distance_matrix: object = None  # km, pandas DataFrame over branches and accommodations

    def route_cost(self, route, vtype):
        v = self.vehicles[vtype]
        return v.per_km * route.distance_km + v.per_min * route.travel_min

    def vehicle_types_for(self, route):
        cluster = self.acc_cluster[route.acc]
        return [k for k, v in self.vehicles.items() if v.cluster == cluster]


def _vehicle_types(data_file):
    """Vehicle types per (cluster, kind). As in the routing code, a bus's own costs in `bus_info` take
    precedence over the generic `vehicle_cost_model`."""
    vehicles = pd.read_excel(data_file, sheet_name="vehicles")
    models = pd.read_excel(data_file, sheet_name="vehicle_cost_model").set_index("id")
    bus_info = pd.read_excel(data_file, sheet_name="bus_info").set_index("id")
    out = {}
    for _, row in vehicles.iterrows():
        if str(row.get("status", "available")).lower() != "available":
            continue
        kind = "Large Bus" if "large" in str(row["vehicle_type"]).lower() else "Small Bus"
        key = f"{row['cluster_id']}|{kind}"
        m = models.loc[row["cost_model_id"]]
        info = bus_info.loc[row["id"]] if row["id"] in bus_info.index else {}
        pick = lambda col, fallback: float(info[col]) if col in info and pd.notna(info[col]) else float(fallback)
        # Scenario switch: BUS_COST_SCALE multiplies every bus cost (fixed, per km, per minute), e.g. 3 for the
        # "transport matters more" scenarios. Default 1.
        f = float(os.environ.get("BUS_COST_SCALE", "1"))
        if key not in out:
            out[key] = VehicleType(key, str(row["cluster_id"]), kind, int(row["capacity"]),
                                   f * pick("fixed_cost", m["fixed_cost"]), f * pick("cost_per_km", m["cost_per_km"]),
                                   f * pick("cost_per_min", float(m["cost_per_hour"]) / 60.0), 0)
        out[key].available += 1
    # Scenario switch: BUS_OWNED="s,l" sets the owned small and large buses of every cluster (hard-fleet
    # experiments, together with a prohibitive BUS_RENTAL_FACTOR). Default: the fleet in the data.
    owned = os.environ.get("BUS_OWNED")
    if owned:
        n_small, n_large = (int(v) for v in owned.split(","))
        for v in out.values():
            v.available = n_large if v.kind == "Large Bus" else n_small
    return out


def build_instance(branches, args, name=None, max_stops=5, data_file=DATA_FILE):
    args.data_file = str(data_file)
    staff = BranchSubsetLoader(args, branches).load_and_preprocess()
    acc_of_branch = {b: a for b, a in allowed_accommodations(data_file).items() if b in set(branches)}
    acc_cluster = accommodation_cluster(data_file)

    xls = pd.ExcelFile(data_file)
    br = xls.parse("branches")
    br.columns = [c.strip() for c in br.columns]
    lon_col = "longitude" if "longitude" in br.columns else "longitute"
    acc = xls.parse("accommodation")
    coords = {r.branch_id: (r.latitude, getattr(r, lon_col)) for r in br.itertuples()}
    coords.update({str(r.id): (r.latitude, r.longitude) for r in acc.itertuples()})
    dist = lambda u, v: haversine_km(*coords[u], *coords[v])
    ids = sorted(set(branches) | {a for accs in acc_of_branch.values() for a in accs})
    dm = pd.DataFrame([[dist(u, v) for v in ids] for u in ids], index=ids, columns=ids)

    riders = {b: [] for b in branches}
    for i in staff.non_saudi_staff_ids:
        riders[staff.staff_branch_map[i]].append(i)

    routes = []
    by_acc = {}  # accommodation -> branches a route from it may serve
    for b in sorted(branches):
        if riders[b]:
            for a in acc_of_branch[b]:
                by_acc.setdefault(a, []).append(b)
    for a, bs in sorted(by_acc.items()):
        for k in range(1, min(max_stops, len(bs)) + 1):
            for subset in itertools.combinations(bs, k):
                for seq in itertools.permutations(subset):
                    km = dist(a, seq[0]) + sum(dist(seq[j], seq[j + 1]) for j in range(k - 1)) + dist(seq[-1], a)
                    routes.append(Route(f"{a[:4]}:{'-'.join(s[-4:] for s in seq)}", a, seq, km,
                                        60.0 * km / TRAVEL_SPEED_KMH))

    return Instance(name or f"{len(branches)}br", staff, sorted(branches), acc_of_branch, acc_cluster,
                    riders, _vehicle_types(data_file), routes, dm)


def cluster_groups(data_file=DATA_FILE):
    """cluster -> sorted list of its branches. Clusters are independent (own buses, own accommodations),
    so the integrated problem separates exactly by cluster."""
    groups = {}
    for b, c in branch_cluster(data_file).items():
        groups.setdefault(c, []).append(b)
    return {c: sorted(bs) for c, bs in groups.items()}
