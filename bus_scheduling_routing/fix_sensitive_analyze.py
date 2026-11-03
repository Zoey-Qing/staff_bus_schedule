import time
import math
import itertools
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from docplex.mp.model import Model

from bus_riders import keep_bus_riders, serves_own_accommodation

warnings.filterwarnings("ignore")

# ============================================================
# 0. Baseline parameters
# ============================================================

BASE_PARAMS = {
    "TRAVEL_SPEED": 40,
    "BOARDING_TIME": 0.5,
    "ALIGHTING_TIME": 0.3,
    "EARLIEST_DEPARTURE_SHIFT_MIN": -30.0,
    "PENALTY_BOARD_LATE": 20,
    "PENALTY_ALIGHT_LATE": 20,
    "REWARD_ALIGHT_EARLY": 2,
    "TAU_SOFT_WINDOW_MIN": 5.0,
    "TAU_VIOLATION_PENALTY": 50.0,
    "TAU_VIOLATION_PENALTY_MULTIPLIER": 4.0,
    "MIN_BREAK_BETWEEN_SHIFTS": 60,
    "NUMERICAL_EPS": 1e-9,
    "DEMAND_EPS": 1e-6,
    "CPLEX_LOG_OUTPUT": False,
    "WEEK_HOURS": 168,
    "TIME_SHIFT_HOURS": 0,
    "TOP_K_PERIODS": None,
    "max_stops": 20,
    "max_per_combo": 10000,
    "max_routes": 70000,
}

# one-factor-at-a-time sensitivity analysis
SENSITIVITY_CONFIG = {
    "TRAVEL_SPEED": [30, 50, 60],
    "BOARDING_TIME": [0.3, 0.8],
    "ALIGHTING_TIME": [0.2, 0.5],
    "EARLIEST_DEPARTURE_SHIFT_MIN": [-60.0, -30.0, -10.0],
    "PENALTY_BOARD_LATE": [10, 40],
    "PENALTY_ALIGHT_LATE": [10, 40],
    "REWARD_ALIGHT_EARLY": [0, 5],
    "TAU_SOFT_WINDOW_MIN": [2.0, 8.0],
    "TAU_VIOLATION_PENALTY": [20.0, 80.0],
    "TAU_VIOLATION_PENALTY_MULTIPLIER": [2.0, 6.0],
    "max_stops": [3, 4, 5],
    # "max_routes": [30000, 100000],
}

DATA_FILE = "data/Berkeley_Mock_Data.xlsx"
SCHEDULE_FILE = "data/schedule_result.csv"


# ============================================================
# 1. Data loading
# ============================================================

def _build_counts_from_demand_file(demand_file, params):
    demand_df = pd.read_csv(demand_file)
    required = {"time", "branch_id", "board", "alight"}
    missing = required.difference(demand_df.columns)
    if missing:
        raise ValueError(
            f"Demand file {demand_file} is missing required columns: {sorted(missing)}"
        )

    demand_df = demand_df.copy()
    demand_df["time"] = (
        demand_df["time"].astype(int) - params["TIME_SHIFT_HOURS"]
    ) % params["WEEK_HOURS"]
    demand_df["branch_id"] = demand_df["branch_id"].astype(str)
    demand_df["board"] = pd.to_numeric(demand_df["board"], errors="coerce").fillna(0.0)
    demand_df["alight"] = pd.to_numeric(demand_df["alight"], errors="coerce").fillna(0.0)

    start_counts = (
        demand_df.loc[demand_df["board"] > 0, ["time", "branch_id", "board"]]
        .rename(columns={"time": "Start", "board": "staff_starting_count"})
        .copy()
    )
    start_counts["staff_starting_count"] = (
        start_counts["staff_starting_count"].round().astype(int)
    )
    start_counts = (
        start_counts.groupby(["Start", "branch_id"], as_index=False)["staff_starting_count"]
        .sum()
        .sort_values(["Start", "branch_id"])
        .reset_index(drop=True)
    )

    end_counts = (
        demand_df.loc[demand_df["alight"] > 0, ["time", "branch_id", "alight"]]
        .rename(columns={"time": "End", "alight": "staff_ending_count"})
        .copy()
    )
    end_counts["staff_ending_count"] = (
        end_counts["staff_ending_count"].round().astype(int)
    )
    end_counts = (
        end_counts.groupby(["End", "branch_id"], as_index=False)["staff_ending_count"]
        .sum()
        .sort_values(["End", "branch_id"])
        .reset_index(drop=True)
    )

    return start_counts, end_counts


def load_data(data_file=DATA_FILE, schedule_file=SCHEDULE_FILE, demand_file=None, params=None):
    if params is None:
        params = BASE_PARAMS

    if demand_file:
        start_counts, end_counts = _build_counts_from_demand_file(demand_file, params)
    else:
        staff_schedule = keep_bus_riders(pd.read_csv(schedule_file), data_file)
        staff_schedule["Start"] = (
            staff_schedule["Start"].astype(int) - params["TIME_SHIFT_HOURS"]
        ) % params["WEEK_HOURS"]
        staff_schedule["End"] = (
            staff_schedule["End"].astype(int) - params["TIME_SHIFT_HOURS"]
        ) % params["WEEK_HOURS"]

        sample_data_staff = pd.read_excel(data_file, sheet_name="staff")
        merged = pd.merge(staff_schedule, sample_data_staff, left_on="Staff", right_on="staff_id")

        start_counts = (
            merged.groupby(["Start", "branch_id"])
            .size()
            .reset_index(name="staff_starting_count")
        )
        end_counts = (
            merged.groupby(["End", "branch_id"])
            .size()
            .reset_index(name="staff_ending_count")
        )

    sample_data_dist = pd.read_excel(data_file, sheet_name="branches")

    acc_raw = pd.read_excel(data_file, sheet_name="accommodation")
    acc = acc_raw.copy()
    acc["branch_id"] = acc["id"].astype(str)
    acc = (
        acc[["branch_id", "latitude", "longitude", "cluster_id"]]
        .dropna(subset=["branch_id", "latitude", "longitude"])
        .drop_duplicates(subset=["branch_id"], keep="first")
        .reset_index(drop=True)
    )

    bus_info = pd.read_excel(data_file, sheet_name="bus_info")
    vehicles_df = pd.read_excel(data_file, sheet_name="vehicles")
    vehicle_cost_model = pd.read_excel(data_file, sheet_name="vehicle_cost_model")
    clusters_df = pd.read_excel(data_file, sheet_name="clusters")

    return (
        start_counts,
        end_counts,
        sample_data_dist,
        acc,
        bus_info,
        vehicles_df,
        vehicle_cost_model,
        clusters_df,
    )


# ============================================================
# 2. Shared preprocessing
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_shared_context(start_counts, end_counts, sample_data_dist,
                         acc, bus_info, vehicles_df, vehicle_cost_model,
                         clusters_df):
    all_locs = pd.concat([sample_data_dist, acc], ignore_index=True)
    loc_coords = all_locs.set_index("branch_id")[["latitude", "longitude"]]
    loc_ids = loc_coords.index.tolist()
    n = len(loc_ids)

    distance_matrix = pd.DataFrame(np.zeros((n, n)), index=loc_ids, columns=loc_ids)
    for a in loc_ids:
        la, lo = loc_coords.loc[a]
        for b in loc_ids:
            if a == b:
                continue
            lb, lb2 = loc_coords.loc[b]
            distance_matrix.loc[a, b] = haversine_km(la, lo, lb, lb2)

    def normalize_vehicle_type(x):
        s = str(x).strip().lower()
        if "small" in s:
            return "Small Bus"
        if "large" in s:
            return "Large Bus"
        return None

    cost_model_by_id = vehicle_cost_model.set_index("id").to_dict("index")
    bus_info_by_id = bus_info.set_index("id").to_dict("index")

    VEHICLE_TYPES = {}
    for _, row in vehicles_df.iterrows():
        is_active = bool(row.get("active", True))
        status = str(row.get("status", "available")).lower()
        if not is_active or status != "available":
            continue

        vt = normalize_vehicle_type(row.get("vehicle_type"))
        if vt is None:
            continue

        cm = cost_model_by_id.get(row.get("cost_model_id"))
        if cm is None:
            continue

        cap = row.get("capacity")
        if pd.isna(cap):
            cap = bus_info_by_id.get(row.get("id"), {}).get("capacity")
        if pd.isna(cap) or cap is None:
            continue

        cid = row.get("cluster_id")
        if pd.isna(cid):
            cid = bus_info_by_id.get(row.get("id"), {}).get("cluster_id")
        if pd.isna(cid) or cid is None:
            continue

        bus_info = bus_info_by_id.get(row.get("id"), {})
        fixed_cost = bus_info.get("fixed_cost", None)
        if pd.isna(fixed_cost):
            fixed_cost = cm.get("fixed_cost", 0.0)

        cost_per_km = bus_info.get("cost_per_km", None)
        if pd.isna(cost_per_km):
            cost_per_km = cm.get("cost_per_km", 0.0)

        cost_per_min = bus_info.get("cost_per_min", None)
        if pd.isna(cost_per_min):
            cost_per_hour = cm.get("cost_per_hour", None)
            cost_per_min = 0.0 if pd.isna(cost_per_hour) else float(cost_per_hour) / 60.0

        key = f"{cid}__{vt.replace(' ', '_')}"
        if key not in VEHICLE_TYPES:
            VEHICLE_TYPES[key] = {
                "capacity": int(cap),
                "fixed_cost": float(fixed_cost),
                "cost_per_km": float(cost_per_km),
                "cost_per_min": float(cost_per_min),
                "cluster_id": cid,
                "vehicle_type": vt,
            }

    BRANCH_CLUSTER = dict(zip(sample_data_dist["branch_id"], sample_data_dist["cluster_id"]))
    CLUSTERS = sorted(sample_data_dist["cluster_id"].dropna().unique().tolist())
    BRANCHES_BY_CLUSTER = {
        c: sorted([b for b, cc in BRANCH_CLUSTER.items() if cc == c])
        for c in CLUSTERS
    }

    acc_by_cluster = (
        acc.groupby("cluster_id")["branch_id"]
        .apply(lambda s: sorted({str(x) for x in s if pd.notna(x)}))
        .to_dict()
    )
    DEPOTS_BY_CLUSTER = {c: acc_by_cluster.get(c, []) for c in CLUSTERS}

    demand = {}
    branches = sorted(set(start_counts["branch_id"]).union(set(end_counts["branch_id"])))
    horizon = max(int(start_counts["Start"].max()), int(end_counts["End"].max())) + 1

    for t in range(horizon):
        for b in branches:
            demand[(t, b)] = {"board": 0, "alight": 0}

    for row in start_counts.itertuples(index=False):
        demand[(int(row.Start), row.branch_id)]["alight"] = int(row.staff_starting_count)  # dropped off

    for row in end_counts.itertuples(index=False):
        demand[(int(row.End), row.branch_id)]["board"] = int(row.staff_ending_count)  # picked up

    return distance_matrix, VEHICLE_TYPES, CLUSTERS, BRANCHES_BY_CLUSTER, DEPOTS_BY_CLUSTER, demand


# ============================================================
# 3. Route class
# ============================================================

class Route:
    def __init__(self, route_id, stops, distance_matrix, time_slot, depot, params):
        self.route_id = route_id
        self.stops = stops
        self.time_slot = time_slot
        self.depot = depot
        self.params = params
        self.distance = self.calculate_distance(distance_matrix)
        self.time_feasible = False
        self.best_departure_shift = 0.0
        self.tau_by_stop = {}
        self.time_cost = 0.0

    def calculate_distance(self, distance_matrix):
        if len(self.stops) == 0:
            return 0.0
        total = float(distance_matrix.at[self.depot, self.stops[0]])
        for i in range(len(self.stops) - 1):
            total += float(distance_matrix.at[self.stops[i], self.stops[i + 1]])
        total += float(distance_matrix.at[self.stops[-1], self.depot])
        return total

    def _travel_minutes(self, km):
        return 60.0 * float(km) / float(self.params["TRAVEL_SPEED"])

    def travel_minutes(self):
        return self._travel_minutes(self.distance)

    def _arrival_offsets(self, demand_data, distance_matrix):
        offsets = {}
        if not self.stops:
            return offsets

        elapsed = self._travel_minutes(float(distance_matrix.at[self.depot, self.stops[0]]))
        offsets[self.stops[0]] = elapsed

        for i in range(len(self.stops) - 1):
            cur, nxt = self.stops[i], self.stops[i + 1]
            d = demand_data.get((self.time_slot, cur), {"board": 0, "alight": 0})
            service = (
                d["board"] * self.params["BOARDING_TIME"]
                + d["alight"] * self.params["ALIGHTING_TIME"]
            )
            elapsed += service
            elapsed += self._travel_minutes(float(distance_matrix.at[cur, nxt]))
            offsets[nxt] = elapsed

        return offsets

    def calculate_load_profile(self, demand_data):
        changes = []
        for b in self.stops:
            d = demand_data.get((self.time_slot, b), {"board": 0, "alight": 0})
            changes.append(d["board"] - d["alight"])

        cum = 0
        min_cum = 0
        for c in changes:
            cum += c
            min_cum = min(min_cum, cum)

        init_load = sum(demand_data.get((self.time_slot, b), {"board": 0, "alight": 0})["alight"]
                        for b in self.stops)  # physical rule: the bus leaves with every drop-off rider on board

        load_profile = []
        cur = init_load
        for c in changes:
            cur += c
            load_profile.append(cur)

        return init_load, load_profile

    def is_capacity_feasible(self, vehicle_capacity, demand_data):
        # Riders must return to their own accommodation.
        if not serves_own_accommodation(self.depot, self.stops):
            return False
        init_load, load_profile = self.calculate_load_profile(demand_data)
        if init_load > vehicle_capacity:
            return False
        return all(0 <= load <= vehicle_capacity for load in load_profile)

    def evaluate_time_feasibility(self, demand_data, distance_matrix):
        offsets = self._arrival_offsets(demand_data, distance_matrix)
        lower = float(self.params["EARLIEST_DEPARTURE_SHIFT_MIN"])
        tol = max(0.0, float(self.params["TAU_SOFT_WINDOW_MIN"]))
        candidates = [lower]

        for b in self.stops:
            for bp in (-offsets[b] - tol, -offsets[b], -offsets[b] + tol):
                if bp >= lower - 1e-9:
                    candidates.append(float(bp))

        def violation_penalty(delta):
            total = 0.0
            base_rate = max(0.0, float(self.params["TAU_VIOLATION_PENALTY"]))
            excess_rate = base_rate * max(1.0, float(self.params["TAU_VIOLATION_PENALTY_MULTIPLIER"]))
            for b in self.stops:
                d = demand_data.get((self.time_slot, b), {"board": 0, "alight": 0})
                tau = delta + offsets[b]
                if d["board"] > 0:
                    violation_board = max(0.0, -tau)
                    inside = min(violation_board, tol)
                    outside = max(0.0, violation_board - tol)
                    total += base_rate * inside + excess_rate * outside
                if d["alight"] > 0:
                    violation_alight = max(0.0, tau)
                    inside = min(violation_alight, tol)
                    outside = max(0.0, violation_alight - tol)
                    total += base_rate * inside + excess_rate * outside
            return total

        def time_cost(delta):
            total = 0.0
            for b in self.stops:
                tau = delta + offsets[b]
                d = demand_data.get((self.time_slot, b), {"board": 0, "alight": 0})
                total += self.params["PENALTY_BOARD_LATE"] * d["board"] * max(0.0, tau)
                total += self.params["PENALTY_ALIGHT_LATE"] * d["alight"] * max(0.0, tau)
                if d["alight"] > 0:
                    total -= self.params["REWARD_ALIGHT_EARLY"] * d["alight"] * max(0.0, -tau)
            return total + violation_penalty(delta)

        best_delta = min(candidates, key=time_cost)
        self.tau_by_stop = {b: best_delta + offsets[b] for b in self.stops}
        self.time_feasible = True
        self.best_departure_shift = float(best_delta)
        self.time_cost = max(0.0, float(time_cost(best_delta)))
        return True

    def calculate_cost(self, vehicle_type, VEHICLE_TYPES):
        p = VEHICLE_TYPES[vehicle_type]
        return float(
            p["fixed_cost"]
            + p["cost_per_km"] * self.distance
            + p["cost_per_min"] * self.travel_minutes()
            + self.time_cost
        )


# ============================================================
# 4. Route generation
# ============================================================

def generate_routes_for_period(
    time_slot, demand_data, branches_list, distance_matrix, depot_ids, params
):
    routes = []

    active_branches = []
    for b in branches_list:
        d = demand_data.get((time_slot, b), {"board": 0, "alight": 0})
        if d["board"] > 0 or d["alight"] > 0:
            active_branches.append(b)

    if not active_branches:
        return routes

    depots = [str(d) for d in depot_ids if pd.notna(d)]
    depots = list(dict.fromkeys(depots))
    if not depots:
        return routes

    per_depot_max_routes = max(1, int(params["max_routes"]) // len(depots))

    for dep_idx, depot_id in enumerate(depots):
        depot_routes = []
        route_id = 0

        for b in active_branches:
            route = Route(
                f"R_{time_slot}_{dep_idx}_{route_id}",
                [b],
                distance_matrix,
                time_slot,
                depot_id,
                params,
            )
            route.evaluate_time_feasibility(demand_data, distance_matrix)
            depot_routes.append(route)
            route_id += 1
            if len(depot_routes) >= per_depot_max_routes:
                break

        if len(depot_routes) < per_depot_max_routes:
            max_len = min(params["max_stops"], len(active_branches))
            for num_stops in range(2, max_len + 1):
                for combo in itertools.combinations(active_branches, num_stops):
                    for perm in itertools.islice(
                        itertools.permutations(combo), params["max_per_combo"]
                    ):
                        route = Route(
                            f"R_{time_slot}_{dep_idx}_{route_id}",
                            list(perm),
                            distance_matrix,
                            time_slot,
                            depot_id,
                            params,
                        )
                        route.evaluate_time_feasibility(demand_data, distance_matrix)
                        depot_routes.append(route)
                        route_id += 1
                        if len(depot_routes) >= per_depot_max_routes:
                            break
                    if len(depot_routes) >= per_depot_max_routes:
                        break
                if len(depot_routes) >= per_depot_max_routes:
                    break

        routes.extend(depot_routes)

    return routes


def build_all_routes(CLUSTERS, BRANCHES_BY_CLUSTER, DEPOTS_BY_CLUSTER,
                     demand_data, distance_matrix, params):
    selected_periods_by_cluster = {}
    all_routes_by_cluster = {}

    for c in CLUSTERS:
        branches_c = BRANCHES_BY_CLUSTER[c]
        depots_c = DEPOTS_BY_CLUSTER[c]

        periods_with_demand = sorted({
            t for (t, b), d in demand_data.items()
            if b in branches_c and (d["board"] + d["alight"] > 0)
        })

        if params["TOP_K_PERIODS"] is None:
            selected_periods = periods_with_demand
        else:
            demand_totals = []
            for t in periods_with_demand:
                total = sum(
                    demand_data[(t, b)]["board"] + demand_data[(t, b)]["alight"]
                    for b in branches_c
                )
                demand_totals.append((t, total))
            demand_totals.sort(key=lambda x: x[1], reverse=True)
            selected_periods = sorted(
                [t for t, _ in demand_totals[: min(params["TOP_K_PERIODS"], len(demand_totals))]]
            )

        selected_periods_by_cluster[c] = selected_periods
        all_routes_by_cluster[c] = {}

        for t in selected_periods:
            all_routes_by_cluster[c][t] = generate_routes_for_period(
                t, demand_data, branches_c, distance_matrix, depots_c, params
            )

    return all_routes_by_cluster, selected_periods_by_cluster


# ============================================================
# 5. CPLEX strict solver
# ============================================================

def solve_bus_scheduling_strict_cplex(
    all_routes_by_cluster,
    demand_data,
    branches_by_cluster,
    selected_periods_by_cluster,
    vehicle_types,
):
    model = Model(name="Bus_Fleet_Scheduling_Strict_Sensitivity")

    lambda_vars = {}
    route_cost = {}
    route_map = {}

    num_constraints = 0
    num_demand_points = 0
    num_uncovered_points = 0

    for c, periods in selected_periods_by_cluster.items():
        for t in periods:
            for route in all_routes_by_cluster[c][t]:
                route_map[(c, route.route_id, t)] = route
                for v_type, params in vehicle_types.items():
                    v_cluster = params.get("cluster_id", None)
                    if pd.notna(v_cluster) and v_cluster is not None and v_cluster != c:
                        continue

                    if not route.is_capacity_feasible(params["capacity"], demand_data):
                        continue
                    if not route.time_feasible:
                        continue

                    cost_val = float(route.calculate_cost(v_type, vehicle_types))
                    if not np.isfinite(cost_val):
                        continue

                    key = (c, route.route_id, v_type, t)
                    lambda_vars[key] = model.binary_var(name=f"lam_{c}_{route.route_id}_{v_type}_{t}")
                    route_cost[key] = cost_val

    if not lambda_vars:
        return {
            "success": False,
            "status": "no variables",
            "objective": np.nan,
            "num_valid_vars": 0,
            "num_constraints": 0,
            "num_demand_points": 0,
            "num_uncovered_points": 0,
            "selected_routes": [],
        }

    model.minimize(model.sum(route_cost[k] * v for k, v in lambda_vars.items()))

    for c, periods in selected_periods_by_cluster.items():
        for t in periods:
            for b in branches_by_cluster[c]:
                d = demand_data.get((t, b), {"board": 0, "alight": 0})
                if d["board"] + d["alight"] <= 0:
                    continue

                num_demand_points += 1
                terms = []
                for route in all_routes_by_cluster[c][t]:
                    if b not in route.stops:
                        continue
                    for v_type in vehicle_types.keys():
                        key = (c, route.route_id, v_type, t)
                        if key in lambda_vars:
                            terms.append(lambda_vars[key])

                if terms:
                    model.add_constraint(model.sum(terms) == 1)
                    num_constraints += 1
                else:
                    num_uncovered_points += 1

    solve_start = time.perf_counter()
    sol = model.solve(log_output=False)
    solve_time = time.perf_counter() - solve_start

    if sol is None:
        return {
            "success": False,
            "status": model.solve_details.status,
            "objective": np.nan,
            "num_valid_vars": len(lambda_vars),
            "num_constraints": num_constraints,
            "num_demand_points": num_demand_points,
            "num_uncovered_points": num_uncovered_points,
            "selected_routes": [],
            "solve_time_s": solve_time,
        }

    selected_routes = []
    for key, var in lambda_vars.items():
        if var.solution_value > 0.5:
            c, route_id, v_type, t = key
            route = route_map[(c, route_id, t)]
            selected_routes.append({
                "cluster": c,
                "time": t,
                "route_id": route_id,
                "vehicle_type": v_type,
                "bus_kind": vehicle_types[v_type].get("vehicle_type"),
                "stops": route.stops,
                "distance": route.distance,
                "cost": route_cost[key],
            })

    return {
        "success": True,
        "status": "optimal/feasible",
        "objective": float(sol.objective_value),
        "num_valid_vars": len(lambda_vars),
        "num_constraints": num_constraints,
        "num_demand_points": num_demand_points,
        "num_uncovered_points": num_uncovered_points,
        "selected_routes": selected_routes,
        "solve_time_s": solve_time,
    }


# ============================================================
# 6. Run one scenario
# ============================================================

def run_one_scenario(params, scenario_name):
    (
        start_counts,
        end_counts,
        sample_data_dist,
        acc,
        bus_info,
        vehicles_df,
        vehicle_cost_model,
        clusters_df,
    ) = load_data(params=params)

    (
        distance_matrix,
        VEHICLE_TYPES,
        CLUSTERS,
        BRANCHES_BY_CLUSTER,
        DEPOTS_BY_CLUSTER,
        demand_data,
    ) = build_shared_context(
        start_counts,
        end_counts,
        sample_data_dist,
        acc,
        bus_info,
        vehicles_df,
        vehicle_cost_model,
        clusters_df,
    )

    rg_start = time.perf_counter()
    all_routes_by_cluster, selected_periods_by_cluster = build_all_routes(
        CLUSTERS,
        BRANCHES_BY_CLUSTER,
        DEPOTS_BY_CLUSTER,
        demand_data,
        distance_matrix,
        params,
    )
    route_generation_time = time.perf_counter() - rg_start

    num_candidate_routes = int(
        sum(len(v) for cluster_map in all_routes_by_cluster.values() for v in cluster_map.values())
    )

    total_start = time.perf_counter()
    result = solve_bus_scheduling_strict_cplex(
        all_routes_by_cluster=all_routes_by_cluster,
        demand_data=demand_data,
        branches_by_cluster=BRANCHES_BY_CLUSTER,
        selected_periods_by_cluster=selected_periods_by_cluster,
        vehicle_types=VEHICLE_TYPES,
    )
    total_runtime = time.perf_counter() - total_start + route_generation_time

    selected_routes = result["selected_routes"]
    if selected_routes:
        df_routes = pd.DataFrame(selected_routes)
        num_selected_routes = len(df_routes)
        total_distance = float(df_routes["distance"].sum())
        avg_route_distance = float(df_routes["distance"].mean())
        small_bus_routes = int((df_routes["bus_kind"] == "Small Bus").sum())
        large_bus_routes = int((df_routes["bus_kind"] == "Large Bus").sum())
    else:
        num_selected_routes = 0
        total_distance = np.nan
        avg_route_distance = np.nan
        small_bus_routes = 0
        large_bus_routes = 0

    out = {
        "scenario": scenario_name,
        "objective": result["objective"],
        "status": result["status"],
        "success": result["success"],
        "route_generation_time_s": route_generation_time,
        "solve_time_s": result.get("solve_time_s", np.nan),
        "total_runtime_s": total_runtime,
        "num_candidate_routes": num_candidate_routes,
        "num_valid_vars": result["num_valid_vars"],
        "num_constraints": result["num_constraints"],
        "num_demand_points": result["num_demand_points"],
        "num_uncovered_points": result["num_uncovered_points"],
        "num_selected_routes": num_selected_routes,
        "total_distance": total_distance,
        "avg_route_distance": avg_route_distance,
        "small_bus_routes": small_bus_routes,
        "large_bus_routes": large_bus_routes,
    }

    for k, v in params.items():
        out[f"param_{k}"] = v

    return out


# ============================================================
# 7. Full sensitivity analysis
# ============================================================

def run_sensitivity_analysis():
    all_results = []

    baseline_result = run_one_scenario(deepcopy(BASE_PARAMS), "baseline")
    all_results.append(baseline_result)

    for param_name, values in SENSITIVITY_CONFIG.items():
        for val in values:
            params = deepcopy(BASE_PARAMS)
            params[param_name] = val
            scenario_name = f"{param_name}={val}"
            print(f"Running scenario: {scenario_name}")
            res = run_one_scenario(params, scenario_name)
            all_results.append(res)

    results_df = pd.DataFrame(all_results)

    baseline_obj = results_df.loc[results_df["scenario"] == "baseline", "objective"].iloc[0]
    baseline_runtime = results_df.loc[results_df["scenario"] == "baseline", "total_runtime_s"].iloc[0]

    results_df["delta_objective_vs_baseline"] = results_df["objective"] - baseline_obj
    results_df["pct_objective_vs_baseline"] = (
        (results_df["objective"] - baseline_obj) / baseline_obj * 100.0
    )

    results_df["delta_runtime_vs_baseline"] = results_df["total_runtime_s"] - baseline_runtime
    results_df["pct_runtime_vs_baseline"] = (
        (results_df["total_runtime_s"] - baseline_runtime) / baseline_runtime * 100.0
    )

    results_df.to_csv("results/fleet_sensitivity_results.csv", index=False)
    print("Saved: fleet_sensitivity_results.csv")

    return results_df


if __name__ == "__main__":
    df = run_sensitivity_analysis()
    print(df.head(20))
