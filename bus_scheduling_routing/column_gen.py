# ============================================================
# Column Generation for Bus Fleet Scheduling
# ============================================================

import argparse
import ast
import os
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import itertools
import math

# ============================================================
# 0. Runtime parameters / data loading
# ============================================================

DEFAULT_PARAMS = {
    "TRAVEL_SPEED": 40.0,
    "BOARDING_TIME": 0.5,
    "ALIGHTING_TIME": 0.3,
    "EARLIEST_DEPARTURE_SHIFT_MIN": -30.0,
    "PENALTY_BOARD_LATE": 20.0,
    "PENALTY_ALIGHT_LATE": 20.0,
    "REWARD_ALIGHT_EARLY": 2.0,
    "TAU_SOFT_WINDOW_MIN": 5.0,
    "TAU_VIOLATION_PENALTY": 50.0,
    "TAU_VIOLATION_PENALTY_MULTIPLIER": 4.0,
    "WEEK_HOURS": 168,
    "TIME_SHIFT_HOURS": 0,
}

TRAVEL_SPEED = None
BOARDING_TIME = None
ALIGHTING_TIME = None
EARLIEST_DEPARTURE_SHIFT_MIN = None
PENALTY_BOARD_LATE = None
PENALTY_ALIGHT_LATE = None
REWARD_ALIGHT_EARLY = None
TAU_SOFT_WINDOW_MIN = None
TAU_VIOLATION_PENALTY = None
TAU_VIOLATION_PENALTY_MULTIPLIER = None


def apply_runtime_params(params):
    global TRAVEL_SPEED, BOARDING_TIME, ALIGHTING_TIME
    global EARLIEST_DEPARTURE_SHIFT_MIN, PENALTY_BOARD_LATE, PENALTY_ALIGHT_LATE, REWARD_ALIGHT_EARLY
    global TAU_SOFT_WINDOW_MIN, TAU_VIOLATION_PENALTY, TAU_VIOLATION_PENALTY_MULTIPLIER

    TRAVEL_SPEED = float(params["TRAVEL_SPEED"])
    BOARDING_TIME = float(params["BOARDING_TIME"])
    ALIGHTING_TIME = float(params["ALIGHTING_TIME"])
    EARLIEST_DEPARTURE_SHIFT_MIN = float(params["EARLIEST_DEPARTURE_SHIFT_MIN"])
    PENALTY_BOARD_LATE = float(params["PENALTY_BOARD_LATE"])
    PENALTY_ALIGHT_LATE = float(params["PENALTY_ALIGHT_LATE"])
    REWARD_ALIGHT_EARLY = float(params["REWARD_ALIGHT_EARLY"])
    TAU_SOFT_WINDOW_MIN = float(params["TAU_SOFT_WINDOW_MIN"])
    TAU_VIOLATION_PENALTY = float(params["TAU_VIOLATION_PENALTY"])
    TAU_VIOLATION_PENALTY_MULTIPLIER = float(params["TAU_VIOLATION_PENALTY_MULTIPLIER"])


apply_runtime_params(DEFAULT_PARAMS)


def _truthy(v, default=True):
    if pd.isna(v):
        return default
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "active", "available"}


def _safe_str(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _parse_stops(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def load_project_data(
    data_file="data/Berkeley_Mock_Data.xlsx",
    schedule_file="data/schedule_result.csv",
    params=None,
):
    if params is None:
        params = DEFAULT_PARAMS

    branches_df = pd.read_excel(data_file, sheet_name="branches")
    acc_df = pd.read_excel(data_file, sheet_name="accommodation")
    vehicles_df = pd.read_excel(data_file, sheet_name="vehicles")
    cost_df = pd.read_excel(data_file, sheet_name="vehicle_cost_model")
    bus_info_df = pd.read_excel(data_file, sheet_name="bus_info")
    staff_df = pd.read_excel(data_file, sheet_name="staff")[["staff_id", "branch_id"]]
    schedule_df = pd.read_csv(schedule_file)

    schedule_df["Start"] = (
        schedule_df["Start"].astype(int) - int(params["TIME_SHIFT_HOURS"])
    ) % int(params["WEEK_HOURS"])
    schedule_df["End"] = (
        schedule_df["End"].astype(int) - int(params["TIME_SHIFT_HOURS"])
    ) % int(params["WEEK_HOURS"])

    merged = pd.merge(schedule_df, staff_df, left_on="Staff", right_on="staff_id")

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

    branch_rows = []
    for row in branches_df.itertuples(index=False):
        bid = _safe_str(getattr(row, "branch_id", None))
        lat = getattr(row, "latitude", None)
        lon = getattr(row, "longitude", None)
        cid = _safe_str(getattr(row, "cluster_id", None))
        if bid is None or pd.isna(lat) or pd.isna(lon):
            continue
        branch_rows.append((bid, float(lat), float(lon), cid))

    depot_rows = []
    for row in acc_df.itertuples(index=False):
        did = _safe_str(getattr(row, "id", None))
        lat = getattr(row, "latitude", None)
        lon = getattr(row, "longitude", None)
        cid = _safe_str(getattr(row, "cluster_id", None))
        active = _truthy(getattr(row, "active", True), default=True)
        if did is None or cid is None or pd.isna(lat) or pd.isna(lon) or not active:
            continue
        depot_rows.append((did, float(lat), float(lon), cid))

    if not branch_rows:
        raise ValueError("No valid branches found in branches sheet.")
    if not depot_rows:
        raise ValueError("No active depots found in accommodation sheet.")

    branch_cluster = {bid: cid for bid, _lat, _lon, cid in branch_rows if cid is not None}

    depot_by_cluster = {}
    for did, _lat, _lon, cid in depot_rows:
        depot_by_cluster.setdefault(cid, did)

    loc_ids = [bid for bid, *_rest in branch_rows] + [did for did, *_rest in depot_rows]
    loc_coords = {
        bid: (lat, lon) for bid, lat, lon, _cid in branch_rows
    }
    loc_coords.update({
        did: (lat, lon) for did, lat, lon, _cid in depot_rows
    })

    distance_matrix = pd.DataFrame(
        np.zeros((len(loc_ids), len(loc_ids))),
        index=loc_ids,
        columns=loc_ids,
    )
    for a in loc_ids:
        la, loa = loc_coords[a]
        for b in loc_ids:
            if a == b:
                continue
            lb, lob = loc_coords[b]
            distance_matrix.loc[a, b] = haversine_km(la, loa, lb, lob)

    cost_by_id = cost_df.set_index("id").to_dict("index")
    bus_info_by_id = bus_info_df.set_index("id").to_dict("index")

    vehicle_types = {}
    for row in vehicles_df.itertuples(index=False):
        vid = _safe_str(getattr(row, "id", None))
        if vid is None:
            continue

        active = _truthy(getattr(row, "active", True), default=True)
        status = _safe_str(getattr(row, "status", "available")) or "available"
        if not active or status.lower() != "available":
            continue

        cap = getattr(row, "capacity", None)
        if pd.isna(cap):
            cap = bus_info_by_id.get(vid, {}).get("capacity")
        if pd.isna(cap):
            continue

        cid = _safe_str(getattr(row, "cluster_id", None))
        if cid is None:
            cid = _safe_str(bus_info_by_id.get(vid, {}).get("cluster_id"))
        if cid is None or cid not in depot_by_cluster:
            continue

        cmid = getattr(row, "cost_model_id", None)
        cm = cost_by_id.get(cmid, {})

        fixed_cost = cm.get("fixed_cost", 0.0)
        cost_per_km = cm.get("cost_per_km", 0.0)
        cost_per_min = cm.get("cost_per_hour", None)
        if pd.isna(cost_per_min):
            cost_per_min = bus_info_by_id.get(vid, {}).get("cost_per_min", 0.0)
        else:
            cost_per_min = float(cost_per_min) / 60.0

        vehicle_type = _safe_str(getattr(row, "vehicle_type", "unknown")) or "unknown"
        key = (
            f"{cid}::{vehicle_type}::cap{int(cap)}::fc{float(fixed_cost):.4f}"
            f"::dkm{float(cost_per_km):.4f}::dmin{float(cost_per_min):.4f}"
        )
        if key not in vehicle_types:
            vehicle_types[key] = {
                "capacity": int(cap),
                "fixed_cost": float(fixed_cost),
                "cost_per_km": float(cost_per_km),
                "cost_per_min": float(cost_per_min),
                "cluster_id": cid,
                "vehicle_type": vehicle_type,
            }

    if not vehicle_types:
        raise ValueError("No available vehicle types could be built from vehicles sheet.")

    branches = sorted({b for b, *_rest in branch_rows})
    horizon = 0
    if not start_counts.empty:
        horizon = max(horizon, int(start_counts["Start"].max()) + 1)
    if not end_counts.empty:
        horizon = max(horizon, int(end_counts["End"].max()) + 1)

    demand_data = {}
    for t in range(horizon):
        for b in branches:
            demand_data[(t, b)] = {"board": 0, "alight": 0}

    for row in start_counts.itertuples(index=False):
        demand_data[(int(row.Start), row.branch_id)]["board"] = int(row.staff_starting_count)
    for row in end_counts.itertuples(index=False):
        demand_data[(int(row.End), row.branch_id)]["alight"] = int(row.staff_ending_count)

    return demand_data, distance_matrix, vehicle_types, depot_by_cluster, branch_cluster


def initialize_from_fixed_solution(
    rmp,
    fixed_routes_csv,
    demand_data,
    distance_matrix,
    vehicle_types,
    depot_by_cluster,
    verbose=True,
):
    if not fixed_routes_csv or not os.path.exists(fixed_routes_csv):
        return set()

    fixed_df = pd.read_csv(fixed_routes_csv)
    covered = set()
    added = 0

    depot_cluster_lookup = {}
    for cluster_id, depot in depot_by_cluster.items():
        depot_cluster_lookup[str(depot)] = str(cluster_id)

    def normalize_bus_kind(value):
        s = str(value).strip().lower().replace("__", " ").replace("_", " ")
        if "small" in s:
            return "small bus"
        if "large" in s:
            return "large bus"
        return s

    def match_vehicle_type(csv_row, depot_id):
        raw_vtype = getattr(csv_row, "vehicle_type", None)
        raw_bus_kind = getattr(csv_row, "bus_kind", None)
        raw_capacity = getattr(csv_row, "capacity", None)

        if raw_vtype in vehicle_types:
            return raw_vtype

        depot_cluster = depot_cluster_lookup.get(str(depot_id))
        desired_kind = normalize_bus_kind(raw_bus_kind if raw_bus_kind is not None else raw_vtype)
        desired_cap = None if pd.isna(raw_capacity) else int(raw_capacity)

        candidates = []
        for key, params in vehicle_types.items():
            if depot_cluster is not None and str(params.get("cluster_id")) != depot_cluster:
                continue
            kind = normalize_bus_kind(params.get("vehicle_type", key))
            cap = int(params.get("capacity", -1))
            score = 0
            if kind == desired_kind:
                score += 10
            if desired_cap is not None and cap == desired_cap:
                score += 5
            if raw_vtype is not None and normalize_bus_kind(raw_vtype) == kind:
                score += 3
            candidates.append((score, -cap, key))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]

    for row in fixed_df.itertuples(index=False):
        t = int(getattr(row, "time", -1))
        depot_id = getattr(row, "depot_id", None)
        if depot_id is None:
            depot_id = getattr(row, "depot", None)
        v_type = match_vehicle_type(row, depot_id)
        stops = _parse_stops(getattr(row, "stops", None))
        if t < 0 or depot_id is None or v_type is None or not stops:
            continue

        route = Route(
            route_id=f"FIXED_{t}_{depot_id}_{added}",
            stops=stops,
            distance_matrix=distance_matrix,
            time_slot=t,
            depot=depot_id,
        )
        route.evaluate_time_feasibility(demand_data, distance_matrix)
        if not route.time_feasible:
            continue
        if not route.is_capacity_feasible(vehicle_types[v_type]["capacity"], demand_data):
            continue

        params = vehicle_types[v_type]
        cost = float(
            params["fixed_cost"]
            + params["cost_per_km"] * route.distance
            + params["cost_per_min"] * route.travel_minutes()
            + route.time_cost
        )
        coverage = {
            (t, b)
            for b in stops
            if demand_data.get((t, b), {"board": 0, "alight": 0})["board"]
            + demand_data.get((t, b), {"board": 0, "alight": 0})["alight"] > 0
        }
        if coverage and rmp.add_column(cost, coverage, route, v_type):
            covered |= coverage
            added += 1

    if verbose:
        print(f"Warm-started {added} columns from fixed solution.")
    return covered


class Route:
    def __init__(self, route_id, stops, distance_matrix, time_slot, depot):
        self.route_id = route_id
        self.stops = list(stops)
        self.time_slot = time_slot
        self.depot = depot
        self.distance = self._calc_dist(distance_matrix)
        self.time_feasible = False
        self.best_departure_shift = 0.0
        self.tau_by_stop = {}
        self.time_cost = 0.0

    def _calc_dist(self, dm):
        if not self.stops:
            return 0.0
        total = float(dm.at[self.depot, self.stops[0]])
        for i in range(len(self.stops) - 1):
            total += float(dm.at[self.stops[i], self.stops[i + 1]])
        total += float(dm.at[self.stops[-1], self.depot])
        return total

    def _travel_min(self, km):
        return 60.0 * km / TRAVEL_SPEED

    def travel_minutes(self):
        return self._travel_min(self.distance)

    def _arrival_offsets(self, demand, dm):
        if not self.stops:
            return {}
        offsets = {}
        elapsed = self._travel_min(float(dm.at[self.depot, self.stops[0]]))
        offsets[self.stops[0]] = elapsed
        for i in range(len(self.stops) - 1):
            cur, nxt = self.stops[i], self.stops[i + 1]
            d = demand.get((self.time_slot, cur), {"board": 0, "alight": 0})
            elapsed += d["board"] * BOARDING_TIME + d["alight"] * ALIGHTING_TIME
            elapsed += self._travel_min(float(dm.at[cur, nxt]))
            offsets[nxt] = elapsed
        return offsets

    def _load_profile(self, demand):
        changes = [
            demand.get((self.time_slot, b), {"board": 0, "alight": 0})["board"]
            - demand.get((self.time_slot, b), {"board": 0, "alight": 0})["alight"]
            for b in self.stops
        ]
        cum = 0
        mn = 0
        for c in changes:
            cum += c
            mn = min(mn, cum)
        init = max(0, -mn)
        cur = init
        profile = []
        for c in changes:
            cur += c
            profile.append(cur)
        return init, profile

    def is_capacity_feasible(self, cap, demand):
        init, profile = self._load_profile(demand)
        return init <= cap and all(0 <= l <= cap for l in profile)

    def evaluate_time_feasibility(self, demand, dm):
        offsets = self._arrival_offsets(demand, dm)
        lo = float(EARLIEST_DEPARTURE_SHIFT_MIN)
        tol = max(0.0, float(TAU_SOFT_WINDOW_MIN))
        candidates = [lo]
        for b in self.stops:
            for candidate in (-offsets[b] - tol, -offsets[b], -offsets[b] + tol):
                if candidate >= lo - 1e-9:
                    candidates.append(float(candidate))

        def violation_penalty(delta):
            total = 0.0
            base_rate = max(0.0, float(TAU_VIOLATION_PENALTY))
            excess_rate = base_rate * max(1.0, float(TAU_VIOLATION_PENALTY_MULTIPLIER))
            for b in self.stops:
                d = demand.get((self.time_slot, b), {"board": 0, "alight": 0})
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
                d = demand.get((self.time_slot, b), {"board": 0, "alight": 0})
                tau = delta + offsets[b]
                total += PENALTY_BOARD_LATE * d["board"] * max(0.0, tau)
                total += PENALTY_ALIGHT_LATE * d["alight"] * max(0.0, tau)
                if d["alight"] > 0:
                    total -= REWARD_ALIGHT_EARLY * d["alight"] * max(0.0, -tau)
            return total + violation_penalty(delta)

        best = min(candidates, key=time_cost)
        self.tau_by_stop = {b: best + offsets[b] for b in self.stops}

        self.time_feasible = True
        self.best_departure_shift = float(best)
        self.time_cost = max(0.0, float(time_cost(best)))
        return True

# ============================================================
# 1. Restricted Master Problem (RMP)
# ============================================================

class RestrictedMasterProblem:
    """
    LP Relaxation of Set Partitioning:
    min  sum_{r,v,t} c_{r,v,t} * lambda_{r,v,t}
    s.t. sum_{r,v} a_{b,r} * lambda_{r,v,t} = 1  ∀(t,b)
         lambda_{r,v,t} >= 0
    """
    
    def __init__(self, demand_points):
        """
        demand_points: list of (t, b) tuples with positive demand
        """
        self.demand_points = demand_points
        self.demand_index = {dp: i for i, dp in enumerate(demand_points)}
        self.n_constraints = len(demand_points)
        
        # columns: list of (cost, coverage_set, route_obj, v_type)
        self.columns = []
        self.column_signatures = set()
        
        self.model = None
        self.lambda_vars = []
        self.dual_values = None
        
    def add_column(self, cost, coverage_set, route_obj, v_type):
        """
        coverage_set: set of (t,b) tuples this route covers
        """
        signature = (
            route_obj.time_slot,
            route_obj.depot,
            v_type,
            tuple(route_obj.stops),
        )
        if signature in self.column_signatures:
            return False

        self.columns.append({
            'cost': cost,
            'coverage': coverage_set,
            'route': route_obj,
            'v_type': v_type
        })
        self.column_signatures.add(signature)
        return True
    
    def build_model(self):
        """Build LP relaxation model."""
        self.model = Model(name='RMP')
        n_cols = len(self.columns)
        
        # Variables: lambda >= 0 (LP relaxation)
        self.lambda_vars = [
            self.model.continuous_var(lb=0.0, name=f'lam_{i}')
            for i in range(n_cols)
        ]
        
        # Objective
        self.model.minimize(
            self.model.sum(
                self.columns[i]['cost'] * self.lambda_vars[i]
                for i in range(n_cols)
            )
        )
        
        # Coverage constraints: = 1 for each demand point
        self.coverage_constraints = []
        for dp in self.demand_points:
            terms = [
                self.lambda_vars[i]
                for i, col in enumerate(self.columns)
                if dp in col['coverage']
            ]
            if not terms:
                raise ValueError(
                    f"No initial column covers demand point {(dp[0], dp[1])}."
                )
            ct = self.model.add_constraint(
                self.model.sum(terms) == 1,
                ctname=f'cov_{dp[0]}_{dp[1]}'
            )
            self.coverage_constraints.append(ct)
        
        return self.model
    
    def add_column_to_model(self, cost, coverage_set):
        """Rebuild the RMP after adding columns.

        Docplex constraints are not safely mutable in the way the original
        incremental update assumed, so we rebuild from the current column set.
        """
        self.build_model()
    
    def solve(self):
        """Solve RMP and return dual values."""
        solution = self.model.solve(log_output=False)
        if not solution:
            return None, None
        
        obj = solution.objective_value
        
        # Extract dual values (shadow prices)
        self.dual_values = {}
        for i, dp in enumerate(self.demand_points):
            ct = self.coverage_constraints[i]
            dual = self.model.dual_values([ct])[0]
            self.dual_values[dp] = dual
        
        return obj, self.dual_values
    
    def get_solution_vars(self):
        return [(i, self.lambda_vars[i].solution_value) 
                for i in range(len(self.lambda_vars))
                if self.lambda_vars[i].solution_value > 1e-6]


# ============================================================
# 2. Pricing Problem (Subproblem)
# ============================================================

class PricingProblem:
    """
    For each time slot t and vehicle type v:
    Find route r that minimizes reduced cost:
    
    reduced_cost = c_{r,v,t} - sum_{b in r} pi_{t,b}
    
    where pi_{t,b} are dual variables from RMP.
    
    If min reduced_cost < -epsilon, add column to RMP.
    """
    
    def __init__(self, demand_data, distance_matrix, vehicle_types,
                 depot_by_cluster, branch_cluster, max_stops=10,
                 max_greedy_starts=10, max_savings_pairs=100,
                 full_permutation_limit=6, max_orderings_per_stop_list=120):
        self.demand_data = demand_data
        self.distance_matrix = distance_matrix
        self.vehicle_types = vehicle_types
        self.depot_by_cluster = depot_by_cluster
        self.branch_cluster = branch_cluster
        self.max_stops = max_stops
        self.max_greedy_starts = max_greedy_starts
        self.max_savings_pairs = max_savings_pairs
        self.full_permutation_limit = full_permutation_limit
        self.max_orderings_per_stop_list = max_orderings_per_stop_list
    
    def _compute_route_cost(self, route_obj, v_type):
        """Compute actual route cost."""
        params = self.vehicle_types[v_type]
        return float(
            params['fixed_cost']
            + params['cost_per_km'] * route_obj.distance
            + params['cost_per_min'] * route_obj.travel_minutes()
            + route_obj.time_cost
        )
    
    def _reduced_cost(self, route_obj, v_type, t, dual_values):
        """
        reduced_cost = c_{r,v,t} - sum_{b in stops} pi_{t,b}
        """
        cost = self._compute_route_cost(route_obj, v_type)
        dual_sum = sum(
            dual_values.get((t, b), 0.0)
            for b in route_obj.stops
        )
        return cost - dual_sum

    def _allowed_depot(self, depot_id, stops):
        return True

    def _best_feasible_cost(self, route_obj, feasible_vtypes):
        costs = [
            self._compute_route_cost(route_obj, v_type)
            for v_type in feasible_vtypes
            if route_obj.is_capacity_feasible(
                self.vehicle_types[v_type]['capacity'],
                self.demand_data,
            )
        ]
        if not costs:
            return float("inf")
        return min(costs)

    def _generate_stop_lists(self, active_branches, t, dual_values, depot_id):
        stop_lists = []
        depot_cluster = next(
            (cluster_id for cluster_id, did in self.depot_by_cluster.items() if did == depot_id),
            None,
        )
        feasible_vtypes = [
            v_type
            for v_type, params in self.vehicle_types.items()
            if depot_cluster is None or params.get("cluster_id") == depot_cluster
        ]
        if not feasible_vtypes:
            feasible_vtypes = list(self.vehicle_types.keys())

        sorted_branches = sorted(
            active_branches,
            key=lambda b: dual_values.get((t, b), 0.0),
            reverse=True,
        )

        for b in active_branches:
            stop_lists.append([b])

        for start in sorted_branches[: min(self.max_greedy_starts, len(sorted_branches))]:
            route = [start]
            remaining = set(active_branches) - {start}

            while remaining and len(route) < self.max_stops:
                current_route = Route(
                    f"greedy_{t}_{depot_id}_{'_'.join(route)}",
                    route,
                    self.distance_matrix,
                    t,
                    depot_id,
                )
                current_route.evaluate_time_feasibility(
                    self.demand_data, self.distance_matrix
                )
                current_cost = (
                    self._best_feasible_cost(current_route, feasible_vtypes)
                    if current_route.time_feasible
                    else float("inf")
                )
                if not math.isfinite(current_cost):
                    break

                best = None
                best_delta_rc = float("inf")

                for candidate in remaining:
                    trial_stops = route + [candidate]
                    trial_route = Route(
                        f"greedy_{t}_{depot_id}_{'_'.join(trial_stops)}",
                        trial_stops,
                        self.distance_matrix,
                        t,
                        depot_id,
                    )
                    trial_route.evaluate_time_feasibility(
                        self.demand_data, self.distance_matrix
                    )
                    if not trial_route.time_feasible:
                        continue

                    trial_cost = self._best_feasible_cost(trial_route, feasible_vtypes)
                    if not math.isfinite(trial_cost):
                        continue

                    delta_rc = (
                        trial_cost
                        - current_cost
                        - dual_values.get((t, candidate), 0.0)
                    )
                    if delta_rc < best_delta_rc:
                        best_delta_rc = delta_rc
                        best = candidate

                if best is None or best_delta_rc >= 0.0:
                    break
                route.append(best)
                remaining.remove(best)

            if len(route) > 1:
                stop_lists.append(route)

        savings = []
        for i, j in itertools.combinations(active_branches, 2):
            s = (
                float(self.distance_matrix.at[depot_id, i])
                + float(self.distance_matrix.at[depot_id, j])
                - float(self.distance_matrix.at[i, j])
            )
            weighted_s = s * (
                dual_values.get((t, i), 0.0)
                + dual_values.get((t, j), 0.0)
                + 1.0
            )
            savings.append((weighted_s, i, j))

        savings.sort(reverse=True)
        current_routes = [[b] for b in active_branches]
        changed = True
        while changed:
            changed = False
            for _, i, j in savings[: self.max_savings_pairs]:
                route_i = next((r for r in current_routes if r and r[-1] == i), None)
                route_j = next((r for r in current_routes if r and r[0] == j), None)
                if route_i is None or route_j is None or route_i is route_j:
                    continue
                merged = route_i + route_j
                if len(set(merged)) != len(merged):
                    continue
                if len(merged) > self.max_stops:
                    continue
                current_routes.remove(route_i)
                current_routes.remove(route_j)
                current_routes.append(merged)
                changed = True
                break

        stop_lists.extend(current_routes)

        unique = []
        seen = set()
        for stop_list in stop_lists:
            key = tuple(stop_list)
            if key in seen:
                continue
            seen.add(key)
            unique.append(stop_list)
        return unique

    def _expand_stop_list_orderings(self, stop_list):
        if len(stop_list) <= 1:
            return [list(stop_list)]

        orderings = []
        seen = set()

        def add_order(order):
            key = tuple(order)
            if key in seen:
                return
            seen.add(key)
            orderings.append(list(order))

        add_order(stop_list)
        add_order(list(reversed(stop_list)))

        if len(stop_list) <= self.full_permutation_limit:
            for perm in itertools.permutations(stop_list):
                add_order(perm)
                if len(orderings) >= self.max_orderings_per_stop_list:
                    break

        return orderings

    def solve_for_period(self, t, active_branches, dual_values, epsilon=1e-6):
        """
        Heuristic pricing: generate a limited set of promising stop lists
        guided by dual values and route savings.
        Returns list of (reduced_cost, route_obj, v_type, coverage_set)
        with negative reduced cost.
        """
        if not active_branches:
            return []

        new_columns = []
        seen_signatures = set()

        for v_type, params in self.vehicle_types.items():
            for cluster_id, depot_id in self.depot_by_cluster.items():
                for stop_list in self._generate_stop_lists(active_branches, t, dual_values, depot_id):
                    for stops in self._expand_stop_list_orderings(stop_list):
                        if not self._allowed_depot(depot_id, stops):
                            continue

                        signature = (t, depot_id, v_type, tuple(stops))
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)

                        route = Route(
                            f'CG_{t}_{v_type}_{depot_id}_{"_".join(stops)}',
                            stops,
                            self.distance_matrix,
                            t,
                            depot_id
                        )

                        route.evaluate_time_feasibility(
                            self.demand_data, self.distance_matrix
                        )
                        if not route.time_feasible:
                            continue
                        if not route.is_capacity_feasible(
                            params['capacity'], self.demand_data
                        ):
                            continue

                        rc = self._reduced_cost(route, v_type, t, dual_values)

                        if rc < -epsilon:
                            coverage = {
                                (t, b) for b in stops
                                if self.demand_data.get(
                                    (t, b), {'board': 0, 'alight': 0}
                                )['board'] + self.demand_data.get(
                                    (t, b), {'board': 0, 'alight': 0}
                                )['alight'] > 0
                            }
                            if coverage:
                                new_columns.append((rc, route, v_type, coverage))

        # Sort by most negative reduced cost
        new_columns.sort(key=lambda x: x[0])
        return new_columns


# ============================================================
# 3. Column Generation Main Loop
# ============================================================

def column_generation(
    demand_data,
    distance_matrix,
    vehicle_types,
    depot_by_cluster,
    branch_cluster,
    fixed_init_csv="",
    max_iterations=500,
    max_stops=10,
    max_new_cols_per_period=5,
    max_greedy_starts=10,
    max_savings_pairs=100,
    epsilon=1e-6,
    verbose=True
):
    """
    Main Column Generation loop.
    
    Returns:
        - Final restricted-master objective
        - Selected columns
        - Convergence history
    """
    
    # ---- Step 1: Identify all demand points ----
    demand_points = [
        (t, b) for (t, b), d in demand_data.items()
        if d['board'] + d['alight'] > 0
    ]
    demand_points = sorted(set(demand_points))
    
    active_by_time = {}
    for (t, b) in demand_points:
        if t not in active_by_time:
            active_by_time[t] = []
        active_by_time[t].append(b)
    
    if verbose:
        print(f"Demand points: {len(demand_points)}")
        print(f"Active time slots: {len(active_by_time)}")
    
    # ---- Step 2: Initialize RMP with single-stop routes ----
    rmp = RestrictedMasterProblem(demand_points)
    
    pricing = PricingProblem(
        demand_data=demand_data,
        distance_matrix=distance_matrix,
        vehicle_types=vehicle_types,
        depot_by_cluster=depot_by_cluster,
        branch_cluster=branch_cluster,
        max_stops=max_stops,
        max_greedy_starts=max_greedy_starts,
        max_savings_pairs=max_savings_pairs,
    )
    
    covered = initialize_from_fixed_solution(
        rmp=rmp,
        fixed_routes_csv=fixed_init_csv,
        demand_data=demand_data,
        distance_matrix=distance_matrix,
        vehicle_types=vehicle_types,
        depot_by_cluster=depot_by_cluster,
        verbose=verbose,
    )

    # Add initial columns (single-stop routes for uncovered demand points)
    if verbose:
        print("Initializing remaining uncovered demand points with single-stop routes...")
    
    for (t, b) in demand_points:
        if (t, b) in covered:
            continue
        for v_type, params in vehicle_types.items():
            # Find nearest depot
            best_depot = min(
                depot_by_cluster.values(),
                key=lambda d: float(distance_matrix.at[d, b])
            )
            
            route = Route(
                f'INIT_{t}_{b}_{v_type}',
                [b],
                distance_matrix,
                t,
                best_depot
            )
            route.evaluate_time_feasibility(demand_data, distance_matrix)
            
            if not route.time_feasible:
                continue
            if not route.is_capacity_feasible(
                params['capacity'], demand_data
            ):
                continue
            
            cost = float(
                params['fixed_cost']
                + params['cost_per_km'] * route.distance
                + params['cost_per_min'] * route.travel_minutes()
                + route.time_cost
            )

            coverage = {(t, b)}
            if rmp.add_column(cost, coverage, route, v_type):
                break  # one feasible singleton column is enough
    
    if verbose:
        print(f"Initial columns: {len(rmp.columns)}")
    
    uncovered = [
        dp for dp in demand_points
        if not any(dp in col['coverage'] for col in rmp.columns)
    ]
    if uncovered:
        raise ValueError(
            f"Failed to build a feasible initial RMP; uncovered demand points: {uncovered[:10]}"
        )

    # Build initial model
    rmp.build_model()
    
    # ---- Step 3: Column Generation Loop ----
    history = []
    
    for iteration in range(max_iterations):
        
        # Solve RMP
        lp_obj, dual_values = rmp.solve()
        
        if lp_obj is None:
            print(f"RMP infeasible at iteration {iteration}")
            break
        
        history.append(lp_obj)
        
        if verbose:
            print(f"Iter {iteration:3d}: LP obj = ${lp_obj:,.2f}, "
                  f"columns = {len(rmp.columns)}")
        
        # Pricing: find columns with negative reduced cost
        total_new = 0
        
        for t, active in active_by_time.items():
            new_cols = pricing.solve_for_period(
                t, active, dual_values, epsilon
            )
            
            # Add most promising columns (avoid too many at once)
            added = 0
            for rc, route, v_type, coverage in new_cols[:max_new_cols_per_period]:
                cost = float(
                    vehicle_types[v_type]['fixed_cost']
                    + vehicle_types[v_type]['cost_per_km'] * route.distance
                    + vehicle_types[v_type]['cost_per_min'] 
                    * route.travel_minutes()
                    + route.time_cost
                )
                if rmp.add_column(cost, coverage, route, v_type):
                    added += 1
            
            total_new += added

        if total_new > 0:
            rmp.build_model()
        
        if verbose:
            print(f"         Added {total_new} new columns")
        
        # Convergence check
        if total_new == 0:
            if verbose:
                print(f"Converged at iteration {iteration}!")
                print(f"Final restricted-master objective: ${lp_obj:,.2f}")
            break
    
    return lp_obj, rmp, history


# ============================================================
# 4. Integer Solution (after CG)
# ============================================================

def solve_integer_after_cg(rmp, demand_points, vehicle_types):
    """
    Solve integer program using columns from CG.
    This gives the final integer solution.
    """
    model = Model(name='IP_after_CG')
    
    # Binary variables
    lambda_vars = [
        model.binary_var(name=f'lam_int_{i}')
        for i in range(len(rmp.columns))
    ]
    
    # Objective
    model.minimize(model.sum(
        rmp.columns[i]['cost'] * lambda_vars[i]
        for i in range(len(rmp.columns))
    ))
    
    # Coverage constraints: = 1 (exact cover)
    for dp in demand_points:
        terms = [
            lambda_vars[i]
            for i, col in enumerate(rmp.columns)
            if dp in col['coverage']
        ]
        if terms:
            model.add_constraint(
                model.sum(terms) == 1,
                ctname=f'cov_{dp[0]}_{dp[1]}'
            )
    
    solution = model.solve(log_output=True)
    
    if not solution:
        print("Integer solution not found!")
        return None, []
    
    print(f"\nInteger solution: ${solution.objective_value:,.2f}")
    
    selected = []
    for i, var in enumerate(lambda_vars):
        if var.solution_value > 0.5:
            col = rmp.columns[i]
            selected.append({
                'route': col['route'],
                'v_type': col['v_type'],
                'cost': col['cost'],
                'coverage': col['coverage'],
                'stops': col['route'].stops,
                'time': col['route'].time_slot,
                'depot': col['route'].depot,
            })
    
    return solution, selected


# 5. Main
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Column generation for bus fleet scheduling."
    )
    parser.add_argument(
        "--data-file",
        default="data/Berkeley_Mock_Data.xlsx",
        help="Path to Excel data file.",
    )
    parser.add_argument(
        "--schedule-file",
        default="data/schedule_result.csv",
        help="Path to schedule CSV file.",
    )
    parser.add_argument(
        "--fixed-init-csv",
        default="results/bus_fleet_schedule.csv",
        help="Optional fixed-model routes CSV used to warm-start the initial column pool.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum CG iterations.",
    )
    parser.add_argument(
        "--max-stops",
        type=int,
        default=10,
        help="Maximum route length considered in pricing.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Reduced-cost threshold.",
    )
    parser.add_argument(
        "--tau-soft-window-min",
        type=float,
        default=DEFAULT_PARAMS["TAU_SOFT_WINDOW_MIN"],
        help="Directional tau violation window in minutes before heavier penalties apply.",
    )
    parser.add_argument(
        "--penalty-board-late",
        type=float,
        default=DEFAULT_PARAMS["PENALTY_BOARD_LATE"],
        help="Per-passenger per-minute penalty for late arrival at boarding stops.",
    )
    parser.add_argument(
        "--penalty-alight-late",
        type=float,
        default=DEFAULT_PARAMS["PENALTY_ALIGHT_LATE"],
        help="Per-passenger per-minute penalty for late arrival at alighting stops.",
    )
    parser.add_argument(
        "--reward-alight-early",
        type=float,
        default=DEFAULT_PARAMS["REWARD_ALIGHT_EARLY"],
        help="Per-passenger per-minute reward for early arrival at alighting stops.",
    )
    parser.add_argument(
        "--tau-violation-penalty",
        type=float,
        default=DEFAULT_PARAMS["TAU_VIOLATION_PENALTY"],
        help="Base per-minute penalty for violating the desired tau direction.",
    )
    parser.add_argument(
        "--tau-violation-penalty-multiplier",
        type=float,
        default=DEFAULT_PARAMS["TAU_VIOLATION_PENALTY_MULTIPLIER"],
        help="Multiplier applied to the penalty beyond the soft tau window.",
    )
    args = parser.parse_args()

    runtime_params = dict(DEFAULT_PARAMS)
    runtime_params["PENALTY_BOARD_LATE"] = float(args.penalty_board_late)
    runtime_params["PENALTY_ALIGHT_LATE"] = float(args.penalty_alight_late)
    runtime_params["REWARD_ALIGHT_EARLY"] = float(args.reward_alight_early)
    runtime_params["TAU_SOFT_WINDOW_MIN"] = float(args.tau_soft_window_min)
    runtime_params["TAU_VIOLATION_PENALTY"] = float(args.tau_violation_penalty)
    runtime_params["TAU_VIOLATION_PENALTY_MULTIPLIER"] = float(args.tau_violation_penalty_multiplier)
    apply_runtime_params(runtime_params)

    print("=" * 60)
    print("Starting Column Generation...")
    print("=" * 60)

    demand_data, distance_matrix, vehicle_types, depot_by_cluster, branch_cluster = load_project_data(
        data_file=args.data_file,
        schedule_file=args.schedule_file,
        params=runtime_params,
    )

    lp_bound, rmp_final, cg_history = column_generation(
        demand_data=demand_data,
        distance_matrix=distance_matrix,
        vehicle_types=vehicle_types,
        depot_by_cluster=depot_by_cluster,
        branch_cluster=branch_cluster,
        fixed_init_csv=args.fixed_init_csv,
        max_iterations=args.max_iterations,
        max_stops=args.max_stops,
        epsilon=args.epsilon,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Solving Integer Program with CG columns...")
    print("=" * 60)

    demand_points_all = sorted([
        (t, b) for (t, b), d in demand_data.items()
        if d['board'] + d['alight'] > 0
    ])

    int_solution, selected_cg = solve_integer_after_cg(
        rmp=rmp_final,
        demand_points=demand_points_all,
        vehicle_types=vehicle_types
    )

    if int_solution:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"Heuristic CG RMP:  ${lp_bound:>12,.2f}")
        print(f"Restricted IP:     ${int_solution.objective_value:>12,.2f}")

        gap = (int_solution.objective_value - lp_bound) / max(abs(lp_bound), 1e-9) * 100
        print(f"\nRestricted gap: {gap:.2f}%")

        os.makedirs("results", exist_ok=True)
        df_cg = pd.DataFrame([{
            'time': r['time'],
            'stops': r['stops'],
            'v_type': r['v_type'],
            'depot': r['depot'],
            'cost': r['cost'],
        } for r in selected_cg])
        df_cg.to_csv('results/bus_fleet_schedule_cg.csv', index=False)
        print("\nSaved to results/bus_fleet_schedule_cg.csv")

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(cg_history, marker='o', linewidth=2)
    plt.xlabel('CG Iteration')
    plt.ylabel('LP Objective ($)')
    plt.title('Column Generation Convergence')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/cg_convergence.png', dpi=100)
    plt.show()
    print("Saved convergence plot.")
