"""Who rides the company bus, and which accommodations may serve them.

Only non-Saudi staff are transported by the employer-operated buses; Saudi staff
arrange their own commute. Every routing method must build its boarding/alighting
demand from bus riders only, using the same rule as the staff scheduling model
(staff_scheduling/data_loader.py).

Accommodation rule (problem statement, Fleet Routing): clusters and their buses are fixed,
which "determines the accommodations the buses need to serve"; workers are tied to a branch,
not to an accommodation, and the bus-to-accommodation assignment is a decision. So a route
may start at any accommodation of the cluster its branches belong to. The environment
variable BUS_ACCOMMODATION_RULE selects the rule:
  cluster (default)  any accommodation in the branch's cluster
  branch             only the accommodation recorded for the branch's riders in the staff data
  none               any accommodation in the network (the relaxation: clusters not fixed)
BUS_IGNORE_ACCOMMODATION=1 is kept as an alias for "none".
"""
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

SAUDI_NATIONALITY_VALUES = {"saudi", "saudi arabia"}
DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "data" / "Berkeley_Mock_Data.xlsx"


def accommodation_rule() -> str:
    if os.environ.get("BUS_IGNORE_ACCOMMODATION") == "1":
        return "none"
    rule = os.environ.get("BUS_ACCOMMODATION_RULE", "cluster").lower()
    if rule not in {"cluster", "branch", "none"}:
        raise ValueError(f"BUS_ACCOMMODATION_RULE must be cluster, branch or none, not {rule!r}")
    return rule


def is_bus_rider(nationality: pd.Series) -> pd.Series:
    return ~nationality.fillna("").astype(str).str.strip().str.lower().isin(SAUDI_NATIONALITY_VALUES)


def load_bus_rider_ids(data_file) -> set:
    staff = pd.read_excel(data_file, sheet_name="staff")
    if "Nationality" not in staff.columns:
        staff["Nationality"] = "Non-Saudi"
    return set(staff.loc[is_bus_rider(staff["Nationality"]), "staff_id"].astype(str))


def keep_bus_riders(schedule_df: pd.DataFrame, data_file, staff_col: str = "Staff") -> pd.DataFrame:
    """Drop schedule rows of staff who do not ride the bus."""
    rider_ids = load_bus_rider_ids(data_file)
    return schedule_df[schedule_df[staff_col].astype(str).isin(rider_ids)].copy()


@lru_cache(maxsize=None)
def accommodation_cluster(data_file=DEFAULT_DATA_FILE) -> dict:
    """Active accommodation id -> cluster id it belongs to (whose buses are based there)."""
    acc = pd.read_excel(data_file, sheet_name="accommodation")
    if "active" in acc.columns:
        acc = acc[acc["active"].fillna(True).astype(bool)]
    return dict(zip(acc["id"].astype(str), acc["cluster_id"].astype(str)))


@lru_cache(maxsize=None)
def branch_cluster(data_file=DEFAULT_DATA_FILE) -> dict:
    br = pd.read_excel(data_file, sheet_name="branches")
    return dict(zip(br["branch_id"], br["cluster_id"].astype(str)))


@lru_cache(maxsize=None)
def _recorded_accommodation(data_file) -> dict:
    """branch_id -> the accommodation recorded for its riders in the staff sheet (rule 'branch')."""
    staff = pd.read_excel(data_file, sheet_name="staff")
    if "Nationality" not in staff.columns:
        staff["Nationality"] = "Non-Saudi"
    riders = staff[is_bus_rider(staff["Nationality"])]
    acc = riders.groupby("branch_id")["accommodation_id"].agg(lambda s: s.dropna().astype(str).unique().tolist())
    mixed = {b: a for b, a in acc.items() if len(a) > 1}
    if mixed:
        raise ValueError(f"Branches whose riders live in more than one accommodation: {mixed}")
    return {b: a[0] for b, a in acc.items() if a}


@lru_cache(maxsize=None)
def _allowed(data_file, rule) -> dict:
    accs = accommodation_cluster(data_file)
    if rule == "branch":
        return {b: [a] for b, a in _recorded_accommodation(data_file).items()}
    by_cluster = {}
    for a, c in sorted(accs.items()):
        by_cluster.setdefault(c, []).append(a)
    every = sorted(accs)
    return {b: (by_cluster.get(c, []) if rule == "cluster" else every) for b, c in branch_cluster(data_file).items()}


def allowed_accommodations(data_file=DEFAULT_DATA_FILE) -> dict:
    """branch_id -> list of accommodations a route serving that branch may start and end at."""
    return _allowed(data_file, accommodation_rule())


def serves_own_accommodation(depot, stops, data_file=DEFAULT_DATA_FILE) -> bool:
    """True if the route's start accommodation is allowed for every stop under the current rule.
    A stop that is not a branch of `data_file` raises an error, so a wrong data file can't silently pass."""
    allowed = allowed_accommodations(data_file)
    unknown = [b for b in stops if b not in allowed]
    if unknown:
        raise KeyError(f"stops not in {data_file}: {unknown[:5]} (wrong data file for this network?)")
    return all(str(depot) in allowed[b] for b in stops)


def branch_accommodation(data_file=DEFAULT_DATA_FILE) -> dict:
    """branch_id -> its single allowed accommodation, for branches that have exactly one (rule 'branch',
    or a cluster with one accommodation). Callers that need all options use allowed_accommodations()."""
    return {b: a[0] for b, a in allowed_accommodations(data_file).items() if len(a) == 1}
