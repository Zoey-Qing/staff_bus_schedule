"""Rebalance staff home branches so every branch can cover its own demand.

The synthetic dataset assigned 5-18 staff to branches whose weekly demand is
roughly equal (550-870 staff-hours), so 9 of 25 branches could not be covered
by their own staff even at the 78 h weekly maximum. This script re-assigns
home branches in proportion to each branch's weekly demand-hours, keeping
everything else about every staff member (id, role, nationality, skill,
wages) unchanged.

Rule (documented for the paper's data-generation section):
  1. Target headcount per branch: n_b = largest-remainder rounding of
     N * D_b / sum(D), where D_b is the branch's weekly demand-hours.
     Managers get at least MIN_MANAGERS per branch.
  2. Staff are split into strata (role x Saudi/non-Saudi x skill level).
     Each stratum is shared out across branches in proportion to n_b
     (largest remainder, ties broken toward the branch furthest below its
     target), so every branch gets a representative mix.
  3. Within a stratum, staff stay at their original branch while that
     branch's quota allows; the rest move, preferring branches in their
     original cluster.
  4. accommodation_id is set to the accommodation of the new home branch.

Usage:
  python rebalance_staff.py <input.xlsx> <output_staff.csv> [--report report.csv]
The output CSV holds the new staff sheet; apply_staff_sheet() writes it into
the Excel files.
"""
import argparse
import math
from collections import defaultdict

import pandas as pd

MIN_MANAGERS = 2
SAUDI = {"saudi", "saudi arabia"}


def largest_remainder(total, weights, floor_min=None, tie_key=None):
    """Integer apportionment of `total` across keys in proportion to `weights`."""
    keys = list(weights)
    wsum = float(sum(weights.values()))
    raw = {k: total * weights[k] / wsum for k in keys}
    out = {k: int(math.floor(raw[k])) for k in keys}
    if floor_min is not None:
        for k in keys:
            out[k] = max(out[k], floor_min)
    left = total - sum(out.values())
    order = sorted(keys, key=lambda k: (-(raw[k] - math.floor(raw[k])), tie_key(k) if tie_key else 0))
    i = 0
    while left > 0:
        out[order[i % len(order)]] += 1
        left -= 1
        i += 1
    while left < 0:  # floor_min overshoot: take back from the largest surplus
        k = max(keys, key=lambda k: out[k] - raw[k])
        if out[k] > (floor_min or 0):
            out[k] -= 1
            left += 1
    return out


def branch_demand_hours(xlsx):
    d = pd.read_excel(xlsx, sheet_name="staff_demand")
    return d.groupby("branch_id")["staff_demand"].sum().to_dict()


def rebalance(xlsx):
    staff = pd.read_excel(xlsx, sheet_name="staff")
    staff.columns = [c.strip() for c in staff.columns]
    branches = pd.read_excel(xlsx, sheet_name="branches")
    branch_cluster = dict(zip(branches["branch_id"], branches["cluster_id"]))
    acc_by_branch = staff.groupby("branch_id")["accommodation_id"].agg(lambda s: s.mode().iat[0]).to_dict()
    demand = branch_demand_hours(xlsx)
    branch_ids = sorted(demand)

    skill_col = [c for c in staff.columns if c.startswith("skill_level")][0]
    staff["_saudi"] = staff["Nationality"].astype(str).str.strip().str.lower().isin(SAUDI)
    staff["_stratum"] = list(zip(staff["role"], staff["_saudi"], staff[skill_col]))

    n_total = len(staff)
    target = largest_remainder(n_total, demand)
    assigned = defaultdict(int)

    # Managers first so the per-branch minimum is applied before other strata.
    strata = sorted(staff["_stratum"].unique(), key=lambda s: (s[0] != "Manager", -len(staff[staff["_stratum"] == s])))
    manager_quota = largest_remainder(int((staff["role"] == "Manager").sum()), target, floor_min=MIN_MANAGERS)
    manager_assigned = defaultdict(int)

    new_branch = {}
    for stratum in strata:
        members = staff[staff["_stratum"] == stratum]
        is_mgr = stratum[0] == "Manager"
        if is_mgr:
            remaining = {b: manager_quota[b] - manager_assigned[b] for b in branch_ids}
            weights = {b: max(remaining[b], 0) + 1e-9 for b in branch_ids}
        else:
            weights = {b: max(target[b] - assigned[b], 0) + 1e-9 for b in branch_ids}
        quota = largest_remainder(len(members), weights, tie_key=lambda b: -(target[b] - assigned[b]))

        movers = []
        for b in branch_ids:  # keep staff at their original branch where the quota allows
            here = members[members["branch_id"] == b]["staff_id"].tolist()
            keep = here[: quota[b]]
            for sid in keep:
                new_branch[sid] = b
            quota[b] -= len(keep)
            movers += [(sid, b) for sid in here[len(keep):]]
        for sid, old_b in movers:  # place the rest, preferring the original cluster
            open_b = [b for b in branch_ids if quota[b] > 0]
            same = [b for b in open_b if branch_cluster[b] == branch_cluster[old_b]]
            pick = max(same or open_b, key=lambda b: (quota[b], target[b] - assigned[b]))
            new_branch[sid] = pick
            quota[pick] -= 1
        for sid in members["staff_id"]:
            assigned[new_branch[sid]] += 1
            if is_mgr:
                manager_assigned[new_branch[sid]] += 1

    out = staff.drop(columns=["_saudi", "_stratum"]).copy()
    out["original_branch_id"] = out["branch_id"]
    out["branch_id"] = out["staff_id"].map(new_branch)
    out["accommodation_id"] = out["branch_id"].map(acc_by_branch)

    report = pd.DataFrame({
        "branch_id": branch_ids,
        "cluster_id": [branch_cluster[b] for b in branch_ids],
        "demand_hours": [demand[b] for b in branch_ids],
        "staff_before": [int((staff["branch_id"] == b).sum()) for b in branch_ids],
        "staff_after": [int((out["branch_id"] == b).sum()) for b in branch_ids],
        "managers_after": [int(((out["branch_id"] == b) & (out["role"] == "Manager")).sum()) for b in branch_ids],
        "saudi_after": [int(((out["branch_id"] == b) & staff["_saudi"]).sum()) for b in branch_ids],
    })
    report["hours_per_staff_before"] = report["demand_hours"] / report["staff_before"]
    report["hours_per_staff_after"] = report["demand_hours"] / report["staff_after"]
    return out, report


def apply_staff_sheet(xlsx_path, staff_df, keep_columns):
    """Overwrite the branch_id/accommodation_id cells of the staff sheet in place, leaving every other sheet untouched."""
    from openpyxl import load_workbook

    wb = load_workbook(xlsx_path)
    # The source workbook stores flags as formulas (=TRUE()). openpyxl does not keep
    # their cached results, so pandas would read them back as empty. Store the
    # computed values instead.
    cached = load_workbook(xlsx_path, data_only=True)
    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if cell.data_type == "f":
                    cell.value = cached[sheet.title][cell.coordinate].value
    ws = wb["staff"]
    header = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    col_idx = {name: header.index(name) + 1 for name in keep_columns}
    sid_col = header.index("staff_id") + 1
    lookup = staff_df.set_index("staff_id")
    for row in range(2, ws.max_row + 1):
        sid = ws.cell(row=row, column=sid_col).value
        if sid is None:
            continue
        for name in keep_columns:
            ws.cell(row=row, column=col_idx[name]).value = lookup.at[sid, name]
    wb.save(xlsx_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("xlsx")
    p.add_argument("out_csv")
    p.add_argument("--report")
    a = p.parse_args()
    new_staff, rep = rebalance(a.xlsx)
    new_staff.to_csv(a.out_csv, index=False)
    if a.report:
        rep.to_csv(a.report, index=False)
    print(rep.to_string(index=False))
    print("staff moved:", int((new_staff["branch_id"] != new_staff["original_branch_id"]).sum()), "of", len(new_staff))
