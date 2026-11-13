"""Staff-roster column generation with adaptive bucket pricing and Benders cuts, one cluster at a time.

A new method next to net_benders.py and branch_dw.py, which it imports unchanged. The staff model is decomposed with one
block per employee. A weekly roster is a nested path (van Rossum, van Lieshout and Jacquillat 2026): its subpaths are the
day rosters (one shift, or two shifts with the minimum break), which carry the daily rules; the path carries the weekly
rules (weekly hours, work days) and the rest across midnight and across the end of the week.

Column:   a weekly roster of one employee that meets every hard rule of that employee. It carries its wages and overtime,
          the employee's own violations of each priority level, the hours it covers and the stops its rider uses.
Master:   min  sum_c cost_c lam_c + theta
          s.t. sum_{c of i} lam_c = 1                                   every employee i
               coverage, onsite minimums and manager opening/closing    every branch hour / day, with their slacks
               sum_c lev_{l,c} lam_c + slacks of level l <= Vbar_l      every level l (the service limits)
               y_s >= sum_{c of i} a_{c,s} lam_c                        every rider i and stop s
               y_s <= sum_{c} a_{c,s} lam_c                             every stop s
               theta >= sum_s pi^k_s y_s                                every Benders cut k
Pricing:  per employee, labelling over the seven days. Exact by adaptive partitioning: the day rosters are split into
          buckets; each bucket is represented by its roster of least reduced cost and by the most favourable value of each
          weekly resource among its rosters. The labelling over buckets gives a lower bound on the least reduced cost; if
          the representatives form a feasible roster it is optimal, otherwise the buckets on the path are split and the
          labelling is repeated. With --pricing full every bucket holds the day rosters with equal resources
          (the enumeration of non-dominated subpaths, the benchmark of that paper).
Bound:    master LP value + sum_i min(0, lower bound of the least reduced cost of i), valid at every iteration.
Primal:   the master as an integer program over the generated columns, routed exactly; a new cut, and column generation
          resumes, as in branch_dw.py.

Usage: python staff_dw.py <data.xlsx> <benders_dir> <out_dir> --cluster <id prefix> [--time 6000 --pricing adaptive|full]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cplex
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "staff_scheduling"))
sys.path.insert(0, str(HERE.parents[2] / "bus_scheduling_routing"))
import net_benders as B  # noqa: E402
from bus_riders import branch_cluster  # noqa: E402

LEVELS = (1, 2, 3, 4)
WEEK = 168
NO_DAY0 = 24  # first-start offset of a roster without a shift on day 0 (never binds the rest across the week)


class Rules:
    """The per-employee rules of the staff model (solver_cplex_new.py), by employee category."""

    def __init__(self, data, args):
        g = lambda k, v: getattr(args, k, v)  # noqa: E731
        expat, mgr = set(data.non_saudi_staff_ids), set(data.manager_ids)
        self.brk, self.max_week, self.work_days = g("min_break_hours", 4), g("max_weekly_hours", 78), 7 - g("days_off", 1)
        self.max_daily_all = 16
        assert g("max_shift_hours", 16) <= self.max_daily_all, "a single shift longer than the daily cap"
        self.cat = {}
        for i in data.staff_ids:
            if i not in expat and i not in mgr:      # Saudi crew
                r = dict(min_shift=g("min_shift_saudi_not_mgr", 6), max_shift=None, soft_shift=g("max_shift_saudi_not_mgr_soft", 6),
                         max_daily=None, soft_daily=g("max_daily_saudi_not_mgr_soft", 6), daily_level=2,
                         ot_from=g("min_daily_saudi_not_mgr", 6), min_daily=g("min_daily_saudi_not_mgr", 6))
            elif i not in expat:                     # Saudi manager
                r = dict(min_shift=g("min_shift_saudi_mgr", 8), max_shift=None, soft_shift=None, max_daily=None,
                         soft_daily=g("max_daily_saudi_mgr_soft", 8), daily_level=3, ot_from=None, min_daily=g("min_daily_saudi_mgr", 8))
            elif i in mgr:                           # expatriate manager
                r = dict(min_shift=g("min_shift_expat", 9), max_shift=None, soft_shift=g("max_shift_expat_mgr_soft", 9), max_daily=None,
                         soft_daily=g("max_daily_expat_mgr_soft", 9), daily_level=3, ot_from=None, min_daily=g("min_daily_expat", 9))
            else:                                    # expatriate crew
                r = dict(min_shift=g("min_shift_expat", 9), max_shift=g("max_shift_expat_not_mgr_hard", 16),
                         soft_shift=g("max_shift_expat_not_mgr_soft", 12), max_daily=g("max_daily_expat_not_mgr_hard", 16),
                         soft_daily=g("max_daily_expat_not_mgr_soft", 12), daily_level=3, ot_from=g("min_daily_expat", 9),
                         min_daily=g("min_daily_expat", 9))
            r["mgr"] = i in mgr
            self.cat[i] = r
        assert not g("enforce_bus", False)


class DayRosters:
    """The day rosters of one rule set (shared by every employee of the same category): one shift, or two with the break."""

    def __init__(self, shifts, rules, r):
        self.days = []
        for d in range(7):
            sh = sorted((s for s in shifts if s["start"] // 24 == d and s["duration"] >= r["min_shift"]
                         and (r["max_shift"] is None or s["duration"] <= r["max_shift"])), key=lambda s: s["start"])
            cap = min(rules.max_daily_all, r["max_daily"] or rules.max_daily_all)
            rows = [(k, -1) for k in range(len(sh)) if r["min_daily"] <= sh[k]["duration"] <= cap]
            for k, a in enumerate(sh):
                for j in range(k + 1, len(sh)):
                    b = sh[j]
                    if b["start"] >= a["end"] + rules.brk and r["min_daily"] <= a["duration"] + b["duration"] <= cap:
                        rows.append((k, j))
            assert 3 * r["min_shift"] > cap, "three shifts a day are possible; extend DayRosters"
            a_idx = np.array([p for p, _ in rows], dtype=int)
            b_idx = np.array([q for _, q in rows], dtype=int)
            dur = np.array([s["duration"] for s in sh] + [0])
            start = np.array([s["start"] for s in sh] + [0])
            end = np.array([s["end"] for s in sh] + [0])
            h = dur[a_idx] + dur[b_idx]                      # index -1 picks the trailing 0
            last_end = np.where(b_idx >= 0, end[b_idx], end[a_idx])
            soft_shift = 0 if r["soft_shift"] is None else (np.maximum(0, dur[a_idx] - r["soft_shift"])
                                                             + np.where(b_idx >= 0, np.maximum(0, dur[b_idx] - r["soft_shift"]), 0))
            v_daily = np.maximum(0, h - r["soft_daily"])
            self.days.append(dict(
                shifts=[s["id"] for s in sh], a=a_idx, b=b_idx, h=h,
                f=start[a_idx] - 24 * d,                                      # first start, offset in the day
                ex=np.maximum(0, last_end + rules.brk - 24 * (d + 1)),          # rest that runs into the next day
                ot=np.maximum(0, h - r["ot_from"]) if r["ot_from"] is not None else np.zeros(len(rows)),
                lev2=v_daily if r["daily_level"] == 2 else np.zeros(len(rows)),
                lev3=(v_daily if r["daily_level"] == 3 else 0) + soft_shift))


class Pricer:
    """Exact least reduced cost of one employee's weekly roster by labelling over buckets of day rosters."""

    def __init__(self, dr, rules, mode):
        self.dr, self.rules, self.mode = dr, rules, mode
        self.buckets = []
        for day in dr.days:
            n = len(day["h"])
            if mode == "full" or n == 0:
                key = day["h"] * 10000 + np.minimum(day["f"], 24) * 100 + day["ex"]
                _u, inv = np.unique(key, return_inverse=True)
                self.buckets.append([np.flatnonzero(inv == k) for k in range(inv.max() + 1)] if n else [])
            else:
                self.buckets.append([np.arange(n)])

    def _label(self, reps, opt, g_w):
        """Labelling over the days. reps[d]: list of (rc, h, f, ex, bucket index) per bucket. Returns (value, choice)."""
        H = self.rules.max_week
        labels = {(0, 0, NO_DAY0): [(0, 0.0, ())]}  # (work days, rest carried, first start on day 0) -> [(hours, rc, path)]
        for d in range(7):
            new = {}

            def put(key, hrs, rc, path):
                lst = new.setdefault(key, [])
                for k, (h2, r2, _p) in enumerate(lst):
                    if h2 <= hrs and r2 <= rc + 1e-12:
                        return
                lst[:] = [e for e in lst if not (hrs <= e[0] and rc <= e[1] + 1e-12)]
                lst.append((hrs, rc, path))

            for (w, ex, ff), lst in labels.items():
                for hrs, rc, path in lst:
                    put((w, 0, ff), hrs, rc, path + (None,))
                    if w >= self.rules.work_days:
                        continue
                    for (brc, bh, bf, bex, k) in reps[d]:
                        if bf < ex or hrs + bh > H:
                            continue
                        put((w + 1, bex, bf if d == 0 else ff), hrs + bh, rc + brc, path + (k,))
            labels = new
        best, choice = math.inf, None
        for (w, ex, ff), lst in labels.items():
            if ex > ff:  # the rest after day 6 runs into the first shift of day 0
                continue
            for hrs, rc, path in lst:
                v = rc + g_w[w]
                if v < best:
                    best, choice = v, path
        return best, choice

    def price(self, rc_day, g_w, max_rounds=100000):
        """rc_day[d]: reduced cost of every day roster of day d. Returns (lower bound, roster or None, rounds)."""
        days = self.dr.days
        for rnd in range(1, max_rounds + 1):
            reps, info = [], []
            for d, day in enumerate(days):
                rd, ri = [], []
                for k, idx in enumerate(self.buckets[d]):
                    j = idx[np.argmin(rc_day[d][idx])]
                    rd.append((rc_day[d][j], day["h"][idx].min(), day["f"][idx].max(), day["ex"][idx].min(), k))
                    ri.append(j)
                reps.append(rd)
                info.append(ri)
            low, choice = self._label(reps, True, g_w)
            if choice is None:
                return low, None, rnd
            pick = [None if k is None else info[d][k] for d, k in enumerate(choice)]
            bad = self._violations(pick)
            if not bad:
                return low, pick, rnd
            split = False
            for d, attr in bad:
                k = choice[d]
                idx = self.buckets[d][k]
                v = days[d][attr][idx]
                if v.min() == v.max():
                    continue
                cut = np.median(v)
                lo, hi = idx[v <= cut], idx[v > cut]
                if len(hi) == 0:
                    lo, hi = idx[v < v.max()], idx[v == v.max()]
                self.buckets[d][k] = lo
                self.buckets[d].append(hi)
                split = True
            assert split, "no bucket to split"
        return low, None, max_rounds

    def _violations(self, pick):
        """The (day, attribute) pairs whose actual values break the path built from the representatives."""
        days, bad = self.dr.days, []
        on = [d for d in range(7) if pick[d] is not None]
        if sum(days[d]["h"][pick[d]] for d in on) > self.rules.max_week:
            bad += [(d, "h") for d in on]
        for d in on:
            nxt = (d + 1) % 7
            if pick[nxt] is not None and days[d]["ex"][pick[d]] > days[nxt]["f"][pick[nxt]]:
                bad += [(d, "ex"), (nxt, "f")]
        return bad


class Problem:
    """The cluster: the employees, their rules and day rosters, the coupling rows and the stops."""

    def __init__(self, prob, mode):
        self.prob, d = prob, prob.data
        self.data = d
        self.rules = Rules(d, prob.args)
        self.staff = list(d.staff_ids)
        self.branch = {i: d.staff_branch_map[i] for i in self.staff}
        self.not_new = set(d.non_new_staff_ids)
        dr_cache = {}
        self.dr = {}
        for i in self.staff:
            r = self.rules.cat[i]
            key = (r["min_shift"], r["max_shift"], r["soft_shift"], r["max_daily"], r["soft_daily"], r["daily_level"], r["ot_from"], r["min_daily"])
            if key not in dr_cache:
                dr_cache[key] = DayRosters(d.shifts, self.rules, r)
            self.dr[i] = dr_cache[key]
        self.pricer = {i: Pricer(self.dr[i], self.rules, mode) for i in self.staff}
        self.shift = prob.shift
        self.active = [(b, t) for b in d.branch_ids for t in d.time_slots if d.branch_demand.get((b, t), 0) > 0]
        work_t = {t: False for t in d.time_slots}
        for s in d.shifts:
            for t in s["covers"]:
                work_t[t] = True
        staff_b = {b: [i for i in self.staff if self.branch[i] == b] for b in d.branch_ids}
        self.hard = [(b, t) for (b, t) in self.active if work_t[t] and staff_b[b]]
        mgr_b = {b: [i for i in staff_b[b] if self.rules.cat[i]["mgr"]] for b in d.branch_ids}
        self.open_close, self.lev_const = [], {l: 0.0 for l in LEVELS}
        for b, lim in d.branch_operation_limits.items():
            if not mgr_b.get(b):
                self.lev_const[3] += 2 * len(d.days)   # no manager: both slacks are at least 1 every day
                continue
            for day, tt in lim.items():
                self.open_close.append((b, tt["first_hour"]))
                self.open_close.append((b, tt["last_hour"]))
        self.rider = {}
        for (b, a), riders in prob.net.riders.items():
            for i in riders:
                self.rider[i] = (b, a)
        self.stops = [s for s in prob.possible_stops()]

    def shifts_by_day(self, i, x_i):
        """The roster x_i (shift ids) as one day-roster index per day, or None if it breaks a hard rule of i."""
        dr = self.dr[i]
        pick = []
        for d, day in enumerate(dr.days):
            ids = sorted((s for s in x_i if self.shift[s]["start"] // 24 == d), key=lambda s: self.shift[s]["start"])
            if not ids:
                pick.append(None)
                continue
            pos = {s: k for k, s in enumerate(day["shifts"])}
            if any(s not in pos for s in ids) or len(ids) > 2:
                return None
            a, b = pos[ids[0]], pos[ids[1]] if len(ids) == 2 else -1
            j = np.flatnonzero((day["a"] == a) & (day["b"] == b))
            if len(j) != 1:
                return None
            pick.append(int(j[0]))
        return pick

    def column(self, i, pick):
        """Cost, own level violations, covered hours and stops of the roster of i given as one day roster per day."""
        dr, r = self.dr[i], self.rules.cat[i]
        wage, ot = self.data.wage_reg[i], self.data.wage_ot[i]
        x, hours, cost, lev = [], [], 0.0, {l: 0.0 for l in LEVELS}
        on = 0
        for d, j in enumerate(pick):
            if j is None:
                continue
            day = dr.days[d]
            on += 1
            for k in (day["a"][j], day["b"][j]):
                if k >= 0:
                    s = day["shifts"][k]
                    x.append(s)
                    hours.extend(self.shift[s]["covers"])
            cost += wage * day["h"][j] + ot * day["ot"][j]
            lev[2] += day["lev2"][j]
            lev[3] += day["lev3"][j]
        lev[1] += max(0, 1 - on)
        lev[3] += max(0, self.rules.work_days - on)
        stops = set()
        if i in self.rider:
            b, a = self.rider[i]
            for s in x:
                stops.add((b, a, self.shift[s]["start"] % WEEK))
                stops.add((b, a, self.shift[s]["end"] % WEEK))
        return dict(i=i, x=frozenset((i, s) for s in x), pick=tuple(pick), cost=float(cost), lev={l: float(v) for l, v in lev.items()},
                    hours=sorted(set(hours)), stops=frozenset(stops))


class Master:
    """Restricted master LP over the staff columns, through the CPLEX callable library."""

    def __init__(self, P, limits):
        self.P = P
        c = self.c = cplex.Cplex()
        for stream in (c.set_log_stream, c.set_results_stream, c.set_warning_stream):
            stream(None)
        c.objective.set_sense(c.objective.sense.minimize)
        dem = P.data.branch_demand
        g = lambda k, v: getattr(P.prob.args, k, v)  # noqa: E731
        rows = [(f"conv_{i}", "E", 1.0) for i in P.staff]
        for (b, t) in P.active:
            rows += [(f"dem_{b}_{t}", "G", float(dem[b, t])), (f"nn_{b}_{t}", "G", float(g("min_onsite_not_new_soft", 1))),
                     (f"all_{b}_{t}", "G", float(g("min_onsite_all_soft", 2))), (f"mgr_{b}_{t}", "G", float(g("min_onsite_mgr_soft", 1))),
                     (f"nm_{b}_{t}", "G", float(g("min_onsite_not_mgr_soft", 2)))]
        rows += [(f"hard_{b}_{t}", "G", float(g("min_onsite_all_hard", 1))) for (b, t) in P.hard]
        # opening and closing rows; a first and a last hour on the same row name would merge, so they are counted
        self.oc = {}
        for (b, t) in P.open_close:
            self.oc[b, t] = self.oc.get((b, t), 0) + 1
        rows += [(f"oc_{b}_{t}", "G", float(n) * g("min_onsite_open_mgr_soft", 1)) for (b, t), n in self.oc.items()]
        assert g("min_onsite_open_mgr_soft", 1) == g("min_onsite_close_mgr_soft", 1) == 1, "opening/closing rows assume 1 manager"
        rows += [(f"lev_{l}", "L", float(limits[l]) - P.lev_const[l]) for l in LEVELS]
        self.ry = [(i, s) for s in P.stops for i in P.prob.net.riders[s[0], s[1]]]
        rows += [(f"ry_{i}_{k}", "G", 0.0) for k, (i, s) in enumerate(self.ry)]
        rows += [(f"ld_{k}", "L", 0.0) for k in range(len(P.stops))]
        c.linear_constraints.add(names=[r[0] for r in rows], senses="".join(r[1] for r in rows), rhs=[r[2] for r in rows])
        self.row = {r[0]: k for k, r in enumerate(rows)}
        self.ry_row = {(i, s): self.row[f"ry_{i}_{k}"] for k, (i, s) in enumerate(self.ry)}
        self.ld_row = {s: self.row[f"ld_{k}"] for k, s in enumerate(P.stops)}
        # slacks with their level
        names, cols, = [], []
        for (b, t) in P.active:
            for pre, lev in (("dem", 1), ("nn", 2), ("all", 3), ("mgr", 3), ("nm", 4)):
                names.append(f"s{pre}_{b}_{t}")
                cols.append(cplex.SparsePair([self.row[f"{pre}_{b}_{t}"], self.row[f"lev_{lev}"]], [1.0, 1.0]))
        for (b, t), n in self.oc.items():
            names.append(f"soc_{b}_{t}")
            cols.append(cplex.SparsePair([self.row[f"oc_{b}_{t}"], self.row["lev_3"]], [1.0, 1.0]))
        c.variables.add(obj=[0.0] * len(names), lb=[0.0] * len(names), names=names, columns=cols)
        self.n_slack = len(names)
        # stops y
        ycols = []
        for s in P.stops:
            ind = [self.ld_row[s]] + [self.ry_row[i, s] for i in P.prob.net.riders[s[0], s[1]]]
            ycols.append(cplex.SparsePair(ind, [1.0] * len(ind)))
        self.y0 = c.variables.get_num()
        c.variables.add(obj=[0.0] * len(P.stops), lb=[0.0] * len(P.stops), ub=[1.0] * len(P.stops),
                        names=[f"y_{k}" for k in range(len(P.stops))], columns=ycols)
        self.theta = c.variables.get_num()
        c.variables.add(obj=[1.0], lb=[0.0], names=["theta"])
        self.cols, self.cuts, self.keys = [], [], set()
        self.oc_of = {}
        for (b, t) in self.oc:
            self.oc_of.setdefault(b, set()).add(t)
        self.active = set(P.active)
        self.hard = set(P.hard)

    def add_cut(self, prices):
        k = len(self.cuts)
        self.cuts.append(prices)
        ind = [self.theta] + [self.y0 + j for j, s in enumerate(self.P.stops) if prices.get(s, 0.0) != 0.0]
        val = [1.0] + [-prices[s] for s in self.P.stops if prices.get(s, 0.0) != 0.0]
        self.c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses="G", rhs=[0.0], names=[f"cut_{k}"])

    def add_column(self, col):
        key = (col["i"], col["x"])
        if key in self.keys:
            return False
        self.keys.add(key)
        P, i = self.P, col["i"]
        b, mgr, nn = P.branch[i], P.rules.cat[i]["mgr"], i in P.not_new
        ind, val = [self.row[f"conv_{i}"]], [1.0]
        for t in col["hours"]:
            if (b, t) in self.active:
                for pre in ("dem", "all") + (("mgr",) if mgr else ("nm",)) + (("nn",) if nn else ()):
                    ind.append(self.row[f"{pre}_{b}_{t}"])
                    val.append(1.0)
                if (b, t) in self.hard:
                    ind.append(self.row[f"hard_{b}_{t}"])
                    val.append(1.0)
            if mgr and t in self.oc_of.get(b, ()):
                ind.append(self.row[f"oc_{b}_{t}"])
                val.append(float(self.oc[b, t]))  # a first and a last hour at the same hour: both rows in one
        for l in LEVELS:
            if col["lev"][l]:
                ind.append(self.row[f"lev_{l}"])
                val.append(col["lev"][l])
        for s in col["stops"]:
            ind += [self.ry_row[i, s], self.ld_row[s]]
            val += [-1.0, -1.0]
        self.cols.append(col)
        self.c.variables.add(obj=[col["cost"]], lb=[0.0], names=[f"lam_{len(self.cols) - 1}"], columns=[cplex.SparsePair(ind, val)])
        return True

    def solve_lp(self):
        self.c.set_problem_type(self.c.problem_type.LP)
        self.c.solve()
        st = self.c.solution.get_status()
        assert self.c.solution.is_primal_feasible(), f"master LP status {st}"
        return self.c.solution.get_objective_value(), np.array(self.c.solution.get_dual_values())

    def reduced_costs(self, i, pi):
        """Per day, the reduced cost of every day roster of employee i, and the work-day term g(w), at duals pi."""
        P = self.P
        b, mgr, nn = P.branch[i], P.rules.cat[i]["mgr"], i in P.not_new
        cov = np.zeros(WEEK)
        for t in range(WEEK):
            if (b, t) in self.active:
                cov[t] = sum(pi[self.row[f"{pre}_{b}_{t}"]] for pre in ("dem", "all") + (("mgr",) if mgr else ("nm",)) + (("nn",) if nn else ()))
                if (b, t) in self.hard:
                    cov[t] += pi[self.row[f"hard_{b}_{t}"]]
            if mgr and t in self.oc_of.get(b, ()):
                cov[t] += self.oc[b, t] * pi[self.row[f"oc_{b}_{t}"]]
        stop_w = np.zeros(WEEK)
        if i in P.rider:
            ba = P.rider[i]
            for t in range(WEEK):
                s = (ba[0], ba[1], t)
                if s in self.ld_row:
                    stop_w[t] = pi[self.ry_row[i, s]] + pi[self.ld_row[s]]
        p = {l: -pi[self.row[f"lev_{l}"]] for l in LEVELS}
        wage, ot = P.data.wage_reg[i], P.data.wage_ot[i]
        out = []
        for day in P.dr[i].days:
            sw = np.array([-cov[list(P.shift[s]["covers"])].sum() + stop_w[P.shift[s]["start"] % WEEK] + stop_w[P.shift[s]["end"] % WEEK]
                           for s in day["shifts"]] + [0.0])
            out.append(sw[day["a"]] + sw[day["b"]] + wage * day["h"] + ot * day["ot"] + p[2] * day["lev2"] + p[3] * day["lev3"])
        wd = P.rules.work_days
        g_w = [p[1] * max(0, 1 - w) + p[3] * max(0, wd - w) - pi[self.row[f"conv_{i}"]] for w in range(wd + 1)]
        return out, g_w

    def solve_ip(self, time_limit, threads):
        n = self.c.variables.get_num()
        types = ["C"] * self.n_slack + ["B"] * len(self.P.stops) + ["C"] + ["B"] * len(self.cols)
        self.c.variables.set_types(list(zip(range(n), types)))
        self.c.parameters.timelimit.set(time_limit)
        self.c.parameters.threads.set(threads)
        self.c.parameters.mip.tolerances.mipgap.set(1e-6)
        self.c.solve()
        ok = self.c.solution.is_primal_feasible()
        v = self.c.solution.get_values() if ok else []
        chosen = [self.cols[j] for j in range(len(self.cols)) if v[self.theta + 1 + j] > 0.5] if ok else []
        value = self.c.solution.get_objective_value() if ok else math.inf
        self.c.variables.set_types([(j, "C") for j in range(n)])
        self.c.set_problem_type(self.c.problem_type.LP)
        return chosen, value


def run(prob, x0, limits, time_budget, mode, threads, log, warm=()):
    t0 = time.time()
    P = Problem(prob, mode)
    master = Master(P, limits)
    evaluator = B.StaffModel(prob.args, prob.data)
    vals0, plan0, z0 = prob.plan_cost(evaluator, x0)
    best = dict(z=z0, x=set(x0), vals=vals0, plan=plan0, source="sequential")
    seen = set()

    def add_schedule(name, x):
        n = 0
        for i in P.staff:
            pick = P.shifts_by_day(i, [s for j, s in x if j == i])
            assert pick is not None, f"{name}: roster of {i} breaks a hard rule"
            n += master.add_column(P.column(i, pick))
        stops = frozenset(prob.net.demand(prob.events(x))[2])
        if stops not in seen:
            seen.add(stops)
            master.add_cut(prob.cut(x)[0])
        return n

    add_schedule("x0", x0)
    for name, xw in warm:
        vals, plan, z = prob.plan_cost(evaluator, xw)
        ok = all(vals["levels"][l] <= limits[l] + 1e-6 for l in LEVELS)
        log(f"warm {name}: routed Z {z:,.2f}  within limits {ok}")
        if ok and z < best["z"]:
            best = dict(z=z, x=set(xw), vals=vals, plan=plan, source=f"warm {name}")
        add_schedule(name, xw)
    value, _pi = master.solve_lp()
    log(f"setup {time.time() - t0:.0f}s  sequential Z {z0:,.2f}  master LP at the start {value:,.2f}  staff {len(P.staff)}  "
        f"rows {master.c.linear_constraints.get_num()}  limits {limits}")
    lower, history, outer = -math.inf, [], 0
    while time.time() - t0 < time_budget:
        while time.time() - t0 < time_budget:
            value, pi = master.solve_lp()
            tp = time.time()
            bound, added, rounds = value, 0, []
            for i in P.staff:
                rc_day, g_w = master.reduced_costs(i, pi)
                low, pick, rnd = P.pricer[i].price(rc_day, g_w)
                rounds.append(rnd)
                bound += min(0.0, low)
                if pick is not None and low < -1e-6 * max(1.0, abs(value)):
                    added += master.add_column(P.column(i, pick))
            lower = max(lower, bound)
            buckets = sum(len(bk) for i in P.staff for bk in P.pricer[i].buckets)
            history.append(dict(outer=outer, lp=value, lb=lower, bound=bound, added=added, columns=len(master.cols), cuts=len(master.cuts),
                                best=best["z"], seconds=time.time() - t0, pricing_seconds=time.time() - tp,
                                rounds_mean=float(np.mean(rounds)), rounds_max=int(max(rounds)), buckets=buckets))
            log(f"outer {outer}  LP {value:,.2f}  bound {bound:,.2f}  LB {lower:,.2f}  best {best['z']:,.2f}  "
                f"gap {(best['z'] - lower) / best['z']:.3%}  +{added} cols ({len(master.cols)})  pricing {time.time() - tp:.1f}s "
                f"rounds {np.mean(rounds):.1f}/{max(rounds)}  buckets {buckets}  {time.time() - t0:.0f}s")
            if added == 0:
                break
        chosen, ip_value = master.solve_ip(min(600, max(30, time_budget - (time.time() - t0))), threads)
        if not chosen:
            log("integer master found no solution")
            break
        x_new = set().union(*[set(c["x"]) for c in chosen])
        vals, plan, z = prob.plan_cost(evaluator, x_new)
        ok = all(vals["levels"][l] <= limits[l] + 1e-6 for l in LEVELS)
        log(f"outer {outer}: integer master {ip_value:,.2f}  routed Z {z:,.2f}  levels {vals['levels']}  within limits {ok}")
        if ok and z < best["z"]:
            best = dict(z=z, x=x_new, vals=vals, plan=plan, source=f"outer {outer}")
        history.append(dict(outer=outer, ip=ip_value, routed=z, within=ok, best=best["z"], lb=lower, seconds=time.time() - t0))
        outer += 1
        stops = frozenset(prob.net.demand(prob.events(x_new))[2])
        if stops in seen:
            log("the integer master repeats a stop set: no new cut")
            break
        seen.add(stops)
        master.add_cut(prob.cut(x_new)[0])
        if (best["z"] - lower) / best["z"] <= 1e-4:
            break
    return best, lower, history, master


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("benders_dir", help="folder with <stem>_<cluster>_x0.csv and the Benders json (limits)")
    p.add_argument("out_dir")
    p.add_argument("--cluster", required=True)
    p.add_argument("--time", type=float, default=None, help="seconds; default: the time of the Benders run")
    p.add_argument("--pricing", choices=["adaptive", "full"], default="adaptive")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--warm", nargs="*", default=[], help="schedule csv files whose rosters start the master as columns")
    p.add_argument("--tag", default=None, help="output name suffix; default: the --pricing value")
    p.add_argument("--no-lift", action="store_true",
                   help="Benders cuts without lifting (closed stops at min(0, lifted price)): the literature method without our cut")
    a = p.parse_args()
    assert not (a.no_lift and a.warm), "--no-lift is the literature baseline: no --warm"
    t_all = time.time()
    cluster = next(c for c in set(branch_cluster(a.data).values()) if c.startswith(a.cluster))
    stem = Path(a.data).stem
    bres = json.load(open(Path(a.benders_dir) / f"{stem}_{cluster[:8]}.json"))
    prob = B.ClusterProblem(cluster, a.data, B.staff_args(), a.threads)
    if a.no_lift:
        prob.closed_price = "zero"
    x0 = B.load_schedule(Path(a.benders_dir) / f"{stem}_{cluster[:8]}_x0.csv", prob.data.staff_ids)
    limits = {int(k): v for k, v in bres["limits"].items()}
    budget = a.time if a.time else bres["benders_seconds"]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warm = [(Path(w).name, B.load_schedule(w, prob.data.staff_ids)) for w in a.warm]
    best, lower, history, master = run(prob, x0, limits, budget, a.pricing, a.threads, lambda s: print(s, flush=True), warm)
    z0 = bres["z_seq"]
    res = dict(cluster=cluster[:8], pricing=a.pricing, z_seq=z0, z=best["z"], lb=lower, gap=(best["z"] - lower) / best["z"],
               saving=(z0 - best["z"]) / z0, benders_lb=bres["lb"], benders_z=bres["z"], source=best["source"],
               wages=best["vals"]["j_reg"], overtime=best["vals"]["j_over"], levels=best["vals"]["levels"], limits=limits,
               plan=B._parts(best["plan"]), columns=len(master.cols), cuts=len(master.cuts), budget=budget,
               seconds=time.time() - t_all, history=history)
    name = f"{stem}_{cluster[:8]}_{a.tag or a.pricing}"
    with open(out / f"{name}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    B.save_schedule(best["x"], prob.shift, out / f"{name}_schedule.csv")
    print(f"DONE z {best['z']:,.2f} lb {lower:,.2f} gap {res['gap']:.3%} saving {res['saving']:.3%}  (Benders: lb {bres['lb']:,.2f} z {bres['z']:,.2f})")


if __name__ == "__main__":
    main()
