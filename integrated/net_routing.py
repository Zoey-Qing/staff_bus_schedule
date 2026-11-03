"""Bus routing on a time-space network, as in the paper's bus section.

A stop is (branch, accommodation, hour). A trip is a route with a departure minute. Every minute between the bus and
the shift change costs the riders' wages, every stop must be reached within 30 minutes of the shift change, and the
fleet follows from a flow of buses through the minutes of the week.

The LP is solved by column generation over trips and empty runs. Waiting arcs join consecutive event minutes of each
(bus type, accommodation), which gives the same LP as one arc per minute: the duals of the minutes in between are taken
from the next event minute, and they satisfy every waiting arc, so a pricing pass that finds nothing proves optimality.
"""
import math
import os
from dataclasses import dataclass, field

import cplex
import numpy as np
import pandas as pd

import instance as I
from bus_riders import branch_cluster, is_bus_rider

WEEK = 10080
# Scenario switches for the routing experiments; the defaults are the model of the paper.
WINDOW = int(os.environ.get("NET_WINDOW", "30"))
MAX_STOPS = 5
SPEED = float(os.environ.get("NET_SPEED", "40"))
STEP = int(os.environ.get("NET_STEP", "1"))
USE_THETA = os.environ.get("NET_THETA", "1") == "1"
USE_EMPTY = os.environ.get("NET_EMPTY", "1") == "1"
FIXED_SCALE = float(os.environ.get("NET_FIXED_SCALE", "1"))
EPS = 1e-7


@dataclass
class BusType:
    key: str
    capacity: int
    fixed: float
    per_km: float
    per_min: float


@dataclass(frozen=True)
class Trip:
    k: int
    acc: str
    t: int
    seq: tuple
    delta: int
    dep: int
    dur: int
    drive: float
    theta: float

    @property
    def stops(self):
        return [(b, self.acc, self.t) for b in self.seq]


@dataclass(frozen=True)
class EmptyRun:
    k: int
    a: str
    a2: str
    dep: int
    dur: int
    cost: float


class RouteTable:
    """Every route of one accommodation and bus type with every departure offset, as arrays. Costs that do not depend
    on the schedule are computed once; the hour only shifts the departure minute."""

    def __init__(self, net, acc, k):
        self.acc, self.k = acc, k
        bus = net.buses[k]
        self.branches = net.branches_of[acc]
        col = {b: j for j, b in enumerate(self.branches)}
        self.seqs, self.drive, self.taus = [], [], []
        delta, dur, start, D = [], [], [], []
        pos = 0

        def add(seq):
            nonlocal pos
            tau, prev, km = [], acc, 0.0
            elapsed = 0
            for b in seq:
                elapsed += net.tau[prev, b]
                km += net.km[prev, b]
                tau.append(elapsed)
                prev = b
            lo, hi = -WINDOW - tau[0], WINDOW - tau[-1]
            if lo > hi:
                return False
            total = elapsed + net.tau[prev, acc]
            km += net.km[prev, acc]
            ds = np.arange(lo, hi + 1)
            ds = ds[ds % STEP == 0]
            if not len(ds):
                return False
            row = np.zeros((len(ds), len(self.branches)))
            for b, x in zip(seq, tau):
                row[:, col[b]] = np.abs(ds + x)
            self.seqs.append(seq)
            self.drive.append(bus.per_km * km + bus.per_min * total)
            self.taus.append(tau)
            start.append(pos)
            delta.append(ds)
            dur.append(np.full(len(ds), total))
            D.append(row)
            pos += len(ds)
            return True

        def extend(seq):
            for b in self.branches:
                if b not in seq and len(seq) < MAX_STOPS:
                    # a longer route keeps the earlier arrivals, so a route that breaks the window ends its branch
                    s2 = seq + (b,)
                    if add(s2):
                        extend(s2)

        extend(())
        self.start = np.array(start, dtype=int)
        self.delta = np.concatenate(delta)
        self.dur = np.concatenate(dur)
        self.D = np.vstack(D)
        self.drive = np.array(self.drive)
        self.drive_entry = np.repeat(self.drive, np.diff(np.append(self.start, len(self.delta))))
        self.mem = np.zeros((len(self.seqs), len(self.branches)), dtype=bool)
        for r, seq in enumerate(self.seqs):
            for b in seq:
                self.mem[r, col[b]] = True
        self.col = col


class Network:
    """One cluster: its locations, bus types, riders and route tables."""

    def __init__(self, cluster, data_file):
        self.cluster = cluster
        x = pd.ExcelFile(data_file)
        br = x.parse("branches")
        br.columns = [c.strip() for c in br.columns]
        lon = "longitude" if "longitude" in br.columns else "longitute"
        cl = branch_cluster(data_file)
        pos = {r.branch_id: (r.latitude, getattr(r, lon)) for r in br.itertuples() if cl[r.branch_id] == cluster}
        acc = x.parse("accommodation")
        acc = acc[acc.cluster_id.astype(str) == cluster]
        self.accs = sorted(str(a) for a in acc.id)
        pos.update({str(r.id): (r.latitude, r.longitude) for r in acc.itertuples()})
        self.km, self.tau = {}, {}
        for u in pos:
            for v in pos:
                d = I.haversine_km(*pos[u], *pos[v])
                self.km[u, v] = d
                self.tau[u, v] = int(math.ceil(60.0 * d / SPEED - 1e-9))
        # buses of one type in a cluster may list different fixed costs: a type uses their mean, so the result does
        # not depend on the order of the rows
        veh = x.parse("vehicles")
        veh = veh[veh.cluster_id.astype(str) == cluster].merge(x.parse("bus_info")[["id", "fixed_cost"]], on="id")
        veh["kind"] = ["Large Bus" if "large" in str(s).lower() else "Small Bus" for s in veh.vehicle_type]
        mean_fixed = veh.groupby("kind").fixed_cost.mean().to_dict()
        self.buses = [BusType(v.key, v.capacity, mean_fixed[v.kind] * I.FIXED_COST_PERIOD_FRACTION * float(os.environ.get("BUS_COST_SCALE", "1")) * FIXED_SCALE, v.per_km, v.per_min)
                      for v in I._vehicle_types(data_file).values() if v.cluster == cluster]
        staff = x.parse("staff")
        staff = staff[is_bus_rider(staff["Nationality"])]
        staff = staff[staff.branch_id.isin(list(cl)) & staff.branch_id.map(cl).eq(cluster)]
        self.home = {r.staff_id: (r.branch_id, str(r.accommodation_id)) for r in staff.itertuples()}
        self.wage = {r.staff_id: r.hourly_wage / 60.0 for r in staff.itertuples()}
        self.riders = {}
        for i, key in self.home.items():
            self.riders.setdefault(key, []).append(i)
        self.low_wage = {key: min(self.wage[i] for i in rs) for key, rs in self.riders.items()}
        self.branches_of = {}
        for b, a in self.riders:
            self.branches_of.setdefault(a, []).append(b)
        self.branches_of = {a: sorted(bs) for a, bs in self.branches_of.items()}
        self.tables = {}

    def table(self, a, k):
        if (a, k) not in self.tables:
            self.tables[a, k] = RouteTable(self, a, k)
        return self.tables[a, k]

    def demand(self, events):
        """events: (staff, start hour, end hour) of the assigned shifts. Returns q, p and the wages e per stop."""
        q, p, e = {}, {}, {}
        for i, s, f in events:
            if i not in self.home:
                continue
            b, a = self.home[i]
            for t, d in ((s % 168, q), (f % 168, p)):
                d[b, a, t] = d.get((b, a, t), 0) + 1
                e[b, a, t] = e.get((b, a, t), 0.0) + self.wage[i]
        return q, p, e


def fits(seq, a, t, q, p, cap):
    load = sum(q.get((b, a, t), 0) for b in seq)
    if load > cap:
        return False
    for b in seq:
        load += p.get((b, a, t), 0) - q.get((b, a, t), 0)
        if load > cap:
            return False
    return True


class Master:
    """The restricted LP or integer program over the generated trips and empty runs."""

    def __init__(self, net, stops, rhs):
        self.net, self.stops, self.rhs = net, list(stops), rhs
        self.trips, self.runs = {}, {}

    def add_trip(self, h):
        new = h not in self.trips
        self.trips[h] = True
        return new

    def add_run(self, g):
        new = g not in self.runs
        self.runs[g] = True
        return new

    def _events(self):
        ev = {}
        for h in self.trips:
            s = ev.setdefault((h.k, h.acc), set())
            s.add(h.dep)
            s.add((h.dep + h.dur) % WEEK)
        for g in self.runs:
            ev.setdefault((g.k, g.a), set()).add(g.dep)
            ev.setdefault((g.k, g.a2), set()).add((g.dep + g.dur) % WEEK)
        return {key: sorted(v) for key, v in ev.items()}

    def build(self, integer=False, threads=0):
        ev = self._events()
        row = {("c", s): j for j, s in enumerate(self.stops)}
        for key, minutes in sorted(ev.items()):
            for w in minutes:
                row["f", key, w] = len(row)
        for k in range(len(self.net.buses)):
            row["n", k] = len(row)
        m = len(self.stops)
        rhs = [float(self.rhs[s]) for s in self.stops] + [0.0] * (len(row) - m)
        sense = "E" * (len(row) - len(self.net.buses)) + "L" * len(self.net.buses)
        cols, obj, types, names = [], [], [], []

        def column(pairs, cost, kind, name):
            cols.append(cplex.SparsePair(ind=[i for i, _ in pairs], val=[v for _, v in pairs]))
            obj.append(cost)
            types.append(kind)
            names.append(name)

        for h in self.trips:
            pairs = [(row["c", s], 1.0) for s in h.stops]
            ret = h.dep + h.dur
            pairs += [(row["f", (h.k, h.acc), ret % WEEK], 1.0), (row["f", (h.k, h.acc), h.dep], -1.0)]
            if ret >= WEEK:
                pairs.append((row["n", h.k], 1.0))
            column(pairs, h.drive + h.theta, "B", h)
        for g in self.runs:
            ret = g.dep + g.dur
            pairs = [(row["f", (g.k, g.a2), ret % WEEK], 1.0), (row["f", (g.k, g.a), g.dep], -1.0)]
            if ret >= WEEK:
                pairs.append((row["n", g.k], 1.0))
            column(pairs, g.cost, "C", g)
        for key, minutes in ev.items():
            for j, w in enumerate(minutes):
                nxt = minutes[(j + 1) % len(minutes)]
                pairs = []
                if len(minutes) > 1:
                    pairs = [(row["f", key, nxt], 1.0), (row["f", key, w], -1.0)]
                if j == len(minutes) - 1:
                    pairs.append((row["n", key[0]], 1.0))
                column(pairs, 0.0, "C", ("wait", key, w))
        for k, bus in enumerate(self.net.buses):
            column([(row["n", k], -1.0)], bus.fixed, "I", ("n", k))
        c = cplex.Cplex()
        c.set_log_stream(None)
        c.set_results_stream(None)
        c.set_warning_stream(None)
        c.set_error_stream(None)
        if threads:
            c.parameters.threads.set(threads)
        c.objective.set_sense(c.objective.sense.minimize)
        c.linear_constraints.add(rhs=rhs, senses=sense)
        lb = [0.0] * len(cols)
        ub = [cplex.infinity] * len(cols)
        if integer:
            ub = [1.0 if tp == "B" else cplex.infinity for tp in types]
            c.variables.add(obj=obj, lb=lb, ub=ub, types=types, columns=cols)
        else:
            c.variables.add(obj=obj, lb=lb, ub=ub, columns=cols)
        return c, row, names, ev

    def solve_lp(self, threads=0):
        c, row, names, ev = self.build(False, threads)
        c.solve()
        y = c.solution.get_dual_values()
        pi = {s: y[row["c", s]] for s in self.stops}
        mu = [y[row["n", k]] for k in range(len(self.net.buses))]
        rho = {}
        for key, minutes in ev.items():
            rho[key] = (np.array(minutes), np.array([y[row["f", key, w]] for w in minutes]))
        return c.solution.get_objective_value(), pi, rho, mu

    def solve_ip(self, time_limit=120, threads=0, gap=1e-5):
        c, row, names, ev = self.build(True, threads)
        c.parameters.timelimit.set(time_limit)
        c.parameters.mip.tolerances.mipgap.set(gap)
        c.solve()
        x = c.solution.get_values()
        return c.solution.get_objective_value(), {n: v for n, v in zip(names, x) if v > 1e-6}, c.solution.MIP.get_best_objective()


def extend_duals(rho, mu, net):
    """rho at every minute and accommodation: the value of the next event minute, or after the last event the first
    event's value plus mu, so that every waiting arc is satisfied."""
    out = {}
    for k in range(len(net.buses)):
        for a in net.accs:
            arr = np.zeros(WEEK)
            if (k, a) in rho:
                minutes, vals = rho[k, a]
                idx = np.searchsorted(minutes, np.arange(WEEK))
                wrap = idx == len(minutes)
                arr = vals[np.where(wrap, 0, idx)] + np.where(wrap, mu[k], 0.0)
            out[k, a] = arr
    return out


def rho_of(arr, mu, dep, dur):
    """Net price of the buses for a trip leaving at dep: rho(return) - rho(leave), with mu if it crosses the week start."""
    ret = dep + dur
    cross = ret >= WEEK
    return arr[ret % WEEK] - arr[dep] + np.where(cross, mu, 0.0)


def price(net, stops_by, pi, rho, mu, cost_of, q=None, p=None, keep=20):
    """Exact pricing. cost_of(a, t) gives the wage per minute of each branch of the table. Returns trips with negative
    reduced cost, the best departure of each route, at most `keep` per accommodation, bus type and hour."""
    arrs = extend_duals(rho, mu, net)
    new = []
    net.min_run_rc = 0.0
    for (a, t), open_b in stops_by.items():
        for k in range(len(net.buses)):
            tb = net.table(a, k)
            isopen = np.array([b in open_b for b in tb.branches])
            valid = ~(tb.mem & ~isopen).any(axis=1)
            if not valid.any():
                continue
            base = tb.drive_entry + tb.D @ cost_of(a, t, tb)
            dep = (60 * t + tb.delta) % WEEK
            val = base - rho_of(arrs[k, a], mu[k], dep, tb.dur)
            best = np.minimum.reduceat(val, tb.start)
            gain = tb.mem @ np.array([pi.get((b, a, t), 0.0) for b in tb.branches])
            rc = best - gain
            rc[~valid] = np.inf
            order = np.argsort(rc)
            taken = 0
            ends = np.append(tb.start, len(tb.delta))
            for r in order:
                if rc[r] >= -EPS or taken >= keep:
                    break
                seq = tb.seqs[r]
                if q is not None and not fits(seq, a, t, q, p, net.buses[k].capacity):
                    continue
                lo, hi = tb.start[r], ends[r + 1]
                # prefer the latest of the tied departures: riders wait least
                j = lo + int(np.flatnonzero(val[lo:hi] <= val[lo:hi].min() + 1e-9)[-1])
                new.append(Trip(k, a, t, seq, int(tb.delta[j]), int(dep[j]), int(tb.dur[j]), float(tb.drive[r]),
                                float(base[j] - tb.drive[r])))
                taken += 1
    if len(net.accs) > 1 and USE_EMPTY:
        for k, bus in enumerate(net.buses):
            for a in net.accs:
                for a2 in net.accs:
                    if a == a2:
                        continue
                    dur = net.tau[a, a2]
                    cost = bus.per_km * net.km[a, a2] + bus.per_min * dur
                    deps = np.arange(WEEK)
                    ret = deps + dur
                    val = cost - (arrs[k, a2][ret % WEEK] - arrs[k, a][deps] + np.where(ret >= WEEK, mu[k], 0.0))
                    net.min_run_rc = min(net.min_run_rc, float(val.min()))
                    for w in np.argsort(val)[:5]:
                        if val[w] >= -EPS:
                            break
                        new.append(EmptyRun(k, a, a2, int(w), dur, cost))
    return new


def direct_trips(net, stops, cost_fn, q=None, p=None):
    """A direct trip on every bus type that fits, arriving on time, or as near as the departure step allows."""
    out = []
    for b, a, t in stops:
        for k, bus in enumerate(net.buses):
            if q is not None and not fits((b,), a, t, q, p, bus.capacity):
                continue
            tb = net.table(a, k)
            r = tb.seqs.index((b,))
            tau = tb.taus[r][0]
            total = net.tau[a, b] + net.tau[b, a]
            d = -tau if STEP == 1 else STEP * round(-tau / STEP)
            theta = 0.0 if d == -tau else float(cost_fn(a, t, tb)[tb.col[b]]) * abs(d + tau)
            out.append(Trip(k, a, t, (b,), d, (60 * t + d) % WEEK, total, float(tb.drive[r]), theta))
    return out


def column_generation(net, stops, rhs, cost_of, q=None, p=None, threads=0, max_iter=2000, log=None):
    """Solve the routing LP. Returns the master with its columns, the LP value and the final duals."""
    stops = list(stops)
    stops_by = {}
    for b, a, t in stops:
        if rhs[b, a, t] > 0:
            stops_by.setdefault((a, t), set()).add(b)
    master = Master(net, stops, rhs)
    for h in direct_trips(net, [s for s in stops if rhs[s] > 0], cost_of, q, p):
        master.add_trip(h)
    value, pi, rho, mu = None, {}, {}, [0.0] * len(net.buses)
    for it in range(max_iter):
        value, pi, rho, mu = master.solve_lp(threads)
        cols = price(net, stops_by, pi, rho, mu, cost_of, q, p)
        added = 0
        for col in cols:
            added += master.add_trip(col) if isinstance(col, Trip) else master.add_run(col)
        if log:
            log(f"    iter {it}: LP {value:,.2f}  trips {len(master.trips)}  runs {len(master.runs)}  new {added}")
        if not added:
            # nothing may price out below the tolerance, or the LP is not solved and a cut would be invalid
            assert net.min_run_rc >= -1e-5, net.min_run_rc
            return master, value, pi, rho, mu, it + 1
    raise RuntimeError("column generation hit the iteration limit")


@dataclass
class Plan:
    cost: float = 0.0
    lp: float = 0.0
    driving: float = 0.0
    punctuality: float = 0.0
    empty: float = 0.0
    fleet: float = 0.0
    buses: dict = field(default_factory=dict)
    trips: list = field(default_factory=list)
    runs: list = field(default_factory=list)
    iterations: int = 0


def route_schedule(net, events, threads=0, time_limit=120, log=None):
    """The routing model of the paper for a fixed schedule: column generation, then an integer program over the
    generated trips. The LP value is a lower bound on the routing cost and the integer plan an upper bound."""
    q, p, e = net.demand(events)
    stops = sorted(e)
    if not stops:
        return Plan()

    def cost_of(a, t, tb):
        return np.array([e.get((b, a, t), 0.0) if USE_THETA else 0.0 for b in tb.branches])

    master, lp, *_rest, iters = column_generation(net, stops, {s: 1 for s in stops}, cost_of, q, p, threads, log=log)
    # direct trips were created with no punctuality cost, which is right: they arrive on time
    value, sol, bound = master.solve_ip(time_limit, threads)
    plan = Plan(cost=value, lp=lp, iterations=iters)
    for name, v in sol.items():
        if isinstance(name, Trip) and v > 0.5:
            plan.trips.append(name)
            plan.driving += name.drive
            plan.punctuality += name.theta
        elif isinstance(name, EmptyRun):
            plan.runs.append((name, v))
            plan.empty += name.cost * v
        elif isinstance(name, tuple) and name[0] == "n":
            plan.buses[net.buses[name[1]].key] = round(v)
            plan.fleet += net.buses[name[1]].fixed * round(v)
    if not USE_THETA:
        # routed without the cost of riders' time: the plan still pays it
        for h in plan.trips:
            tb = net.table(h.acc, h.k)
            tau = tb.taus[tb.seqs.index(h.seq)]
            plan.punctuality += sum(e[b, h.acc, h.t] * abs(h.delta + x) for b, x in zip(h.seq, tau))
        plan.cost += plan.punctuality
    return plan
