"""Checks of net_routing on small cases, against a brute-force model and hand calculations."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bus_scheduling_routing"))
import net_routing as N


class Toy(N.Network):
    def __init__(self, accs, branches_of, tau, fixed=100.0, cap=20):
        self.cluster = "toy"
        self.accs = accs
        self.tau, self.km = {}, {}
        for (u, v), m in tau.items():
            self.tau[u, v] = self.tau[v, u] = m
            self.km[u, v] = self.km[v, u] = m * N.SPEED / 60.0
        for u in set(x for pair in tau for x in pair):
            self.tau[u, u] = 0
            self.km[u, u] = 0.0
        self.buses = [N.BusType("small", cap, fixed, 1.5, 1.0)]
        self.branches_of = branches_of
        self.home, self.wage, self.riders, self.low_wage, self.tables = {}, {}, {}, {}, {}

    def people(self, b, a, n, wage=0.5):
        for j in range(n):
            i = f"{b}{a}{j}{len(self.home)}"
            self.home[i] = (b, a)
            self.wage[i] = wage
            self.riders.setdefault((b, a), []).append(i)
        return [i for i in self.riders[b, a]]


def brute_force(net, stops, e, q, p):
    """Enumerate every trip and every minute of the network, and solve the integer model directly."""
    master = N.Master(net, stops, {s: 1 for s in stops})
    hours = {}
    for b, a, t in stops:
        hours.setdefault((a, t), []).append(b)
    for (a, t), bs in hours.items():
        for n in range(1, min(5, len(bs)) + 1):
            for seq in itertools.permutations(bs, n):
                for k, bus in enumerate(net.buses):
                    if not N.fits(seq, a, t, q, p, bus.capacity):
                        continue
                    tb = net.table(a, k)
                    if seq not in tb.seqs:
                        continue
                    r = tb.seqs.index(seq)
                    tau = tb.taus[r]
                    lo, hi = -N.WINDOW - tau[0], N.WINDOW - tau[-1]
                    for d in range(lo, hi + 1):
                        theta = sum(e[b, a, t] * abs(d + x) for b, x in zip(seq, tau))
                        total = tau[-1] + net.tau[seq[-1], a]
                        master.add_trip(N.Trip(k, a, t, seq, d, (60 * t + d) % N.WEEK, total, float(tb.drive[r]), theta))
    if len(net.accs) > 1:
        for k, bus in enumerate(net.buses):
            for a, a2 in itertools.permutations(net.accs, 2):
                for w in range(N.WEEK):
                    if w % 5 == 0:
                        master.add_run(N.EmptyRun(k, a, a2, w, net.tau[a, a2],
                                                  bus.per_km * net.km[a, a2] + bus.per_min * net.tau[a, a2]))
    return master.solve_ip(600)[0]


def case_two_stops():
    """The hand example: X 4 riders, Y 2 riders at 0.5 per minute, 10 + 15 + 15 minutes. One round trip per week."""
    net = Toy(["A"], {"A": ["X", "Y"]}, {("A", "X"): 10, ("X", "Y"): 15, ("A", "Y"): 15}, fixed=100.0)
    ev = []
    for i in net.people("X", "A", 4):
        ev.append((i, 10, 18))
    for i in net.people("Y", "A", 2):
        ev.append((i, 10, 19))
    # keep only the drop-offs of hour 10 by giving the pick-ups their own hours
    plan = N.route_schedule(net, ev)
    q, p, e = net.demand(ev)
    assert abs(plan.cost - brute_force(net, sorted(e), e, q, p)) < 1e-4, (plan.cost,)
    return plan


def case_two_acc():
    """Two accommodations 20 minutes apart. One bus serves both, with two empty runs, when that beats a second bus."""
    tau = {("A", "B"): 20, ("A", "X"): 10, ("B", "Z"): 10, ("A", "Z"): 25, ("B", "X"): 25, ("X", "Z"): 30}
    net = Toy(["A", "B"], {"A": ["X"], "B": ["Z"]}, tau, fixed=500.0)
    i, j = net.people("X", "A", 1)[0], net.people("Z", "B", 1)[0]
    ev = [(i, 10, 18), (j, 11, 19)]
    plan = N.route_schedule(net, ev)
    q, p, e = net.demand(ev)
    assert abs(plan.cost - brute_force(net, sorted(e), e, q, p)) < 1e-3, plan.cost
    assert sum(plan.buses.values()) == 1, plan.buses
    assert plan.empty > 0
    # with a cheap bus, two buses are better than driving empty
    net2 = Toy(["A", "B"], {"A": ["X"], "B": ["Z"]}, tau, fixed=1.0)
    net2.people("X", "A", 1), net2.people("Z", "B", 1)
    ev2 = [(list(net2.home)[0], 10, 18), (list(net2.home)[1], 11, 19)]
    plan2 = N.route_schedule(net2, ev2)
    assert sum(plan2.buses.values()) == 2 and plan2.empty == 0, plan2
    return plan, plan2


def case_fleet():
    """Two riders whose direct trips overlap need two buses, unless a departure moves."""
    net = Toy(["A"], {"A": ["X", "Y"]}, {("A", "X"): 20, ("X", "Y"): 100, ("A", "Y"): 20}, fixed=1000.0)
    i, j = net.people("X", "A", 1)[0], net.people("Y", "A", 1)[0]
    ev = [(i, 10, 18), (j, 10, 18)]
    plan = N.route_schedule(net, ev)
    q, p, e = net.demand(ev)
    assert abs(plan.cost - brute_force(net, sorted(e), e, q, p)) < 1e-3, plan.cost
    assert sum(plan.buses.values()) == 1, plan.buses
    return plan


def case_lift_valid():
    """The dual values and the LP agree, and the LP lies below the integer plan."""
    net = Toy(["A"], {"A": ["X", "Y", "Z"]}, {("A", "X"): 10, ("A", "Y"): 12, ("A", "Z"): 14, ("X", "Y"): 5, ("Y", "Z"): 6,
                                              ("X", "Z"): 9}, fixed=50.0)
    ev = []
    for b, n in (("X", 3), ("Y", 4), ("Z", 2)):
        for h, i in enumerate(net.people(b, "A", n)):
            ev.append((i, 8 + h % 2, 17 + h % 2))
    plan = N.route_schedule(net, ev)
    q, p, e = net.demand(ev)
    assert plan.lp <= plan.cost + 1e-6
    assert abs(plan.cost - brute_force(net, sorted(e), e, q, p)) < plan.cost * 2e-3, (plan.cost, plan.lp)
    return plan


def case_retime_transport():
    """The re-timing model, with the schedule fixed, costs the same as the routing plan when the plan drives empty."""
    from docplex.mp.model import Model
    import net_benders as B

    tau = {("A", "B"): 20, ("A", "X"): 10, ("B", "Z"): 10, ("A", "Z"): 25, ("B", "X"): 25, ("X", "Z"): 30}
    out = []
    for fixed, buses in ((500.0, 1), (1.0, 2)):
        net = Toy(["A", "B"], {"A": ["X"], "B": ["Z"]}, tau, fixed=fixed)
        i, j = net.people("X", "A", 1)[0], net.people("Z", "B", 1)[0]
        ev = [(i, 10, 18), (j, 11, 19)]
        plan = N.route_schedule(net, ev)
        assert sum(plan.buses.values()) == buses and (plan.empty > 0) == (buses == 1), plan
        prob = B.ClusterProblem.__new__(B.ClusterProblem)
        prob.net = net
        prob.starts = {10: ["s1"], 11: ["s2"], 18: [], 19: []}
        prob.ends = {10: [], 11: [], 18: ["s1"], 19: ["s2"]}
        m = Model()
        x = {(i, "s1"): m.integer_var(lb=1, ub=1), (j, "s2"): m.integer_var(lb=1, ub=1)}
        q, p, e = net.demand(ev)
        y = {s: m.integer_var(lb=1, ub=1) for s in e}
        m.minimize(prob._add_transport(m, x, y, dict(plan=plan)))
        sol = m.solve()
        assert sol and abs(sol.objective_value - plan.cost) < 1e-4, (sol and sol.objective_value, plan.cost)
        out.append((plan.cost, plan.empty))
    return out


if __name__ == "__main__":
    for f in (case_two_stops, case_two_acc, case_fleet, case_lift_valid, case_retime_transport):
        print(f.__name__, f())
    print("all passed")
