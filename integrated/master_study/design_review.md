# Benders master: design review

Scope: `net_benders.py` (`add_stops`, `cut`, `benders`), `net_routing.py` (`column_generation`, `extend_duals`, `price`), `staff_scheduling/solver_cplex_new.py`, spec Sec. "Master Problem" / "Routing LP and Cuts". Notation follows the spec: `T_low(y)` is the routing LP (eq:lp) with costs `c_low_h` (lowest wage per stop), cover rows `sum_{h in H(s)} lambda_h = y_s`.

## Facts from the code and runs that drive the ranking

* All five clusters have exactly 5 (branch, accommodation) pairs and 168 event hours, so |y| = 840; each accommodation has 2-5 branches (a1e8: 3+2, 51 riders, 6/16/14/5/10 per stop).
* Final plans: driving 85-95% of transport, punctuality 1-12%, fleet 1-8% (2 bus types, 86 and 111 per bus-week), empty runs 0, route LP = IP.
* Round time is about 3,600 s master + ~20 s for CG and plan evaluation. CG is cheap, so the bottleneck is the master's LP relaxation and not the cut.
* `cut()` gives the cut at the stops of `x_new` with rhs 1. The lifted prices of closed stops make `(pi, rho, mu)` dual feasible for every rhs, so the cut is valid for any `y >= 0`, including fractional `y`.
* With the master's current linking `y <= load <= n*y`, the LP value of `y` is `load/n`. With 5-16 riders per stop and prices of 1.4-88, the LP value of theta is close to zero. That is why the master gap goes from 0.5% (round 0) to 4-6% once cuts exist: the LP value is about the staff bound, and theta is recovered only by branching on 840 y.

## Ranked recommendations

| # | Change | Impact | Effort | Validity risk |
|---|---|---|---|---|
| 1 | Rider-hour linking `y >= sum x`, y continuous | High | ~10 lines | None |
| 2 | Route block in the master (partial Benders) | Very high | ~40 lines | None |
| 3 | Seed cuts + CG column pool | Medium | ~30 lines | None |
| 4 | Round schedule (short early rounds, emphasis on bound last) | Medium | parameters | None |
| 5 | Single tree with lazy cuts at candidates | Medium (after 1-2) | ~80 lines | Low (implementation) |
| 6 | LP phase / in-out stabilisation | Low after 1-2 | ~50 lines | None |
| 7 | Multi-cut by accommodation/day/bus type | Do not use: invalid | - | High |

### 1. Disaggregated linking (replace `add_stops`)

The staff model has the clique `sum_{s covers [start, end+min_break)} x[i,s] <= 1` for every (i, t) (solver line 302, `min_break` = 4). Every shift that starts at t or ends at t covers hour t in this extended range. So for each rider, `sum_{s in starts[t] ∪ ends[t]} x[i,s] <= 1`. The tight rider-hour form is therefore:

```python
def add_stops(self, sm):
    m, x, y = sm.mdl, sm.solver.x, {}
    for (b, a, t) in self.possible_stops():
        y[b, a, t] = m.continuous_var(lb=0, ub=1, name=f"y_{b}_{a[:6]}_{t}")
        ev = self.starts[t] + self.ends[t]
        for i in self.net.riders[b, a]:
            m.add_constraint(y[b, a, t] >= m.sum(x[i, s] for s in ev))   # disaggregated
        m.add_constraint(y[b, a, t] <= m.sum(x[i, s] for i in self.net.riders[b, a] for s in ev))  # optional
    return y
```

* Size: one row per (rider, event hour), about 51 x 168 = 8.6k rows and about 120k nonzeros for a1e8. The (i, s) form would need about 10 times as many rows and is weaker. These rows can be added directly; user cuts are not needed at this size.
* Validity: at integer x, `max_i (sum_s x[i,s]) = 1` exactly when the stop is open, so the constraints describe the same set. `load <= n*y` is implied (sum the rows) and can be dropped.
* Is `y <= load` needed? No for validity. Dropping an upper bound on y only enlarges the master, so the bound stays valid, and the routing at the next round uses `demand(events(x))`, not y. It is also not needed for tightness while every coefficient of y is >= 0, which holds for every cut so far and for item 2. A lifted price can be negative in principle (cover rows are equalities). Keeping the 840 rows costs nothing, so keep them.
* y continuous: at integer x, the rows force y = 1 when some rider has an event and y = 0 otherwise (with `y <= load`). This removes 840 binaries from branching.

### 2. Route block in the master (the valid partial disaggregation)

**Multi-cut is not valid as written.** The dual feasible set of `T_low` is not a product:
* `mu_k` (fleet row eq:fleet, one per bus type) is shared by all accommodations, days and hours.
* Empty runs `g: a -> a2` put `rho_{k,a}` and `rho_{k,a2}` in one dual row.
* The cyclic flow rows link all hours of the week.
* All bus types share the cover duals `pi`.

Combining pieces of different cuts (for example `theta_a >= pi^j_a y_a`) can therefore exceed `T_low(y)`. The split by accommodation is exact only if the model has no empty runs, and the model allows them even though the plans use none.

**Valid separable relaxation.** In (eq:lp), drop the flow, fleet and empty-run rows and their costs. The removed costs are all >= 0, and every trip still costs at least the best departure of its route. What remains separates by (a, t) into a set-partitioning LP:

`T_low(y) >= sum_{a,t} SP_{a,t}(y) = sum_{a,t} min { sum_S c_{a,S} z_{a,t,S} : sum_{S ∋ b} z_{a,t,S} = y_{b,a,t}, z >= 0 }`,

where `c_{a,S} = min_{k, seq of S in RouteTable(a,k)} min_delta (drive + D @ low_wage)`. This cost does not depend on t or on the schedule. These are exactly the `base` values already used in `price()` and `cut()`. With 2-5 branches per accommodation there are at most 31 subsets, so the block is at most 168 x 31 continuous variables per accommodation (about 6k in a1e8) and 840 rows. This is an exact model of the per-hour routing inside the master: partial Benders (Crainic, Hewitt, Maggioni & Rei, 2021).

```python
def add_route_block(self, m, y, theta):
    net, cost = self.net, {}
    for a in net.accs:
        for k in range(len(net.buses)):
            tb = net.table(a, k)
            best = np.minimum.reduceat(tb.drive_entry + tb.D @ self.low_cost(a, None, tb), tb.start)
            for r, seq in enumerate(tb.seqs):
                key = (a, frozenset(seq)); cost[key] = min(cost.get(key, np.inf), float(best[r]))
    by = {}
    for (b, a, t) in y: by.setdefault((a, t), []).append(b)
    terms = []
    for (a, t), bs in by.items():
        z = {S: m.continuous_var(lb=0) for (aa, S) in cost if aa == a and S <= set(bs)}
        for b in bs:
            m.add_constraint(m.sum(v for S, v in z.items() if b in S) == y[b, a, t])
        terms += [cost[a, S] * v for S, v in z.items()]
    m.add_constraint(theta >= m.sum(terms), ctname="route_block")
```

Call this in `benders()` right after `add_stops`. Keep `theta >= pi.y` cuts as well: each is a valid lower bound, so their maximum is valid.

* Validity: the block is a relaxation of `T_low(y)`, which in turn is <= the real transport (spec eq:lp-bound).
* Effect: theta becomes about 92-99% of the true transport from round 0, and in the LP relaxation, given item 1. The Benders cuts then only need to recover the fleet and flow coupling (1-8%). Making z binary is also valid, because the integer set partition is still <= the real IP cost, but it is unnecessary: with at most 5 elements the LP is almost always integral.
* Projected a1e8 round-0 LB: 112.3k (staff) + about 12.5k (routes) = about 124.8k, against UB 125.9k, a gap near 1% instead of 4.1%. This is a projection, not a measured result.

### 3. Seeding and column reuse

* Before round 0, add cuts at `x0`, `x_u`, and any surrogate-evaluated schedules (`self.cut(x)` for each, ~15 s each), plus one cut with all 840 stops open. Any `x` gives a dual feasible lifted point, so each cut is valid everywhere. After item 2, these mainly add the fleet term.
* Per-stop bound (a one-row fallback if item 2 is not adopted): `theta >= sum_s l_s y_s` with `l_{b,a} = min_{S ∋ b} c_{a,S}/|S|`. Proof: `sum_S c_S z_S = sum_S sum_{b in S} (c_S/|S|) z_S >= sum_b l_b y_b`. A fleet constant is not valid for the LP: fractional trips can be chained on one fractional bus, so the LP fleet can be below one bus.
* Column pool: store every `Trip` from earlier CG runs. Before CG, seed `Master` with the pool trips whose stops are all open (`set(h.stops) <= open`) and with all pooled runs. Under `low_cost`, the costs of cut columns do not depend on the schedule, so reuse is exact. The final pricing pass still proves optimality, so the cut is unchanged. Expect about 10-30 CG iterations instead of about 100. This needs an optional `init_cols` argument to `column_generation`.

### 4. Round schedule

Round 0 currently spends 3,600 s on a master with theta = 0. After items 1-3, set early rounds to 600-900 s with `mipgap` 1% and `parameters.emphasis.mip = 1`, and spend the rest of the budget on the last round with `emphasis.mip = 3` (bound). The LB is the best bound of any round, so loose early rounds lose nothing. Also pass `best["x"]` as the MIP start, not only `x_on` (the round-1 master objective 126,039 was worse than the known UB of 125,942). CPLEX completes the continuous y, z and theta from a partial start.

### 5. Single-tree (branch-and-Benders-cut)

This is worth doing only if items 1-4 still leave more than ~1% gap. Use the generic callback with candidate context on `m.get_cplex()`:

```python
def invoke(self, ctx):                       # Context.id.candidate
    if not ctx.is_candidate_point(): return
    v = ctx.get_candidate_point(self.x_idx); x_on = {k for k, j in zip(self.x_keys, v) if j > .5}
    stops = frozenset(net.demand(events(x_on))[2]); th = ctx.get_candidate_point(self.theta_idx)
    with self.lock: prices, val = self.cache.get(stops) or self.cache.setdefault(stops, self.cut(x_on)[:2])
    if th < val - 1e-6 * max(1, val):
        ctx.reject_candidate(constraints=[cplex.SparsePair([self.theta_idx] + yi, [1] + [-p for p in pr])],
                             senses="G", rhs=[0.0])
```

* Cost: about 15 s of CG per new stop set, or about 3 s with the pool. A few hundred incumbents is still well under a 5-hour budget.
* Validity: the cuts are global, as in item 2. Rejecting candidates only removes points with theta < `T_low`.
* Evaluate the real UB (`plan_cost`) outside the callback, for the final incumbent and a few improving ones.
* Use a lock, because CG is not thread-safe through `net.min_run_rc`.

### 6. LP phase / stabilisation

The cut is well defined at fractional y. `column_generation` already accepts `rhs`. Call it with `stops = {s: y_s > 1e-9}` and `rhs = y`, lift prices for `y_s = 0` exactly as now, and change the check to `sum pi_s y_s == value`. The LP is feasible for any `0 <= y <= 1` (scaled direct trips, continuous flows). Kelley cuts on the LP master with in-out separation (`y_sep = 0.5 y_LP + 0.5 y(x0)`) converge to `min LP master + T_low`. After item 2, only the fleet residual remains to be gained, so this is low priority.

### 7. Other items

* The staff model has little symmetry: wages differ per person. Check with `len({(home[i], wage_reg[i], type(i))})` versus the number of riders before investing here.
* `b_start`/`b_end` stay in the master with zero cost. Presolve removes them, so no change is needed.
* Local branching around `best["x"]` overlaps with `retime`, so it is not recommended.
* The remaining gap between theta and the real transport after the cut is (i) the lowest-wage punctuality, (ii) LP versus IP and seat limits, and (iii) fleet. Item 2 does not change (i), so it bounds the achievable LB at roughly 0.3-1%.

**Suggested order:** items 1 + 2 together (one master build, about 50 lines in `add_stops` plus a new `add_route_block`), then item 3, then re-run a1e8 with item 4. Consider item 5 only if the gap is still above 1%. All changes are additive to the master; the verified CG and cut code are untouched, apart from an optional `init_cols` argument.
