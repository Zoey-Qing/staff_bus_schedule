# Benders master formulation study, cluster a1e8b5a7

Master rebuilt exactly as in `net_benders.ClusterProblem.benders`. It uses the stored limits {1: 175, 2: 83, 3: 780, 4: 28}, which match the levels of x0. It also has all 5 cuts stored by the 60-min run. Every run used 4 threads, a MIP gap of 1e-3, a 600 s limit and a MIP start from x0, with y set to the stops of x0 and theta set to the largest cut value at those stops. The runs were done two at a time.
Scripts: `master_study.py <variant>`, `summarize.py`. Full table: `master_study_results.csv`. CPLEX logs: `run_<variant>_cplex.log`.

| variant | rows added | LP | LP theta | root LP (presolved) | root bound (after cuts) | 600 s bound | incumbent | gap |
|---|---|---|---|---|---|---|---|---|
| no cuts (theta = 0) | 0 | 101,622 | 0 | 111,404 | 112,179 | 112,279 | 113,023 (x0) | 0.66% |
| A: current (aggregated link) | 0 | 103,805 | 2,174 | 113,724 | 118,843 | **118,846** | 125,601 | 5.38% |
| B: A + y >= x[i,s] | 177,276 | 104,034 | 2,360 | 114,512 | 116,399 | 116,409 | 125,861 (x0) | 7.51% |
| C: B without y <= load | 177,276 (−840) | 104,034 | 2,360 | 114,513 | 116,482 | 116,486 | 125,861 (x0) | 7.45% |
| D: B rows as user cuts | 177,276 user cuts | 103,805 | 2,174 | 113,724 | 118,851 | **118,851** | 125,601 | 5.37% |
| E: A + y >= sum_s x[i,s] per rider | 8,568 | 104,276 | 2,625 | 114,496 | 116,252 | 116,256 | 125,861 (x0) | 7.63% |
| E2: E with y continuous | 8,568 | 104,276 | 2,625 | 114,496 | 115,081 | 115,096 | 125,861 (x0) | 8.55% |
| theta only (min theta, A rows) | 0 | 1,440 | 1,440 | 1,691 | 2,420 | 2,420 | 9,631 | 74.9% |

For comparison, the 60-min run reached a Benders LB of 120,734, and its best plan costs 126,270.

## Findings

1. **The tighter links do not raise the 600 s bound. Added as hard rows, they lower it.** The disaggregated rows lift the LP by only 230 to 470 and the presolved root LP by about 800. With the rows in the model, CPLEX applies roughly 300 cuts of its own, against about 2,270 in A (GUB cover, clique, MIR and zero-half). The root bound therefore ends 2.4k to 3.8k lower. The root LP is also about twice as slow. As user cuts (D), the rows are almost neutral: the bound gains 6, and 439 of them are used. CPLEX's own cuts in A already capture what these links add.
2. **E is valid.** The staff model has a rest constraint: `sum x[i,s] <= 1` over the shifts whose [start, end + 4h) covers t. Every shift that starts or ends at t is in that set, so the right side of E is at most 1 at every point, including LP points. As a check, x0, the stored best schedule and all incumbents have at most 1 start or end per rider-hour. E is the strongest LP of the variants, with LP theta 2,625, but it gives a lower MIP bound than A. E2 (continuous y) is weaker again, because CPLEX loses the binary structure it uses for its cuts.
3. **The staff part is not what limits the bound.** Without cuts the master is nearly solved: x0 is within 0.66% of the staff optimum (112.3k). The whole gap comes from theta. In A the bound proves about 6.5k of theta, while the incumbents carry 12.5k to 12.8k.
4. **What limits the bound is the LP of theta under the cuts.** With theta as the only objective, the master proves only theta >= 2,420 after 600 s, and it is still at the root. Yet the cheapest theta found is 9,631, and that schedule has wages and overtime about 21k above x0. In the LP, x spreads each rider's shifts over many start and end hours, so every y takes a small fractional value. Each stop is priced between about 1.4 and 88, so the cut sums stay small. Linking y to individual x variables cannot fix this, because each x is itself small.

## Next steps

Inequalities that count stops per day, rather than per (shift, hour), are the promising direction. For a rider i of (b,a) and a day d, an example is `sum_{t in day d} y[b,a,t] >= w[i,d]`, applied separately to start stops and end stops. The number of working days w is forced by the minimum-work-day rule. These rows would give theta a floor of about (working days) × (cheapest price in the day) per rider group. Rows that link to x only do not give such a floor.
