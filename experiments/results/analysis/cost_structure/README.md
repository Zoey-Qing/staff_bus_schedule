# Why every method saves little over the sequential schedule (5 base clusters, seed 1)

Script: ../cost_structure.py (per cluster); table: cost_structure_summary.csv.

- labour0, transport0: wages + overtime and exact transport of x0. Z0 = labour0 + transport0 (no penalty term).
- bigM_x0: sum_l M_l V_l(x0), M = (1000, 100, 10, 1). It is in the sequential objective, never in Z.
- labour_floor: best bound of min J_reg + J_over over all schedules within the service limits, buses ignored
  (900 s, CPLEX, 3 threads); labour_min_found is the best schedule found. Every feasible schedule pays at least the floor.
- LB: best valid lower bound on Z* from compare_all (M2 or M5-eq stage 1; L3 on C2).
- max_saving_pct = (Z0 - LB)/Z0: no method can save more than this.
- m5_*: M5-eq result; saving split into labour and transport changes; saving as a share of transport0, of Z0 - LB,
  and of Z0 - labour_floor.

Findings: labour is 89-96% of Z0; x0's labour is 0.3-1.2% above the labour floor; the provable maximum saving is
1.0-3.3% of Z0; M5-eq saves 3-22% of the transport cost and 8-52% of Z0 - LB.
