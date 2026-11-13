Ablations of the main method M5-eq (five base clusters, seed 1). See run_ablation.sh for the six variants and how
each uses the time budget T. The full method and the single-stage rows (M2, M3, L3) are in ../compare_all/.
surrogate/benders_sglns.py gained --no-lift (default off, old behaviour unchanged) for A6.

Results (2026-11-09): summarize.py -> summary.csv (one row per cluster and variant) and pivot.csv.
saving = (Z0 - Z)/Z0; share = (Z0 - Z)/(Z0 - LB*), LB* the best bound of the cluster over compare_all/ and these runs
(on C3, A4 gives a higher bound than compare_all/, so the C3 shares differ from analysis/cost_structure/).
Notes:
 - A2_start_x0 was written to <c>/A1_prices_direct/A2_start_x0/ (a variable in run_ablation.sh was reused); the runs
   are complete and correct, only the folder is nested.
 - df75e3dc/A1_prices_direct/ stopped at the end of its search with a negative CPLEX time limit, before re-timing and
   before writing its result. It was rerun in df75e3dc/A1_prices_direct_rerun/ (see NOTE.txt there); summarize.py
   uses the rerun. The crashed log is kept.
 - All pipeline runs (A3-A6) finished within T (wall.txt); all stage-1 checks pass.
