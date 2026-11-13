Equal-time comparison on the five base clusters (seed 1); see run_all.sh. Per cluster folder: L1, L3, L4, M2 and M5eq
(stage 1 + stage 2) run at the same time, each with T = the Benders time of the cluster and 3 CPLEX threads.
C1 is rerun here under the same load as the other clusters; the earlier C1 runs (compare_c1/, cg_sglns_fair/) are kept.
Reused results: B1 and M1 from benders/, L2 from benders_nolift/ (z_benders, before our re-timing), M3 from
ablation_lns_fit/ (seed 1, run_equal_time.py, 5 jobs x 3 threads). summarize.py builds summary.csv.
