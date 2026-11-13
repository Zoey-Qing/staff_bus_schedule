One-cluster comparison (base Cluster 1 = 1aea822f, seed 1), every method given T = 6206 s (the Benders time of this cluster),
3 CPLEX threads each, run at the same time. Sequential and classical Benders without lifting: existing results
(benders/, benders_nolift/; report z_benders of the latter, before re-timing).
  ours:    sg_bpb.py --sglns-frac 0.3            -> ours/
  ormeci:  ormeci_baseline.py                    -> ormeci/
  dw_lit:  branch_dw.py --cuts own --no-lift     -> dw_lit/
  sdw_lit: staff_dw.py --no-lift                 -> sdw_lit/
  ours_nosg: sg_bpb.py --sglns-frac 0 (ablation: no SG-LNS phase) -> ours_nosg/
