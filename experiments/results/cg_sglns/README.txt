SG-LNS seeded by branch-roster column generation with lifted cuts (C1 = 1aea822f, seed 1).
Stage 1: sg_bpb.py --sglns-frac 0 (compare_c1/ours_nosg/, time T = 6206 s): lower bound 112,094.94 and schedule x^CG.
Stage 2: surrogate/benders_sglns.py unchanged, --start benders --prices benders, run on cg_sglns_input/, where
  *_x0.csv is a copy of benders/*_x0.csv and *_schedule.csv is a copy of the stage-1 schedule. So "benders" in the log
  means x^CG: the start, the lifted prices computed at x^CG, and the history {x0, x^CG}. Search time T = 6206 s.
Total time 2T, as for benders_sglns/ (the same pipeline started from the Benders + CG result).
