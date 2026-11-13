Equal-time version of "branch CG + lifted cuts -> SG-LNS" on C1 (1aea822f), seed 1. See run.sh.
Stage 1 (stage1/, stage1.log): sg_bpb.py --sglns-frac 0 --time 4965 (0.8 T): lower bound and schedule x^CG.
Stage 2 (stage2.log, *_s1.json): surrogate/benders_sglns.py unchanged on input/ (x0 copy + stage-1 schedule), so
"benders" in its log means x^CG. Wall clock of both stages in wall.txt; the total is to stay within T = 6206 s.
