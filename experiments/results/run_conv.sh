#!/bin/bash
# Convergence check of SG-LNS: three times the Benders time, seed 1, base network and medium families.
# Starts after run_tre2.sh (large network) so the runs do not compete for the CPU.
O=/e/schedule/revision/results_new/original
until grep -q "TRE2 DONE" $O/run_tre2.log 2>/dev/null; do sleep 120; done
cd /e/schedule/surrogate
python -u run_equal_time.py $O/benders $O/conv_x3/base --seeds 1 --ablation lns_fit --time-factor 3 --jobs 6 --threads 2 > $O/conv_x3_base.log 2>&1
python -u run_equal_time.py $O/families/benders $O/conv_x3/medium --seeds 1 --ablation lns_fit --time-factor 3 --jobs 6 --threads 2 > $O/conv_x3_medium.log 2>&1
echo CONV DONE
