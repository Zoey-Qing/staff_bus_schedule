#!/usr/bin/env bash
# Scale test on a generated network four times the size per cluster (net100_k10x2_s1: 20 branches, ~200 staff, two
# accommodations per cluster): sequential + Benders (20-min masters), then SG-LNS at the same budget.
O=/e/schedule/revision/results_new/original
cd /e/schedule/integrated
python -u run_benders_batch.py $O/net100/benders data_gen/net100_k10x2_s1.xlsx --jobs 5 --threads 3 > $O/net100_benders_batch.log 2>&1
cd /e/schedule/surrogate
python -u run_equal_time.py $O/net100/benders $O/net100/lns_fit --seeds 1 --ablation lns_fit > $O/net100_lns_fit_batch.log 2>&1
echo SCALE DONE
