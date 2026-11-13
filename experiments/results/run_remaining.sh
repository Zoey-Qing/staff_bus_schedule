#!/usr/bin/env bash
# Remaining experiments on the original data, one batch at a time (5 jobs x 3 threads), after the surrogate seeds 2-3:
#   far x2 (accommodations at twice the distance): sequential + Benders (20-min masters), then the surrogate at equal time;
#   ablations of the surrogate at the Benders budgets: lns (no upper-level solve) and static (no regression).
O=/e/schedule/revision/results_new/original
until grep -q "ALL DONE" $O/surrogate_seeds23_batch.log 2>/dev/null; do sleep 60; done
cd /e/schedule/integrated
python -u run_benders_batch.py $O/far2/benders data_gen/Berkeley_Mock_Data_25_Stores_original_far2.xlsx --jobs 5 --threads 3 > $O/far2_benders_batch.log 2>&1
cd /e/schedule/surrogate
python -u run_equal_time.py $O/far2/benders $O/far2/surrogate --seeds 1 > $O/far2_surrogate_batch.log 2>&1
python -u run_equal_time.py $O/benders $O/ablation_lns --seeds 1 --ablation lns > $O/ablation_lns_batch.log 2>&1
python -u run_equal_time.py $O/benders $O/ablation_static --seeds 1 --ablation static > $O/ablation_static_batch.log 2>&1
echo REMAINING DONE
