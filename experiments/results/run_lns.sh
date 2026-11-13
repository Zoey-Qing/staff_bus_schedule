#!/usr/bin/env bash
# Follow-up of the ablations: lns with the regression (seed 1), then lns seeds 2-3, at the Benders budgets.
O=/e/schedule/revision/results_new/original
cd /e/schedule/surrogate
python -u run_equal_time.py $O/benders $O/ablation_lns_fit --seeds 1 --ablation lns_fit > $O/ablation_lns_fit_batch.log 2>&1
python -u run_equal_time.py $O/benders $O/ablation_lns --seeds 2,3 --ablation lns > $O/ablation_lns_seeds23_batch.log 2>&1
echo LNS DONE
