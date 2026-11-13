#!/usr/bin/env bash
# lns_fit (neighbor search with the regression, no upper-level solve): seeds 2-3 on the original data, then seed 1 on far x2.
O=/e/schedule/revision/results_new/original
cd /e/schedule/surrogate
python -u run_equal_time.py $O/benders $O/ablation_lns_fit --seeds 2,3 --ablation lns_fit > $O/ablation_lns_fit_seeds23_batch.log 2>&1
python -u run_equal_time.py $O/far2/benders $O/far2/lns_fit --seeds 1 --ablation lns_fit > $O/far2_lns_fit_batch.log 2>&1
echo LNSFIT DONE
