#!/bin/bash
# Follows run_tre.sh: the large network regenerated from the original base data (the earlier net100 instance took
# its branch templates from the rebalanced workbook). Benders, then SG-LNS seed 1 at equal time.
O=/e/schedule/revision/results_new/original
D=/e/schedule/integrated/data_gen
until grep -q "TRE DONE" $O/run_tre.log 2>/dev/null; do sleep 300; done
cd /e/schedule/integrated
python -u run_benders_batch.py $O/large/benders $D/syn_large_s1.xlsx --jobs 5 --threads 3 > $O/large_benders_batch.log 2>&1
cd /e/schedule/surrogate
python -u run_equal_time.py $O/large/benders $O/large/lns_fit --seeds 1 --ablation lns_fit > $O/large_lns_fit_batch.log 2>&1
echo TRE2 DONE
