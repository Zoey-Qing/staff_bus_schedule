#!/bin/bash
# Does the learned estimate gain with a longer search? C3 (ba5b9796) and C4 (df75e3dc), seed 1, three times the
# search time of the main run, learned (benders_sglns.py, as in the method) vs fixed direct-trip estimate
# (lns_variants.py --guide fixed --prices direct). Same start x^1. Four runs at once. 2026-11-11.
R=E:/schedule/revision/results_new/original; O=$R/search_long
D=E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
S=Berkeley_Mock_Data_25_Stores_original_unbalanced; SU=E:/schedule/surrogate
one() {  # cluster name script flags
  local c=$1; local w=$O/$1/$2; mkdir -p $w
  local left=$(awk '{print 3*$4}' $R/ablation_m5eq/$1/A4_no_smooth/wall.txt)
  (cd $SU && python -u $3 --cluster $c --data $D --benders-dir $R/ablation_m5eq/$1/A4_no_smooth/input --time-limit $left \
     --seed 1 --threads 3 $4 --out $w/${S}_${c}_s1.json > $w/search.log 2>&1)
}
one ba5b9796 learned benders_sglns.py "--start benders --prices benders" &
one ba5b9796 fixed_direct lns_variants.py "--guide fixed --prices direct" &
one df75e3dc learned benders_sglns.py "--start benders --prices benders" &
one df75e3dc fixed_direct lns_variants.py "--guide fixed --prices direct" &
wait
echo ALL_DONE > $O/done.txt
