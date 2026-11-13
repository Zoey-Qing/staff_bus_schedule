#!/bin/bash
# Independent check of the lifted cuts against the full routing dual (integrated/verify_cuts.py, unchanged), at the
# sequential schedule x0 and at our final schedule, four clusters. 2026-11-11.
R=E:/schedule/revision/results_new/original; O=$R/verify_cuts
D=E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
S=Berkeley_Mock_Data_25_Stores_original_unbalanced
for c in 1aea822f a1e8b5a7 ba5b9796 df75e3dc; do
  A4=$R/ablation_m5eq/$c/A4_no_smooth
  (cd E:/schedule/integrated && python -u verify_cuts.py $D $A4/input/${S}_${c}_x0.csv --cluster $c --random 10 --threads 2 --out $O/${c}_x0.json > $O/${c}_x0.log 2>&1)
  (cd E:/schedule/integrated && python -u verify_cuts.py $D $A4/${S}_${c}_s1.schedule.csv --cluster $c --random 10 --threads 2 --out $O/${c}_ours.json > $O/${c}_ours.log 2>&1)
done
echo ALL_DONE > $O/done.txt
