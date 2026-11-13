#!/bin/bash
# Stage-2 ablation on C4 (ba5b9796, the largest gap once C2 is dropped): plain LNS vs LNS with a fixed transport
# estimate vs SG-LNS with the learned surrogate, same start x^1 and same search time as the main run
# (../ablation_m5eq/ba5b9796/A4_no_smooth: stage-2 search 1122 s). Seeds 1-3; one wave per seed, four runs at once.
R=E:/schedule/revision/results_new/original; O=$R/surrogate_ablation_c4
D=E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
S=Berkeley_Mock_Data_25_Stores_original_unbalanced; c=ba5b9796
IN=$R/ablation_m5eq/$c/A4_no_smooth/input
LEFT=$(awk '{print $4}' $R/ablation_m5eq/$c/A4_no_smooth/wall.txt)
one() {  # name guide prices seed
  local w=$O/$1_s$4; mkdir -p $w
  (cd E:/schedule/surrogate && python -u lns_variants.py --cluster $c --data $D --benders-dir $IN --guide $2 --prices $3 \
     --time-limit $LEFT --seed $4 --threads 3 --out $w/${S}_${c}_s$4.json > $w/stage2.log 2>&1)
}
for s in 1 2 3; do
  echo "$(date) seed $s" >> $O/progress.txt
  one plain none benders $s &
  one fixed_cut fixed benders $s &
  one fixed_direct fixed direct $s &
  one learned learned benders $s &
  wait
done
echo ALL_DONE >> $O/progress.txt
