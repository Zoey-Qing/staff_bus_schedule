#!/bin/bash
# Search ablations on all four test clusters, seed 1 (2026-11-11, reviewer requests):
#  (a) what guides the search: plain LNS / fixed estimate (stop prices) / fixed estimate (direct trips);
#      the learned estimate is the main run (ablation_m5eq/<c>/A4_no_smooth); C3=ba5b9796 (a) runs reused from
#      ../surrogate_ablation_c4/*_s1.
#  (b) lifting in the cuts vs lifting in the starting values of the search:
#      liftcut_noliftseed  = x^1 of the main run, search seeded with min(0, lifted price) (benders_sglns --no-lift)
#      noliftcut_liftseed  = x^1 of the no-lift column generation (ablation_nosmooth/<c>/A6_no_lift), lifted prices
# Same start and same search time as the source run (its wall.txt). Four runs at once, 3 CPLEX threads each.
R=E:/schedule/revision/results_new/original; O=$R/search_ablation_all
D=E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
S=Berkeley_Mock_Data_25_Stores_original_unbalanced; SU=E:/schedule/surrogate
var() {  # cluster name script input_dir wall_file "flags"
  local c=$1; local w=$O/$1/$2; mkdir -p $w; local left=$(awk '{print $4}' $5)
  (cd $SU && python -u $3 --cluster $c --data $D --benders-dir $4 --time-limit $left --seed 1 --threads 3 $6 \
     --out $w/${S}_${c}_s1.json > $w/search.log 2>&1)
}
jobs_list=()
for c in 1aea822f a1e8b5a7 ba5b9796 df75e3dc; do
  A4=$R/ablation_m5eq/$c/A4_no_smooth; A6=$R/ablation_nosmooth/$c/A6_no_lift
  if [ $c != ba5b9796 ]; then
    jobs_list+=("$c plain lns_variants.py $A4/input $A4/wall.txt --guide|none|--prices|benders")
    jobs_list+=("$c fixed_cut lns_variants.py $A4/input $A4/wall.txt --guide|fixed|--prices|benders")
    jobs_list+=("$c fixed_direct lns_variants.py $A4/input $A4/wall.txt --guide|fixed|--prices|direct")
  fi
  jobs_list+=("$c liftcut_noliftseed benders_sglns.py $A4/input $A4/wall.txt --start|benders|--prices|benders|--no-lift")
  jobs_list+=("$c noliftcut_liftseed benders_sglns.py $A6/input $A6/wall.txt --start|benders|--prices|benders")
done
i=0
for j in "${jobs_list[@]}"; do
  set -- $j
  var $1 $2 $3 $4 $5 "$(echo $6 | tr '|' ' ')" &
  i=$((i+1)); echo "$(date) start $1 $2" >> $O/progress.txt
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
echo ALL_DONE >> $O/progress.txt
