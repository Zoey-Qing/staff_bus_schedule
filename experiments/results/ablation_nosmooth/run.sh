#!/bin/bash
# Ablations of the main method WITHOUT dual smoothing (decision 2026-11-09: the main method uses --smooth 0).
# The full method without smoothing is ../ablation_m5eq/<c>/A4_no_smooth/ (seed 1, same budget and load pattern).
# Here every ablation of ../ablation_m5eq/ is rerun with --smooth 0, so that each one differs from the main method in
# one part only. One cluster at a time, five slots at once (3 CPLEX threads each), as before.
#  A1 prices_direct : stage 2 from the stage-1 schedule of A4_no_smooth, one-trip opening prices
#  A2 start_x0      : stage 2 from x0 with cut prices at x0 and history {x0, x1}
#     A1 and A2 reuse the stage 1 of A4_no_smooth and get its stage-2 search time (wall.txt); one slot, in turn.
#  A3 no_heur       : exact pricing only
#  A5 split06       : stage 1 = 0.6 T
#  A6 no_lift       : cuts and stage-2 prices without lifting
#  M2 stage1_only   : stage 1 with the whole budget T
#  Stage 2 alone from x0 (M3) and the literature methods have no smoothing and are not rerun (../compare_all/).
R=E:/schedule/revision/results_new/original; O=$R/ablation_nosmooth; AB=$R/ablation_m5eq
D=E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
S=Berkeley_Mock_Data_25_Stores_original_unbalanced
RC=E:/schedule/integrated/roster_cg; SU=E:/schedule/surrogate
pipeline() {  # cluster T base name frac "stage1 flags" "stage2 flags"
  local c=$1 T=$2 w=$3/$4 f=$5; mkdir -p $w; local t0=$(date +%s)
  (cd $RC && python -u sg_bpb.py $D $R/benders $w/stage1 --cluster $c --time $(python -c "print(round($f*$T))") --sglns-frac 0 --threads 3 --seed 1 --tag stage1 --smooth 0 $6 > $w/stage1.log 2>&1)
  local t1=$(date +%s); mkdir -p $w/input
  cp $R/benders/${S}_${c}_x0.csv $w/input/; cp $w/stage1/${S}_${c}_stage1_s1_schedule.csv $w/input/${S}_${c}_schedule.csv
  local left=$((T - (t1-t0) - 400))
  (cd $SU && python -u benders_sglns.py --cluster $c --data $D --benders-dir $w/input --time-limit $left --seed 1 --threads 3 --start benders --prices benders $7 --out $w/${S}_${c}_s1.json > $w/stage2.log 2>&1)
  local t2=$(date +%s); echo "stage1 $((t1-t0)) stage2_search $left stage2 $((t2-t1)) total $((t2-t0)) T $T" > $w/wall.txt
}
stage1_only() {  # cluster T base name
  local c=$1 T=$2 w=$3/$4; mkdir -p $w; local t0=$(date +%s)
  (cd $RC && python -u sg_bpb.py $D $R/benders $w --cluster $c --time $T --sglns-frac 0 --threads 3 --seed 1 --tag m2 --smooth 0 > $w/stage1.log 2>&1)
  echo "total $(( $(date +%s) - t0 )) T $T" > $w/wall.txt
}
stage2_only() {  # cluster base name "flags"
  local c=$1 w=$2/$3; mkdir -p $w; local src=$AB/$c/A4_no_smooth
  local left=$(awk '{print $4}' $src/wall.txt); local t1=$(date +%s)
  (cd $SU && python -u benders_sglns.py --cluster $c --data $D --benders-dir $src/input --time-limit $left --seed 1 --threads 3 $4 --out $w/${S}_${c}_s1.json > $w/stage2.log 2>&1)
  echo "stage2_search $left stage2 $(( $(date +%s) - t1 ))" > $w/wall.txt
}
for c in 1aea822f 662b3495 a1e8b5a7 ba5b9796 df75e3dc; do
  base=$O/$c; [ -f $base/WAVE_DONE ] && continue; mkdir -p $base
  T=$(python -c "import json;print(round(json.load(open('$R/benders/${S}_${c}.json'))['benders_seconds']))")
  echo "$(date) start $c T=$T" >> $O/progress.txt
  (stage2_only $c $base A1_prices_direct "--start benders --prices direct"; stage2_only $c $base A2_start_x0 "--start x0 --prices benders") &
  pipeline $c $T $base A3_no_heur 0.8 "--exact-every 0" "" &
  pipeline $c $T $base A5_split06 0.6 "" "" &
  pipeline $c $T $base A6_no_lift 0.8 "--no-lift" "--no-lift" &
  stage1_only $c $T $base M2_stage1_only &
  wait
  touch $base/WAVE_DONE; echo "$(date) done $c" >> $O/progress.txt
done
echo ALL_DONE >> $O/progress.txt
