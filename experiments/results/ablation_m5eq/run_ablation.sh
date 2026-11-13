#!/bin/bash
# Ablations of M5-eq on the five base clusters, seed 1. One cluster at a time, five slots at once (3 CPLEX threads each),
# as in compare_all/. T = Benders time of the cluster. The full method is compare_all/<c>/m5eq_*.
#  A1 prices_direct : stage 2 from the same stage-1 schedule with the one-trip opening prices (benders_sglns --prices direct)
#  A2 start_x0      : stage 2 from x0 with cut prices at x0 and history {x0, x^CG} (benders_sglns --start x0)
#     A1 and A2 reuse the stage-1 result of compare_all/ and get the same stage-2 search time (m5eq_wall.txt); one slot, in turn.
#  A3 no_heur       : M5-eq with exact pricing only (sg_bpb --exact-every 0)
#  A4 no_smooth     : M5-eq without dual smoothing (sg_bpb --smooth 0)
#  A5 split06       : M5-eq with stage 1 = 0.6 T
#  A6 no_lift       : M5-eq with cuts and stage-2 prices without lifting (sg_bpb --no-lift, benders_sglns --no-lift)
#  Stage 2 alone from x0 (M3), stage 1 alone (M2) and stage 1 without lifting (L3) are in compare_all/.
R=E:/schedule/revision/results_new/original; O=$R/ablation_m5eq; CA=$R/compare_all
D=E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
S=Berkeley_Mock_Data_25_Stores_original_unbalanced
RC=E:/schedule/integrated/roster_cg; SU=E:/schedule/surrogate
pipeline() {  # cluster T dir name frac "stage1 flags" "stage2 flags"
  c=$1; T=$2; d=$3/$4; f=$5; mkdir -p $d; t0=$(date +%s)
  (cd $RC && python -u sg_bpb.py $D $R/benders $d/stage1 --cluster $c --time $(python -c "print(round($f*$T))") --sglns-frac 0 --threads 3 --seed 1 --tag stage1 $6 > $d/stage1.log 2>&1)
  t1=$(date +%s); mkdir -p $d/input
  cp $R/benders/${S}_${c}_x0.csv $d/input/; cp $d/stage1/${S}_${c}_stage1_s1_schedule.csv $d/input/${S}_${c}_schedule.csv
  left=$((T - (t1-t0) - 400))
  (cd $SU && python -u benders_sglns.py --cluster $c --data $D --benders-dir $d/input --time-limit $left --seed 1 --threads 3 --start benders --prices benders $7 --out $d/${S}_${c}_s1.json > $d/stage2.log 2>&1)
  t2=$(date +%s); echo "stage1 $((t1-t0)) stage2_search $left stage2 $((t2-t1)) total $((t2-t0)) T $T" > $d/wall.txt
}
stage2_only() {  # cluster dir name "flags"
  c=$1; d=$2/$3; mkdir -p $d; left=$(awk '{print $4}' $CA/$c/m5eq_wall.txt)
  t1=$(date +%s)
  (cd $SU && python -u benders_sglns.py --cluster $c --data $D --benders-dir $CA/$c/m5eq_input --time-limit $left --seed 1 --threads 3 $4 --out $d/${S}_${c}_s1.json > $d/stage2.log 2>&1)
  echo "stage2_search $left stage2 $(( $(date +%s) - t1 ))" > $d/wall.txt
}
for c in 1aea822f 662b3495 a1e8b5a7 ba5b9796 df75e3dc; do
  d=$O/$c; [ -f $d/WAVE_DONE ] && continue; mkdir -p $d
  T=$(python -c "import json;print(round(json.load(open('$R/benders/${S}_${c}.json'))['benders_seconds']))")
  echo "$(date) start $c T=$T" >> $O/progress.txt
  (stage2_only $c $d A1_prices_direct "--start benders --prices direct"; stage2_only $c $d A2_start_x0 "--start x0 --prices benders") &
  pipeline $c $T $d A3_no_heur 0.8 "--exact-every 0" "" &
  pipeline $c $T $d A4_no_smooth 0.8 "--smooth 0" "" &
  pipeline $c $T $d A5_split06 0.6 "" "" &
  pipeline $c $T $d A6_no_lift 0.8 "--no-lift" "--no-lift" &
  wait
  touch $d/WAVE_DONE; echo "$(date) done $c" >> $O/progress.txt
done
echo ALL_DONE >> $O/progress.txt
