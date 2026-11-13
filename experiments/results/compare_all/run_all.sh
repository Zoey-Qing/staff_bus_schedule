#!/bin/bash
# Equal-time comparison on the five base clusters, seed 1. One cluster at a time; the five new methods of a cluster run
# at the same time (5 jobs x 3 CPLEX threads), so every method of a cluster sees the same machine load.
# T = benders_seconds of the cluster (benders/*.json). Every method starts from the same x0 (benders/*_x0.csv).
#   L1 ormeci_baseline.py            L3 branch_dw.py --cuts own --no-lift     L4 staff_dw.py --no-lift
#   M2 sg_bpb.py --sglns-frac 0      M5eq sg_bpb.py --sglns-frac 0 (0.8 T) then benders_sglns.py (rest of T, 400 s kept)
# Reused, not rerun: B1 / M1 (benders/), L2 (benders_nolift/, z_benders), M3 (ablation_lns_fit/ seed 1, run_equal_time.py).
R=E:/schedule/revision/results_new/original; O=$R/compare_all
D=E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
S=Berkeley_Mock_Data_25_Stores_original_unbalanced
RC=E:/schedule/integrated/roster_cg; SU=E:/schedule/surrogate
m5eq() {  # cluster T dir
  c=$1; T=$2; d=$3; t0=$(date +%s)
  (cd $RC && python -u sg_bpb.py $D $R/benders $d/m5eq_stage1 --cluster $c --time $(python -c "print(round(0.8*$T))") --sglns-frac 0 --threads 3 --seed 1 --tag m5eq_stage1 > $d/m5eq_stage1.log 2>&1)
  t1=$(date +%s); mkdir -p $d/m5eq_input
  cp $R/benders/${S}_${c}_x0.csv $d/m5eq_input/
  cp $d/m5eq_stage1/${S}_${c}_m5eq_stage1_s1_schedule.csv $d/m5eq_input/${S}_${c}_schedule.csv
  left=$((T - (t1-t0) - 400))
  (cd $SU && python -u benders_sglns.py --cluster $c --data $D --benders-dir $d/m5eq_input --time-limit $left --seed 1 --threads 3 --start benders --prices benders --out $d/m5eq_${S}_${c}_s1.json > $d/m5eq_stage2.log 2>&1)
  t2=$(date +%s); echo "stage1 $((t1-t0)) stage2_search $left stage2 $((t2-t1)) total $((t2-t0)) T $T" > $d/m5eq_wall.txt
}
for c in 1aea822f 662b3495 a1e8b5a7 ba5b9796 df75e3dc; do
  d=$O/$c; [ -f $d/WAVE_DONE ] && continue; mkdir -p $d
  T=$(python -c "import json;print(round(json.load(open('$R/benders/${S}_${c}.json'))['benders_seconds']))")
  echo "$(date) start $c T=$T" >> $O/progress.txt
  (cd $RC && python -u ormeci_baseline.py $D $R/benders $d/L1 --cluster $c --time $T --threads 3 > $d/L1.log 2>&1) &
  (cd $RC && python -u branch_dw.py $D $R/benders $d/L3 --cluster $c --cuts own --no-lift --time $T --threads 3 --tag nolift > $d/L3.log 2>&1) &
  (cd $RC && python -u staff_dw.py $D $R/benders $d/L4 --cluster $c --no-lift --time $T --threads 3 --tag nolift > $d/L4.log 2>&1) &
  (cd $RC && python -u sg_bpb.py $D $R/benders $d/M2 --cluster $c --time $T --sglns-frac 0 --threads 3 --seed 1 --tag m2 > $d/M2.log 2>&1) &
  m5eq $c $T $d &
  wait
  touch $d/WAVE_DONE; echo "$(date) done $c" >> $O/progress.txt
done
echo ALL_DONE >> $O/progress.txt
