#!/bin/bash
# Hybrid: branch-roster DW warm-started with the three SG-LNS schedules (seeds 1-3) as columns and cuts; time of the Benders run.
# Usage: run_dw_warm.sh <data.xlsx> <benders_dir> <lns_dir> <out_dir> <parallel jobs> <threads per job>
DATA=$1; BD=$2; LD=$3; OUT=$4; P=$5; T=$6
STEM=$(basename $DATA .xlsx)
mkdir -p $OUT
cd /e/schedule/integrated/roster_cg
for f in $BD/*_x0.csv; do
  c=$(basename $f _x0.csv); c=${c: -8}
  [ -f "$OUT/${STEM}_${c}_warm.json" ] && continue
  echo $c
done | xargs -P $P -I{} sh -c "python -u branch_dw.py $DATA $BD $OUT --cluster {} --cuts own --threads $T --tag warm --warm $LD/${STEM}_{}_s1.schedule.csv $LD/${STEM}_{}_s2.schedule.csv $LD/${STEM}_{}_s3.schedule.csv > $OUT/{}.log 2>&1; echo finished {}"
echo ALL DONE
