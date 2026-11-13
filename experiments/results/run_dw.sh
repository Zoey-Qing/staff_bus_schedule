#!/bin/bash
# Branch-roster column generation with Benders cuts (integrated/roster_cg/branch_dw.py), own cuts, time of the Benders run.
# Usage: run_dw.sh <data.xlsx> <benders_dir> <out_dir> <parallel jobs> <threads per job>
DATA=$1; BD=$2; OUT=$3; P=$4; T=$5
mkdir -p $OUT
cd /e/schedule/integrated/roster_cg
for f in $BD/*_x0.csv; do
  c=$(basename $f _x0.csv); c=${c: -8}
  [ -f "$OUT/$(basename $DATA .xlsx)_${c}_own.json" ] && continue
  echo $c
done | xargs -P $P -I{} sh -c "python -u branch_dw.py $DATA $BD $OUT --cluster {} --cuts own --threads $T > $OUT/{}.log 2>&1; echo finished {}"
echo ALL DONE
