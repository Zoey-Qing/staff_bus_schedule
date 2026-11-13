#!/bin/bash
# Staff-roster column generation with adaptive bucket pricing (integrated/roster_cg/staff_dw.py), time of the Benders run.
# Usage: run_sdw.sh <data.xlsx> <benders_dir> <out_dir> <parallel jobs> <threads per job> <pricing>
DATA=$1; BD=$2; OUT=$3; P=$4; T=$5; PR=$6
mkdir -p $OUT
cd /e/schedule/integrated/roster_cg
for f in $BD/*_x0.csv; do
  c=$(basename $f _x0.csv); c=${c: -8}
  [ -f "$OUT/$(basename $DATA .xlsx)_${c}_${PR}.json" ] && continue
  echo $c
done | xargs -P $P -I{} sh -c "python -u staff_dw.py $DATA $BD $OUT --cluster {} --pricing $PR --threads $T > $OUT/{}.log 2>&1; echo finished {}"
echo ALL DONE
