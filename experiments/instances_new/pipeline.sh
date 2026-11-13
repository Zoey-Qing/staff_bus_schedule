#!/bin/bash
# One run of our method on a new instance or a sensitivity case (2026-11-11). Same pipeline as the main runs
# (ablation_m5eq A4_no_smooth): sequential schedule x0, column generation with lifted cuts for 0.8 Delta
# (sg_bpb.py --sglns-frac 0 --smooth 0), then the neighborhood search (benders_sglns.py) for the rest minus 400 s.
# Usage: pipeline.sh <name> <data.xlsx> <cluster prefix> <Delta seconds> [ENV=VALUE ...]
name=$1; D=$2; c=$3; T=$4; shift 4
for kv in "$@"; do export "$kv"; done
O=E:/schedule/revision/instances_new/${RUNS:-runs}/$name; mkdir -p $O/bd $O/input
stem=$(basename $D .xlsx); RC=E:/schedule/integrated/roster_cg; SU=E:/schedule/surrogate; I=E:/schedule/integrated
echo "$(date) start $name T=$T env=$*" >> E:/schedule/revision/instances_new/progress.txt
if [ -n "$X0DIR" ]; then
  cp $X0DIR/${stem}_${c}_x0.csv $X0DIR/${stem}_${c}.json $O/bd/
else
  (cd $I && python -u net_benders.py $D $O/bd/${stem}_${c} --cluster $c --seq-only --threads 3 > $O/seq.log 2>&1)
fi
python - "$O/bd/${stem}_${c}.json" "$T" <<'PY'
import json, sys
p, T = sys.argv[1], float(sys.argv[2])
d = json.load(open(p))
d.setdefault("limits", d.get("levels"))
d["benders_seconds"] = T
d.setdefault("z_seq", d["z"])  # --seq-only output: z is the cost of the sequential schedule
d.setdefault("lb", float("-inf")); d.setdefault("cuts", [])
json.dump(d, open(p, "w"), indent=1)
PY
t0=$(date +%s)
(cd $RC && python -u sg_bpb.py $D $O/bd $O/stage1 --cluster $c --time $(python -c "print(round(0.8*$T))") --sglns-frac 0 \
   --threads 3 --seed 1 --tag stage1 --smooth 0 > $O/stage1.log 2>&1)
t1=$(date +%s)
cp $O/bd/${stem}_${c}_x0.csv $O/input/; cp $O/stage1/${stem}_${c}_stage1_s1_schedule.csv $O/input/${stem}_${c}_schedule.csv
left=$((T - (t1-t0) - 400))
(cd $SU && python -u benders_sglns.py --cluster $c --data $D --benders-dir $O/input --time-limit $left --seed 1 --threads 3 \
   --start benders --prices benders --out $O/${stem}_${c}_s1.json > $O/stage2.log 2>&1)
t2=$(date +%s); echo "stage1 $((t1-t0)) stage2_search $left stage2 $((t2-t1)) total $((t2-t0)) T $T" > $O/wall.txt
echo "$(date) done $name" >> E:/schedule/revision/instances_new/progress.txt
