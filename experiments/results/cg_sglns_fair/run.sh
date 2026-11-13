#!/bin/bash
# Equal-time version on C1 (seed 1): whole pipeline within T = 6206 s of wall clock.
# Stage 1: sg_bpb.py --sglns-frac 0 with 0.8 T. Stage 2: benders_sglns.py (unchanged) from the stage-1 schedule, with the
# search time = T - elapsed - 400 s (kept for set-up, prices and the final re-timing). The wall clock of both stages is logged.
R=E:/schedule/revision/results_new/original; O=$R/cg_sglns_fair; T=6206
D=E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
S=Berkeley_Mock_Data_25_Stores_original_unbalanced_1aea822f
t0=$(date +%s)
cd E:/schedule/integrated/roster_cg
python -u sg_bpb.py $D $R/benders $O/stage1 --cluster 1aea --time 4965 --sglns-frac 0 --threads 3 --seed 1 --tag sgcg_frac0_t08 > $O/stage1.log 2>&1
t1=$(date +%s); echo "stage 1 wall $((t1-t0)) s" >> $O/wall.txt
cp $R/benders/${S}_x0.csv $O/input/
cp $O/stage1/${S}_sgcg_frac0_t08_s1_schedule.csv $O/input/${S}_schedule.csv
left=$((T - (t1-t0) - 400))
echo "stage 2 search time $left s" >> $O/wall.txt
cd E:/schedule/surrogate
python -u benders_sglns.py --cluster 1aea822f --data $D --benders-dir $O/input --time-limit $left --seed 1 --threads 3 --start benders --prices benders --out $O/${S}_s1.json > $O/stage2.log 2>&1
t2=$(date +%s); echo "stage 2 wall $((t2-t1)) s" >> $O/wall.txt; echo "total wall $((t2-t0)) s of $T" >> $O/wall.txt
echo FINISHED >> $O/wall.txt
