#!/bin/bash
# Additional experiments for the TR-E revision (started 2026-11-05). Every step skips runs that already have results.
O=/e/schedule/revision/results_new/original
D=/e/schedule/integrated/data_gen
BASE=/e/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
STEM=Berkeley_Mock_Data_25_Stores_original_unbalanced

# 1. numerical check of the cuts (Proposition 2): dual feasibility of the full routing LP and random stop sets
cd /e/schedule/integrated
mkdir -p $O/verify_cuts
for c in 1aea822f 662b3495 a1e8b5a7 ba5b9796 df75e3dc; do
  [ -f $O/verify_cuts/base_$c.json ] || python -u verify_cuts.py $BASE $O/benders/${STEM}_${c}_x0.csv --cluster $c --random 10 --threads 6 --out $O/verify_cuts/base_$c.json > $O/verify_cuts/base_$c.log 2>&1
done
for c in 1273d27b 69f22899; do
  [ -f $O/verify_cuts/net100_$c.json ] || python -u verify_cuts.py $D/net100_k10x2_s1.xlsx $O/net100/benders/net100_k10x2_s1_${c}_x0.csv --cluster $c --random 5 --threads 6 --out $O/verify_cuts/net100_$c.json > $O/verify_cuts/net100_$c.log 2>&1
done
echo STEP1 DONE

# 2. small instances: Benders with up to ten 10-minute masters, then SG-LNS with three seeds at equal time
python -u run_benders_batch.py $O/small/benders $D/syn_small_s1.xlsx $D/syn_small_s2.xlsx $D/syn_small_s3.xlsx --jobs 5 --threads 3 \
  --master-times 600,600,600,600,600,600,600,600,600,600 > $O/small_benders_batch.log 2>&1
cd /e/schedule/surrogate
python -u run_equal_time.py $O/small/benders $O/small/lns_fit --seeds 1,2,3 --ablation lns_fit > $O/small_lns_fit_batch.log 2>&1
echo STEP2 DONE

# 3. lifting ablation on the base network: closed stops priced at min(0, lifted price), same start schedules
mkdir -p $O/benders_nolift
cp -n $O/benders/*_x0.csv $O/benders_nolift/
cd /e/schedule/integrated
python -u run_benders_batch.py $O/benders_nolift $BASE --jobs 5 --threads 3 --closed-price zero > $O/benders_nolift_batch.log 2>&1
echo STEP3 DONE

# 4. structural families of medium size (10 branches per cluster): compact, far (same draws, distances x2), two accommodations
python -u run_benders_batch.py $O/families/benders $D/syn_m_compact_s1.xlsx $D/syn_m_far2_s1.xlsx $D/syn_m_2acc_s1.xlsx \
  --jobs 6 --threads 2 > $O/families_benders_batch.log 2>&1
cd /e/schedule/surrogate
python -u run_equal_time.py $O/families/benders $O/families/lns_fit --seeds 1 --ablation lns_fit --jobs 6 --threads 2 > $O/families_lns_fit_batch.log 2>&1
echo STEP4 DONE

# 5. SG-LNS without the random price perturbation, base network, three seeds
python -u run_equal_time.py $O/benders $O/ablation_lns_fit_noise0 --seeds 1,2,3 --ablation lns_fit --noise 0 > $O/ablation_noise0_batch.log 2>&1
echo TRE DONE
