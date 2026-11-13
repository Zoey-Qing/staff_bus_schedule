#!/bin/bash
# Replaces steps 2-5 of run_tre.sh after the small-network Benders runs (the small masters do not close their gap,
# so SG-LNS on them gets one seed and moves to the end). Every step skips runs that already have results.
O=/e/schedule/revision/results_new/original
D=/e/schedule/integrated/data_gen
BASE=/e/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx
until grep -q "ALL DONE" $O/small_benders_batch.log 2>/dev/null; do sleep 120; done

# lifting ablation on the base network: closed stops priced at min(0, lifted price), same start schedules
mkdir -p $O/benders_nolift
cp -n $O/benders/*_x0.csv $O/benders_nolift/
cd /e/schedule/integrated
python -u run_benders_batch.py $O/benders_nolift $BASE --jobs 5 --threads 3 --closed-price zero > $O/benders_nolift_batch.log 2>&1
echo NOLIFT DONE

# structural families of medium size: compact, far (same draws, distances x2), two accommodations per cluster
python -u run_benders_batch.py $O/families/benders $D/syn_m_compact_s1.xlsx $D/syn_m_far2_s1.xlsx $D/syn_m_2acc_s1.xlsx \
  --jobs 6 --threads 2 > $O/families_benders_batch.log 2>&1
cd /e/schedule/surrogate
python -u run_equal_time.py $O/families/benders $O/families/lns_fit --seeds 1 --ablation lns_fit --jobs 6 --threads 2 > $O/families_lns_fit_batch.log 2>&1
echo FAMILIES DONE

# SG-LNS without the random perturbation of the coefficients, base network, seed 1
python -u run_equal_time.py $O/benders $O/ablation_lns_fit_noise0 --seeds 1 --ablation lns_fit --noise 0 > $O/ablation_noise0_batch.log 2>&1
echo NOISE0 DONE

# SG-LNS on the small networks, seed 1
python -u run_equal_time.py $O/small/benders $O/small/lns_fit --seeds 1 --ablation lns_fit --jobs 6 --threads 2 > $O/small_lns_fit_batch.log 2>&1
echo TRE DONE >> $O/run_tre.log
echo TRE DONE
