Ablations of the main method without dual smoothing (seed 1, five base clusters, same budget T as compare_all).
Main method = ../ablation_m5eq/<c>/A4_no_smooth. Every row here differs from it in one part only (see run.sh).
summarize.py -> summary.csv, pivot.csv. All solution and bound checks passed.
Best bound LB* per cluster (now all from no-smoothing runs): C1 112081.97 (A3 stage 1), C2 138067.48 (M2),
C3 122818.74 (M2), C4 115911.74 (M2), C5 114605.41 (A3 stage 1). Provable max saving 1.02/2.21/3.01/2.61/1.50 %.
Mean saving % / share of provable max %: main 0.683/33.0; A1 0.669/32.4; A2 0.209/9.5; A3 0.648/32.4;
A5 0.628/31.1; A6 0.360/17.7; M2 0.535/26.8; with smoothing 0.634/30.8; M3 0.396/18.8; L3 0.239/14.2.
