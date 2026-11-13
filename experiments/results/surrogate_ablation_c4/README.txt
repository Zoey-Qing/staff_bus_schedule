Stage-2 ablation on C3 of the paper (cluster ba5b9796; C4 in the old five-cluster numbering), 2026-11-10.
Same start x^1 (ablation_m5eq/ba5b9796/A4_no_smooth/input), same search time 1122 s, seeds 1-3, four runs at once.
Code: surrogate/lns_variants.py (imports benders_sglns.py and staff_surrogate.py unchanged). Script: run.sh.
  plain         repair minimizes labor cost only (no transport term, no refit)
  fixed_cut     transport term = stop prices of the lifted cut at x^1, never refitted
  fixed_direct  transport term = cheapest direct round trip, never refitted
  learned       SG-LNS of the paper (stop prices at x^1, ridge refit every round); seed 1 reproduces the main run
z(x0) = 119,022.93, z(x^1) = 118,524.80 (saving 0.419%), LB* = 115,911.74.
Mean +- sd over three seeds (final z after re-timing):
  variant        saving %        share %       cost reduction over x^1   rounds
  plain          0.526 +- 0.016  20.1 +- 0.6   128.4 +- 18.7             20
  fixed_cut      0.560 +- 0.014  21.4 +- 0.6   168.0 +- 17.2             18
  fixed_direct   0.560 +- 0.021  21.4 +- 0.8   168.0 +- 24.6             18
  learned        0.535 +- 0.029  20.5 +- 1.1   139.1 +- 35.0             18
Reading: a transport estimate in the repair problem helps over plain LNS; refitting it by ridge regression does not
help on this instance.
Paper (2026-11-10, user: seed 1 only): Table tab:stage2 reports seed 1; seed 1 corr(pred,real): fixed_cut 0.58, fixed_direct 0.87, learned 0.85.
