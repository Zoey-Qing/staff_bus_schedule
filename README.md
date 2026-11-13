# Staff Scheduling and Bus Scheduling

## Code owner
Qing Lyu, Xuanmian He

## Last Update Date
2026/11/13

## File Description
- staff_scheduling/

  -- args.py         Set constraint hyperparameters, data/results saving path
  
  -- data_loader.py  Load dataset
  
  -- evaluation.py   Evaluate the demand shortage, overtime work conditions, etc.
  
  -- solver_##.py    Implementation on cplex and highs
  
  -- main.py

## Run codes
```sh
python staff_scheduling/main.py --solver cplex
```


## Bus Scheduling and Routing

The `bus_scheduling_routing` module includes three methods:

1. **Column Generation**  
   - File: `bus_scheduling_routing/column_gen.py`  
   - Description: Implements a column generation approach for large-scale bus scheduling or routing problems.

2. **Fixed Cluster**  
   - File: `bus_scheduling_routing/fleet_schedule.ipynb`  
   - Description: Implements a fixed cluster based approach.

3. **Route Pool Generation**  
   - File: `bus_scheduling_routing/fleet_schedule.ipynb`  
   - Description: Implements a route pool generation approach.


## Integrated staff and bus planning

Each cluster is solved as its own instance. Both methods below use the same start schedule, service limits and routing model, and score a schedule by the same realized cost.

- `integrated/net_routing.py`: routing on a time-space network, solved by column generation and an integer program over the generated trips
- `integrated/net_benders.py`: Benders decomposition with column generation, and the re-timing step
- `surrogate/staff_surrogate.py`: surrogate-fit heuristic, with the same routing model and re-timing step
- `integrated/test_net_routing.py`: checks of the routing on small cases against a brute-force model

```sh
python integrated/net_benders.py staff_scheduling/Berkeley_Mock_Data_25_Stores.xlsx <out_prefix> --cluster <id prefix>
python surrogate/staff_surrogate.py --cluster <id prefix> --data <data.xlsx> --schedule <out_prefix>_x0.csv --time-limit <seconds> --out <file.json>
```

CPLEX and docplex are required. All data are synthetic.

## Method of the paper (branch-roster column generation + SG-LNS)

The main method runs in two steps on one cluster:

1. `integrated/roster_cg/sg_bpb.py`: column generation over branch rosters with lifted Benders cuts. It returns a
   lower bound LB, the cut prices, and a schedule x1.
2. `surrogate/benders_sglns.py`: the surrogate-guided neighborhood search (SG-LNS) started from x1, with opening
   prices taken from the lifted cuts; the transport-cost estimate is updated by ridge regression after every round.

```sh
cd integrated/roster_cg
python sg_bpb.py <data.xlsx> <benders_dir> <out_dir> --cluster <prefix> --time <seconds> --sglns-frac 0 --seed 1 --tag stage1
cd ../../surrogate
python benders_sglns.py --cluster <prefix> --data <data.xlsx> --benders-dir <dir with x0 and stage-1 schedule> \
    --time-limit <seconds> --seed 1 --start benders --prices benders --out <file.json>
```

`experiments/results/ablation_m5eq/run_ablation.sh` shows the full pipeline as it was run.

Other files:
- `integrated/roster_cg/branch_dw.py`, `staff_dw.py`: branch-roster and staff-roster column generation with Benders cuts
  (baselines and building blocks); `ormeci_baseline.py`: transport-aware staff MIP baseline; `check_*.py`: small checks.
- `integrated/verify_cuts.py`, `verify_routing.py`: numerical checks of the Benders cut and of the routing.
- `integrated/master_study/`: study of the Benders master problem.
- `surrogate/lns_variants.py`, `sglns_cutprices.py`: search variants used in the ablations.
- `integrated/data_gen/make_far2_original.py`: the "far x2" variant of the original data.

## Experiments of the paper (`experiments/`)

- `experiments/results/`: all runs on the original data, seed 1, with the run scripts (`*.sh`), the summaries
  (`summarize.py`, `summary.csv`) and the raw outputs (json, schedules as csv). `README.txt` gives the order of the runs.
  Log files are not included.
- `experiments/instances_new/`: generated instances (tiny T1-T3, larger G1-G5) from `make_instances.py`, the run
  pipeline `pipeline.sh`, the exact check on the tiny instances `exact_tiny.py` (`exact_T*.json`), and `summarize.py`.
- `experiments/figures/`: the matplotlib scripts of the paper figures (PDF and PNG).

The scripts were run from a working copy at `E:/schedule` and use absolute paths; change `R`, `D`, `RC`, `SU` at the top
of a script to your own paths. They refer to the data file as `Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx`;
this is the file `staff_scheduling/Berkeley_Mock_Data_25_Stores.xlsx` of this repository (identical content).

## Data

- `staff_scheduling/Berkeley_Mock_Data_25_Stores.xlsx`: the original synthetic data. The experiments use this file.
- `staff_scheduling/Berkeley_Mock_Data_25_Stores_rebalanced.xlsx`: the same data with staff home branches re-assigned in proportion to branch demand, so that each branch can cover its own demand. Only the `branch_id` and `accommodation_id` columns of the `staff` sheet differ (94 of 260 staff move). It is made by `staff_scheduling/rebalance_staff.py` from the original file; `staff_scheduling/rebalance_report.csv` lists the result per branch.
- `integrated/data_gen/`: networks generated by `integrated/generate_network.py`, with the rebalanced data as template. `net25_s*` keep the base distances; `net25_far*` multiply the branch-to-accommodation distances by 1.5, 2 or 3.
