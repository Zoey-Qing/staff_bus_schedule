# Staff Scheduling and Bus Scheduling

## Code owner
Qing Lyu, Xuanmian He

## Last Update Date
2026/03/27

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
