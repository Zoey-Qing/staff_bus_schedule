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
