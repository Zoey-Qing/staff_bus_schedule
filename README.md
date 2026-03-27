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
