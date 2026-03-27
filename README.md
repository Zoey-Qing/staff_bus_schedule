# Staff Scheduling and Bus Scheduling

## Code owner: 
Qing Lyu, Xuanmian He

## Last Update Date: 
2026/03/27

## File Description:
--staff_scheduling/
  --args.py  \t Set constraint hyperparameters, data/results saving path
  --data_loader.py  \t load dataset
  --evaluation.py  \t evaluate the demand shortage, overtime work conditions, etc.
  --solver_##.py  \t Implementation on cplex and highs
  --main.py  \t

```sh
python main.py --solver cplex
```
