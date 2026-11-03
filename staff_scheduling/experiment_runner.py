import os
import sys
import time
import random
import gc
import pandas as pd
import numpy as np

from args import get_args
from data_loader import DataLoader
from solver_cplex_new import StaffSchedulingSolver


def extract_and_save_detailed_kpis(solver, loader, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    df_viol = solver.export_violate_registry()
    viol_path = os.path.join(run_dir, "violation_registry.csv")
    if not df_viol.empty:
        df_viol.to_csv(viol_path, index=False)
    else:
        pd.DataFrame(columns=["Staff", "Branch", "Day/Time", "Constraint", "Violation_Amount"]).to_csv(viol_path,
                                                                                                       index=False)

    gaps = []
    if solver.solution:
        for t in loader.time_slots:
            gap_val = solver.solution.get_value(solver.sl[t])
            if gap_val > 0.01:
                gaps.append({'Time': t, 'Missing_Staff': round(gap_val, 2)})

    gap_path = os.path.join(run_dir, "demand_gaps.csv")
    df_gaps = pd.DataFrame(gaps)
    if not df_gaps.empty:
        df_gaps.to_csv(gap_path, index=False)
    else:
        pd.DataFrame(columns=["Time", "Missing_Staff"]).to_csv(gap_path, index=False)


def run_comprehensive_experiments():
    args = get_args()
    TARGET_CLUSTER = "df75e3dc-d14a-4ed7-a41b-10ef3979ac47"
    args.solver_engine = 'cplex'
    args.time_limit = 1200

    BASE_OUTPUT_DIR = "academic_experiments"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    results_summary = []

    print("=" * 60)
    print(f"=== INITIATING ACADEMIC GRID SEARCH EXPERIMENTS ===")
    print("=" * 60)

    # =========================================================
    # [COMMENTED OUT] EXPERIMENT 1: Big-M Sensitivity Analysis
    # =========================================================
    print("\n>>> [Experiment 1] Big-M Sensitivity Analysis")
    m3_grid = [10, 100, 1000, 5000]

    for m3_val in m3_grid:
        scenario_name = f"Exp1_M3_{m3_val}"
        run_dir = os.path.join(BASE_OUTPUT_DIR, scenario_name)
        print(f"  -> Testing {scenario_name}...")

        loader = DataLoader(args, target_cluster=TARGET_CLUSTER)
        loader.load_and_preprocess()

        args.M1 = 1000
        args.M2 = 100
        args.M3 = m3_val
        args.M4 = 1
        args.noise_level = 0.001

        solver = StaffSchedulingSolver(args, loader)
        solver.build_model()

        start_t = time.time()
        is_solved = solver.solve()
        solve_t = time.time() - start_t

        obj_val = solver.get_objective_value() if is_solved else "N/A"

        extract_and_save_detailed_kpis(solver, loader, run_dir)

        results_summary.append({
            'Phase': 'Big-M',
            'Scenario': scenario_name,
            'Solve_Time_s': round(solve_t, 2),
            'Obj': obj_val
        })

        solver.mdl.end()
        del solver
        gc.collect()

    summary_csv_exp1 = os.path.join(BASE_OUTPUT_DIR, "macro_summary_exp1.csv")
    pd.DataFrame(results_summary).to_csv(summary_csv_exp1, index=False)
    print(f"\n>>> Exp 1 Grid Search Completed. Summary saved to {summary_csv_exp1}")

    # =========================================================
    # EXPERIMENT 3: Demand / Staffing Sensitivity
    # =========================================================
    print("\n>>> [Experiment 3] Demand Sensitivity (Workload Stress Test)")
    demand_scales = [0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2]

    for scale in demand_scales:
        scenario_name = f"Exp3_Demand_{scale}"
        run_dir = os.path.join(BASE_OUTPUT_DIR, scenario_name)
        print(f"  -> Testing {scenario_name}...")

        loader = DataLoader(args, target_cluster=TARGET_CLUSTER)
        loader.load_and_preprocess()

        # Scale the demand in-place to simulate busy/quiet seasons
        if hasattr(loader, 'demand'):
            for t in loader.demand:
                loader.demand[t] = np.round(loader.demand[t] * scale, 0).astype(int)
                # Round to nearest integer after scaling
            for bt in loader.branch_demand:
                loader.branch_demand[bt] = int(np.round(loader.branch_demand[bt] * scale, 0))

        # Keep M-weights and noise stable for this experiment
        args.M1, args.M2, args.M3, args.M4 = 1000, 100, 10, 1
        args.noise_level = 0.001

        solver = StaffSchedulingSolver(args, loader)
        solver.build_model()

        start_t = time.time()
        is_solved = solver.solve()
        solve_t = time.time() - start_t

        obj_val = solver.get_objective_value() if is_solved else "N/A"

        extract_and_save_detailed_kpis(solver, loader, run_dir)

        results_summary.append({
            'Phase': 'Demand', 'Scenario': scenario_name, 'Solve_Time_s': round(solve_t, 2), 'Obj': obj_val
        })

        solver.mdl.end()
        del solver
        gc.collect()

    summary_csv = os.path.join(BASE_OUTPUT_DIR, "macro_summary_exp3.csv")
    pd.DataFrame(results_summary).to_csv(summary_csv, index=False)
    print(f"\n>>> Exp 3 Grid Search Completed. Summary saved to {summary_csv}")


if __name__ == "__main__":
    run_comprehensive_experiments()