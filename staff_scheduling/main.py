import sys
import os
import datetime
import pandas as pd
from args import get_args
from data_loader import DataLoader
from evaluation_new import run_evaluation
from visualization import generate_visualizations

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    args = get_args()
    print("=== Staff Scheduling System (Cluster-By-Cluster) ===")
    solver_choice = getattr(args, 'solver_engine', 'ortools').lower()
    if solver_choice == 'cplex':
        print("  - Solver Engine: CPLEX")
        from solver_cplex_new import StaffSchedulingSolver
    elif solver_choice == 'highs':
        print("  - Solver Engine: HiGHS")
        from solver_highs_new import StaffSchedulingSolver
    else:
        print("  - Solver Engine Not Found. Choose 'cplex' or 'highs'.")
        exit(1)

    output_dir = os.path.join("results", f"{timestamp}_{solver_choice}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  - Output Directory: {output_dir}")

    log_filename = os.path.join(output_dir, f"run_log_{timestamp}.txt")
    sys.stdout = Logger(log_filename)
    print(f"=== All console output will be saved to: {log_filename} ===")

    original_schedule_file = getattr(args, 'schedule_file', 'schedule_result.csv')
    original_report_file = getattr(args, 'report_file', 'verification_report.xlsx')

    args.schedule_file = os.path.join(output_dir, os.path.basename(original_schedule_file))
    args.report_file = os.path.join(output_dir, os.path.basename(original_report_file))
    args.plot_dir = os.path.join(output_dir, "plots")

    baseline_assignments = None
    baseline_file = getattr(args, 'baseline_schedule', 'schedule_result.csv')
    bus_file = getattr(args, 'bus_fleet_schedule', 'bus_fleet_schedule.csv')

    if args.enforce_bus:
        print(f"  - Bus Constraint: ON (Using {bus_file})")
        if os.path.exists(baseline_file):
            print(f"  - Mode: FINETUNE (Found baseline: {baseline_file})")
            try:
                df_baseline = pd.read_csv(baseline_file)
                baseline_assignments = set()
                for _, row in df_baseline.iterrows():
                    baseline_assignments.add((row['Staff'], row['Shift']))
            except Exception as e:
                print(f"Error reading baseline schedule '{baseline_file}': {e}")
                sys.exit(1)

            args.schedule_file = os.path.join(output_dir, "finetuned_" + os.path.basename(original_schedule_file))
            args.report_file = os.path.join(output_dir, "finetuned_" + os.path.basename(original_report_file))
        else:
            print(f"  - Mode: NORMAL (From scratch with bus constraints)")
    else:
        print("  - Mode: NORMAL (From scratch without bus constraints)")
        print("  - Bus Constraint: OFF")

    try:
        df_branches = pd.read_excel(args.data_file, sheet_name="branches", header=0)
        df_branches.columns = [c.strip() for c in df_branches.columns]
        all_clusters = df_branches['cluster_id'].unique().tolist()
    except Exception as e:
        print(f"Failed to read clusters from branches sheet: {e}")
        sys.exit(1)

    all_results = []
    all_violations = []

    for cluster_id in all_clusters:
        print(f"\n{'=' * 60}")
        print(f"SOLVING FOR CLUSTER: {cluster_id}")
        print(f"{'=' * 60}")

        loader = DataLoader(args, target_cluster=cluster_id)
        try:
            loader.load_and_preprocess()
        except Exception as e:
            print(f"Error loading data for cluster {cluster_id}: {e}")
            continue

        if not loader.staff_ids:
            print(f">>> Skipping cluster {cluster_id} (No staff assigned).")
            continue

        solver = StaffSchedulingSolver(args, loader, baseline_assignments=baseline_assignments)
        solver.build_model()
        is_solved = solver.solve()

        if is_solved:
            obj_val = solver.get_objective_value()
            print(f">>> Objective Value for {cluster_id}: {obj_val:.2f}")

            df_res = solver.export_results()
            all_results.append(df_res)

            # --- NEW: Extract soft constraint violations for this cluster ---
            if hasattr(solver, 'export_violate_registry'):
                df_viol = solver.export_violate_registry()
                if not df_viol.empty:
                    # Optional: Add cluster ID column for tracking
                    df_viol.insert(0, 'Cluster', cluster_id)
                    all_violations.append(df_viol)
        else:
            print(f">>> Optimization failed for cluster {cluster_id}.")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(args.schedule_file, index=False)
        print(f"\n>>> Global raw schedule saved to: {args.schedule_file}")

        if all_violations:
            final_viol_df = pd.concat(all_violations, ignore_index=True)
            viol_file_path = os.path.join(output_dir, "violation_registry.csv")
            final_viol_df.to_csv(viol_file_path, index=False)
            print(f">>> Soft Constraint Violations Registry saved to: {viol_file_path}")
        else:
            print("\n>>> No soft constraints were violated globally.")

        print("\n>>> Generating Global Evaluation Report & Plots...")
        global_loader = DataLoader(args, target_cluster=None)
        global_loader.load_and_preprocess()

        run_evaluation(args, final_df, global_loader)
        generate_visualizations(args)
    else:
        print("\n>>> Critical Error: No solutions found for any cluster.")


if __name__ == "__main__":
    main()