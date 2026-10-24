import sys
import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from args import get_args
from data_loader import DataLoader
from solver_cplex_new import StaffSchedulingSolver
from evaluation_new import run_evaluation
from visualization import generate_visualizations


def extract_kpis_from_dir(directory):
    kpis = {
        "Total_Understaffing": 0,
        "Max_Daily_Hours_Violations": 0,
        "Max_Shift_Length_Violations": 0
    }

    demand_file = os.path.join(directory, "verification_report.xlsx")
    viol_file = os.path.join(directory, "violation_registry.csv")

    try:
        if os.path.exists(demand_file):
            df_demand = pd.read_excel(demand_file, sheet_name="Demand_Check")
            understaffing = df_demand[df_demand['Gap'] < 0]['Gap'].sum()
            kpis["Total_Understaffing"] = abs(understaffing)

        if os.path.exists(viol_file):
            df_viol = pd.read_csv(viol_file)
            grouped = df_viol.groupby('Constraint')['Violation_Amount'].sum()
            kpis["Max_Daily_Hours_Violations"] = grouped.get("Max Daily Hours", 0.0)
            kpis["Max_Shift_Length_Violations"] = grouped.get("Max Shift Length", 0.0)
    except Exception as e:
        print(f"Warning: Could not extract full KPIs from {directory}. Details: {e}")

    return kpis


def plot_milestone_bus_comparison(df_buses, routes_dict, output_path):
    plt.figure(figsize=(12, 6))

    colors = ['gray', '#2ca02c', '#ff7f0e', '#d62728']
    linestyles = ['--', '-', '-', '-']

    idx = 0
    for label, active_routes in routes_dict.items():
        if active_routes is None:
            continue

        if "Baseline" in label:
            df_subset = df_buses.copy()
        else:
            df_subset = df_buses[df_buses['route_id'].isin(active_routes)].copy()

        counts = df_subset.groupby('time_mod')['route_id'].nunique().reindex(range(168), fill_value=0)

        plt.plot(counts.index, counts.values, label=label,
                 color=colors[idx % len(colors)],
                 linestyle=linestyles[idx % len(linestyles)],
                 linewidth=2.5 if "Baseline" not in label else 2.0,
                 alpha=0.8)
        idx += 1

    plt.title("Bus Units Required per Hour at Different Capacity Milestones", fontsize=14, fontweight='bold')
    plt.xlabel("Hour of Week (0-167)", fontsize=12)
    plt.ylabel("Number of Active Bus Routes", fontsize=12)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] Milestone bus usage comparison chart saved to: {output_path}")


def plot_pareto_frontier(df_pareto, output_path):
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    df_plot = df_pareto[df_pareto['Peak_Buses'] >= 1].copy()
    df_sorted = df_plot.sort_values(by="Peak_Buses", ascending=False)
    x = df_sorted["Peak_Buses"].astype(str)

    ax1.plot(x, df_sorted["Max_Daily_Hours_Violations"], color='#d62728', marker='o', label='Overtime (Hours)',
             linewidth=2.5)
    ax2.plot(x, df_sorted["Total_Understaffing"], color='#1f77b4', marker='s', label='Understaffing (Hours)',
             linewidth=2.5)

    ax1.set_xlabel("Peak Bus Capacity Limit", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Total Overtime Hours", color='#d62728', fontsize=12, fontweight='bold')
    ax2.set_ylabel("Total Understaffing Hours", color='#1f77b4', fontsize=12, fontweight='bold')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2,
               frameon=False)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Pareto frontier chart saved to: {output_path}")


def main():
    args = get_args()

    args.enforce_bus = True
    args.mip_gap = 0.01
    args.time_limit = 600

    baseline_dir = getattr(args, 'baseline_dir', os.path.normpath("results/20260405_094036_cplex"))

    base_folder_name = os.path.basename(baseline_dir)
    parent_dir = os.path.dirname(baseline_dir)
    output_dir = os.path.join(parent_dir, f"{base_folder_name}_finetuned_iterative_2")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Iterative Finetuning for Absolute Minimum Bus Capacity")
    print(f"Baseline Directory: {baseline_dir}")
    print(f"Output Directory  : {output_dir}")
    print("=" * 60)

    loader = DataLoader(args)
    loader.load_and_preprocess()

    baseline_file = os.path.join(baseline_dir, 'schedule_result.csv')
    if not os.path.exists(baseline_file):
        print(f"Error: Baseline schedule not found at '{baseline_file}'")
        sys.exit(1)

    baseline_assignments = set()
    try:
        df_baseline = pd.read_csv(baseline_file)
        for _, row in df_baseline.iterrows():
            baseline_assignments.add((row['Staff'], row['Shift']))
        print(f"[Data] Loaded {len(baseline_assignments)} assignments from baseline.")
    except Exception as e:
        print(f"Error reading baseline schedule: {e}")
        sys.exit(1)

    bus_schedule_file = 'bus_scheduling_results/bus_fleet_schedule_dynamic.csv'
    if not os.path.exists(bus_schedule_file):
        bus_schedule_file = 'bus_fleet_schedule_dynamic.csv'
        if not os.path.exists(bus_schedule_file):
            print(f"Error: Dynamic bus file not found.")
            sys.exit(1)

    df_buses = pd.read_csv(bus_schedule_file)
    df_buses['time_mod'] = df_buses['time'].astype(int) % 168
    route_ids = df_buses['route_id'].tolist()

    routes_per_time = df_buses.groupby('time_mod')['route_id'].nunique()
    peak_buses_before = routes_per_time.max()

    print(f"[Data] Peak concurrent buses (Bottleneck) before: {peak_buses_before}")

    target_75 = int(round(peak_buses_before * 0.75))
    target_50 = int(round(peak_buses_before * 0.50))
    target_25 = int(round(peak_buses_before * 0.25))

    milestone_routes = {
        'Baseline (100%)': route_ids,
        f'75% Capacity (Peak {target_75})': None,
        f'50% Capacity (Peak {target_50})': None,
        f'25% Capacity (Peak {target_25})': None
    }

    branch_time_to_routes = {}
    time_to_routes = {t: [] for t in range(168)}

    for _, row in df_buses.iterrows():
        r_id = row['route_id']
        t = row['time_mod']

        time_to_routes[t].append(r_id)

        try:
            stops = ast.literal_eval(row['stops'])
        except:
            stops = [row['stops']]

        for branch in stops:
            if (branch, t) not in branch_time_to_routes:
                branch_time_to_routes[(branch, t)] = []
            branch_time_to_routes[(branch, t)].append(r_id)

    kpi_records = []
    baseline_kpis = extract_kpis_from_dir(baseline_dir)
    baseline_kpis["Peak_Buses"] = peak_buses_before
    baseline_kpis["Iteration"] = "Baseline"
    kpi_records.append(baseline_kpis)

    current_peak_buses = peak_buses_before + 1

    while current_peak_buses > 0:
        target_peak_buses = current_peak_buses - 1
        print("\n" + "-" * 50)
        print(f"[Iteration] Attempting to compress network to Max {target_peak_buses} buses/hour")
        print("-" * 50)

        solver = StaffSchedulingSolver(args, loader, baseline_assignments=baseline_assignments)
        solver.build_model()
        mdl = solver.mdl

        y_route = mdl.binary_var_dict(route_ids, name="dyn_route_active")

        for t in range(168):
            if len(time_to_routes[t]) > 0:
                mdl.add_constraint(mdl.sum(y_route[r] for r in time_to_routes[t]) <= target_peak_buses, f"cap_time_{t}")

        expat_set = set(loader.non_saudi_staff_ids)

        for i in loader.staff_ids:
            if i not in expat_set:
                continue

            branch = loader.staff_branch_map.get(i)
            if not branch:
                continue

            for s in loader.shifts:
                sid = s['id']
                if (i, sid) not in solver.x:
                    continue

                var_x = solver.x[i, sid]
                t_start = s['start'] % 168
                t_end = s['end'] % 168

                valid_starts = branch_time_to_routes.get((branch, t_start), [])
                valid_ends = branch_time_to_routes.get((branch, t_end), [])

                if not valid_starts or not valid_ends:
                    mdl.add_constraint(var_x == 0)
                else:
                    mdl.add_constraint(var_x <= mdl.sum(y_route[r] for r in valid_starts))
                    mdl.add_constraint(var_x <= mdl.sum(y_route[r] for r in valid_ends))

        is_solved = solver.solve()

        if is_solved:
            print(
                f"[Success] Found feasible schedule for Peak = {target_peak_buses}. Obj: {solver.get_objective_value():.2f}")

            active_routes = [r for r in route_ids if solver.solution.get_value(y_route[r]) >= 0.5]

            if target_peak_buses == target_75:
                milestone_routes[f'75% Capacity (Peak {target_75})'] = active_routes
            elif target_peak_buses == target_50:
                milestone_routes[f'50% Capacity (Peak {target_50})'] = active_routes
            elif target_peak_buses == target_25:
                milestone_routes[f'25% Capacity (Peak {target_25})'] = active_routes

            iter_dir = os.path.join(output_dir, f"peak_capacity_{target_peak_buses}")
            os.makedirs(iter_dir, exist_ok=True)

            df_res = solver.export_results()
            out_schedule = os.path.join(iter_dir, "schedule_result.csv")
            df_res.to_csv(out_schedule, index=False)

            args.schedule_file = out_schedule
            args.report_file = os.path.join(iter_dir, "verification_report.xlsx")
            args.plot_dir = os.path.join(iter_dir, "plots")
            os.makedirs(args.plot_dir, exist_ok=True)

            run_evaluation(args, df_res, loader)
            df_viol = solver.export_violate_registry()
            if not df_viol.empty:
                df_viol.to_csv(os.path.join(iter_dir, "violation_registry.csv"), index=False)
            generate_visualizations(args)

            iter_kpis = extract_kpis_from_dir(iter_dir)
            iter_kpis["Peak_Buses"] = target_peak_buses
            iter_kpis["Iteration"] = f"Peak_{target_peak_buses}"
            kpi_records.append(iter_kpis)

            current_peak_buses = target_peak_buses

        else:
            print(
                f"\n[Infeasible] System reached theoretical limit. Cannot satisfy constraints with {target_peak_buses} buses.")
            break

    print("\n" + "=" * 60)
    print("Generating Global Pareto Analysis Metrics")
    print("=" * 60)

    df_pareto = pd.DataFrame(kpi_records)
    cols = ["Iteration", "Peak_Buses", "Total_Understaffing", "Max_Daily_Hours_Violations",
            "Max_Shift_Length_Violations"]
    df_pareto = df_pareto[cols]

    pareto_csv_path = os.path.join(output_dir, "pareto_frontier_summary.csv")
    df_pareto.to_csv(pareto_csv_path, index=False)

    print("\nIterative Optimization Summary:")
    print(df_pareto.to_string(index=False))

    plot_pareto_frontier(df_pareto, os.path.join(output_dir, "pareto_frontier_tradeoff.png"))

    plot_milestone_bus_comparison(df_buses, milestone_routes,
                                  os.path.join(output_dir, "milestone_bus_usage_comparison.png"))

    print(f"\n[Done] All iterative finetuning results and Pareto analyses saved to: {output_dir}")


if __name__ == "__main__":
    main()