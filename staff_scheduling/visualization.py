import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cur_time = time.strftime("%m%d%H%M", time.localtime())

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_demand_vs_supply(df_demand, output_file):
    plt.figure(figsize=(12, 6))
    plt.plot(df_demand['hour_of_week'], df_demand['Target_Demand'], label='Demand', color='#0052cc', linewidth=2)
    plt.plot(df_demand['hour_of_week'], df_demand['Actual_Supply'], label='Supply', color='#ff9900', linestyle='--',
             linewidth=2)
    plt.xlabel('Hour of Week')
    plt.ylabel('Staff Number')
    plt.title('Staff Supply & Demand', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_gantt(df_schedule, staff_roles, output_file):
    staff_subset = df_schedule['Staff'].unique()
    df_subset = df_schedule[df_schedule['Staff'].isin(staff_subset)]
    if df_subset.empty:
        return

    plt.figure(figsize=(18, 36))
    staff_to_y = {staff_id: i for i, staff_id in enumerate(staff_subset)}

    COLOR_MANAGER = '#003366'
    COLOR_STAFF = '#66b3ff'

    for staff_id in staff_subset:
        y = staff_to_y[staff_id]
        role = staff_roles.get(staff_id, 'Crew')
        color = COLOR_MANAGER if role == 'Manager' else COLOR_STAFF

        # 1. Shifts
        staff_shifts = df_subset[df_subset['Staff'] == staff_id]
        worked_days = set(staff_shifts['Day'].unique())

        for _, row in staff_shifts.iterrows():
            plt.barh(y=y, width=row['Duration'], left=row['Start'], height=0.6, align='center',
                     color=color, edgecolor='black', linewidth=0.5)

        # 2. Days Off
        all_days = set(range(7))
        days_off = all_days - worked_days
        for day in days_off:
            plt.barh(y=y, width=24, left=day * 24, height=0.6, align='center',
                     color='none', edgecolor='black', hatch='////', linewidth=0, alpha=0.3)

    plt.yticks(list(staff_to_y.values()), list(staff_to_y.keys()), fontsize=8)
    plt.xlabel('Hour of Week')
    for d in range(1, 8):
        plt.axvline(x=d * 24, color='gray', linestyle='--')
    plt.title('Weekly Staff Schedule', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def generate_violation_distributions(violation_csv_path, output_dir):
    """
    Automatically generate magnitude distribution plots (Histograms)
    for each type of constraint violation and save them to the figures directory.
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    try:
        df_viol = pd.read_csv(violation_csv_path)
    except FileNotFoundError:
        print(f"Warning: Violation registry not found at {violation_csv_path}")
        return

    if df_viol.empty:
        print("No violations to plot. Perfect schedule!")
        return

    print(f"=== Generating Violation Distribution Plots into {figures_dir} ===")

    constraints = df_viol['Constraint'].unique()

    for constraint in constraints:
        subset = df_viol[df_viol['Constraint'] == constraint]

        plt.figure(figsize=(8, 5), dpi=300)
        max_val = subset['Violation_Amount'].max()

        if max_val <= 25:
            bins = np.arange(1, max_val + 2) - 0.5
            xticks = range(1, int(max_val) + 1)
        else:
            bins = 15
            xticks = None

        sns.histplot(
            data=subset,
            x='Violation_Amount',
            color='#d62728',
            stat='count',
            bins=bins,
            edgecolor='black',
            linewidth=1.2,
            alpha=0.75
        )

        plt.title(f'Distribution of [{constraint}] Violations', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Violation Magnitude (e.g., Excess Hours / Missing Staff)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency (Number of Incidents)', fontsize=12, fontweight='bold')

        if xticks is not None:
            plt.xticks(xticks)

        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', constraint.replace(' ', '_'))
        filename = os.path.join(figures_dir, f'violation_{safe_name}.png')

        plt.savefig(filename, format='png', bbox_inches='tight')
        plt.close()

        print(f"  -> Saved: {filename}")


def generate_visualizations(args):
    """
    Main entry point for visualization generation.
    """
    print("=== Starting Visualization Generation ===")

    if not os.path.exists(args.schedule_file):
        print(f"Error: Schedule file '{args.schedule_file}' not found.")
        return
    if not os.path.exists(args.report_file):
        print(f"Error: Report file '{args.report_file}' not found.")
        return

    df_schedule = pd.read_csv(args.schedule_file)

    try:
        df_demand = pd.read_excel(args.report_file, sheet_name='Demand_Check')
    except ValueError as e:
        print(f"Warning: Could not read Excel sheets ({e}).")
        return

    staff_roles = {}
    try:
        df_staff = pd.read_excel(args.data_file, sheet_name='staff', header=0)
        df_staff.columns = [c.strip() for c in df_staff.columns]
        if 'staff_id' in df_staff.columns and 'role' in df_staff.columns:
            staff_roles = df_staff.set_index('staff_id')['role'].to_dict()
        else:
            print("Warning: 'staff_id' or 'role' column not found in staff sheet.")
    except Exception as e:
        print(f"Warning: Could not load staff roles ({e}). Defaulting to default colors.")

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    # 1. Plot Demand vs Supply
    plot_demand_vs_supply(df_demand, os.path.join(args.plot_dir, f'1_demand_vs_supply_{cur_time}.png'))

    # 2. Plot Staff Gantt Chart
    plot_gantt(df_schedule, staff_roles, os.path.join(args.plot_dir, f'2_staff_gantt_chart_{cur_time}.png'))

    # 3. Generate Violation Distribution Histograms
    # Dynamically find the violation_registry.csv based on the report_file directory
    base_dir = os.path.dirname(args.report_file)
    violation_csv_path = os.path.join(base_dir, 'violation_registry.csv')
    generate_violation_distributions(violation_csv_path, args.plot_dir)

    print("=== Visualization Complete ===")


if __name__ == "__main__":
    from args import get_args
    results_dir = os.path.join("results", f"20260407_143845_cplex")
    args = get_args()
    args.schedule_file = os.path.join(results_dir, args.schedule_file)
    args.report_file = os.path.join(results_dir, args.report_file)
    args.plot_dir = os.path.join(results_dir, "plots")

    print("\n--- Running Visualization in Standalone Mode ---")

    generate_visualizations(args)