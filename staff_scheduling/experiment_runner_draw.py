import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

def get_clean_label(scenario_name):
    if "Exp1_M3_" in scenario_name:
        val = scenario_name.split("_")[-1]
        return f"M3 = {val}"
    elif "Exp2_Noise_" in scenario_name:
        val = scenario_name.split("_")[-1]
        return f"Noise = {val}"
    elif "Exp3_Demand_" in scenario_name:
        val = scenario_name.split("_")[-1]
        return f"Demand = {val}x"
    return scenario_name

def load_experiment_data(base_dir="academic_experiments"):
    exp1_data = {'understaffing': [], 'max_daily': [], 'max_shift': []}
    exp2_data = {'understaffing': [], 'max_daily': [], 'max_shift': []}
    exp3_data = {'understaffing': [], 'max_daily': [], 'max_shift': []}

    exp1_scenarios = {}
    exp2_scenarios = {}
    exp3_scenarios = {}

    if not os.path.exists(base_dir):
        print(f"[Error] Directory not found: {base_dir}")
        return None, None, None, None, None, None

    for folder in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, folder)
        if not os.path.isdir(dir_path):
            continue

        clean_label = get_clean_label(folder)
        try:
            sort_key = float(folder.split("_")[-1])
        except ValueError:
            sort_key = 0.0

        if "Exp1" in folder:
            exp1_scenarios[clean_label] = sort_key
            target_dict = exp1_data
        elif "Exp2" in folder:
            exp2_scenarios[clean_label] = sort_key
            target_dict = exp2_data
        elif "Exp3" in folder:
            exp3_scenarios[clean_label] = sort_key
            target_dict = exp3_data
        else:
            continue

        gap_file = os.path.join(dir_path, "demand_gaps.csv")
        if os.path.exists(gap_file):
            df_gap = pd.read_csv(gap_file)
            if not df_gap.empty and 'Missing_Staff' in df_gap.columns:
                df_gap['Scenario'] = clean_label
                df_gap['Sort_Key'] = sort_key
                target_dict['understaffing'].append(df_gap[['Scenario', 'Missing_Staff', 'Sort_Key']])

        viol_file = os.path.join(dir_path, "violation_registry.csv")
        if os.path.exists(viol_file):
            df_viol = pd.read_csv(viol_file)
            if not df_viol.empty:
                df_viol['Scenario'] = clean_label
                df_viol['Sort_Key'] = sort_key

                daily_viol = df_viol[df_viol['Constraint'] == 'Max Daily Hours'].copy()
                if not daily_viol.empty:
                    target_dict['max_daily'].append(daily_viol[['Scenario', 'Violation_Amount', 'Sort_Key']])

                shift_viol = df_viol[df_viol['Constraint'] == 'Max Shift Length'].copy()
                if not shift_viol.empty:
                    target_dict['max_shift'].append(shift_viol[['Scenario', 'Violation_Amount', 'Sort_Key']])

    for d in [exp1_data, exp2_data, exp3_data]:
        for key in d.keys():
            if d[key]:
                combined = pd.concat(d[key], ignore_index=True)
                d[key] = combined
            else:
                d[key] = pd.DataFrame()

    return exp1_data, exp2_data, exp3_data, exp1_scenarios, exp2_scenarios, exp3_scenarios

def fill_missing_scenarios(df, value_col, expected_scenarios):
    if not expected_scenarios:
        return df

    if df is None or df.empty:
        df = pd.DataFrame(columns=['Scenario', value_col, 'Sort_Key'])

    existing_scenarios = df['Scenario'].unique()
    missing_rows = []

    for scen, skey in expected_scenarios.items():
        if scen not in existing_scenarios:
            missing_rows.append({'Scenario': scen, value_col: 0.0, 'Sort_Key': skey})

    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    df.sort_values(by='Sort_Key', inplace=True)
    return df

def plot_boxplot(df, value_col, exp_name, metric_name, ylabel, output_dir):
    if df.empty:
        return

    plt.figure(figsize=(8, 5), dpi=300)
    sns.boxplot(x='Scenario', y=value_col, data=df, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 6},
                boxprops={'alpha': 0.8}, flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5})

    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(0, None)
    plt.xlabel('Scenarios', fontsize=16)
    plt.xticks(fontsize=16, rotation=30)

    y_max = df[value_col].max()
    plt.ylim(-0.5 if y_max > 0 else -0.1, y_max * 1.1 + 1)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{exp_name}_Boxplot_{metric_name.replace(' ', '_')}.png")
    plt.savefig(filename)
    plt.close('all')

def plot_stacked_distributions(df, value_col, exp_name, metric_name, xlabel, output_dir):
    if df.empty:
        return

    scenarios = df['Scenario'].unique()
    num_scenarios = len(scenarios)

    fig, axes = plt.subplots(nrows=num_scenarios, ncols=1, figsize=(8, 2.5 * num_scenarios), sharex=True, dpi=300)
    if num_scenarios == 1:
        axes = [axes]

    max_val = df[value_col].max()
    bins = np.arange(-0.5, max_val + 1.5, 1) if max_val < 30 else 20

    for ax, scenario in zip(axes, scenarios):
        subset = df[df['Scenario'] == scenario]

        is_zero_variance = (subset[value_col].nunique() <= 1)
        is_perfect = is_zero_variance and (subset[value_col].max() == 0.0)

        sns.histplot(data=subset, x=value_col, bins=bins, ax=ax,
                     color='steelblue' if not is_perfect else 'mediumseagreen',
                     edgecolor='black', alpha=0.6)

        if is_perfect:
            ax.text(0.5, 0.5, 'No Violation', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=16, color='darkgreen', fontweight='bold')
            ax.axvline(0, color='red', linestyle='dashed', linewidth=1.5, label='Median: 0.0')
        else:
            median_val = subset[value_col].median()
            ax.axvline(median_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.1f}')

        ax.legend(loc='upper right', fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.xlabel(xlabel, fontsize=16, fontweight='bold')
    plt.xticks(rotation=30)
    plt.ylim(0, None)
    plt.suptitle(f'Distribution of {metric_name}', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{exp_name}_DistPlot_{metric_name.replace(' ', '_')}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')

def plot_3d_distributions(df, value_col, exp_name, metric_name, xlabel, output_dir):
    if df.empty:
        return

    scenarios = df['Scenario'].unique()
    if 'demand' in exp_name.lower():
        scenarios_short = [s.split('=')[1].strip() for s in scenarios]
    else:
        scenarios_short = scenarios
    num_scenarios = len(scenarios)

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    max_val = df[value_col].max()
    min_val = df[value_col].min()
    if max_val - min_val < 30:
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
    else:
        bins = np.linspace(min_val, max_val, 20)

    colors = plt.cm.viridis(np.linspace(0, 1, num_scenarios))
    yticks = np.arange(num_scenarios)

    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        subset = df[df['Scenario'] == scenario]

        if 'Excess Hours' in xlabel:
            subset = subset[subset[value_col] <= 10]
        
        subset_data = subset[value_col].dropna()

        # Plot Histogram Bars
        hist, bin_edges = np.histogram(subset_data, bins=bins)
        x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        ax.bar(x_centers, hist, zs=i, zdir='y', width=bin_width * 0.8,
               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Overlay KDE Line
        if len(subset_data) >= 3:
            try:
                kde = gaussian_kde(subset_data)
                x_grid = np.linspace(subset_data.min()-1, subset_data.max()+1, 200)
                y_kde = kde(x_grid)

                # Scale KDE to match histogram frequency
                y_kde_scaled = y_kde * len(subset_data) * bin_width

                ax.plot(x_grid, y_kde_scaled, zs=i, zdir='y', color='black', linewidth=2.0)
            except (np.linalg.LinAlgError, ValueError):
                # Handle cases where KDE fails (e.g., no variance)
                pass

    ax.set_yticks(yticks)
    ax.set_yticklabels(scenarios_short)

    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    if 'demand' in exp_name.lower():
        ax.set_ylabel('Demand Scale', fontsize=16, fontweight='bold', labelpad=15)
    else:
        ax.set_ylabel('Scenario', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_zlabel('Frequency', fontsize=16, fontweight='bold', labelpad=10)

    ax.view_init(elev=25, azim=-50)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(linestyle=':', alpha=0.6)

    plt.tight_layout()

    filename = os.path.join(output_dir, f"{exp_name}_3D_Distplot_{metric_name.replace(' ', '_')}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
    print(f"[Plot] 3D Histogram with KDE saved to {filename}")

def main():
    BASE_DIR = "academic_experiments"
    OUTPUT_DIR = os.path.join(BASE_DIR, "academic_separated_plots/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Extracting Data and Rendering Separated Plots ===")
    exp1_data, exp2_data, exp3_data, exp1_scenarios, exp2_scenarios, exp3_scenarios = load_experiment_data(BASE_DIR)

    if exp1_data is None:
        return

    plot_configs = [
        ('understaffing', 'Missing_Staff', 'Demand Shortage', 'Missing Staff per Hour'),
        ('max_daily', 'Violation_Amount', 'Max Daily Overtime', 'Excess Hours'),
        ('max_shift', 'Violation_Amount', 'Max Shift Extension', 'Excess Hours')
    ]

    print("\n>>> Plotting Experiment 1: Big-M Sensitivity")
    for key, val_col, title, label in plot_configs:
        filled_df = fill_missing_scenarios(exp1_data[key], val_col, exp1_scenarios)
        plot_boxplot(filled_df, val_col, "M3 Sensitivity", title, label, OUTPUT_DIR)
        plot_stacked_distributions(filled_df, val_col, "M3 Sensitivity", title, label, OUTPUT_DIR)
        plot_3d_distributions(filled_df, val_col, "M3 Sensitivity", title, label, OUTPUT_DIR)

    print("\n>>> Plotting Experiment 2: White-Noise Perturbation")
    for key, val_col, title, label in plot_configs:
        filled_df = fill_missing_scenarios(exp2_data[key], val_col, exp2_scenarios)
        plot_boxplot(filled_df, val_col, "Perturbation", title, label, OUTPUT_DIR)
        plot_stacked_distributions(filled_df, val_col, "Perturbation", title, label, OUTPUT_DIR)
        plot_3d_distributions(filled_df, val_col, "Perturbation", title, label, OUTPUT_DIR)

    print("\n>>> Plotting Experiment 3: Demand Sensitivity")
    for key, val_col, title, label in plot_configs:
        filled_df = fill_missing_scenarios(exp3_data[key], val_col, exp3_scenarios)
        plot_boxplot(filled_df, val_col, "Demand Sensitivity", title, label, OUTPUT_DIR)
        plot_stacked_distributions(filled_df, val_col, "Demand Sensitivity", title, label, OUTPUT_DIR)
        plot_3d_distributions(filled_df, val_col, "Demand Sensitivity", title, label, OUTPUT_DIR)

    print(f"\n=== All plots saved successfully in: {OUTPUT_DIR} ===")

if __name__ == "__main__":
    main()
