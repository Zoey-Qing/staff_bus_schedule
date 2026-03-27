import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Staff Scheduling Optimization")

    parser.add_argument('--solver_engine', type=str, default='cplex', choices=['cplex', 'highs'],
                        help='Select optimization engine')

    # --- File Paths ---
    parser.add_argument('--data_file', type=str, default='../Berkeley_Mock_Data_25_Stores.xlsx', help='Path to input Excel file')
    parser.add_argument('--schedule_file', type=str, default='schedule_result.csv', help='Path to save schedule CSV')
    parser.add_argument('--report_file', type=str, default='verification_report.xlsx', help='Path to save verification report')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')

    # --- New: Finetuning & Bus Fleet ---
    parser.add_argument('--baseline_schedule', type=str, default='schedule_result.csv', help='Path to baseline staff schedule for finetuning')
    parser.add_argument('--bus_fleet_schedule', type=str, default='bus_fleet_schedule.csv', help='Path to bus fleet schedule')

    # --- Constraints --
    parser.add_argument('--min_shift_hours', type=int, default=6, help='Critical: Minimum duration of a single shift')
    parser.add_argument('--max_shift_hours', type=int, default=13, help='Optional: Maximum duration of a single shift')

    parser.add_argument('--standard_weekly_hours', type=int, default=48, help='Standard working hours before overtime')
    parser.add_argument('--max_weekly_hours', type=int, default=78, help='Max working hours per week')

    parser.add_argument('--min_daily_hours', type=int, default=8, help='Critical: Minimum daily working hours')
    parser.add_argument('--max_daily_hours', type=int, default=13, help='Optional: Maximum daily working hours')

    parser.add_argument('--days_off', type=int, default=1, help='Critical: Days off per week')
    parser.add_argument('--min_break_hours', type=int, default=4, help='Critical: Minimum break between shifts')

    # --- Bus Constraint ---
    parser.add_argument('--enforce_bus', action='store_true', help='Enforce bus schedule for non-Saudi staff')

    # --- Weights & Penalties ---
    parser.add_argument('--w_understaffing', type=float, default=500, help='Penalty cost per unit of understaffing')
    parser.add_argument('--w_manager_absence', type=float, default=100, help='Penalty cost because of manager absence')
    parser.add_argument('--w_experienced', type=float, default=0.0, help='Reward weight for assigning experienced staff during peak hours')
    parser.add_argument('--w_bus_trip', type=float, default=1.0, help='Penalty for each unique bus trip used by non-Saudi staff')
    parser.add_argument('--w_change', type=float, default=100.0, help='Penalty for changing baseline assignment during finetuning')

    # --- Solver Settings ---
    parser.add_argument('--time_limit', type=int, default=600, help='Solver time limit in seconds')

    return parser.parse_args()