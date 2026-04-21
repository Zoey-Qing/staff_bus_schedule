# import argparse
#
# def get_args():
#     parser = argparse.ArgumentParser(description="Staff Scheduling Optimization")
#
#     parser.add_argument('--solver_engine', type=str, default='highs', choices=['cplex', 'ortools', 'highs'],
#                         help='Select optimization engine')
#
#     # --- File Paths ---
#     parser.add_argument('--data_file', type=str, default='Berkeley_Mock_Data_25_Stores.xlsx', help='Path to input Excel file')
#     parser.add_argument('--schedule_file', type=str, default='schedule_result.csv', help='Path to save schedule CSV')
#     parser.add_argument('--report_file', type=str, default='verification_report.xlsx', help='Path to save verification report')
#     parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
#
#     # --- New: Finetuning & Bus Fleet ---
#     parser.add_argument('--baseline_schedule', type=str, default='schedule_result.csv', help='Path to baseline staff schedule for finetuning')
#     parser.add_argument('--bus_fleet_schedule', type=str, default='bus_fleet_schedule.csv', help='Path to bus fleet schedule')
#
#     # --- Constraints --
#     parser.add_argument('--min_shift_hours', type=int, default=6, help='Critical: Minimum duration of a single shift')
#     parser.add_argument('--max_shift_hours', type=int, default=13, help='Optional: Maximum duration of a single shift')
#
#     parser.add_argument('--standard_weekly_hours', type=int, default=48, help='Standard working hours before overtime')
#     parser.add_argument('--max_weekly_hours', type=int, default=78, help='Max working hours per week')
#
#     parser.add_argument('--min_daily_hours', type=int, default=8, help='Critical: Minimum daily working hours')
#     parser.add_argument('--max_daily_hours', type=int, default=13, help='Optional: Maximum daily working hours')
#
#     parser.add_argument('--days_off', type=int, default=1, help='Critical: Days off per week')
#     parser.add_argument('--min_break_hours', type=int, default=4, help='Critical: Minimum break between shifts')
#
#     # --- Bus Constraint ---
#     parser.add_argument('--enforce_bus', action='store_true', help='Enforce bus schedule for non-Saudi staff')
#
#     # --- Weights & Penalties ---
#     parser.add_argument('--w_understaffing', type=float, default=500, help='Penalty cost per unit of understaffing')
#     parser.add_argument('--w_manager_absence', type=float, default=100, help='Penalty cost because of manager absence')
#     parser.add_argument('--w_experienced', type=float, default=0.0, help='Reward weight for assigning experienced staff during peak hours')
#     parser.add_argument('--w_bus_trip', type=float, default=1.0, help='Penalty for each unique bus trip used by non-Saudi staff')
#     parser.add_argument('--w_change', type=float, default=100.0, help='Penalty for changing baseline assignment during finetuning')
#
#     # --- Solver Settings ---
#     parser.add_argument('--time_limit', type=int, default=300, help='Solver time limit in seconds')
#
#     return parser.parse_args()


import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Staff Scheduling Optimization")

    parser.add_argument('--solver_engine', type=str, default='cplex', choices=['cplex', 'ortools', 'highs'],
                        help='Select optimization engine')

    # --- File Paths ---
    parser.add_argument('--data_file', type=str, default='Berkeley_Mock_Data_25_Stores.xlsx',
                        help='Path to input Excel file')
    parser.add_argument('--schedule_file', type=str, default='schedule_result.csv', help='Path to save schedule CSV')
    parser.add_argument('--report_file', type=str, default='verification_report.xlsx',
                        help='Path to save verification report')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--baseline_schedule', type=str, default='schedule_result.csv',
                        help='Path to baseline staff schedule for finetuning')
    parser.add_argument('--bus_fleet_schedule', type=str, default='bus_fleet_schedule.csv',
                        help='Path to bus fleet schedule')

    # --- Global Standard Constraints ---
    parser.add_argument('--standard_weekly_hours', type=int, default=48, help='Standard working hours before overtime')
    parser.add_argument('--max_weekly_hours', type=int, default=78, help='Max working hours per week')
    parser.add_argument('--days_off', type=int, default=1, help='Critical: Days off per week')
    parser.add_argument('--min_break_hours', type=int, default=4,
                        help='Critical: Minimum break between shifts for all staff')
    # --- Global Shift Generation Limits (Used by DataLoader to generate the shift pool) ---
    parser.add_argument('--min_shift_hours', type=int, default=6, help='Absolute min shift length for shift generation')
    parser.add_argument('--max_shift_hours', type=int, default=16,
                        help='Absolute max shift length for shift generation (fallback is 16)')

    # --- CRITICAL CONSTRAINTS (HARD LIMITS) ---
    # Daily Minimums
    parser.add_argument('--min_daily_saudi_not_mgr', type=int, default=6,
                        help='Critical: Min daily hours for Saudi AND NOT Manager')
    parser.add_argument('--min_daily_saudi_mgr', type=int, default=8,
                        help='Critical: Min daily hours for Saudi AND Manager')
    parser.add_argument('--min_daily_expat', type=int, default=9, help='Critical: Min daily hours for NOT Saudi')

    # Shift Length Minimums
    parser.add_argument('--min_shift_saudi_not_mgr', type=int, default=6,
                        help='Critical: Min shift length for Saudi AND NOT Manager')
    parser.add_argument('--min_shift_saudi_mgr', type=int, default=8,
                        help='Critical: Min shift length for Saudi AND Manager')
    parser.add_argument('--min_shift_expat', type=int, default=9, help='Critical: Min shift length for NOT Saudi')


    # --- OPTIONAL CONSTRAINTS (SOFT LIMITS) ---
    # Daily Maximums
    parser.add_argument('--max_daily_saudi_not_mgr_soft', type=int, default=6,
                        help='Optional: Max daily hours for Saudi AND NOT Manager')
    parser.add_argument('--max_daily_saudi_mgr_soft', type=int, default=8,
                        help='Optional: Max daily hours for Saudi AND Manager')
    parser.add_argument('--max_daily_expat_mgr_soft', type=int, default=9,
                        help='Optional: Max daily hours for Manager AND NOT Saudi')
    parser.add_argument('--max_daily_expat_not_mgr_soft', type=int, default=12,
                        help='Optional: Max daily hours for NOT (Saudi OR Manager)')

    # Shift Length Maximums
    parser.add_argument('--max_shift_saudi_not_mgr_soft', type=int, default=6,
                        help='Optional: Max shift length for Saudi AND NOT Manager')
    parser.add_argument('--max_shift_expat_mgr_soft', type=int, default=9,
                        help='Optional: Max shift length for Manager AND NOT Saudi')
    parser.add_argument('--max_shift_expat_not_mgr_soft', type=int, default=12,
                        help='Optional: Max shift length for NOT (Saudi OR Manager)')

    # Onsite Minimums (Soft)
    parser.add_argument('--min_onsite_all_soft', type=int, default=2, help='Optional: Min number onsite for ALL STAFF')
    parser.add_argument('--min_onsite_open_mgr_soft', type=int, default=1,
                        help='Optional: Min Manager onsite at opening (first hour)')
    parser.add_argument('--min_onsite_close_mgr_soft', type=int, default=1,
                        help='Optional: Min Manager onsite at closing (last hour)')
    parser.add_argument('--min_onsite_mgr_soft', type=int, default=1, help='Optional: Min Manager onsite')
    parser.add_argument('--min_onsite_not_mgr_soft', type=int, default=2, help='Optional: Min NOT Manager onsite')

    # --- RESTORED: Hard Fallback Constraints ---
    parser.add_argument('--min_onsite_all_hard', type=int, default=1,
                        help='Critical: Absolute min number onsite for ALL STAFF if 2 is not possible')
    parser.add_argument('--max_daily_expat_not_mgr_hard', type=int, default=16,
                        help='Critical: Max daily hours if 12 is not possible')
    parser.add_argument('--max_shift_expat_not_mgr_hard', type=int, default=16,
                        help='Critical: Max shift length if 12 is not possible')

    # --- NEW: Soft Constraint for NOT New Staff ---
    parser.add_argument('--min_onsite_not_new_soft', type=int, default=1,
                        help='Most Important: Min NOT New staff onsite')

    # --- Bus Constraint ---
    parser.add_argument('--enforce_bus', action='store_true', help='Enforce bus schedule for non-Saudi staff')

    # --- BIG-M PENALTY WEIGHTS (HIERARCHY) ---
    parser.add_argument('--w_demand_shortage', type=float, default=100000.0, help='Penalty for hard demand shortage')
    parser.add_argument('--w_priority_1', type=float, default=10000.0,
                        help='Weight for Most Important optional constraints')
    parser.add_argument('--w_priority_2', type=float, default=1000.0,
                        help='Weight for High priority optional constraints')
    parser.add_argument('--w_priority_3', type=float, default=100.0,
                        help='Weight for Medium priority optional constraints')
    parser.add_argument('--w_priority_4', type=float, default=10.0,
                        help='Weight for Least Important optional constraints')

    # --- General Weights ---
    parser.add_argument('--w_understaffing', type=float, default=500,
                        help='Legacy penalty cost per unit of understaffing')
    parser.add_argument('--w_manager_absence', type=float, default=100, help='Penalty cost because of manager absence')
    parser.add_argument('--w_experienced', type=float, default=0.0,
                        help='Reward weight for assigning experienced staff')
    parser.add_argument('--w_bus_trip', type=float, default=1.0,
                        help='Penalty for each unique bus trip used by non-Saudi staff')
    parser.add_argument('--w_change', type=float, default=100.0, help='Penalty for changing baseline assignment')

    # --- Solver Settings ---
    parser.add_argument('--time_limit', type=int, default=1200, help='Solver time limit in seconds')
    parser.add_argument('--mip_gap', type=float, default=0.01, help='MIP gap for optimization')

    return parser.parse_args()