import pandas as pd
import numpy as np


def run_evaluation(args, df_schedule, data_loader):
    print(f"\n>>> [Evaluation] Generating report: {args.report_file}...")
    non_saudi_set = set(data_loader.non_saudi_staff_ids)
    manager_set = set(data_loader.manager_ids)

    # === Bus Trip Consolidation Stats by Cluster ===
    df_ns = df_schedule[df_schedule['Staff'].isin(non_saudi_set)].copy()
    if not df_ns.empty:
        df_ns['Cluster'] = df_ns['Staff'].map(data_loader.staff_cluster_map)
        df_ns['Start_Time'] = df_ns['Start'] % 168
        df_ns['End_Time'] = (df_ns['Start'] + df_ns['Duration']) % 168

        print("\n[Bus Trip Consolidation Stats by Cluster]")
        total_cluster_trips = 0
        for cluster, group in df_ns.groupby('Cluster'):
            c_starts = set(group['Start_Time'].unique())
            c_ends = set(group['End_Time'].unique())
            c_total = len(c_starts) + len(c_ends)
            total_cluster_trips += c_total
            print(f"  Cluster '{cluster}': {c_total} Trips (Starts: {len(c_starts)}, Ends: {len(c_ends)})")
        print(f"  Total Unique Bus Trips Accrued Penalty: {total_cluster_trips}")

    # === Demand Check ===
    df_verify = pd.DataFrame({'hour_of_week': range(168)})
    df_verify['Target_Demand'] = df_verify['hour_of_week'].map(data_loader.demand).fillna(0)

    supply_total = np.zeros(168)
    for _, row in df_schedule.iterrows():
        s, d = int(row['Start']), int(row['Duration'])
        for t in range(s, s + d):
            if 0 <= t < 168: supply_total[t] += 1

    df_verify['Actual_Supply'] = supply_total
    df_verify['Gap'] = df_verify['Actual_Supply'] - df_verify['Target_Demand']
    df_verify['Status'] = df_verify['Gap'].apply(lambda x: 'OK' if x == 0 else ('Over' if x > 0 else 'Under'))

    inv_day_map = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
    df_verify['Day'] = (df_verify['hour_of_week'] // 24).map(inv_day_map)
    df_verify['Rel_Hour'] = df_verify['hour_of_week'] % 24
    df_verify['Clock_Hour'] = (df_verify['Rel_Hour'] + 5) % 24

    cols = ['hour_of_week', 'Day', 'Rel_Hour', 'Clock_Hour', 'Target_Demand', 'Actual_Supply', 'Gap', 'Status']
    df_verify = df_verify[cols]

    # === Staff Rules Check ===
    staff_stats = []
    grouped = df_schedule.groupby('Staff')

    for staff_id in data_loader.staff_ids:
        if staff_id not in grouped.groups:
            staff_stats.append({'Staff': staff_id, 'Status': 'Idle', 'Total_Hours': 0})
            continue

        shifts = grouped.get_group(staff_id).sort_values('Start')
        total_hours = shifts['Duration'].sum()
        min_shift_actual = shifts['Duration'].min()
        max_shift_actual = shifts['Duration'].max()

        daily_hours = shifts.groupby('Day')['Duration'].sum()
        min_daily_actual = daily_hours.min()
        max_daily_actual = daily_hours.max()

        worked_days = len(daily_hours)
        days_off = 7 - worked_days

        breaks = []
        if len(shifts) > 1:
            starts = shifts['Start'].values
            ends = starts + shifts['Duration'].values
            for k in range(len(shifts) - 1): breaks.append(starts[k + 1] - ends[k])
        min_break = min(breaks) if breaks else 999

        # 1. 识别员工身份
        is_saudi = staff_id not in non_saudi_set
        is_mgr = staff_id in manager_set

        # 2. 匹配其对应的精确规则下限/上限
        if is_saudi and not is_mgr:
            req_min_shift = args.min_shift_saudi_not_mgr
            req_min_daily = args.min_daily_saudi_not_mgr
            req_max_shift_soft = args.max_shift_saudi_not_mgr_soft
            req_max_daily_soft = args.max_daily_saudi_not_mgr_soft
        elif is_saudi and is_mgr:
            req_min_shift = args.min_shift_saudi_mgr
            req_min_daily = args.min_daily_saudi_mgr
            req_max_shift_soft = 999  # 无特定上限
            req_max_daily_soft = args.max_daily_saudi_mgr_soft
        elif not is_saudi and is_mgr:
            req_min_shift = args.min_shift_expat
            req_min_daily = args.min_daily_expat
            req_max_shift_soft = args.max_shift_expat_mgr_soft
            req_max_daily_soft = args.max_daily_expat_mgr_soft
        else:  # not saudi, not mgr
            req_min_shift = args.min_shift_expat
            req_min_daily = args.min_daily_expat
            req_max_shift_soft = args.max_shift_expat_not_mgr_soft
            req_max_daily_soft = args.max_daily_expat_not_mgr_soft

        issues = []
        soft_violations = []

        # 3. 检查绝对硬约束 (Critical)
        if min_shift_actual < req_min_shift:
            issues.append(f"Shift<{req_min_shift}")
        if min_daily_actual < req_min_daily:
            issues.append(f"Daily<{req_min_daily}")
        if days_off < args.days_off:  # 注意：改成 <，允许休息更多天
            issues.append(f"DaysOff<1({days_off})")
        if min_break < args.min_break_hours:
            issues.append(f"Break<{args.min_break_hours}")
        if total_hours > args.max_weekly_hours:
            issues.append("WeeklyMaxOver")

        # 4. 检查软约束 (Optional/Violate Registry)
        if max_daily_actual > req_max_daily_soft:
            soft_violations.append(f"SoftDaily>{req_max_daily_soft}")
        if max_shift_actual > req_max_shift_soft:
            soft_violations.append(f"SoftShift>{req_max_shift_soft}")

        if args.enforce_bus and not is_saudi:
            invalid_shifts = shifts[shifts['Bus_Compatible'] == False]
            if not invalid_shifts.empty:
                issues.append("BusViolation")

        # 判断最终状态
        if issues:
            status = 'FAIL (Hard)'
        elif soft_violations:
            status = 'PASS (with Soft Violations)'
        else:
            status = 'PASS (Perfect)'

        all_notes = "; ".join(issues + soft_violations)

        staff_stats.append({
            'Staff': staff_id,
            'Status': status,
            'Issues/Violations': all_notes,
            'Total_Hours': total_hours,
            'Days_Off': days_off,
            'Min_Break': min_break
        })

    df_staff = pd.DataFrame(staff_stats)

    try:
        with pd.ExcelWriter(args.report_file, engine='openpyxl') as writer:
            df_verify.to_excel(writer, sheet_name='Demand_Check', index=False)
            df_staff.to_excel(writer, sheet_name='Staff_Check', index=False)
        print(f">>> Report saved: {args.report_file}")
    except Exception as e:
        print(f"Error saving report: {e}")


if __name__ == "__main__":
    import os
    import sys
    import glob
    from args import get_args
    from data_loader import DataLoader
    from visualization import generate_visualizations

    args = get_args()

    results_dirs = glob.glob(os.path.join("results", "*"))
    if not results_dirs:
        print("Error: No 'results' directory found. Please run main.py first.")
        sys.exit(1)

    latest_dir = max(results_dirs, key=os.path.getctime)
    print(f"Detect the latest run directory: {latest_dir}")

    original_schedule_file = os.path.basename(getattr(args, 'schedule_file', 'schedule_result.csv'))
    original_report_file = os.path.basename(getattr(args, 'report_file', 'verification_report.xlsx'))

    args.schedule_file = os.path.join(latest_dir, original_schedule_file)
    args.report_file = os.path.join(latest_dir, original_report_file)
    args.plot_dir = os.path.join(latest_dir, "plots")

    if not os.path.exists(args.schedule_file):
        print(f"Error: Schedule file not found at {args.schedule_file}")
        sys.exit(1)

    df_schedule = pd.read_csv(args.schedule_file)
    global_loader = DataLoader(args, target_cluster=None)
    global_loader.load_and_preprocess()

    run_evaluation(args, df_schedule, global_loader)

    print(f"Generating visualizations into: {args.plot_dir}")
    os.makedirs(args.plot_dir, exist_ok=True)
    generate_visualizations(args)

    print("\n>>> Evaluation Complete!")
