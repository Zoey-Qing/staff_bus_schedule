import pandas as pd
import numpy as np


def run_evaluation(args, df_schedule, data_loader):
    print(f"\n>>> [Evaluation] Generating report: {args.report_file}...")
    non_saudi_set = set(data_loader.non_saudi_staff_ids)

    # === 新增：按 Cluster 统计 Bus Trip 次数 ===
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
            # 每发一次班车(不管接/送)均算作1次Trip（与目标函数公式完全对应）
            c_total = len(c_starts) + len(c_ends)
            total_cluster_trips += c_total
            print(f"  Cluster '{cluster}': {c_total} Trips (Starts: {len(c_starts)}, Ends: {len(c_ends)})")
        print(f"  Total Unique Bus Trips Accrued Penalty: {total_cluster_trips}")

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

    staff_stats = []
    grouped = df_schedule.groupby('Staff')

    for staff_id in data_loader.staff_ids:
        if staff_id not in grouped.groups:
            staff_stats.append({'Staff': staff_id, 'Status': 'Idle', 'Total_Hours': 0})
            continue

        shifts = grouped.get_group(staff_id).sort_values('Start')
        total_hours = shifts['Duration'].sum()
        min_shift = shifts['Duration'].min()

        daily_hours = shifts.groupby('Day')['Duration'].sum()
        worked_days = len(daily_hours)
        days_off = 7 - worked_days

        breaks = []
        if len(shifts) > 1:
            starts = shifts['Start'].values
            ends = starts + shifts['Duration'].values
            for k in range(len(shifts) - 1): breaks.append(starts[k + 1] - ends[k])
        min_break = min(breaks) if breaks else 999

        issues = []
        if min_shift < args.min_shift_hours: issues.append("Shift<6")
        if daily_hours.min() < args.min_daily_hours: issues.append("Daily<8")
        if daily_hours.max() > args.max_daily_hours: issues.append("DailyMax")
        if days_off != args.days_off: issues.append(f"DaysOff!={args.days_off}({days_off})")
        if min_break < args.min_break_hours: issues.append("Break<4")

        if args.enforce_bus and (staff_id in non_saudi_set):
            invalid_shifts = shifts[shifts['Bus_Compatible'] == False]
            if not invalid_shifts.empty:
                issues.append("BusViolation")

        status = 'FAIL' if issues else 'PASS'
        staff_stats.append({
            'Staff': staff_id, 'Status': status, 'Issues': "; ".join(issues),
            'Total_Hours': total_hours, 'Days_Off': days_off, 'Min_Break': min_break
        })

    df_staff = pd.DataFrame(staff_stats)

    try:
        with pd.ExcelWriter(args.report_file, engine='openpyxl') as writer:
            df_verify.to_excel(writer, sheet_name='Demand_Check', index=False)
            df_staff.to_excel(writer, sheet_name='Staff_Check', index=False)
        print(f">>> Report saved: {args.report_file}")
    except Exception as e:
        print(e)