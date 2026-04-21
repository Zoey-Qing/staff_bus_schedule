import pandas as pd
import pulp
import os


class StaffSchedulingSolver:
    def __init__(self, args, data, baseline_assignments=None):
        self.args = args
        self.data = data
        self.baseline_assignments = baseline_assignments

        mode_name = "Finetune" if baseline_assignments else "Normal"
        self.mdl = pulp.LpProblem(f"Staff_Scheduling_ShiftBased_{mode_name}", pulp.LpMinimize)

        self.x = {}
        self.o_daily = {}
        self.w = {}
        self.b_start = {}
        self.b_end = {}

        self.sl = {}
        self.v_min_onsite_not_new = {}
        self.v_max_daily = {}
        self.v_max_shift = {}
        self.v_min_onsite_all = {}
        self.v_min_onsite_mgr = {}
        self.sl_mgr_first = {}
        self.sl_mgr_last = {}
        self.v_min_onsite_not_mgr = {}
        self.v_min_work_days = {}
        self.v_idle = {}

    def build_model(self):
        staff_ids = self.data.staff_ids
        shifts = self.data.shifts
        time_slots = self.data.time_slots
        days = self.data.days
        clusters = self.data.cluster_ids

        expat_set = set(self.data.non_saudi_staff_ids)
        saudi_set = set(staff_ids) - expat_set
        manager_set = set(self.data.manager_ids)
        non_manager_set = set(staff_ids) - manager_set
        not_new_set = set(self.data.non_new_staff_ids)

        saudi_not_mgr = saudi_set & non_manager_set
        saudi_mgr = saudi_set & manager_set
        expat_mgr = expat_set & manager_set
        expat_not_mgr = expat_set & non_manager_set

        managers_by_branch = {}
        staff_by_branch = {}
        for i in staff_ids:
            bid = self.data.staff_branch_map.get(i)
            if bid:
                if bid not in staff_by_branch: staff_by_branch[bid] = []
                staff_by_branch[bid].append(i)
                if i in manager_set:
                    if bid not in managers_by_branch: managers_by_branch[bid] = []
                    managers_by_branch[bid].append(i)

        shift_ids = [s['id'] for s in shifts]

        # --- Variables Definition (PuLP) ---
        for i in staff_ids:
            safe_i = str(i).replace('-', '_')
            self.v_idle[i] = pulp.LpVariable(f"v_idle_{safe_i}", lowBound=0, cat='Continuous')
            self.v_min_work_days[i] = pulp.LpVariable(f"v_min_work_days_{safe_i}", lowBound=0, cat='Continuous')

            for s in shifts:
                sid = s['id']
                self.x[i, sid] = pulp.LpVariable(f"x_{safe_i}_{sid}", cat='Binary')
                self.v_max_shift[i, sid] = pulp.LpVariable(f"v_max_shift_{safe_i}_{sid}", lowBound=0, cat='Continuous')

            for d in days:
                self.w[i, d] = pulp.LpVariable(f"w_{safe_i}_{d}", cat='Binary')
                self.o_daily[i, d] = pulp.LpVariable(f"o_daily_{safe_i}_{d}", lowBound=0, cat='Continuous')
                self.v_max_daily[i, d] = pulp.LpVariable(f"v_max_daily_{safe_i}_{d}", lowBound=0, cat='Continuous')

        for c in clusters:
            safe_c = str(c).replace('-', '_')
            for t in time_slots:
                self.b_start[c, t] = pulp.LpVariable(f"bs_{safe_c}_{t}", cat='Binary')
                self.b_end[c, t] = pulp.LpVariable(f"be_{safe_c}_{t}", cat='Binary')

        for t in time_slots:
            self.sl[t] = pulp.LpVariable(f"sl_{t}", lowBound=0, cat='Continuous')
            self.v_min_onsite_all[t] = pulp.LpVariable(f"v_onsite_all_{t}", lowBound=0, cat='Continuous')
            self.v_min_onsite_mgr[t] = pulp.LpVariable(f"v_onsite_mgr_{t}", lowBound=0, cat='Continuous')
            self.v_min_onsite_not_mgr[t] = pulp.LpVariable(f"v_onsite_not_mgr_{t}", lowBound=0, cat='Continuous')
            self.v_min_onsite_not_new[t] = pulp.LpVariable(f"v_onsite_not_new_{t}", lowBound=0, cat='Continuous')

        branch_days = [(b, d) for b in self.data.branch_operation_limits for d in days]
        for bd in branch_days:
            safe_b = str(bd[0]).replace('-', '_')
            self.sl_mgr_first[bd] = pulp.LpVariable(f"sl_mgr_first_{safe_b}_{bd[1]}", lowBound=0, cat='Continuous')
            self.sl_mgr_last[bd] = pulp.LpVariable(f"sl_mgr_last_{safe_b}_{bd[1]}", lowBound=0, cat='Continuous')

        # --- Base Objective Terms ---
        term_wage = pulp.lpSum(
            self.data.wage_reg[i] * s['duration'] * self.x[i, s['id']] for i in staff_ids for s in shifts)
        term_ot = pulp.lpSum(self.data.wage_ot[i] * self.o_daily[i, d] for i in staff_ids for d in days)

        if getattr(self.args, 'enforce_bus', False):
            term_bus_trip = pulp.lpSum(
                getattr(self.args, 'w_bus_trip', 1.0) * (self.b_start[c, t] + self.b_end[c, t]) for c in clusters for t
                in time_slots)
        else:
            term_bus_trip = 0

        term_deviation = 0
        if self.baseline_assignments is not None:
            w_change = getattr(self.args, 'w_change', 100.0)
            for i in staff_ids:
                for s in shifts:
                    if (i, s['id']) in self.baseline_assignments:
                        term_deviation += w_change * (1 - self.x[i, s['id']])
                    else:
                        term_deviation += w_change * self.x[i, s['id']]

        # --- Big-M Penalties ---
        M1 = 1000
        M2 = 1000
        M3 = 100
        M4 = 1

        penalty_m1 = M1 * (
                pulp.lpSum(self.sl[t] for t in time_slots) +
                pulp.lpSum(self.v_idle[i] for i in staff_ids)
        )

        penalty_m2 = M2 * (
                pulp.lpSum(self.v_max_daily[i, d] for i in saudi_not_mgr for d in days) +
                pulp.lpSum(self.v_min_onsite_not_new[t] for t in time_slots)
        )

        penalty_m3 = M3 * (
                pulp.lpSum(self.v_min_work_days[i] for i in staff_ids) +
                pulp.lpSum(self.v_max_daily[i, d] for i in saudi_mgr for d in days) +
                pulp.lpSum(self.v_max_shift[i, sid] for i in saudi_not_mgr for sid in shift_ids) +
                pulp.lpSum(self.v_max_daily[i, d] for i in expat_mgr for d in days) +
                pulp.lpSum(self.v_max_shift[i, sid] for i in expat_mgr for sid in shift_ids) +
                pulp.lpSum(self.v_max_shift[i, sid] for i in expat_not_mgr for sid in shift_ids) +
                pulp.lpSum(self.v_max_daily[i, d] for i in expat_not_mgr for d in days) +
                pulp.lpSum(self.v_min_onsite_all[t] for t in time_slots) +
                pulp.lpSum(self.v_min_onsite_mgr[t] for t in time_slots) +
                pulp.lpSum(self.sl_mgr_first[bd] + self.sl_mgr_last[bd] for bd in branch_days)
        )

        penalty_m4 = M4 * pulp.lpSum(self.v_min_onsite_not_mgr[t] for t in time_slots)

        term_penalties = penalty_m1 + penalty_m2 + penalty_m3 + penalty_m4\

        lambda_l1 = 0.1
        term_l1_reg = pulp.lpSum(lambda_l1 * self.x[i, s['id']] for i in staff_ids for s in shifts)

        self.mdl += (term_wage + term_ot + term_bus_trip + term_deviation + term_penalties + term_l1_reg), "Total_Objective"

        # --- Pre-processing ---
        shifts_at_work_t = {t: [] for t in time_slots}
        shifts_at_extended_t = {t: [] for t in time_slots}
        shifts_at_day = {d: [] for d in days}
        shifts_starting_at = {t: [] for t in time_slots}
        shifts_ending_at = {t: [] for t in time_slots}

        min_break = getattr(self.args, 'min_break_hours', 4)
        for s in shifts:
            sid = s['id']
            for t in s['covers']: shifts_at_work_t[t].append(sid)
            extended_end = min(168, s['end'] + min_break)
            for t in range(s['start'], extended_end): shifts_at_extended_t[t].append(sid)
            day_idx = s['start'] // 24
            if 0 <= day_idx < 7: shifts_at_day[day_idx].append(sid)
            shifts_starting_at[s['start'] % 168].append(sid)
            shifts_ending_at[s['end'] % 168].append(sid)

        # --- Bus Consolidation Constraints ---
        if getattr(self.args, 'enforce_bus', False):
            for c in clusters:
                staff_in_cluster = [i for i in expat_set if self.data.staff_cluster_map.get(i) == c]
                M_val = len(staff_in_cluster)
                if M_val > 0:
                    for t in time_slots:
                        if shifts_starting_at[t]:
                            self.mdl += (pulp.lpSum(
                                self.x[i, sid] for i in staff_in_cluster for sid in shifts_starting_at[t]) <= M_val *
                                         self.b_start[c, t])
                        else:
                            self.mdl += (self.b_start[c, t] == 0)
                        if shifts_ending_at[t]:
                            self.mdl += (pulp.lpSum(
                                self.x[i, sid] for i in staff_in_cluster for sid in shifts_ending_at[t]) <= M_val *
                                         self.b_end[c, t])
                        else:
                            self.mdl += (self.b_end[c, t] == 0)
                else:
                    for t in time_slots:
                        self.mdl += (self.b_start[c, t] == 0)
                        self.mdl += (self.b_end[c, t] == 0)

            for s in shifts:
                for i in expat_set:
                    branch = self.data.staff_branch_map.get(i)
                    if branch:
                        valid_times = self.data.branch_valid_bus_times.get(branch, set())
                        if (s['start'] % 168) not in valid_times or (s['end'] % 168) not in valid_times:
                            self.mdl += (self.x[i, s['id']] == 0)

        # --- Shift Constraints ---
        for i in staff_ids:
            for s in shifts:
                sid = s['id']
                dur = s['duration']

                if i in saudi_not_mgr and dur < getattr(self.args, 'min_shift_saudi_not_mgr', 6):
                    self.mdl += (self.x[i, sid] == 0)
                elif i in saudi_mgr and dur < getattr(self.args, 'min_shift_saudi_mgr', 8):
                    self.mdl += (self.x[i, sid] == 0)
                elif i in expat_set and dur < getattr(self.args, 'min_shift_expat', 9):
                    self.mdl += (self.x[i, sid] == 0)

                if i in expat_not_mgr and dur > getattr(self.args, 'max_shift_expat_not_mgr_hard', 16):
                    self.mdl += (self.x[i, sid] == 0)

                if i in saudi_not_mgr:
                    self.mdl += (dur * self.x[i, sid] - self.v_max_shift[i, sid] <= getattr(self.args,
                                                                                            'max_shift_saudi_not_mgr_soft',
                                                                                            6))
                elif i in expat_mgr:
                    self.mdl += (dur * self.x[i, sid] - self.v_max_shift[i, sid] <= getattr(self.args,
                                                                                            'max_shift_expat_mgr_soft',
                                                                                            9))
                elif i in expat_not_mgr:
                    self.mdl += (dur * self.x[i, sid] - self.v_max_shift[i, sid] <= getattr(self.args,
                                                                                            'max_shift_expat_not_mgr_soft',
                                                                                            12))

        # --- Daily Constraints ---
        for i in staff_ids:
            for d in days:
                relevant = shifts_at_day[d]
                if not relevant:
                    self.mdl += (self.o_daily[i, d] == 0)
                    self.mdl += (self.w[i, d] == 0)
                    continue

                daily_hours = pulp.lpSum(self.x[i, sid] * self.data.shift_map[sid]['duration'] for sid in relevant)
                self.mdl += (daily_hours <= 24 * self.w[i, d])
                self.mdl += (daily_hours >= 0.5 * self.w[i, d])

                if i in saudi_not_mgr:
                    standard_min = getattr(self.args, 'min_daily_saudi_not_mgr', 6)
                    self.mdl += (self.o_daily[i, d] >= daily_hours - standard_min)
                elif i in expat_not_mgr:
                    standard_min = getattr(self.args, 'min_daily_expat', 9)
                    self.mdl += (self.o_daily[i, d] >= daily_hours - standard_min)
                else:
                    self.mdl += (self.o_daily[i, d] == 0)

                if i in saudi_not_mgr:
                    self.mdl += (daily_hours >= getattr(self.args, 'min_daily_saudi_not_mgr', 6) * self.w[i, d])
                elif i in saudi_mgr:
                    self.mdl += (daily_hours >= getattr(self.args, 'min_daily_saudi_mgr', 8) * self.w[i, d])
                elif i in expat_set:
                    self.mdl += (daily_hours >= getattr(self.args, 'min_daily_expat', 9) * self.w[i, d])

                if i in expat_not_mgr:
                    self.mdl += (daily_hours <= getattr(self.args, 'max_daily_expat_not_mgr_hard', 16) * self.w[i, d])

                if i in saudi_not_mgr:
                    self.mdl += (daily_hours - self.v_max_daily[i, d] <= getattr(self.args,
                                                                                 'max_daily_saudi_not_mgr_soft', 6) *
                                 self.w[i, d])
                elif i in saudi_mgr:
                    self.mdl += (daily_hours - self.v_max_daily[i, d] <= getattr(self.args, 'max_daily_saudi_mgr_soft',
                                                                                 8) * self.w[i, d])
                elif i in expat_mgr:
                    self.mdl += (daily_hours - self.v_max_daily[i, d] <= getattr(self.args, 'max_daily_expat_mgr_soft',
                                                                                 9) * self.w[i, d])
                elif i in expat_not_mgr:
                    self.mdl += (daily_hours - self.v_max_daily[i, d] <= getattr(self.args,
                                                                                 'max_daily_expat_not_mgr_soft', 12) *
                                 self.w[i, d])

            self.mdl += (pulp.lpSum(self.w[i, d] for d in days) + self.v_idle[i] >= 1)

            target_work_days = 7 - getattr(self.args, 'days_off', 1)
            self.mdl += (pulp.lpSum(self.w[i, d] for d in days) + self.v_min_work_days[i] >= target_work_days)
            self.mdl += (pulp.lpSum(self.w[i, d] for d in days) <= target_work_days)

            total_weekly = pulp.lpSum(s['duration'] * self.x[i, s['id']] for s in shifts)
            self.mdl += (total_weekly <= getattr(self.args, 'max_weekly_hours', 78))

        # --- No Overlap Constraints ---
        for i in staff_ids:
            for t in time_slots:
                if shifts_at_extended_t[t]:
                    self.mdl += (pulp.lpSum(self.x[i, sid] for sid in shifts_at_extended_t[t]) <= 1)

        # --- Coverage & Onsite Constraints ---
        for t in time_slots:
            supply_all = pulp.lpSum(self.x[i, sid] for i in staff_ids for sid in shifts_at_work_t[t])
            supply_mgr = pulp.lpSum(self.x[i, sid] for i in manager_set for sid in shifts_at_work_t[t])
            supply_non_mgr = pulp.lpSum(self.x[i, sid] for i in non_manager_set for sid in shifts_at_work_t[t])
            supply_not_new = pulp.lpSum(self.x[i, sid] for i in not_new_set for sid in shifts_at_work_t[t])

            self.mdl += (supply_all + self.sl[t] >= self.data.demand[t])

            if self.data.demand[t] > 0:
                if shifts_at_work_t[t]:
                    self.mdl += (supply_all >= getattr(self.args, 'min_onsite_all_hard', 1))

                self.mdl += (supply_not_new + self.v_min_onsite_not_new[t] >= getattr(self.args,
                                                                                      'min_onsite_not_new_soft', 1))
                self.mdl += (supply_all + self.v_min_onsite_all[t] >= getattr(self.args, 'min_onsite_all_soft', 2))
                self.mdl += (supply_mgr + self.v_min_onsite_mgr[t] >= getattr(self.args, 'min_onsite_mgr_soft', 1))
                self.mdl += (supply_non_mgr + self.v_min_onsite_not_mgr[t] >= getattr(self.args,
                                                                                      'min_onsite_not_mgr_soft', 2))

        # --- Manager Opening/Closing Constraints ---
        for branch_id, limits in self.data.branch_operation_limits.items():
            branch_managers = managers_by_branch.get(branch_id, [])
            if not branch_managers:
                for d in days:
                    self.mdl += (self.sl_mgr_first[branch_id, d] >= getattr(self.args, 'min_onsite_open_mgr_soft', 1))
                    self.mdl += (self.sl_mgr_last[branch_id, d] >= getattr(self.args, 'min_onsite_close_mgr_soft', 1))
                continue

            for d, times in limits.items():
                t_first = times['first_hour']
                t_last = times['last_hour']

                mgr_supply_first = pulp.lpSum(
                    self.x[i, sid] for i in branch_managers for sid in shifts_at_work_t[t_first])
                self.mdl += (mgr_supply_first + self.sl_mgr_first[branch_id, d] >= getattr(self.args,
                                                                                           'min_onsite_open_mgr_soft',
                                                                                           1))

                mgr_supply_last = pulp.lpSum(
                    self.x[i, sid] for i in branch_managers for sid in shifts_at_work_t[t_last])
                self.mdl += (mgr_supply_last + self.sl_mgr_last[branch_id, d] >= getattr(self.args,
                                                                                         'min_onsite_close_mgr_soft',
                                                                                         1))

    def solve(self):
        mode = "Finetune" if self.baseline_assignments else "Normal"
        print(f"====== Solving {mode} Model with HiGHS (via PuLP CMD) ======")

        highs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highs/bin/highs.exe")

        if not os.path.exists(highs_path):
            print(f"!!! Error: Can't find solver engine {highs_path} !!!")
            return False

        min_gap = getattr(self.args, 'mip_rel_gap', 0.01)
        solver = pulp.HiGHS_CMD(
            path=highs_path,
            timeLimit=getattr(self.args, 'time_limit', 600),
            msg=True,
            keepFiles=False,
            options=[f'mip_rel_gap={min_gap}']
        )

        status = self.mdl.solve(solver)

        if status == pulp.LpStatusOptimal or (
                status == pulp.LpStatusNotSolved and self.get_objective_value() is not None):
            print(">>> Solution Found!")
            return True
        else:
            print(">>> No Solution Found or Infeasible.")
            return False

    def get_objective_value(self):
        return pulp.value(self.mdl.objective)

    def export_violate_registry(self):
        if self.mdl.status not in [pulp.LpStatusOptimal, pulp.LpStatusNotSolved]:
            return pd.DataFrame()

        violations = []
        tol = 0.01

        for i in self.data.staff_ids:
            val_idle = pulp.value(self.v_idle[i]) or 0.0
            if val_idle > tol:
                violations.append({
                    "Staff": i, "Branch": self.data.staff_branch_map.get(i, "Unknown"),
                    "Day/Time": "Weekly", "Constraint": "Completely Idle (0 shifts)",
                    "Violation_Amount": round(val_idle, 2)
                })

            val_min_days = pulp.value(self.v_min_work_days[i]) or 0.0
            if val_min_days > tol:
                violations.append({
                    "Staff": i, "Branch": self.data.staff_branch_map.get(i, "Unknown"),
                    "Day/Time": "Weekly", "Constraint": "Too Many Days Off",
                    "Violation_Amount": round(val_min_days, 2)
                })

            for d in self.data.days:
                val_daily = pulp.value(self.v_max_daily[i, d]) or 0.0
                if val_daily > tol:
                    violations.append({
                        "Staff": i, "Branch": self.data.staff_branch_map.get(i, "Unknown"),
                        "Day/Time": f"Day {d}", "Constraint": "Max Daily Hours",
                        "Violation_Amount": round(val_daily, 2)
                    })
            for s in self.data.shifts:
                val_shift = pulp.value(self.v_max_shift[i, s['id']]) or 0.0
                if val_shift > tol:
                    violations.append({
                        "Staff": i, "Branch": self.data.staff_branch_map.get(i, "Unknown"),
                        "Day/Time": f"Shift {s['id']}", "Constraint": "Max Shift Length",
                        "Violation_Amount": round(val_shift, 2)
                    })

        for t in self.data.time_slots:
            val_all = pulp.value(self.v_min_onsite_all[t]) or 0.0
            if val_all > tol:
                violations.append(
                    {"Staff": "ALL", "Branch": "Global", "Day/Time": f"Time {t}", "Constraint": "Min Onsite All",
                     "Violation_Amount": round(val_all, 2)})

            val_mgr = pulp.value(self.v_min_onsite_mgr[t]) or 0.0
            if val_mgr > tol:
                violations.append(
                    {"Staff": "Manager", "Branch": "Global", "Day/Time": f"Time {t}", "Constraint": "Min Onsite Mgr",
                     "Violation_Amount": round(val_mgr, 2)})

            val_not_mgr = pulp.value(self.v_min_onsite_not_mgr[t]) or 0.0
            if val_not_mgr > tol:
                violations.append({"Staff": "Not Manager", "Branch": "Global", "Day/Time": f"Time {t}",
                                   "Constraint": "Min Onsite Not Mgr", "Violation_Amount": round(val_not_mgr, 2)})

            val_not_new = pulp.value(self.v_min_onsite_not_new[t]) or 0.0
            if val_not_new > tol:
                violations.append({"Staff": "Not New", "Branch": "Global", "Day/Time": f"Time {t}",
                                   "Constraint": "Min Onsite Not New", "Violation_Amount": round(val_not_new, 2)})

        for bd in self.sl_mgr_first:
            branch_id, d = bd
            val_first = pulp.value(self.sl_mgr_first[bd]) or 0.0
            if val_first > tol:
                violations.append({"Staff": "Manager", "Branch": branch_id, "Day/Time": f"Day {d} Open",
                                   "Constraint": "Min Onsite Open Mgr", "Violation_Amount": round(val_first, 2)})

            val_last = pulp.value(self.sl_mgr_last[bd]) or 0.0
            if val_last > tol:
                violations.append({"Staff": "Manager", "Branch": branch_id, "Day/Time": f"Day {d} Close",
                                   "Constraint": "Min Onsite Close Mgr", "Violation_Amount": round(val_last, 2)})

        return pd.DataFrame(violations)

    def export_results(self):
        if self.mdl.status not in [pulp.LpStatusOptimal, pulp.LpStatusNotSolved]:
            return pd.DataFrame()

        records = []
        for i in self.data.staff_ids:
            branch = self.data.staff_branch_map.get(i)
            valid_times = self.data.branch_valid_bus_times.get(branch, set())

            for s in self.data.shifts:
                val_x = pulp.value(self.x[i, s['id']])
                if val_x is not None and val_x > 0.5:
                    day_idx = s['start'] // 24
                    t_start = s['start'] % 168
                    t_end = s['end'] % 168
                    bus_compatible = (t_start in valid_times) and (t_end in valid_times)

                    records.append({
                        'Staff': i, 'Shift': s['id'], 'Day': day_idx,
                        'Start': s['start'], 'Duration': s['duration'], 'End': s['end'],
                        'Bus_Compatible': bus_compatible
                    })
        return pd.DataFrame(records)