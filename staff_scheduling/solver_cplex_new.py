from docplex.mp.model import Model
import pandas as pd


class StaffSchedulingSolver:
    def __init__(self, args, data, baseline_assignments=None):
        self.args = args
        self.data = data
        self.baseline_assignments = baseline_assignments

        mode_name = "Finetune" if baseline_assignments else "Normal"
        self.mdl = Model(name=f"Staff_Scheduling_ShiftBased_{mode_name}")

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
        self.v_idle = {}  # NEW: Slack for completely idle staff

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

        self.x = self.mdl.binary_var_matrix(staff_ids, shift_ids, name="x")
        self.w = self.mdl.binary_var_matrix(staff_ids, days, name="w")
        self.o_daily = self.mdl.continuous_var_matrix(staff_ids, days, lb=0, name="o_daily")

        self.b_start = self.mdl.binary_var_matrix(clusters, time_slots, name="bus_start")
        self.b_end = self.mdl.binary_var_matrix(clusters, time_slots, name="bus_end")

        self.sl = self.mdl.continuous_var_dict(time_slots, lb=0, name="slack_demand")
        self.v_max_daily = self.mdl.continuous_var_matrix(staff_ids, days, lb=0, name="v_max_daily")
        self.v_max_shift = self.mdl.continuous_var_matrix(staff_ids, shift_ids, lb=0, name="v_max_shift")
        self.v_min_onsite_all = self.mdl.continuous_var_dict(time_slots, lb=0, name="v_min_onsite_all")
        self.v_min_onsite_mgr = self.mdl.continuous_var_dict(time_slots, lb=0, name="v_min_onsite_mgr")
        self.v_min_onsite_not_mgr = self.mdl.continuous_var_dict(time_slots, lb=0, name="v_min_onsite_not_mgr")
        self.v_min_onsite_not_new = self.mdl.continuous_var_dict(time_slots, lb=0, name="v_min_onsite_not_new")
        self.v_min_work_days = self.mdl.continuous_var_dict(staff_ids, lb=0, name="v_min_work_days")
        self.v_idle = self.mdl.continuous_var_dict(staff_ids, lb=0, name="v_idle")

        branch_days = [(b, d) for b in self.data.branch_operation_limits for d in days]
        self.sl_mgr_first = self.mdl.continuous_var_dict(branch_days, lb=0, name="sl_mgr_first")
        self.sl_mgr_last = self.mdl.continuous_var_dict(branch_days, lb=0, name="sl_mgr_last")

        term_wage = self.mdl.sum(
            self.data.wage_reg[i] * s['duration'] * self.x[i, s['id']] for i in staff_ids for s in shifts)
        term_ot = self.mdl.sum(self.data.wage_ot[i] * self.o_daily[i, d] for i in staff_ids for d in days)
        term_bus_trip = self.mdl.sum(
            getattr(self.args, 'w_bus_trip', 1.0) * (self.b_start[c, t] + self.b_end[c, t]) for c in clusters for t in
            time_slots)

        term_deviation = 0
        if self.baseline_assignments is not None:
            w_change = getattr(self.args, 'w_change', 100.0)
            for i in staff_ids:
                for s in shifts:
                    if (i, s['id']) in self.baseline_assignments:
                        term_deviation += w_change * (1 - self.x[i, s['id']])
                    else:
                        term_deviation += w_change * self.x[i, s['id']]

        M1 = 1000
        M2 = 1000
        M3 = 10
        M4 = 1

        penalty_m1 = M1 * (
                self.mdl.sum(self.sl[t] for t in time_slots) +
                self.mdl.sum(self.v_idle[i] for i in staff_ids)  # Penalize heavily if completely idle
        )

        penalty_m2 = M2 * (
                self.mdl.sum(self.v_max_daily[i, d] for i in saudi_not_mgr for d in days) +
                self.mdl.sum(self.v_min_onsite_not_new[t] for t in time_slots)
        )

        penalty_m3 = M3 * (
                self.mdl.sum(self.v_min_work_days[i] for i in staff_ids) +
                self.mdl.sum(self.v_max_daily[i, d] for i in saudi_mgr for d in days) +
                self.mdl.sum(self.v_max_shift[i, sid] for i in saudi_not_mgr for sid in shift_ids) +
                self.mdl.sum(self.v_max_daily[i, d] for i in expat_mgr for d in days) +
                self.mdl.sum(self.v_max_shift[i, sid] for i in expat_mgr for sid in shift_ids) +
                self.mdl.sum(self.v_max_shift[i, sid] for i in expat_not_mgr for sid in shift_ids) +
                self.mdl.sum(self.v_max_daily[i, d] for i in expat_not_mgr for d in days) +
                self.mdl.sum(self.v_min_onsite_all[t] for t in time_slots) +
                self.mdl.sum(self.v_min_onsite_mgr[t] for t in time_slots) +
                self.mdl.sum(self.sl_mgr_first[bd] + self.sl_mgr_last[bd] for bd in branch_days)
        )

        penalty_m4 = M4 * self.mdl.sum(self.v_min_onsite_not_mgr[t] for t in time_slots)

        # Extra: L1 regularization
        lambda_l1 = 0.1
        term_l1_reg = self.mdl.sum(lambda_l1 * self.x[i, s['id']] for i in staff_ids for s in shifts)

        # Extra: Cost Perturbation
        term_perturbation = 0
        for idx, i in enumerate(staff_ids):
            pert_val = 0.0001 * idx
            for s in shifts:
                term_perturbation += pert_val * self.x[i, s['id']]

        term_penalties = penalty_m1 + penalty_m2 + penalty_m3 + penalty_m4

        self.mdl.minimize(term_wage + term_ot + term_bus_trip + term_deviation + term_penalties + term_l1_reg + term_perturbation)

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

        for c in clusters:
            staff_in_cluster = [i for i in expat_set if self.data.staff_cluster_map.get(i) == c]
            M_val = len(staff_in_cluster)
            if M_val > 0:
                for t in time_slots:
                    if shifts_starting_at[t]:
                        self.mdl.add_constraint(self.mdl.sum(
                            self.x[i, sid] for i in staff_in_cluster for sid in shifts_starting_at[t]) <= M_val *
                                                self.b_start[c, t])
                    else:
                        self.mdl.add_constraint(self.b_start[c, t] == 0)
                    if shifts_ending_at[t]:
                        self.mdl.add_constraint(self.mdl.sum(
                            self.x[i, sid] for i in staff_in_cluster for sid in shifts_ending_at[t]) <= M_val *
                                                self.b_end[c, t])
                    else:
                        self.mdl.add_constraint(self.b_end[c, t] == 0)
            else:
                for t in time_slots:
                    self.mdl.add_constraint(self.b_start[c, t] == 0)
                    self.mdl.add_constraint(self.b_end[c, t] == 0)

        if getattr(self.args, 'enforce_bus', False):
            for s in shifts:
                for i in expat_set:
                    branch = self.data.staff_branch_map.get(i)
                    if branch:
                        valid_times = self.data.branch_valid_bus_times.get(branch, set())
                        if (s['start'] % 168) not in valid_times or (s['end'] % 168) not in valid_times:
                            self.mdl.add_constraint(self.x[i, s['id']] == 0)

        for i in staff_ids:
            for s in shifts:
                sid = s['id']
                dur = s['duration']

                if i in saudi_not_mgr and dur < getattr(self.args, 'min_shift_saudi_not_mgr', 6):
                    self.mdl.add_constraint(self.x[i, sid] == 0)
                elif i in saudi_mgr and dur < getattr(self.args, 'min_shift_saudi_mgr', 8):
                    self.mdl.add_constraint(self.x[i, sid] == 0)
                elif i in expat_set and dur < getattr(self.args, 'min_shift_expat', 9):
                    self.mdl.add_constraint(self.x[i, sid] == 0)

                if i in expat_not_mgr and dur > getattr(self.args, 'max_shift_expat_not_mgr_hard', 16):
                    self.mdl.add_constraint(self.x[i, sid] == 0)

                if i in saudi_not_mgr:
                    self.mdl.add_constraint(dur * self.x[i, sid] - self.v_max_shift[i, sid] <= getattr(self.args,
                                                                                                       'max_shift_saudi_not_mgr_soft',
                                                                                                       6))
                elif i in expat_mgr:
                    self.mdl.add_constraint(dur * self.x[i, sid] - self.v_max_shift[i, sid] <= getattr(self.args,
                                                                                                       'max_shift_expat_mgr_soft',
                                                                                                       9))
                elif i in expat_not_mgr:
                    self.mdl.add_constraint(dur * self.x[i, sid] - self.v_max_shift[i, sid] <= getattr(self.args,
                                                                                                       'max_shift_expat_not_mgr_soft',
                                                                                                       12))

        for i in staff_ids:
            for d in days:
                relevant = shifts_at_day[d]
                if not relevant:
                    self.mdl.add_constraint(self.o_daily[i, d] == 0)
                    self.mdl.add_constraint(self.w[i, d] == 0)
                    continue

                daily_hours = self.mdl.sum(self.x[i, sid] * self.data.shift_map[sid]['duration'] for sid in relevant)
                self.mdl.add_constraint(daily_hours <= 24 * self.w[i, d])
                self.mdl.add_constraint(daily_hours >= 0.5 * self.w[i, d])

                if i in saudi_not_mgr:
                    standard_min = getattr(self.args, 'min_daily_saudi_not_mgr', 6)
                    self.mdl.add_constraint(self.o_daily[i, d] >= daily_hours - standard_min)
                elif i in expat_not_mgr:
                    standard_min = getattr(self.args, 'min_daily_expat', 9)
                    self.mdl.add_constraint(self.o_daily[i, d] >= daily_hours - standard_min)
                else:
                    self.mdl.add_constraint(self.o_daily[i, d] == 0)


                if i in manager_set:
                    self.mdl.add_constraint(self.o_daily[i, d] == 0)

                if i in saudi_not_mgr:
                    self.mdl.add_constraint(
                        daily_hours >= getattr(self.args, 'min_daily_saudi_not_mgr', 6) * self.w[i, d])
                elif i in saudi_mgr:
                    self.mdl.add_constraint(daily_hours >= getattr(self.args, 'min_daily_saudi_mgr', 8) * self.w[i, d])
                elif i in expat_set:
                    self.mdl.add_constraint(daily_hours >= getattr(self.args, 'min_daily_expat', 9) * self.w[i, d])

                if i in expat_not_mgr:
                    self.mdl.add_constraint(
                        daily_hours <= getattr(self.args, 'max_daily_expat_not_mgr_hard', 16) * self.w[i, d])

                if i in saudi_not_mgr:
                    self.mdl.add_constraint(
                        daily_hours - self.v_max_daily[i, d] <= getattr(self.args, 'max_daily_saudi_not_mgr_soft', 6) *
                        self.w[i, d])
                elif i in saudi_mgr:
                    self.mdl.add_constraint(
                        daily_hours - self.v_max_daily[i, d] <= getattr(self.args, 'max_daily_saudi_mgr_soft', 8) *
                        self.w[i, d])
                elif i in expat_mgr:
                    self.mdl.add_constraint(
                        daily_hours - self.v_max_daily[i, d] <= getattr(self.args, 'max_daily_expat_mgr_soft', 9) *
                        self.w[i, d])
                elif i in expat_not_mgr:
                    self.mdl.add_constraint(
                        daily_hours - self.v_max_daily[i, d] <= getattr(self.args, 'max_daily_expat_not_mgr_soft', 12) *
                        self.w[i, d])

            # FIXED: Avoid 0 >= 1 paradox by using slack variable
            self.mdl.add_constraint(self.mdl.sum(self.w[i, d] for d in days) + self.v_idle[i] >= 1)

            target_work_days = 7 - getattr(self.args, 'days_off', 1)
            self.mdl.add_constraint(
                self.mdl.sum(self.w[i, d] for d in days) + self.v_min_work_days[i] >= target_work_days)
            self.mdl.add_constraint(self.mdl.sum(self.w[i, d] for d in days) <= target_work_days)

            total_weekly = self.mdl.sum(s['duration'] * self.x[i, s['id']] for s in shifts)
            self.mdl.add_constraint(total_weekly <= getattr(self.args, 'max_weekly_hours', 78))

        for i in staff_ids:
            for t in time_slots:
                if shifts_at_extended_t[t]:
                    self.mdl.add_constraint(self.mdl.sum(self.x[i, sid] for sid in shifts_at_extended_t[t]) <= 1)

        for t in time_slots:
            supply_all = self.mdl.sum(self.x[i, sid] for i in staff_ids for sid in shifts_at_work_t[t])
            supply_mgr = self.mdl.sum(self.x[i, sid] for i in manager_set for sid in shifts_at_work_t[t])
            supply_non_mgr = self.mdl.sum(self.x[i, sid] for i in non_manager_set for sid in shifts_at_work_t[t])
            supply_not_new = self.mdl.sum(self.x[i, sid] for i in not_new_set for sid in shifts_at_work_t[t])

            self.mdl.add_constraint(supply_all + self.sl[t] >= self.data.demand[t])

            if self.data.demand[t] > 0:
                # FIXED: Protect against empty shift pools at specific hours
                if shifts_at_work_t[t]:
                    self.mdl.add_constraint(supply_all >= getattr(self.args, 'min_onsite_all_hard', 1))

                self.mdl.add_constraint(
                    supply_not_new + self.v_min_onsite_not_new[t] >= getattr(self.args, 'min_onsite_not_new_soft', 1))
                self.mdl.add_constraint(
                    supply_all + self.v_min_onsite_all[t] >= getattr(self.args, 'min_onsite_all_soft', 2))
                self.mdl.add_constraint(
                    supply_mgr + self.v_min_onsite_mgr[t] >= getattr(self.args, 'min_onsite_mgr_soft', 1))
                self.mdl.add_constraint(
                    supply_non_mgr + self.v_min_onsite_not_mgr[t] >= getattr(self.args, 'min_onsite_not_mgr_soft', 2))

        for branch_id, limits in self.data.branch_operation_limits.items():
            branch_managers = managers_by_branch.get(branch_id, [])
            if not branch_managers:
                for d in days:
                    self.mdl.add_constraint(
                        self.sl_mgr_first[branch_id, d] >= getattr(self.args, 'min_onsite_open_mgr_soft', 1))
                    self.mdl.add_constraint(
                        self.sl_mgr_last[branch_id, d] >= getattr(self.args, 'min_onsite_close_mgr_soft', 1))
                continue

            for d, times in limits.items():
                t_first = times['first_hour']
                t_last = times['last_hour']

                mgr_supply_first = self.mdl.sum(
                    self.x[i, sid] for i in branch_managers for sid in shifts_at_work_t[t_first])
                self.mdl.add_constraint(
                    mgr_supply_first + self.sl_mgr_first[branch_id, d] >= getattr(self.args, 'min_onsite_open_mgr_soft',
                                                                                  1))

                mgr_supply_last = self.mdl.sum(
                    self.x[i, sid] for i in branch_managers for sid in shifts_at_work_t[t_last])
                self.mdl.add_constraint(
                    mgr_supply_last + self.sl_mgr_last[branch_id, d] >= getattr(self.args, 'min_onsite_close_mgr_soft',
                                                                                1))

    def solve(self):
        mode = "Finetune" if self.baseline_assignments else "Normal"
        print(f"====== Solving {mode} Model with CPLEX ======")
        self.mdl.set_time_limit(getattr(self.args, 'time_limit', 300))

        target_gap = getattr(self.args, 'mip_gap', 0.01)
        self.mdl.parameters.mip.tolerances.mipgap = target_gap

        self.solution = self.mdl.solve(log_output=True)

        if self.solution:
            print(">>> Solution Found!")
            return True
        else:
            print(">>> No Solution Found.")
            return False

    def get_objective_value(self):
        if self.solution:
            return self.solution.objective_value
        return None

    def export_violate_registry(self):
        if not self.solution: return pd.DataFrame()

        violations = []
        tol = 0.01

        for i in self.data.staff_ids:
            val_idle = self.solution.get_value(self.v_idle[i])
            if val_idle > tol:
                violations.append({
                    "Staff": i, "Branch": self.data.staff_branch_map.get(i, "Unknown"),
                    "Day/Time": "Weekly", "Constraint": "Completely Idle (0 shifts)",
                    "Violation_Amount": round(val_idle, 2)
                })

            val_min_days = self.solution.get_value(self.v_min_work_days[i])
            if val_min_days > tol:
                violations.append({
                    "Staff": i, "Branch": self.data.staff_branch_map.get(i, "Unknown"),
                    "Day/Time": "Weekly", "Constraint": "Too Many Days Off",
                    "Violation_Amount": round(val_min_days, 2)
                })

            for d in self.data.days:
                val_daily = self.solution.get_value(self.v_max_daily[i, d])
                if val_daily > tol:
                    violations.append({
                        "Staff": i, "Branch": self.data.staff_branch_map.get(i, "Unknown"),
                        "Day/Time": f"Day {d}", "Constraint": "Max Daily Hours",
                        "Violation_Amount": round(val_daily, 2)
                    })
            for s in self.data.shifts:
                val_shift = self.solution.get_value(self.v_max_shift[i, s['id']])
                if val_shift > tol:
                    violations.append({
                        "Staff": i, "Branch": self.data.staff_branch_map.get(i, "Unknown"),
                        "Day/Time": f"Shift {s['id']}", "Constraint": "Max Shift Length",
                        "Violation_Amount": round(val_shift, 2)
                    })

        for t in self.data.time_slots:
            val_all = self.solution.get_value(self.v_min_onsite_all[t])
            if val_all > tol:
                violations.append(
                    {"Staff": "ALL", "Branch": "Global", "Day/Time": f"Time {t}", "Constraint": "Min Onsite All",
                     "Violation_Amount": round(val_all, 2)})

            val_mgr = self.solution.get_value(self.v_min_onsite_mgr[t])
            if val_mgr > tol:
                violations.append(
                    {"Staff": "Manager", "Branch": "Global", "Day/Time": f"Time {t}", "Constraint": "Min Onsite Mgr",
                     "Violation_Amount": round(val_mgr, 2)})

            val_not_mgr = self.solution.get_value(self.v_min_onsite_not_mgr[t])
            if val_not_mgr > tol:
                violations.append({"Staff": "Not Manager", "Branch": "Global", "Day/Time": f"Time {t}",
                                   "Constraint": "Min Onsite Not Mgr", "Violation_Amount": round(val_not_mgr, 2)})

            val_not_new = self.solution.get_value(self.v_min_onsite_not_new[t])
            if val_not_new > tol:
                violations.append({"Staff": "Not New", "Branch": "Global", "Day/Time": f"Time {t}",
                                   "Constraint": "Min Onsite Not New", "Violation_Amount": round(val_not_new, 2)})

        for bd in self.sl_mgr_first:
            branch_id, d = bd
            val_first = self.solution.get_value(self.sl_mgr_first[bd])
            if val_first > tol:
                violations.append({"Staff": "Manager", "Branch": branch_id, "Day/Time": f"Day {d} Open",
                                   "Constraint": "Min Onsite Open Mgr", "Violation_Amount": round(val_first, 2)})

            val_last = self.solution.get_value(self.sl_mgr_last[bd])
            if val_last > tol:
                violations.append({"Staff": "Manager", "Branch": branch_id, "Day/Time": f"Day {d} Close",
                                   "Constraint": "Min Onsite Close Mgr", "Violation_Amount": round(val_last, 2)})

        return pd.DataFrame(violations)

    def export_results(self):
        if not self.solution: return pd.DataFrame()
        records = []
        for i in self.data.staff_ids:
            branch = self.data.staff_branch_map.get(i)
            valid_times = self.data.branch_valid_bus_times.get(branch, set())

            for s in self.data.shifts:
                if self.solution.get_value(self.x[i, s['id']]) > 0.5:
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