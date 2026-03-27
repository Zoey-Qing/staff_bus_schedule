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
        self.sl = {}
        self.w = {}

        self.b_start = {}  # b_start[c, t]: 1 if cluster c has a start trip at time t
        self.b_end = {}  # b_end[c, t]: 1 if cluster c has an end trip at time t

        self.sl_mgr_first = {}
        self.sl_mgr_last = {}

    def build_model(self):
        staff_ids = self.data.staff_ids
        shifts = self.data.shifts
        time_slots = self.data.time_slots
        days = self.data.days
        clusters = self.data.cluster_ids

        manager_set = set(self.data.manager_ids)
        non_saudi_set = set(self.data.non_saudi_staff_ids)

        managers_by_branch = {}
        for mid in self.data.manager_ids:
            bid = self.data.staff_branch_map.get(mid)
            if bid:
                if bid not in managers_by_branch: managers_by_branch[bid] = []
                managers_by_branch[bid].append(mid)

        shift_ids = [s['id'] for s in shifts]
        self.x = self.mdl.binary_var_matrix(staff_ids, shift_ids, name="x")
        self.sl = self.mdl.continuous_var_dict(time_slots, lb=0, name="slack")
        self.w = self.mdl.binary_var_matrix(staff_ids, days, name="w")
        self.o_daily = self.mdl.continuous_var_matrix(staff_ids, days, lb=0, name="o_daily")

        self.b_start = self.mdl.binary_var_matrix(clusters, time_slots, name="bus_start")
        self.b_end = self.mdl.binary_var_matrix(clusters, time_slots, name="bus_end")

        self.sl_mgr_first = self.mdl.continuous_var_dict(
            [(b, d) for b in self.data.branch_operation_limits for d in days], lb=0, name="sl_mgr_first")
        self.sl_mgr_last = self.mdl.continuous_var_dict(
            [(b, d) for b in self.data.branch_operation_limits for d in days], lb=0, name="sl_mgr_last")

        term_wage = self.mdl.sum(
            self.data.wage_reg[i] * s['duration'] * self.x[i, s['id']] for i in staff_ids for s in shifts)
        term_ot = self.mdl.sum(self.data.wage_ot[i] * self.o_daily[i, d] for i in staff_ids for d in days)
        term_penalty = self.mdl.sum(self.args.w_understaffing * self.sl[t] for t in time_slots)

        term_bus_trip = self.mdl.sum(
            self.args.w_bus_trip * (self.b_start[c, t] + self.b_end[c, t]) for c in clusters for t in time_slots)

        w_mgr_abs = getattr(self.args, 'w_manager_absence', 100.0)
        term_mgr_penalty = self.mdl.sum(
            w_mgr_abs * (self.sl_mgr_first[b, d] + self.sl_mgr_last[b, d]) for b in self.data.branch_operation_limits
            for d in days)

        term_reward = 0
        for i in staff_ids:
            if self.data.skill_score[i] > 0:
                for s in shifts:
                    peak_h = sum(self.data.is_peak[t] for t in s['covers'])
                    if peak_h > 0:
                        term_reward += self.args.w_experienced * self.data.skill_score[i] * peak_h * self.x[i, s['id']]

        term_deviation = 0
        if self.baseline_assignments is not None:
            W_CHANGE = getattr(self.args, 'w_change', 1000.0)
            for i in staff_ids:
                for s in shifts:
                    if (i, s['id']) in self.baseline_assignments:
                        term_deviation += W_CHANGE * (1 - self.x[i, s['id']])
                    else:
                        term_deviation += W_CHANGE * self.x[i, s['id']]

        self.mdl.minimize(
            term_wage + term_ot + term_penalty + term_bus_trip + term_mgr_penalty + term_deviation - term_reward)

        shifts_at_work_t = {t: [] for t in time_slots}
        for s in shifts:
            for t in s['covers']: shifts_at_work_t[t].append(s['id'])

        shifts_at_extended_t = {t: [] for t in time_slots}
        for s in shifts:
            extended_end = min(168, s['end'] + self.args.min_break_hours)
            for t in range(s['start'], extended_end): shifts_at_extended_t[t].append(s['id'])

        shifts_at_day = {d: [] for d in days}
        for s in shifts:
            day_idx = s['start'] // 24
            if 0 <= day_idx < 7: shifts_at_day[day_idx].append(s['id'])

        shifts_starting_at = {t: [] for t in time_slots}
        shifts_ending_at = {t: [] for t in time_slots}
        for s in shifts:
            t_start = s['start'] % 168
            t_end = s['end'] % 168
            shifts_starting_at[t_start].append(s['id'])
            shifts_ending_at[t_end].append(s['id'])

        # === Bus Trip Consolidation Constraints===
        for c in clusters:
            staff_in_cluster = [i for i in non_saudi_set if self.data.staff_cluster_map.get(i) == c]
            M = len(staff_in_cluster)

            if M > 0:
                for t in time_slots:
                    relevant_starts = shifts_starting_at[t]
                    if relevant_starts:
                        self.mdl.add_constraint(
                            self.mdl.sum(self.x[i, sid] for i in staff_in_cluster for sid in relevant_starts) <= M *
                            self.b_start[c, t]
                        )
                    else:
                        self.mdl.add_constraint(self.b_start[c, t] == 0)

                    relevant_ends = shifts_ending_at[t]
                    if relevant_ends:
                        self.mdl.add_constraint(
                            self.mdl.sum(self.x[i, sid] for i in staff_in_cluster for sid in relevant_ends) <= M *
                            self.b_end[c, t]
                        )
                    else:
                        self.mdl.add_constraint(self.b_end[c, t] == 0)
            else:
                for t in time_slots:
                    self.mdl.add_constraint(self.b_start[c, t] == 0)
                    self.mdl.add_constraint(self.b_end[c, t] == 0)

        # Precise Bus Enforce Constraint
        if self.args.enforce_bus:
            for s in shifts:
                t_start = s['start'] % 168
                t_end = s['end'] % 168
                for i in non_saudi_set:
                    branch = self.data.staff_branch_map.get(i)
                    if branch:
                        valid_times = self.data.branch_valid_bus_times.get(branch, set())
                        if t_start not in valid_times or t_end not in valid_times:
                            self.mdl.add_constraint(self.x[i, s['id']] == 0)

        # Demand Coverage
        for t in time_slots:
            supply = self.mdl.sum(self.x[i, sid] for i in staff_ids for sid in shifts_at_work_t[t])
            self.mdl.add_constraint(supply + self.sl[t] >= self.data.demand[t], f"Dem_{t}")
            if self.data.demand[t] > 0:
                skill_supply = self.mdl.sum(
                    self.x[i, sid] for i in self.data.non_new_staff_ids for sid in shifts_at_work_t[t])
                self.mdl.add_constraint(skill_supply >= 1, f"Skill_All_{t}")
                mgr_supply_global = self.mdl.sum(
                    self.x[i, sid] for i in self.data.manager_ids for sid in shifts_at_work_t[t])
                self.mdl.add_constraint(mgr_supply_global >= 1, f"Mgr_Global_{t}")

        # Manager Constraint
        for branch_id, limits in self.data.branch_operation_limits.items():
            branch_managers = managers_by_branch.get(branch_id, [])
            if not branch_managers:
                for d in days:
                    self.mdl.add_constraint(self.sl_mgr_first[branch_id, d] >= 1)
                    self.mdl.add_constraint(self.sl_mgr_last[branch_id, d] >= 1)
                continue
            for d, times in limits.items():
                t_first = times['first_hour']
                t_last = times['last_hour']
                mgr_supply_first = self.mdl.sum(
                    self.x[i, sid] for i in branch_managers for sid in shifts_at_work_t[t_first])
                self.mdl.add_constraint(mgr_supply_first + self.sl_mgr_first[branch_id, d] >= 1)
                mgr_supply_last = self.mdl.sum(
                    self.x[i, sid] for i in branch_managers for sid in shifts_at_work_t[t_last])
                self.mdl.add_constraint(mgr_supply_last + self.sl_mgr_last[branch_id, d] >= 1)

        # No Overlap
        for i in staff_ids:
            for t in time_slots:
                if shifts_at_extended_t[t]:
                    self.mdl.add_constraint(self.mdl.sum(self.x[i, sid] for sid in shifts_at_extended_t[t]) <= 1)

        # Workload & Days Off
        for i in staff_ids:
            for d in days:
                relevant = shifts_at_day[d]
                if not relevant:
                    self.mdl.add_constraint(self.o_daily[i, d] == 0)
                    continue
                daily_hours = self.mdl.sum(self.x[i, sid] * self.data.shift_map[sid]['duration'] for sid in relevant)
                self.mdl.add_constraint(daily_hours <= 24 * self.w[i, d])
                self.mdl.add_constraint(daily_hours <= self.args.max_daily_hours * self.w[i, d])
                self.mdl.add_constraint(daily_hours >= self.args.min_daily_hours * self.w[i, d])
                self.mdl.add_constraint(self.o_daily[i, d] >= daily_hours - self.args.min_daily_hours)

                if i in manager_set:
                    self.mdl.add_constraint(self.o_daily[i, d] == 0)

            self.mdl.add_constraint(self.mdl.sum(self.w[i, d] for d in days) == 7 - self.args.days_off, f"DaysOff_{i}")
            total_weekly = self.mdl.sum(s['duration'] * self.x[i, s['id']] for s in shifts)
            self.mdl.add_constraint(total_weekly <= self.args.max_weekly_hours)

    def solve(self):
        mode = "Finetune" if self.baseline_assignments else "Normal"
        print(f"====== Solving {mode} Model with CPLEX ======")
        self.mdl.set_time_limit(self.args.time_limit)

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