import pandas as pd
import pulp


class StaffSchedulingSolver:
    def __init__(self, args, data, baseline_assignments=None):
        self.args = args
        self.data = data
        self.baseline_assignments = baseline_assignments

        mode_name = "Finetune" if baseline_assignments else "Normal"
        self.mdl = pulp.LpProblem(f"Staff_Scheduling_{mode_name}", pulp.LpMinimize)

        self.x = {}
        self.o_daily = {}
        self.sl = {}
        self.w = {}
        self.b_start = {}
        self.b_end = {}
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

        # Decision Variables
        for i in staff_ids:
            for s in shifts:
                self.x[i, s['id']] = pulp.LpVariable(f"x_{i}_{s['id']}", cat='Binary')
            for d in days:
                self.w[i, d] = pulp.LpVariable(f"w_{i}_{d}", cat='Binary')
                self.o_daily[i, d] = pulp.LpVariable(f"o_daily_{i}_{d}", lowBound=0, cat='Continuous')

        for t in time_slots:
            self.sl[t] = pulp.LpVariable(f"sl_{t}", lowBound=0, cat='Continuous')

        for c in clusters:
            for t in time_slots:
                self.b_start[c, t] = pulp.LpVariable(f"b_start_{c}_{t}", cat='Binary')
                self.b_end[c, t] = pulp.LpVariable(f"b_end_{c}_{t}", cat='Binary')

        for b in self.data.branch_operation_limits:
            for d in days:
                self.sl_mgr_first[b, d] = pulp.LpVariable(f"sl_mgr_first_{b}_{d}", lowBound=0, cat='Continuous')
                self.sl_mgr_last[b, d] = pulp.LpVariable(f"sl_mgr_last_{b}_{d}", lowBound=0, cat='Continuous')

        shifts_at_work_t = {t: [] for t in time_slots}
        shifts_at_extended_t = {t: [] for t in time_slots}
        shifts_at_day = {d: [] for d in days}
        shifts_starting_at = {t: [] for t in time_slots}
        shifts_ending_at = {t: [] for t in time_slots}

        for s in shifts:
            sid = s['id']
            for t in s['covers']: shifts_at_work_t[t].append(sid)
            extended_end = min(168, s['end'] + self.args.min_break_hours)
            for t in range(s['start'], extended_end): shifts_at_extended_t[t].append(sid)
            day_idx = s['start'] // 24
            if 0 <= day_idx < 7: shifts_at_day[day_idx].append(sid)
            shifts_starting_at[s['start'] % 168].append(sid)
            shifts_ending_at[s['end'] % 168].append(sid)

        # Objective Function
        term_wage = pulp.lpSum(
            self.data.wage_reg[i] * s['duration'] * self.x[i, s['id']] for i in staff_ids for s in shifts)
        term_ot = pulp.lpSum(self.data.wage_ot[i] * self.o_daily[i, d] for i in staff_ids for d in days)
        term_penalty = pulp.lpSum(self.args.w_understaffing * self.sl[t] for t in time_slots)
        term_bus_trip = pulp.lpSum(
            self.args.w_bus_trip * (self.b_start[c, t] + self.b_end[c, t]) for c in clusters for t in time_slots)

        w_mgr_abs = getattr(self.args, 'w_manager_absence', 100.0)
        term_mgr_penalty = pulp.lpSum(
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

        self.mdl += (
                    term_wage + term_ot + term_penalty + term_bus_trip + term_mgr_penalty + term_deviation - term_reward), "Objective"

        # Constraints
        # Bus Trip
        for c in clusters:
            staff_in_cluster = [i for i in non_saudi_set if self.data.staff_cluster_map.get(i) == c]
            M = len(staff_in_cluster)
            if M > 0:
                for t in time_slots:
                    relevant_starts = shifts_starting_at[t]
                    if relevant_starts:
                        self.mdl += (pulp.lpSum(
                            self.x[i, sid] for i in staff_in_cluster for sid in relevant_starts) <= M * self.b_start[
                                         c, t])
                    else:
                        self.mdl += (self.b_start[c, t] == 0)

                    relevant_ends = shifts_ending_at[t]
                    if relevant_ends:
                        self.mdl += (
                                    pulp.lpSum(self.x[i, sid] for i in staff_in_cluster for sid in relevant_ends) <= M *
                                    self.b_end[c, t])
                    else:
                        self.mdl += (self.b_end[c, t] == 0)
            else:
                for t in time_slots:
                    self.mdl += (self.b_start[c, t] == 0)
                    self.mdl += (self.b_end[c, t] == 0)

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
                            self.mdl += (self.x[i, s['id']] == 0)

        # Demand Coverage
        for t in time_slots:
            supply = pulp.lpSum(self.x[i, sid] for i in staff_ids for sid in shifts_at_work_t[t])
            self.mdl += (supply + self.sl[t] >= self.data.demand[t])

            if self.data.demand[t] > 0:
                skill_supply = pulp.lpSum(
                    self.x[i, sid] for i in self.data.non_new_staff_ids for sid in shifts_at_work_t[t])
                self.mdl += (skill_supply >= 1)
                mgr_supply_global = pulp.lpSum(
                    self.x[i, sid] for i in self.data.manager_ids for sid in shifts_at_work_t[t])
                self.mdl += (mgr_supply_global >= 1)

        # Manager Constraint
        for branch_id, limits in self.data.branch_operation_limits.items():
            branch_managers = managers_by_branch.get(branch_id, [])
            if not branch_managers:
                for d in days:
                    self.mdl += (self.sl_mgr_first[branch_id, d] >= 1)
                    self.mdl += (self.sl_mgr_last[branch_id, d] >= 1)
                continue
            for d, times in limits.items():
                mgr_supply_first = pulp.lpSum(
                    self.x[i, sid] for i in branch_managers for sid in shifts_at_work_t[times['first_hour']])
                self.mdl += (mgr_supply_first + self.sl_mgr_first[branch_id, d] >= 1)

                mgr_supply_last = pulp.lpSum(
                    self.x[i, sid] for i in branch_managers for sid in shifts_at_work_t[times['last_hour']])
                self.mdl += (mgr_supply_last + self.sl_mgr_last[branch_id, d] >= 1)

        # No Overlap
        for i in staff_ids:
            for t in time_slots:
                if shifts_at_extended_t[t]:
                    self.mdl += (pulp.lpSum(self.x[i, sid] for sid in shifts_at_extended_t[t]) <= 1)

        # Workload & Days Off
        for i in staff_ids:
            for d in days:
                relevant = shifts_at_day[d]
                if not relevant:
                    self.mdl += (self.o_daily[i, d] == 0)
                    continue
                daily_hours = pulp.lpSum(self.x[i, sid] * self.data.shift_map[sid]['duration'] for sid in relevant)
                self.mdl += (daily_hours <= 24 * self.w[i, d])
                self.mdl += (daily_hours <= self.args.max_daily_hours * self.w[i, d])
                self.mdl += (daily_hours >= self.args.min_daily_hours * self.w[i, d])
                self.mdl += (self.o_daily[i, d] >= daily_hours - self.args.min_daily_hours)

                if i in manager_set:
                    self.mdl += (self.o_daily[i, d] == 0)

            self.mdl += (pulp.lpSum(self.w[i, d] for d in days) == 7 - self.args.days_off)
            total_weekly = pulp.lpSum(s['duration'] * self.x[i, s['id']] for s in shifts)
            self.mdl += (total_weekly <= self.args.max_weekly_hours)

    def solve(self):
        mode = "Finetune" if self.baseline_assignments else "Normal"
        print(f"====== Solving {mode} Model with HiGHS (via PuLP CMD) ======")

        import os
        highs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highs/bin/highs.exe")

        if not os.path.exists(highs_path):
            print(f"Error:Can't find solver engine {highs_path} !!!")
            return False

        solver = pulp.HiGHS_CMD(
            path=highs_path,
            timeLimit=self.args.time_limit,
            msg=True,
            keepFiles=False
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

    def export_results(self):
        records = []
        for i in self.data.staff_ids:
            branch = self.data.staff_branch_map.get(i)
            valid_times = self.data.branch_valid_bus_times.get(branch, set())

            for s in self.data.shifts:
                if pulp.value(self.x[i, s['id']]) is not None and pulp.value(self.x[i, s['id']]) > 0.5:
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