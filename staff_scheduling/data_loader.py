import pandas as pd
import numpy as np
import ast
import os

HOURS_PER_WEEK = 168
DAYS_PER_WEEK = 7
START_HOUR_OFFSET = 5  # Sunday 05:00 AM is t=0


class DataLoader:
    def __init__(self, args, target_cluster=None):
        self.args = args
        self.target_cluster = target_cluster

        self.staff_ids = []
        self.manager_ids = []
        self.non_new_staff_ids = []
        self.non_saudi_staff_ids = []
        self.staff_branch_map = {}

        self.branch_cluster_map = {}
        self.staff_cluster_map = {}
        self.cluster_ids = []

        self.time_slots = list(range(HOURS_PER_WEEK))
        self.days = list(range(DAYS_PER_WEEK))

        self.wage_reg = {}
        self.wage_ot = {}
        self.skill_score = {}
        self.demand = {}
        self.is_peak = {}
        self.operating_hours = []

        self.shifts = []
        self.shift_map = {}

        self.branch_valid_bus_times = {}
        self.branch_operation_limits = {}

    def load_and_preprocess(self):
        self._load_raw_data()
        self._load_operation_hours()

        if getattr(self.args, 'enforce_bus', False):
            self._load_bus_fleet_schedule()

        self._generate_shifts()
        return self

    def _load_raw_data(self):
        df_branches = pd.read_excel(self.args.data_file, sheet_name="branches", header=0)
        df_branches.columns = [c.strip() for c in df_branches.columns]

        if self.target_cluster:
            df_branches = df_branches[df_branches['cluster_id'] == self.target_cluster]

        self.branch_cluster_map = df_branches.set_index('branch_id')['cluster_id'].to_dict()
        valid_branches = set(self.branch_cluster_map.keys())
        self.cluster_ids = list(set(self.branch_cluster_map.values()))

        df_staff = pd.read_excel(self.args.data_file, sheet_name="staff", header=0)
        df_staff.columns = [c.strip() for c in df_staff.columns]

        if self.target_cluster:
            df_staff = df_staff[df_staff['branch_id'].isin(valid_branches)]

        if 'Nationality' not in df_staff.columns:
            df_staff['Nationality'] = 'Non-Saudi'

        skill_mapping = {'New': 0, 'Experienced': 1, 'Top Performer': 2}
        col_skill = [c for c in df_staff.columns if 'skill_level' in c][0]
        df_staff['skill_score'] = df_staff[col_skill].map(skill_mapping).fillna(0)

        self.staff_ids = df_staff['staff_id'].unique().tolist()
        self.manager_ids = df_staff[df_staff['role'] == 'Manager']['staff_id'].tolist()
        self.staff_branch_map = df_staff.set_index('staff_id')['branch_id'].to_dict()

        self.staff_cluster_map = {i: self.branch_cluster_map.get(self.staff_branch_map.get(i)) for i in self.staff_ids}

        self.non_new_staff_ids = df_staff[df_staff[col_skill] != 'New']['staff_id'].tolist()
        self.non_saudi_staff_ids = df_staff[
            ~df_staff['Nationality'].fillna('').astype(str).str.strip().str.lower().isin(['saudi', 'saudi arabia'])
        ]['staff_id'].tolist()

        self.wage_reg = df_staff.set_index('staff_id')['hourly_wage'].to_dict()
        self.wage_ot = df_staff.set_index('staff_id')['overtime_hourly_wage'].to_dict()
        self.skill_score = df_staff.set_index('staff_id')['skill_score'].to_dict()

        df_demand = pd.read_excel(self.args.data_file, sheet_name="staff_demand", header=0)

        if self.target_cluster:
            df_demand = df_demand[df_demand['branch_id'].isin(valid_branches)]

        day_map = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                   'Thursday': 4, 'Friday': 5, 'Saturday': 6}

        raw_abs_h = df_demand['day'].map(day_map) * 24 + df_demand['hour']
        df_demand['hour_of_week'] = (raw_abs_h - START_HOUR_OFFSET) % 168

        demand_series = df_demand.groupby('hour_of_week')['staff_demand'].sum()
        demand_dict = demand_series.to_dict()

        self.demand = {t: demand_dict.get(t, 0) for t in self.time_slots}
        self.operating_hours = [t for t, d in self.demand.items() if d > 0]

        avg_demand = np.mean(list(self.demand.values())) if self.demand.values() else 0
        self.is_peak = {t: 1 if self.demand[t] > avg_demand else 0 for t in self.time_slots}

    def _load_operation_hours(self):
        try:
            df_op = pd.read_excel(self.args.data_file, sheet_name="operation_hours", header=1)
            df_op.columns = [c.strip() for c in df_op.columns]

            def parse_h(t_str):
                return int(str(t_str).split(':')[0])

            day_map_rev = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

            for _, row in df_op.iterrows():
                branch = row['branch_id']
                if self.target_cluster and branch not in self.branch_cluster_map:
                    continue

                day_raw = row['week_day']
                if day_raw not in day_map_rev: continue
                day_idx = day_map_rev[day_raw]

                start_h = parse_h(row['start_time'])
                end_h = parse_h(row['end_time'])

                abs_start = day_idx * 24 + start_h
                abs_end = (day_idx + 1) * 24 + end_h if end_h <= start_h else day_idx * 24 + end_h

                t_first = (abs_start - START_HOUR_OFFSET) % 168
                t_last = (abs_end - 1 - START_HOUR_OFFSET) % 168

                if branch not in self.branch_operation_limits:
                    self.branch_operation_limits[branch] = {}
                self.branch_operation_limits[branch][day_idx] = {'first_hour': t_first, 'last_hour': t_last}
        except Exception as e:
            print(f"    Warning: Could not load operation hours ({e}).")

    def _load_bus_fleet_schedule(self):
        bus_file = getattr(self.args, 'bus_fleet_schedule', 'bus_fleet_schedule.csv')
        try:
            if os.path.exists(bus_file):
                df_bus = pd.read_csv(bus_file)
                for _, row in df_bus.iterrows():
                    t = int(row['time']) % 168
                    stops = ast.literal_eval(row['stops']) if isinstance(row['stops'], str) else row['stops']
                    for branch in stops:
                        if self.target_cluster and branch not in self.branch_cluster_map:
                            continue
                        if branch not in self.branch_valid_bus_times:
                            self.branch_valid_bus_times[branch] = set()
                        self.branch_valid_bus_times[branch].add(t)
            else:
                print(f"    Warning: Bus fleet schedule '{bus_file}' not found.")
        except Exception as e:
            print(f"    Error reading bus fleet schedule: {e}")

    def _generate_shifts(self):
        shift_id_counter = 0
        self.shifts = []
        for d in self.days:
            for start_h in range(24):
                abs_start = d * 24 + start_h
                for duration in range(self.args.min_shift_hours, self.args.max_shift_hours + 1):
                    if abs_start + duration > HOURS_PER_WEEK: continue
                    shift_id = f"S_{shift_id_counter}"
                    shift_obj = {
                        'id': shift_id, 'day': d, 'start': abs_start, 'end': abs_start + duration,
                        'duration': duration, 'covers': list(range(abs_start, abs_start + duration))
                    }
                    self.shifts.append(shift_obj)
                    self.shift_map[shift_id] = shift_obj
                    shift_id_counter += 1