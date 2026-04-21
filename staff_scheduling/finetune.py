import sys
import os
import pandas as pd
from args import get_args
from data_loader import DataLoader
from solver_cplex import StaffSchedulingSolver
from evaluation import run_evaluation
from visualization import generate_visualizations


def main():
    args = get_args()

    # 强制在微调模式下开启公交车约束
    args.enforce_bus = True

    print("=== Finetuning Staff Schedule Based on Partner's Bus Fleet ===")

    # 1. 加载基础数据
    loader = DataLoader(args)
    loader.load_and_preprocess()

    # 你的同伴给你的文件，请确保将其重命名为 partner_schedule.csv
    partner_file = 'partner_schedule.csv'

    if not os.path.exists(partner_file):
        print(f"Error: 请将同伴提供的排班文件重命名为 '{partner_file}' 并放在当前目录下。")
        sys.exit(1)

    # 2. 提取基准数据 (Baseline Assignments)
    baseline_assignments = set()
    try:
        df_partner = pd.read_csv(partner_file)
        # 提取同伴的原始排班作为微调基准 (Baseline)
        for _, row in df_partner.iterrows():
            baseline_assignments.add((row['Staff'], row['Shift']))
        print(f">>> 提取到 {len(baseline_assignments)} 个同伴排好的班次作为 Baseline 锚点。")

    except Exception as e:
        print(f"Error reading {partner_file}: {e}")
        sys.exit(1)

    # 3. 拦截更新 DataLoader 生成的 shifts 标志 (可选：如果你在 data_loader 里提取了 valid_bus_times，这一步可以省略，但这里做二次保险)
    valid_bus_times = loader.valid_bus_times
    for s in loader.shifts:
        t_start = s['start'] % 168
        t_end = s['end'] % 168
        s['bus_compatible'] = (t_start in valid_bus_times) and (t_end in valid_bus_times)

    # 4. 使用统一的 Solver 进行微调求解
    # 重点：通过传入 baseline_assignments 激活 solver 内部的微调模式
    solver = StaffSchedulingSolver(args, loader, baseline_assignments=baseline_assignments)
    solver.build_model()
    solution = solver.solve()

    if solution:
        print(f"Objective Value: {solution.objective_value:.2f}")
        df_res = solver.export_results(solution)

        # 导出微调后的结果
        finetune_out = "finetuned_" + args.schedule_file
        df_res.to_csv(finetune_out, index=False)
        print(f">>> 微调后的排班已保存至: {finetune_out}")

        # 更新报告和图片
        args.schedule_file = finetune_out
        args.report_file = "finetuned_" + args.report_file
        run_evaluation(args, df_res, loader)
        generate_visualizations(args)
    else:
        print(">>> Optimization failed.")


if __name__ == "__main__":
    main()