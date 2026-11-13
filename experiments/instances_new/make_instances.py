"""New test instances (2026-11-11). Base: the original IntelMatrix workbook without cluster 662b3495 (not used in the
paper), so that no generated branch copies its data. Generator: integrated/generate_network.py (unchanged).
  T1-T3  tiny instances with one branch and one accommodation (seeds 1-3), for the exact check of the bounds;
  G1     3 branches, 1 accommodation           G2  8 branches, 2 accommodations of 4 branches
  G3     12 branches, 2 accommodations of 6    G4  10 branches, 1 accommodation
  G5     8 branches, 2 accommodations of 4, branch-to-accommodation distances x2
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "E:/schedule/integrated")
import generate_network as G  # noqa: E402

SRC = "E:/schedule/staff_scheduling/Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx"
OUT = Path("E:/schedule/revision/instances_new")
DROP = "662b3495"


def filtered_base():
    x = pd.ExcelFile(SRC)
    br = x.parse("branches")
    drop_br = set(br.branch_id[br.cluster_id.astype(str).str.startswith(DROP)])
    sheets = {}
    for s in x.sheet_names:
        if s == "operation_hours":
            continue
        df = x.parse(s)
        if "cluster_id" in df.columns:
            df = df[~df.cluster_id.astype(str).str.startswith(DROP)]
        if "branch_id" in df.columns:
            df = df[~df.branch_id.isin(drop_br)]
        if s == "clusters":
            df = df[~df.id.astype(str).str.startswith(DROP)]
        sheets[s] = df
    ops = x.parse("operation_hours", header=1)
    ops = ops[~ops.branch_id.isin(drop_br)]
    path = OUT / "base_without_662b3495.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame([["Week day goes from 1 (Sunday) to 7 (Saturday)", None, None, None]]) \
            .to_excel(w, sheet_name="operation_hours", index=False, header=False)
        ops.to_excel(w, sheet_name="operation_hours", index=False, startrow=1)
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    return path


SPECS = {  # name: branches, per accommodation, accommodations per cluster, distance scale, seed
    "T1": (1, 1, 1, 1.0, 1), "T2": (1, 1, 1, 1.0, 2), "T3": (1, 1, 1, 1.0, 3),
    "G1": (3, 3, 1, 1.0, 1), "G2": (8, 4, 2, 1.0, 1), "G3": (12, 6, 2, 1.0, 1), "G4": (10, 10, 1, 1.0, 1),
    "G5": (8, 4, 2, 2.0, 1),
}

if __name__ == "__main__":
    base = filtered_base()
    for name, (n, k, a, dscale, seed) in SPECS.items():
        sheets, ops, sched, tpl = G.generate(n, k, a, 2.0, seed, base_xlsx=base, distance_scale=dscale)
        G.write(OUT / name, sheets, ops, sched, tpl)
        st = sheets["staff"]
        riders = (~st.Nationality.astype(str).str.lower().isin(["saudi", "saudi arabia"])).sum()
        print(name, "branches", n, "accommodations", len(sheets["accommodation"]), "staff", len(st), "riders", riders,
              "cluster", sheets["clusters"].id.iloc[0][:8], "templates", ",".join(tpl.template_branch))
