"""Far x2 version of the original data: every accommodation moves away from the centroid of its cluster's branches, to
twice its distance. Only the accommodation coordinates change; the workbook is edited in place on a copy, so all other
sheets (operation hours included) stay as they are.

Usage: python data_gen/make_far2_original.py   (run from E:\schedule\integrated)
"""
from pathlib import Path

import openpyxl
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "staff_scheduling" / "Berkeley_Mock_Data_25_Stores_original_unbalanced.xlsx"
OUT = Path(__file__).resolve().parent / "Berkeley_Mock_Data_25_Stores_original_far2.xlsx"
FACTOR = 2.0

br = pd.read_excel(SRC, "branches")
centre = br.groupby(br.cluster_id.astype(str))[["latitude", "longitute"]].mean()
wb = openpyxl.load_workbook(SRC)
ws = wb["accommodation"]
col = {c.value: c.column for c in ws[1]}
for row in range(2, ws.max_row + 1):
    c = str(ws.cell(row, col["cluster_id"]).value)
    if c not in centre.index:
        continue
    lat, lon = ws.cell(row, col["latitude"]), ws.cell(row, col["longitude"])
    clat, clon = centre.loc[c, "latitude"], centre.loc[c, "longitute"]
    lat.value, lon.value = clat + FACTOR * (lat.value - clat), clon + FACTOR * (lon.value - clon)
wb.save(OUT)
print(OUT)
