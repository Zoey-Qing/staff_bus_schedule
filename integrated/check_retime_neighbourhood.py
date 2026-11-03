"""Check a re-timed schedule against the schedule it came from: each shift stays, moves its start by one hour, or changes its
length by one hour, and no shift appears or disappears.

Usage: python check_retime_neighbourhood.py <before.csv> <after.csv>, with columns Staff, Start, Duration.
"""
import sys

import pandas as pd


def by_staff(path):
    out = {}
    for r in pd.read_csv(path).itertuples():
        out.setdefault(r.Staff, []).append((int(r.Start), int(r.Duration)))
    return {i: sorted(v) for i, v in out.items()}


if __name__ == "__main__":
    before, after = by_staff(sys.argv[1]), by_staff(sys.argv[2])
    assert set(before) == set(after), "staff differ"
    moved = 0
    for i, shifts in before.items():
        assert len(shifts) == len(after[i]), (i, shifts, after[i])
        for (s0, d0), (s1, d1) in zip(shifts, after[i]):
            ds, dd = abs(s1 - s0), abs(d1 - d0)
            assert (ds, dd) in ((0, 0), (1, 0), (0, 1)), (i, s0, d0, s1, d1)
            moved += (ds, dd) != (0, 0)
    print("neighbourhood holds; shifts moved:", moved)
