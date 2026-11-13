"""Analyses of the finished runs, no new solves. Writes CSV files next to this script.

  retiming.csv          value of the final re-timing step for Benders and SG-LNS
  surrogate_quality.csv how well the transport-cost surrogate predicts and ranks the neighbours it generates
  seeds.csv             saving of SG-LNS and LNS with fixed prices per seed, mean, standard deviation and 95% interval
  gap_parts.csv         the Benders gap split into the master's own gap and the cut's underestimate of transport
  time_to_target.csv    time SG-LNS needs to reach the Benders solution (before re-timing)
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SETS = {"base": ("benders", "ablation_lns_fit", "ablation_lns", "surrogate"),
        "far2": ("far2/benders", "far2/lns_fit", None, "far2/surrogate"),
        "net100": ("net100/benders", "net100/lns_fit", None, None)}


def cluster_of(path):
    return os.path.basename(path).split("_s")[0].rsplit("_", 1)[-1][:8] if "_s" in os.path.basename(path)[-8:] \
        else os.path.basename(path)[:-5].rsplit("_", 1)[-1][:8]


def benders(folder):
    out = {}
    for f in glob.glob(os.path.join(ROOT, folder, "*.json")):
        d = json.load(open(f))
        if "z_seq" in d:
            out[d["cluster"]] = d
    return out


def runs(folder):
    out = {}
    if folder is None:
        return out
    for f in glob.glob(os.path.join(ROOT, folder, "*_s[0-9].json")):
        base = os.path.basename(f)[:-5]
        stem, seed = base.rsplit("_s", 1)
        out[stem.rsplit("_", 1)[-1][:8], int(seed)] = json.load(open(f))
    return out


def main():
    rt, sq, sd, gp, tt = [], [], [], [], []
    for name, (bdir, mdir, ldir, sdir) in SETS.items():
        bd, main_runs, lns_runs, sur_runs = benders(bdir), runs(mdir), runs(ldir), runs(sdir)
        for c, b in sorted(bd.items()):
            rt.append(dict(set=name, cluster=c, method="Benders", before=b["z_benders"], after=b["z"],
                           gain_pct=100 * (b["z_benders"] - b["z"]) / b["z_seq"]))
            last = [h for h in b["history"] if h.get("theta") is not None][-1]
            gp.append(dict(set=name, cluster=c, gap_pct=100 * b["gap"], rounds=b["rounds"],
                           master_gap_pct=100 * (last["master"] - b["lb"]) / b["z"],
                           cut_under_pct=100 * (last["transport"] - last["theta"]) / b["z"],
                           theta_share=last["theta"] / last["transport"]))
        for label, rr in (("SG-LNS", main_runs), ("LNS fixed prices", lns_runs), ("Full surrogate", sur_runs)):
            for (c, s), r in sorted(rr.items()):
                if c not in bd:
                    continue
                z0 = bd[c]["z_seq"]
                if label == "SG-LNS":
                    rt.append(dict(set=name, cluster=c, method=f"SG-LNS s{s}", before=r["z_search"], after=r["z"],
                                   gain_pct=100 * (r["z_search"] - r["z"]) / z0))
                dp, dr = np.array(r.get("dpred", [])), np.array(r.get("dreal", []))
                ok = len(dp) > 2 and dp.std() > 0 and dr.std() > 0
                sq.append(dict(set=name, cluster=c, seed=s, method=label, neighbours=len(dp),
                               pearson=float(np.corrcoef(dp, dr)[0, 1]) if ok else np.nan,
                               spearman=float(stats.spearmanr(dp, dr)[0]) if ok else np.nan,
                               sign_agree=float(np.mean(np.sign(dp) == np.sign(dr))) if len(dp) else np.nan,
                               real_lower_share=float(np.mean(dr < 0)) if len(dr) else np.nan,
                               pred_lower_share=float(np.mean(dp < 0)) if len(dp) else np.nan,
                               mae=float(np.mean(np.abs(dp - dr))) if len(dp) else np.nan))
                sd.append(dict(set=name, cluster=c, seed=s, method=label, saving_pct=100 * (z0 - r["z"]) / z0,
                               saving_search_pct=100 * (z0 - r["z_search"]) / z0))
                if label == "SG-LNS":
                    target = bd[c]["z_benders"]
                    hit = [h["seconds"] for h in r["history"] if h["best_j_op"] <= target + 1e-6]
                    tt.append(dict(set=name, cluster=c, seed=s, benders_seconds=bd[c]["benders_seconds"],
                                   seconds_to_benders=hit[0] if hit else np.nan, reached=bool(hit)))
    pd.DataFrame(rt).to_csv(os.path.join(HERE, "retiming.csv"), index=False)
    pd.DataFrame(sq).to_csv(os.path.join(HERE, "surrogate_quality.csv"), index=False)
    pd.DataFrame(gp).to_csv(os.path.join(HERE, "gap_parts.csv"), index=False)
    pd.DataFrame(tt).to_csv(os.path.join(HERE, "time_to_target.csv"), index=False)
    seeds = pd.DataFrame(sd)
    seeds.to_csv(os.path.join(HERE, "seeds.csv"), index=False)
    agg = []
    for (st, c, m), g in seeds.groupby(["set", "cluster", "method"]):
        v = g.saving_pct.to_numpy()
        half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        agg.append(dict(set=st, cluster=c, method=m, n=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                        ci_low=v.mean() - half, ci_high=v.mean() + half, min=v.min(), max=v.max()))
    pd.DataFrame(agg).to_csv(os.path.join(HERE, "seeds_summary.csv"), index=False)
    # paired by cluster and seed: SG-LNS minus LNS with fixed prices
    piv = seeds[seeds.set == "base"].pivot_table(index=["cluster", "seed"], columns="method", values="saving_pct")
    if {"SG-LNS", "LNS fixed prices"} <= set(piv.columns):
        d = (piv["SG-LNS"] - piv["LNS fixed prices"]).dropna()
        w = stats.wilcoxon(d) if len(d) > 5 else None
        pd.DataFrame([dict(pairs=len(d), mean_diff=d.mean(), sd=d.std(ddof=1), positive=int((d > 0).sum()),
                           wilcoxon_p=w.pvalue if w else np.nan, t_p=stats.ttest_1samp(d, 0).pvalue)]) \
            .to_csv(os.path.join(HERE, "paired_fit_vs_fixed.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
