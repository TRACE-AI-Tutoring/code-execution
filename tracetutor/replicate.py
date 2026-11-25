"""Re-estimate Bastani et al. Table 1 (and perception / calibration gaps) from the public data,
in pure numpy: OLS with teacher, session, grader and grade fixed effects + prior GPA,
CR1 standard errors clustered by classroom.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ols_cluster(y, X, clusters):
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    meat = np.zeros((k, k))
    groups = pd.factorize(clusters)[0]
    G = groups.max() + 1
    for g in range(G):
        idx = groups == g
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtX_inv @ meat @ XtX_inv
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return beta, np.sqrt(np.diag(V)), r2


def design(df):
    X = pd.DataFrame({"const": 1.0, "GPTBase": df["GPTBase"].astype(float),
                      "GPTTutor": df["GPTTutor"].astype(float), "gpa_prev": df["gpa_prev"]})
    for fe in ("teacher", "Session", "Grader", "Year"):
        X = pd.concat([X, pd.get_dummies(df[fe].astype(str), prefix=fe, drop_first=True, dtype=float)], axis=1)
    return X


def replicate_table1(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = panel.copy()
    df["gap"] = df["perc_perf"] - df["Part3Tot"]          # overconfidence: perceived - actual
    df["abs_gap"] = df["gap"].abs()
    for outcome, label in (("Part2Tot", "Practice (assisted)"), ("Part3Tot", "Exam (unassisted)"),
                           ("perc_perf", "Perceived exam performance"),
                           ("gap", "Overconfidence (perceived - exam)"),
                           ("abs_gap", "Calibration error |perceived - exam|")):
        d = df.dropna(subset=[outcome])
        X = design(d)
        b, se, r2 = ols_cluster(d[outcome].to_numpy(float), X.to_numpy(float), d["Class"].to_numpy())
        cm = d.loc[d["Treatment arm"] == "control", outcome].mean()
        names = list(X.columns)
        for v in ("GPTBase", "GPTTutor", "gpa_prev"):
            j = names.index(v)
            rows.append(dict(outcome=label, term=v, coef=b[j], se=se[j], t=b[j] / se[j],
                             control_mean=cm, rel=b[j] / cm if v != "gpa_prev" else np.nan,
                             n=len(d), r2=r2))
    return pd.DataFrame(rows)


def heterogeneity(panel: pd.DataFrame) -> pd.DataFrame:
    """Arm effects estimated separately by session and by prior-GPA tercile (not used for
    calibration -> out-of-sample validation targets for the simulator)."""
    rows = []
    df = panel.copy()
    df["tercile"] = pd.qcut(df["gpa_prev"], 3, labels=["low", "mid", "high"])
    for by in ("Session", "tercile"):
        for lev, d in df.groupby(by, observed=True):
            for outcome in ("Part2Tot", "Part3Tot"):
                X = design(d)
                X = X.loc[:, X.std() > 0].copy(); X["const"] = 1.0
                b, se, _ = ols_cluster(d[outcome].to_numpy(float), X.to_numpy(float), d["Class"].to_numpy())
                names = list(X.columns)
                for v in ("GPTBase", "GPTTutor"):
                    j = names.index(v)
                    rows.append(dict(by=by, level=str(lev), outcome=outcome, term=v, coef=b[j], se=se[j]))
    return pd.DataFrame(rows)


def problem_level(repo_mr: str, items: dict) -> pd.DataFrame:
    """Per-problem practice success of GPT Base minus Control, against audited GPT-4 accuracy."""
    import os
    p2 = pd.read_csv(os.path.join(repo_mr, "problem_part2.csv"))
    p2 = p2[(p2["Honors"] == 0)]
    m = p2.groupby(["Problem", "Treatment arm"])["Score"].mean().unstack()
    q = {pid: qq for d in items.values() for pid, qq in zip(d["ids"], d["q_base"])}
    m["q_base"] = m.index.map(q)
    m["base_minus_ctrl"] = m["vanilla"] - m["control"]
    m["tutor_minus_ctrl"] = m["augmented"] - m["control"]
    return m.reset_index()
