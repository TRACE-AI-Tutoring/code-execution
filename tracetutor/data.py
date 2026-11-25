"""Data loading and calibration targets from the Bastani et al. (PNAS 2025) replication package.

The replication package (https://github.com/obastani/GenAICanHarmLearning) is NOT redistributed
here (it carries no licence). Clone it and point TRACE_REPO (env var) or --repo to the clone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_REPO = os.environ.get(
    "TRACE_REPO",
    os.path.join(os.path.dirname(__file__), "..", "data", "GenAICanHarmLearning"),
)

PERF_MAP = {
    "very poorly": 0.0, "below average": 0.25, "average": 0.5,
    "above average": 0.75, "excellent": 1.0,
}


def _logit(p):
    p = np.clip(p, 0.03, 0.97)
    return np.log(p / (1 - p))


@dataclass
class StudyData:
    students: pd.DataFrame            # one row per student: id, grade, gpa, gpa_z
    items: dict                       # (session, grade) -> dict(b_prac, b_exam, q_base, ids)
    targets: dict                     # calibration moments
    target_se: dict                   # their standard errors (for MSM weighting)
    panel: pd.DataFrame               # student-session panel (non-honors)
    gpa_mean: float = 0.0
    gpa_sd: float = 1.0
    extra: dict = field(default_factory=dict)


def load_study(repo: str = DEFAULT_REPO) -> StudyData:
    repo = os.path.abspath(repo)
    mr = os.path.join(repo, "main_regressions")
    if not os.path.isdir(mr):
        raise FileNotFoundError(
            f"Replication package not found at {repo}.\n"
            "Run:  git clone https://github.com/obastani/GenAICanHarmLearning data/GenAICanHarmLearning\n"
            "or set the TRACE_REPO environment variable / --repo argument.")
    df = pd.read_csv(os.path.join(mr, "final_data.csv"))
    df = df[df["Honors"] == 0].copy()
    df = df[df["gpa_prev"].notna()].copy()        # paper's estimation sample (N = 2,848)

    # perception (perceived exam performance on a [0,1] scale)
    perc = pd.read_stata(os.path.join(repo, "additional_results", "df_perception.dta"))
    perc["perc_perf"] = perc["perceived_performance"].str.strip().str.lower().map(PERF_MAP)
    perc = perc[["Student_ID", "Session", "perc_perf"]].rename(columns={"Student_ID": "Student ID"})
    df = df.merge(perc, on=["Student ID", "Session"], how="left")

    students = (df.groupby("Student ID").agg(grade=("Year", "first"), gpa=("gpa_prev", "first"),
                                             arm=("Treatment arm", "first"), cls=("Class", "first"))
                .reset_index())
    gm, gs = students.gpa.mean(), students.gpa.std()
    students["gpa_z"] = (students.gpa - gm) / gs

    # ---- item bank: difficulty from control-arm problem-level success; GPT accuracy from audit ----
    p2 = pd.read_csv(os.path.join(mr, "problem_part2.csv"))
    p3 = pd.read_csv(os.path.join(mr, "problem_part3.csv"))
    gpt = pd.read_csv(os.path.join(mr, "gpt_answers_full.csv"))
    gpt.columns = [c.replace("﻿", "") for c in gpt.columns]
    mp = pd.read_csv(os.path.join(mr, "problem_mapping.csv"))
    c2 = p2[(p2["Treatment arm"] == "control") & (p2["Honors"] == 0)].groupby("Problem")["Score"].mean()
    c3 = p3[(p3["Treatment arm"] == "control") & (p3["Honors"] == 0)].groupby("Problem")["Score"].mean()
    qmap = dict(zip(gpt["problem"], gpt["total_correct"] / 10.0))
    items = {}
    for _, row in mp.iterrows():
        pid, eid = row["part2"], row["part3"]
        s, g, _k = pid.split("_")
        key = (int(s[1:]), int(g))
        d = items.setdefault(key, {"ids": [], "b_prac": [], "b_exam": [], "q_base": []})
        d["ids"].append(pid)
        d["b_prac"].append(-_logit(c2.get(pid, c2.mean())))
        d["b_exam"].append(-_logit(c3.get(eid, c3.mean())))
        d["q_base"].append(qmap.get(pid, 0.51))
    for d in items.values():
        for k in ("b_prac", "b_exam", "q_base"):
            d[k] = np.asarray(d[k], float)

    # ---- calibration targets: published Table 1 regression-adjusted effects + moments ----
    t = {
        "ctrl_prac": 0.284, "ctrl_exam": 0.321,
        "base_prac_eff": 0.137, "tutor_prac_eff": 0.361,
        "base_exam_eff": -0.054, "tutor_exam_eff": -0.004,
        "exam_gpa_slope": 1.334,             # per GPA unit (Table 1)
        "base_answer_share_s1": 0.67,        # first-interaction answer seeking, session 1
        "tutor_answer_share_s1": 0.37,
    }
    se = {
        "ctrl_prac": 0.287 / np.sqrt(1076), "ctrl_exam": 0.277 / np.sqrt(1076),
        "base_prac_eff": 0.031, "tutor_prac_eff": 0.032,
        "base_exam_eff": 0.022, "tutor_exam_eff": 0.013,
        "exam_gpa_slope": 0.15, "base_answer_share_s1": 0.04, "tutor_answer_share_s1": 0.04,
    }
    ctrl = df[df["Treatment arm"] == "control"]
    t["ctrl_perceived"] = float(ctrl["perc_perf"].mean())
    se["ctrl_perceived"] = float(ctrl["perc_perf"].std() / np.sqrt(ctrl["perc_perf"].notna().sum()))
    # regression-adjusted perception effects (same specification as Table 1; see replicate.py)
    from .replicate import replicate_table1
    rep = replicate_table1(df)
    print("\n--- SAMPLE SIZE CHECK ---")
    print("Total panel rows:", len(df))
    print("Unique students:", df["Student ID"].nunique())
    print(
    "Unique student-session pairs:",
    df[["Student ID", "Session"]].drop_duplicates().shape[0]
)
    print(
    "Duplicate student-session rows:",
    df.duplicated(["Student ID", "Session"]).sum()
)

    print("\nObservations per outcome:")
    print(rep.groupby("outcome")["n"].first())




    for term, nm in (("GPTBase", "base"), ("GPTTutor", "tutor")):
        r = rep[(rep.outcome == "Perceived exam performance") & (rep.term == term)].iloc[0]
        t[f"{nm}_perceived_eff"], se[f"{nm}_perceived_eff"] = float(r.coef), float(r.se)
    return StudyData(students=students, items=items, targets=t, target_se=se, panel=df,
                     gpa_mean=gm, gpa_sd=gs)
