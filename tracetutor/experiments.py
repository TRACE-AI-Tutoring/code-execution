"""All experiments reported in the paper. Each stage caches CSVs under results/."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from .calibrate import (MOMENTS, SPEC_BKT, calibrate, calibrate_global, load_params, save_params,
                        sim_moments)
from .data import load_study
from .policies import (TRACE, Control, FixedFading, GPTBase, GPTTutor, GreedyAssist, LambdaPenalty,
                       MasteryLadder, Overassist, TSBandit)
from .replicate import heterogeneity, problem_level, replicate_table1
from .simulator import ACTIONS, Cohort

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TEST_SEEDS = tuple(range(10))
METRICS = ["practice", "exam", "exam_exp", "nlg", "delayed", "overconf", "calib_err", "leak", "off",
           "r_end"] + [f"act_{a}" for a in ACTIONS]


def path(name):
    os.makedirs(RES, exist_ok=True)
    return os.path.join(RES, name)


def policies(Ph, seed=0, which="all"):
    pol = [Control(), GPTBase(), GPTTutor(), FixedFading(), MasteryLadder(Ph, seed), GreedyAssist(Ph, seed),
           TSBandit(Ph, seed), Overassist(Ph, seed), LambdaPenalty(Ph, 2.0, seed), TRACE(Ph, seed)]
    if which == "core":
        pol = [pol[i] for i in (0, 1, 2, 4, 5, 9)]
    return pol


def ablations(Ph, seed=0):
    return [TRACE(Ph, seed),
            TRACE(Ph, seed, use_r=False, name="TRACE - reliance state"),
            TRACE(Ph, seed, use_F=False, name="TRACE - frustration model"),
            TRACE(Ph, seed, use_c=False, name="TRACE - calibration term"),
            TRACE(Ph, seed, lookahead=False, name="TRACE - lookahead (myopic)"),
            TRACE(Ph, seed, use_r=False, use_F=False, use_c=False, lookahead=False,
                  name="TRACE - all (belief only)")]


def run_set(study, Ptrue, pols_fn, seeds, Ph=None, sessions=(1, 2, 3, 4), tag=None):
    Ph = Ptrue if Ph is None else Ph
    out = []
    for sd in seeds:
        C = Cohort(study, Ptrue, sd, sessions=sessions)
        for pol in pols_fn(Ph, sd):
            d = C.run(pol)
            if tag is not None:
                d["tag"] = tag
            out.append(d)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------------------------------
def stage_replication(study):
    t1 = replicate_table1(study.panel); t1.to_csv(path("real_table1.csv"), index=False)
    het = heterogeneity(study.panel); het.to_csv(path("real_heterogeneity.csv"), index=False)
    pl = problem_level(os.path.join(os.path.dirname(__file__), "..", "data", "GenAICanHarmLearning",
                                    "main_regressions"), study.items)
    pl.to_csv(path("real_problem_level.csv"), index=False)
    return t1


def stage_calibration(study, force=False):
    from .simulator import Params
    f = path("params_logistic.json")
    if os.path.exists(f) and not force:
        P = load_params(f)
    else:
        P, J = calibrate_global(study, Params(), maxiter=60, popsize=6, polish_iter=600, verbose=False)
        save_params(P, f)
    # fit on held-out simulation seeds
    rows = []
    for sd in range(20, 25):
        m = sim_moments(study, P, (sd,))
        rows += [dict(moment=k, seed=sd, sim=m[k], target=study.targets[k], se=study.target_se[k]) for k in MOMENTS]
    pd.DataFrame(rows).to_csv(path("calibration_fit.csv"), index=False)
    # out-of-sample validation: per-session, per-tercile, per-problem
    val = []
    for sd in range(20, 25):
        C = Cohort(study, P, sd)
        c, b, t = C.run(Control()), C.run(GPTBase()), C.run(GPTTutor())
        cut = np.quantile(c.gz, [1 / 3, 2 / 3])
        for d, nm in ((b, "GPTBase"), (t, "GPTTutor")):
            for col, real in (("practice", "Part2Tot"), ("exam", "Part3Tot")):
                for s in (1, 2, 3, 4):
                    val.append(dict(by="Session", level=str(s), outcome=real, term=nm, seed=sd,
                                    sim=d[d.session == s][col].mean() - c[c.session == s][col].mean()))
                for i, lev in enumerate(("low", "mid", "high")):
                    tc = np.digitize(c.gz, cut) == i
                    val.append(dict(by="tercile", level=lev, outcome=real, term=nm, seed=sd,
                                    sim=d[col][tc].mean() - c[col][tc].mean()))
    pd.DataFrame(val).to_csv(path("validation_sim.csv"), index=False)
    C = Cohort(study, P, 20)
    C.run(Control()); ic = C.item_df.copy()
    C.run(GPTBase()); ib = C.item_df.copy()
    ib["base_minus_ctrl"] = ib.practice - ic.practice
    ib.to_csv(path("validation_items_sim.csv"), index=False)
    return P


def stage_bkt(study, P, force=False):
    f = path("params_bkt.json")
    if os.path.exists(f) and not force:
        return load_params(f)
    base = P.copy(family="bkt")
    mom = [m for m in MOMENTS if "perceived" not in m]
    Pb, J, _ = calibrate(study, base, spec=SPEC_BKT, moments=mom, seeds=(0,), maxiter=700, verbose=False)
    save_params(Pb, f)
    return Pb


def summarise(df, by=("policy",)):
    by = list(by)
    per_seed = df.groupby(by + ["seed"])[METRICS].mean().reset_index()
    agg = per_seed.groupby(by)[METRICS].agg(["mean", "std", "count"])
    return per_seed, agg


def stage_main(study, P):
    df = run_set(study, P, policies, TEST_SEEDS)
    df.to_parquet(path("main_students.parquet")) if _has_pq() else df.to_csv(path("main_students.csv.gz"), index=False)
    per_seed = df.groupby(["policy", "seed"])[METRICS].mean().reset_index()
    per_seed.to_csv(path("main_per_seed.csv"), index=False)
    # student-level paired contrasts (pooled over seeds) with a student cluster bootstrap
    rng = np.random.default_rng(0)
    ctrl = df[df.policy == "Control (no AI)"].set_index(["seed", "sid", "session"])
    trace = df[df.policy == "TRACE (ours)"].set_index(["seed", "sid", "session"])
    sids = df.sid.unique()
    rows = []
    for pol, d in df.groupby("policy"):
        d = d.set_index(["seed", "sid", "session"])
        for ref_name, ref in (("Control (no AI)", ctrl), ("TRACE (ours)", trace)):
            diff = (d["exam_exp"] - ref["exam_exp"]).groupby(level="sid").mean()
            boots = [diff.loc[rng.choice(sids, len(sids))].mean() for _ in range(1000)]
            ps = per_seed[per_seed.policy == pol].set_index("seed")["exam_exp"]
            pr = per_seed[per_seed.policy == ref_name].set_index("seed")["exam_exp"]
            tt = stats.ttest_rel(ps, pr) if pol != ref_name else None
            rows.append(dict(policy=pol, ref=ref_name, diff=diff.mean(), lo=np.quantile(boots, .025),
                             hi=np.quantile(boots, .975),
                             d_cohen=diff.mean() / ref["exam_exp"].std(),
                             p_seed=np.nan if tt is None else tt.pvalue))
    con = pd.DataFrame(rows)
    # Holm correction within each reference family
    for ref in con.ref.unique():
        m = (con.ref == ref) & con.p_seed.notna()
        p = con.loc[m, "p_seed"].to_numpy(); order = np.argsort(p); k = len(p)
        adj = np.empty(k); run = 0
        for i, j in enumerate(order):
            run = max(run, min(1, (k - i) * p[j])); adj[j] = run
        con.loc[m, "p_holm"] = adj
    con.to_csv(path("main_contrasts.csv"), index=False)
    # equity: effect vs control by prior-GPA tercile
    cut = np.quantile(study.students.gpa_z, [1 / 3, 2 / 3])
    df["tercile"] = pd.Categorical(np.array(["low", "mid", "high"])[np.digitize(df.gz, cut)],
                                   ["low", "mid", "high"])
    eq = df.groupby(["policy", "seed", "tercile"], observed=True)[["exam_exp", "overconf", "practice"]].mean().reset_index()
    eq.to_csv(path("equity.csv"), index=False)
    # adaptivity: TRACE action mix by tercile and session
    ad = df[df.policy == "TRACE (ours)"].groupby(["tercile", "session"], observed=True)[
        [f"act_{a}" for a in ACTIONS]].mean().reset_index()
    ad.to_csv(path("trace_actions.csv"), index=False)
    # dynamics over sessions
    dyn = df.groupby(["policy", "seed", "session"])[["exam_exp", "practice", "r_end", "overconf", "off"]].mean().reset_index()
    dyn.to_csv(path("dynamics_sessions.csv"), index=False)
    return per_seed, con


def stage_ablation(study, P):
    df = run_set(study, P, ablations, TEST_SEEDS)
    per = df.groupby(["policy", "seed"])[METRICS].mean().reset_index()
    c = run_set(study, P, lambda Ph, sd: [Control()], TEST_SEEDS).groupby("seed")[METRICS].mean()
    per["d_exam"] = per.apply(lambda r: r.exam_exp - c.loc[r.seed, "exam_exp"], axis=1)
    per.to_csv(path("ablation_per_seed.csv"), index=False)
    return per


def stage_pareto(study, P):
    def fn(Ph, sd):
        pol = [Control(), GreedyAssist(Ph, sd), GPTTutor(), MasteryLadder(Ph, sd)]
        pol += [TRACE(Ph, sd, w_c=w, name=f"TRACE w_c={w:g}") for w in (0.0, 0.1, 0.25, 0.5, 1.0)]
        pol += [LambdaPenalty(Ph, lam, sd) for lam in (0.25, 0.5, 1.0, 2.0)]
        return pol
    df = run_set(study, P, fn, (0, 1, 2))
    per = df.groupby(["policy", "seed"])[METRICS].mean().reset_index()
    per.to_csv(path("pareto_per_seed.csv"), index=False)
    return per


PERTURB = ["eta", "rho_g", "k_up", "k_dn", "k_ref", "w1", "pf", "h_scale", "c_min", "omega",
           "lam_c", "zpd_amp", "mu_r0", "L_proc"]


def stage_robust(study, P, Pb, n_draws=30):
    rng = np.random.default_rng(2026)
    out = []
    for i in range(n_draws):
        kw = {}
        for k in PERTURB:
            v = getattr(P, k) * float(np.exp(rng.normal(0, 0.30)))
            if k in ("rho_g", "c_min", "mu_r0"):
                v = float(np.clip(v, 0.02, 0.95))
            kw[k] = v
        Pt = P.copy(**kw)
        d = run_set(study, Pt, policies, (i,), Ph=P, tag=f"perturb_{i}")
        out.append(d.groupby(["policy", "tag"])[METRICS].mean().reset_index())
    for sd in range(5):
        d = run_set(study, Pb, policies, (sd,), Ph=P, tag=f"bkt_{sd}")
        out.append(d.groupby(["policy", "tag"])[METRICS].mean().reset_index())
    r = pd.concat(out, ignore_index=True)
    r.to_csv(path("robustness.csv"), index=False)
    return r


def stage_ai_accuracy(study, P):
    out = []
    for q in (0.51, 0.75, 0.90, 0.97):
        Pq = P.copy(q_ground=q)
        d = run_set(study, Pq, lambda Ph, sd: policies(Ph, sd, "core"), (0, 1, 2), tag=f"q={q}")
        d["q"] = q
        out.append(d.groupby(["policy", "seed", "q"])[METRICS].mean().reset_index())
    r = pd.concat(out); r.to_csv(path("ai_accuracy.csv"), index=False)
    return r


def stage_horizon(study, P, n_sessions=12):
    d = run_set(study, P, lambda Ph, sd: policies(Ph, sd, "core"), (0, 1, 2),
                sessions=tuple(range(1, n_sessions + 1)))
    r = d.groupby(["policy", "seed", "session"])[METRICS].mean().reset_index()
    r.to_csv(path("horizon.csv"), index=False)
    return r


def stage_timing(study, P):
    C = Cohort(study, P, 0)
    rows = []
    for pol in policies(P, 0):
        t = time.perf_counter(); C.run(pol); dt = time.perf_counter() - t
        n_dec = C.N * sum(len(study.items[(s, 9)]["b_prac"]) for s in (1, 2, 3, 4)) * P.T_steps
        rows.append(dict(policy=pol.name, seconds=dt, us_per_student_decision=1e6 * dt / n_dec))
    r = pd.DataFrame(rows); r.to_csv(path("timing.csv"), index=False)
    return r


def _has_pq():
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False
