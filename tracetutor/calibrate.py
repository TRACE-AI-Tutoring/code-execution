"""Method-of-simulated-moments calibration of the learner simulator to Bastani et al. (2025).

The three arms of the field experiment (Control, GPT Base, GPT Tutor) are re-created in silico on
the real student roster (grade, prior GPA) and item bank (control-arm item difficulties, audited
GPT-4 accuracy per problem). Free parameters are chosen to minimise the inverse-SE-weighted
distance to 12 empirical moments; common random numbers keep the objective smooth.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize, differential_evolution

from .policies import Control, GPTBase, GPTTutor
from .simulator import Cohort, Params

SPEC_LOGISTIC = {   # name: (low, high)
    "a0": (-3.0, 2.0), "a1": (0.0, 1.5), "d_exam": (-2.5, 2.5), "eta": (0.02, 1.5),
    "h_scale": (0.3, 1.6), "mu_r0": (0.2, 0.95), "k_ref": (0.0, 0.6),
    "c0": (0.05, 0.8), "lam_c": (0.0, 0.5), "c0_gpa": (0.0, 0.3),
    "c_min": (0.0, 0.95), "L_proc": (0.0, 1.0), "pf": (0.0, 0.9), "psi": (0.3, 1.0),
}
SPEC_BKT = {
    "a0": (-3.0, 2.0), "a1": (0.0, 1.5), "bkt_T": (0.01, 0.9), "h_scale": (0.3, 1.6),
    "bkt_guess": (0.02, 0.4), "bkt_slip": (0.02, 0.4),
}
MOMENTS = ["ctrl_prac", "ctrl_exam", "base_prac_eff", "tutor_prac_eff", "base_exam_eff",
           "tutor_exam_eff", "exam_gpa_slope", "base_answer_share_s1", "tutor_answer_share_s1",
           "ctrl_perceived", "base_perceived_eff", "tutor_perceived_eff"]


def sim_moments(study, P, seeds=(0,)):
    out = []
    for sd in seeds:
        C = Cohort(study, P, sd)
        c, b, t = C.run(Control()), C.run(GPTBase()), C.run(GPTTutor())
        gpa = study.gpa_mean + study.gpa_sd * c.gz
        slope = np.polyfit(gpa, c.exam, 1)[0]
        m = dict(ctrl_prac=c.practice.mean(), ctrl_exam=c.exam.mean(),
                 base_prac_eff=b.practice.mean() - c.practice.mean(),
                 tutor_prac_eff=t.practice.mean() - c.practice.mean(),
                 base_exam_eff=b.exam.mean() - c.exam.mean(),
                 tutor_exam_eff=t.exam.mean() - c.exam.mean(),
                 exam_gpa_slope=slope,
                 base_answer_share_s1=b[b.session == 1].ans_req.mean(),
                 tutor_answer_share_s1=t[t.session == 1].ans_req.mean(),
                 ctrl_perceived=c.perceived.mean(),
                 base_perceived_eff=b.perceived.mean() - c.perceived.mean(),
                 tutor_perceived_eff=t.perceived.mean() - c.perceived.mean())
        out.append(m)
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def _to_params(x, spec, base):
    kw = {}
    for xi, (k, (lo, hi)) in zip(x, spec.items()):
        kw[k] = lo + (hi - lo) / (1 + np.exp(-xi))
    return base.copy(**kw)


def _from_params(P, spec):
    x = []
    for k, (lo, hi) in spec.items():
        v = np.clip((getattr(P, k) - lo) / (hi - lo), 1e-3, 1 - 1e-3)
        x.append(np.log(v / (1 - v)))
    return np.array(x)


def calibrate(study, base: Params, spec=SPEC_LOGISTIC, moments=MOMENTS, seeds=(0, 1),
              maxiter=1200, verbose=True):
    tgt = np.array([study.targets[m] for m in moments])
    se = np.array([study.target_se[m] for m in moments])
    hist = []

    def obj(x):
        P = _to_params(x, spec, base)
        m = sim_moments(study, P, seeds)
        z = (np.array([m[k] for k in moments]) - tgt) / se
        f = float(z @ z)
        hist.append(f)
        if verbose and len(hist) % 100 == 0:
            print(f"  iter {len(hist):5d}  J = {min(hist):.3f}", flush=True)
        return f

    x0 = _from_params(base, spec)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options=dict(maxiter=maxiter, maxfev=maxiter, xatol=1e-3, fatol=1e-3, adaptive=True))
    # polish
    res = minimize(obj, res.x, method="Nelder-Mead",
                   options=dict(maxiter=maxiter // 2, maxfev=maxiter // 2, xatol=1e-4, fatol=1e-4, adaptive=True))
    P = _to_params(res.x, spec, base)
    return P, res.fun, hist


def save_params(P, path):
    with open(path, "w") as f:
        json.dump(P.as_dict(), f, indent=2)


def load_params(path):
    with open(path) as f:
        d = json.load(f)
    for k in ("L_succ", "L_fail", "shallow", "H"):
        d[k] = tuple(d[k])
    return Params(**d)


# ---------------- global search (differential evolution on the unit cube) ----------------
_G = {}


def _de_obj(z):
    study, base, spec, moments, seeds, tgt, se = (_G[k] for k in
                                                  ("study", "base", "spec", "moments", "seeds", "tgt", "se"))
    kw = {k: lo + (hi - lo) * zi for zi, (k, (lo, hi)) in zip(z, spec.items())}
    m = sim_moments(study, base.copy(**kw), seeds)
    d = (np.array([m[k] for k in moments]) - tgt) / se
    return float(d @ d)


def calibrate_global(study, base: Params, spec=SPEC_LOGISTIC, moments=MOMENTS, seeds=(0,),
                     maxiter=30, popsize=8, seed=0, polish_iter=800, verbose=True):
    _G.update(study=study, base=base, spec=spec, moments=moments, seeds=seeds,
              tgt=np.array([study.targets[m] for m in moments]),
              se=np.array([study.target_se[m] for m in moments]))
    x0 = np.array([np.clip((getattr(base, k) - lo) / (hi - lo), 0, 1) for k, (lo, hi) in spec.items()])
    res = differential_evolution(_de_obj, [(0, 1)] * len(spec), maxiter=maxiter, popsize=popsize,
                                 seed=seed, x0=x0, polish=False, tol=1e-6, updating="deferred",
                                 workers=1, disp=verbose)
    kw = {k: lo + (hi - lo) * zi for zi, (k, (lo, hi)) in zip(res.x, spec.items())}
    P = base.copy(**kw)
    P, J, hist = calibrate(study, P, spec, moments, seeds=(0, 1), maxiter=polish_iter, verbose=verbose)
    return P, J
