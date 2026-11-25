"""Learner simulator calibrated to Bastani et al. (PNAS 2025).

Two ground-truth families are implemented so that policies can be stress-tested under
model misspecification:
  * 'logistic'  - continuous logit-skill learner (IRT response model) with assistance-dependent
                  learning, zone-of-proximal-development weighting, expertise reversal for worked
                  examples, reliance-driven help abuse, frustration/off-task dynamics, and
                  self-assessment (confidence) dynamics with credit misattribution.
  * 'bkt'       - Bayesian-knowledge-tracing learner (binary mastery, guess/slip) with the same
                  reliance, frustration and confidence mechanics.

Help levels (actions):
  0 NONE     attempt alone (textbook/notes)          3 STEPS   step-by-step hints
  1 REFLECT  predict-confidence-then-attempt          4 WORKED  worked example on an isomorphic item
  2 HINT     conceptual hint                          5 ANSWER  full solution
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict

import numpy as np
import pandas as pd

ACTIONS = ["none", "reflect", "hint", "steps", "worked", "answer"]
NA = len(ACTIONS)
LEAK = np.array([0.0, 0.0, 0.0, 0.2, 0.5, 1.0])          # solution-leakage weight of each level


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Params:
    family: str = "logistic"
    # prior skill:  theta0 = a0 + a1 * gpa_z + u_i + eps_ik
    a0: float = 0.0
    a1: float = 0.55
    sig_u: float = 0.5
    sig_e: float = 0.5
    d_exam: float = 0.0              # exam difficulty offset
    # learning
    eta: float = 0.30
    zpd_lo: float = 0.4
    zpd_amp: float = 2.4
    L_succ: tuple = (1.00, 1.00, 0.85, 0.55, 0.0, 0.0)
    L_fail: tuple = (0.35, 0.45, 0.40, 0.30, 0.0, 0.0)
    pf: float = 0.35                 # productive-failure learning when an unaided attempt fails
    w0: float = 0.25                 # worked example: w0 + w1 * (1 - p0)   (expertise reversal)
    w1: float = 0.95
    L_proc: float = 0.40             # learning from a full solution that is processed, not copied
    c_min: float = 0.30              # copy prob = c_min + (1 - c_min) * reliance
    rho_g: float = 0.60              # help abuse: learning from help scaled by (1 - rho_g * r)
    shallow: tuple = (0.10, 0.10, 0.20, 0.40, 0.35, 0.80)
    # immediate help strength
    H: tuple = (0.0, 0.0, 0.35, 0.60, 0.50, 1.0)
    h_scale: float = 1.0
    q_ground: float = 0.97           # accuracy of a solution-grounded tutor
    # reliance dynamics
    mu_r0: float = 0.6
    conc_r0: float = 6.0
    k_up: float = 0.10
    k_dn: float = 0.03
    k_meta: float = 0.08
    k_ref: float = 0.15
    psi: float = 0.8                 # uptake: P(student asks the assistant for help on a step)
    req_p3: float = 0.6              # non-answer requests: steps w.p. req_p3 else hint
    # frustration / off-task
    off_base: float = 0.03
    omega: float = 0.5
    gam: float = 1.5
    F0: float = 2.0
    # self-assessment
    c0: float = 0.35
    c0_gpa: float = 0.08
    lam_c: float = 0.15
    lam_m: float = 0.50
    # retention
    decay_deep: float = 0.10
    decay_shallow: float = 1.00
    # BKT family
    bkt_T: float = 0.30
    bkt_guess: float = 0.15
    bkt_slip: float = 0.10
    T_steps: int = 3

    def copy(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)


class Cohort:
    """Vectorised classroom simulator over all students; one policy per run."""

    def __init__(self, study, P: Params, seed: int, sessions=(1, 2, 3, 4)):
        self.S, self.P, self.seed, self.sessions = study, P, seed, sessions
        st = study.students
        self.N = len(st)
        self.gz = st.gpa_z.to_numpy(float)
        self.grade = st.grade.to_numpy(int)
        self.sid = st["Student ID"].to_numpy()
        rs = np.random.default_rng(10_000 + seed)
        self.u = rs.normal(0, P.sig_u, self.N)
        a, b = P.mu_r0 * P.conc_r0, (1 - P.mu_r0) * P.conc_r0
        self.r0 = rs.beta(a, b, self.N)
        # pre-draw initial skills per session (common random numbers across policies)
        self.eps = {s: rs.normal(0, P.sig_e, (self.N, 5)) for s in sessions}
        self.minit = {s: rs.random((self.N, 5)) for s in sessions}
        self.exam_u = {s: rs.random((self.N, 5)) for s in sessions}

    def _items(self, s):
        Kmax = 5
        bp = np.zeros((self.N, Kmax)); be = np.zeros((self.N, Kmax)); qb = np.zeros((self.N, Kmax))
        mask = np.zeros((self.N, Kmax), bool)
        for g in np.unique(self.grade):
            it = self.S.items[(((s - 1) % 4) + 1, int(g))]       # sessions > 4 recycle topics
            K = len(it["b_prac"]); idx = self.grade == g
            bp[idx, :K] = it["b_prac"]; be[idx, :K] = it["b_exam"]; qb[idx, :K] = it["q_base"]
            mask[idx, :K] = True
        return bp, be, qb, mask

    def run(self, policy, record_steps=False):
        P, N = self.P, self.N
        rng = np.random.default_rng(self.seed * 7919 + 17)
        r = self.r0.copy()
        policy.reset(self, P)
        rows, steps, item_rows = [], [], []
        Ls, Lf, sh, Hh = map(np.asarray, (P.L_succ, P.L_fail, P.shallow, P.H))
        Lf = Lf.copy(); Lf[0] = P.pf; Lf[1] = P.pf + 0.10
        Hh = np.clip(Hh * np.r_[1, 1, P.h_scale, P.h_scale, P.h_scale, 1], 0, 1)
        n_total_skills = sum(self._items(s)[3].sum(1) for s in self.sessions)
        done_skills = np.zeros(N)
        for s in self.sessions:
            bp, be, qb, mask = self._items(s)
            Kmax = bp.shape[1]
            th0 = P.a0 + P.a1 * self.gz[:, None] + self.u[:, None] + self.eps[s]
            m = self.minit[s] < sig(th0 - bp)                     # BKT mastery
            D = np.zeros((N, Kmax)); Sh = np.zeros((N, Kmax))
            chat = np.clip(P.c0 + P.c0_gpa * self.gz[:, None] + np.zeros((N, Kmax)), 0.02, 0.98)
            F = np.zeros(N)
            succ_k = np.zeros((N, Kmax))
            succ = np.zeros(N); nst = np.zeros(N); leak = np.zeros(N); off_n = np.zeros(N)
            ans_req = np.zeros(N); nreq = np.zeros(N); acts = np.zeros((N, NA))
            r_start = r.copy()
            q_all = qb if not policy.grounded else np.full_like(qb, P.q_ground)
            for k in range(Kmax):
                mk = mask[:, k]
                remaining = n_total_skills - done_skills
                policy.start_skill(dict(session=s, k=k, b_prac=bp[:, k], b_exam=be[:, k] + P.d_exam,
                                        q=q_all[:, k], mask=mk, remaining=remaining, T=P.T_steps))
                for t in range(P.T_steps):
                    if P.family == "logistic":
                        p0 = sig(th0[:, k] + D[:, k] + Sh[:, k] - bp[:, k])
                    else:
                        p0 = np.where(m[:, k], 1 - P.bkt_slip, P.bkt_guess)
                    u = rng.random((N, 5))
                    req = np.where(u[:, 0] < r, 5, np.where(u[:, 1] < P.req_p3, 3, 2))
                    req = np.where(u[:, 4] < P.psi, req, 0)                 # 0 = no request
                    a = np.asarray(policy.act(req, t), int)
                    if not policy.has_ai:
                        a = np.zeros(N, int)
                    q = q_all[:, k]
                    off_p = P.off_base + (P.omega * sig(P.gam * (F - P.F0)) if policy.has_ai else 0.0)
                    off = u[:, 2] < off_p
                    ps = np.where(a == 5, q, 1 - (1 - p0) * (1 - Hh[a] * q))
                    ps = np.where(off, 0.5 * p0, ps)
                    y = rng.random(N) < ps
                    copy = u[:, 3] < P.c_min + (1 - P.c_min) * r
                    Lv = np.where(y, Ls[a], Lf[a])
                    Lv = np.where(a == 4, P.w0 + P.w1 * (1 - p0), Lv)
                    Lv = np.where(a == 5, np.where(copy, 0.0, P.L_proc), Lv)
                    game = np.where((a >= 2) & (a <= 4), 1 - P.rho_g * r, 1.0)
                    eng = (~off) & mk
                    if P.family == "logistic":
                        Z = P.zpd_lo + P.zpd_amp * p0 * (1 - p0)
                        dth = P.eta * Z * Lv * game * eng
                        D[:, k] += (1 - sh[a]) * dth
                        Sh[:, k] += sh[a] * dth
                    else:
                        learn = rng.random(N) < np.clip(P.bkt_T * Lv * game, 0, 1)
                        m[:, k] |= learn & eng
                    if policy.has_ai:
                        refused = (req == 5) & (a < 5) & mk
                        r = r - P.k_ref * r * refused
                        r = r + P.k_up * (1 - r) * (y & (a >= 4) & eng)
                        r = r - P.k_dn * r * (y & (a <= 2) & eng)
                        r = r - P.k_meta * r * ((a == 1) & mk)
                        F = np.where(mk, np.where(y, 0.5 * F, F + ((req > a) & ~y)), F)
                        ans_req += (req == 5) & mk; nreq += (req > 0) & mk
                    c = chat[:, k]
                    c = c + P.lam_c * (y - c) * mk
                    c = c + P.lam_m * (p0 - c) * ((a == 1) & mk)
                    chat[:, k] = c
                    succ_k[:, k] += y & mk
                    succ += y & mk; nst += mk; leak += (a == 5) & mk; off_n += off & mk
                    acts[np.arange(N), a] += mk
                    if record_steps:
                        steps.append(dict(session=s, k=k, t=t, mean_r=r[mk].mean(),
                                          share_answer=(a[mk] == 5).mean()))
                    policy.observe(a, req, y)
                policy.end_skill()
                done_skills += mk
            # ---- unassisted exam (paired near-transfer item per skill) ----
            if P.family == "logistic":
                thf = th0 + D + Sh
                pe = sig(thf - be - P.d_exam)
                pre = sig(th0 - be - P.d_exam)
                pdl = sig(th0 + D * np.exp(-P.decay_deep) + Sh * np.exp(-P.decay_shallow) - be - P.d_exam)
            else:
                m0 = self.minit[s] < sig(th0 - bp)
                pe = np.where(m, 1 - P.bkt_slip, P.bkt_guess)
                pre = np.where(m0, 1 - P.bkt_slip, P.bkt_guess)
                pdl = pe
            ye = self.exam_u[s] < pe
            K = mask.sum(1)
            mean = lambda x: (x * mask).sum(1) / K
            exam_exp, pre_exp = mean(pe), mean(pre)
            perceived = mean(chat)
            df = pd.DataFrame(dict(
                sid=self.sid, session=s, gz=self.gz, grade=self.grade,
                practice=succ / nst, exam=mean(ye.astype(float)), exam_exp=exam_exp, pre_exp=pre_exp,
                nlg=(exam_exp - pre_exp) / (1 - pre_exp), delayed=mean(pdl), perceived=perceived,
                overconf=perceived - exam_exp, calib_err=np.abs(perceived - exam_exp),
                leak=leak / nst, off=off_n / nst, r_start=r_start, r_end=r.copy(),
                ans_req=np.where(nreq > 0, ans_req / np.maximum(nreq, 1), np.nan),
                **{f"act_{n}": acts[:, i] / nst for i, n in enumerate(ACTIONS)}))
            rows.append(df)
            for g in np.unique(self.grade):
                idx = self.grade == g
                for k in range(int(mask[idx][0].sum())):
                    item_rows.append(dict(session=s, grade=int(g), k=k, practice=succ_k[idx, k].mean() / P.T_steps,
                                          exam=pe[idx, k].mean(), q_base=qb[idx, k][0]))
        self.item_df = pd.DataFrame(item_rows)
        out = pd.concat(rows, ignore_index=True)
        out["policy"] = policy.name
        out["seed"] = self.seed
        return (out, pd.DataFrame(steps)) if record_steps else out
