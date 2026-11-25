"""Assistance policies: baselines from the literature and TRACE.

All model-based policies share a `Belief` that runs on the tutor's *internal* (nominal) model
`Ph`.  When the ground-truth simulator is perturbed or switched to the BKT family, the internal
model is NOT updated - this is how robustness to misspecification is evaluated.

Observations available to every policy: the student's help request (before acting), the
correctness of each attempt, item metadata (difficulty, AI accuracy) and prior GPA.
Latent skill, reliance, frustration and confidence are never observed directly.
"""
from __future__ import annotations

import numpy as np

from .simulator import NA, LEAK, sig

GRID_M = 32


# --------------------------------------------------------------------------------------------
# simple, model-free policies
# --------------------------------------------------------------------------------------------
class Policy:
    name, has_ai, grounded = "policy", True, True

    def reset(self, cohort, P_true):
        self.N = cohort.N

    def start_skill(self, ctx):
        self.ctx = ctx

    def act(self, req, t):
        raise NotImplementedError

    def observe(self, a, req, y):
        pass

    def end_skill(self):
        pass


class Control(Policy):
    name, has_ai = "Control (no AI)", False

    def act(self, req, t):
        return np.zeros(self.N, int)


class GPTBase(Policy):
    """Unguarded chat assistant: delivers whatever is asked, with GPT-4's audited accuracy."""
    name, grounded = "GPT Base", False

    def act(self, req, t):
        return req


class GPTTutor(Policy):
    """Static guardrail (Bastani et al.): solution-grounded, never reveals the full answer."""
    name = "GPT Tutor (static)"

    def act(self, req, t):
        return np.minimum(req, 3)


class FixedFading(Policy):
    """Worked example -> conceptual hint -> independent attempt (Renkl & Atkinson fading)."""
    name = "Fixed fading"
    schedule = (4, 4, 1)          # tuned on development seeds (see experiments.py)

    def act(self, req, t):
        return np.full(self.N, self.schedule[min(t, len(self.schedule) - 1)])


# --------------------------------------------------------------------------------------------
# belief over latent learner state on the tutor's internal model
# --------------------------------------------------------------------------------------------
class Belief:
    def __init__(self, Ph, N, gz, rng, track_r=True, track_F=True, M=GRID_M):
        self.Ph, self.N, self.gz, self.M, self.rng = Ph, N, gz, M, rng
        self.track_r, self.track_F = track_r, track_F
        self.Hh = np.clip(np.asarray(Ph.H) * np.r_[1, 1, Ph.h_scale, Ph.h_scale, Ph.h_scale, 1], 0, 1)
        self.Ls, self.Lf = np.asarray(Ph.L_succ), np.asarray(Ph.L_fail).copy()
        self.Lf[0] = Ph.pf; self.Lf[1] = Ph.pf + 0.10
        self.u_sum = np.zeros(N); self.u_n = np.zeros(N)
        self.r = np.full(N, Ph.mu_r0)
        self.F = np.zeros(N)
        self.session = None
        self.z = rng.standard_normal(M)           # common quantiles for particle init

    # ---- lifecycle ----
    def start_skill(self, ctx):
        Ph = self.Ph
        if ctx["session"] != self.session:
            self.session = ctx["session"]; self.F[:] = 0.0
        shrink = self.u_n / (self.u_n + (Ph.sig_e ** 2) / (Ph.sig_u ** 2))
        u_hat = np.where(self.u_n > 0, self.u_sum / np.maximum(self.u_n, 1), 0.0) * shrink
        sd = np.sqrt(Ph.sig_e ** 2 + Ph.sig_u ** 2 * (1 - shrink))
        self.mu0 = Ph.a0 + Ph.a1 * self.gz + u_hat
        self.th = self.mu0[:, None] + sd[:, None] * self.z[None, :]
        self.th_init = self.th.copy()
        self.w = np.full((self.N, self.M), 1.0 / self.M)
        self.cum = np.zeros((self.N, self.M))
        self.b, self.be, self.q, self.mask = ctx["b_prac"], ctx["b_exam"], ctx["q"], ctx["mask"]
        self.c = np.clip(Ph.c0 + Ph.c0_gpa * self.gz, 0.02, 0.98)

    def end_skill(self):
        Ph = self.Ph
        post0 = (self.w * self.th_init).sum(1)        # posterior mean of initial skill
        resid = post0 - (Ph.a0 + Ph.a1 * self.gz)
        self.u_sum += resid * self.mask; self.u_n += self.mask

    # ---- model pieces ----
    def off_prob(self, F):
        Ph = self.Ph
        if not self.track_F:
            return np.full_like(F, Ph.off_base, dtype=float)
        return Ph.off_base + Ph.omega * sig(Ph.gam * (F - Ph.F0))

    def succ_prob(self, a, p0, q):
        a = np.broadcast_to(a, p0.shape[:1])
        H = self.Hh[a][:, None]
        return np.where((a == 5)[:, None], q[:, None], 1 - (1 - p0) * (1 - H * q[:, None]))

    def learn_amount(self, a, p0, y_prob, r):
        """Expected logit-skill gain given engagement, as a function of P(success)."""
        Ph = self.Ph
        a = np.broadcast_to(a, p0.shape[:1])
        Ls, Lf = self.Ls[a][:, None], self.Lf[a][:, None]
        L = y_prob * Ls + (1 - y_prob) * Lf
        L = np.where((a == 4)[:, None], Ph.w0 + Ph.w1 * (1 - p0), L)
        copy = Ph.c_min + (1 - Ph.c_min) * r
        L = np.where((a == 5)[:, None], ((1 - copy) * Ph.L_proc)[:, None], L)
        game = np.where(((a >= 2) & (a <= 4)), 1 - Ph.rho_g * r, 1.0)[:, None]
        Z = Ph.zpd_lo + Ph.zpd_amp * p0 * (1 - p0)
        return Ph.eta * Z * L * game

    def p0(self, th=None):
        return sig((self.th if th is None else th) - self.b[:, None])

    def p_exam(self, th=None, w=None):
        th = self.th if th is None else th
        w = self.w if w is None else w
        return (w * sig(th - self.be[:, None])).sum(1)

    def p_unassisted(self):
        return (self.w * self.p0()).sum(1)

    # ---- filtering ----
    def observe_request(self, req):
        if self.track_r:
            asked = (req > 0) & self.mask
            self.r = self.r + 0.12 * ((req == 5).astype(float) - self.r) * asked

    def update(self, a, req, y):
        Ph = self.Ph
        p0 = self.p0()
        off = self.off_prob(self.F)[:, None]
        ps = self.succ_prob(a, p0, self.q)
        like1 = (1 - off) * ps + off * 0.5 * p0
        like = np.where(y[:, None], like1, 1 - like1)
        w = self.w * like
        w = w / w.sum(1, keepdims=True)
        eng_post = np.where(y[:, None], (1 - off) * ps / np.maximum(like1, 1e-9),
                            (1 - off) * (1 - ps) / np.maximum(1 - like1, 1e-9))
        yv = y[:, None].astype(float)
        dth = eng_post * self.learn_amount(a, p0, yv, self.r)
        self.th = self.th + dth * self.mask[:, None]
        self.w = np.where(self.mask[:, None], w, self.w)
        # resample when effective sample size collapses
        ess = 1.0 / (self.w ** 2).sum(1)
        bad = ess < self.M / 3
        if bad.any():
            idx = np.array([self.rng.choice(self.M, self.M, p=wi) for wi in self.w[bad]])
            rows = np.where(bad)[0][:, None]
            jit = self.rng.normal(0, 0.05, idx.shape)
            self.th[bad] = self.th[rows, idx] + jit
            self.th_init[bad] = self.th_init[rows, idx] + jit
            self.w[bad] = 1.0 / self.M
        # reliance / frustration / confidence replicas (known structural dynamics)
        mk = self.mask
        if self.track_r:
            r = self.r
            r = r - Ph.k_ref * r * ((req == 5) & (a < 5) & mk)
            r = r + Ph.k_up * (1 - r) * (y & (a >= 4) & mk)
            r = r - Ph.k_dn * r * (y & (a <= 2) & mk)
            r = r - Ph.k_meta * r * ((a == 1) & mk)
            self.r = r
        self.F = np.where(mk, np.where(y, 0.5 * self.F, self.F + ((req > a) & ~y)), self.F)
        c = self.c + Ph.lam_c * (y - self.c) * mk
        self.c = c + Ph.lam_m * ((self.w * p0).sum(1) - c) * ((a == 1) & mk)

    # ---- expected one-step transition used for planning ----
    def expect_step(self, state, a, req_known=None):
        """state = (th, r, F, c, prac). Returns the expected next state for action array a."""
        Ph = self.Ph
        th, r, F, c, prac = state
        p0 = sig(th - self.b[:, None])
        off = self.off_prob(F)[:, None]
        ps = self.succ_prob(a, p0, self.q)
        dth = (1 - off) * self.learn_amount(a, p0, ps, r)
        pbar = (self.w * ((1 - off) * ps + off * 0.5 * p0)).sum(1)
        p0bar = (self.w * p0).sum(1)
        if req_known is not None:
            p_ans = (req_known == 5).astype(float)
            p_gt = (req_known > a).astype(float)
        else:
            p_ans = Ph.psi * r
            p_gt = Ph.psi * (r * (a < 5) + (1 - r) * (Ph.req_p3 * (a < 3) + (1 - Ph.req_p3) * (a < 2)))
        if self.track_r:
            r = r - Ph.k_ref * r * p_ans * (a < 5)
            r = r + Ph.k_up * (1 - r) * pbar * (a >= 4)
            r = r - Ph.k_dn * r * pbar * (a <= 2)
            r = r - Ph.k_meta * r * (a == 1)
        F = pbar * 0.5 * F + (1 - pbar) * (F + p_gt)
        c = c + Ph.lam_c * (pbar - c)
        c = np.where(a == 1, c + Ph.lam_m * (p0bar - c), c)
        return (th + dth, r, F, c, prac + pbar)


# --------------------------------------------------------------------------------------------
# model-based baselines
# --------------------------------------------------------------------------------------------
class _ModelBased(Policy):
    track_r, track_F = False, False

    def __init__(self, Ph, seed=0):
        self.Ph, self.seed = Ph, seed

    def reset(self, cohort, P_true):
        super().reset(cohort, P_true)
        self.B = Belief(self.Ph, cohort.N, cohort.gz, np.random.default_rng(self.seed + 99),
                        track_r=self.track_r, track_F=self.track_F)

    def start_skill(self, ctx):
        self.ctx = ctx; self.B.start_skill(ctx)

    def observe(self, a, req, y):
        self.B.update(a, req, y)

    def end_skill(self):
        self.B.end_skill()

    def state(self):
        B = self.B
        return (B.th, B.r, B.F, B.c, np.zeros(self.N))


class MasteryLadder(_ModelBased):
    """Mastery-banded help cap (Pisan 2026; Kadir 2026): more help when estimated mastery is low."""
    name = "Mastery ladder"
    lo, hi = 0.2, 0.5             # tuned on development seeds

    def act(self, req, t):
        p = self.B.p_unassisted()
        return np.where(p < self.lo, 3, np.where(p < self.hi, 2, 0))


class GreedyAssist(_ModelBased):
    """Assistance-formula optimum: the level with the largest expected one-step learning gain
    (Koedinger & Pavlik), ignoring reliance and frustration."""
    name = "Greedy learning-gain"

    def act(self, req, t):
        st = self.state()
        best, val = np.zeros(self.N, int), np.full(self.N, -np.inf)
        for a in range(NA):
            th2 = self.B.expect_step(st, np.full(self.N, a), req)[0]
            v = self.B.p_exam(th2)
            better = v > val
            best, val = np.where(better, a, best), np.where(better, v, val)
        return best


class Overassist(_ModelBased):
    """Maximises immediate correctness (the over-assisting behaviour documented by Teo et al.)."""
    name = "Max-correctness"

    def act(self, req, t):
        p0 = self.B.p0()
        vals = np.stack([(self.B.w * self.B.succ_prob(np.full(self.N, a), p0, self.B.q)).sum(1)
                         for a in range(NA)], 1)
        return vals.argmax(1)


class LambdaPenalty(_ModelBased):
    """Correctness minus a fixed leakage penalty (Dinucu-Jianu et al. 2025 reward shape)."""
    def __init__(self, Ph, lam=2.0, seed=0):
        super().__init__(Ph, seed); self.lam = lam
        self.name = f"Leak-penalised RL reward (lambda={lam:g})"

    def act(self, req, t):
        p0 = self.B.p0()
        vals = np.stack([(self.B.w * self.B.succ_prob(np.full(self.N, a), p0, self.B.q)).sum(1)
                         - self.lam * LEAK[a] for a in range(NA)], 1)
        return vals.argmax(1)


class TSBandit(_ModelBased):
    """Contextual Thompson sampling over help levels, reward = learning progress
    (change in estimated exam success), contexts = prior-GPA terciles (ZPDES-style)."""
    name = "Thompson bandit (LP reward)"

    def reset(self, cohort, P_true):
        super().reset(cohort, P_true)
        self.ctxid = np.digitize(cohort.gz, np.quantile(cohort.gz, [1 / 3, 2 / 3]))
        self.sum = np.zeros((3, NA)); self.cnt = np.zeros((3, NA)); self.sq = np.zeros((3, NA))
        self.rng = np.random.default_rng(self.seed + 5)

    def act(self, req, t):
        self.pe_before = self.B.p_exam()
        mean = self.sum / np.maximum(self.cnt, 1)
        sd = 0.05 / np.sqrt(self.cnt + 1)
        draws = self.rng.normal(mean[self.ctxid], sd[self.ctxid])
        self.a = draws.argmax(1)
        return self.a

    def observe(self, a, req, y):
        super().observe(a, req, y)
        rew = self.B.p_exam() - self.pe_before
        mk = self.B.mask
        np.add.at(self.sum, (self.ctxid[mk], a[mk]), rew[mk])
        np.add.at(self.cnt, (self.ctxid[mk], a[mk]), 1)


# --------------------------------------------------------------------------------------------
# TRACE
# --------------------------------------------------------------------------------------------
class TRACE(_ModelBased):
    """Tracking Reliance And Calibration for Effortful learning.

    Belief:   particle posterior over current-skill logit mastery (hierarchical prior from GPA and
              earlier skills) + filtered reliance r_hat (from help requests and known dynamics)
              + frustration and self-assessment replicas.
    Planner:  rollout (one-step policy improvement, Bertsekas) over the remaining steps of the
              current item with an r-aware greedy base heuristic; terminal value
                 V = P(exam success) - w_c |c_hat - P(exam)| - g_r(gpa) * remaining * (r_T - r_0)
                     - g_F * F_T + w_p * E[practice successes]
              where g_r and g_F are sensitivities of downstream learning to reliance/frustration,
              computed once by finite differences on the internal model.
    """
    track_r, track_F = True, True

    def __init__(self, Ph, seed=0, use_r=True, use_c=True, use_F=True, lookahead=True,
                 w_c=0.25, w_p=0.02, name="TRACE (ours)"):
        super().__init__(Ph, seed)
        self.use_r, self.use_c, self.use_F, self.lookahead = use_r, use_c, use_F, lookahead
        self.track_r, self.track_F = use_r, use_F
        self.w_c = w_c if use_c else 0.0
        self.w_p = w_p
        self.name = name

    # finite-difference sensitivities of next-skill learning to reliance / frustration
    def _sensitivities(self):
        B, N, Ph = self.B, self.N, self.Ph
        ctx = dict(session=-1, k=0, b_prac=np.zeros(N), b_exam=np.zeros(N) + Ph.d_exam,
                   q=np.full(N, Ph.q_ground), mask=np.ones(N, bool), remaining=np.ones(N), T=Ph.T_steps)
        B.start_skill(ctx)

        def value(r, F):
            st = (B.th.copy(), r, F, B.c.copy(), np.zeros(N))
            for t in range(Ph.T_steps):
                a = self._greedy(st)
                st = B.expect_step(st, a)
            return B.p_exam(st[0])
        base = value(np.full(N, Ph.mu_r0), np.zeros(N))
        g_r = (base - value(np.full(N, Ph.mu_r0) + 0.1, np.zeros(N))) / 0.1
        g_F = (base - value(np.full(N, Ph.mu_r0), np.ones(N))) / 1.0
        B.u_sum[:] = 0; B.u_n[:] = 0; B.session = None
        return np.maximum(g_r, 0), np.maximum(g_F, 0)

    def reset(self, cohort, P_true):
        super().reset(cohort, P_true)
        self.g_r, self.g_F = self._sensitivities() if (self.use_r or self.use_F) else (0, 0)
        if not self.use_r:
            self.g_r = np.zeros(self.N)
        if not self.use_F:
            self.g_F = np.zeros(self.N)

    def _greedy(self, st, req=None):
        best, val = np.zeros(self.N, int), np.full(self.N, -np.inf)
        for a in range(NA):
            nxt = self.B.expect_step(st, np.full(self.N, a), req)
            v = self.B.p_exam(nxt[0])
            better = v > val
            best, val = np.where(better, a, best), np.where(better, v, val)
        return best

    def _terminal(self, st, r0):
        th, r, F, c, prac = st
        pe = self.B.p_exam(th)
        rem = np.maximum(self.ctx["remaining"] - 1, 0)
        v = pe + self.w_p * prac
        v = v - self.w_c * np.abs(c - pe)
        v = v - self.g_r * np.minimum(rem, 6) * (r - r0) * 0.5
        v = v - self.g_F * F * 0.5
        return v

    def act(self, req, t):
        B = self.B
        B.observe_request(req)
        st0 = self.state()
        T = self.ctx["T"]
        r0 = B.r.copy()
        best, val = np.zeros(self.N, int), np.full(self.N, -np.inf)
        for a in range(NA):
            st = B.expect_step(st0, np.full(self.N, a), req)
            if self.lookahead:
                for _ in range(t + 1, T):
                    st = B.expect_step(st, self._greedy(st))
                v = self._terminal(st, r0)
            else:
                v = B.p_exam(st[0])
            better = v > val + 1e-12
            best, val = np.where(better, a, best), np.where(better, v, val)
        return best
