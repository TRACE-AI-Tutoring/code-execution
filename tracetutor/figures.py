"""Publication figures (PDF + PNG) from cached results/."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

# validated categorical palette (fixed order) + neutral inks
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#6250d6", red="#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"
TRACE_NAME = "TRACE (ours)"
SHORT = {"Control (no AI)": "Control", "GPT Base": "GPT Base", "GPT Tutor (static)": "GPT Tutor",
         "Fixed fading": "Fixed fading", "Mastery ladder": "Mastery ladder",
         "Greedy learning-gain": "Greedy LG", "Thompson bandit (LP reward)": "TS bandit",
         "Max-correctness": "Max-correct", "Leak-penalised RL reward (lambda=2)": "Leak-pen. RL",
         TRACE_NAME: "TRACE"}
FOCUS = {"GPT Base": C["orange"], "GPT Tutor (static)": C["aqua"], "Mastery ladder": C["yellow"],
         "Greedy learning-gain": C["violet"], TRACE_NAME: C["blue"]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf")); fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def _ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _col(p):
    return C["blue"] if p == TRACE_NAME else MUTED


# ------------------------------------------------------------------------------------------
def fig_calibration():
    fit = pd.read_csv(os.path.join(RES, "calibration_fit.csv"))
    g = fit.groupby("moment").agg(sim=("sim", "mean"), sd=("sim", "std"), target=("target", "first"),
                                  se=("se", "first")).reset_index()
    labels = {"ctrl_prac": "Control practice", "ctrl_exam": "Control exam",
              "base_prac_eff": "Base: practice effect", "tutor_prac_eff": "Tutor: practice effect",
              "base_exam_eff": "Base: exam effect", "tutor_exam_eff": "Tutor: exam effect",
              "exam_gpa_slope": "Exam-GPA slope (/10)", "base_answer_share_s1": "Base: answer requests S1",
              "tutor_answer_share_s1": "Tutor: answer requests S1", "ctrl_perceived": "Control perceived perf.",
              "base_perceived_eff": "Base: perceived effect", "tutor_perceived_eff": "Tutor: perceived effect"}
    g["label"] = g.moment.map(labels)
    sc = np.where(g.moment == "exam_gpa_slope", 0.1, 1.0)
    order = list(labels)
    g = g.set_index("moment").loc[order].reset_index()
    sc = np.where(g.moment == "exam_gpa_slope", 0.1, 1.0)

    het = pd.read_csv(os.path.join(RES, "real_heterogeneity.csv"))
    vs = pd.read_csv(os.path.join(RES, "validation_sim.csv")).groupby(["by", "level", "outcome", "term"]).sim.mean().reset_index()
    het["level"] = het.level.astype(str); vs["level"] = vs.level.astype(str)
    m = het.merge(vs, on=["by", "level", "outcome", "term"])
    pr = pd.read_csv(os.path.join(RES, "real_problem_level.csv")).dropna(subset=["base_minus_ctrl"])
    ps = pd.read_csv(os.path.join(RES, "validation_items_sim.csv"))

    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    y = np.arange(len(g))[::-1]
    a = ax[0]
    a.errorbar(g.target * sc, y, xerr=1.96 * g.se * sc, fmt="o", color=INK2, ms=4, lw=1, capsize=0,
               label="Empirical target (95% CI)")
    a.plot(g.sim * sc, y, "D", color=C["blue"], ms=5, label="Calibrated simulator", mec="white", mew=1)
    a.set_yticks(y); a.set_yticklabels(g.label); a.axvline(0, color=MUTED, lw=0.8)
    a.set_xlabel("Value (score scale)"); a.set_title("(a) Targeted moments", loc="left")
    a.legend(loc="lower right", fontsize=7.5)

    a = ax[1]
    for (by, oc), mk in zip([("Session", "Part2Tot"), ("Session", "Part3Tot"), ("tercile", "Part2Tot"),
                             ("tercile", "Part3Tot")], ["o", "s", "^", "v"]):
        d = m[(m.by == by) & (m.outcome == oc)]
        lab = f"{'per session' if by == 'Session' else 'per GPA tercile'}, {'practice' if oc == 'Part2Tot' else 'exam'}"
        a.errorbar(d.coef, d.sim, xerr=1.96 * d.se, fmt=mk, ms=4.5, lw=0.8, color=C["blue"] if oc == "Part3Tot" else C["orange"],
                   mfc="white" if by == "tercile" else None, label=lab, alpha=0.9)
    lim = [-0.15, 0.55]
    a.plot(lim, lim, color=MUTED, lw=1, ls="--"); a.set_xlim(lim); a.set_ylim(lim)
    r = np.corrcoef(m.coef, m.sim)[0, 1]
    a.text(0.03, 0.95, f"r = {r:.2f}  (n = {len(m)}, not targeted)", transform=a.transAxes, va="top", color=INK2)
    a.set_xlabel("Real effect (field RCT)"); a.set_ylabel("Simulated effect")
    a.set_title("(b) Out-of-sample heterogeneity", loc="left"); a.legend(fontsize=6.5, loc="lower right")

    a = ax[2]
    a.scatter(pr.q_base, pr.base_minus_ctrl, s=18, color=INK2, alpha=0.7, label="Real problems (n=57)", lw=0)
    a.scatter(ps.q_base, ps.base_minus_ctrl, s=18, color=C["blue"], alpha=0.7, label="Simulated items", lw=0, marker="D")
    for d, col in ((pr, INK2), (ps, C["blue"])):
        b = np.polyfit(d.q_base, d.base_minus_ctrl, 1); xx = np.linspace(0, 1, 10)
        a.plot(xx, np.polyval(b, xx), color=col, lw=1.6)
        a.text(0.02, 0.97 - (0.07 if col == C["blue"] else 0) - 0.12, f"{'simulated' if col == C['blue'] else 'real'} slope {b[0]:.2f}",
               color=col, transform=a.transAxes, va="top", fontsize=7.5)
    a.set_xlabel("Audited GPT-4 accuracy on the problem"); a.set_ylabel("Practice gain of GPT Base over Control")
    a.set_title("(c) Problem-level mechanism", loc="left"); a.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); _save(fig, "fig2_calibration")


def fig_main():
    p = pd.read_csv(os.path.join(RES, "main_per_seed.csv"))
    con = pd.read_csv(os.path.join(RES, "main_contrasts.csv"))
    g = p.groupby("policy").agg(prac=("practice", "mean"), exam=("exam_exp", "mean")).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
    a = ax[0]
    ctrl = g[g.policy == "Control (no AI)"].iloc[0]
    a.axhline(ctrl.exam, color=MUTED, lw=0.8, ls="--"); a.axvline(ctrl.prac, color=MUTED, lw=0.8, ls="--")
    for _, r in g.iterrows():
        col = FOCUS.get(r.policy, MUTED)
        a.scatter(r.prac, r.exam, s=70 if r.policy == TRACE_NAME else 40, color=col, edgecolor="white", lw=1.5, zorder=3)
        dx, dy, ha = 0.012, 0.0, "left"
        if r.policy in ("Max-correctness", "Greedy learning-gain"):
            dx, ha = -0.012, "right"
        if r.policy == "Fixed fading":
            dy = -0.0012
        a.text(r.prac + dx, r.exam + dy, SHORT[r.policy], fontsize=7.5, va="center", ha=ha,
               color=INK if r.policy == TRACE_NAME else INK2, weight="bold" if r.policy == TRACE_NAME else None)
    a.set_xlabel("Assisted practice score"); a.set_ylabel("Unassisted exam score (expected)")
    a.set_title("(a) Practice performance is not learning", loc="left")
    a.text(0.97, 0.55, "dashed lines = Control (no AI)", transform=a.transAxes, ha="right", fontsize=7, color=INK2)

    a = ax[1]
    d = con[(con.ref == "Control (no AI)") & (con.policy != "Control (no AI)")].sort_values("diff")
    y = np.arange(len(d))
    cols = [FOCUS.get(pp, MUTED) for pp in d.policy]
    a.barh(y, d["diff"] * 100, color=cols, height=0.62, edgecolor="white", linewidth=2)
    a.errorbar(d["diff"] * 100, y, xerr=[(d["diff"] - d.lo) * 100, (d.hi - d["diff"]) * 100], fmt="none", color=INK, lw=0.9)
    a.set_yticks(y); a.set_yticklabels([SHORT[x] for x in d.policy]); a.axvline(0, color=INK2, lw=0.8)
    for yi, v in zip(y, d["diff"] * 100):
        a.text(v + (0.25 if v >= 0 else -0.25), yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
    a.set_xlabel("Exam effect vs Control (percentage points, 95% CI)")
    a.set_title("(b) Effect on unassisted learning", loc="left"); a.grid(axis="y", visible=False)
    a.set_xlim(-9.5, 3.2)
    fig.tight_layout(); _save(fig, "fig3_main")


def fig_dimensions():
    p = pd.read_csv(os.path.join(RES, "main_per_seed.csv"))
    mets = [("nlg", "Normalised learning gain", 1), ("delayed", "Delayed retention score", 1),
            ("overconf", "Overconfidence (perceived - actual)", 1), ("calib_err", "Calibration error |perceived - actual|", 1),
            ("off", "Off-task share of steps", 1), ("leak", "Full-solution leakage rate", 1)]
    order = p.groupby("policy").exam_exp.mean().sort_values().index
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    for a, (m, lab, _) in zip(axs.ravel(), mets):
        y = np.arange(len(order))
        for yi, pol in zip(y, order):
            mu, ci = _ci(p[p.policy == pol][m])
            a.errorbar(mu, yi, xerr=ci, fmt="o", color=FOCUS.get(pol, MUTED), ms=6 if pol == TRACE_NAME else 4.5,
                       lw=1, mec="white", mew=0.8)
        ctrl = p[p.policy == "Control (no AI)"][m].mean()
        a.axvline(ctrl, color=MUTED, lw=0.8, ls="--")
        a.set_title(lab, loc="left"); a.set_yticks(y); a.set_yticklabels([SHORT[o] for o in order])
    fig.text(0.99, 0.005, "Policies sorted by exam score; dashed = Control; 95% CI over 10 seeds.", ha="right", fontsize=7, color=INK2)
    fig.tight_layout(); _save(fig, "fig4_dimensions")


def fig_pareto():
    p = pd.read_csv(os.path.join(RES, "pareto_per_seed.csv"))
    g = p.groupby("policy")[["exam_exp", "overconf", "practice"]].mean()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    tr = g[g.index.str.startswith("TRACE")].copy()
    tr["w"] = tr.index.str.extract(r"w_c=([\d.]+)")[0].astype(float).values; tr = tr.sort_values("w")
    lp = g[g.index.str.startswith("Leak")].copy()
    lp["l"] = lp.index.str.extract(r"lambda=([\d.]+)")[0].astype(float).values; lp = lp.sort_values("l")
    for a, xm, xl in ((ax[0], "overconf", "Overconfidence (perceived - actual exam)"),
                      (ax[1], "practice", "Assisted practice score")):
        a.plot(tr[xm], tr.exam_exp, "-o", color=C["blue"], ms=6, mec="white", mew=1.2, label="TRACE (w_c sweep)")
        for _, r in (tr.iterrows() if xm == "overconf" else []):
            a.annotate(f"w_c={r.w:g}", (r[xm], r.exam_exp), textcoords="offset points", xytext=(5, 4), fontsize=6.5, color=C["blue"])
        a.plot(lp[xm], lp.exam_exp, "-s", color=C["orange"], ms=5, mec="white", mew=1, label="Leak-penalised RL (lambda sweep)")
        for nm, col, mk in (("Control (no AI)", INK2, "X"), ("GPT Tutor (static)", C["aqua"], "P"),
                            ("Greedy learning-gain", C["violet"], "D"), ("Mastery ladder", C["yellow"], "^")):
            a.scatter(g.loc[nm, xm], g.loc[nm, "exam_exp"], color=col, marker=mk, s=55, zorder=3, label=SHORT[nm], edgecolor="white")
        a.set_xlabel(xl); a.set_ylabel("Unassisted exam score")
    ax[0].set_title("(a) Learning vs metacognitive calibration", loc="left")
    ax[1].set_title("(b) Learning vs practice performance", loc="left")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); _save(fig, "fig5_pareto")


def fig_dynamics():
    h = pd.read_csv(os.path.join(RES, "horizon.csv"))
    ctrl = h[h.policy == "Control (no AI)"].set_index(["seed", "session"]).exam_exp
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for pol, col in FOCUS.items():
        d = h[h.policy == pol]
        r = d.groupby("session").r_end.mean()
        ax[0].plot(r.index, r.values, "-o", color=col, ms=3.5, label=SHORT[pol])
        de = (d.set_index(["seed", "session"]).exam_exp - ctrl).groupby(level="session").mean() * 100
        ax[1].plot(de.index, de.values, "-o", color=col, ms=3.5, label=SHORT[pol])
    ax[0].axhline(h[h.policy == "Control (no AI)"].r_end.mean(), color=MUTED, ls="--", lw=0.8)
    ax[0].set_ylabel("Mean reliance (P(ask for answer))"); ax[0].set_xlabel("Session")
    ax[0].set_title("(a) Reliance trajectories (12 sessions)", loc="left")
    ax[1].axhline(0, color=MUTED, ls="--", lw=0.8)
    ax[1].set_ylabel("Exam effect vs Control (pp)"); ax[1].set_xlabel("Session")
    ax[1].set_title("(b) Learning effect over a longer horizon", loc="left")
    ax[0].legend(fontsize=7, ncol=1, loc="center right")
    for a in ax:
        a.set_xticks(range(1, 13))
    fig.tight_layout(); _save(fig, "fig6_dynamics")


def fig_equity_adaptivity():
    eq = pd.read_csv(os.path.join(RES, "equity.csv"))
    ad = pd.read_csv(os.path.join(RES, "trace_actions.csv"))
    ctrl = eq[eq.policy == "Control (no AI)"].set_index(["seed", "tercile"]).exam_exp
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    terc = ["low", "mid", "high"]
    pols = list(FOCUS)
    w = 0.16
    for i, pol in enumerate(pols):
        d = eq[eq.policy == pol].set_index(["seed", "tercile"]).exam_exp - ctrl
        mu = [d.xs(t, level="tercile").mean() * 100 for t in terc]
        ci = [1.96 * d.xs(t, level="tercile").std() / np.sqrt(10) * 100 for t in terc]
        x = np.arange(3) + (i - 2) * w
        ax[0].bar(x, mu, w * 0.88, color=FOCUS[pol], label=SHORT[pol], edgecolor="white", linewidth=1)
        ax[0].errorbar(x, mu, yerr=ci, fmt="none", color=INK, lw=0.8)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["Low prior GPA", "Middle", "High prior GPA"])
    ax[0].axhline(0, color=INK2, lw=0.8); ax[0].set_ylabel("Exam effect vs Control (pp)")
    ax[0].set_title("(a) Effects by prior achievement tercile", loc="left"); ax[0].legend(fontsize=7, ncol=3, loc="lower left")
    ax[0].grid(axis="x", visible=False)
    acts = ["none", "reflect", "hint", "steps", "worked", "answer"]
    nm = ["Attempt alone", "Reflect-then-attempt", "Conceptual hint", "Step hints", "Worked example", "Full answer"]
    cols = ["#d6e4f5", "#a9c8ee", "#76a7e2", "#2a78d6", "#1d5ca8", "#123d73"]   # ordinal: help level
    g = ad.groupby("tercile")[[f"act_{a}" for a in acts]].mean().loc[terc]
    left = np.zeros(3)
    for a_, n_, c_ in zip(acts, nm, cols):
        v = g[f"act_{a_}"].values
        ax[1].barh(range(3), v, left=left, color=c_, label=n_, edgecolor="white", linewidth=2, height=0.6)
        for j in range(3):
            if v[j] > 0.07:
                ax[1].text(left[j] + v[j] / 2, j, f"{v[j]:.0%}", ha="center", va="center", fontsize=7,
                           color="white" if c_ in cols[3:] else INK)
        left += v
    ax[1].set_yticks(range(3)); ax[1].set_yticklabels(["Low prior GPA", "Middle", "High prior GPA"])
    ax[1].set_xlim(0, 1); ax[1].set_xlabel("Share of TRACE decisions"); ax[1].grid(visible=False)
    ax[1].set_title("(b) TRACE help level by learner (darker = more help)", loc="left")
    ax[1].legend(fontsize=6.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); _save(fig, "fig7_equity_adaptivity")


def fig_robustness():
    r = pd.read_csv(os.path.join(RES, "robustness.csv"))
    r["family"] = np.where(r.tag.str.startswith("bkt"), "bkt", "perturb")
    ctrl = r[r.policy == "Control (no AI)"].set_index("tag").exam_exp
    r["d"] = (r.set_index("tag").exam_exp - ctrl.reindex(r.tag).values).values * 100 if False else \
        r.exam_exp.values - ctrl.reindex(r.tag).values
    r["d"] *= 100
    order = r[r.family == "perturb"].groupby("policy").d.median().sort_values().index
    order = [o for o in order if o != "Control (no AI)"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    data = [r[(r.family == "perturb") & (r.policy == o)].d.values for o in order]
    bp = ax[0].boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, o in zip(bp["boxes"], order):
        patch.set_facecolor(FOCUS.get(o, "#d9d8d2")); patch.set_edgecolor("white"); patch.set_alpha(0.9)
    for i, d in enumerate(data):
        ax[0].scatter(d, np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), s=5, color=INK2, alpha=0.5, lw=0, zorder=3)
    ax[0].set_yticks(range(1, len(order) + 1)); ax[0].set_yticklabels([SHORT[o] for o in order])
    ax[0].axvline(0, color=INK2, lw=0.8); ax[0].set_xlabel("Exam effect vs Control (pp)")
    ax[0].set_title("(a) 30 misspecified ground-truth learners", loc="left")
    b = r[r.family == "bkt"].groupby("policy").d.agg(["mean", "std"]).loc[order]
    y = np.arange(len(order))
    ax[1].barh(y, b["mean"], color=[FOCUS.get(o, "#d9d8d2") for o in order], height=0.6, edgecolor="white", linewidth=2)
    ax[1].errorbar(b["mean"], y, xerr=1.96 * b["std"] / np.sqrt(5), fmt="none", color=INK, lw=0.8)
    ax[1].set_yticks(y); ax[1].set_yticklabels([SHORT[o] for o in order]); ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_xlabel("Exam effect vs Control (pp)"); ax[1].grid(axis="y", visible=False)
    ax[1].set_title("(b) Different learner family (BKT ground truth)", loc="left")
    fig.tight_layout(); _save(fig, "fig8_robustness")


def fig_ablation():
    a = pd.read_csv(os.path.join(RES, "ablation_per_seed.csv"))
    order = ["TRACE (ours)", "TRACE - reliance state", "TRACE - frustration model", "TRACE - calibration term",
             "TRACE - lookahead (myopic)", "TRACE - all (belief only)"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
    for k, (m, lab, sc) in enumerate((("d_exam", "Exam effect vs Control (pp)", 100),
                                      ("overconf", "Overconfidence", 1), ("off", "Off-task share", 1))):
        mu = [a[a.policy == o][m].mean() * sc for o in order]
        ci = [1.96 * a[a.policy == o][m].std() / np.sqrt(10) * sc for o in order]
        y = np.arange(len(order))[::-1]
        ax[k].barh(y, mu, color=[C["blue"]] + ["#bdbcb6"] * 5, height=0.6, edgecolor="white", linewidth=2)
        ax[k].errorbar(mu, y, xerr=ci, fmt="none", color=INK, lw=0.8)
        ax[k].set_title(lab, loc="left"); ax[k].grid(axis="y", visible=False)
        ax[k].set_yticks(y); ax[k].set_yticklabels([o.replace("TRACE - ", "w/o ").replace(" (ours)", " (full)") for o in order])
    fig.tight_layout(); _save(fig, "fig9_ablation")


def make_all():
    for f in (fig_calibration, fig_main, fig_dimensions, fig_pareto, fig_dynamics, fig_equity_adaptivity,
              fig_ablation):
        f(); print("  ", f.__name__)
    if os.path.exists(os.path.join(RES, "robustness.csv")):
        fig_robustness(); print("   fig_robustness")
