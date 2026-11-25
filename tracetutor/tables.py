"""LaTeX tables + a macro file of every number quoted in the paper text (no hand transcription)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, TAB = os.path.join(ROOT, "results"), os.path.join(ROOT, "tables")
TRACE = "TRACE (ours)"
SHORT = {"Control (no AI)": "Control (no AI)", "GPT Base": "GPT Base \\citep{bastani2025}",
         "GPT Tutor (static)": "GPT Tutor \\citep{bastani2025}", "Fixed fading": "Fixed fading \\citep{renkl2003}",
         "Mastery ladder": "Mastery ladder \\citep{pisan2026,kadir2026}",
         "Greedy learning-gain": "Greedy learning gain \\citep{koedinger2007}",
         "Thompson bandit (LP reward)": "TS bandit, LP reward \\citep{clement2015}",
         "Max-correctness": "Max-correctness \\citep{teo2026}",
         "Leak-penalised RL reward (lambda=2)": "Leak-penalised reward \\citep{dinucu2025}",
         TRACE: "\\textbf{TRACE (ours)}"}
ORDER = ["Control (no AI)", "GPT Base", "GPT Tutor (static)", "Fixed fading", "Mastery ladder",
         "Greedy learning-gain", "Thompson bandit (LP reward)", "Max-correctness",
         "Leak-penalised RL reward (lambda=2)", TRACE]
MACROS = {}


def _w(name, s):
    import re
    os.makedirs(TAB, exist_ok=True)
    # scale wide tables to the text block (robust to the journal's column layout)
    s = re.sub(r"\\begin\{tabular\*\}\{\\(textwidth|columnwidth)\}\{@\{\\extracolsep\\fill\}([^}]*)\}",
               lambda m: "\\adjustbox{max width=\\textwidth}{\\begin{tabular}{" + m.group(2) + "}", s)
    s = s.replace("\\end{tabular*}", "\\end{tabular}}")
    with open(os.path.join(TAB, name), "w") as f:
        f.write(s)


def mac(name, val):
    MACROS[name] = val


def _ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def star(p):
    return "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


# ------------------------------------------------------------------------------------------
def t_real():
    t = pd.read_csv(os.path.join(RES, "real_table1.csv"))
    from scipy.stats import norm
    lines = []
    for oc in t.outcome.unique():
        d = t[t.outcome == oc].set_index("term")
        cells = []
        for v in ("GPTBase", "GPTTutor", "gpa_prev"):
            p = 2 * (1 - norm.cdf(abs(d.loc[v, "t"])))
            cells.append(f"${d.loc[v, 'coef']:.3f}{star(p)}$ $({d.loc[v, 'se']:.3f})$")
        lines.append(f"{oc} & " + " & ".join(cells) + f" & {d.iloc[0].control_mean:.3f} & {int(d.iloc[0].n):,} \\\\")
    s = r"""\begin{table}[t]
\caption{Re-estimation of the field experiment of \citet{bastani2025} from the public replication data
(OLS with teacher, session, grader and grade fixed effects and prior GPA; classroom-clustered CR1 standard
errors in parentheses). Rows 1--2 reproduce the published Table~1; rows 3--5 are our secondary analysis of
the post-exam self-assessment survey (perceived performance mapped to $[0,1]$).}
\label{tab:real}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccccc}
\toprule
Outcome & GPT Base & GPT Tutor & Prior GPA & Control mean & $N$ \\
\midrule
""" + "\n".join(lines) + r"""
\botrule
\end{tabular*}
\footnotesize{$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}
\end{table}
"""
    _w("tab_real.tex", s)
    d = t.set_index(["outcome", "term"])
    oc = "Overconfidence (perceived - exam)"
    mac("realOCbase", f"{d.loc[(oc, 'GPTBase'), 'coef']:.3f}")
    mac("realOCtutor", f"{d.loc[(oc, 'GPTTutor'), 'coef']:.3f}")
    mac("realOCbaseRel", f"{100 * d.loc[(oc, 'GPTBase'), 'rel']:.0f}")
    mac("realOCtutorRel", f"{100 * d.loc[(oc, 'GPTTutor'), 'rel']:.0f}")
    mac("realOCctrl", f"{d.loc[(oc, 'GPTBase'), 'control_mean']:.3f}")


def t_params():
    P = json.load(open(os.path.join(RES, "params_logistic.json")))
    cal = [("a0", r"$\alpha_0$", "prior-skill intercept"), ("a1", r"$\alpha_1$", "prior-skill slope on GPA ($z$)"),
           ("d_exam", r"$\delta_e$", "exam difficulty offset"), ("eta", r"$\eta$", "learning rate"),
           ("h_scale", r"$s_h$", "strength of hints/worked examples"), ("mu_r0", r"$\bar r_0$", "initial reliance (mean)"),
           ("k_ref", r"$\kappa_{\mathrm{ref}}$", "reliance extinction after refusal"), ("psi", r"$\psi$", "help-request uptake"),
           ("c_min", r"$c_{\min}$", "copy floor"), ("L_proc", r"$L_{\mathrm{proc}}$", "learning from a processed solution"),
           ("pf", r"$\phi$", "productive-failure learning"), ("c0", r"$c_0$", "prior self-assessment"),
           ("c0_gpa", r"$c_1$", "self-assessment slope on GPA"), ("lam_c", r"$\lambda_c$", "confidence update rate")]
    rows = [f"{sym} & {desc} & {P[k]:.3f} \\\\" for k, sym, desc in cal]
    s = r"""\begin{table}[t]
\caption{Parameters estimated by simulated moments (left) against the 12 empirical moments of Table~\ref{tab:fit}.
All other mechanism parameters are fixed a priori from the learning-sciences literature (Appendix~\ref{app:sim}).}
\label{tab:params}
\footnotesize
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}llc}
\toprule
Symbol & Meaning & Estimate \\
\midrule
""" + "\n".join(rows) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    _w("tab_params.tex", s)


def t_fit():
    f = pd.read_csv(os.path.join(RES, "calibration_fit.csv"))
    g = f.groupby("moment").agg(sim=("sim", "mean"), sd=("sim", "std"), target=("target", "first"), se=("se", "first"))
    names = {"ctrl_prac": "Control practice score", "ctrl_exam": "Control exam score",
             "base_prac_eff": "GPT Base effect, practice", "tutor_prac_eff": "GPT Tutor effect, practice",
             "base_exam_eff": "GPT Base effect, exam", "tutor_exam_eff": "GPT Tutor effect, exam",
             "exam_gpa_slope": "Exam slope on prior GPA", "base_answer_share_s1": "Answer-seeking share, Base (S1)",
             "tutor_answer_share_s1": "Answer-seeking share, Tutor (S1)", "ctrl_perceived": "Perceived performance, Control",
             "base_perceived_eff": "GPT Base effect, perceived perf.", "tutor_perceived_eff": "GPT Tutor effect, perceived perf."}
    rows = []
    for k, n in names.items():
        r = g.loc[k]; z = (r.sim - r.target) / r.se
        rows.append(f"{n} & ${r.target:+.3f}$ & ${r.se:.3f}$ & ${r.sim:+.3f}$ & ${z:+.2f}$ \\\\")
    zz = (g.sim - g.target) / g.se
    mac("fitMaxZ", f"{zz.abs().max():.2f}")
    mac("fitJ", f"{(zz ** 2).sum():.2f}")
    s = r"""\begin{table}[t]
\caption{Calibration fit on five held-out simulation seeds: every targeted moment is reproduced within one
empirical standard error ($|z|\le """ + f"{zz.abs().max():.2f}" + r"""$).}
\label{tab:fit}
\footnotesize
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}lcccc}
\toprule
Moment & Target & SE & Simulated & $z$ \\
\midrule
""" + "\n".join(rows) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    _w("tab_fit.tex", s)
    # out-of-sample validation correlation
    het = pd.read_csv(os.path.join(RES, "real_heterogeneity.csv"))
    vs = pd.read_csv(os.path.join(RES, "validation_sim.csv")).groupby(["by", "level", "outcome", "term"]).sim.mean().reset_index()
    het["level"] = het.level.astype(str); vs["level"] = vs.level.astype(str)
    m = het.merge(vs, on=["by", "level", "outcome", "term"])
    mac("oosR", f"{np.corrcoef(m.coef, m.sim)[0, 1]:.2f}")
    mac("oosN", f"{len(m)}")
    e = m[m.outcome == "Part3Tot"]
    mac("oosRexam", f"{np.corrcoef(e.coef, e.sim)[0, 1]:.2f}")
    mac("oosCover", f"{100 * np.mean(np.abs(m.coef - m.sim) <= 1.96 * m.se):.0f}")
    pr = pd.read_csv(os.path.join(RES, "real_problem_level.csv")).dropna(subset=["base_minus_ctrl"])
    ps = pd.read_csv(os.path.join(RES, "validation_items_sim.csv"))
    mac("probSlopeReal", f"{np.polyfit(pr.q_base, pr.base_minus_ctrl, 1)[0]:.2f}")
    mac("probSlopeSim", f"{np.polyfit(ps.q_base, ps.base_minus_ctrl, 1)[0]:.2f}")
    mac("probR", f"{np.corrcoef(pr.q_base, pr.base_minus_ctrl)[0, 1]:.2f}")


def t_main():
    p = pd.read_csv(os.path.join(RES, "main_per_seed.csv"))
    con = pd.read_csv(os.path.join(RES, "main_contrasts.csv"))
    ctrl = p[p.policy == "Control (no AI)"].set_index("seed")
    cols = [("practice", 3), ("exam_exp", 3), ("nlg", 3), ("delayed", 3), ("overconf", 3), ("calib_err", 3),
            ("leak", 2), ("off", 2)]
    best = {}
    for c, _ in cols:
        g = p.groupby("policy")[c].mean()
        if c in ("exam_exp", "nlg", "delayed"):
            best[c] = g.idxmax()
        elif c in ("calib_err",):
            best[c] = g.idxmin()
    rows = []
    for pol in ORDER:
        d = p[p.policy == pol].set_index("seed")
        cells = []
        for c, nd in cols:
            mu, ci = _ci(d[c])
            txt = f"{mu:.{nd}f}"
            if best.get(c) == pol:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        dx = (d.exam_exp - ctrl.exam_exp)
        rel = 100 * dx.mean() / ctrl.exam_exp.mean()
        cells.insert(2, f"{100 * dx.mean():+.2f} ({rel:+.1f}\\%)" if pol != "Control (no AI)" else "--")
        rows.append(SHORT[pol] + " & " + " & ".join(cells) + " \\\\")
    s = r"""\begin{table*}[t]
\caption{Main results on 10 held-out test seeds (825 real student profiles $\times$ 4 sessions each; mean over seeds,
all 95\% half-widths $\le$ """ + f"{_maxci(p):.3f}" + r"""). Practice is assisted; all other learning outcomes are unassisted.
$\Delta$Exam is in percentage points relative to Control. NLG: normalised learning gain; Overconf.: perceived minus
actual exam score; Cal.\ err.: absolute calibration error; Leak: share of steps that reveal the full solution;
Off: off-task share of steps. Best value per learning column in bold.}
\label{tab:main}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccccccccc}
\toprule
Policy & Practice & Exam & $\Delta$Exam (pp) & NLG & Delayed & Overconf. & Cal.\ err. & Leak & Off \\
\midrule
""" + "\n".join(rows) + r"""
\botrule
\end{tabular*}
\end{table*}
"""
    _w("tab_main.tex", s)
    # contrasts vs TRACE
    rows = []
    c2 = con[con.ref == TRACE].set_index("policy")
    c1 = con[con.ref == "Control (no AI)"].set_index("policy")
    for pol in ORDER:
        if pol == TRACE:
            continue
        r = c2.loc[pol]
        rows.append(f"{SHORT[pol]} & ${-100 * r['diff']:+.2f}$ & $[{-100 * r.hi:+.2f}, {-100 * r.lo:+.2f}]$ & "
                    f"${-r.d_cohen:.3f}$ & {_p(r.p_holm)} \\\\")
    s = r"""\begin{table}[t]
\caption{Paired contrasts of TRACE against each policy on the unassisted exam (percentage points; positive favours
TRACE). CI: student-cluster bootstrap (1{,}000 resamples, pooled over seeds); $d$: standardised by the
comparison policy's student-level SD; $p$: paired $t$-test over seeds, Holm-adjusted.}
\label{tab:contrasts}
\footnotesize
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}lcccc}
\toprule
TRACE vs. & $\Delta$ (pp) & 95\% CI & $d$ & $p_{\mathrm{Holm}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    _w("tab_contrasts.tex", s)
    g = p.groupby("policy").mean(numeric_only=True)
    cm = g.loc["Control (no AI)"]
    for key, pol in (("Trace", TRACE), ("Base", "GPT Base"), ("Tutor", "GPT Tutor (static)"),
                     ("Greedy", "Greedy learning-gain"), ("Ladder", "Mastery ladder"), ("Fade", "Fixed fading"),
                     ("Maxc", "Max-correctness"), ("Ctrl", "Control (no AI)")):
        r = g.loc[pol]
        mac(f"exam{key}", f"{r.exam_exp:.3f}"); mac(f"prac{key}", f"{r.practice:.3f}")
        mac(f"dExam{key}", f"{100 * (r.exam_exp - cm.exam_exp):+.2f}")
        mac(f"relExam{key}", f"{100 * (r.exam_exp - cm.exam_exp) / cm.exam_exp:+.1f}")
        mac(f"nlg{key}", f"{r.nlg:.3f}"); mac(f"oc{key}", f"{r.overconf:.3f}"); mac(f"ce{key}", f"{r.calib_err:.3f}")
        mac(f"off{key}", f"{100 * r.off:.1f}"); mac(f"del{key}", f"{r.delayed:.3f}"); mac(f"rEnd{key}", f"{r.r_end:.2f}")
        mac(f"leak{key}", f"{100 * r.leak:.1f}")
    T, Tu, G = g.loc[TRACE], g.loc["GPT Tutor (static)"], g.loc["Greedy learning-gain"]
    mac("nlgGainVsCtrl", f"{100 * (T.nlg / cm.nlg - 1):.0f}")
    mac("nlgGainVsTutor", f"{100 * (T.nlg / Tu.nlg - 1):.0f}")
    mac("dExamTraceTutor", f"{100 * (T.exam_exp - Tu.exam_exp):.2f}")
    mac("relExamTraceTutor", f"{100 * (T.exam_exp - Tu.exam_exp) / Tu.exam_exp:.1f}")
    mac("ocCutVsTutor", f"{100 * (1 - T.overconf / Tu.overconf):.0f}")
    mac("ceCutVsTutor", f"{100 * (1 - T.calib_err / Tu.calib_err):.0f}")
    mac("ocCutVsCtrl", f"{100 * (1 - T.overconf / cm.overconf):.0f}")
    r = c2.loc["Greedy learning-gain"]
    mac("pGreedy", _p(r.p_holm).strip("$")); mac("dGreedy", f"{-100 * r['diff']:.2f}")
    mac("dCohenTraceCtrl", f"{c1.loc[TRACE].d_cohen:.3f}")
    mac("dCohenBaseCtrl", f"{c1.loc['GPT Base'].d_cohen:.3f}")
    mac("ciTraceCtrlLo", f"{100 * c1.loc[TRACE].lo:.2f}"); mac("ciTraceCtrlHi", f"{100 * c1.loc[TRACE].hi:.2f}")


def _maxci(p):
    out = 0
    for pol, d in p.groupby("policy"):
        for c in ("practice", "exam_exp", "nlg", "delayed", "overconf", "calib_err"):
            out = max(out, _ci(d[c])[1])
    return out


def _p(p):
    if not np.isfinite(p):
        return "--"
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def t_ablation():
    a = pd.read_csv(os.path.join(RES, "ablation_per_seed.csv"))
    order = ["TRACE (ours)", "TRACE - reliance state", "TRACE - frustration model", "TRACE - calibration term",
             "TRACE - lookahead (myopic)", "TRACE - all (belief only)"]
    rows = []
    for o in order:
        d = a[a.policy == o]
        nm = o.replace("TRACE - ", "$-$ ").replace("TRACE (ours)", "TRACE (full)")
        rows.append(f"{nm} & {100 * d.d_exam.mean():+.2f} & {d.nlg.mean():.3f} & {d.overconf.mean():.3f} & "
                    f"{d.calib_err.mean():.3f} & {100 * d.off.mean():.1f} & {d.practice.mean():.3f} \\\\")
    g = a.groupby("policy").mean(numeric_only=True)
    mac("ablMyopicD", f"{100 * g.loc['TRACE - lookahead (myopic)'].d_exam:+.2f}")
    mac("ablMyopicOC", f"{g.loc['TRACE - lookahead (myopic)'].overconf:.3f}")
    mac("ablNoCD", f"{100 * g.loc['TRACE - calibration term'].d_exam:+.2f}")
    mac("ablNoCOC", f"{g.loc['TRACE - calibration term'].overconf:.3f}")
    mac("ablNoROff", f"{100 * g.loc['TRACE - reliance state'].off:.1f}")
    mac("ablNoFOff", f"{100 * g.loc['TRACE - frustration model'].off:.1f}")
    mac("ablAllD", f"{100 * g.loc['TRACE - all (belief only)'].d_exam:+.2f}")
    s = r"""\begin{table}[t]
\caption{Ablation of TRACE (10 test seeds). Each row removes one component; ``belief only'' keeps the particle
belief and greedy one-step selection.}
\label{tab:ablation}
\footnotesize
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}lcccccc}
\toprule
Variant & $\Delta$Exam & NLG & Overconf. & Cal.\ err. & Off (\%) & Practice \\
\midrule
""" + "\n".join(rows) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    _w("tab_ablation.tex", s)


def t_robust():
    f = os.path.join(RES, "robustness.csv")
    if not os.path.exists(f):
        return
    r = pd.read_csv(f)
    r["family"] = np.where(r.tag.str.startswith("bkt"), "bkt", "perturb")
    ctrl = r[r.policy == "Control (no AI)"].set_index("tag").exam_exp
    r["d"] = 100 * (r.exam_exp.values - ctrl.reindex(r.tag).values)
    tr = r[r.policy == TRACE].set_index("tag").exam_exp
    rows = []
    for pol in ORDER:
        if pol == "Control (no AI)":
            continue
        d = r[(r.policy == pol) & (r.family == "perturb")]
        b = r[(r.policy == pol) & (r.family == "bkt")]
        win = "--" if pol == TRACE else f"{100 * np.mean(tr.reindex(d.tag).values > d.exam_exp.values):.0f}"
        rows.append(f"{SHORT[pol]} & {d.d.median():+.2f} & [{d.d.quantile(.1):+.2f}, {d.d.quantile(.9):+.2f}] & "
                    f"{100 * (d.d > 0).mean():.0f} & {win} & {b.d.mean():+.2f} \\\\")
    d = r[(r.policy == TRACE) & (r.family == "perturb")]
    mac("robTraceMedian", f"{d.d.median():+.2f}"); mac("robTracePos", f"{100 * (d.d > 0).mean():.0f}")
    mac("robNdraws", f"{len(d)}")
    others = [o for o in ORDER if o not in (TRACE, "Control (no AI)")]
    wins = {o: np.mean(tr.reindex(r[(r.policy == o) & (r.family == 'perturb')].tag).values >
                       r[(r.policy == o) & (r.family == 'perturb')].exam_exp.values) for o in others}
    mac("robMinWin", f"{100 * min(wins.values()):.0f}")
    mac("robMinWinPolicy", min(wins, key=wins.get))
    gb = r[r.family == "bkt"].groupby("policy").d.mean()
    mac("bktTrace", f"{gb.loc[TRACE]:+.2f}"); mac("bktBase", f"{gb.loc['GPT Base']:+.2f}")
    mac("bktTutor", f"{gb.loc['GPT Tutor (static)']:+.2f}"); mac("bktGreedy", f"{gb.loc['Greedy learning-gain']:+.2f}")
    mac("bktBest", SHORT[gb.drop('Control (no AI)').idxmax()].split(" \\cite")[0])
    # Pareto membership on (exam up, overconfidence down) per draw; overconfidence win-rates
    pm = {}
    for tag, d in r.groupby("tag"):
        d = d.set_index("policy")
        for pol in d.index:
            dom = ((d.exam_exp >= d.loc[pol, "exam_exp"]) & (d.overconf <= d.loc[pol, "overconf"]) &
                   ((d.exam_exp > d.loc[pol, "exam_exp"]) | (d.overconf < d.loc[pol, "overconf"]))).any()
            pm.setdefault(pol, []).append((tag.startswith("bkt"), not dom))
    pareto = {pol: 100 * np.mean([x for b, x in v if not b]) for pol, v in pm.items()}
    pareto_b = {pol: 100 * np.mean([x for b, x in v if b]) for pol, v in pm.items()}
    trc = r[r.policy == TRACE].set_index("tag").overconf
    gre = r[r.policy == "Greedy learning-gain"].set_index("tag")
    pert = [t for t in gre.index if not t.startswith("bkt")]
    mac("robOCwinGreedy", f"{100 * np.mean(trc.loc[pert] < gre.loc[pert, 'overconf']):.0f}")
    mac("robParetoTrace", f"{pareto[TRACE]:.0f}"); mac("robParetoTraceBkt", f"{pareto_b[TRACE]:.0f}")
    mac("robParetoGreedy", f"{pareto['Greedy learning-gain']:.0f}"); mac("robParetoFade", f"{pareto['Fixed fading']:.0f}")
    rows = [rw.replace(" \\\\", "") for rw in rows]
    rows = [rw + f" & {pareto[pol]:.0f} \\\\" for rw, pol in zip(rows, [o for o in ORDER if o != "Control (no AI)"])]
    s = r"""\begin{table}[t]
\caption{Robustness to simulator misspecification. Policies keep the nominal internal model while the ground-truth
learner is (i) re-drawn """ + f"{len(d)}" + r""" times with 14 mechanism parameters perturbed by log-normal noise
($\sigma=0.30$) or (ii) replaced by a separately calibrated Bayesian-knowledge-tracing learner (5 seeds).
Exam effects vs Control in percentage points. Pareto: share of perturbed draws in which the policy is not dominated
on (exam score, overconfidence) by any other policy.}
\label{tab:robust}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccccc}
\toprule
& \multicolumn{4}{c}{Perturbed logistic learners} & BKT learners & \\
\cmidrule{2-5}\cmidrule{6-6}
Policy & Median $\Delta$ & 10--90\% range & \% draws $>0$ & \% TRACE better & Mean $\Delta$ & Pareto (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\botrule
\end{tabular*}
\end{table}
"""
    _w("tab_robust.tex", s)


def t_misc():
    a = pd.read_csv(os.path.join(RES, "ai_accuracy.csv"))
    g = a.groupby(["q", "policy"]).exam_exp.mean().unstack()
    c = g["Control (no AI)"]
    qn = {0.51: "Raw", 0.75: "Mid", 0.9: "High", 0.97: "Grounded"}
    for q in g.index:
        mac(f"accTrace{qn[round(q, 2)]}", f"{100 * (g.loc[q, TRACE] - c.loc[q]):+.2f}")
        mac(f"accTutor{qn[round(q, 2)]}", f"{100 * (g.loc[q, 'GPT Tutor (static)'] - c.loc[q]):+.2f}")
    h = pd.read_csv(os.path.join(RES, "horizon.csv"))
    hb = h[h.policy == "GPT Base"].groupby("session").r_end.mean()
    mac("hBaseRfirst", f"{hb.iloc[0]:.2f}"); mac("hBaseRlast", f"{hb.iloc[-1]:.2f}")
    ht = h[h.policy == TRACE].groupby("session").r_end.mean()
    mac("hTraceRmin", f"{ht.min():.2f}"); mac("hTraceRmax", f"{ht.max():.2f}")
    ctrl = h[h.policy == "Control (no AI)"].groupby("session").exam_exp.mean()
    for key, pol in (("Trace", TRACE), ("Tutor", "GPT Tutor (static)"), ("Base", "GPT Base"), ("Greedy", "Greedy learning-gain")):
        d = h[h.policy == pol].groupby("session").exam_exp.mean() - ctrl
        mac(f"h{key}Mean", f"{100 * d.mean():+.2f}")
    tm = pd.read_csv(os.path.join(RES, "timing.csv")).set_index("policy")
    mac("timeTraceUs", f"{tm.loc[TRACE].us_per_student_decision:.0f}")
    mac("timeTraceRun", f"{tm.loc[TRACE].seconds:.1f}")
    eq = pd.read_csv(os.path.join(RES, "equity.csv"))
    ce = eq[eq.policy == "Control (no AI)"].set_index(["seed", "tercile"]).exam_exp
    for key, pol in (("Trace", TRACE), ("Base", "GPT Base"), ("Tutor", "GPT Tutor (static)"), ("Greedy", "Greedy learning-gain")):
        d = (eq[eq.policy == pol].set_index(["seed", "tercile"]).exam_exp - ce).groupby(level="tercile").mean()
        for t in ("low", "mid", "high"):
            mac(f"eq{key}{t.capitalize()}", f"{100 * d.loc[t]:+.2f}")
    ad = pd.read_csv(os.path.join(RES, "trace_actions.csv")).groupby("tercile").mean(numeric_only=True)
    mac("actWorkedLow", f"{100 * ad.loc['low', 'act_worked']:.0f}"); mac("actWorkedHigh", f"{100 * ad.loc['high', 'act_worked']:.0f}")
    mac("actAloneHigh", f"{100 * (ad.loc['high', 'act_none']):.0f}"); mac("actReflectHigh", f"{100 * ad.loc['high', 'act_reflect']:.0f}")
    pa = pd.read_csv(os.path.join(RES, "pareto_per_seed.csv")).groupby("policy").mean(numeric_only=True)
    for w, k in (("0", "Zero"), ("0.1", "Tenth"), ("0.25", "Quarter"), ("0.5", "Half"), ("1", "One")):
        mac(f"par{k}Exam", f"{pa.loc[f'TRACE w_c={w}'].exam_exp:.3f}"); mac(f"par{k}OC", f"{pa.loc[f'TRACE w_c={w}'].overconf:.3f}")


def make_all():
    for f in (t_real, t_params, t_fit, t_main, t_ablation, t_robust, t_misc):
        f()
    def esc(v):
        return str(v).replace("%", r"\%").replace("_", r"\_")
    lines = ["% auto-generated by tracetutor/tables.py -- do not edit"]
    for k, v in MACROS.items():
        assert k.isalpha(), k
        lines.append(f"\\newcommand{{\\{k}}}{{{esc(v)}}}")
    _w("macros.tex", "\n".join(lines) + "\n")
    print(f"   {len(MACROS)} macros")
