"""Reproduce every result, table and figure of the paper.

Usage (from the project root, e.g. in the VS Code terminal):
    python run_all.py                 # everything (~25-40 min on a laptop)
    python run_all.py --stages main ablation figures tables
    python run_all.py --repo /path/to/GenAICanHarmLearning
"""
import argparse
import os
import time

STAGES = ["replication", "calibration", "bkt", "main", "ablation", "pareto", "robust",
          "ai_accuracy", "horizon", "timing", "figures", "tables"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--repo", default=None, help="path to a clone of obastani/GenAICanHarmLearning")
    ap.add_argument("--recalibrate", action="store_true")
    a = ap.parse_args()
    if a.repo:
        os.environ["TRACE_REPO"] = a.repo
    from tracetutor import experiments as E
    from tracetutor.data import load_study
    study = load_study(os.environ.get("TRACE_REPO", os.path.join("data", "GenAICanHarmLearning")))
    P = Pb = None
    for st in a.stages:
        t = time.time(); print(f"== {st} ==", flush=True)
        if st == "replication":
            print(E.stage_replication(study).round(4).to_string())
        elif st == "calibration":
            P = E.stage_calibration(study, force=a.recalibrate)
        elif st == "figures":
            from tracetutor import figures; figures.make_all()
        elif st == "tables":
            from tracetutor import tables; tables.make_all()
        else:
            if P is None:
                P = E.stage_calibration(study)
            if st == "bkt":
                Pb = E.stage_bkt(study, P, force=a.recalibrate)
            elif st == "robust":
                Pb = Pb or E.stage_bkt(study, P)
                E.stage_robust(study, P, Pb)
            else:
                getattr(E, f"stage_{st}")(study, P)
        print(f"   done in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
