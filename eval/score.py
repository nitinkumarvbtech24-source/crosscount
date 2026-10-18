"""Scores a recorded feed's crossings against the hand-verified ground truth.

A predicted crossing matches a true one on the same scene and line when the
class and direction agree and the two are at most MATCH_S apart; each true
crossing matches at most once (closest first). From the matches:

  recall     true crossings that were counted, with the right class and direction
  precision  counted crossings that are real, with the right class and direction
  count err  |counted - true| / true per class: what a city dashboard would show

"Any class" repeats the matching ignoring class, which separates the detector
and tracker missing a road user from it being counted under the wrong class.

    python -m eval.score <feed.jsonl> [--name fp16] [--gt eval/ground_truth.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from crosscount.counter import CATEGORIES
from crosscount.offline import count_feed

ROOT = Path(__file__).resolve().parent.parent
MATCH_S = 1.5
FPS = 30.0


def match(pred, true, use_class=True):
    """pred/true: lists of (frame, category, direction). Returns matched pairs."""
    pairs, used = [], set()
    cands = []
    for i, p in enumerate(pred):
        for j, t in enumerate(true):
            if p[2] != t[2] or (use_class and p[1] != t[1]):
                continue
            gap = abs(p[0] - t[0]) / FPS
            if gap <= MATCH_S:
                cands.append((gap, i, j))
    taken_p = set()
    for gap, i, j in sorted(cands):
        if i in taken_p or j in used:
            continue
        taken_p.add(i)
        used.add(j)
        pairs.append((pred[i], true[j]))
    return pairs


def score(feed, gt_path, config, manifest):
    gt = json.loads(Path(gt_path).read_text())["scenes"]
    c, frames, secs = count_feed(feed, config, manifest)
    pred = defaultdict(list)
    for e in c.events:
        pred[(e.scene, e.line)].append((e.frame_index, e.category, e.direction))
    per_cat = {k: {"tp": 0, "pred": 0, "true": 0} for k in CATEGORIES}
    any_tp = any_pred = any_true = 0
    per_scene = {}
    for scene, lines in gt.items():
        s_tp = s_pred = s_true = 0
        for line, body in lines.items():
            true = [(t["frame"], t["category"], t["direction"]) for t in body["events"]]
            p = pred.get((scene, line), [])
            for x in p:
                per_cat[x[1]]["pred"] += 1
            for x in true:
                per_cat[x[1]]["true"] += 1
            pairs = match(p, true)
            for _, t in pairs:
                per_cat[t[1]]["tp"] += 1
            any_tp += len(match(p, true, use_class=False))
            any_pred += len(p)
            any_true += len(true)
            s_tp, s_pred, s_true = s_tp + len(pairs), s_pred + len(p), s_true + len(true)
        per_scene[scene] = {"tp": s_tp, "pred": s_pred, "true": s_true}
    return per_cat, per_scene, (any_tp, any_pred, any_true), frames, secs


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def report(name, per_cat, per_scene, any_counts, frames, secs) -> str:
    out = [f"## {name}", "", "| Class | True | Counted | Matched | Recall | Precision | Count error |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    T = P = M = 0
    for k, v in per_cat.items():
        if not v["true"] and not v["pred"]:
            continue
        err = f"{100 * abs(v['pred'] - v['true']) / v['true']:.1f}%" if v["true"] else "-"
        out.append(f"| {k} | {v['true']} | {v['pred']} | {v['tp']} | {pct(v['tp'], v['true'])} | {pct(v['tp'], v['pred'])} | {err} |")
        T, P, M = T + v["true"], P + v["pred"], M + v["tp"]
    out.append(f"| **all, class must match** | {T} | {P} | {M} | {pct(M, T)} | {pct(M, P)} | {pct(abs(P - T), T)} |")
    a_tp, a_pred, a_true = any_counts
    out.append(f"| **all, any class** | {a_true} | {a_pred} | {a_tp} | {pct(a_tp, a_true)} | {pct(a_tp, a_pred)} | |")
    out += ["", "| Scene | True | Counted | Matched (class + direction) |", "|---|---:|---:|---:|"]
    for s, v in per_scene.items():
        out.append(f"| {s} | {v['true']} | {v['pred']} | {v['tp']} |")
    out += ["", f"{frames} frames; counting took {1000 * secs / max(frames, 1):.3f} ms per frame on the CPU.", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feed")
    ap.add_argument("--name", default=None)
    ap.add_argument("--gt", default=str(ROOT / "eval" / "ground_truth.json"))
    ap.add_argument("--config", default=str(ROOT / "config" / "scenes.json"))
    ap.add_argument("--manifest", default=str(ROOT / "eval" / "reel.json"))
    ap.add_argument("--out", help="also write the report (markdown) here")
    a = ap.parse_args()
    md = report(a.name or Path(a.feed).stem, *score(a.feed, a.gt, a.config, a.manifest))
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)


if __name__ == "__main__":
    main()
