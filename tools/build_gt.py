"""Builds eval/ground_truth.json from the reference crossings and the review.

The reference feed (tools/reference_tracks.py: YOLOv8x at 1280 px + ByteTrack)
is counted with the same lines, then every one of its crossings is checked by
eye on contact sheets (tools/contact_sheet.py), and the slit-scan of each line
(tools/slitscan.py) is searched for crossings it missed. eval/review.json
records the outcome:

  "fix":   {"<scene>/<line>/<tracker id>/<direction>": "fp" | {"category": ..., "direction": ...}}
           fp = not a real crossing (a rider counted as a pedestrian, a duplicate
           track); otherwise the corrected class or direction. The direction is
           part of the key because a tracker can hand one id to two road users
           going opposite ways; "<scene>/<line>/<tracker id>" also works.
  "added": [{"scene", "line", "frame", "category", "direction", "note"}]
           real crossings the reference missed
  "reviewed": ["<scene>/<line>", ...]   only these lines go into the ground truth

    python tools/build_gt.py data/reference_tracks.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crosscount import counter as counter_rules  # noqa: E402
from crosscount.offline import count_feed  # noqa: E402

# The review was done on the counter as it was before duplicate-box suppression
# existed (it was added because of what the review found). Turn it off so this
# rebuild sees the same reference crossings the reviewer saw.
counter_rules.DUP_IOU = 2.0


def main(feed):
    review = json.loads((ROOT / "eval" / "review.json").read_text())
    c, _, _ = count_feed(feed, ROOT / "config" / "scenes.json", ROOT / "eval" / "reel.json")
    fixes, reviewed = review.get("fix", {}), set(review.get("reviewed", []))
    gt, tally = {}, Counter()
    for e in c.events:
        key = f"{e.scene}/{e.line}"
        if key not in reviewed:
            continue
        fix = fixes.get(f"{key}/{e.tracker_id}/{e.direction}", fixes.get(f"{key}/{e.tracker_id}"))
        if fix == "fp":
            tally["removed"] += 1
            continue
        ev = {"frame": e.frame_index, "category": e.category, "direction": e.direction, "ref_track": e.tracker_id}
        if isinstance(fix, dict):
            ev.update(fix)
            tally["corrected"] += 1
        gt.setdefault(e.scene, {}).setdefault(e.line, {"events": []})["events"].append(ev)
        tally["kept"] += 1
    for a in review.get("added", []):
        if f"{a['scene']}/{a['line']}" in reviewed:
            gt.setdefault(a["scene"], {}).setdefault(a["line"], {"events": []})["events"].append(
                {k: a[k] for k in ("frame", "category", "direction")} | {"note": a.get("note", "added from slit-scan")})
            tally["added"] += 1
    for lines in gt.values():
        for body in lines.values():
            body["events"].sort(key=lambda x: x["frame"])
    out = {"_doc": "Hand-verified line crossings for the reel; built by tools/build_gt.py from eval/review.json.",
           "scenes": gt}
    (ROOT / "eval" / "ground_truth.json").write_text(json.dumps(out, indent=1))
    n = sum(len(b["events"]) for lines in gt.values() for b in lines.values())
    print(f"ground truth: {n} crossings on {len(reviewed)} lines  ({dict(tally)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "reference_tracks.jsonl"))
