"""Slit-scan of a counting line: what crossed it, independent of any detector.

Every frame, the pixels under the counting line become one row; stacking the
rows over time gives a picture where each road user that crossed the line is
a separate streak (time runs down, position along the line runs across).
Counted crossings are marked on it, so a streak with no mark is a crossing the
counter missed. This is how misses are found when building the ground truth.

    python tools/slitscan.py <reel.mp4> <reel.json> <events.jsonl> <scene> <line> <out folder>
        [--config config/scenes.json] [--step 2] [--rows 360]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reel")
    ap.add_argument("manifest")
    ap.add_argument("events")
    ap.add_argument("scene")
    ap.add_argument("line")
    ap.add_argument("out")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config" / "scenes.json"))
    ap.add_argument("--step", type=int, default=2, help="use every Nth frame")
    ap.add_argument("--rows", type=int, default=360, help="rows per output image")
    a = ap.parse_args()

    sc = next(s for s in json.load(open(a.manifest))["scenes"] if s["scene"] == a.scene)
    line = next(l for l in json.load(open(a.config))["scenes"][a.scene]["lines"] if l["name"] == a.line)
    events = [json.loads(l) for l in open(a.events) if l.strip()]
    events = [e for e in events if e["scene"] == a.scene and e["line"] == a.line]
    cap = cv2.VideoCapture(a.reel)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ax, ay, bx, by = line["a"][0] * w, line["a"][1] * h, line["b"][0] * w, line["b"][1] * h
    n = int(np.hypot(bx - ax, by - ay))
    xs = np.linspace(ax, bx, n).astype(np.float32)
    ys = np.linspace(ay, by, n).astype(np.float32)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sc["start_frame"])
    rows, frames = [], []
    for f in range(sc["start_frame"], sc["end_frame"]):
        ok, img = cap.read()
        if not ok:
            break
        if (f - sc["start_frame"]) % a.step:
            continue
        rows.append(cv2.remap(img, xs[None, :], ys[None, :], cv2.INTER_LINEAR)[0])
        frames.append(f)
    strip = np.stack(rows)                                  # time x position x 3
    scale = max(1, 900 // n) if n < 900 else 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for part, start in enumerate(range(0, len(strip), a.rows)):
        img = np.ascontiguousarray(strip[start:start + a.rows])
        img = cv2.resize(img, (n * scale, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((img.shape[0], img.shape[1] + 70, 3), np.uint8)
        canvas[:, 70:] = img
        f0, f1 = frames[start], frames[min(start + a.rows, len(frames)) - 1]
        for r in range(0, img.shape[0], 30):   # frame numbers down the left edge
            cv2.putText(canvas, str(frames[start + r // 2] if start + r // 2 < len(frames) else ""), (2, r + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        for e in events:
            if f0 <= e["frame_index"] <= f1:
                r = (e["frame_index"] - f0) // a.step * 2
                t = ((e["point"][0] - ax) * (bx - ax) + (e["point"][1] - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)
                x = 70 + int(t * n * scale)
                cv2.drawMarker(canvas, (x, r), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(canvas, f"{e['category'][:3]} {e['direction'][:6]}", (x + 8, r + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        cv2.imwrite(str(out / f"{a.scene}_{a.line}_{part + 1:02d}.png"), canvas)
    print(f"{len(frames)} rows, line {n} px, {len(events)} counted crossings marked -> {out}")


if __name__ == "__main__":
    main()
