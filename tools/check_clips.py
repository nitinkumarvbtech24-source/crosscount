"""Is each downloaded clip usable as a fixed traffic camera?

A line-crossing count assumes the line stays on the same piece of road, so
a clip that pans, zooms or shakes is out. For each clip this samples a frame
every half second, measures how far the whole picture has shifted from the
first sample (phase correlation, which the static background dominates), and
writes a mid-clip still for picking the counting line.

    python tools/check_clips.py <raw clip folder> <still output folder>
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def drift(path: Path, still_dir: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / 2)))
    ref, shifts, i = None, [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            _, frame = cap.retrieve()
            if i // step == (n // step) // 2:
                cv2.imwrite(str(still_dir / f"{path.stem}.jpg"), cv2.resize(frame, (960, int(960 * h / w))))
            g = cv2.cvtColor(cv2.resize(frame, (480, int(480 * h / w))), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if ref is None:
                ref = g
            else:
                (dx, dy), _ = cv2.phaseCorrelate(ref, g)
                shifts.append(np.hypot(dx, dy) * w / 480)   # back to source pixels
        i += 1
    cap.release()
    s = np.array(shifts or [0.0])
    return fps, i, w, h, float(np.median(s)), float(s.max())


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'clip':16s} {'fps':>5s} {'frames':>6s} {'secs':>5s} {'size':>9s} {'median shift':>12s} {'max shift':>9s}")
    for p in sorted(src.glob("*.mp4")):
        fps, n, w, h, med, mx = drift(p, out)
        print(f"{p.stem:16s} {fps:5.1f} {n:6d} {n / fps:5.1f} {w}x{h} {med:10.1f}px {mx:8.1f}px")
