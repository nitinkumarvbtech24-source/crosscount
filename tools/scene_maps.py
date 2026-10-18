"""Per-scene pictures from the reel and a recorded feed:

  data/stills/<scene>.jpg   heavily blurred, darkened still for the dashboard
                            background (no face or plate is readable)
  <out>/<scene>_flows.jpg   every track's ground point drawn over a sharp
                            still, coloured by class, plus the counting lines:
                            shows where traffic actually flows, which is
                            where a counting line belongs

    python tools/scene_maps.py <reel.mp4> <reel.json> <feed.jsonl> <out folder> [config/scenes.json]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
COLOURS = {"person": (248, 189, 56), "bicycle": (128, 222, 74), "motorcycle": (36, 191, 251),
           "car": (252, 132, 192), "bus": (133, 113, 251), "truck": (184, 163, 148)}


def main(reel, manifest, feed, out, config=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "stills").mkdir(parents=True, exist_ok=True)
    scenes = json.load(open(manifest))["scenes"]
    lines_cfg = json.load(open(config))["scenes"] if config and Path(config).exists() else {}
    points = defaultdict(list)   # scene -> [(x, y, cls)]
    with open(feed) as f:
        for raw in f:
            d = json.loads(raw)
            s = next((s["scene"] for s in scenes if s["start_frame"] <= d["frame_index"] < s["end_frame"]), None)
            if s:
                for o in d["objects"]:
                    x1, y1, x2, y2 = o["bounding_box"]
                    points[s].append((int((x1 + x2) / 2), int(y2), o["class"]))
    cap = cv2.VideoCapture(reel)
    for s in scenes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, (s["start_frame"] + s["end_frame"]) // 2)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        bg = cv2.GaussianBlur(cv2.resize(img, (w // 4, h // 4)), (0, 0), 6)
        cv2.imwrite(str(ROOT / "data" / "stills" / f"{s['scene']}.jpg"), cv2.resize(bg, (w, h)))
        flows = (img * 0.55).astype("uint8")
        for x, y, cls in points[s["scene"]]:
            cv2.circle(flows, (x, y), 2, COLOURS.get(cls, (255, 255, 255)), -1)
        for line in lines_cfg.get(s["scene"], {}).get("lines", []):
            a = (int(line["a"][0] * w), int(line["a"][1] * h))
            b = (int(line["b"][0] * w), int(line["b"][1] * h))
            cv2.line(flows, a, b, (238, 211, 34), 3)
            cv2.putText(flows, line["name"], (a[0] + 6, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 211, 34), 2)
        for i in range(1, 10):
            cv2.putText(flows, f".{i}", (w * i // 10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(flows, f".{i}", (2, h * i // 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"{s['scene']}_flows.jpg"), flows)


if __name__ == "__main__":
    main(*sys.argv[1:6])
