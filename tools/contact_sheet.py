"""Crops every counted crossing out of the reel and tiles them into sheets,
numbered, so each count can be checked by eye (eval/README.md: building the
ground truth). Each tile shows the object's box at the moment it crossed,
its class and direction, and the counting line.

    python tools/contact_sheet.py <reel.mp4> <feed.jsonl> <events.jsonl> <out folder>
        [--config config/scenes.json] [--only category] [--scene name]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TILE_W, TILE_H, COLS, ROWS = 320, 240, 4, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reel")
    ap.add_argument("feed")
    ap.add_argument("events")
    ap.add_argument("out")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config" / "scenes.json"))
    ap.add_argument("--only")
    ap.add_argument("--scene")
    a = ap.parse_args()

    events = [json.loads(l) for l in open(a.events) if l.strip()]
    events = [e for e in events if (not a.only or e["category"] == a.only) and (not a.scene or e["scene"] == a.scene)]
    wanted = {e["frame_index"] for e in events}
    boxes = {}
    with open(a.feed) as f:
        for raw in f:
            d = json.loads(raw)
            if d["frame_index"] in wanted:
                boxes[d["frame_index"]] = {o["tracker_id"]: o for o in d["objects"]}
    lines = json.load(open(a.config))["scenes"]
    cap = cv2.VideoCapture(a.reel)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for n, e in enumerate(events, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["frame_index"])
        ok, img = cap.read()
        o = boxes.get(e["frame_index"], {}).get(e["tracker_id"])
        if not ok or o is None:
            continue
        h, w = img.shape[:2]
        for l in lines.get(e["scene"], {}).get("lines", []):
            cv2.line(img, (int(l["a"][0] * w), int(l["a"][1] * h)), (int(l["b"][0] * w), int(l["b"][1] * h)),
                     (238, 211, 34), 1)
        x1, y1, x2, y2 = (int(v) for v in o["bounding_box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Crop around the box with context, keeping the tile's 4:3 shape.
        bw, bh = max(x2 - x1, 40), max(y2 - y1, 40)
        cw = max(bw * 2.2, bh * 2.2 * TILE_W / TILE_H)
        ch = cw * TILE_H / TILE_W
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        X1, Y1 = int(max(0, min(w - cw, cx - cw / 2))), int(max(0, min(h - ch, cy - ch / 2)))
        crop = img[Y1:int(Y1 + ch), X1:int(X1 + cw)]
        tile = cv2.resize(crop, (TILE_W, TILE_H))
        label = f"{n} {e['category']} {e['direction']} #{e['tracker_id']}"
        cv2.rectangle(tile, (0, 0), (TILE_W, 20), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(tile)
    per = COLS * ROWS
    for s in range(0, len(tiles), per):
        sheet = np.zeros((TILE_H * ROWS, TILE_W * COLS, 3), np.uint8)
        for i, t in enumerate(tiles[s:s + per]):
            r, c = divmod(i, COLS)
            sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = t
        cv2.imwrite(str(out / f"sheet_{s // per + 1:02d}.jpg"), sheet)
    print(f"{len(tiles)} crossings -> {(len(tiles) + per - 1) // per} sheets in {out}")


if __name__ == "__main__":
    main()
