"""Offline reference tracks for the reel, in PipeGen's MQTT message format.

This is NOT part of the product: PipeGen's TensorRT pipeline is. It is a slow,
large model (YOLOv8x at 1280 px + ByteTrack, PyTorch) used for two jobs:
  1. a first draft of the ground truth, which is then checked by hand
     (eval/README.md says how), and
  2. a stand-in feed with the same JSON shape PipeGen publishes, so CrossCount
     can be tested end to end before the PipeGen run exists.

Each output line is one frame:
  {"frame_index": 12, "timestamp_s": 0.4, "objects": [{"tracker_id": 3,
   "class": "car", "confidence": 0.91, "bounding_box": [x1, y1, x2, y2]}]}

    python tools/reference_tracks.py <reel.mp4> <out.jsonl> [--model yolov8x.pt] [--imgsz 1280]
"""
import argparse
import json
import time

import cv2
from ultralytics import YOLO

ROAD_USERS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reel")
    ap.add_argument("out")
    ap.add_argument("--model", default="yolov8x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.reel)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t0, i = time.time(), 0
    with open(a.out, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=a.imgsz, conf=a.conf,
                            classes=list(ROAD_USERS), verbose=False)[0]
            objs = []
            if r.boxes is not None and r.boxes.id is not None:
                for box, tid, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(),
                                               r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    objs.append({"tracker_id": int(tid), "class": ROAD_USERS[int(cls)],
                                 "confidence": round(float(conf), 3),
                                 "bounding_box": [round(v, 1) for v in box]})
            f.write(json.dumps({"frame_index": i, "timestamp_s": round(i / fps, 3), "objects": objs}) + "\n")
            i += 1
            if i % 500 == 0:
                print(f"{i}/{total} frames, {i / (time.time() - t0):.1f} fps", flush=True)
    print(f"done: {i} frames in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
