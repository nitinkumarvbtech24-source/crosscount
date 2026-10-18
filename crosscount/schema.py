"""PipeGen's per-frame MQTT message, parsed into plain objects.

The shape PipeGen 0.5.1's generated DeepStream app publishes (frame_json.h in
the app's parsers/, payload mode "frame"):
  {"frame": 12, "source_id": 0, "pts_ns": 400000000, "width": 1280, "height": 720,
   "objects": [{"id": 3, "class_id": 2, "label": "car", "confidence": 0.91,
                "bbox": {"left": 10.0, "top": 20.0, "width": 50.0, "height": 30.0}}]}
"id" is null for an object the tracker has not confirmed yet. Boxes are in the
stream muxer's resolution ("width"/"height"), which need not be the video's.
The parser also accepts the other spellings seen in PipeGen projects
("frame_index", "timestamp_s", "tracker_id", "class", "bounding_box" as
[x1, y1, x2, y2]) so a change in the generated app's JSON does not silently
zero the counts.
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

# Class names as the detector may spell them -> COCO names used below.
CLASS_ALIASES = {
    "pedestrian": "person", "people": "person", "persons": "person",
    "bike": "bicycle", "cycle": "bicycle",
    "motorbike": "motorcycle", "motor": "motorcycle", "scooter": "motorcycle",
    "auto": "car", "van": "car",
    "lorry": "truck",
}


@dataclass
class Detection:
    tid: int
    cls: str
    conf: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Frame:
    index: int
    ts: float
    objects: List[Detection] = field(default_factory=list)
    size: Optional[Tuple[int, int]] = None   # coordinate space of the boxes, when the message says


def _box(o: dict) -> Tuple[float, float, float, float]:
    b = o.get("bounding_box", o.get("bbox", o.get("box", o.get("rect_params"))))
    if isinstance(b, dict):
        x, y = float(b.get("left", b.get("x", 0))), float(b.get("top", b.get("y", 0)))
        return (x, y, x + float(b.get("width", b.get("w", 0))), y + float(b.get("height", b.get("h", 0))))
    x1, y1, x2, y2 = (float(v) for v in b)
    return (x1, y1, x2, y2)


def normalise_class(name: str) -> str:
    n = str(name).strip().lower()
    return CLASS_ALIASES.get(n, n)


def parse(message: Union[bytes, str, dict], fps: float = 30.0) -> Frame:
    d = message if isinstance(message, dict) else json.loads(message)
    index = int(d.get("frame_index", d.get("frame_number", d.get("frame", 0))))
    if "timestamp_s" in d or "timestamp" in d:
        ts = float(d.get("timestamp_s", d.get("timestamp")))
    elif d.get("pts_ns") is not None:
        ts = int(d["pts_ns"]) / 1e9          # the buffer's presentation time
    else:
        ts = index / fps                     # speeds and timeouts still need a clock
    size = (int(d["width"]), int(d["height"])) if d.get("width") and d.get("height") else None
    objects = []
    for o in d.get("objects", d.get("detections", [])) or []:
        tid = o.get("tracker_id", o.get("object_id", o.get("id")))
        if tid is None or int(tid) < 0:
            continue  # untracked detections cannot be counted once
        objects.append(Detection(int(tid), normalise_class(o.get("class", o.get("label", ""))),
                                 float(o.get("confidence", o.get("conf", 1.0))), _box(o)))
    return Frame(index, ts, objects, size)
