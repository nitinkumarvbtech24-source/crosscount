"""Turns PipeGen's tracked detections into line-crossing counts per class.

Per track it keeps a class vote, a rider vote and, per counting line, the last
side of the line the track was clearly on. A crossing is counted when a track
is seen clearly on one side after being clearly on the other, and the step
between the two passes through the drawn segment (not the line's extension).

Road-user categories:
  pedestrian   person not riding anything
  cyclist      bicycle (its rider is not counted again)
  two_wheeler  motorcycle / scooter (riders not counted again: one bike with
               two people on it is one two-wheeler)
  car, bus, truck
"""
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .geometry import Point, along, anchor, crossing_point, iou, overlap_share, signed_distance
from .schema import Frame

CATEGORIES = ("pedestrian", "cyclist", "two_wheeler", "car", "bus", "truck")
VEHICLE_CATEGORY = {"bicycle": "cyclist", "motorcycle": "two_wheeler", "car": "car", "bus": "bus", "truck": "truck"}
RIDEABLE = ("bicycle", "motorcycle")

DEAD_BAND = 0.006      # x frame height: ignore anchor jitter this close to the line
END_MARGIN = 0.03      # a crossing may pass this far (x line length) beyond either end
RIDER_OVERLAP = 0.25   # share of a person's box over a bike's box that makes them its rider
BIKE_INSIDE = 0.5      # ...or share of the bike's box inside the person's (one box drawn round rider and bike)
WALK_MAX = 1.6         # body heights per second: faster than any walk, so a "person" this fast is riding
                       # something (a missed bike, a bus seen through its window). Walking is ~0.8.
SPEED_WINDOW_S = 0.2   # speed is measured over at least this much track history
MIN_TIMED = 4          # speed readings needed before a person can be called a pedestrian
PENDING_MAX_S = 1.5    # a person's crossing still undecided this long after it happened is dropped
DUP_IOU = 0.6          # two vehicle boxes overlapping this much...
DUP_FRAMES = 3         # ...in this many frames are one vehicle boxed twice ("twins"). Two cars
                       # close together reach 0.45 for a frame or two; one van boxed twice sits at 0.6-0.9.
DUP_WINDOW_S = 1.5     # a twin's crossing of the same line, same way, this close in time is the same vehicle
TRACK_TTL_S = 3.0      # forget a track not seen for this long


@dataclass
class Line:
    name: str
    a: Point             # normalised 0..1
    b: Point
    pos: str = "pos"     # label for crossing onto the positive side
    neg: str = "neg"     # label for crossing onto the negative side


@dataclass
class Scene:
    name: str
    lines: List[Line]
    start_frame: Optional[int] = None   # None: every frame (a live camera)
    end_frame: Optional[int] = None


@dataclass
class Crossing:
    frame_index: int
    ts: float
    scene: str
    line: str
    tracker_id: int
    category: str
    direction: str
    point: Point


@dataclass
class _Track:
    votes: Counter = field(default_factory=Counter)
    person_frames: int = 0
    rider_frames: int = 0
    fast_frames: int = 0
    timed_frames: int = 0
    twins: Counter = field(default_factory=Counter)   # other track id -> frames boxed on the same vehicle
    counted_at: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (line, direction) -> time counted
    last_ts: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=30))   # (ts, ground point, box height)
    sides: Dict[str, Tuple[int, Point]] = field(default_factory=dict)
    counted: set = field(default_factory=set)
    pending: list = field(default_factory=list)   # crossings waiting for the class to be known

    def undecided(self) -> bool:
        """A person seen too briefly to tell walking from riding. Riders' tracks
        often start only a few frames before the line, so their first crossing
        has to wait for a few speed readings."""
        return bool(self.votes) and self.votes.most_common(1)[0][0] == "person" and self.timed_frames < MIN_TIMED

    def observe_speed(self, ts: float, p: Point, height: float):
        """Ground speed in body heights per second, over SPEED_WINDOW_S or more.
        Heights make it scale-free: a far cyclist and a near one read the same."""
        self.history.append((ts, p, height))
        old = next((h for h in self.history if ts - h[0] >= SPEED_WINDOW_S), None)
        if old is None:
            return
        heights = sorted(h[2] for h in self.history)
        speed = ((p[0] - old[1][0]) ** 2 + (p[1] - old[1][1]) ** 2) ** 0.5 / max(heights[len(heights) // 2], 1.0) / (ts - old[0])
        self.timed_frames += 1
        self.fast_frames += speed > WALK_MAX

    def category(self) -> Optional[str]:
        if not self.votes:
            return None
        cls = self.votes.most_common(1)[0][0]
        if cls == "person":
            # Riding for most of its life, or moving faster than anyone walks:
            # the vehicle's own track carries the count, not the person.
            if self.rider_frames * 2 > self.person_frames or self.fast_frames * 2 > self.timed_frames > 0:
                return None
            return "pedestrian"
        return VEHICLE_CATEGORY.get(cls)


def overlapping_vehicles(objects):
    """Pairs of vehicle boxes in this frame that overlap like one vehicle boxed
    twice. The detector suppresses overlapping boxes only within a class, so a
    van can carry a 'car' box and a 'truck' box, each with its own track."""
    vehicles = [o for o in objects if o.cls in VEHICLE_CATEGORY]
    for i, a in enumerate(vehicles):
        for b in vehicles[i + 1:]:
            if iou(a.box, b.box) >= DUP_IOU:
                yield a.tid, b.tid


def riders(objects) -> set:
    """Tracker ids of people sitting on a bicycle or motorcycle in this frame."""
    bikes = [o.box for o in objects if o.cls in RIDEABLE]
    out = set()
    for o in objects:
        if o.cls != "person":
            continue
        for b in bikes:
            # The bike overlaps the person's body (or sits inside a box drawn
            # round both) and is not wholly above them.
            overlaps = overlap_share(o.box, b) >= RIDER_OVERLAP or overlap_share(b, o.box) >= BIKE_INSIDE
            if overlaps and b[3] >= o.box[1] + 0.5 * (o.box[3] - o.box[1]):
                out.add(o.tid)
                break
    return out


class CrossCounter:
    def __init__(self, scenes: List[Scene], frame_size: Tuple[int, int]):
        self.scenes = scenes
        self.w, self.h = frame_size
        self.scene: Optional[Scene] = None
        self.tracks: Dict[int, _Track] = {}
        self.totals = defaultdict(lambda: defaultdict(Counter))   # scene -> line -> "category|direction" -> n
        self.events: List[Crossing] = []
        self.dropped: List[dict] = []   # crossings not counted (riders, passengers, duplicate boxes), with the reason

    def _scene_for(self, index: int) -> Optional[Scene]:
        for s in self.scenes:
            if s.start_frame is None or s.start_frame <= index < s.end_frame:
                return s
        return None

    def _px(self, p: Point) -> Point:
        return (p[0] * self.w, p[1] * self.h)

    def _resolve(self, tid: int, t: _Track, now: float, final: bool = False) -> List[Crossing]:
        """Count a track's pending crossings once its class is known (at once
        for vehicles; after a few speed readings for people)."""
        if not t.pending or (t.undecided() and not final and now - t.pending[0][3] < PENDING_MAX_S):
            return []
        cat = None if t.undecided() else t.category()
        out = []
        for line, direction, index, ts, x in t.pending:
            why = None
            if cat is None:
                if t.undecided():
                    why = "seen too briefly"
                else:
                    why = "riding" if t.rider_frames * 2 > t.person_frames else "faster than walking"
            elif self._twin_counted(t, line, direction, ts):
                why = "duplicate box"
            if why:
                # Kept for review: which crossings were not counted, and why.
                self.dropped.append({"frame_index": index, "scene": self.scene.name, "line": line,
                                     "tracker_id": tid, "direction": direction, "why": why})
                continue
            t.counted_at[(line, direction)] = ts
            c = Crossing(index, ts, self.scene.name, line, tid, cat, direction, (round(x[0], 1), round(x[1], 1)))
            self.totals[self.scene.name][line][f"{cat}|{direction}"] += 1
            self.events.append(c)
            out.append(c)
        t.pending = []
        return out

    def _twin_counted(self, t: _Track, line: str, direction: str, ts: float) -> bool:
        """Has a track boxed on this same vehicle already counted this line,
        this way, moments ago? Whichever of the two boxes crosses first counts."""
        for other, frames in t.twins.items():
            o = self.tracks.get(other)
            if frames >= DUP_FRAMES and o is not None:
                when = o.counted_at.get((line, direction))
                if when is not None and abs(ts - when) <= DUP_WINDOW_S:
                    return True
        return False

    def flush(self) -> List[Crossing]:
        """Settle every pending crossing (end of a feed, or a scene change)."""
        out = []
        if self.scene is not None:
            for tid, t in self.tracks.items():
                out += self._resolve(tid, t, 0.0, final=True)
        return out

    def update(self, frame: Frame) -> List[Crossing]:
        scene = self._scene_for(frame.index)
        out = []
        if scene is not self.scene:
            # New street: tracks from the last scene must not cross this one's lines.
            out += self.flush()
            self.scene, self.tracks = scene, {}
        if scene is None:
            return out
        if frame.size and tuple(frame.size) != (self.w, self.h):
            # Boxes arrive in the stream muxer's resolution; the lines are drawn
            # on the video's. Bring the boxes over before anything is measured.
            sx, sy = self.w / frame.size[0], self.h / frame.size[1]
            for o in frame.objects:
                o.box = (o.box[0] * sx, o.box[1] * sy, o.box[2] * sx, o.box[3] * sy)
        ridden = riders(frame.objects)
        for a, b in overlapping_vehicles(frame.objects):
            self.tracks.setdefault(a, _Track()).twins[b] += 1
            self.tracks.setdefault(b, _Track()).twins[a] += 1
        band = DEAD_BAND * self.h
        for o in frame.objects:
            t = self.tracks.setdefault(o.tid, _Track())
            t.votes[o.cls] += o.conf
            t.last_ts = frame.ts
            p = anchor(o.box)
            if o.cls == "person":
                t.person_frames += 1
                t.rider_frames += o.tid in ridden
                t.observe_speed(frame.ts, p, o.box[3] - o.box[1])
            for line in scene.lines:
                a, b = self._px(line.a), self._px(line.b)
                d = signed_distance(a, b, p)
                if abs(d) <= band:
                    continue
                side = 1 if d > 0 else -1
                prev = t.sides.get(line.name)
                t.sides[line.name] = (side, p)
                if prev is None or prev[0] == side:
                    continue
                x = crossing_point(a, b, prev[1], p)
                if x is None or not -END_MARGIN <= along(a, b, x) <= 1 + END_MARGIN:
                    continue
                direction = line.pos if side > 0 else line.neg
                if (line.name, direction) in t.counted:
                    continue
                t.counted.add((line.name, direction))
                t.pending.append((line.name, direction, frame.index, frame.ts, x))
        for tid, t in list(self.tracks.items()):
            gone = frame.ts - t.last_ts > TRACK_TTL_S
            out += self._resolve(tid, t, frame.ts, final=gone)
            if gone:
                del self.tracks[tid]
        return out

    def summary(self) -> dict:
        """{scene: {line: {category: {direction: n}}}}"""
        out = {}
        for scene, lines in self.totals.items():
            for line, c in lines.items():
                for key, n in c.items():
                    cat, direction = key.split("|")
                    out.setdefault(scene, {}).setdefault(line, {}).setdefault(cat, {})[direction] = n
        return out


def load_scenes(config_path, manifest_path=None, only=None) -> Tuple[List[Scene], Tuple[int, int]]:
    """Scenes and counting lines from config/scenes.json. With the reel's
    manifest, each scene gets its frame range; without it, one scene (`only`,
    or the first) applies to every frame: a single live camera."""
    cfg = json.loads(Path(config_path).read_text())
    ranges = {}
    if manifest_path:
        for s in json.loads(Path(manifest_path).read_text())["scenes"]:
            ranges[s["scene"]] = (s["start_frame"], s["end_frame"])
    scenes = []
    for name, sc in cfg["scenes"].items():
        if ranges and name not in ranges:
            continue
        start, end = ranges.get(name, (None, None))
        lines = [Line(l["name"], tuple(l["a"]), tuple(l["b"]), l.get("pos", "pos"), l.get("neg", "neg"))
                 for l in sc["lines"]]
        scenes.append(Scene(name, lines, start, end))
    if not ranges:
        scenes = [s for s in scenes if s.name == only] if only else scenes[:1]
        if not scenes:
            raise ValueError(f"no scene named {only!r} in {config_path}")
    return scenes, tuple(cfg.get("frame_size", (1280, 720)))


def crossing_dict(c: Crossing) -> dict:
    return asdict(c)
