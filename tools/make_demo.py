"""Renders the demo video from a PipeGen run.

Inputs are what a PipeGen run leaves behind: its saved output video (boxes
drawn and heads pixelated by the pipeline) and its MQTT feed recorded as JSONL.
The feed goes through the same CrossCount counter the live service runs, frame
by frame in step with the video, and each output frame shows:

  - the PipeGen video, with CrossCount's counting lines and a flash where each
    crossing happened
  - per-class totals for the scene, split by direction
  - a ticker of the latest crossings and the pipeline facts (model, precision)

framed by a title card, a how-it-works slide, closing results slides (from a
small JSON file of measured numbers) and any screenshots passed with --slide.

    python tools/make_demo.py --video <pipegen output.mp4> --feed data/pipegen_fp16.jsonl \
        --stats eval/results/demo_stats.json --slide ir_graph.png --slide speed_report.png \
        --out data/crosscount_demo.mp4

--video can be the raw reel for a dry run; then pass --draw-boxes --blur-heads
--watermark "TEST: reference tracks, not PipeGen" so the result cannot be
mistaken for a PipeGen run. The output is MPEG-4 (mp4v); re-encode to H.264
for upload with ffmpeg -c:v libx264 -pix_fmt yuv420p.
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crosscount.counter import CATEGORIES, CrossCounter, load_scenes  # noqa: E402
from crosscount.schema import parse  # noqa: E402

W, H, FPS = 1920, 1080, 30
VX, VY, VW, VH = 40, 120, 1280, 720            # where the pipeline video sits
PX = VX + VW + 36                               # right panel left edge
BG, PANEL, LINE, TEXT, MUTED, ACCENT = (11, 16, 23), (18, 26, 36), (31, 42, 55), (230, 237, 243), (139, 152, 168), (34, 211, 238)
COLOURS = {"pedestrian": (56, 189, 248), "cyclist": (74, 222, 128), "two_wheeler": (251, 191, 36),
           "car": (192, 132, 252), "bus": (251, 113, 133), "truck": (148, 163, 184)}
LABEL = {"pedestrian": "Pedestrians", "cyclist": "Cyclists", "two_wheeler": "Two-wheelers",
         "car": "Cars", "bus": "Buses", "truck": "Trucks"}
CLASS_OF = {"person": "pedestrian", "bicycle": "cyclist", "motorcycle": "two_wheeler", "car": "car", "bus": "bus", "truck": "truck"}
# Stretches of the reel worth watching (reel frame numbers, 30 fps): each scene's busiest part.
DEFAULT_SEGMENTS = [(0, 845), (2450, 3050), (3420, 4320), (7150, 7750), (7813, 8300)]
SCENE_TITLE = {"in_citycentre": "Indian city road, elevated", "nl_amsterdam": "Amsterdam junction",
               "wroclaw_junction": "Wroclaw junction", "tram_junction": "Tram stop crossing",
               "wroclaw_zebra": "Wroclaw zebra crossing"}


def font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        for base in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")):
            p = base / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F = {k: font(s, b) for k, (s, b) in {"h1": (54, True), "h2": (34, True), "b": (26, False), "bb": (26, True),
                                       "s": (21, False), "n": (46, True), "xs": (18, False)}.items()}


def rgb(c):
    return tuple(int(v) for v in c)


def slide(lines, sub=None):
    """A full-screen text slide: lines = [(text, font key, colour)]."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 300 if sub is None else 220
    for text, key, colour in lines:
        d.text((140, y), text, font=F[key], fill=colour)
        y += int(F[key].size * 1.6)
    if sub:
        sub(d, y + 30)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def how_it_works():
    def boxes(d, y):
        steps = [("Camera / video", "1280x720, 30 fps"), ("PipeGen pipeline", "DeepStream + TensorRT\nyolo11s detector + tracker\nheads pixelated in-pipeline"),
                 ("MQTT", "boxes, classes, track ids\nno pixels"), ("CrossCount", "rider + duplicate rules\ncounting lines per scene"),
                 ("Outputs", "per-class counts by direction\nMQTT events + dashboard")]
        x, bw, gap = 140, 300, 40
        for i, (title, body) in enumerate(steps):
            colour = ACCENT if i == 1 else (LINE if i != 3 else (74, 222, 128))
            d.rounded_rectangle((x, y, x + bw, y + 220), radius=16, fill=PANEL, outline=colour, width=3)
            d.text((x + 20, y + 20), title, font=F["bb"], fill=TEXT)
            d.multiline_text((x + 20, y + 70), body, font=F["s"], fill=MUTED, spacing=8)
            if i < len(steps) - 1:
                d.text((x + bw + 6, y + 90), "->", font=F["bb"], fill=MUTED)
            x += bw + gap
    return slide([("How it works", "h1", TEXT),
                  ("PipeGen does all the seeing. CrossCount turns its tracks into counts a city can act on.", "b", MUTED)], boxes)


def image_slide(path, caption):
    img = cv2.imread(str(path))
    canvas = np.full((H, W, 3), BG[::-1], np.uint8)
    if img is None:
        return canvas
    s = min((W - 160) / img.shape[1], (H - 220) / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    y, x = 150, (W - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((80, 50), caption, font=F["h2"], fill=TEXT)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def results_slide(stats):
    def body(d, y):
        for row in stats.get("rows", []):
            d.text((140, y), row[0], font=F["b"], fill=MUTED)
            d.text((820, y), row[1], font=F["bb"], fill=TEXT)
            y += 52
        for note in stats.get("notes", []):
            y += 10
            d.text((140, y), note, font=F["s"], fill=MUTED)
            y += 34
    return slide([(stats.get("title", "Measured"), "h1", TEXT)], body)


def pixelate_heads(frame, objs, scale):
    """Dry runs only: the raw reel has faces in it, PipeGen's output does not."""
    h, w = frame.shape[:2]
    for o in objs:
        if o.cls != "person":
            continue
        x1, y1, x2, y2 = (int(v * scale) for v in o.box)
        bh = y2 - y1
        hx1, hy1, hx2, hy2 = max(0, x1 - 2), max(0, y1 - 2), min(w, x2 + 2), min(h, y1 + int(0.24 * bh) + 2)
        if hx2 - hx1 < 2 or hy2 - hy1 < 2:
            continue
        roi = frame[hy1:hy2, hx1:hx2]
        small = cv2.resize(roi, (max(1, (hx2 - hx1) // 10), max(1, (hy2 - hy1) // 10)), interpolation=cv2.INTER_AREA)
        frame[hy1:hy2, hx1:hx2] = cv2.resize(small, (hx2 - hx1, hy2 - hy1), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--feed", required=True)
    ap.add_argument("--config", default=str(ROOT / "config" / "scenes.json"))
    ap.add_argument("--manifest", default=str(ROOT / "eval" / "reel.json"))
    ap.add_argument("--stats", help="JSON: {title, rows: [[label, value]], notes: [..]} for the results slide")
    ap.add_argument("--slide", action="append", default=[], help="screenshot to show: path or 'path|caption'")
    ap.add_argument("--pipeline", default="PipeGen  |  DeepStream 7.1  |  TensorRT FP16  |  yolo11s + tracker")
    ap.add_argument("--segments", help="comma list of start-end reel frames (default: busiest stretch of each scene)")
    ap.add_argument("--draw-boxes", action="store_true", help="draw boxes from the feed (raw-reel dry runs)")
    ap.add_argument("--blur-heads", action="store_true", help="pixelate heads from the feed (raw-reel dry runs)")
    ap.add_argument("--watermark")
    ap.add_argument("--out", default=str(ROOT / "data" / "crosscount_demo.mp4"))
    a = ap.parse_args()

    segments = DEFAULT_SEGMENTS if not a.segments else [tuple(int(v) for v in s.split("-")) for s in a.segments.split(",")]
    wanted = set()
    for s, e in segments:
        wanted.update(range(s, e))
    scenes, size = load_scenes(a.config, a.manifest)
    counter = CrossCounter(scenes, size)
    feed = {}
    with open(a.feed) as f:
        for line in f:
            if line.strip():
                fr = parse(line)
                feed[fr.index] = fr
    cap = cv2.VideoCapture(a.video)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or VW
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    def hold(img, seconds):
        for _ in range(int(seconds * FPS)):
            out.write(img)

    hold(slide([("CrossCount", "h1", TEXT),
                ("Traffic and crosswalk analytics on CraftifAI PipeGen", "h2", ACCENT),
                ("Per-class, per-direction line-crossing counts for pedestrians, cyclists, two-wheelers, cars, buses and trucks", "b", MUTED),
                ("Embedded SF Physical AI Hackathon  |  Track A (PipeGen)  |  Problem 4: Traffic and Crosswalk Analytics", "s", MUTED)]), 4)
    hold(how_it_works(), 6)

    flashes = deque()          # (until frame, point, category, direction)
    ticker = deque(maxlen=6)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr = feed.get(index)
        if fr is not None:
            for c in counter.update(fr):
                flashes.append((index + 20, c.point, c.category, c.direction))
                ticker.appendleft(c)
        if index in wanted and counter.scene is not None:
            frame = cv2.resize(frame, (VW, VH)) if frame.shape[1] != VW else frame
            objs = fr.objects if fr is not None else []
            if a.blur_heads:
                pixelate_heads(frame, objs, 1.0)     # feed boxes are already in counter space (1280x720)
            if a.draw_boxes:
                for o in objs:
                    x1, y1, x2, y2 = (int(v) for v in o.box)
                    col = COLOURS.get(CLASS_OF.get(o.cls, ""), (200, 200, 200))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col[::-1], 1)
            for line in counter.scene.lines:
                p1 = (int(line.a[0] * VW), int(line.a[1] * VH))
                p2 = (int(line.b[0] * VW), int(line.b[1] * VH))
                cv2.line(frame, p1, p2, ACCENT[::-1], 3)
                cv2.circle(frame, p1, 6, ACCENT[::-1], -1)
                cv2.circle(frame, p2, 6, ACCENT[::-1], -1)
            while flashes and flashes[0][0] < index:
                flashes.popleft()
            for until, (x, y), cat, direction in flashes:
                r = 10 + (until - index)
                cv2.circle(frame, (int(x), int(y)), r, COLOURS[cat][::-1], 3)
            canvas = np.full((H, W, 3), BG[::-1], np.uint8)
            canvas[VY:VY + VH, VX:VX + VW] = frame
            pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            name = counter.scene.name
            d.text((VX, 34), f"{SCENE_TITLE.get(name, name)}", font=F["h2"], fill=TEXT)
            d.text((VX + 700, 44), a.pipeline, font=F["s"], fill=MUTED)
            # Right panel: totals for this scene, by class and direction.
            totals = counter.summary().get(name, {})
            d.text((PX, VY - 4), "Crossings this scene", font=F["bb"], fill=MUTED)
            y = VY + 44
            for cat in CATEGORIES:
                n = sum(v for line in totals.values() for v in line.get(cat, {}).values())
                dirs = {}
                for line in totals.values():
                    for k, v in line.get(cat, {}).items():
                        dirs[k] = dirs.get(k, 0) + v
                d.rounded_rectangle((PX, y, W - 40, y + 98), radius=12, fill=PANEL, outline=LINE)
                d.rectangle((PX, y, PX + 8, y + 98), fill=COLOURS[cat])
                d.text((PX + 26, y + 12), str(n), font=F["n"], fill=TEXT)
                d.text((PX + 150, y + 16), LABEL[cat], font=F["bb"], fill=TEXT)
                d.text((PX + 150, y + 54), "   ".join(f"{k} {v}" for k, v in sorted(dirs.items())) or "-", font=F["s"], fill=MUTED)
                y += 108
            # Bottom: the latest crossings and the clock.
            y0 = VY + VH + 24
            d.text((VX, y0), "Latest crossings", font=F["bb"], fill=MUTED)
            x = VX
            for c in list(ticker)[:5]:
                tag = f"{LABEL[c.category][:-1] if c.category != 'bus' else 'Bus'}  {c.direction}"
                tw = d.textlength(tag, font=F["s"]) + 30
                d.rounded_rectangle((x, y0 + 44, x + tw, y0 + 84), radius=8, fill=COLOURS[c.category])
                d.text((x + 15, y0 + 51), tag, font=F["s"], fill=BG)
                x += tw + 12
            d.text((VX, y0 + 110), f"reel {index / FPS:6.1f} s   |   MQTT feed: boxes, classes, track ids only; heads pixelated inside the pipeline",
                   font=F["xs"], fill=MUTED)
            if a.watermark:
                d.text((VX + 20, VY + 20), a.watermark, font=F["h2"], fill=(248, 113, 113))
            out.write(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        index += 1
    counter.flush()

    if a.stats and Path(a.stats).exists():
        hold(results_slide(json.loads(Path(a.stats).read_text())), 8)
    for s in a.slide:
        path, _, caption = s.partition("|")
        hold(image_slide(path, caption or Path(path).stem.replace("_", " ")), 5)
    hold(slide([("CrossCount", "h1", TEXT), ("PipeGen sees. CrossCount counts. Nothing but metadata leaves the pipeline.", "b", MUTED)]), 3)
    out.release()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
