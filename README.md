# CrossCount: traffic and crosswalk analytics on PipeGen

Embedded SF / CraftifAI Physical AI Hackathon, **suggested problem #4: Traffic
and Crosswalk Analytics**, Track A (PipeGen).

Cities lack live counts of pedestrians, cyclists and vehicles at crossings.
CrossCount gives a pole-mounted camera per-class, per-direction line-crossing
counts: a GPU vision pipeline built with **PipeGen** (DeepStream, TensorRT
FP16 and INT8) sees and tracks every road user, and CrossCount turns those
tracks into counts that a traffic team can act on, over MQTT and a live
dashboard.

```
 video / RTSP camera
        │
 ┌──────▼─────────────────────── PipeGen pipeline (DeepStream) ─────────────────────┐
 │ decode → COCO detector on TensorRT (FP16 / INT8) → multi-object tracker          │
 │        → pixelate heads + plates (before any frame is shown, saved or streamed)  │
 │        → MQTT crosscount/detections (JSON per frame)  · RTSP · saved video       │
 └──────┬───────────────────────────────────────────────────────────────────────────┘
        │ boxes, classes, track ids only: no pixels leave the pipeline
 ┌──────▼──────────────── CrossCount (crosscount/) ───────────────┐
 │ rider logic: a person on a bike is part of that bike           │
 │ counting lines per scene, direction from the side crossed onto │
 │ per-class, per-direction totals; 10 s bins                     │
 └──────┬─────────────────────────────────────────────────────────┘
        ├─ MQTT crosscount/events   one message per crossing
        ├─ MQTT crosscount/counts   running totals (retained)
        ├─ data/events.jsonl        crossing log
        └─ http://localhost:8097    live dashboard
```

**PipeGen is the perception system, not a plug-in.** Every box, class and
track id CrossCount counts comes from the PipeGen-generated DeepStream app, and
the privacy redaction happens inside it. Without PipeGen there is nothing to
count; the INT8 engine PipeGen compiles is what makes the pole-mounted
(low-power) version credible.

## Road-user classes

| Category | From the detector | Rule |
|---|---|---|
| pedestrian | `person` | a person who is neither riding nor moving faster than walking pace |
| cyclist | `bicycle` | the rider is not counted again |
| two_wheeler | `motorcycle` | one bike with two people on it is one two-wheeler |
| car, bus, truck | `car`, `bus`, `truck` | class by confidence-weighted vote over the track's life |

Why the rider rules exist: on the Indian scene the detector often boxes a rider
without the bike, or rider and bike together, and the person box crosses the
line too. Counting people naively reported 12 "pedestrians" crossing a busy
road in 28 s, and every one was a rider or a bus passenger seen through the
window. A person is a rider when a bicycle or motorcycle box overlaps them (or
sits inside a box drawn round both), or when they move faster than anyone
walks (1.6 body heights per second; walking is about 0.8). A person who has
only just appeared waits for a few speed readings (at most 1.5 s) before
their crossing is counted or dropped.

A crossing is counted when a track's ground point (bottom centre of its box)
is seen clearly on one side of the line after being clearly on the other, and
the step between the two passes through the drawn segment. A dead band of
0.6% of the frame height stops box jitter on the line from counting.

## Footage

Five fixed-camera street scenes from Pexels (free licence), joined into one
1280x720, 30 fps reel so a single PipeGen run covers them all, with a 1 s black
gap between scenes so tracks end at each cut. `eval/reel.json` lists each
scene's frame range; `config/scenes.json` holds each scene's counting line.

| Scene | Source | Length | What crosses |
|---|---|---|---|
| in_citycentre | [pexels 28925595](https://www.pexels.com/video/28925595/) | 28 s | two-wheelers, cars, autos, buses, trucks (Indian city road, elevated view) |
| nl_amsterdam | [pexels 12545270](https://www.pexels.com/video/12545270/) | 84 s | cyclists, cars, pedestrians, trams |
| wroclaw_junction | [pexels 33865601](https://www.pexels.com/video/33865601/) | 114 s | cars, pedestrians, cyclists (Pexels tags it Bratislava; the street sign, Grabiszyńska, is Wrocław) |
| tram_junction | [pexels 40260363](https://www.pexels.com/video/40260363/) | 30 s | cars, pedestrians, cyclists, trams |
| wroclaw_zebra | [pexels 40260120](https://www.pexels.com/video/40260120/) | 22 s | pedestrians on a zebra crossing, cars (street sign: Podwale, Wrocław) |

A sixth clip (pexels 30850558) was downloaded and dropped: the camera pans
110 px, which no fixed counting line survives (`tools/check_clips.py`).

## Run it

PipeGen side (WSL Ubuntu-22.04, NVIDIA GPU):

```bash
bash tools/fetch_clips.sh ~/videos/crosscount/raw         # download the five clips (+1 dropped)
bash tools/make_reel.sh ~/videos/crosscount/raw ~/videos/crosscount
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 pipegen      # paste docs/pipegen_requirement.md, run all five stages
```

CrossCount side (Windows or Linux, Python 3.10+, `uv`):

```bash
uv sync
uv run python -m crosscount.service --manifest eval/reel.json --record data/pipegen_fp16.jsonl
```

Open http://localhost:8097. For a single live camera instead of the reel, drop
`--manifest` and pass `--scene <name>` to pick that camera's counting line.

Without PipeGen running (UI work, demos): `uv run python tools/replay.py <recorded feed.jsonl>`
publishes a recorded feed on the same topic at 30 fps.

Tests: `uv run pytest` (counter behaviour on synthetic tracks, message parsing).

## Demo video

`tools/make_demo.py` renders the demo: title, how it works, each scene with
its counting line and live per-class counts, measured numbers, and the
PipeGen screenshots in `docs/evidence/`.

The cut in `data/crosscount_demo_h264.mp4` was made before the PipeGen run
finished, so its counting clips use the offline reference tracks and say so
on screen. To rebuild it from the PipeGen run (after **Run** in PipeGen, with
`crosscount.service --record data/pipegen_fp16.jsonl` running):

```bash
python tools/make_demo.py --video <PipeGen saved video: output/output.mp4> --feed data/pipegen_fp16.jsonl --stats eval/results/demo_stats.json --slide "docs/evidence/pipeline_ir_graph.png|PipeGen Pipeline IR" --slide "docs/evidence/yolo11s_fp16_engine.png|PipeGen TensorRT FP16 engine" --out data/crosscount_demo.mp4
```

PipeGen's saved video already has its boxes and pixelated heads, so leave
out `--draw-boxes`, `--blur-heads` and `--watermark`.

## Measuring it

`eval/README.md` explains how the ground truth was built and how a recorded
PipeGen run is scored (`uv run python -m eval.score data/pipegen_fp16.jsonl`).

## Privacy

Heads and number plates are pixelated inside the PipeGen pipeline before a
frame is displayed, saved or streamed. CrossCount receives only boxes, classes
and track ids, so it cannot leak a face or a plate; the dashboard's scene
background is a heavily blurred still.

## Limitations

- COCO has no auto-rickshaw class: autos are detected as car, truck or not at
  all. The ground truth counts them as cars.
- A rider whose bike the detector never finds, moving at walking pace in a jam,
  can still be counted as a pedestrian.
- One counting line per scene in this reel; the code takes any number.
