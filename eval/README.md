# Measuring CrossCount

## What is scored

A *crossing* is one road user passing one counting line in one direction.
`eval/score.py` runs a recorded feed (PipeGen's MQTT messages saved as JSONL by
`crosscount.service --record`) through the same counter the live service uses
and matches its crossings to `ground_truth.json`: same scene and line, same
class, same direction, at most 1.5 s apart, one-to-one, closest first.

It reports per class:

- **recall**: true crossings that were counted with the right class and direction
- **precision**: counted crossings that are real, with the right class and direction
- **count error**: |counted − true| / true, what a city dashboard would be off by

and the same matching ignoring class ("any class"), which tells a road user
the detector or tracker missed apart from one counted under the wrong class.

## How the ground truth was built

No public dataset has labelled line crossings for mixed pedestrians, cyclists
and vehicles from a fixed pole-height camera, so the ground truth was built
for this reel:

1. **Reference tracks.** `tools/reference_tracks.py` runs a much larger model
   than the one deployed (YOLOv8x at 1280 px, PyTorch, ByteTrack) over the
   reel, about 6 fps. Its tracks are counted with the same lines and rules.
2. **Every counted crossing checked by eye.** `tools/contact_sheet.py` crops
   each crossing at the moment it happened, with the box and the line drawn.
   Each tile was marked right, wrong class, wrong direction, or not a crossing
   (a duplicate box, a driver's head boxed as a bicycle).
3. **Misses searched for, detector-free.** `tools/slitscan.py` stacks the
   pixels under each counting line over time, so everything that crossed shows
   as a streak whatever the detector saw; counted crossings are marked on it, so
   an unmarked streak is a miss. Doubtful streaks were checked on full frames.
4. **People the counter dropped were checked too.** The counter logs every
   person crossing it did not count and why (riding, faster than walking, seen
   too briefly); the contact sheet of those confirmed they were riders.

`eval/review.json` records every decision with the reason, and
`tools/build_gt.py` turns the reference crossings plus the review into
`ground_truth.json`. The ground truth is frozen once built; changing the
counter's rules afterwards does not regenerate it.

Class policy: car = cars, SUVs, small vans and auto-rickshaws (COCO has no
auto class); truck = pickups, lorries and large panel vans; trams are not
counted (no category); a person pushing or riding a bike is the bike.

**What is and is not scored.** Two counting lines are shown in the demo but
not scored: the crosswalk in `tram_junction` (a crowd crossing at once in
front of a queue of cars) and the far pavement in `wroclaw_zebra` (a dense
two-way stream of people). Neither could be labelled crossing by crossing by
eye in the time available, and a partial ground truth would mark real people
that PipeGen counts as false. The other five lines are scored. On the Indian
scene the slit-scan is too crowded to search exhaustively for misses, so its
ground truth is every reference crossing that survived review; a road user
both the reference model and the reviewer missed is not in it.

**Who checked it:** the review was done by eye on the contact sheets and
slit-scans by the AI assistant that built CrossCount, not by an independent
annotator, and the reference model's detections were the starting point. In
dense traffic (the Indian scene) a road user hidden behind a bus for the whole
crossing can be missed by both the model and the slit-scan. Treat the numbers
as careful, not as an audited benchmark.

## Scoring a run

```bash
uv run python -m eval.score data/pipegen_fp16.jsonl --name "PipeGen FP16" --out eval/results/fp16.md
uv run python -m eval.score data/pipegen_int8.jsonl --name "PipeGen INT8" --out eval/results/int8.md
```
