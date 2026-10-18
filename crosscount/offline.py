"""Run the counter over a recorded feed (JSONL, one PipeGen message per line)
with no MQTT in between: the same code path the live service uses, so a
recorded FP16 run and a recorded INT8 run can be scored side by side.

    python -m crosscount.offline <feed.jsonl> --manifest <reel.json> [--json out.json] [--events out.jsonl]
"""
import argparse
import json
import time
from pathlib import Path

from .counter import CATEGORIES, CrossCounter, crossing_dict, load_scenes
from .schema import parse

ROOT = Path(__file__).resolve().parent.parent


def count_feed(feed, config, manifest=None, scene=None):
    scenes, size = load_scenes(config, manifest, scene)
    c = CrossCounter(scenes, size)
    frames, t0 = 0, time.perf_counter()
    with open(feed) as f:
        for line in f:
            if line.strip():
                c.update(parse(line))
                frames += 1
    c.flush()
    return c, frames, time.perf_counter() - t0


def table(summary: dict) -> str:
    rows = [f"{'scene':16s} {'line':12s} " + " ".join(f"{k[:11]:>11s}" for k in CATEGORIES)]
    for scene, lines in summary.items():
        for line, cats in lines.items():
            cells = []
            for k in CATEGORIES:
                d = cats.get(k, {})
                cells.append(f"{'/'.join(str(v) for v in d.values()) if d else '-':>11s}")
            rows.append(f"{scene:16s} {line:12s} " + " ".join(cells))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feed")
    ap.add_argument("--config", default=str(ROOT / "config" / "scenes.json"))
    ap.add_argument("--manifest")
    ap.add_argument("--scene")
    ap.add_argument("--json", help="write the totals here")
    ap.add_argument("--events", help="write every crossing here (JSONL)")
    a = ap.parse_args()
    c, frames, secs = count_feed(a.feed, a.config, a.manifest, a.scene)
    print(table(c.summary()))
    print(f"\n{frames} frames, {len(c.events)} crossings, counting took {1000 * secs / max(frames, 1):.3f} ms/frame")
    if a.json:
        Path(a.json).write_text(json.dumps(c.summary(), indent=1))
    if a.events:
        with open(a.events, "w") as f:
            for e in c.events:
                f.write(json.dumps(crossing_dict(e)) + "\n")


if __name__ == "__main__":
    main()
