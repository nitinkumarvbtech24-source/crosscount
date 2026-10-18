"""Live CrossCount service: PipeGen's MQTT feed in, counts out.

  subscribes  crosscount/detections   PipeGen's per-frame tracked objects
  publishes   crosscount/events       one message per line crossing
              crosscount/counts       running totals (retained), once a second
  serves      http://localhost:8097   live dashboard (metadata only: no video)

    python -m crosscount.service --config config/scenes.json --manifest <reel.json>
    python -m crosscount.service --config config/scenes.json --scene in_citycentre   # one live camera

Only PipeGen touches pixels. What reaches this process is boxes, classes and
track ids, so nothing here can leak a face or a number plate.
"""
import argparse
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import paho.mqtt.client as mqtt

from .counter import CATEGORIES, CrossCounter, crossing_dict, load_scenes
from .schema import parse

ROOT = Path(__file__).resolve().parent.parent
BIN_S = 10.0   # chart bin width, seconds of video time


class Live:
    """Counter plus everything the dashboard shows, behind one lock."""

    def __init__(self, counter: CrossCounter, record_path=None, events_path=None):
        self.c = counter
        self.lock = threading.Lock()
        self.frame = None
        self.arrivals = deque(maxlen=90)          # wall-clock arrival times -> pipeline fps
        self.proc_ms = deque(maxlen=300)          # time spent counting each frame
        self.recent = deque(maxlen=25)
        self.bins = {}                            # (scene, bin index) -> {category: n}
        self.messages = 0
        self.connected = False
        self.record = open(record_path, "a", buffering=1) if record_path else None
        self.events_log = open(events_path, "a", buffering=1) if events_path else None

    def on_payload(self, payload: bytes):
        now = time.time()
        frame = parse(payload)
        t0 = time.perf_counter()
        with self.lock:
            events = self.c.update(frame)
            self.proc_ms.append((time.perf_counter() - t0) * 1000)
            self.arrivals.append(now)
            self.frame = frame
            self.messages += 1
            for e in events:
                d = crossing_dict(e)
                d["wall_time"] = now
                self.recent.appendleft(d)
                b = self.bins.setdefault((e.scene, int(e.ts // BIN_S)), {})
                b[e.category] = b.get(e.category, 0) + 1
                if self.events_log:
                    self.events_log.write(json.dumps(d) + "\n")
        if self.record:
            self.record.write(payload.decode() if isinstance(payload, bytes) else payload)
            self.record.write("\n")
        return events

    def snapshot(self) -> dict:
        with self.lock:
            f, scene = self.frame, self.c.scene
            objects = []
            if f is not None and scene is not None:
                for o in f.objects:
                    t = self.c.tracks.get(o.tid)
                    # Not counted: a person riding something (their vehicle carries the count).
                    cat = (t.category() if t else None) or "rider"
                    objects.append({"id": o.tid, "category": cat, "box": [round(v, 1) for v in o.box]})
            fps = 0.0
            if len(self.arrivals) > 1:
                span = self.arrivals[-1] - self.arrivals[0]
                fps = (len(self.arrivals) - 1) / span if span > 0 else 0.0
            ms = sorted(self.proc_ms)
            series = [{"scene": s, "t0": i * BIN_S, "counts": c} for (s, i), c in sorted(self.bins.items())
                      if scene is None or s == scene.name]
            return {
                "connected": self.connected,
                "messages": self.messages,
                "scene": scene.name if scene else None,
                "lines": [{"name": l.name, "a": l.a, "b": l.b, "pos": l.pos, "neg": l.neg} for l in scene.lines] if scene else [],
                "frame_index": f.index if f else None,
                "video_ts": f.ts if f else None,
                "frame_size": [self.c.w, self.c.h],
                "pipeline_fps": round(fps, 1),
                "count_ms_avg": round(sum(ms) / len(ms), 3) if ms else None,
                "count_ms_p95": round(ms[int(0.95 * (len(ms) - 1))], 3) if ms else None,
                "objects": objects,
                "totals": self.c.summary(),
                "recent": list(self.recent),
                "series": series[-30:],
                "categories": CATEGORIES,
            }


def make_handler(live: Live):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, (ROOT / "dashboard" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send(200, json.dumps(live.snapshot()).encode(), "application/json")
            if self.path.startswith("/stills/") and self.path.endswith(".jpg"):
                p = ROOT / "data" / "stills" / Path(self.path).name
                if p.exists():
                    return self._send(200, p.read_bytes(), "image/jpeg")
            self._send(404, b"not found", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker", default="localhost:1883")
    ap.add_argument("--topic", default="crosscount/detections")
    ap.add_argument("--out-prefix", default="crosscount")
    ap.add_argument("--config", default=str(ROOT / "config" / "scenes.json"))
    ap.add_argument("--manifest", help="reel manifest (scene frame ranges); omit for one live camera")
    ap.add_argument("--scene", help="with no manifest: which scene's lines to use")
    ap.add_argument("--http", type=int, default=int(os.environ.get("PORT", 8097)))
    ap.add_argument("--record", help="append every raw PipeGen message to this JSONL file (for scoring)")
    ap.add_argument("--events", default=str(ROOT / "data" / "events.jsonl"))
    a = ap.parse_args()

    scenes, size = load_scenes(a.config, a.manifest, a.scene)
    Path(a.events).parent.mkdir(parents=True, exist_ok=True)
    live = Live(CrossCounter(scenes, size), a.record, a.events)

    host, port = a.broker.rsplit(":", 1)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"crosscount-{int(time.time())}")

    def on_connect(c, userdata, flags, rc, props=None):
        live.connected = rc == 0
        c.subscribe(a.topic, qos=0)
        print(f"MQTT {a.broker}: subscribed to {a.topic}", flush=True)

    def on_disconnect(c, userdata, flags, rc, props=None):
        live.connected = False

    def on_message(c, userdata, msg):
        for e in live.on_payload(msg.payload):
            c.publish(f"{a.out_prefix}/events", json.dumps(crossing_dict(e)))

    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message
    client.connect(host, int(port))
    client.loop_start()

    def publish_counts():
        while True:
            time.sleep(1.0)
            with live.lock:
                body = json.dumps({"totals": live.c.summary(), "frame_index": live.frame.index if live.frame else None})
            client.publish(f"{a.out_prefix}/counts", body, retain=True)

    threading.Thread(target=publish_counts, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", a.http), make_handler(live))
    print(f"dashboard: http://localhost:{a.http}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
