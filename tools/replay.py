"""Publishes a recorded feed (JSONL, one PipeGen message per line) to MQTT at
the video's frame rate, as if PipeGen were running. For testing the service
and dashboard, and for re-showing a recorded PipeGen run.

    python tools/replay.py data/pipegen_fp16.jsonl [--fps 30] [--speed 1] [--loop]
"""
import argparse
import time

import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feed")
    ap.add_argument("--broker", default="localhost:1883")
    ap.add_argument("--topic", default="crosscount/detections")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--start", type=int, default=0, help="skip this many frames")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()

    host, port = a.broker.rsplit(":", 1)
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.connect(host, int(port))
    c.loop_start()
    period = 1.0 / (a.fps * a.speed)
    while True:
        with open(a.feed) as f:
            nxt = time.perf_counter()
            for i, line in enumerate(f):
                if i < a.start or not line.strip():
                    continue
                c.publish(a.topic, line.strip())
                nxt += period
                time.sleep(max(0.0, nxt - time.perf_counter()))
        if not a.loop:
            break
    time.sleep(0.5)
    c.loop_stop()


if __name__ == "__main__":
    main()
