"""Counter behaviour on synthetic tracks (fake data, 1000x1000 frame)."""
from crosscount.counter import CrossCounter, Line, Scene
from crosscount.schema import Detection, Frame, parse

W = H = 1000
# A vertical line at x=500 from y=200 to y=800, drawn top to bottom: moving
# right lands on its negative side, so neg is labelled "east".
LINE = Line("road", (0.5, 0.2), (0.5, 0.8), pos="west", neg="east")


def counter(scenes=None):
    return CrossCounter(scenes or [Scene("s", [LINE])], (W, H))


def box(cx, bottom, w=40, h=100):
    return (cx - w / 2, bottom - h, cx + w / 2, bottom)


def run(c, tracks, n=40, start=0):
    """tracks: {tid: (cls, fn(i) -> box)}; feeds n frames at 30 fps."""
    out = []
    for i in range(start, start + n):
        objs = [Detection(tid, cls, 0.9, f(i)) for tid, (cls, f) in tracks.items() if f(i) is not None]
        out += c.update(Frame(i, i / 30, objs))
    return out


def test_car_crossing_counts_once_with_direction():
    c = counter()
    ev = run(c, {1: ("car", lambda i: box(300 + 10 * i, 500))})
    assert [(e.category, e.direction) for e in ev] == [("car", "east")]


def test_direction_labels_are_opposite():
    c1, c2 = counter(), counter()
    a = run(c1, {1: ("car", lambda i: box(300 + 10 * i, 500))})
    b = run(c2, {1: ("car", lambda i: box(700 - 10 * i, 500))})
    assert a[0].direction != b[0].direction


def test_jitter_on_the_line_is_not_counted():
    c = counter()
    ev = run(c, {1: ("person", lambda i: box(500 + (3 if i % 2 else -3), 500, h=90))}, n=60)
    assert ev == []


def test_crossing_past_the_end_of_the_line_is_not_counted():
    c = counter()
    ev = run(c, {1: ("car", lambda i: box(300 + 10 * i, 950))})   # below y=800
    assert ev == []


def test_cyclist_counts_as_one_cyclist_not_a_pedestrian():
    c = counter()
    ev = run(c, {
        1: ("bicycle", lambda i: box(300 + 10 * i, 500, w=60, h=50)),
        2: ("person", lambda i: box(300 + 10 * i, 480, w=40, h=90)),   # sits on the bike
    })
    assert [e.category for e in ev] == ["cyclist"]


def test_two_people_on_a_motorcycle_count_as_one_two_wheeler():
    c = counter()
    ev = run(c, {
        1: ("motorcycle", lambda i: box(300 + 10 * i, 500, w=80, h=60)),
        2: ("person", lambda i: box(290 + 10 * i, 480, w=40, h=90)),
        3: ("person", lambda i: box(315 + 10 * i, 480, w=40, h=90)),
    })
    assert [e.category for e in ev] == ["two_wheeler"]


def walker(start, step=3, bottom=500):
    """A person walking at ~1 body height per second (3 px/frame, 90 px tall, 30 fps)."""
    return lambda i: box(start + step * i, bottom, w=40, h=90)


def test_pedestrian_next_to_parked_bikes_is_still_a_pedestrian():
    c = counter()
    parked = box(520, 500, w=60, h=50)
    ev = run(c, {1: ("bicycle", lambda i: parked), 2: ("person", walker(440))}, n=60)
    assert [e.category for e in ev] == ["pedestrian"]


def test_person_faster_than_walking_is_not_a_pedestrian():
    # A rider whose bike the detector missed, or a passenger seen through a bus window.
    c = counter()
    ev = run(c, {1: ("person", walker(300, step=10))})
    assert ev == []
    assert [d["why"] for d in c.dropped] == ["faster than walking"]


def test_one_box_round_rider_and_bike_marks_the_rider():
    c = counter()
    ev = run(c, {
        1: ("motorcycle", lambda i: box(300 + 10 * i, 495, w=60, h=45)),
        2: ("person", lambda i: box(300 + 10 * i, 500, w=90, h=110)),   # drawn round both
    })
    assert [e.category for e in ev] == ["two_wheeler"]


def test_one_van_boxed_as_car_and_truck_counts_once():
    c = counter()
    ev = run(c, {
        1: ("car", lambda i: box(300 + 10 * i, 500, w=120, h=80)),
        2: ("truck", lambda i: box(302 + 10 * i, 498, w=118, h=82) if i >= 5 else None),  # second box, appears later
    })
    assert [e.category for e in ev] == ["car"]
    assert [d["why"] for d in c.dropped] == ["duplicate box"]


def test_two_vehicles_side_by_side_both_count():
    c = counter()
    ev = run(c, {
        1: ("car", lambda i: box(300 + 10 * i, 500, w=120, h=80)),
        2: ("car", lambda i: box(300 + 10 * i, 560, w=120, h=80)),    # next lane, boxes overlap a little
    })
    assert len(ev) == 2


def test_class_vote_wins_over_flicker():
    c = counter()
    ev = run(c, {1: ("car", lambda i: box(300 + 10 * i, 500))}, n=15)
    ev += run(c, {1: ("truck", lambda i: box(300 + 10 * i, 500))}, n=5, start=15)
    ev += run(c, {1: ("car", lambda i: box(300 + 10 * i, 500))}, n=20, start=20)
    assert [e.category for e in ev] == ["car"]


def test_scene_change_resets_tracks():
    s1 = Scene("one", [LINE], 0, 10)
    s2 = Scene("two", [LINE], 10, 40)
    c = counter([s1, s2])
    # Same tracker id: left of the line in scene one, right of it in scene two.
    ev = run(c, {1: ("car", lambda i: box(400 if i < 10 else 600, 500))})
    assert ev == []


def test_totals_and_summary():
    c = counter()
    run(c, {1: ("car", lambda i: box(300 + 10 * i, 500)),
            2: ("person", walker(560, step=-3, bottom=600))}, n=60)
    s = c.summary()["s"]["road"]
    assert sum(s["car"].values()) == 1 and sum(s["pedestrian"].values()) == 1


def test_parse_pipegen_message():
    f = parse('{"frame_index": 7, "timestamp_s": 0.23, "objects": [{"tracker_id": 4, "class": "Motorbike",'
              ' "confidence": 0.8, "bounding_box": [1, 2, 11, 22]}, {"tracker_id": -1, "class": "car",'
              ' "confidence": 0.5, "bounding_box": [0, 0, 1, 1]}]}')
    assert f.index == 7 and len(f.objects) == 1
    assert f.objects[0].cls == "motorcycle" and f.objects[0].box == (1, 2, 11, 22)


def test_parse_pipegen_deepstream_frame():
    # The exact shape of PipeGen 0.5.1's frame_json.h record.
    f = parse('{"frame": 12, "source_id": 0, "pts_ns": 400000000, "width": 1920, "height": 1080,'
              ' "objects": [{"id": 3, "class_id": 2, "label": "car", "confidence": 0.9100,'
              ' "bbox": {"left": 10.0, "top": 20.0, "width": 50.0, "height": 30.0}, "tracker_confidence": 0.8},'
              ' {"id": null, "class_id": 0, "label": "person", "confidence": 0.5,'
              ' "bbox": {"left": 1, "top": 1, "width": 1, "height": 1}}]}')
    assert (f.index, f.ts, f.size) == (12, 0.4, (1920, 1080))
    assert len(f.objects) == 1 and f.objects[0].tid == 3 and f.objects[0].cls == "car"
    assert f.objects[0].box == (10.0, 20.0, 60.0, 50.0)


def test_boxes_in_muxer_resolution_are_scaled_to_the_lines():
    # Same car as test_car_crossing..., but reported in a 2000x2000 muxer space.
    c = counter()
    ev = []
    for i in range(40):
        b = box(300 + 10 * i, 500)
        f = Frame(i, i / 30, [Detection(1, "car", 0.9, tuple(v * 2 for v in b))], size=(2000, 2000))
        ev += c.update(f)
    assert [(e.category, e.direction) for e in ev] == [("car", "east")]


def test_parse_rect_params_variant():
    f = parse({"frame_number": 3, "objects": [{"object_id": 9, "label": "person",
                                               "rect_params": {"left": 10, "top": 20, "width": 5, "height": 8}}]})
    assert f.objects[0].box == (10, 20, 15, 28) and f.objects[0].tid == 9
