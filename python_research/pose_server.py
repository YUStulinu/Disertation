"""
Module 1 + 2: pose and hand co-process for the C++ Qt application.

The Qt app starts this script with QProcess and talks to it over stdin/stdout:
  C++ writes, one per line:  "<capture time in s> <scale> <base64 JPEG frame>", or a command -
                             "#next" (every hand gets the next object),
                             "#objects 0" / "#objects 1" (hands and objects off / on)
  C++ reads, one line per frame, a JSON object:
    pose       33 [x, y] keypoints (normalized 0-1) of the nearest person, [] if nobody is in
               view, or null if no new pose is ready - for the action classifier (Module 1)
    pose_time  the capture time of the frame that pose comes from
    objects    the museum objects to draw in the hands held out (HandheldScene.describe in
               common/handheld.py), for cpp_app/object_renderer.cpp
    hands      every hand found: 21 [x, y] points (pixels), its centre and how it is held - for the debug view
The frame is a copy of the camera image made small for the detection (540 rows: the hand and pose
models work on smaller images anyway); <scale> is how much larger the camera image the app draws
into is (Full HD: 2), and every place, size and point in the reply is given at that scale.
    harmony    true once the harmonization networks are loaded

Each frame only waits for the near-hand tracking; the pose of everyone in view (several people,
~35 ms), their ids and the far-hand search run in a background thread on the newest frame
(BodyThread, at the classifier's SAMPLE_RATE, as in ar_objects.py), and the harmonization networks
in their own process. The app keeps two frames here at once, and the next one is read and decoded
in a thread while the hands of this one are found (incoming): no frame waits for another step.
--no-objects keeps only the pose, computed on every frame (Module 1 alone).
Not meant to be run manually.
"""

import argparse
import base64
import json
import queue
import sys
import threading
import time
import traceback

import cv2
import numpy as np

from common.pose import SAMPLE_RATE, PoseTracker


def keypoints(poses: list[np.ndarray]) -> list:
    """x, y of the nearest person's 33 landmarks - what the ONNX model expects - or []."""
    return poses[0][:, :2].astype(float).round(5).tolist() if poses else []


def incoming(stream):
    """The app's lines in order, read - and their frames decoded - ahead in a thread: (command words,
    None) for a command, (None, (capture time, scale, frame or None)) for a frame."""
    items: queue.Queue = queue.Queue(maxsize=4)

    def read() -> None:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):  # a command from the app, no reply
                items.put((line[1:].split(), None))
                continue
            try:
                fields = line.split(" ")
                t = float(fields[0]) if len(fields) > 1 else time.perf_counter()
                scale, data = float(fields[1]) if len(fields) > 2 else 1.0, fields[-1]
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                t, scale, frame = time.perf_counter(), 1.0, None
            items.put((None, (t, scale, frame)))
        items.put(None)  # the end of input: the app is closing

    threading.Thread(target=read, daemon=True).start()
    while (item := items.get()) is not None:
        yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose and hand co-process for the C++ app")
    parser.add_argument("--no-objects", action="store_true", help="Pose only: no hands, objects or harmonization")
    parser.add_argument("--people", type=int, default=2, help="People tracked (each gets their own objects)")
    args = parser.parse_args()

    with_objects = not args.no_objects
    pose = PoseTracker(num_poses=args.people if with_objects else 1)
    if with_objects:
        from common.handheld import HandheldScene, load_catalog
        from common.hands import BodyThread, HandTracker, merge_hands

        hands = HandTracker(max_hands=2)
        body = BodyThread(pose, hands, SAMPLE_RATE)
        scene = HandheldScene(load_catalog(), "ensemble")  # starts loading the harmonization networks
        body.start()
    enabled, last_pose = with_objects, None
    print("pose_server ready", file=sys.stderr, flush=True)
    try:
        for command, job in incoming(sys.stdin):
            if command is not None:
                if with_objects and command == ["next"]:
                    scene.next_object()
                elif with_objects and command[:1] == ["objects"]:
                    enabled = command[1:] == ["1"]
                    if not enabled:
                        scene.tracks, scene.holders = [], {}
                continue
            reply = {"pose": None, "pose_time": None, "objects": [], "hands": [], "harmony": False}
            t, scale, frame = job
            try:
                if frame is not None and not with_objects:
                    reply["pose"], reply["pose_time"] = keypoints(pose.detect_all(frame)), t
                elif frame is not None:
                    near = hands.detect_near(frame) if enabled else []
                    body.submit(frame, t, near, far=enabled)
                    pose_time, poses, people, far = body.result
                    if pose_time is not None and pose_time != last_pose:  # a new pose: for the classifier
                        reply["pose"], reply["pose_time"], last_pose = keypoints(poses), pose_time, pose_time
                    if enabled:
                        found = merge_hands(near, far)
                        scene.update(found, people)
                        reply["objects"] = scene.describe(frame, scale)
                        reply["hands"] = [{"points": np.round(h.points * scale, 1).tolist(),
                                           "centre": np.round(h.centre * scale, 1).tolist(), "hold": h.hold} for h in found]
                        reply["harmony"] = scene.worker.ready
            except Exception:
                traceback.print_exc(file=sys.stderr)
            sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
            sys.stdout.flush()
    finally:
        if with_objects:
            body.stop()
            scene.worker.close()
            hands.close()
        pose.close()


if __name__ == "__main__":
    main()
