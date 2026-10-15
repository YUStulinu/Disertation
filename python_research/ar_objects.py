"""
Module 1 + 2 together: museum objects appear in the hands held out to the camera (live).

Hold one hand out, palm open: a small object stands on it (N and H change it). Hold out both
hands close together: one larger object appears between them - held at its belly by open hands,
or by its handles when the fingers are bent as in a grip; hold them far apart: a small object
in each. With both hands up they are busy, so the objects change by themselves every 4 s
(--cycle), each shown harmonized for 2 s and then raw for 2 s (--alternate). Every person
gets their own objects; hands of different people never share one. Objects are drawn at their
real size, harmonized with the webcam image and given its sharpness and noise
(common/handheld.py); the Module 1 action classifier runs alongside and shows what the person
is doing. The webcam is read in Full HD (--resolution): the hands and poses are found in a
540-row copy (half of it) - the models work on smaller images anyway - and the objects are drawn
into the Full HD frame, at the detail of the museum photos.

Keys: H = harmonization on/off (the objects stay, shown raw, to compare), O = objects on/off,
      N = next object, D = hand landmarks on/off, S = save the frame shown (images/python_app/
      objects_<n>.png), Q or Esc = quit (closing the window quits too).

Usage:
    python ar_objects.py                                   # webcam
    python ar_objects.py --out demo.mp4                    # webcam, also recorded (H.264, real speed)
    python ar_objects.py --video clip.mp4 --out result.mp4 # a recorded video, frame by frame
    python ar_objects.py --image photo.jpg [more.jpg ...]  # stills -> data/ar_results/
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from common.handheld import COLOUR_SHARE, HandheldScene, load_catalog
from common.hands import BodyThread, HandTracker, PeopleTracker, draw_hands, merge_hands
from common.imaging import (NOTE_S, RateMeter, labeled_grid, read_image, save_shot, show_live, video_writer, with_note,
                            write_image)
from common.paths import DATA_DIR, SCREENSHOTS_DIR
from common.pose import ONNX_PATH, SAMPLE_RATE, LiveActionClassifier, PoseTracker, open_webcam

OUTPUT_DIR = DATA_DIR / "ar_results"
WINDOW = "Museum Objects in Your Hands (Q = Quit)"
WINDOW_SCALE = 1.3  # the window is a little larger than the camera image, if the screen allows
CAMERA_SIZE = "1920x1080"  # the webcam's resolution: Full HD (it gives the closest one it has)
DETECT_ROWS = 540   # the hands and poses are found in a copy of the frame this tall (Full HD / 2: a fast resize)
RECORD_FPS = 30.0   # --out, live: the recording's frame rate (frames are written on the clock)
_BAR = {}           # the status bar last drawn: anti-aliased text in Full HD takes ~5 ms, so it is kept


def draw_status(frame: np.ndarray, scene: HandheldScene, action, fps: float, harmony: bool, objects: bool) -> None:
    h, w = frame.shape[:2]
    u = h / 480  # sizes are for a 480-row frame
    grey, green, red = (210, 210, 210), (90, 220, 90), (80, 80, 255)
    lines = [(f"FPS {fps:.0f} | H Harmonization | O Objects | N Next Object | S Save | Q Quit", grey)]
    shown = [t for t in scene.tracks if t.opacity > 0 and t.artefact]
    if not objects:
        lines.append(("Objects Hidden (O Shows Them)", red))
    elif harmony and all(scene.harmonized(t) for t in shown):  # both hands switch it on and off by themselves
        lines.append(("Harmonization ON: Light, Sharpness and Noise Matched to the Camera", green))
    else:
        lines.append(("Harmonization OFF: Objects Shown Raw, as Photographed in the Museum", red))
    if objects:
        lines += [(f"{len(t.hands)} Hand{'s' * (len(t.hands) > 1)}, {'Grip' if t.grip else 'Open'}: "
                   f"{t.artefact.name.replace('_', ' ').title()} ({100 * t.artefact.size_m:.0f} cm)"
                   + (f" - {'Harmonized' if harmony and scene.harmonized(t) else 'Raw'}" if t.busy else ""), grey)
                  for t in shown]
    if _BAR.get("key") != (lines, w, h):  # drawn again only when a line changes (mostly the FPS figure)
        step = round(20 * u)
        bar = np.full((round(22 * u) + step * len(lines) + 1, w, 3), 20, np.uint8)  # the rows cv2.rectangle filled
        for i, (line, colour) in enumerate(lines):
            cv2.putText(bar, line, (round(8 * u), round(18 * u) + step * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * u, colour,
                        max(1, round(u)), cv2.LINE_AA)
        _BAR.update(key=(lines, w, h), image=bar)
    frame[:_BAR["image"].shape[0]] = _BAR["image"]
    if action:
        label, conf, _ = action
        cv2.putText(frame, f"{label} {100 * conf:.0f}%", (round(8 * u), h - round(14 * u)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 * u, (0, 220, 0) if conf > 0.7 else (0, 180, 255), max(2, round(2 * u)), cv2.LINE_AA)


def open_window(size: tuple[int, int]) -> None:
    """A resizable window, centred on the screen: WINDOW_SCALE times the camera image, or less so that
    a Full HD image fits (at most 62 % of the screen's height, 90 % of its width)."""
    try:
        from ctypes import windll
    except ImportError:  # not Windows
        windll = None
    screen = (windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)) if windll else (1920, 1080)
    scale = min(WINDOW_SCALE, 0.62 * screen[1] / size[1], 0.9 * screen[0] / size[0])
    w, h = round(size[0] * scale), round(size[1] * scale)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, w, h)
    cv2.moveWindow(WINDOW, max(0, (screen[0] - w) // 2), max(0, (screen[1] - h) // 2 - 20))  # 20 px: the title bar


def draw_pairs(frame: np.ndarray, scene: HandheldScene, scale: float = 1.0) -> None:
    """Debug view: a line between the two hands that share an object."""
    for tr in scene.tracks:
        if len(tr.hands) == 2 and tr.opacity > 0:
            a, b = ((h.centre * scale).astype(int) for h in tr.hands)
            cv2.line(frame, tuple(a), tuple(b), (0, 255, 255), max(2, round(2 * frame.shape[0] / 480)))


def detection_copy(frame: np.ndarray) -> np.ndarray:
    """The frame made DETECT_ROWS tall for the hand and pose models (they work on smaller images anyway)."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (round(w * DETECT_ROWS / h), DETECT_ROWS), interpolation=cv2.INTER_AREA) if h > DETECT_ROWS else frame


class LiveSource(threading.Thread):
    """Live, in parallel with the main loop's drawing (MediaPipe and OpenCV release the GIL): a reader
    thread takes the webcam's frames as they come, and this one mirrors the newest, makes its
    detection copy and finds its near hands. take() returns the newest (frame, copy, near hands)
    not taken yet."""

    def __init__(self, cap: cv2.VideoCapture, hands: HandTracker, body: BodyThread):
        super().__init__(daemon=True)
        self.cap, self.hands, self.body, self.stopped = cap, hands, body, False
        self._frame, self._latest, self._lock = None, None, threading.Lock()
        self._frame_ready, self._ready = threading.Event(), threading.Event()

    def _read(self) -> None:
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_ready.set()

    def run(self) -> None:
        reader = threading.Thread(target=self._read, daemon=True)
        reader.start()
        while not self.stopped:
            if not self._frame_ready.wait(0.2):
                continue
            with self._lock:
                frame, self._frame = self._frame, None
                self._frame_ready.clear()
            frame = cv2.flip(frame, 1)  # mirror view, as in the other live demos
            small = detection_copy(frame)
            near = self.hands.detect_near(small)
            self.body.submit(small, time.perf_counter(), near)
            with self._lock:
                self._latest = (frame, small, near)
                self._ready.set()
        reader.join(1.0)

    def take(self, timeout: float = 1.0) -> tuple | None:
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            taken, self._latest = self._latest, None
            self._ready.clear()
        return taken


def run_images(paths: list[Path], scene: HandheldScene) -> None:
    pose, hands_tracker = PoseTracker(num_poses=4, video=False), HandTracker(video=False)
    for path in paths:
        frame = read_image(path)
        poses = pose.detect_all(frame)
        hands = hands_tracker.detect(frame, people=poses)
        scene.tracks, scene.holders = [], {}
        scene.update(hands, PeopleTracker().update(poses, frame.shape[1::-1]), instant=True)
        scene.refresh_harmony(frame)
        raw, result = scene.render(frame, harmonize=False), scene.render(frame)
        debug = frame.copy()
        draw_hands(debug, hands)
        figure = labeled_grid([(debug, f"Hands: {sum(h.presented for h in hands)} held out of {len(hands)}"),
                               (raw, "Objects, raw"), (result, "Harmonized + camera match")])
        print(f"{path.name}: {len(hands)} hands, objects: {[t.artefact.name for t in scene.tracks]} ->",
              write_image(OUTPUT_DIR / f"{path.stem}_ar.jpg", figure).name)


def run_stream(args, scene: HandheldScene) -> None:
    cap = cv2.VideoCapture(str(args.video)) if args.video else open_webcam(args.camera, args.resolution)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose, hands_tracker = PoseTracker(num_poses=args.people), HandTracker(max_hands=args.max_hands)
    people_tracker = PeopleTracker()
    classifier = LiveActionClassifier() if args.actions and ONNX_PATH.exists() else None
    live_action = [None]

    def classify(poses, t):  # on the body thread, with every new pose
        if classifier and poses:
            live_action[0] = classifier.push(poses[0], t) or live_action[0]

    # live: the pose of everyone, their ids and the far hands at the classifier's 15 Hz, in parallel
    body = None if args.video else BodyThread(pose, hands_tracker, SAMPLE_RATE, classify)
    source = None if args.video else LiveSource(cap, hands_tracker, body)  # live: the next frame's hands meanwhile
    if body:
        body.start()
        source.start()
    writer, action, meter, harmony, objects, debug, i, note = None, None, RateMeter(), True, True, False, 0, None
    recording, written = 0.0, 0  # live recording: when it started, frames written
    try:
        while True:
            if args.video:  # frame by frame, on the video's own clock: reproducible
                ok, frame = cap.read()
                if not ok:
                    break
                small = detection_copy(frame)  # the hands and poses are found here, the objects drawn into frame
                t = i / video_fps
                poses = pose.detect_all(small, t)
                hands = hands_tracker.detect(small, t, poses)
                people = people_tracker.update(poses, small.shape[1::-1], t)
                if classifier and poses:
                    action = classifier.push(poses[0], t) or action
            else:  # the newest frame, its near hands found by the source thread meanwhile
                taken = source.take()
                if taken is None:
                    continue
                frame, small, near = taken
                _, _, people, far = body.result
                hands, action = merge_hands(near, far), live_action[0]
            scale = frame.shape[0] / small.shape[0]
            i += 1
            scene.update(hands, people, t if args.video else None)
            out = scene.render(frame, harmony, small) if objects else frame.copy()
            fps = meter.tick()
            if debug:
                draw_hands(out, hands, scale)
                draw_pairs(out, scene, scale)
            draw_status(out, scene, action, fps, harmony, objects)
            if args.out:
                if writer is None:
                    writer = video_writer(args.out, video_fps if args.video else RECORD_FPS, out.shape[1::-1])
                    recording = time.perf_counter()
                # a video file: every frame; live: as many copies as the clock asks for (none if the loop
                # ran ahead), so the recording plays at real speed whatever the loop's frame rate
                due = i if args.video else int((time.perf_counter() - recording) * RECORD_FPS) + 1
                for _ in range(due - written):
                    writer.write(out)
                written = max(written, due)
            if args.no_show:
                continue
            if i == 1:
                open_window(out.shape[1::-1])
            key = show_live(WINDOW, with_note(out, note))  # Q, Esc or the window's X button quit
            if key == ord("q"):
                break
            if key == ord("s"):  # the frame shown, without the note
                path = save_shot(out, "objects", SCREENSHOTS_DIR)
                note = (f"Saved {path.name}", time.perf_counter() + NOTE_S)
                print("Saved:", path)
            harmony ^= key == ord("h")
            objects ^= key == ord("o")
            debug ^= key == ord("d")
            if key == ord("n"):
                scene.next_object()
    finally:
        if source:
            source.stopped = True
            source.join(1.0)
        if body:
            body.stopped = True
            body.join(1.0)
        if scene.worker:
            scene.worker.close()
        cap.release()
        if writer:
            writer.release()
            print("Saved:", args.out)
        cv2.destroyAllWindows()
        pose.close()
        hands_tracker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Museum objects in the hands held out to the camera")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Webcam index")
    source.add_argument("--video", type=Path, help="A recorded video instead of the webcam")
    source.add_argument("--image", type=Path, nargs="+", help="Still images (results in data/ar_results)")
    parser.add_argument("--resolution", default=CAMERA_SIZE, metavar="WxH",
                        type=lambda text: tuple(int(v) for v in text.lower().split("x")),
                        help="Webcam resolution (default: Full HD; the webcam gives the closest one it has)")
    parser.add_argument("--out", type=Path,
                        help="Also save what is shown as an .mp4 video (H.264: plays in browsers; live at real speed)")
    parser.add_argument("--harmony", default="ensemble", help="Harmonization method (see harmonize.py), or none")
    parser.add_argument("--max-hands", type=int, default=2,
                        help="Hands tracked on the whole frame (more people's hands are found around their wrists)")
    parser.add_argument("--cycle", type=float, default=4.0, metavar="S",
                        help="Both hands up: seconds between objects (0 = keep them)")
    parser.add_argument("--alternate", type=float, default=2.0, metavar="S",
                        help="Both hands up: seconds harmonized, then raw (0 = always harmonized)")
    parser.add_argument("--colour", type=float, default=COLOUR_SHARE, metavar="SHARE",
                        help="Share of the harmonization's colour change given to the objects (their lightness is always "
                             "matched): 0 = their own colours, 1 = all of it")
    parser.add_argument("--people", type=int, default=2, help="People tracked (each gets their own objects)")
    parser.add_argument("--actions", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the Module 1 action classifier's output")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    scene = HandheldScene(load_catalog(), args.harmony, sync=bool(args.image or args.video), cycle_s=args.cycle,
                          alternate_s=args.alternate, colour_share=args.colour)
    if args.image:
        run_images(args.image, scene)
    else:
        run_stream(args, scene)


if __name__ == "__main__":
    main()
