"""
Module 1 + 2 interaction: hand tracking with the MediaPipe Hand Landmarker (21 landmarks per
hand, several hands - of several people - at once).

Besides the image landmarks the model returns metric "world" landmarks (metres, camera-aligned
axes), which give what the object placement needs:
  hold       how the hand is held out, if it is (it must not hang down along the body):
               "open" - at least three straight fingers (measured as 3D joint angles, so the
                        palm may face the camera or the ceiling), palm not turned to the floor
                        (that is a hand resting on a lap or a table): an object stands on it
               "grip" - at least three bent fingers, as when gripping a vase by its handles
                        (the back of the hand is usually what the camera sees)
  px_per_m   the image scale at the hand: knuckle span in pixels / in metres, so an object of
             known real height can be drawn at its real size in the hand
Every hand is also attached to a person (PeopleTracker gives the pose model's people ids that
stay the same from frame to frame) and to one of that person's arms, so that hands of different
people are never taken for one pair.
"""

import threading
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from common.paths import MODELS_DIR

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/latest/hand_landmarker.task")
PALM = [0, 5, 9, 13, 17]                  # wrist + the knuckles of the four fingers
FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]  # thumb..pinky
STRAIGHT_DEG = 140.0                      # a finger is straight when its middle joint opens wider than this
BENT_DEG = 130.0                          # ... and bent (gripping) when it opens less than this
HANGING_COS = 0.8                         # wrist -> knuckles within ~37 deg of straight down: the hand hangs
PALM_DOWN = 0.5                           # palm normal pointing down more than this: the palm is turned to the floor
SHOULDERS_M, TORSO_M = 0.36, 0.5          # typical adult shoulder-joint width and shoulder-to-hip height
WRISTS = (15, 16)                         # pose landmarks of the two wrists: arm 0 and arm 1


@dataclass
class Hand:
    points: np.ndarray  # (21, 2) image pixels
    world: np.ndarray   # (21, 3) metres around the hand centre, x right, y down, z away from the camera
    side: str           # "Left" / "Right", as MediaPipe classifies the hand's appearance
    score: float        # confidence of that classification
    person: int | None = None  # id of the person it belongs to (assign_people), None if no body was found
    arm: int | None = None     # 0 / 1: which of that person's arms

    @property
    def centre(self) -> np.ndarray:
        return self.points[PALM].mean(0)

    @property
    def width_px(self) -> float:
        return float(np.linalg.norm(self.points[5] - self.points[17]))

    @property
    def px_per_m(self) -> float:
        return self.width_px / max(float(np.linalg.norm(self.world[5] - self.world[17])), 1e-3)

    def finger_angles(self) -> np.ndarray:
        """Opening of the middle joint of the index..pinky fingers, degrees (180 = straight)."""
        angles = []
        for mcp, pip, _, tip in FINGERS[1:]:
            a, b = self.world[mcp] - self.world[pip], self.world[tip] - self.world[pip]
            cos = a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9)
            angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        return np.array(angles)

    @property
    def hanging(self) -> bool:
        d = self.world[9] - self.world[0]
        return d[1] / max(float(np.linalg.norm(d)), 1e-9) > HANGING_COS

    @property
    def palm_normal(self) -> np.ndarray:
        """Unit vector out of the palm. Which way the knuckle cross product points depends on the
        hand's side (checked on photos: z < 0 = the palm faces the camera)."""
        n = np.cross(self.world[5] - self.world[0], self.world[17] - self.world[0])
        return (-n if self.side == "Left" else n) / max(float(np.linalg.norm(n)), 1e-9)

    @property
    def hold(self) -> str | None:
        """"open", "grip" or None (not held out) - see the module docstring."""
        if self.hanging:
            return None
        angles = self.finger_angles()
        if (angles < BENT_DEG).sum() >= 3:
            return "grip"
        palm_down = self.score >= 0.7 and self.palm_normal[1] > PALM_DOWN  # trust the side only when sure
        return "open" if (angles > STRAIGHT_DEG).sum() >= 3 and not palm_down else None

    @property
    def presented(self) -> bool:
        return self.hold is not None


def _mp_image(bgr: np.ndarray):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def _hands(result, scale: tuple[float, float], offset: tuple[float, float] = (0, 0)) -> list[Hand]:
    return [Hand(np.array([[p.x * scale[0] + offset[0], p.y * scale[1] + offset[1]] for p in lm], np.float32),
                 np.array([[p.x, p.y, p.z] for p in wl], np.float32), hd[0].category_name, float(hd[0].score))
            for lm, wl, hd in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness)]


class HandTracker:
    """Two passes, so that both near and far hands are found:
    near  the whole frame (VIDEO mode with temporal tracking for live/video input, IMAGE mode for
          stills): hands that are large in the image, even when the body is out of view;
    far   for every person found by the pose model whose wrist is raised but has no hand yet, a
          crop around that wrist, enlarged - the hand detector sees the whole frame at 192 px,
          too coarse for the small hands of people further away.
    The two passes use separate detectors, so they can run in different threads."""

    def __init__(self, max_hands: int = 4, min_confidence: float = 0.5, video: bool = True, crop_confidence: float = 0.4,
                 max_crops: int = 2):
        from mediapipe.tasks.python import BaseOptions, vision

        if not HAND_MODEL_PATH.exists():
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

        def landmarker(mode, n: int, confidence: float):
            return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)), running_mode=mode, num_hands=n,
                min_hand_detection_confidence=confidence, min_hand_presence_confidence=confidence,
                min_tracking_confidence=confidence))

        self.full = landmarker(vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE, max_hands, min_confidence)
        self.crop = landmarker(vision.RunningMode.IMAGE, 1, crop_confidence)  # a crop is already known to hold a wrist
        self.video, self.max_crops, self.t0, self.last_ms = video, max_crops, time.perf_counter(), -1

    def detect_near(self, frame_bgr: np.ndarray, t: float | None = None) -> list[Hand]:
        """Hands found on the whole frame; t = frame time in seconds for video files (default: now)."""
        h, w = frame_bgr.shape[:2]
        image = _mp_image(frame_bgr)
        if not self.video:
            return _hands(self.full.detect(image), (w, h))
        now = time.perf_counter() - self.t0 if t is None else t
        self.last_ms = max(int(now * 1000), self.last_ms + 1)
        return _hands(self.full.detect_for_video(image, self.last_ms), (w, h))

    def detect_far(self, frame_bgr: np.ndarray, people: list[np.ndarray], known: list[Hand]) -> list[Hand]:
        """Hands at the raised wrists of `people` (PoseTracker.detect_all) that `known` does not cover."""
        h, w = frame_bgr.shape[:2]
        found: list[Hand] = []
        for person in people:
            pts = person[:, :2] * (w, h)
            for wrist, elbow, hip in ((15, 13, 23), (16, 14, 24)):
                forearm = pts[wrist] - pts[elbow]
                size = 1.6 * np.linalg.norm(forearm)
                if (len(found) >= self.max_crops or person[wrist, 3] < 0.5 or pts[wrist, 1] > pts[hip, 1] or size < 16
                        or any(np.linalg.norm(hand.centre - pts[wrist]) < size for hand in known + found)):
                    continue  # enough crops, unsure, hanging below the hip, too small, or already found
                c = pts[wrist] + 0.35 * forearm  # the hand continues the forearm
                x1, y1 = (max(0, int(v - size)) for v in c)
                x2, y2 = min(w, int(c[0] + size)), min(h, int(c[1] + size))
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                crop = cv2.resize(frame_bgr[y1:y2, x1:x2], (256, max(16, round((y2 - y1) * 256 / (x2 - x1)))))
                found += _hands(self.crop.detect(_mp_image(crop)), (x2 - x1, y2 - y1), (x1, y1))
        return found

    def detect(self, frame_bgr: np.ndarray, t: float | None = None, people: list[np.ndarray] = ()) -> list[Hand]:
        """Both passes, one after the other."""
        near = self.detect_near(frame_bgr, t)
        return near + self.detect_far(frame_bgr, list(people), near)

    def close(self) -> None:
        self.full.close()
        self.crop.close()


class BodyThread(threading.Thread):
    """Live: the pose of everyone in view (PoseTracker.detect_all, several people), their ids
    (PeopleTracker) and the far-hand search, on the newest frame, at most rate_hz times a second,
    in parallel with the caller's loop (MediaPipe releases the GIL) - the multi-person pose costs
    ~35 ms, so a live loop only waits for the near hands. `result` holds the latest
    (frame time, poses, people, far hands); on_pose(poses, t) is called with every new pose."""

    def __init__(self, pose, hands: HandTracker, rate_hz: float = 20.0, on_pose=None):
        super().__init__(daemon=True)
        self.pose, self.hands, self.period, self.on_pose = pose, hands, 1.0 / rate_hz, on_pose
        self.people = PeopleTracker()
        self.result: tuple = (None, [], [], [])
        self._job, self._lock, self._wake, self.stopped = None, threading.Lock(), threading.Event(), False

    def submit(self, frame: np.ndarray, t: float, near: list[Hand], far: bool = True) -> None:
        """A frame taken at time t (time.perf_counter); far=False skips the far-hand search."""
        with self._lock:
            self._job = (frame, t, near, far)
        self._wake.set()

    def run(self) -> None:
        while not self.stopped:
            if not self._wake.wait(0.2):
                continue
            start = time.perf_counter()
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is not None:
                frame, t, near, search = job
                poses = self.pose.detect_all(frame)
                far = self.hands.detect_far(frame, poses, near) if search else []
                self.result = (t, poses, self.people.update(poses, frame.shape[1::-1], t), far)
                if self.on_pose:
                    self.on_pose(poses, t)
            time.sleep(max(0.0, self.period - (time.perf_counter() - start)))

    def stop(self) -> None:
        self.stopped = True
        self.join(1.0)


def merge_hands(near: list[Hand], far: list[Hand]) -> list[Hand]:
    """near + the far hands that no near hand already covers (the passes may run on different frames)."""
    return near + [f for f in far if all(np.linalg.norm(f.centre - n.centre) > 1.5 * n.width_px for n in near)]


@dataclass
class Person:
    id: int
    points: np.ndarray      # (33, 2) pose landmarks in pixels
    visibility: np.ndarray  # (33,)

    @property
    def px_per_m(self) -> float:
        """Image scale at the body, from the shoulder width or the torso height, whichever is
        larger (turning sideways shrinks the first, leaning the second)."""
        shoulders = float(np.linalg.norm(self.points[11] - self.points[12])) / SHOULDERS_M
        if self.visibility[[23, 24]].min() < 0.5:  # hips out of view
            return shoulders
        mid = lambda a, b: (self.points[a] + self.points[b]) / 2
        return max(shoulders, float(np.linalg.norm(mid(11, 12) - mid(23, 24))) / TORSO_M)

    @property
    def chest(self) -> np.ndarray:
        return (self.points[11] + self.points[12]) / 2


class PeopleTracker:
    """Ids that stay the same from frame to frame for the people PoseTracker.detect_all finds:
    each person takes the id of the nearest person seen in the last keep_s seconds (chest points
    less than half a metre apart) - so someone the pose model misses for a moment keeps their id -
    the others get new ids."""

    def __init__(self, keep_s: float = 1.0):
        self.people: list[Person] = []                   # the people of the last call
        self.known: list[tuple[Person, float]] = []      # everyone seen in the last keep_s seconds, and when
        self.next_id, self.keep_s = 0, keep_s

    def update(self, poses: list[np.ndarray], size: tuple[int, int], t: float | None = None) -> list[Person]:
        """t = frame time in seconds (default: now)."""
        now = time.perf_counter() if t is None else t
        current = [Person(-1, pose[:, :2] * size, pose[:, 3]) for pose in poses]
        recent = [(q, seen) for q, seen in self.known if now - seen <= self.keep_s]
        unmatched = list(range(len(recent)))
        for person in sorted(current, key=lambda p: -p.px_per_m):  # nearest people choose first
            close = [i for i in unmatched if np.linalg.norm(recent[i][0].chest - person.chest) < 0.5 * person.px_per_m]
            if close:
                match = min(close, key=lambda i: np.linalg.norm(recent[i][0].chest - person.chest))
                person.id = recent[match][0].id
                unmatched.remove(match)
            else:
                person.id, self.next_id = self.next_id, self.next_id + 1
        self.known = [(p, now) for p in current] + [recent[i] for i in unmatched]
        self.people = current
        return current


def assign_people(hands: list[Hand], people: list[Person], recent=None) -> None:
    """Attach every hand to a person and arm, in place: first to the pose wrist nearest to the
    hand's own wrist (within 25 cm - the pose may be a frame or two older than the hand), then -
    recent(hand, taken) -> (person, arm) or None, if given - to the person it was with a moment
    ago (the pose model may miss a body for a frame or two, and a hand does not change owners),
    then any hand left over to the nearest person within arm's reach (1 m from the chest) who
    still has a free arm. Hands with no body nearby keep person None."""
    for hand in hands:  # far hands are reused for a few frames: attach them again
        hand.person = hand.arm = None
    taken: set[tuple[int, int]] = set()
    pairs = sorted((float(np.linalg.norm(hand.points[0] - person.points[wrist])), i, person.id, arm)
                   for i, hand in enumerate(hands) for person in people
                   for arm, wrist in enumerate(WRISTS) if person.visibility[wrist] > 0.3)
    for d, i, pid, arm in pairs:
        person = next(p for p in people if p.id == pid)
        if hands[i].person is None and (pid, arm) not in taken and d < max(0.25 * person.px_per_m, 2.5 * hands[i].width_px):
            hands[i].person, hands[i].arm = pid, arm
            taken.add((pid, arm))
    for hand in hands:
        place = recent(hand, taken) if recent and hand.person is None else None
        if place:
            hand.person, hand.arm = place
            taken.add(place)
    for hand in hands:
        if hand.person is not None:
            continue
        reach = [(float(np.linalg.norm(hand.centre - p.chest)) / p.px_per_m, p) for p in people]
        for dist, person in sorted(reach, key=lambda r: r[0]):
            free = [arm for arm in (0, 1) if (person.id, arm) not in taken]
            if dist > 1.0 or not free:
                continue
            arm = min(free, key=lambda a: np.linalg.norm(hand.centre - person.points[11 + a]))  # the nearer shoulder
            hand.person, hand.arm = person.id, arm
            taken.add((person.id, arm))
            break


def skin_mask(points: np.ndarray, shape: tuple[int, int], thickness: float = 0.2) -> np.ndarray:
    """uint8 mask (0/1) of the hand itself: the palm polygon and the fingers drawn as lines
    `thickness` knuckle spans thick - unlike the convex hull, no background between the fingers."""
    mask = np.zeros(shape, np.uint8)
    pts = points.astype(np.int32)
    cv2.fillPoly(mask, [pts[[0, 1, 5, 9, 13, 17]]], 1)
    width = max(2, round(thickness * float(np.linalg.norm(points[5] - points[17]))))
    cv2.polylines(mask, [pts[list(chain)] for chain in FINGERS], False, 1, width)
    return mask


def draw_hands(frame: np.ndarray, hands: list[Hand], scale: float = 1.0) -> None:
    """Hand skeletons, in place: green when held open, orange when gripping, grey otherwise.
    scale = how much larger the frame is than the one the hands were found in."""
    thick, radius = max(2, round(2 * frame.shape[0] / 480)), max(4, round(4 * frame.shape[0] / 480))
    for hand in hands:
        colour = {"open": (0, 220, 0), "grip": (0, 160, 255)}.get(hand.hold, (160, 160, 160))
        pts = (hand.points * scale).astype(int)
        for chain in FINGERS:
            cv2.polylines(frame, [pts[[0, *chain]]], False, colour, thick)
        cv2.polylines(frame, [pts[[5, 9, 13, 17]]], False, colour, thick)
        cv2.circle(frame, tuple((hand.centre * scale).astype(int)), radius, (0, 0, 255), -1)
