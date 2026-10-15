"""
Module 1 + 2 together: museum objects appear in the hands people hold out to the camera.

Each held-out hand (common/hands.py) belongs to a person - hands of different people never
share an object - and each person's hands get:
  one hand                      a small object (up to 25 cm, or one the catalog gives to one hand)
  both hands, closer than 50 cm one larger object held with both hands, first the one whose real
                                width best fits the gap
  both hands, farther than 70 cm one small object in each hand
With one hand up, the N and H keys change the object and switch harmonization. With both hands
up they are busy, so - with one large or two small objects - the objects change by themselves
every 4 s, each shown harmonized for its first 2 s and raw for the next 2 s, on a clock that
starts when the second hand comes up and keeps running through brief losses of a hand and
changes between one large and two small objects, so the difference can always be seen.
The objects come in shuffled rounds, one per size: every object once before any comes again,
never the same one in two hands at once, and none straight back from the other hand at the
next change.
Between 50 and 70 cm nothing changes (hysteresis), a change must hold for 0.35 s and comes at
least 1.5 s after the previous one, and a hand lost for a few frames keeps its place for 0.8 s -
so the objects never flicker between one large and two small ones. Likewise a hand stays with its
person while the pose model misses the body for a moment, and if the person gets a new id or the
arms are swapped, the objects (and the rhythm) follow the hands. Two hands with no body in view
are told apart by position when the hand model gives both the same side.
The object is held the way the hands are held:
  open hands      it stands on the palm(s), or is held at its belly when the hands are about as
                  far apart as it is wide (a sword lying lengthwise always rests on the palms)
  gripping hands  it is held by its handles, which are put in the hands (a weapon by its grip, an
                  object without handles as open hands hold it)
Every object is drawn at its real size - the hands give the pixels per metre at their depth
(not the body: hands held out towards the camera are much nearer to it than the shoulders) -
adjusted by at most 30 % so that two hands really touch its belly or handles. Position and size are smoothed with a One Euro filter:
steady while the hands are still, little lag when they move.

The objects are Module 2 cut-outs (build_catalog.py) and get the Module 2 realism cues, made
cheap enough for live video:
  harmonization  a background process harmonizes the raw composite twice per second and fits,
                 per object, the colour matrix that maps its raw colours to the harmonized ones -
                 with the networks' lightness change whole but only a quarter of their colour
                 change (COLOUR_SHARE): they are trained to repair objects whose colours are wrong,
                 while a museum photo's colours are right and only the light differs; live, each
                 update runs one network of the ensemble in turn and the matrix is the mean of the
                 last three fits - the fit is linear in its target, so that is (nearly) the
                 ensemble's fit at a third of the cost. The sprite is re-coloured once per update,
                 not once per frame
  camera match   the camera's sensor noise, measured on the image the objects are drawn into; the
                 object keeps the sharpness of its photo (MAX_BLUR): the webcam's sharpest edges are
                 sharper than the cut-outs', and the hands' smooth skin always read as blurred
The scene works in the image the hands were found in - live, a 540-row copy of the Full HD
camera frame - and the objects are drawn into the camera frame itself, at its scale (layers,
render and describe take that scale), from cut-outs at their photos' full resolution (up to 2400 px tall;
a sword, lying, ~3600 px long).
  occlusion      the hands stay in front of an object held at its belly or handles
  shadow         a soft contact shadow on the palm under an object standing on it
"""

import contextlib
import json
import os
import queue
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from common.hands import Hand, Person, assign_people, skin_mask
from common.paths import DATA_DIR

CATALOG_DIR = DATA_DIR / "objects"
CATALOG_PATH = CATALOG_DIR / "catalog.json"
ONE_HAND_MAX_M = 0.25              # one hand holds the objects up to 25 cm tall, two hands the larger ones
                                   # (unless the catalog says which: a dagger, a sword lying lengthwise)
PAIR_M, SPLIT_M = 0.5, 0.7         # a person's two hands share one object below PAIR_M apart, get one each above SPLIT_M
CONFIRM_S, DWELL_S = 0.35, 1.5     # a pair/split change must hold CONFIRM_S, at least DWELL_S after the last one
GRACE_S = 0.8                      # a hand lost for less than this keeps its place
BUSY_GRACE_S = 1.0                 # both hands count as up through gaps shorter than this
KEEP_S = 1.0                       # an object whose hands vanished is kept (hidden) this long, to resume
APPEAR_S = 0.25                    # a new object waits this long (the second hand may be on its way)
SWITCH = 12                        # frames before an object is swapped for a better fit
CYCLE_S = 4.0                      # both hands up: seconds between objects (0 = keep them)
ALTERNATE_S = 2.0                  # both hands up: harmonized, then raw, this long each (0 = always harmonized)
FIT_RANGE = (0.7, 1.3)             # how much two hands may resize an object to touch it
HARMONY_PERIOD_S = 0.5             # live: seconds between two harmonization updates
HARMONY_EVERY = 6                  # video files and stills: frames between two updates (reproducible)
MAX_BLUR = 0.0                     # px per 480 image rows: the most the camera match may blur an object in the
                                   # hands - none: the webcam's sharpest edges (10-90% in 1.25 px at 1080p) are
                                   # sharper than a cut-out's drawn unblurred (1.75 px), while the hands' smooth
                                   # skin always reads blurred (the estimate sat at the old 1.5 px cap)
COLOUR_SHARE = 0.25                # share of the harmonization's colour (a, b) change given to an object (lightness: all)
SAMPLED_RATIO = 1.4                # an object drawn taller than its cut-out / this is sampled where it is seen, every
                                   # frame (bilinear: within 44 dB of INTER_AREA up to there); smaller, resized whole
IDENTITY = np.vstack([np.eye(3), np.zeros((1, 3))]).astype(np.float32)  # [rgb, 1] @ IDENTITY = rgb
WEIGHT = np.arange(256, dtype=np.float32) * np.float32(1 / 255)  # alpha (uint8) -> blending weight, by cv2.LUT
REST = 1 - WEIGHT
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")  # math libraries' thread pools
ENSEMBLE = ("harmonizer", "pctnet", "pctnet_cnn")


# ── Object library ────────────────────────────────────────────────────────

@dataclass(eq=False)
class Artefact:
    name: str
    height_m: float
    sprite: np.ndarray                         # BGRA cut-out, cropped to the object
    handles_at: float | bool | None = None     # catalog "handles": None = its handles are the widest row of its
                                               # upper half; a fraction of the height from the top = a grip holds it
                                               # there (a weapon's grip); False = none: held as by open hands
    hands: int | None = None                   # catalog "hands": 1 or 2 hold it, whatever its height
    size_m: float | None = None                # the size shown: the catalog's (a sword lying lengthwise: its length)
    lengthwise: bool = False                   # lies lengthwise (catalog "width_m"): carried on the palms, never
                                               # held at its sides
    belly: float = field(init=False)           # widest row of the lower body (fraction of the height from the top)
    belly_width_m: float = field(init=False)   # its real width there: open hands hold it at its sides there
    handles: float | None = field(init=False)  # where a grip holds it (fraction of the height from the top)
    handles_width_m: float = field(init=False)

    def __post_init__(self):
        widths = (self.sprite[..., 3] > 127).sum(1)
        n = len(widths)

        def widest(lo: float, hi: float) -> tuple[float, float]:
            row = int(lo * n) + int(np.argmax(widths[int(lo * n):int(hi * n) + 1]))
            return row / n, widths[row] / n * self.height_m

        self.belly, self.belly_width_m = widest(0.4, 0.85)
        if self.handles_at is False:  # no handles: gripping hands hold it as open hands do
            self.handles, self.handles_width_m = None, 0.0
        elif self.handles_at is None:
            self.handles, self.handles_width_m = widest(0.05, 0.5)
        else:
            row = min(n - 1, int(self.handles_at * n))
            self.handles, self.handles_width_m = row / n, widths[row] / n * self.height_m
        self.size_m = self.size_m or self.height_m

    @property
    def one_hand(self) -> bool:
        """Held in one hand: small (ONE_HAND_MAX_M), unless the catalog says which."""
        return self.hands == 1 if self.hands else self.height_m <= ONE_HAND_MAX_M

    @cached_property
    def texture(self) -> np.ndarray:
        """What the object is drawn from: the cut-out with its colours bled into its transparent
        pixels (_bleed), made the first time it is needed."""
        return _bleed(self.sprite)


def catalog_entries() -> list[dict]:
    """catalog.json: per object the photo (in photos/) and its real height in metres - or its width,
    for an object lying lengthwise - and how it is held, if not as usual (see build_catalog.py)."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def sprite_path(entry: dict) -> Path:
    return CATALOG_DIR / f"{Path(entry['photo']).stem}.png"


def load_catalog() -> list[Artefact]:
    """The objects of catalog.json that have a cut-out, smallest first. One given by its width ("width_m":
    a sword lying lengthwise) gets the height that width makes in its cut-out."""
    arts = []
    for e in catalog_entries():
        path = sprite_path(e)
        if path.exists():
            sprite = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            height = e["height_m"] if "height_m" in e else e["width_m"] * sprite.shape[0] / sprite.shape[1]
            arts.append(Artefact(path.stem, height, sprite, e.get("handles"), e.get("hands"), e.get("height_m", e.get("width_m")),
                                 "height_m" not in e))
    if not arts:
        raise FileNotFoundError(f"No object cut-outs in {CATALOG_DIR}: run python build_catalog.py first.")
    return sorted(arts, key=lambda a: a.height_m)


# ── Compositing helpers ───────────────────────────────────────────────────

def _paste(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    """dst = alpha * src + (1 - alpha) * dst inside the frame, in place (float arrays, alpha 0-1)."""
    h, w = src.shape[:2]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x2 > x1 and y2 > y1:
        part = (slice(y1 - y, y2 - y), slice(x1 - x, x2 - x))
        roi = dst[y1:y2, x1:x2]
        roi += alpha[part] * (src[part] - roi)


def _resize_bgra(bgra: np.ndarray, height: int) -> np.ndarray:
    """Float BGRA resize with premultiplied alpha, so no background colour bleeds into the edges."""
    pm = bgra.astype(np.float32)
    pm[..., :3] *= pm[..., 3:] / 255
    size = (max(1, round(bgra.shape[1] * height / bgra.shape[0])), height)
    pm = cv2.resize(pm, size, interpolation=cv2.INTER_AREA if height < bgra.shape[0] else cv2.INTER_LINEAR)
    pm[..., :3] /= np.maximum(pm[..., 3:] / 255, 1e-3)
    return pm


def _to_u8(image: np.ndarray) -> np.ndarray:
    """Rounded to the nearest (ties to even) and saturated, like OpenCV's convertTo."""
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _bleed(sprite: np.ndarray) -> np.ndarray:
    """The cut-out (uint8 BGRA) with every transparent pixel given the colour of the object around it -
    pull-push through a premultiplied pyramid: down to 1 px with cv2.pyrDown, then back up with
    cv2.pyrUp, each level keeping its own colours wherever it has any alpha - so that resizing or
    blurring its straight colours never pulls the photo's background or a dark rim into its edge."""
    premultiplied = sprite.astype(np.float32)
    premultiplied[..., 3] /= 255
    premultiplied[..., :3] *= premultiplied[..., 3:]
    levels = [premultiplied]
    while min(levels[-1].shape[:2]) > 1:
        levels.append(cv2.pyrDown(levels[-1]))
    colours = np.zeros(levels[-1].shape[:2] + (3,), np.float32)
    for level in reversed(levels):
        if colours.shape[:2] != level.shape[:2]:
            colours = cv2.pyrUp(colours, dstsize=level.shape[1::-1])
        np.divide(level[..., :3], level[..., 3:], out=colours, where=level[..., 3:] > 0)
    return np.dstack([_to_u8(colours), sprite[..., 3]])


def _resized(bgra: np.ndarray, w: int, h: int) -> np.ndarray:
    """A uint8 cut-out at w x h: INTER_AREA when smaller (every source pixel counts, as on a camera's
    sensor), bilinear when larger."""
    return cv2.resize(bgra, (w, h), interpolation=cv2.INTER_AREA if h < bgra.shape[0] else cv2.INTER_LINEAR)


def _blend(dst: np.ndarray, src: np.ndarray | None, weight: np.ndarray, x: int, y: int,
           rest: np.ndarray | None = None) -> None:
    """dst = weight * src + (1 - weight) * dst where weight (one-channel float, 0-1) placed at x, y
    overlaps dst, in place: uint8 images (src None = black, src as large as weight; rest = 1 - weight,
    if made already), through cv2.blendLinear - in parallel, one weight for the three channels,
    rounded to uint8."""
    h, w = weight.shape
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x2 > x1 and y2 > y1:
        part, inside = dst[y1:y2, x1:x2], (slice(y1 - y, y2 - y), slice(x1 - x, x2 - x))
        cv2.blendLinear(part, np.zeros_like(part) if src is None else src[inside],
                        1 - weight[inside] if rest is None else rest[inside], weight[inside], dst=part)


def _shadowed(layer) -> bool:
    """Whether a contact shadow goes under the object: when it stands on the palm(s) - not under a
    sword lying across two palms, whose base would cast it into the air between the hands."""
    return layer.mode == "palm" and not layer.track.artefact.lengthwise


def _occlude(roi: np.ndarray, camera: np.ndarray, hand: Hand, offset: tuple[int, int], scale: float, soft: float) -> None:
    """The camera's own pixels put back over the object where the hand is (it is in front of it), in
    place (uint8): a soft mask of the hand's skin (skin_mask, fingers as thick as real ones, edge blurred by
    soft px), made only around the hand - its box, the finger width and the blur's reach - where it
    can be above zero. scale = how much larger roi's image is than the one the hand was found in."""
    pts = hand.points * scale - np.asarray(offset, np.float32)
    reach = round(0.15 * float(np.linalg.norm(pts[5] - pts[17]))) + int(np.ceil(4 * soft)) + 3
    (x1, y1), (x2, y2) = np.floor(pts.min(0)).astype(int) - reach, np.ceil(pts.max(0)).astype(int) + reach + 1
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, roi.shape[1]), min(y2, roi.shape[0])
    if x2 > x1 and y2 > y1:
        mask = skin_mask(pts - (x1, y1), (y2 - y1, x2 - x1), thickness=0.3).astype(np.float32)
        _blend(roi[y1:y2, x1:x2], camera[y1:y2, x1:x2], cv2.GaussianBlur(mask, (0, 0), soft), 0, 0)


# ── Scene: tracking the hand groups and drawing their objects ─────────────

class OneEuro:
    """One Euro filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass filter whose cut-off rises
    with the speed, so a still hand is steady and a moving one is followed with little lag."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt, self.t = max(t - self.t, 1e-3), t
        self.dx = self.dx + self._alpha(dt, self.d_cutoff) * ((x - self.x) / dt - self.dx)
        self.x = self.x + self._alpha(dt, self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))) * (x - self.x)
        return self.x


def _filters() -> tuple[OneEuro, OneEuro, OneEuro]:
    """Filters for the centre (pixels), the log of the scale and the gap between the hands (metres)."""
    return OneEuro(1.0, 0.015), OneEuro(0.5, 0.5), OneEuro(0.7, 1.0)


@dataclass(eq=False)
class Track:
    id: int
    key: tuple                          # (person, "pair") or (person, arm): what the object is held by
    hands: list[Hand]
    born: float                         # when its hands were first seen
    centre: np.ndarray | None = None    # filtered centre of its hands (pixels)
    px_per_m: float = 0.0               # filtered image scale at the hands
    gap_m: float = 0.0                  # filtered distance between its two hands
    grip: bool = False                  # the hands grip (handles) instead of holding open
    grip_votes: int = 0                 # frames in a row the hands did the other thing
    busy: bool = False                  # both of the person's hands are up: automatic objects and harmonization
    clock: float = 0.0                  # when both hands came up: the automatic rhythm counts from there
    cycle: int = 0                      # the number of object changes of that rhythm already shown
    filters: tuple = field(default_factory=_filters)
    artefact: Artefact | None = None
    previous: Artefact | None = None    # the object it showed before: not to come back right away
    changed: float = -np.inf            # when its object last changed
    candidate: Artefact | None = None   # a better fitting object, waiting for SWITCH votes
    votes: int = 0
    seen: int = 0
    missing: int = 0
    last_seen: float = 0.0
    opacity: float = 0.0


@dataclass(eq=False)
class Holder:
    """One person's held-out hands, each remembered for GRACE_S, whether they share one object, and
    since when both are up (the clock of the automatic rhythm)."""
    arms: dict = field(default_factory=dict)  # arm -> (Hand, time it was last seen)
    paired: bool | None = None                # one object in both hands (None: not decided yet)
    want_since: float | None = None           # since when the other decision has held
    changed: float = -np.inf                  # when `paired` last changed
    both_since: float | None = None           # when both hands came up
    both_seen: float = -np.inf                # when both hands were last up


@dataclass
class Layer:
    track: Track
    w: int                              # the object's size in the frame
    h: int
    x: int
    y: int
    mode: str                           # "palm" (stands on it), "sides" (held at its belly) or "handles"
    scale: float = 1.0                  # image size / size of the image the hands were found in

    @property
    def bgra(self) -> np.ndarray:
        """The cut-out at the layer's size as uint8 BGRA: what the harmonization worker composites."""
        return _resized(self.track.artefact.texture, self.w, self.h)


class HandheldScene:
    def __init__(self, catalog: list[Artefact], method: str = "ensemble", sync: bool = False, cycle_s: float = CYCLE_S,
                 alternate_s: float = ALTERNATE_S, colour_share: float = COLOUR_SHARE):
        self.catalog, self.cycle_s, self.alternate_s = catalog, cycle_s, alternate_s
        self.tracks, self.next_id, self.frames, self.now = [], 0, 0, 0.0
        self.holders: dict = {}  # person id (or "free") -> Holder
        self._decks: dict[bool, list[Artefact]] = {}  # one hand? -> what is left of its shuffled round
        self._rng = np.random.default_rng(0 if sync else None)  # video files and stills: the same objects every run
        # starts loading the networks now
        self.worker = HarmonyWorker(method, sync, colour_share=colour_share) if method != "none" else None
        self._styled_cache, self._sprites, self._shadows, self._spans, self._next_harmony = {}, {}, {}, {}, 0.0
        self._work: dict[str, np.ndarray] = {}  # arrays made every frame, kept (_buffer)
        self._noise, self._noise_version = 0.0, -1  # the camera's grain, measured on the image drawn into
        self._grain = None  # sensor noise covering a Full HD frame: a random window of it every frame
        # made now, so that no object's first appearance holds up a frame - in parallel: OpenCV frees the
        # GIL, so 8 threads prepare the 30 cut-outs (~40 MP) in ~0.9 s instead of ~3.1 s
        with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
            list(pool.map(lambda art: art.texture, catalog))

    def pool(self, hands: int) -> list[Artefact]:
        """The small objects for one hand, the larger ones for two (any, if the catalog has none)."""
        return ([a for a in self.catalog if a.one_hand == (hands == 1)]
                or [min(self.catalog, key=lambda a: abs(a.height_m - ONE_HAND_MAX_M))])

    def next_object(self) -> None:
        """Every hand or pair of hands gets the next object of its size (the N key)."""
        for tr in self.tracks:
            if tr.artefact is not None:
                self._draw(tr)

    def _draw(self, tr: Track, fit=None) -> None:
        """The track's next object, from its pool's shuffled round: one not shown yet in this round,
        not shown now by any hand and not just replaced in this same change - the first of the
        round's order, or, given fit (object -> how badly it fits), the best fitting. When the round
        has nothing left that no hand shows, a new one is shuffled (the objects shown now or just
        replaced are skipped in it until later). More hands than objects: one has to show twice."""
        pool = self.pool(len(tr.hands))
        deck = self._decks.setdefault(len(tr.hands) == 1, [])
        deck[:] = [a for a in deck if a in pool]
        shown = {t.artefact for t in self.tracks if t is not tr} | {tr.artefact}
        recent = shown | {t.previous for t in self.tracks if t.changed == self.now}  # the other hand's object of a moment ago
        if all(a in shown for a in deck):  # the round is over: a new one
            deck[:] = [pool[i] for i in self._rng.permutation(len(pool))]
        candidates = [a for a in deck if a not in recent] or [a for a in deck if a not in shown] or deck
        pick = candidates[0] if fit is None else min(candidates, key=fit)
        deck.remove(pick)
        tr.previous, tr.artefact, tr.changed = tr.artefact, pick, self.now

    def update(self, hands: list[Hand], people: list[Person] = (), t: float | None = None, instant: bool = False) -> None:
        """Turn this frame's held-out hands into objects. people = PeopleTracker.update's output (the
        hands are attached to them); t = frame time in seconds (default: now); instant = a still image."""
        now = self.now = time.perf_counter() if t is None else t
        assign_people(hands, list(people), None if instant else lambda hand, taken: self._recent_place(hand, taken, now))
        free = sorted((hand for hand in hands if hand.person is None), key=lambda hand: hand.centre[0])
        sides = {id(hand): hand.side for hand in free}
        if len(free) == 2 and free[0].side == free[1].side:  # unsure of the side (palms held up): tell them apart by position
            sides = {id(free[0]): "Left", id(free[1]): "Right"}  # as MediaPipe names them in the mirrored view
        placed = []
        for hand in hands:  # hands with no body in view are grouped by their side, as one "person"
            key, arm = (hand.person, hand.arm) if hand.person is not None else ("free", sides[id(hand)])
            held = key in self.holders and arm in self.holders[key].arms
            if hand.presented or (held and not hand.hanging):  # once held out, a hand stays until it hangs or is lost
                self.holders.setdefault(key, Holder()).arms[arm] = (hand, now)
                placed.append((key, arm, hand))
        for key, arm, hand in placed:  # a hand now seen under another person or arm: its old place is not a second hand
            for other_key, holder in list(self.holders.items()):
                for other_arm, (old, seen) in list(holder.arms.items()):
                    if seen < now and np.linalg.norm(old.centre - hand.centre) < 1.5 * hand.width_px:
                        del holder.arms[other_arm]
                        self._moved(holder, other_key, other_arm, key, arm)
        observed = []
        for key, holder in list(self.holders.items()):
            holder.arms = {arm: v for arm, v in holder.arms.items() if now - v[1] <= (0.0 if instant else GRACE_S)}
            arms = sorted(holder.arms.items(), key=lambda item: str(item[0]))
            if len(arms) >= 2:
                if holder.both_since is None or now - holder.both_seen > BUSY_GRACE_S:
                    holder.both_since = now  # the second hand came up: the automatic rhythm starts
                holder.both_seen = now
            busy = holder.both_since is not None and now - holder.both_seen <= (0.0 if instant else BUSY_GRACE_S)
            clock = holder.both_since if busy else now
            if len(arms) < 2:
                holder.paired = None  # decided again when the second hand comes back
                observed += [((key, arm), [hand], busy, clock) for arm, (hand, _) in arms]
                if not arms and not busy:
                    del self.holders[key]
                continue
            (arm_a, (a, _)), (arm_b, (b, _)) = arms[:2]
            ppm = self._scale([a, b])
            self._decide(holder, float(np.linalg.norm(a.centre - b.centre)) / ppm, now, instant)
            observed += ([((key, "pair"), [a, b], True, clock)] if holder.paired
                         else [((key, arm_a), [a], True, clock), ((key, arm_b), [b], True, clock)])
        tracks = {tr.key: tr for tr in self.tracks}
        holding = {tr.key[0] for tr in self.tracks if tr.opacity > 0}
        for key, group, busy, clock in observed:
            if key not in tracks:  # a person who already holds something only regroups: cross-fade at once
                tracks[key] = Track(self.next_id, key, group, now - APPEAR_S if key[0] in holding else now)
                self.tracks.append(tracks[key])
                self.next_id += 1
            self._observe(tracks[key], group, self._scale(group), now, busy, clock)
        seen = {key for key, *_ in observed}
        for tr in self.tracks:
            tr.missing = 0 if tr.key in seen else tr.missing + 1
            if instant:
                tr.opacity = 0.0 if tr.missing else 1.0
            elif tr.missing:
                tr.opacity = max(0.0, tr.opacity - 0.25)
            elif now - tr.born >= APPEAR_S:
                tr.opacity = min(1.0, tr.opacity + 0.34)
        # an object whose hands vanished stays (hidden) for KEEP_S: if they come back, it resumes as it was
        self.tracks = [tr for tr in self.tracks if not tr.missing or (not instant and now - tr.last_seen <= KEEP_S)]

    def _recent_place(self, hand: Hand, taken: set, now: float) -> tuple | None:
        """The person and arm of a hand held out at about this place in the last GRACE_S, if free."""
        for key, holder in self.holders.items():
            for arm, (old, seen) in holder.arms.items():
                if (key != "free" and (key, arm) not in taken and now - seen <= GRACE_S
                        and np.linalg.norm(old.centre - hand.centre) < 1.5 * hand.width_px):
                    return key, arm
        return None

    def _moved(self, old: Holder, old_key, old_arm, key, arm) -> None:
        """A hand went from (old_key, old_arm) to (key, arm) - the pose model gave its person a new id
        or swapped the arms: its object follows it with its colours and smoothing, so does the object
        it shares with the other hand, and a person new in this frame carries on the old one's rhythm."""
        for before, after in (((old_key, old_arm), (key, arm)), ((old_key, "pair"), (key, "pair"))):
            moved = next((tr for tr in self.tracks if tr.key == before), None)
            there = next((tr for tr in self.tracks if tr.key == after), None)
            if moved and moved is not there and (there is None or there.missing > moved.missing):
                moved.key = after
                if there:  # an object fading out there (the arms swapped back and forth) takes the old place
                    there.key = before
        new = self.holders[key]
        if new is not old and new.paired is None and new.both_since is None:
            new.paired, new.changed, new.both_since, new.both_seen = old.paired, old.changed, old.both_since, old.both_seen

    @staticmethod
    def _scale(group: list[Hand]) -> float:
        """Pixels per metre at the hands (smoothed per object by its One Euro filter)."""
        return float(np.mean([h.px_per_m for h in group]))

    @staticmethod
    def _decide(holder: Holder, gap_m: float, now: float, instant: bool) -> None:
        """One object for both hands or one each, with hysteresis and minimum durations."""
        if holder.paired is None or instant:
            holder.paired, holder.changed, holder.want_since = gap_m < (PAIR_M + SPLIT_M) / 2, now, None
            return
        want = gap_m < (SPLIT_M if holder.paired else PAIR_M)
        if want == holder.paired:
            holder.want_since = None
            return
        holder.want_since = now if holder.want_since is None else holder.want_since
        if now - holder.want_since >= CONFIRM_S and now - holder.changed >= DWELL_S:
            holder.paired, holder.changed, holder.want_since = want, now, None

    def _observe(self, tr: Track, group: list[Hand], ppm: float, now: float, busy: bool, clock: float) -> None:
        centre = np.mean([h.centre for h in group], axis=0)
        gap = float(np.linalg.norm(group[0].centre - group[-1].centre)) / ppm
        holds = [h.hold for h in group if h.hold]  # hands kept up without a clear way of holding do not vote
        grip = 2 * holds.count("grip") >= len(holds) if holds else tr.grip
        if tr.seen == 0:
            tr.grip = grip
        f_centre, f_scale, f_gap = tr.filters
        tr.centre, tr.gap_m = f_centre(centre, now), float(f_gap(gap, now))
        tr.px_per_m = float(np.exp(f_scale(np.log(ppm), now)))
        tr.grip_votes = tr.grip_votes + 1 if grip != tr.grip else 0
        if tr.grip_votes >= 4:  # open <-> grip only once it persists, so the object does not jump
            tr.grip, tr.grip_votes = grip, 0
        tr.hands, tr.seen, tr.last_seen = group, tr.seen + 1, now
        pool = self.pool(len(group))
        width = (lambda a: a.handles_width_m) if tr.grip else (lambda a: a.belly_width_m)
        fit = (lambda a: abs(width(a) - tr.gap_m)) if len(group) > 1 else None  # two hands: first the best fit to the gap
        best = min(pool, key=fit) if fit else None
        cycle = int((now - clock) / self.cycle_s) if busy and self.cycle_s else 0
        if busy and (not tr.busy or tr.clock != clock):  # the rhythm (re)started: follow it from its current step
            tr.cycle = cycle
        tr.busy, tr.clock = busy, clock
        if tr.artefact is None or tr.artefact not in pool:  # its first object
            self._draw(tr, fit)
            tr.votes = 0
        elif busy and cycle != tr.cycle:  # both hands are busy: the next object every cycle_s seconds, on the clock
            self._draw(tr)
            tr.cycle = cycle
            tr.opacity = min(tr.opacity, 0.3)  # a short fade-in marks the change
        elif not busy and best is not None and best is not tr.artefact:  # a better fit must persist before it replaces it
            tr.votes = tr.votes + 1 if best is tr.candidate else 1
            tr.candidate = best
            if tr.votes >= SWITCH:
                tr.previous, tr.artefact, tr.changed, tr.votes = tr.artefact, best, now, 0
        else:
            tr.votes = 0

    def layers(self, scale: float = 1.0) -> list[Layer]:
        """Where every visible object goes, farthest first; scale = how much larger the image they
        are drawn into is than the one the hands were found in (the scene works in the latter)."""
        out = []
        for tr in sorted(self.tracks, key=lambda tr: tr.px_per_m):
            if tr.opacity <= 0 or tr.artefact is None:
                continue
            art, several = tr.artefact, len(tr.hands) > 1
            if tr.grip and art.handles is not None:  # (no handles: held as by open hands)
                mode, row, width = "handles", art.handles, art.handles_width_m
            elif several and not art.lengthwise and tr.gap_m >= 0.6 * art.belly_width_m:
                mode, row, width = "sides", art.belly, art.belly_width_m
            else:
                mode, row, width = "palm", 1.0, 0.0
            fit = float(np.clip(tr.gap_m / width, *FIT_RANGE)) if several and width else 1.0
            h = max(8, round(art.height_m * tr.px_per_m * fit * scale))
            w = max(1, round(art.sprite.shape[1] * h / art.sprite.shape[0]))
            x, y = tr.centre[0] * scale - w / 2, tr.centre[1] * scale - row * h  # that row at the hands
            out.append(Layer(tr, w, h, round(x), round(y), mode, scale))
        return out

    def refresh_harmony(self, frame: np.ndarray) -> None:
        """Harmonize now instead of at the next periodic update (stills)."""
        if self.worker:
            self.worker.submit(frame, self.layers())

    def _sprite(self, layer: Layer, visible: tuple[int, int, int, int], fit=None) -> tuple[np.ndarray, ...]:
        """The part of the layer's object in the frame (visible = x1, y1, x2, y2): (BGR uint8, alpha
        weight 0-1, 1 - weight), re-coloured and re-sharpened as the latest harmonization update says
        when fit = (matrix, t in frame pixels) is given. Drawn well below its cut-out's size, the object
        is resized whole with INTER_AREA - every source pixel counts, as on a camera's sensor - and
        kept while its size and the update stay the same, so a still hand costs only the grain and
        the blending; drawn near or above it (SAMPLED_RATIO: a large vase close to the camera, often
        partly out of the frame) only its visible part is sampled, bilinear, every frame, from the
        cut-out sharpened at its own resolution."""
        from common.harmonization import _sharpness

        art, (x1, y1, x2, y2) = layer.track.artefact, visible
        version = None if fit is None else self.worker.version
        if layer.h * SAMPLED_RATIO > art.texture.shape[0]:
            step = int(np.floor(4 * np.log2(art.texture.shape[0] / layer.h) + 0.5))  # the blur's radius, in 2^(1/4) steps
            texture = self._styled(layer, fit, version, 2 ** (step / 4))
            sx, sy = layer.w / texture.shape[1], layer.h / texture.shape[0]  # cv2.resize's pixel alignment:
            m = np.array([[1 / sx, 0, (x1 - layer.x + 0.5) / sx - 0.5], [0, 1 / sy, (y1 - layer.y + 0.5) / sy - 0.5]])
            size = (y2 - y1, x2 - x1)
            drawn = cv2.warpAffine(texture, m, size[::-1], dst=self._buffer("drawn", size + (4,)),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            alpha = cv2.extractChannel(drawn, 3, dst=self._buffer("alpha", size))
            return (cv2.cvtColor(drawn, cv2.COLOR_BGRA2BGR, dst=self._buffer("colours", size + (3,))),
                    cv2.LUT(alpha, WEIGHT, dst=self._buffer("weight", size, np.float32)),
                    cv2.LUT(alpha, REST, dst=self._buffer("rest", size, np.float32)))
        tid, cached = layer.track.id, self._sprites.get(layer.track.id)
        if cached is None or cached[0] != (art.name, layer.w, layer.h, version):
            size, most = (layer.h, layer.w), (art.texture.shape[0], max(art.texture.shape[1], layer.w))  # kept per object
            texture = self._styled(layer, fit, version)
            sized = cv2.resize(texture, size[::-1], dst=self._buffer((tid, "sized"), size + (4,), most=most),
                               interpolation=cv2.INTER_AREA if layer.h < texture.shape[0] else cv2.INTER_LINEAR)
            colours = cv2.cvtColor(sized, cv2.COLOR_BGRA2BGR, dst=self._buffer((tid, "colours"), size + (3,), most=most))
            alpha = cv2.extractChannel(sized, 3, dst=self._buffer((tid, "alpha"), size, most=most))
            cached = self._sprites[tid] = ((art.name, layer.w, layer.h, version), (
                colours if fit is None else _sharpness(colours, fit[1]),
                cv2.LUT(alpha, WEIGHT, dst=self._buffer((tid, "weight"), size, np.float32, most)),
                cv2.LUT(alpha, REST, dst=self._buffer((tid, "rest"), size, np.float32, most))))
        seen = (slice(y1 - layer.y, y2 - layer.y), slice(x1 - layer.x, x2 - layer.x))
        return tuple(a[seen] for a in cached[1])

    def _buffer(self, name, shape: tuple, dtype=np.uint8, most: tuple[int, int] = (0, 0)) -> np.ndarray:
        """The top-left corner, of that shape, of an array kept under that name (made at least `most`
        rows x cols, so that growing objects do not make it again) for the arrays made every frame or
        size: on Windows fresh memory costs ~0.2-0.5 ms per MB in page faults, as much as the work
        itself. Names (track id, what) belong to one object and go with it."""
        kept = self._work.get(name)
        if kept is None or kept.dtype != dtype or kept.shape[2:] != shape[2:] or kept.shape[0] < shape[0] or kept.shape[1] < shape[1]:
            old = (0, 0) if kept is None else kept.shape[:2]
            rows, cols = max(shape[0], most[0], old[0]), max(shape[1], most[1], old[1])
            kept = self._work[name] = np.empty((rows, cols) + shape[2:], dtype)
        return kept[:shape[0], :shape[1]]

    def _styled(self, layer: Layer, fit, version: int, px: float | None = None) -> np.ndarray:
        """The cut-out (uint8 BGRA) re-coloured as fit says, once per update whichever way the object is
        drawn - and, if px is given and the fit changes the sharpness, sharpened at the cut-out's own
        resolution, px of its pixels per frame pixel (kept beside it, for one px)."""
        from common.harmonization import _sharpness

        art = layer.track.artefact
        if fit is None:
            return art.texture
        matrix, t = fit  # [rgb, 1] @ matrix in 0-1 units = the 3x4 transform below in 0-255
        cached = self._styled_cache.get(layer.track.id)
        if cached is None or cached[0] != (art.name, version):
            bgr = cv2.transform(cv2.cvtColor(art.texture, cv2.COLOR_BGRA2BGR), np.hstack([matrix[:3].T, 255 * matrix[3:].T]))
            cached = self._styled_cache[layer.track.id] = [(art.name, version), np.dstack([bgr, art.texture[..., 3]]), None, None]
        if px is None or 0 <= t <= 0.05:  # the sharpness stays (_sharpness changes nothing there)
            return cached[1]
        if cached[2] != px:
            cached[2:] = [px, np.dstack([_sharpness(cv2.cvtColor(cached[1], cv2.COLOR_BGRA2BGR), t, px), art.texture[..., 3]])]
        return cached[3]

    def _grain_window(self, h: int, w: int) -> np.ndarray:
        """Unit sensor noise, the same in the three channels, as uint8 128 + 32 n: a random window of a
        fixed noise texture that covers a Full HD frame - a new one every frame (fresh beyond Full HD)."""
        def noise(rows: int, cols: int) -> np.ndarray:
            unit = np.empty((rows, cols), np.float32)
            cv2.randn(unit, 0.0, 1.0)
            return cv2.merge([_to_u8(128 + 32 * unit)] * 3)

        if self._grain is None:
            self._grain = noise(1152, 2048)
        if h > self._grain.shape[0] or w > self._grain.shape[1]:
            return noise(h, w)
        y, x = np.random.randint(0, self._grain.shape[0] - h + 1), np.random.randint(0, self._grain.shape[1] - w + 1)
        return self._grain[y:y + h, x:x + w]

    def harmonized(self, tr: Track) -> bool:
        """Whether an object is shown harmonized now: always while one hand holds it (the H key decides),
        but with both of the person's hands up it is shown harmonized and then raw, alternate_s
        seconds each, on the clock that started when both hands came up."""
        return not tr.busy or not self.alternate_s or int((self.now - tr.clock) / self.alternate_s) % 2 == 0

    def _feed_harmony(self, small: np.ndarray, layers: list[Layer] | None = None) -> None:
        """Give the harmonization worker the raw composite in the image the hands were found in
        (layers: at its scale, made only when due): every HARMONY_PERIOD_S live, every HARMONY_EVERY
        frames for video files and stills."""
        if self.worker:
            now = time.perf_counter()
            if (self.frames % HARMONY_EVERY == 0) if self.worker.sync else now >= self._next_harmony:
                self.worker.submit(small, self.layers() if layers is None else layers)
                self._next_harmony = now + HARMONY_PERIOD_S
        self.frames += 1

    def describe(self, frame: np.ndarray, scale: float = 1.0) -> list[dict]:
        """What render() would draw, as data for a renderer outside Python - the C++ app
        (cpp_app/object_renderer.cpp, which must stay in step with render()): per object its
        sprite, size and place in the frame, how it is held, its opacity, whether it is in its raw
        phase, the harmonization fit (4x3 colour matrix row by row, sharpness, noise) and the hands
        that go in front of it. frame = the image the hands were found in; scale = how much larger
        the image the objects are drawn into is: places, sizes, hands and blur are given at that
        scale (the renderer measures the grain on that image itself, camera_noise). Feeds the
        harmonization worker like render()."""
        layers = self.layers(scale)
        self._feed_harmony(frame, layers if scale == 1 else None)
        results = self.worker.results if self.worker else {}
        objects = []
        for layer in layers:
            tr, fit = layer.track, results.get(layer.track.id)
            objects.append({
                "id": tr.id, "name": tr.artefact.name, "w": layer.w, "h": layer.h,
                "x": layer.x, "y": layer.y, "mode": layer.mode, "opacity": round(tr.opacity, 3),
                "raw": not self.harmonized(tr),
                "fit": None if fit is None else [round(float(v), 5) for v in fit[0].ravel()] + [
                    round(float(fit[1]) * (scale if fit[1] > 0 else 1.0), 4), round(float(fit[2]), 4)],
                "version": self.worker.version if self.worker else 0,
                "hands": [np.round(h.points * scale, 3).tolist() for h in tr.hands] if layer.mode != "palm" else [],
                "centres": [np.round(h.centre * scale, 1).tolist() for h in tr.hands],
                "n_hands": len(tr.hands), "grip": tr.grip, "busy": tr.busy, "height_m": tr.artefact.height_m,
                "size_m": tr.artefact.size_m, "shadow": _shadowed(layer)})
        return objects

    def render(self, frame: np.ndarray, harmonize: bool = True, small: np.ndarray | None = None,
               noise: float | None = None) -> np.ndarray:
        """The objects drawn into frame. small = the smaller copy of it the hands were found in, if
        any: the objects are then placed and sized frame / small times larger, and harmonized on the
        copy. noise = the grain to give them (default: the frame's own, camera_noise)."""
        from common.harmonization import camera_noise

        small = frame if small is None else small
        scale = frame.shape[0] / small.shape[0]
        layers = self.layers(scale)
        self._feed_harmony(small, layers if scale == 1 else None)
        results = self.worker.results if harmonize and self.worker else {}
        if noise is None and results and self._noise_version != self.worker.version:
            self._noise, self._noise_version = camera_noise(frame), self.worker.version  # once per update
        noise = self._noise if noise is None else noise
        soft = max(1.0, frame.shape[0] / 480)  # the hands' soft edge, as at 480 rows
        alive = {tr.id for tr in self.tracks}
        for cache in (self._sprites, self._styled_cache):  # forget the objects that are gone
            for tid in [tid for tid in cache if tid not in alive]:
                del cache[tid]
        for name in [name for name in self._work if isinstance(name, tuple) and name[0] not in alive]:
            del self._work[name]
        out, (fh, fw) = frame.copy(), frame.shape[:2]
        for layer in layers:
            op = layer.track.opacity
            fit = results.get(layer.track.id) if self.harmonized(layer.track) else None
            shadow = self._contact_shadow(layer) if _shadowed(layer) else None
            boxes = [(layer.x, layer.y, layer.w, layer.h)] + ([] if shadow is None else [(shadow[1], shadow[2], *shadow[0].shape[1::-1])])
            x1, y1 = max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes))
            x2, y2 = min(fw, max(b[0] + b[2] for b in boxes)), min(fh, max(b[1] + b[3] for b in boxes))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = out[y1:y2, x1:x2]  # drawn into in place
            if shadow is not None:
                _blend(roi, None, shadow[0] * op, shadow[1] - x1, shadow[2] - y1)
            vx1, vy1, vx2, vy2 = max(layer.x, 0), max(layer.y, 0), min(layer.x + layer.w, fw), min(layer.y + layer.h, fh)
            if vx2 > vx1 and vy2 > vy1:  # the part of it in the frame
                drawn, weight, rest = self._sprite(layer, (vx1, vy1, vx2, vy2),
                                                   None if fit is None else (fit[0], fit[1] * scale if fit[1] > 0 else fit[1]))
                if op < 1:
                    weight = weight * op
                    rest = 1 - weight
                if fit is not None and noise > 0:  # fresh sensor noise every frame, like the camera's
                    drawn = cv2.addWeighted(drawn, 1.0, self._grain_window(vy2 - vy1, vx2 - vx1), noise / 32, -4.0 * noise,
                                            dst=self._buffer("grained", (vy2 - vy1, vx2 - vx1, 3)))
                _blend(roi, drawn, weight, vx1 - x1, vy1 - y1, rest)
            if layer.mode != "palm":  # held at its belly or handles: the hands are in front of it
                for hand in layer.track.hands:
                    _occlude(roi, frame[y1:y2, x1:x2], hand, (x1, y1), scale, soft)
        return out

    def _contact_shadow(self, layer: Layer, strength: float = 0.45) -> tuple[np.ndarray, int, int]:
        """Blurred ellipse under the object's base: (alpha canvas, x, y) in frame coordinates."""
        sprite = layer.track.artefact.sprite
        if layer.track.artefact.name not in self._spans:  # the base's width in the cut-out
            base = np.nonzero(sprite[-max(2, sprite.shape[0] // 20):, :, 3].max(0) > 127)[0]
            self._spans[layer.track.artefact.name] = float(np.ptp(base)) if len(base) else sprite.shape[1] / 2
        rx = max(3, int(np.floor(0.6 * self._spans[layer.track.artefact.name] * layer.w / sprite.shape[1] + 0.5)))
        if rx not in self._shadows:
            if len(self._shadows) > 64:
                self._shadows.clear()
            canvas = np.zeros((4 * max(2, rx // 3), 3 * rx), np.float32)
            cv2.ellipse(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2), (rx, max(2, rx // 4)), 0, 0, 360, 1.0, -1)
            self._shadows[rx] = cv2.GaussianBlur(canvas, (0, 0), max(1.0, rx / 4)) * strength
        canvas = self._shadows[rx]
        return canvas, layer.x + layer.w // 2 - canvas.shape[1] // 2, layer.y + layer.h - canvas.shape[0] // 2


# ── Harmonization off the render loop ─────────────────────────────────────

def keep_colours(raw: np.ndarray, harmonized: np.ndarray, colour_share: float = COLOUR_SHARE) -> np.ndarray:
    """The harmonized image with the networks' lightness change (exposure, contrast) kept whole but
    only colour_share of their colour change (Lab a, b). The networks are trained to repair objects
    whose colours are wrong, so they pull an object's colours towards the scene's; a museum photo's
    colours are right - only the light differs (live_harmony_eval.py measures the trade-off)."""
    if colour_share >= 1:
        return harmonized
    before, after = (cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_BGR2Lab) for im in (raw, harmonized))
    after[..., 1:] = before[..., 1:] + colour_share * (after[..., 1:] - before[..., 1:])
    return np.clip(cv2.cvtColor(after, cv2.COLOR_Lab2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)


def fit_colour_matrix(raw: np.ndarray, target: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """4x3 matrix M with [rgb, 1] @ M ~ target on the `inside` pixels (0-1 units), by ridge
    regression towards the identity."""
    n = int(inside.sum())
    x = np.c_[raw[inside] / 255.0, np.ones(n)]
    lam = 0.02 * n
    return np.linalg.solve(x.T @ x + lam * np.eye(4), x.T @ (target[inside] / 255.0) + lam * IDENTITY)


class HarmonyWorker:
    """For each object keeps (matrix, t, noise): the 4x3 colour matrix fitted to the harmonization
    method's output with colour_share of its colour change (keep_colours; rgb' = [rgb, 1] @ matrix,
    by ridge regression towards the identity, fit_colour_matrix), and the
    camera match of harmonization.camera_params. Live, the work runs in a separate process - its
    own interpreter, so it never holds up the render loop - on the newest composite whenever the
    previous one is done, one ensemble network per update (rotate); sync=True runs it in the
    caller with the whole ensemble every time (stills, video files: reproducible)."""

    def __init__(self, method: str = "ensemble", sync: bool = False, max_width: int = 960, rotate: bool = False,
                 live_width: int = 512, colour_share: float = COLOUR_SHARE):
        self.method, self.sync, self.max_width, self.rotate, self.version, self._turn = method, sync, max_width, rotate, 0, 0
        self.colour_share = colour_share
        self.ready = sync  # live: True once the harmonization process has loaded the networks
        self._results: dict[int, tuple[np.ndarray, float, float]] = {}
        self._fits: dict[int, dict[str, np.ndarray]] = {}  # per object: the last matrix fitted to each network
        self._updated: dict[int, int] = {}                 # per object: the update that last saw it
        self._turns = 0                                    # updates done
        if not sync:
            import multiprocessing
            import os

            ctx = multiprocessing.get_context("spawn")
            self._jobs, self._done, self._busy = ctx.Queue(), ctx.Queue(), True  # busy until the networks are loaded
            threads = {k: os.environ.get(k) for k in THREAD_VARS}
            os.environ.update(dict.fromkeys(THREAD_VARS, "2"))  # read by the child's numpy / torch when they load
            try:
                self._process = ctx.Process(target=_serve, args=(method, live_width, colour_share, self._jobs, self._done),
                                            daemon=True)
                with _stdin_hidden():
                    self._process.start()
            finally:
                for k, v in threads.items():
                    os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)

    def _collect(self) -> None:
        """Take what the harmonization process has finished."""
        while not self.sync:
            try:
                self._results, self._busy, self.ready = self._done.get_nowait(), False, True
                self.version += 1
            except queue.Empty:
                return

    @property
    def results(self) -> dict[int, tuple[np.ndarray, float, float]]:
        self._collect()
        return self._results

    def submit(self, frame: np.ndarray, layers: list[Layer]) -> None:
        if not layers:
            return
        objects = [(layer.track.id, layer.bgra, layer.x, layer.y,
                    [hand.points for hand in layer.track.hands]) for layer in layers]
        if self.sync:
            self._run((frame, objects))
            self.version += 1
            return
        self._collect()
        if not self._busy:  # composites arriving while one is in progress are dropped
            self._jobs.put((frame, objects))
            self._busy = True

    def close(self) -> None:
        """Stop the harmonization process: asked first, ended if it has not stopped within a second."""
        if not self.sync:
            self._jobs.cancel_join_thread()  # never block the program's exit on a composite still in the pipe
            self._done.cancel_join_thread()
            self._jobs.put(None)
            self._process.join(1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(1.0)

    def _run(self, job) -> None:
        from common.harmonization import camera_params, harmonize

        frame, objects = job
        s = min(1.0, self.max_width / frame.shape[1])
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
        raw, alphas, hands = small.astype(np.float32), {}, {}
        for tid, bgra, x, y, points in objects:
            if s < 1:
                bgra, x, y = _resize_bgra(bgra, max(2, round(bgra.shape[0] * s))), round(x * s), round(y * s)
            a = bgra[..., 3:] / 255.0
            alphas[tid] = np.zeros(small.shape[:2] + (1,), np.float32)
            _paste(alphas[tid], np.ones_like(a), a, x, y)
            _paste(raw, bgra[..., :3], a, x, y)
            hands[tid] = np.zeros(small.shape[:2], np.uint8)
            for pts in points:  # the holding hands: at the object's depth, so in the same focus
                hands[tid] |= skin_mask(pts * s, small.shape[:2])
        union = np.clip(sum(alphas.values())[..., 0] * 255 + 0.5, 0, 255).astype(np.uint8)
        raw8 = np.clip(raw + 0.5, 0, 255).astype(np.uint8)
        method = self.method
        if self.rotate and method == "ensemble":
            method, self._turn = ENSEMBLE[self._turn % len(ENSEMBLE)], self._turn + 1
        out = keep_colours(raw8, harmonize(raw8, union, method), self.colour_share)
        results = {}
        for tid, alpha in alphas.items():
            inside = cv2.erode((alpha[..., 0] > 0.9).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if inside.sum() < 30:
                continue
            fits = self._fits.setdefault(tid, {})
            fits[method] = fit_colour_matrix(raw8, out, inside)
            matrix = np.mean(list(fits.values()), axis=0)  # with rotate: the ensemble of the last three updates
            ys, xs = np.nonzero(inside | (hands[tid] > 0))
            m = int(0.3 * max(np.ptp(ys), np.ptp(xs))) + 4  # the object, its hands and the ring around them
            box = (slice(max(0, ys.min() - m), ys.max() + m + 1), slice(max(0, xs.min() - m), xs.max() + m + 1))
            reference = (hands[tid][box] > 0) & (alpha[box][..., 0] < 0.02)  # the visible part of the hands
            t, noise = camera_params(out[box], (alpha[box][..., 0] * 255).astype(np.uint8), reference)
            t = min(t / s, MAX_BLUR * frame.shape[0] / 480) if t > 0 else t  # smooth skin reads blurrier than it is: cap
            prev = self._results.get(tid)
            camera = (t, noise / s) if prev is None else (0.5 * (prev[1] + t), 0.5 * (prev[2] + noise / s))
            results[tid] = (matrix.astype(np.float32), *camera)
        self._turns += 1
        self._updated.update(dict.fromkeys(results, self._turns))
        keep = {tid for tid, turn in self._updated.items() if self._turns - turn < 20}  # ~10 s: objects briefly out of view
        self._results = {tid: fit for tid, fit in {**self._results, **results}.items() if tid in keep}
        self._fits = {tid: fits for tid, fits in self._fits.items() if tid in keep}
        self._updated = {tid: turn for tid, turn in self._updated.items() if tid in keep}


@contextlib.contextmanager
def _stdin_hidden():
    """Start the harmonization process with NUL as its standard input. Windows gives a new process
    a duplicate of ours, which multiprocessing closes in the child; when our stdin is a pipe opened
    for overlapped I/O - as QProcess makes them, so pose_server.py under the C++ app - that close
    wakes a blocking read of ours with 0 bytes: a false end of input that would end the server."""
    import ctypes

    if not hasattr(ctypes, "WinDLL"):  # not Windows
        yield
        return
    kernel32 = ctypes.WinDLL("kernel32")  # our own function objects: their restype is ours to set
    kernel32.GetStdHandle.restype = kernel32.CreateFileW.restype = ctypes.c_void_p
    saved = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    nul = kernel32.CreateFileW("NUL", 0x80000000, 3, None, 3, 0, None)  # GENERIC_READ, shared, OPEN_EXISTING
    kernel32.SetStdHandle(-10, ctypes.c_void_p(nul))
    try:
        yield
    finally:
        kernel32.SetStdHandle(-10, ctypes.c_void_p(saved))
        kernel32.CloseHandle(ctypes.c_void_p(nul))


def _serve(method: str, max_width: int, colour_share: float, jobs, done) -> None:
    """The harmonization process: load the networks, then answer every composite with its fits.
    It keeps out of the render loop's way - below-normal priority and few CPU threads (the networks
    themselves run on the GPU) - and ends by itself within a second if the program disappears."""
    import multiprocessing
    import os
    import sys

    import torch

    from common.harmonization import harmonize

    try:  # the parent's stdout may be a protocol pipe (pose_server.py <-> the C++ app): keep it clean
        os.dup2(2, 1)
    except OSError:
        pass
    sys.stdout = sys.stderr

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    try:
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    except AttributeError:  # not Windows
        import os

        os.nice(5)
    worker = HarmonyWorker(method, sync=True, max_width=max_width, rotate=True, colour_share=colour_share)
    dummy = np.zeros((64, 64), np.uint8)
    dummy[16:48, 16:48] = 255
    harmonize(np.full((64, 64, 3), 128, np.uint8), dummy, method)  # load the networks before they are needed
    done.put({})
    parent = multiprocessing.parent_process()
    while True:
        try:
            job = jobs.get(timeout=1.0)
        except queue.Empty:  # the program was ended without closing us (e.g. End Task): do not linger
            if parent is not None and not parent.is_alive():
                return
            continue
        if job is None:
            return
        try:
            worker._run(job)
        except Exception:
            traceback.print_exc()
        done.put(worker._results)
