"""
Arm Lab - hand-controlled robot arm on wheels  v10  (Python 3.14)
------------------------------------------------------------------
Install:  py -3.14 -m pip install mediapipe opencv-python numpy pillow
Run:      py -3.14 hand_arm.py

The arm sits on a round wheeled base. One hand, palm facing the camera, in two modes:

ARM MODE (start)
  MOVE your hand       -> steer the gripper (left / right = swing, up / down = far / near)
  PINCH (or FIST)      -> the arm goes down, grabs the cube, lifts it   (automatic)
                          for a fist, tuck your thumb over your fingers
  OPEN your hand       -> the arm goes down, puts the cube down, lifts  (automatic)

DRIVE MODE  (rock sign = index + little finger up, or D)
  Your hand is a joystick: up = forward, down = back, left / right = turn, middle = stop.
  You can switch straight from a pinch: the cube stays in the gripper while you drive.
  Back in arm mode with a cube: pinch, then open over the spot to put it down.

Gestures - hold the pose still for about a second (a ring fills up):
  ROCK SIGN            -> drive mode / arm mode
  THUMBS UP            -> new round
  PEACE SIGN           -> auto mode on / off (the robot drives and plays by itself)
  POINT (index up)     -> pause / resume

Game modes (G):  Tray   put all 3 cubes in the tray
                 Tower  stack all 3 cubes into one tower in the tray
                 Sort   put each cube in the tray of its own colour
                 Free   no clock, just play
The cubes and the trays are far apart: you have to drive. The MAP (left) shows the whole floor.
The clock starts at your first grab. Best times are saved in arm_lab_stats.json
(rounds where auto mode, replay or the tutorial helped are not counted).

Keys:  D drive / arm mode    G mode    A auto mode    T teach (record)    Y replay
       Space pause    R new round    U tutorial    H help    N sound    V auto view
       K switch camera    P phone camera    M mirror    O rotate    Q quit

Phone as camera (same Wi-Fi as the laptop):
  Android: install "IP Webcam", tap "Start server", then press P here and type the
           address it shows, e.g. 192.168.1.5:8080
  iPhone / Android: DroidCam -> press P and type e.g. 192.168.1.5:4747
  Apps that install a PC driver (DroidCam client, iVCam, Camo, Windows Phone Link)
  show up as a normal camera -> just press K.
  Or start with:  py -3.14 hand_arm.py --phone 192.168.1.5:8080
"""
import argparse
import json
import math
import os
import tempfile
import threading
import time
import urllib.request
import wave

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # quiet camera-probe warnings
except AttributeError:
    pass

# =====================================================================
# 1. SETTINGS (tune these)
# =====================================================================
CAM_INDEX = 0                  # camera to start with (K switches while running)
PHONE_ADDRESS = ""             # e.g. "192.168.1.5:8080" (IP Webcam app) or "192.168.1.5:4747" (DroidCam). P sets it.
MIRROR = True                  # mirror the picture (M toggles). Keep on when the camera faces you.
ROTATE = 0                     # 0 / 90 / 180 / 270 degrees (O cycles) - for a phone held upright
CAM_W, CAM_H = 640, 480        # requested size for USB / laptop cameras
MAX_PROC_W = 960               # bigger phone frames are shrunk to this width before hand tracking
CONFIG_FILE = "camera_settings.json"   # remembers your camera, mirror, rotation, mode, sound, tutorial
STATS_FILE = "arm_lab_stats.json"      # best times
WIN_W, WIN_H = 1280, 720
AUTO_BRIGHTEN = True           # boosts dark webcam images before detection

# where your hand works in the camera picture (0..1)
BOX_X, BOX_Y = (0.22, 0.78), (0.18, 0.82)
SWING_RANGE = 1.30             # rad the arm swings each way (hand at the box edge)
REACH_RANGE = (0.57, 0.20)     # m: hand at top of box -> far, bottom -> near

HOVER_Z = 0.20                 # m: travel height of the gripper (cubes are 5 cm)
AIM_ASSIST = True
ASSIST_RADIUS = 0.07           # m: snap starts inside this distance
ASSIST_STRENGTH = 1.0          # 1 = fully centred when very close, 0 = off

AUTO_VIEW = True               # camera follows behind the arm (V toggles)
VIEW_DEADZONE = 0.10           # rad the arm may swing before the camera starts turning
VIEW_SPEED = 3.0               # how fast the camera catches up (1/s)
VIEW_SIDE = 0.40               # rad the camera sits to the side, so the arm never hides the cube below it

# grab = PINCH thumb+index OR make a FIST (thumb tucked).  Open your hand to let go.
# The pinch self-calibrates: 0 = your tightest pinch, 1 = your relaxed open hand.
GRAB_CLOSE, GRAB_OPEN = 0.30, 0.55     # pinch levels to grab / let go (gap between = no flicker)
ALLOW_FIST = True
FIST_CLOSE, FIST_OPEN = 1.35, 1.60     # fingertips-to-wrist / palm size (open hand ~2.0, fist ~1.1)
PINCH_AIM = 0.60               # a pinch aims the thumb at the INDEX tip: thumb-index / thumb-middle below this
GRAB_FRAMES = 2                # frames a grab must hold before it counts (fast)
RELEASE_TIME = 0.30            # s an OPEN hand must be held before letting go (never drops by accident)
REWIND = 0.25                  # s: grab where your hand was just before the pinch (pinching moves the hand)
GRAB_RADIUS = 0.05             # m: how close (sideways) a cube must be to be picked
MIN_HAND_SIZE = 0.035          # ignore "hands" smaller than this (ghosts, far away)

# gestures (thumbs up / peace / point): hold still for GESTURE_HOLD seconds
GESTURES = True
GESTURE_HOLD = 0.8             # s the pose must be held (a bar fills up)
GESTURE_COOLDOWN = 1.2         # s before the next gesture can start
EXTENDED, CURLED = 1.50, 1.25  # finger straightness: tip-to-wrist / knuckle-to-wrist (straight ~1.9, curled ~0.9)
THUMB_TUCK = 0.55              # thumb tip closer than this (x palm size) to the fingers = tucked in (fist)
THUMB_LIFT = 0.30              # thumbs up: thumb tip this much (x palm size) above every finger

# game
START_MODE = "tray"            # tray | tower | sort | free  (G cycles, remembered)
SOUND = True                   # little beeps on Windows (N toggles)
AUTO_SPEED = 0.45              # m/s the robot steers its arm in auto mode

# wheeled base: rock sign (index + little finger up) or D switches between DRIVE and ARM mode
ARENA = 1.8                    # m: the floor is 3.6 x 3.6 m, walls at +-ARENA
BASE_R = 0.14                  # m: radius of the round wheeled base
DRIVE_SPEED, TURN_SPEED = 0.45, 1.4   # m/s forward and rad/s turning, hand at the edge of the box
JOY_DEADZONE = 0.18            # middle of the hand box where the robot stands still (0..1)
CARRY_XY = (0.30, 0.0)         # m: where the arm holds the gripper while driving (in front of the base)

# smoothing
MIN_CUTOFF, BETA = 0.9, 0.6    # One Euro: lower MIN_CUTOFF = smoother, higher BETA = less lag
DEADBAND = 0.003               # m ignored -> no drifting
JOINT_RESPONSE = 10.0          # how fast joints follow (1/s)
MAX_JOINT_SPEED = 3.0          # rad/s cap, like servo motors
JUMP_LIMIT = 0.20              # ignore a hand that teleports more than this in one frame

# =====================================================================
# 2. ROBOT LINKAGE  (5 joints + gripper)
#    J1 base yaw | J2 shoulder | J3 elbow | J4 wrist pitch | J5 wrist roll
# =====================================================================
BASE_H, L1, L2, L3 = 0.12, 0.32, 0.28, 0.07   # base, upper arm, forearm, wrist (m)
FINGER = 0.05
GRASP_DROP = L3 + FINGER * 0.7
CUBE = 0.025
JOINT_LIMITS = np.array([[-2.6, 2.6], [-0.3, 3.0], [-2.8, 0.0], [-4.8, 1.8], [-1.6, 1.6]])
MIN_R, MAX_R = 0.19, 0.57       # reachable ring (grasp point): outside the wheeled base, floor always reachable


def clamp_workspace(t):
    x, y, z = t
    r = math.hypot(x, y)
    rc = min(max(r, MIN_R), MAX_R)
    if r > 1e-6:
        x, y = x * rc / r, y * rc / r
    return np.array([x, y, min(max(z, CUBE - 0.002), 0.40)])


def solve_ik(x, y, z, roll=0.0):
    """Grasp point -> 5 joint angles (elbow up, gripper always pointing straight down)."""
    yaw = math.atan2(y, x)
    r = math.hypot(x, y)
    wz = z + GRASP_DROP - BASE_H
    d = min(max(math.hypot(r, wz), 0.10), L1 + L2 - 1e-3)
    aim = math.atan2(wz, r)
    cos_e = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    elbow = -math.acos(max(-1.0, min(1.0, cos_e)))
    shoulder = aim - math.atan2(L2 * math.sin(elbow), L1 + L2 * math.cos(elbow))
    wrist = -math.pi / 2 - (shoulder + elbow)
    return np.clip(np.array([yaw, shoulder, elbow, wrist, roll]), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def forward_kinematics(q):
    yaw, s, e, w, roll = q
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    pt = lambda r, z: r * fwd + z * up
    shoulder = np.array([0, 0, BASE_H])
    elbow = shoulder + pt(L1 * math.cos(s), L1 * math.sin(s))
    wrist = elbow + pt(L2 * math.cos(s + e), L2 * math.sin(s + e))
    a = s + e + w
    palm = wrist + pt(L3 * math.cos(a), L3 * math.sin(a))
    side = np.array([-math.sin(yaw + roll), math.cos(yaw + roll), 0.0])
    tool = pt(math.cos(a), math.sin(a))
    return dict(base=np.zeros(3), shoulder=shoulder, elbow=elbow, wrist=wrist,
                palm=palm, side=side, tool=tool, grasp=palm + tool * FINGER * 0.7)


# =====================================================================
# 3. SMOOTHING FILTER (One Euro: smooth when still, fast when moving)
# =====================================================================
class OneEuro:
    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None or t - self.t > 1.0:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.dx = self.dx + self._alpha(self.dc, dt) * ((x - self.x) / dt - self.dx)
        self.x = self.x + self._alpha(self.mc + self.beta * np.abs(self.dx), dt) * (x - self.x)
        self.t = t
        return self.x


# =====================================================================
# 4. ONE-HAND CONTROL  (sensing -> steering point, grab, gestures)
# =====================================================================
def palm_size(p):
    return (np.linalg.norm(p[0] - p[5]) + np.linalg.norm(p[0] - p[17])
            + np.linalg.norm(p[5] - p[17])) / 3


def finger_ratios(w):
    """How straight each finger is (index, middle, ring, pinky): tip-to-wrist / knuckle-to-wrist.
    Straight ~1.9, relaxed ~1.7, curled into the palm ~0.9. Uses the 3D hand, so any hand angle works."""
    return [np.linalg.norm(w[tip] - w[0]) / (np.linalg.norm(w[mcp] - w[0]) + 1e-9)
            for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20))]


GESTURE_NAMES = {"thumbs_up": "Thumbs up", "peace": "Peace sign", "point": "Point", "rock": "Rock sign"}


def classify(ext, tuck, lift, nrm):
    """Hand pose from finger straightness, thumb position and the pinch gap.
    ext: 4 finger ratios | tuck: thumb tip to fingers / palm | lift: thumb tip height above
    the fingers / palm (picture) | nrm: pinch gap (0 = pinched, 1 = relaxed open)."""
    straight = [e > EXTENDED for e in ext]
    curled = [e < CURLED for e in ext]
    thumb_free = nrm > GRAB_OPEN                    # thumb away from the index tip = not a pinch
    if all(curled):
        if lift > THUMB_LIFT and tuck > THUMB_TUCK and thumb_free:
            return "thumbs_up"
        if tuck < THUMB_TUCK and lift < 0.10:
            return "fist"
        return "unsure"
    if thumb_free and straight[0] and straight[1] and curled[2] and curled[3]:
        return "peace"
    if thumb_free and straight[0] and curled[1] and curled[2] and straight[3]:
        return "rock"
    if thumb_free and straight[0] and curled[1] and curled[2] and curled[3]:
        return "point"
    if sum(straight) >= 3:
        return "open"
    return "unsure"


class HandControl:
    def __init__(self):
        self.filt = OneEuro()
        self.xy = np.array([0.40, 0.0])     # steering point on the table (m)
        self.closed, self.grab_count, self.release_t0 = False, 0, None
        self.joy = np.zeros(2)               # drive stick: x = turn, y = forward (-1..1)
        self.grip = 0.0                     # 0 = open hand .. 1 = fully closed (for the meter)
        self.open_lvl, self.closed_lvl = 1.0, 0.15   # learned from YOUR hand while you play
        self.pos, self.seen, self.det = None, -9.0, None
        self.recent, self.hist = [], []
        self.pose = "none"                  # open | fist | thumbs_up | peace | point | rock | unsure
        self.g_pose, self.g_t0, self.g_last, self.g_fired, self.g_progress = None, 0.0, 0.0, False, 0.0
        self.g_cool = self.no_grab_until = 0.0
        self.events = []                    # finished gestures, picked up by the game

    def visible(self, t, within=0.25):
        return t - self.seen < within

    @staticmethod
    def describe(h, fw, fh):
        px, w = h["px"], h["world"]
        n = px / [fw, fh]
        p2, p3 = palm_size(px) + 1e-9, palm_size(w) + 1e-9
        pinch = 0.5 * (np.linalg.norm(px[4] - px[8]) / p2      # picture: good when palm faces camera
                       + np.linalg.norm(w[4] - w[8]) / p3)      # 3D: good when the hand is turned
        fist = np.mean([np.linalg.norm(w[i] - w[0]) for i in (8, 12, 16, 20)]) / p3   # fingertips to wrist
        tuck = min(np.linalg.norm(w[4] - w[i]) for i in (6, 7, 10, 11)) / p3          # thumb tip on the fingers?
        lift = (px[5:, 1].min() - px[4, 1]) / p2                                      # thumb tip above all fingers?
        aim = np.linalg.norm(w[4] - w[8]) / (np.linalg.norm(w[4] - w[12]) + 1e-9)    # thumb going to the index tip?
        return dict(h, n=n, c=n[[0, 5, 9, 13, 17]].mean(axis=0),   # knuckles: still when fingers move
                    size=p2 / fw, pinch=pinch, fist=fist, ext=finger_ratios(w), tuck=tuck, lift=lift, aim=aim)

    def choose(self, dets, t):
        """Two hands / ghosts in view -> keep following the same hand."""
        dets = [d for d in dets if d["size"] >= MIN_HAND_SIZE]
        if not dets:
            return None
        if self.pos is not None and t - self.seen < 1.0:
            return min(dets, key=lambda d: np.linalg.norm(d["c"] - self.pos))
        return max(dets, key=lambda d: d["size"])        # new: the biggest (closest) hand

    def rewind(self, t):
        """Pinching / making a gesture moves the hand a little: go back to where it was just before."""
        past = [hxy for ht, hxy in self.hist if ht <= t - REWIND]
        if past:
            self.xy = past[-1].copy()

    def update(self, raw_hands, fw, fh, t):
        d = self.choose([self.describe(h, fw, fh) for h in raw_hands], t)
        if d is None:
            return
        fresh = t - self.seen < 0.5
        if fresh and self.pos is not None and np.linalg.norm(d["c"] - self.pos) > JUMP_LIMIT:
            self.pos = d["c"]                            # teleport = tracking glitch: skip once
            return
        if not fresh:
            self.recent = []
            self.g_pose, self.g_progress = None, 0.0
        self.pos, self.seen, self.det = d["c"], t, d
        self.recent = (self.recent + [[*d["c"], d["pinch"], d["fist"], *d["ext"], d["tuck"], d["lift"], d["aim"]]])[-3:]
        m = np.median(np.array(self.recent), axis=0)    # removes one-frame spikes
        u, v, pinch, fist, ext, tuck, lift, aim = m[0], m[1], m[2], m[3], m[4:8], m[8], m[9], m[10]

        # ---- pinch level, measured against YOUR own open / pinched hand ----
        if fist > FIST_OPEN and not self.closed:          # clearly open hand -> learn your relaxed gap
            self.open_lvl += 0.05 * (pinch - self.open_lvl)
        self.closed_lvl = min(self.closed_lvl + 0.0007, pinch)          # learns your tightest pinch
        self.closed_lvl = float(np.clip(self.closed_lvl, 0.05, 0.40))
        self.open_lvl = float(np.clip(self.open_lvl, self.closed_lvl + 0.25, 1.8))
        nrm = (pinch - self.closed_lvl) / (self.open_lvl - self.closed_lvl)   # 0 = pinched, 1 = open

        # ---- pose + gestures ----
        self.pose = classify(ext, tuck, lift, nrm)
        gesturing = self.gestures(t)

        # ---- steering: left/right = swing around the base, up/down = reach (frozen during a gesture) ----
        swing = np.interp(u, BOX_X, (SWING_RANGE, -SWING_RANGE))
        reach = np.interp(v, BOX_Y, REACH_RANGE)
        swing, reach = self.filt([swing, reach], t)
        if not gesturing:
            goal = reach * np.array([math.cos(swing), math.sin(swing)])
            diff = goal - self.xy
            self.xy = self.xy + np.sign(diff) * np.maximum(np.abs(diff) - DEADBAND, 0)
        self.hist = [(ht, hxy) for ht, hxy in self.hist if t - ht < 1.0] + [(t, self.xy.copy())]

        # ---- drive stick (used in drive mode): hand off-centre in the box = drive / turn ----
        stick = lambda a: math.copysign(max(abs(a) - JOY_DEADZONE, 0.0) / (1 - JOY_DEADZONE), a)
        raw = np.array([stick(np.clip((u - sum(BOX_X) / 2) / ((BOX_X[1] - BOX_X[0]) / 2), -1, 1)),
                        stick(np.clip((sum(BOX_Y) / 2 - v) / ((BOX_Y[1] - BOX_Y[0]) / 2), -1, 1))])
        self.joy = np.zeros(2) if gesturing else self.joy + (raw - self.joy) * 0.4

        # ---- grab: pinch OR fist (thumb tucked).  Never while a gesture is being made ----
        pinch_on, pinch_off = nrm < GRAB_CLOSE and aim < PINCH_AIM, nrm > GRAB_OPEN   # thumb passing by != pinch
        fist_on = ALLOW_FIST and self.pose == "fist"
        fist_off = (not ALLOW_FIST) or fist > FIST_OPEN
        if self.closed:                                   # let go only from a clearly OPEN hand
            want = not (pinch_off and fist_off and self.pose == "open")
        else:
            want = (pinch_on or fist_on) and not gesturing and t >= self.no_grab_until
        fist_grip = (np.interp(fist, (1.0, FIST_CLOSE, 2.0), (1.0, 1 - GRAB_CLOSE, 0.0))
                     if ALLOW_FIST and tuck < THUMB_TUCK else 0.0)
        self.grip = 0.0 if gesturing else float(max(np.clip(1 - nrm, 0, 1), fist_grip))

        if want == self.closed:
            self.grab_count, self.release_t0 = 0, None
            return
        if self.closed:                                   # letting go: open hand held RELEASE_TIME
            self.release_t0 = t if self.release_t0 is None else self.release_t0
            go = t - self.release_t0 >= RELEASE_TIME
        else:                                             # grabbing: fast
            self.grab_count += 1
            go = self.grab_count >= GRAB_FRAMES
        if go:
            self.closed, self.grab_count, self.release_t0 = want, 0, None
            self.rewind(t)

    def gestures(self, t):
        """Hold thumbs up / peace / point still for GESTURE_HOLD s -> one event. Returns True while gesturing."""
        # while grabbing only the rock sign counts (so you can switch to driving with a cube in the gripper)
        g = self.pose if (GESTURES and self.pose in GESTURE_NAMES
                          and (not self.closed or self.pose == "rock")) else None
        if g is not None and g == self.g_pose:
            self.g_last = t
            self.g_progress = min(1.0, (t - self.g_t0) / GESTURE_HOLD)
            if self.g_progress >= 1.0 and not self.g_fired:
                self.g_fired = True
                self.events.append(g)
                self.no_grab_until = t + 0.5
                self.g_cool = t + GESTURE_COOLDOWN
        elif g is not None and t >= self.g_cool:
            if self.g_pose is None:
                self.rewind(t)                           # forming the gesture moved the hand
            self.g_pose, self.g_t0, self.g_last, self.g_fired, self.g_progress = g, t, t, False, 0.0
        elif self.g_pose is not None and t - self.g_last > 0.25:   # left the pose (short flickers forgiven)
            self.g_pose, self.g_progress = None, 0.0
        return self.g_pose is not None or g is not None

# =====================================================================
# 5. CAMERA + HAND TRACKING (background threads)
# =====================================================================
MODEL = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(frame):
    if not AUTO_BRIGHTEN or frame.mean() > 100:
        return frame
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = _clahe.apply(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def phone_url(address):
    """'192.168.1.5:8080' -> 'http://192.168.1.5:8080/video'. Full URLs are used as they are."""
    a = address.strip()
    if not a:
        return ""
    if "://" in a:
        return a
    if ":" not in a.split("/")[0]:
        a = a.split("/")[0] + ":8080" + ("/" + a.split("/", 1)[1] if "/" in a else "")
    if "/" not in a:
        a += "/video"
    return "http://" + a


def source_name(src):
    if isinstance(src, int):
        return "Laptop camera" if src == 0 else f"Camera {src}"
    return "Phone " + src.split("://", 1)[-1].split("/")[0]


def find_local_cameras(max_index=4):
    """Laptop webcam, USB webcams and phone apps that act as a webcam (DroidCam, iVCam, Camo, Phone Link)."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if cap.isOpened() and cap.read()[0]:
            found.append(i)
        cap.release()
    return found


class Camera:
    """Reads the newest frame in a background thread. Can switch source while running,
    and reconnects by itself if a phone stream drops."""

    def __init__(self, source):
        self.frame, self.new, self.fid = None, False, 0
        self.lock = threading.Lock()
        self.source, self.want, self.cap = None, source, None
        self.status, self.running = "Connecting...", True
        threading.Thread(target=self._loop, daemon=True).start()

    def switch(self, source):
        self.want = source

    def _open(self, src):
        if self.cap is not None:
            self.cap.release()
        self.status = f"Connecting to {source_name(src)}..."
        with self.lock:
            self.frame = None
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        else:
            cap = cv2.VideoCapture(src)                  # phone stream over Wi-Fi (MJPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap, self.source = cap, src
        ok = cap.isOpened()
        self.status = "" if ok else f"Can't open {source_name(src)}"
        return ok

    def _loop(self):
        fails, retry_at = 0, 0.0
        while self.running:
            if self.want is not None:
                src, self.want = self.want, None
                fails = 0 if self._open(src) else 999
                retry_at = time.monotonic() + 2.0
                continue
            if self.cap is None or not self.cap.isOpened() or fails > 30:
                if self.source is not None and time.monotonic() > retry_at and not isinstance(self.source, int):
                    ok = self._open(self.source)
                    fails = 0 if ok else 999
                    if not ok:
                        self.status = f"{source_name(self.source)} lost - reconnecting... (same Wi-Fi? app open?)"
                    retry_at = time.monotonic() + 2.0
                time.sleep(0.05)
                continue
            ok, f = self.cap.read()
            if ok and f is not None:
                fails = 0
                with self.lock:
                    self.frame, self.new, self.fid = f, True, self.fid + 1
                if self.status:
                    self.status = ""
            else:
                fails += 1
                if fails > 30:
                    self.status = f"No picture from {source_name(self.source)}"
                    retry_at = time.monotonic() + 1.0
                time.sleep(0.01)

    def read(self):
        with self.lock:
            f, n, self.new = self.frame, self.new, False
        return f, n

    def release(self):
        self.running = False
        time.sleep(0.05)
        if self.cap is not None:
            self.cap.release()


class HandTracker:
    """Runs MediaPipe on the newest camera frame in its own thread."""

    def __init__(self, cam):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as V
        if not os.path.exists(MODEL):
            print("Downloading hand model (one time, ~8 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL)
        self.mp, self.cam = mp, cam
        self.lm = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL),
            running_mode=V.RunningMode.VIDEO, num_hands=2,      # 2 so a second hand can't steal tracking
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5))
        self.result, self.view, self.fps = None, None, 0.0
        self.mirror, self.rotate = MIRROR, ROTATE
        self.ts, self.lock, self.running = 0, threading.Lock(), True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last = time.perf_counter()
        while self.running:
            frame, new = self.cam.read()
            if frame is None or not new:
                time.sleep(0.002)
                continue
            if self.rotate:
                frame = cv2.rotate(frame, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                           270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotate])
            if frame.shape[1] > MAX_PROC_W:                 # big phone frames: shrink for speed
                k = MAX_PROC_W / frame.shape[1]
                frame = cv2.resize(frame, (MAX_PROC_W, int(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
            frame = enhance(cv2.flip(frame, 1) if self.mirror else frame)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.ts = max(self.ts + 1, int(time.monotonic() * 1000))
            res = self.lm.detect_for_video(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.ts)
            fh, fw = frame.shape[:2]
            hands = [{"px": np.array([(p.x * fw, p.y * fh) for p in lms]),
                      "world": np.array([(p.x, p.y, p.z) for p in res.hand_world_landmarks[i]])}
                     for i, lms in enumerate(res.hand_landmarks)]
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now
            with self.lock:
                self.result, self.view = (hands, now), frame

    def get(self):
        with self.lock:
            return self.result, self.view

    def stop(self):
        self.running = False


def draw_camera_overlay(frame, ctrl, t, drive=False):
    """Control box with drafting corners + your hand skeleton, drawn on the camera picture.
    In drive mode the box becomes a joystick: a still zone in the middle, push out to drive."""
    fh, fw = frame.shape[:2]
    lw = max(1, fw // 320)
    dark = (frame * 0.72).astype(np.uint8)                 # dim the picture so the overlay reads
    frame[:] = dark
    x0, x1, y0, y1 = int(BOX_X[0] * fw), int(BOX_X[1] * fw), int(BOX_Y[0] * fh), int(BOX_Y[1] * fh)
    seen = ctrl.visible(t)
    col = (C["teal"] if drive else C["orange"]) if seen else C["paper"]
    tk = max(10, fw // 30)
    for (cx, cy, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * tk, cy), col, 2 * lw, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * tk), col, 2 * lw, cv2.LINE_AA)
    if drive:                                              # joystick: still zone + cross
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        ax, ay = int((x1 - x0) / 2 * JOY_DEADZONE), int((y1 - y0) / 2 * JOY_DEADZONE)
        cv2.ellipse(frame, (mx, my), (ax, ay), 0, 0, 360, C["teal"], lw + 1, cv2.LINE_AA)
        cv2.line(frame, (x0 + tk, my), (x1 - tk, my), C["graphite_light"], lw, cv2.LINE_AA)
        cv2.line(frame, (mx, y0 + tk), (mx, y1 - tk), C["graphite_light"], lw, cv2.LINE_AA)
        if seen and ctrl.det is not None:
            c = (ctrl.det["c"] * [fw, fh]).astype(int)
            cv2.line(frame, (mx, my), tuple(c), C["teal"], 2 * lw, cv2.LINE_AA)
    if seen and ctrl.det is not None:
        pts = ctrl.det["px"].astype(int)
        for a, b in HAND_LINKS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), C["paper"], lw + 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 2 * lw + 1, C["orange"], -1, cv2.LINE_AA)
        if not drive:                                      # grip ring between thumb and index
            gc = C["teal"] if ctrl.closed else C["orange"]
            m = tuple(((pts[4] + pts[8]) / 2).astype(int))
            rr = max(12, fw // 40)
            cv2.circle(frame, m, rr, C["graphite_light"], 2 * lw, cv2.LINE_AA)
            cv2.ellipse(frame, m, (rr, rr), -90, 0, int(360 * ctrl.grip), gc, 3 * lw, cv2.LINE_AA)
        c = (ctrl.det["c"] * [fw, fh]).astype(int)
        cv2.circle(frame, tuple(c), 6 * lw, C["ink_deep"], -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(c), 6 * lw, C["teal"] if drive else C["orange"], 2 * lw, cv2.LINE_AA)


# =====================================================================
# 6. WORLD + GAME RULES: arena, cubes, trays, wheeled base, modes, auto mode, teach & replay
# =====================================================================
TRAY_C, TRAY_HALF = np.array([-0.80, 0.95]), 0.075          # the tray: far from the cubes, you have to drive
SORT_TRAYS = [np.array(p) for p in ((-1.15, -0.25), (-1.15, 0.45), (-0.55, 1.15))]   # red, yellow, blue
CUBE_AREA = ((0.55, 1.40), (-1.40, -0.35))                  # x and y range where cubes appear (m)
CUBE_START = [(0.80, -0.60), (1.10, -1.00), (1.25, -0.50)]   # fallback layout
STACK = 2 * CUBE * 0.9                                       # cubes closer than this (sideways) stack
MODES = {
    "tray": ("Tray", "Put all 3 cubes in the tray."),
    "tower": ("Tower", "Stack all 3 cubes into one tower in the tray."),
    "sort": ("Sort", "Put each cube in the tray of its own colour."),
    "free": ("Free play", "No clock. Stack, sort, build whatever you like."),
}
MODE_ORDER = list(MODES)
COLOR_NAMES = ["Red", "Yellow", "Blue"]


def near(a, b, d=STACK):
    """Sideways overlap test (square footprint, like the cubes)."""
    return np.max(np.abs(np.asarray(a)[:2] - np.asarray(b)[:2])) < d


class World:
    def __init__(self, mode=START_MODE, seed=None):
        self.rng = np.random.default_rng(seed)
        self.layout = 0                     # changes when the trays change (redraws the floor)
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode if mode in MODES else "tray"
        self.trays = [t.copy() for t in SORT_TRAYS] if self.mode == "sort" else [TRAY_C.copy()]
        self.layout += 1
        self.reset()

    def reset(self):
        """New round: three cubes at random spots on the right-hand side of the table."""
        self.cubes = [np.array([x, y, CUBE]) for x, y in self.spawn()]
        self.vz = [0.0] * len(self.cubes)
        self.held = None

    def spawn(self, n=3):
        """Cubes at random spots in the cube area, far enough apart for the base to drive between."""
        pts = []
        for _ in range(3000):
            p = np.array([self.rng.uniform(*CUBE_AREA[0]), self.rng.uniform(*CUBE_AREA[1])])
            if all(np.linalg.norm(p - o) > 0.24 for o in pts):
                pts.append(p)
                if len(pts) == n:
                    return pts
        return [np.array(p) for p in CUBE_START]

    def snapshot(self):
        return dict(mode=self.mode, cubes=[c.copy() for c in self.cubes], vz=list(self.vz), held=self.held)

    def restore(self, s):
        self.mode = s["mode"]
        self.trays = [t.copy() for t in SORT_TRAYS] if self.mode == "sort" else [TRAY_C.copy()]
        self.layout += 1
        self.cubes, self.vz, self.held = [c.copy() for c in s["cubes"]], list(s["vz"]), s["held"]

    # ---- physics ----
    def free(self):
        return [i for i in range(len(self.cubes)) if i != self.held]

    def top_cube_near(self, xy, radius=GRAB_RADIUS):
        """The highest free cube whose centre is within `radius` sideways."""
        best = None
        for i in self.free():
            if np.linalg.norm(self.cubes[i][:2] - xy) < radius and \
                    (best is None or self.cubes[i][2] > self.cubes[best][2]):
                best = i
        return best

    def surface_under(self, xy):
        """Height of whatever a cube would land on at xy (floor or a stack top)."""
        h = 0.0
        for i in self.free():
            if near(self.cubes[i], xy):
                h = max(h, self.cubes[i][2] + CUBE)
        return h

    def try_grab(self, grasp):
        i = self.top_cube_near(grasp[:2])
        if i is not None and abs(self.cubes[i][2] - grasp[2]) < 0.03:
            self.held = i
        return self.held

    def release(self):
        self.held = None

    def support_height(self, i):
        c, h = self.cubes[i], 0.0
        for j, o in enumerate(self.cubes):
            if j != i and j != self.held and near(o, c) and o[2] < c[2]:
                h = max(h, o[2] + CUBE)
        return h

    def step(self, grasp, dt):
        for i, c in enumerate(self.cubes):
            if i == self.held:
                c[:] = grasp
                self.vz[i] = 0.0
                continue
            floor = self.support_height(i) + CUBE
            if c[2] > floor + 1e-4:
                self.vz[i] -= 9.8 * dt
                c[2] = max(floor, c[2] + self.vz[i] * dt)
            else:
                c[2], self.vz[i] = floor, 0.0

    # ---- rules ----
    def resting(self, i):
        return i != self.held and self.vz[i] == 0.0 and \
            abs(self.cubes[i][2] - self.support_height(i) - CUBE) < 1e-3

    def level(self, i):
        return int(round((self.cubes[i][2] - CUBE) / (2 * CUBE)))

    def tray_of(self, i):
        if not self.resting(i):
            return None
        for k, c in enumerate(self.trays):
            if near(self.cubes[i], c, TRAY_HALF):
                return k
        return None

    def column(self, i):
        """Cube i plus everything stacked on / under it, bottom first."""
        col = [j for j in self.free() if near(self.cubes[j], self.cubes[i])]
        return sorted(col, key=lambda j: self.cubes[j][2])

    def column_top(self, i):
        return self.column(i)[-1] if i != self.held else i

    def tower(self):
        """The tallest stack standing on the floor of a tray (list, bottom first)."""
        best = []
        for b in self.free():
            if self.level(b) == 0 and self.tray_of(b) is not None:
                col = [j for j in self.column(b) if self.resting(j)]
                if len(col) > len(best):
                    best = col
        return best

    def done(self, i):
        if not self.resting(i):
            return False
        if self.mode == "sort":
            return self.tray_of(i) == i and self.level(i) == 0
        if self.mode == "tower":
            return i in self.tower()
        return self.tray_of(i) is not None

    def score(self):
        return sum(1 for i in range(len(self.cubes)) if self.done(i))

    def complete(self):
        return self.mode != "free" and self.held is None and self.score() == len(self.cubes)

    def target_for(self, i):
        """Where cube i should go (used by auto mode)."""
        others = [j for j in self.free() if j != i]
        if self.mode == "sort":
            c = self.trays[i]
            for s in [c] + [c + 0.05 * np.array(d) for d in ((1, 1), (-1, 1), (1, -1), (-1, -1))]:
                if all(not near(self.cubes[j], s, 0.048) for j in others):
                    return s.copy()
            return c.copy()
        col = [j for j in self.tower() if j != i]
        return self.cubes[col[0]][:2].copy() if col else self.trays[0].copy()


def aim_assist(xy, world):
    """Pull the steering point onto the nearest useful spot when it is close."""
    if not AIM_ASSIST:
        return xy, None
    spots = [world.cubes[i][:2] for i in world.free()]
    if world.held is not None:
        spots = list(world.trays) + spots                # a tray, or a cube to stack on
    best, bd = None, ASSIST_RADIUS
    for s in spots:
        d = np.linalg.norm(s - xy)
        if d < bd:
            best, bd = s, d
    if best is None:
        return xy, None
    w = ASSIST_STRENGTH * (1 - bd / ASSIST_RADIUS)
    return xy + (best - xy) * w, best


class AutoGrip:
    """Claw-machine helper: pinch = down, grab, up.  Open = down, release, up.
    Uses the game clock t, so pausing never triggers its safety timeout."""

    def __init__(self):
        self.phase = "travel"          # travel | down_pick | down_place | up
        self.lock_xy = None
        self.t0 = 0.0
        self.events = []               # (kind, cube) for sounds, messages and the round clock

    def set(self, phase, t, xy=None, event=None):
        self.phase, self.t0 = phase, t
        if xy is not None:
            self.lock_xy = np.array(xy, float).copy()
        if event:
            self.events.append(event)

    def on_pinch_change(self, closed, xy, world, t):
        if closed:
            if self.phase == "down_place":             # pinched again while placing -> keep it
                self.set("up", t, event=("kept", world.held))
            elif world.held is None and self.phase in ("travel", "up"):
                self.set("down_pick", t, xy, ("pick_start", None))
        else:
            if self.phase == "down_pick":              # opened before reaching it -> cancel
                self.set("up", t)
            elif world.held is not None and self.phase in ("travel", "up"):
                self.set("down_place", t, xy, ("place_start", world.held))

    def target(self, xy, world):
        if self.phase in ("down_pick", "down_place"):
            xy = self.lock_xy
        if self.phase == "down_pick":
            i = world.top_cube_near(xy)
            z = world.cubes[i][2] if i is not None else CUBE
        elif self.phase == "down_place":
            z = world.surface_under(xy) + CUBE + 0.002
        else:
            z = HOVER_Z
        return np.array([xy[0], xy[1], z])

    def step(self, grasp, goal, world, t):
        """Advance when the gripper has actually arrived."""
        timeout = t - self.t0 > 2.5                    # safety: never get stuck
        arrived = np.linalg.norm(grasp - goal) < 0.006 or timeout
        if self.phase == "down_pick" and arrived:
            got = world.try_grab(grasp)
            self.set("up", t, event=("picked", got) if got is not None else ("missed", None))
        elif self.phase == "down_place" and arrived:
            i = world.held
            world.release()
            self.set("up", t, event=("placed", i))
        elif self.phase == "up" and (grasp[2] > HOVER_Z - 0.02 or timeout):
            self.set("travel", t)

    def gripper_closed(self, world, closed):
        if world.held is not None:
            return True
        return closed and self.phase != "down_pick"


def wrap_angle(a):
    return math.remainder(a, 2 * math.pi)


class Base:
    """The round wheeled base (differential drive: two side wheels). Pose = position + heading."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pos, self.th, self.v, self.w, self.blocked = np.zeros(2), 0.0, 0.0, 0.0, False

    def rot(self):
        return np.array([[math.cos(self.th), -math.sin(self.th)], [math.sin(self.th), math.cos(self.th)]])

    def to_world(self, p):
        return self.pos + self.rot() @ np.asarray(p, float)[:2]

    def to_local(self, p):
        return self.rot().T @ (np.asarray(p, float)[:2] - self.pos)

    def to_world3(self, p):
        return np.array([*self.to_world(p[:2]), p[2]])

    def to_local3(self, p):
        return np.array([*self.to_local(p[:2]), p[2]])

    def stop(self):
        self.v = self.w = 0.0

    def drive(self, v_cmd, w_cmd, dt, world):
        """Smooth like real motors, stopped by the walls and by cubes on the floor."""
        self.v += float(np.clip(v_cmd - self.v, -1.2 * dt, 1.2 * dt))
        self.w += float(np.clip(w_cmd - self.w, -5.0 * dt, 5.0 * dt))
        th = self.th + self.w * dt
        new = self.pos + self.v * dt * np.array([math.cos(th), math.sin(th)])
        lim = ARENA - BASE_R
        self.blocked = bool(np.any(np.abs(new) > lim)) and abs(self.v) > 0.01
        new = np.clip(new, -lim, lim)
        for i in world.free():
            c = world.cubes[i][:2]
            d_new = np.linalg.norm(new - c)
            if d_new < BASE_R + CUBE * 1.4 and d_new < np.linalg.norm(self.pos - c):
                new, self.v, self.blocked = self.pos.copy(), 0.0, True
                break
        self.pos, self.th = new, th

    def snapshot(self):
        return (self.pos.copy(), self.th)

    def restore(self, s):
        self.pos, self.th = s[0].copy(), s[1]
        self.stop()


ARM_REACH = (0.22, 0.50)        # auto mode uses the arm when the target is this far from the base...
ARM_ANGLE = 1.0                 # ...and within this angle (rad) of straight ahead
PARK = 0.36                     # m: where auto mode parks the base from its target


class AutoPilot:
    """Auto mode: the robot drives and picks by itself with the same moves you use."""

    def __init__(self, xy, holding=False):
        self.xy = np.array(xy[:2], float)              # arm target (world)
        self.closed, self.drive = holding, False
        self.misses, self.idle, self.best = 0, 0, -1
        self.park, self.park_goal, self.stuck_t, self.replans = None, None, 0.0, 0

    def choose(self, world, base):
        todo = [world.column_top(i) for i in world.free() if not world.done(i)]
        if not todo:
            return None
        return min(todo, key=lambda i: np.linalg.norm(world.cubes[i][:2] - base.pos))

    @staticmethod
    def clear(world, p, r=BASE_R + 0.06):
        return all(np.linalg.norm(p - world.cubes[i][:2]) >= r for i in world.free())

    def plan_park(self, world, base, goal):
        """A spot PARK m from the goal, inside the walls, with a straight clear path to it."""
        away = base.pos - goal
        a0 = math.atan2(away[1], away[0]) if np.linalg.norm(away) > 1e-6 else base.th + math.pi
        a0 += 0.7 * self.replans                        # blocked before: come from another side
        for da in (0, .5, -.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 2.6, -2.6, math.pi):
            p = goal + PARK * np.array([math.cos(a0 + da), math.sin(a0 + da)])
            if np.max(np.abs(p)) > ARENA - BASE_R - 0.05 or not self.clear(world, p):
                continue
            if all(self.clear(world, base.pos + (p - base.pos) * k) for k in np.linspace(0.15, 1, 18)):
                return p
        return goal + PARK * np.array([math.cos(a0), math.sin(a0)])

    def step(self, world, base, grip, grasp, dt):
        """-> (forward speed, turn speed, 'done' | 'stuck' | None)"""
        if grip.phase != "travel":
            return 0.0, 0.0, None
        if world.held is None:
            if self.closed:                            # a grab missed: open first, then try again
                self.closed = False
                return 0.0, 0.0, None
            i = self.choose(world, base)
            if i is None:
                return 0.0, 0.0, "done"
            goal = world.cubes[i][:2].copy()
        else:
            goal = world.target_for(world.held)
        rel = goal - base.pos
        dist, ang = float(np.linalg.norm(rel)), wrap_angle(math.atan2(rel[1], rel[0]) - base.th)

        if ARM_REACH[0] <= dist <= ARM_REACH[1] and abs(ang) < ARM_ANGLE:      # in reach: use the arm
            self.drive, self.park, self.stuck_t = False, None, 0.0
            d = goal - self.xy
            dd, step = float(np.linalg.norm(d)), AUTO_SPEED * dt
            self.xy = goal.copy() if dd <= step else self.xy + d / dd * step
            if dd <= step and np.linalg.norm(grasp[:2] - goal) < 0.008 and grasp[2] > HOVER_Z - 0.03:
                self.closed = world.held is None       # over a cube -> grab, over the target -> let go
            return 0.0, 0.0, None

        # out of reach: tuck the arm and drive to a parking spot facing the goal
        self.drive = True
        self.xy = base.to_world(CARRY_XY)
        if self.park is None or self.park_goal is None or np.linalg.norm(self.park_goal - goal) > 0.01:
            self.park, self.park_goal = self.plan_park(world, base, goal), goal.copy()
        e = self.park - base.pos
        de = float(np.linalg.norm(e))
        if de > 0.04:
            err = wrap_angle(math.atan2(e[1], e[0]) - base.th)
            v = 0.0 if abs(err) > 0.35 else min(DRIVE_SPEED, 1.5 * de)
        else:
            err, v = ang, 0.0                          # parked: turn to face the goal
        w = float(np.clip(3.0 * err, -TURN_SPEED, TURN_SPEED))
        if v > 0.05 and base.blocked:                  # something in the way: plan another route
            self.stuck_t += dt
            if self.stuck_t > 0.8:
                self.replans, self.park, self.stuck_t = self.replans + 1, None, 0.0
                if self.replans > 8:
                    return 0.0, 0.0, "stuck"
        else:
            self.stuck_t = 0.0
        return v, w, None


class Recorder:
    """Teach & replay: records the robot's path, arm target and grabs, then plays them back
    on the same layout (the path is replayed exactly, so drives never drift)."""
    MAX_TIME = 180.0

    def __init__(self):
        self.samples, self.snap = [], None
        self.recording = self.playing = False
        self.t, self.i = 0.0, 0

    def start(self, snap):
        self.samples, self.snap, self.recording, self.playing, self.t = [], snap, True, False, 0.0

    def add(self, dt, pos, th, xy, closed, drive):
        """Returns False when the recording is full."""
        self.t += dt
        self.samples.append((self.t, np.array(pos, float), float(th), np.array(xy, float), bool(closed), bool(drive)))
        if self.t >= self.MAX_TIME:
            self.recording = False
        return self.recording

    def stop(self):
        self.recording = False

    def length(self):
        return self.samples[-1][0] if self.samples else 0.0

    def start_play(self):
        self.playing, self.t, self.i = True, 0.0, 0

    def sample(self, dt):
        """-> (base position, heading, arm target, closed, drive, finished)"""
        self.t += dt
        while self.i + 1 < len(self.samples) and self.samples[self.i + 1][0] <= self.t:
            self.i += 1
        t0, pos, th, xy, closed, drive = self.samples[self.i]
        if self.i + 1 < len(self.samples):
            t1, pos1, th1, xy1, _, _ = self.samples[self.i + 1]
            k = min(max((self.t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
            pos, th, xy = pos + (pos1 - pos) * k, th + (th1 - th) * k, xy + (xy1 - xy) * k
        return pos, th, xy, closed, drive, self.t >= self.length() + 1.5   # +1.5 s lets the last move finish


class Round:
    def __init__(self, mode, practice=False):
        self.mode, self.practice = mode, practice
        self.time, self.running, self.done = 0.0, False, False
        self.assisted, self.picks, self.result = "", 0, None    # assisted: "auto mode" / "replay" helped


TUTORIAL = [
    ("Show your hand", "Hold one hand up inside the corner marks, palm facing the camera."),
    ("Switch to drive mode", "Make a rock sign (index and little finger up) and hold it still for a second."),
    ("Drive to the cubes", "Hand up = forward, down = back, left or right = turn. Middle = stop. Drive next to a cube."),
    ("Back to arm mode", "Rock sign again. Now your hand moves the arm."),
    ("Pick up a cube", "Steer over a cube until the ring turns teal, then pinch and hold."),
    ("Put it in a tray", "Rock sign (the cube stays), drive to a tray, rock sign, then pinch and open over the tray."),
    ("Start a round", "Hold a thumbs up for a second to start a real round."),
]


class Tutorial:
    def __init__(self):
        self.active, self.step = False, 0

    def start(self):
        self.active, self.step, self.t_seen = True, 0, 0.0

    def check(self, ctrl, world, base, drive, now, dt):
        """Steps that finish by themselves. True when a step was just completed."""
        if not self.active:
            return False
        s, ok = self.step, False
        if s == 0:
            self.t_seen = self.t_seen + dt if ctrl.visible(now) else 0.0
            ok = self.t_seen > 0.6
        elif s == 2:
            ok = drive and min(np.linalg.norm(c[:2] - base.pos) for c in world.cubes) < 0.55
        elif s == 4:
            ok = world.held is not None
        if ok:
            self.step += 1
        return ok

    def on_event(self, kind):
        """drive_on, drive_off, placed and new_round finish their steps. -> 'next' | 'finished' | None"""
        want = {1: "drive_on", 3: "drive_off", 5: "placed", 6: "new_round"}.get(self.step)
        if not self.active or kind != want:
            return None
        if self.step == len(TUTORIAL) - 1:
            self.active = False
            return "finished"
        self.step += 1
        return "next"


def write_tones(path, notes, rate=22050):
    """Tiny synth: a list of (frequency Hz, seconds) -> 16-bit mono WAV. Frequency 0 = silence."""
    parts = []
    for f, d in notes:
        t = np.arange(int(rate * d)) / rate
        env = np.minimum(1, t / 0.005) * np.exp(-t * 18)
        parts.append(np.sin(2 * math.pi * f * t) * env * (0.35 if f else 0))
    data = (np.concatenate(parts) * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())


class Sound:
    """Short beeps on Windows (winsound plays them without blocking). Silent elsewhere."""
    NOTES = {
        "pick": [(660, .05), (990, .08)], "place": [(392, .06), (262, .10)],
        "miss": [(220, .08), (0, .04), (196, .10)], "gesture": [(784, .06), (1175, .10)],
        "start": [(523, .06), (784, .12)], "click": [(1200, .03)],
        "win": [(523, .09), (659, .09), (784, .09), (1047, .25)],
    }

    def __init__(self, on=True):
        self.on, self.ok, self.files, self.ws = on, False, {}, None
        try:
            import winsound
            folder = os.path.join(tempfile.gettempdir(), "arm_lab_sounds")
            os.makedirs(folder, exist_ok=True)
            for name, notes in self.NOTES.items():
                self.files[name] = os.path.join(folder, name + ".wav")
                write_tones(self.files[name], notes)
            self.ws, self.ok = winsound, True
        except Exception:
            pass

    def play(self, name):
        if self.on and self.ok and name in self.files:
            try:
                self.ws.PlaySound(self.files[name], self.ws.SND_FILENAME | self.ws.SND_ASYNC | self.ws.SND_NODEFAULT)
            except Exception:
                pass


def load_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def fmt_time(s):
    if s is None:
        return "--"
    s = round(s, 1)
    return f"{int(s // 60)}:{s % 60:04.1f}"

# =====================================================================
# 7. LOOK & FEEL  -  "night-shift engineering blueprint"
#    navy drafting floor, industrial-orange arm, paper-white linework
# =====================================================================
def hexc(h):
    """'#RRGGBB' -> OpenCV BGR tuple."""
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


C = dict(
    ink_deep=hexc("#08111F"), ink=hexc("#0E1B2E"), floor=hexc("#10233D"),
    grid=hexc("#1C3B60"), grid_major=hexc("#2F6399"), paper=hexc("#DCE8F7"), muted=hexc("#7F9BBF"),
    orange=hexc("#FF7A1A"), orange_dark=hexc("#A9480A"), orange_light=hexc("#FFC08F"),
    graphite=hexc("#1E2633"), graphite_light=hexc("#46546A"),
    teal=hexc("#2EC4B6"), yellow=hexc("#FFD23F"), coral=hexc("#FF5A5F"), sky=hexc("#5AB4FF"),
)
CUBE_COLORS = [C["coral"], C["yellow"], C["sky"]]
FONT_FILE = ""                 # optional: path to your own .ttf font


class TextLayer:
    """Smooth TrueType text (via Pillow) drawn in one pass per frame.
    Uses Bahnschrift on Windows; falls back to OpenCV's built-in font if Pillow is missing."""

    CANDIDATES = [
        ("C:/Windows/Fonts/bahnschrift.ttf", "C:/Windows/Fonts/bahnschrift.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"),
        ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]

    def __init__(self):
        self.items, self.cache = [], {}
        self.paths = None
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.Image, self.ImageDraw, self.ImageFont = Image, ImageDraw, ImageFont
            cands = ([(FONT_FILE, FONT_FILE)] if FONT_FILE else []) + self.CANDIDATES
            self.paths = next((c for c in cands if os.path.exists(c[0])), None)
        except ImportError:
            print("Tip: install Pillow for nicer fonts:  py -3.14 -m pip install pillow")

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.cache:
            path = self.paths[1] if bold else self.paths[0]
            f = self.ImageFont.truetype(path, size)
            if "bahnschrift" in path.lower():            # one variable font file, pick the weight
                for name in (("Bold", b"Bold") if bold else ("Regular", b"Regular")):
                    try:
                        f.set_variation_by_name(name)
                        break
                    except Exception:
                        pass
            self.cache[key] = f
        return self.cache[key]

    def width(self, s, size, bold=False):
        if self.paths:
            key = ("w", s, size, bold)
            if key not in self.cache:
                self.cache[key] = int(self.font(size, bold).getlength(s))
            return self.cache[key]
        return cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, size / 34, 2 if bold else 1)[0][0]

    def add(self, s, xy, size=15, color=None, bold=False, anchor="lt"):
        """anchor: 'lt' left-top, 'mm' centre, 'rt' right-top, 'lm' left-middle, 'mt' centre-top."""
        self.items.append((s, (int(xy[0]), int(xy[1])), size, color or C["paper"], bold, anchor))

    def sprite(self, s, size, bold, anchor):
        """Render a piece of text once as an alpha mask; reused every frame after that."""
        key = (s, size, bold, anchor)
        spr = self.cache.get(key)
        if spr is None:
            f = self.font(size, bold)
            x0, y0, x1, y1 = f.getbbox(s, anchor=anchor)
            w, h = max(1, x1 - x0 + 2), max(1, y1 - y0 + 2)
            im = self.Image.new("L", (w, h), 0)
            self.ImageDraw.Draw(im).text((-x0 + 1, -y0 + 1), s, font=f, fill=255, anchor=anchor)
            spr = (np.asarray(im, np.float32)[..., None] / 255.0, x0 - 1, y0 - 1)
            if len(self.cache) > 1500:
                self.cache = {k: v for k, v in self.cache.items() if isinstance(k[0], int)}
            self.cache[key] = spr
        return spr

    def render(self, img):
        if not self.items:
            return
        H_, W_ = img.shape[:2]
        for s, (x, y), size, col, bold, anchor in self.items:
            if self.paths:
                a, ox, oy = self.sprite(s, size, bold, anchor)
                x0, y0 = x + ox, y + oy
                h, w = a.shape[:2]
                cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W_, x0 + w), min(H_, y0 + h)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                aa = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
                reg = img[cy0:cy1, cx0:cx1]
                reg[:] = (reg * (1 - aa) + np.array(col, np.float32) * aa).astype(np.uint8)
            else:
                sc, th = size / 34, 2 if bold else 1
                (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, sc, th)
                x -= w // 2 if anchor[0] == "m" else (w if anchor[0] == "r" else 0)
                y += h if anchor[1] == "t" else (h // 2 if anchor[1] == "m" else 0)
                cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_DUPLEX, sc, col, th, cv2.LINE_AA)
        self.items = []


TXT = TextLayer()


# ---------- small drawing helpers ----------
_FILLS = {}


def blend_rect(img, x0, y0, x1, y1, color, alpha):
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 > x0 and y1 > y0:
        sub = img[y0:y1, x0:x1]
        key = (sub.shape, tuple(color))
        fill = _FILLS.get(key)
        if fill is None:
            fill = _FILLS[key] = np.full_like(sub, color)
        cv2.addWeighted(sub, 1 - alpha, fill, alpha, 0, dst=sub)


def panel(img, x0, y0, x1, y1, accent=None):
    """Glassy ink card with drafting corner marks."""
    blend_rect(img, x0, y0, x1, y1, C["ink"], 0.84)
    cv2.rectangle(img, (x0, y0), (x1, y1), C["grid_major"], 1, cv2.LINE_AA)
    t, col = 12, accent or C["paper"]
    for (cx, cy, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * t, cy), col, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * t), col, 2, cv2.LINE_AA)


def keycap(img, x, y, key, label):
    """[K] Camera  - returns the x where the next keycap goes."""
    w = max(26, TXT.width(key, 14, True) + 14)
    cv2.rectangle(img, (x, y), (x + w, y + 24), C["graphite"], -1)
    cv2.rectangle(img, (x, y), (x + w, y + 24), C["graphite_light"], 1, cv2.LINE_AA)
    cv2.line(img, (x + 2, y + 23), (x + w - 2, y + 23), C["orange"], 2)
    TXT.add(key, (x + w // 2, y + 12), 14, C["paper"], True, "mm")
    TXT.add(label, (x + w + 8, y + 12), 14, C["muted"], False, "lm")
    return x + w + 16 + TXT.width(label, 14)


def dashed(img, pts, color, dash=7, gap=6, thick=1):
    """Dashed polyline (drafting style)."""
    carry, on = 0.0, True
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.array(a, float), np.array(b, float)
        seg = np.linalg.norm(b - a)
        pos = 0.0
        while pos < seg:
            step = min((dash if on else gap) - carry, seg - pos)
            if on:
                p0, p1 = a + (b - a) * pos / seg, a + (b - a) * (pos + step) / seg
                cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), color, thick, cv2.LINE_AA)
            pos += step
            carry += step
            if carry >= (dash if on else gap) - 1e-6:
                carry, on = 0.0, not on


# ---------- 3D view (camera follows behind the arm, and the base as it drives) ----------
class View:
    def __init__(self, el=1.08, dist=1.30, look_ahead=0.34):
        self.az, self.el, self.dist, self.ahead = math.pi + VIEW_SIDE, el, dist, look_ahead
        self.base = np.zeros(2)
        self.f = 820
        self.cx, self.cy = WIN_W * 0.50, WIN_H * 0.47
        self.bg, self.bg_key, self.ver = None, None, 0
        g = np.linspace(0, 1, WIN_H)[:, None, None]
        self.sky = (np.array(C["ink_deep"]) * (1 - g) + np.array(C["ink"]) * g).astype(np.uint8).repeat(WIN_W, 1)
        yy, xx = np.mgrid[0:WIN_H, 0:WIN_W]
        r = np.hypot((xx - WIN_W / 2) / (WIN_W * 0.62), (yy - WIN_H * 0.48) / (WIN_H * 0.70))
        vig = np.clip(1.08 - 0.45 * r ** 2, 0.55, 1.0)
        self.vignette = np.repeat(((1 - vig) * 70)[..., None], 3, 2).astype(np.uint8)  # darkening, subtracted
        fade = np.clip(0.2 + 0.8 * (yy / WIN_H) ** 0.8, 0, 1)                        # grid fades into the distance
        self.fade = np.repeat(((1 - fade) * 110)[..., None], 3, 2).astype(np.uint8)
        self.update()

    def follow(self, yaw, base_pos, dt, driving=False, turn=True):
        """Stay over the base; turn (lazily, past a dead-zone) to look along `yaw`.
        Driving: look further ahead from a little higher up."""
        changed = False
        want_ahead, want_dist = (0.22, 1.80) if driving else (0.34, 1.30)
        k = min(1.0, 3.0 * dt)
        if abs(want_ahead - self.ahead) > 1e-4 or abs(want_dist - self.dist) > 1e-4:
            self.ahead += (want_ahead - self.ahead) * k
            self.dist += (want_dist - self.dist) * k
            changed = True
        if np.linalg.norm(base_pos - self.base) > 1e-5:
            self.base, changed = np.array(base_pos, float), True
        if turn:
            e = math.remainder(yaw + math.pi + VIEW_SIDE - self.az, 2 * math.pi)
            e = math.copysign(max(abs(e) - VIEW_DEADZONE, 0), e)
            if e:
                self.az += e * min(1.0, VIEW_SPEED * dt)
                changed = True
        if changed:
            self.update()

    def update(self):
        self.ver += 1
        self.dir = self.az - math.pi - VIEW_SIDE                                # the way the view looks
        self.look = np.array([*self.base, 0.0]) + self.ahead * np.array([math.cos(self.dir), math.sin(self.dir), 0.0])
        self.eye = self.look + self.dist * np.array([
            math.cos(self.el) * math.cos(self.az), math.cos(self.el) * math.sin(self.az),
            math.sin(self.el)])
        fwd = self.look - self.eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        # roll the view so the arm's direction always points straight UP on screen
        # -> hand up = screen up, hand right = screen right, at every angle
        sc = lambda p: (lambda c: np.array([c[0] / c[2], c[1] / c[2]]))(np.stack([right, up, fwd]) @ (p - self.eye))
        d = sc(self.look) - sc(np.array([*self.base, 0.0]))
        psi = math.atan2(d[1], d[0]) - math.pi / 2
        right, up = math.cos(psi) * right + math.sin(psi) * up, -math.sin(psi) * right + math.cos(psi) * up
        self.R = np.stack([right, up, fwd])
        # the patch of floor that is drawn: in front of the camera only (never behind it)
        f2 = np.array([math.cos(self.dir), math.sin(self.dir)])
        s2 = np.array([-f2[1], f2[0]])
        L = self.look[:2]
        win = [L + f2 * 2.4 + s2 * 2.0, L - f2 * 0.9 + s2 * 2.0, L - f2 * 0.9 - s2 * 2.0, L + f2 * 2.4 - s2 * 2.0]
        self.window = win
        self.floor = clip_poly(win, [(-ARENA, -ARENA), (ARENA, -ARENA), (ARENA, ARENA), (-ARENA, ARENA)])

    def project(self, p):
        c = self.R @ (np.asarray(p) - self.eye)
        z = max(c[2], 1e-3)
        return (int(self.cx + self.f * c[0] / z), int(self.cy - self.f * c[1] / z)), z

    def px(self, p):
        return self.project(p)[0]

    def sees(self, p):
        """Inside the drawn patch of floor (so safe to project)."""
        return inside_poly(p[:2], self.window)

    def background(self, world):
        """Floor, grid, walls and trays: drawn again only when the view or the trays change."""
        key = (self.ver, world.layout)
        if self.bg is None or self.bg_key != key:
            self.bg, self.bg_key = build_background(self, world), key
        return self.bg


def clip_poly(poly, clip):
    """Sutherland-Hodgman: the part of convex polygon `poly` inside convex polygon `clip` (both CCW)."""
    out = [np.array(p, float) for p in poly]
    for i in range(len(clip)):
        a, b = np.array(clip[i], float), np.array(clip[(i + 1) % len(clip)], float)
        side = lambda p, a=a, b=b: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        src, out = out, []
        for j in range(len(src)):
            p, q = src[j], src[(j + 1) % len(src)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                out.append(p + (q - p) * (sp / (sp - sq)))
        if not out:
            return []
    return out


def inside_poly(p, poly):
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0:
            return False
    return True


def clip_segment(a, b, poly):
    """The part of segment a-b inside convex CCW polygon (Cyrus-Beck), or None."""
    a, b = np.array(a, float), np.array(b, float)
    t0, t1, d = 0.0, 1.0, b - a
    for i in range(len(poly)):
        p, q = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
        n = np.array([-(q[1] - p[1]), q[0] - p[0]])          # inward normal for a CCW polygon
        num, den = n @ (a - p), n @ d
        if abs(den) < 1e-12:
            if num < 0:
                return None
            continue
        t = -num / den
        if den > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return a + d * t0, a + d * t1


def tray_color(world, k):
    return CUBE_COLORS[k] if world.mode == "sort" else C["teal"]


def tray_label(world, k):
    return COLOR_NAMES[k] if world.mode == "sort" else ("Tower" if world.mode == "tower" else "Tray")


def tray_corners(view, c, z=0.001):
    return np.array([view.px((c[0] + dx * TRAY_HALF, c[1] + dy * TRAY_HALF, z))
                     for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


def build_background(view, world):
    img = view.sky.copy()
    if len(view.floor) >= 3:
        cv2.fillConvexPoly(img, np.array([view.px((*p, 0)) for p in view.floor]), C["floor"], cv2.LINE_AA)

        # drafting grid: 10 cm minor, 50 cm major, fixed to the floor, fading into the distance
        grid = np.zeros_like(img)
        for v in np.arange(-ARENA, ARENA + 1e-6, 0.1):
            major = abs(round(v / 0.5) * 0.5 - v) < 1e-6
            for a, b in (((v, -ARENA), (v, ARENA)), ((-ARENA, v), (ARENA, v))):
                seg = clip_segment(a, b, view.floor)
                if seg is not None:
                    cv2.line(grid, view.px((*seg[0], 0)), view.px((*seg[1], 0)),
                             C["grid_major"] if major else C["grid"], 1, cv2.LINE_AA)
        cv2.add(img, cv2.subtract(grid, view.fade), dst=img)

    # the arena wall: a low rim around the floor
    corners = [(-ARENA, -ARENA), (ARENA, -ARENA), (ARENA, ARENA), (-ARENA, ARENA)]
    for i in range(4):
        seg = clip_segment(corners[i], corners[(i + 1) % 4], view.window)
        if seg is not None:
            a, b = seg
            quad = np.array([view.px((*a, 0)), view.px((*b, 0)), view.px((*b, 0.06)), view.px((*a, 0.06))])
            cv2.fillConvexPoly(img, quad, C["graphite"], cv2.LINE_AA)
            cv2.line(img, view.px((*a, 0.06)), view.px((*b, 0.06)), C["muted"], 1, cv2.LINE_AA)

    # trays: hatched drop zones (each worked on its own small region only)
    for k, c in enumerate(world.trays):
        if not view.sees(c):
            continue
        col = tray_color(world, k)
        t = tray_corners(view, c)
        x0, y0 = max(0, t[:, 0].min() - 2), max(0, t[:, 1].min() - 2)
        x1, y1 = min(WIN_W, t[:, 0].max() + 3), min(WIN_H, t[:, 1].max() + 3)
        if x1 > x0 and y1 > y0:
            reg = img[y0:y1, x0:x1]
            mask = np.zeros(reg.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, t - [x0, y0], 255, cv2.LINE_AA)
            art = reg.copy()
            cv2.addWeighted(art, 0.84, np.full_like(art, col), 0.16, 0, dst=art)
            hh = y1 - y0
            for k2 in range(-hh, x1 - x0, 9):
                cv2.line(art, (k2, hh), (k2 + hh, 0), col, 1, cv2.LINE_AA)
            m = mask[..., None].astype(np.uint16)
            reg[:] = ((art.astype(np.uint16) * m + reg.astype(np.uint16) * (255 - m)) // 255).astype(np.uint8)
        cv2.polylines(img, [t], True, col, 2, cv2.LINE_AA)

    return cv2.subtract(img, view.vignette)


def shade(color, k):
    return tuple(int(min(255, c * k)) for c in color)


def draw_link(img, view, a, b, radius, color, edge=None, highlight=True):
    (pa, da), (pb, db) = view.project(a), view.project(b)
    w = max(2, int(view.f * radius * 2 / ((da + db) / 2)))
    cv2.line(img, pa, pb, edge or shade(color, 0.5), w + 4, cv2.LINE_AA)
    cv2.line(img, pa, pb, color, w, cv2.LINE_AA)
    if highlight:
        off = np.array([pb[1] - pa[1], pa[0] - pb[0]], float)
        n = np.linalg.norm(off)
        if n > 0:
            off = (off / n * w * 0.25).astype(int)
            cv2.line(img, tuple(pa - off), tuple(pb - off), C["orange_light"] if color == C["orange"] else shade(color, 1.5),
                     max(1, w // 6), cv2.LINE_AA)


def draw_joint(img, view, p, radius):
    c, d = view.project(p)
    r = max(3, int(view.f * radius / d))
    cv2.circle(img, c, r + 2, C["ink_deep"], -1, cv2.LINE_AA)
    cv2.circle(img, c, r, C["graphite"], -1, cv2.LINE_AA)
    cv2.circle(img, c, r, C["orange"], 2, cv2.LINE_AA)
    cv2.circle(img, c, max(2, r // 3), C["graphite_light"], -1, cv2.LINE_AA)


CUBE_FACES = [((0, 1, 3, 2), (-1, 0, 0)), ((4, 5, 7, 6), (1, 0, 0)), ((0, 1, 5, 4), (0, -1, 0)),
              ((2, 3, 7, 6), (0, 1, 0)), ((0, 2, 6, 4), (0, 0, -1)), ((1, 3, 7, 5), (0, 0, 1))]
LIGHT = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])


def draw_cube(img, glow, view, c, color, held, gpts=None):
    corners = [c + CUBE * np.array([dx, dy, dz]) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    pts = [view.px(k) for k in corners]
    for idx, nrm in CUBE_FACES:
        nrm = np.array(nrm, float)
        if np.dot(nrm, c - view.eye) >= 0:
            continue
        poly = np.array([pts[i] for i in idx])
        cv2.fillConvexPoly(img, poly, shade(color, 0.45 + 0.55 * max(0, nrm @ LIGHT)), cv2.LINE_AA)
        cv2.polylines(img, [poly], True, C["paper"] if held else shade(color, 0.35), 1, cv2.LINE_AA)
        if held:
            cv2.polylines(glow, [poly], True, color, 3, cv2.LINE_AA)
            if gpts is not None:
                gpts.extend(poly.tolist())


def add_glow(img, glow, pts, strength=0.9):
    """Soft bloom, computed only in the box around the glowing things."""
    if not pts:
        return
    pts = np.array(pts)
    m = 40
    x0, y0 = max(0, pts[:, 0].min() - m), max(0, pts[:, 1].min() - m)
    x1, y1 = min(WIN_W, pts[:, 0].max() + m), min(WIN_H, pts[:, 1].max() + m)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    g = glow[y0:y1, x0:x1]
    small = cv2.resize(g, ((x1 - x0) // 3 + 1, (y1 - y0) // 3 + 1), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 4)
    big = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    reg = img[y0:y1, x0:x1]
    cv2.addWeighted(reg, 1.0, big, strength, 0, dst=reg)
    cv2.addWeighted(reg, 1.0, g, 0.35, 0, dst=reg)
    g[:] = 0                                       # clear the reused glow buffer where we drew


_GLOW = {}


def draw_scene(view, q, grip_gap, world, lock, base, drive=False):
    img = view.background(world).copy()
    glow = _GLOW.get("buf")
    if glow is None or glow.shape != img.shape:
        glow = _GLOW["buf"] = np.zeros_like(img)
    gpts = []                                      # where things glow (bloom is computed only there)
    k = {n: (base.to_world3(p) if n not in ("side", "tool") else np.array([*base.rot() @ p[:2], p[2]]))
         for n, p in forward_kinematics(q).items()}      # arm points: base frame -> world
    flat = lambda p: (p[0], p[1], 0.0)
    fwd2, side2 = base.rot() @ [1.0, 0.0], base.rot() @ [0.0, 1.0]
    cubes_seen = [i for i, c in enumerate(world.cubes) if view.sees(c)]

    # soft shadows
    shadow = img.copy()
    s0, d0 = view.project((*base.pos, 0))
    cv2.circle(shadow, s0, int(view.f * (BASE_R + 0.02) / d0), C["ink_deep"], -1, cv2.LINE_AA)
    for a, b, r in [("shoulder", "elbow", 0.03), ("elbow", "wrist", 0.025), ("wrist", "palm", 0.02)]:
        (pa, d), pb = view.project(flat(k[a])), view.px(flat(k[b]))
        cv2.line(shadow, pa, pb, C["ink_deep"], max(2, int(view.f * r * 2 / d)), cv2.LINE_AA)
    for i in cubes_seen:
        sp, d = view.project(flat(world.cubes[i]))
        cv2.circle(shadow, sp, int(view.f * CUBE * 1.35 / d), C["ink_deep"], -1, cv2.LINE_AA)
    cv2.addWeighted(shadow, 0.55, img, 0.45, 0, img)

    # carrying a cube: the tray(s) it can go to light up
    if world.held is not None:
        for kk, c in enumerate(world.trays):
            if (world.mode != "sort" or kk == world.held) and view.sees(c):
                t = tray_corners(view, c)
                col = tray_color(world, kk)
                cv2.polylines(img, [t], True, col, 3, cv2.LINE_AA)
                cv2.polylines(glow, [t], True, col, 4, cv2.LINE_AA)
                gpts += t.tolist()

    g = k["grasp"]
    base3 = np.array([*base.pos, 0.0])
    base_pt, foot = view.px(base3), view.px(flat(g))
    if not drive:
        # reach limits as dashed drafting arcs around the base
        for rr in (MIN_R, MAX_R):
            arc = [view.px((*base.to_world((rr * math.cos(t), rr * math.sin(t))), 0.001))
                   for t in np.linspace(-SWING_RANGE, SWING_RANGE, 61)]
            for i in range(0, len(arc) - 1, 2):
                cv2.line(img, arc[i], arc[i + 1], C["muted"], 1, cv2.LINE_AA)
        # drafting annotations: reach (dimension line) and height
        reach_cm, height_cm = np.linalg.norm(g[:2] - base.pos) * 100, g[2] * 100
        vec = np.subtract(foot, base_pt).astype(float)
        if np.linalg.norm(vec) > 30:
            nrm = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec) * 22
            a, b = (np.array(base_pt) + nrm).astype(int), (np.array(foot) + nrm).astype(int)
            cv2.line(img, tuple(a), tuple(b), C["muted"], 1, cv2.LINE_AA)
            for p, q_ in [(base_pt, a), (foot, b)]:
                cv2.line(img, tuple(np.array(p) + (nrm * 0.3).astype(int)), tuple(np.array(q_) + (nrm * 0.25).astype(int)),
                         C["muted"], 1, cv2.LINE_AA)
            u = vec / np.linalg.norm(vec)
            for p, sgn in [(a, 1), (b, -1)]:                  # arrowheads
                tip = np.array(p, float)
                for ang in (0.45, -0.45):
                    r = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) @ (u * sgn)
                    cv2.line(img, tuple(tip.astype(int)), tuple((tip + r * 9).astype(int)), C["muted"], 1, cv2.LINE_AA)
            mid = (a + b) / 2 + nrm * 0.75
            TXT.add(f"{reach_cm:.0f} cm", mid, 14, C["muted"], False, "mm")
        top = view.px(g)
        dashed(img, [top, foot], C["teal"] if lock is not None else C["paper"], 4, 4)
        TXT.add(f"h {height_cm:.0f} cm", ((top[0] + foot[0]) / 2 + 12, (top[1] + foot[1]) / 2), 13, C["muted"], False, "lm")

        # aim marker: crosshair ring, teal + glowing when locked on
        col = C["teal"] if lock is not None else C["paper"]
        r = 10 if lock is not None else 8
        if lock is not None:
            gpts += [(foot[0] - 20, foot[1] - 20), (foot[0] + 20, foot[1] + 20)]
        for layer in ([img, glow] if lock is not None else [img]):
            cv2.circle(layer, foot, r, col, 2, cv2.LINE_AA)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                cv2.line(layer, (foot[0] + dx * (r + 3), foot[1] + dy * (r + 3)),
                         (foot[0] + dx * (r + 8), foot[1] + dy * (r + 8)), col, 2, cv2.LINE_AA)
    else:
        # drive mode: a dashed arrow on the floor showing where the base is heading
        tip = base.pos + fwd2 * (BASE_R + 0.28)
        dashed(img, [view.px((*(base.pos + fwd2 * (BASE_R + 0.04)), 0.002)), view.px((*tip, 0.002))], C["teal"], 6, 5, 2)
        for sgn in (1, -1):
            cv2.line(img, view.px((*tip, 0.002)), view.px((*(tip - fwd2 * 0.06 + sgn * side2 * 0.045), 0.002)),
                     C["teal"], 2, cv2.LINE_AA)

    # cubes + robot, far to near
    items = []
    for i in cubes_seen:
        c = world.cubes[i]
        items.append((view.project(c)[1], lambda c=c, i=i: draw_cube(
            img, glow, view, c, CUBE_COLORS[i % 3], i == world.held, gpts)))
    if world.held is not None and world.held not in cubes_seen:
        c = world.cubes[world.held]
        items.append((view.project(c)[1], lambda c=c: draw_cube(img, glow, view, c, CUBE_COLORS[world.held % 3], True, gpts)))

    def ring(r, z):
        return np.array([view.px((*(base.pos + r * np.array([math.cos(t), math.sin(t)])), z))
                         for t in np.linspace(0, 2 * math.pi, 40)])

    def robot():
        # wheeled base: far wheel, chassis, near wheel
        wheels = sorted([(sgn, view.project((*(base.pos + side2 * sgn * (BASE_R - 0.01)), 0.035))[1]) for sgn in (1, -1)],
                        key=lambda t: -t[1])
        def wheel(sgn):
            cpos = base.pos + side2 * sgn * (BASE_R - 0.01)
            a, b = view.px((*(cpos - fwd2 * 0.04), 0.035)), view.px((*(cpos + fwd2 * 0.04), 0.035))
            d = view.project((*cpos, 0.035))[1]
            wpx = max(4, int(view.f * 0.07 / d))
            cv2.line(img, a, b, C["ink_deep"], wpx + 4, cv2.LINE_AA)
            cv2.line(img, a, b, C["graphite"], wpx, cv2.LINE_AA)
            for k2 in np.linspace(-0.03, 0.03, 4):                      # tread marks
                p = cpos + fwd2 * k2
                cv2.line(img, view.px((*(p + side2 * sgn * 0.012), 0.07)), view.px((*(p + side2 * sgn * 0.012), 0.0)),
                         C["graphite_light"], 1, cv2.LINE_AA)
        wheel(wheels[0][0])
        bottom, top_ = ring(BASE_R, 0.012), ring(BASE_R, 0.07)
        cv2.fillPoly(img, [cv2.convexHull(np.vstack([bottom, top_]).astype(np.int32))], C["graphite"], cv2.LINE_AA)
        cv2.fillPoly(img, [top_], C["graphite_light"], cv2.LINE_AA)
        cv2.polylines(img, [top_], True, C["orange"], 2, cv2.LINE_AA)
        nose = [view.px((*(base.pos + fwd2 * (BASE_R - 0.015)), 0.071)),
                view.px((*(base.pos + fwd2 * (BASE_R - 0.05) + side2 * 0.03), 0.071)),
                view.px((*(base.pos + fwd2 * (BASE_R - 0.05) - side2 * 0.03), 0.071))]
        cv2.fillConvexPoly(img, np.array(nose), C["teal"] if drive else C["orange"], cv2.LINE_AA)
        wheel(wheels[1][0])
        # arm column on top of the base
        for z, rr, colr in [(0.07, 0.075, C["ink_deep"]), (0.085, 0.066, C["graphite"])]:
            cv2.fillPoly(img, [ring(rr, z)], colr, cv2.LINE_AA)
        draw_link(img, view, k["base"] + [0, 0, 0.085], k["shoulder"], 0.045, C["graphite"], C["ink_deep"], False)
        draw_joint(img, view, k["shoulder"], 0.047)
        draw_link(img, view, k["shoulder"], k["elbow"], 0.032, C["orange"], C["orange_dark"])
        draw_joint(img, view, k["elbow"], 0.038)
        draw_link(img, view, k["elbow"], k["wrist"], 0.026, C["orange"], C["orange_dark"])
        draw_joint(img, view, k["wrist"], 0.029)
        draw_link(img, view, k["wrist"], k["palm"], 0.02, C["graphite"], C["ink_deep"], False)
        draw_link(img, view, k["palm"] + k["side"] * 0.042, k["palm"] - k["side"] * 0.042, 0.012,
                  C["graphite_light"], C["ink_deep"], False)
        for sgn in (1, -1):
            f0 = k["palm"] + sgn * k["side"] * grip_gap
            f1 = f0 + k["tool"] * FINGER
            draw_link(img, view, f0, f1, 0.008, C["graphite_light"], C["ink_deep"], False)
            draw_link(img, view, f1 - k["tool"] * 0.012, f1, 0.008, C["orange"], C["orange_dark"], False)
    items.append((view.project(k["elbow"])[1], robot))

    for _, fn in sorted(items, key=lambda t: -t[0]):
        fn()

    for kk, c in enumerate(world.trays):                     # labels on the far side of each tray
        lp = c + np.array([math.cos(view.dir), math.sin(view.dir)]) * (TRAY_HALF * 1.45 + 0.04)
        if view.sees(lp):
            TXT.add(tray_label(world, kk), view.px((*lp, 0)), 14, tray_color(world, kk), True, "mm")
    add_glow(img, glow, gpts)
    return img


# ---------- HUD ----------
def wrap(text, chars):
    lines = []
    for word in text.split():
        if lines and len(lines[-1]) + len(word) < chars:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def grip_gauge(img, cx, cy, r, value, closed):
    """270-degree arc gauge with the grab threshold marked."""
    cv2.ellipse(img, (cx, cy), (r, r), 135, 0, 270, C["graphite_light"], 6, cv2.LINE_AA)
    col = C["teal"] if closed else C["orange"]
    if value > 0.005:
        cv2.ellipse(img, (cx, cy), (r, r), 135, 0, int(270 * min(value, 1)), col, 6, cv2.LINE_AA)
    a = math.radians(135 + 270 * (1 - GRAB_CLOSE))
    cv2.line(img, (int(cx + (r - 9) * math.cos(a)), int(cy + (r - 9) * math.sin(a))),
             (int(cx + (r + 9) * math.cos(a)), int(cy + (r + 9) * math.sin(a))), C["paper"], 2, cv2.LINE_AA)
    TXT.add(f"{int(round(value * 100))}%", (cx, cy - 4), 20, C["paper"], True, "mm")
    TXT.add("grip", (cx, cy + 16), 12, C["muted"], False, "mm")


def bar(img, x0, y, x1, value, color):
    cv2.line(img, (x0, y), (x1, y), C["graphite_light"], 3)
    if value > 0:
        cv2.line(img, (x0, y), (x0 + int((x1 - x0) * min(value, 1.0)), y), color, 3)


POSE_TEXT = {"open": "open hand", "fist": "fist", "thumbs_up": "thumbs up", "peace": "peace sign",
             "point": "pointing", "rock": "rock sign", "unsure": "...", "none": "-"}


def gesture_action(g, name):
    if name == "thumbs_up":
        return "New round"
    if name == "peace":
        return "Auto mode off" if g.pilot is not None else "Auto mode on"
    if name == "rock":
        return "Arm mode" if g.drive else "Drive mode"
    return "Resume" if g.paused else "Pause"


def draw_mission(img, g, x0, y0, x1, y1):
    w, r = g.world, g.round
    panel(img, x0, y0, x1, y1)
    name, goal = MODES[w.mode]
    TXT.add("MODE", (x0 + 16, y0 + 12), 12, C["orange"], True)
    TXT.add(name, (x0 + 16, y0 + 30), 24, C["paper"], True)
    if w.mode != "free":
        TXT.add("TIME", (x1 - 16, y0 + 12), 12, C["orange"], True, "rt")
        TXT.add(fmt_time(r.time), (x1 - 16, y0 + 28), 28, C["teal"] if r.done else C["paper"], True, "rt")
    for i, line in enumerate(wrap(goal, 48)[:2]):
        TXT.add(line, (x0 + 16, y0 + 68 + 19 * i), 14, C["muted"])
    TXT.add("Done", (x0 + 16, y0 + 117), 14, C["paper"])
    for i in range(len(w.cubes)):
        bx, ok = x0 + 62 + i * 24, w.done(i)
        col = CUBE_COLORS[i] if w.mode == "sort" else C["teal"]
        cv2.rectangle(img, (bx, y0 + 116), (bx + 16, y0 + 132), col if ok else C["graphite_light"],
                      -1 if ok else 1, cv2.LINE_AA)
    if w.mode != "free":
        best = g.stats.get("best", {}).get(w.mode)
        TXT.add(f"Best {fmt_time(best)}", (x1 - 16, y0 + 117), 14, C["muted"], False, "rt")


def draw_tutorial(img, g, x0, y0, x1, y1):
    tut = g.tut
    panel(img, x0, y0, x1, y1, C["orange"])
    title, text = TUTORIAL[tut.step]
    TXT.add(f"TUTORIAL  {tut.step + 1} / {len(TUTORIAL)}", (x0 + 16, y0 + 12), 12, C["orange"], True)
    TXT.add("U skip", (x1 - 16, y0 + 12), 12, C["muted"], False, "rt")
    TXT.add(title, (x0 + 16, y0 + 32), 22, C["paper"], True)
    for i, line in enumerate(wrap(text, 48)[:3]):
        TXT.add(line, (x0 + 16, y0 + 66 + 19 * i), 14, C["paper"])
    for i in range(len(TUTORIAL)):
        cx = x0 + 22 + i * 18
        done = i < tut.step
        cv2.circle(img, (cx, y1 - 18), 5, C["teal"] if done else (C["orange"] if i == tut.step else C["graphite_light"]),
                   -1 if done or i == tut.step else 1, cv2.LINE_AA)


def draw_minimap(img, g, x0, y0, size):
    """Top-down map of the whole floor: walls, trays, cubes, the robot and where it faces."""
    w, base = g.world, g.base
    panel(img, x0, y0, x0 + size, y0 + size)
    m = 14
    k = (size - 2 * m) / (2 * ARENA)
    P = lambda p: (int(x0 + m + (p[1] * -1 + ARENA) * k), int(y0 + m + (ARENA - p[0]) * k))   # +x = up, +y = left
    cv2.rectangle(img, P((ARENA, ARENA)), P((-ARENA, -ARENA)), C["grid_major"], 1, cv2.LINE_AA)
    for kk, c in enumerate(w.trays):
        tc = P(c)
        cv2.rectangle(img, (tc[0] - 7, tc[1] - 7), (tc[0] + 7, tc[1] + 7), tray_color(w, kk), 2, cv2.LINE_AA)
    for i, c in enumerate(w.cubes):
        if i != w.held:
            p = P(c)
            cv2.rectangle(img, (p[0] - 3, p[1] - 3), (p[0] + 3, p[1] + 3), CUBE_COLORS[i % 3], -1)
    view_a = g.view.dir                                    # the camera's view: a faint wedge
    for da in (-0.5, 0.5):
        cv2.line(img, P(base.pos), P(base.pos + 0.7 * np.array([math.cos(view_a + da), math.sin(view_a + da)])),
                 C["graphite_light"], 1, cv2.LINE_AA)
    c = P(base.pos)
    r = max(4, int(BASE_R * k))
    cv2.circle(img, c, r, C["teal"] if g.drive else C["orange"], -1, cv2.LINE_AA)
    cv2.line(img, c, P(base.pos + (BASE_R + 0.12) * np.array([math.cos(base.th), math.sin(base.th)])),
             C["paper"], 2, cv2.LINE_AA)
    if w.held is not None:
        cv2.circle(img, c, r + 3, CUBE_COLORS[w.held % 3], 1, cv2.LINE_AA)
    TXT.add("MAP", (x0 + 10, y0 + 4), 11, C["orange"], True)


def draw_badges(img, g, t):
    chips = []
    if g.drive:
        chips.append(("DRIVE", C["teal"], False))
    if g.pilot is not None:
        chips.append(("AUTO", C["sky"], False))
    if g.rec.recording:
        chips.append((f"TEACHING {fmt_time(g.rec.t)}", C["coral"], True))
    if g.replaying:
        chips.append(("REPLAY", C["yellow"], False))
    if g.paused:
        chips.append(("PAUSED", C["paper"], False))
    if not chips:
        return
    widths = [TXT.width(s, 13, True) + 34 for s, _, _ in chips]
    x = WIN_W // 2 - (sum(widths) + 8 * (len(chips) - 1)) // 2
    for (s, col, blink), w in zip(chips, widths):
        blend_rect(img, x, 76, x + w, 102, C["ink"], 0.9)
        cv2.rectangle(img, (x, 76), (x + w, 102), col, 1, cv2.LINE_AA)
        if not blink or int(t * 2) % 2 == 0:
            cv2.circle(img, (x + 13, 89), 4, col, -1, cv2.LINE_AA)
        TXT.add(s, (x + 23, 89), 13, col, True, "lm")
        x += w + 8


def draw_results(img, g):
    r = g.round.result
    w, h = 460, 250
    x0, y0, cx = WIN_W // 2 - w // 2, 150, WIN_W // 2
    blend_rect(img, 0, 0, WIN_W, WIN_H, C["ink_deep"], 0.35)
    panel(img, x0, y0, x0 + w, y0 + h, C["teal"])
    TXT.add("ROUND COMPLETE", (cx, y0 + 28), 15, C["teal"], True, "mm")
    TXT.add(MODES[g.round.mode][0], (cx, y0 + 52), 15, C["muted"], False, "mm")
    TXT.add(fmt_time(r["time"]), (cx, y0 + 100), 52, C["paper"], True, "mm")
    if r["new_best"]:
        line, col = "New best time!", C["orange"]
    elif not r["counts"]:
        line, col = "Helped by " + r["why"] + " - not counted for best time", C["muted"]
    else:
        line, col = f"Best {fmt_time(r['best'])}", C["muted"]
    TXT.add(line, (cx, y0 + 146), 17, col, r["new_best"], "mm")
    TXT.add(f"{r['picks']} grab{'s' if r['picks'] != 1 else ''}", (cx, y0 + 172), 14, C["muted"], False, "mm")
    TXT.add("Hold a thumbs up or press R for a new round", (cx, y0 + h - 30), 15, C["paper"], False, "mm")


HELP_HAND = [("Move your hand", "steer the gripper"), ("Pinch or fist", "grab (tuck your thumb for a fist)"),
             ("Open your hand", "put the cube down"), ("Rock sign, hold", "drive mode / arm mode"),
             ("Thumbs up, hold", "new round"),
             ("Peace sign, hold", "auto mode on / off"), ("Point, hold", "pause / resume")]
HELP_KEYS = [("D", "drive mode / arm mode"), ("G", "change game mode"), ("A", "auto mode on / off"), ("T", "teach: record your moves"),
             ("Y", "replay what you taught"), ("Space", "pause / resume"), ("R", "new round"),
             ("U", "tutorial"), ("N", "sound on / off"), ("V", "auto view on / off"),
             ("K", "switch camera"), ("P", "connect phone camera"), ("M  O", "mirror / rotate picture"),
             ("Q", "quit")]


def draw_help(img):
    w, h = 960, 620
    x0, y0 = WIN_W // 2 - w // 2, WIN_H // 2 - h // 2
    blend_rect(img, 0, 0, WIN_W, WIN_H, C["ink_deep"], 0.6)
    panel(img, x0, y0, x0 + w, y0 + h, C["orange"])
    TXT.add("How to play", (x0 + 32, y0 + 22), 28, C["paper"], True)
    TXT.add("H or Esc to close", (x0 + w - 32, y0 + 32), 14, C["muted"], False, "rt")
    cv2.line(img, (x0 + 32, y0 + 66), (x0 + 72, y0 + 66), C["orange"], 3)
    lx, rx, ty = x0 + 32, x0 + 500, y0 + 90
    TXT.add("YOUR HAND", (lx, ty), 13, C["orange"], True)
    for i, (a, b) in enumerate(HELP_HAND):
        TXT.add(a, (lx, ty + 28 + 28 * i), 16, C["paper"], True)
        TXT.add(b, (lx + 160, ty + 29 + 28 * i), 15, C["muted"])
    my = ty + 28 + 28 * len(HELP_HAND) + 16
    TXT.add("GAME MODES", (lx, my), 13, C["orange"], True)
    for i, m in enumerate(MODE_ORDER):
        name, goal = MODES[m]
        TXT.add(name, (lx, my + 28 + 26 * i), 16, C["paper"], True)
        TXT.add(goal, (lx + 100, my + 29 + 26 * i), 14, C["muted"])
    TXT.add("KEYS", (rx, ty), 13, C["orange"], True)
    for i, (k_, b) in enumerate(HELP_KEYS):
        y = ty + 24 + 29 * i
        kw = max(26, TXT.width(k_, 14, True) + 14)
        cv2.rectangle(img, (rx, y), (rx + kw, y + 24), C["graphite"], -1)
        cv2.rectangle(img, (rx, y), (rx + kw, y + 24), C["graphite_light"], 1, cv2.LINE_AA)
        cv2.line(img, (rx + 2, y + 23), (rx + kw - 2, y + 23), C["orange"], 2)
        TXT.add(k_, (rx + kw // 2, y + 12), 14, C["paper"], True, "mm")
        TXT.add(b, (rx + 80, y + 12), 15, C["muted"], False, "lm")
    TXT.add("Drive mode: hand up = forward, down = back, left / right = turn, middle = stop. Hold gestures still about 1 s.",
            (x0 + w // 2, y0 + h - 30), 14, C["paper"], False, "mm")


def draw_hud(img, g, cam, fps, ai_fps, t, cam_info=None):
    ctrl, world = g.ctrl, g.world
    if g.paused and not g.help:                               # paused: dim the scene
        blend_rect(img, 0, 0, WIN_W, WIN_H, C["ink_deep"], 0.45)
        TXT.add("Paused", (WIN_W // 2, WIN_H // 2 - 20), 44, C["paper"], True, "mm")
        TXT.add("Point again or press Space to resume", (WIN_W // 2, WIN_H // 2 + 24), 16, C["muted"], False, "mm")

    # title
    TXT.add("Arm Lab", (24, 18), 30, C["paper"], True)
    TXT.add("Hand-controlled robot arm", (25, 54), 15, C["muted"])
    cv2.line(img, (24, 80), (64, 80), C["orange"], 3)

    # left column: tutorial step or the round
    if g.tut.active:
        draw_tutorial(img, g, 16, 96, 388, 256)
    else:
        draw_mission(img, g, 16, 96, 388, 246)
    draw_minimap(img, g, 16, 268, 196)

    # camera card (top right)
    bw, bh, pad = 340, 255, 10
    x0, y0 = WIN_W - bw - 2 * pad - 16, 16
    ph, pw, xo = bh, bw, x0 + pad
    if cam is not None:
        ch, cw = cam.shape[:2]
        kk = min(bw / cw, bh / ch)
        pw, ph = int(cw * kk), int(ch * kk)
        xo = x0 + pad + (bw - pw) // 2
    status = cam_info[1] if cam_info else ""
    lines = wrap(status, 40) if status else []
    card_h = pad + ph + 130 + 20 * len(lines)
    panel(img, x0, y0, x0 + bw + 2 * pad, y0 + card_h)
    seen = ctrl.visible(t)
    if cam is not None:
        img[y0 + pad:y0 + pad + ph, xo:xo + pw] = cv2.resize(cam, (pw, ph), interpolation=cv2.INTER_LINEAR)
        # labels for the control box, mapped from the camera picture
        fx0, fx1 = xo + int(BOX_X[0] * pw), xo + int(BOX_X[1] * pw)
        fy0, fy1 = y0 + pad + int(BOX_Y[0] * ph), y0 + pad + int(BOX_Y[1] * ph)
        TXT.add("far", ((fx0 + fx1) // 2, fy0 - 9), 12, C["paper"], False, "mm")
        TXT.add("near", ((fx0 + fx1) // 2, fy1 + 9), 12, C["paper"], False, "mm")
        if seen and ctrl.g_pose is not None and ctrl.det is not None:     # gesture ring on your hand
            hc = (int(xo + ctrl.det["c"][0] * pw), int(y0 + pad + ctrl.det["c"][1] * ph))
            rr = int(np.clip(ctrl.det["size"] * pw * 1.6, 22, 70))
            col = C["teal"] if ctrl.g_fired else C["orange"]
            cv2.circle(img, hc, rr, C["graphite_light"], 2, cv2.LINE_AA)
            cv2.ellipse(img, hc, (rr, rr), -90, 0, int(360 * ctrl.g_progress), col, 4, cv2.LINE_AA)
            bar(img, xo + 4, y0 + pad + ph - 5, xo + pw - 4, ctrl.g_progress, col)
            label = GESTURE_NAMES[ctrl.g_pose] + ": " + gesture_action(g, ctrl.g_pose)
            lw_ = TXT.width(label, 14, True) + 16
            ly = max(y0 + pad + 4, hc[1] - rr - 30)
            lx = min(max(hc[0] - lw_ // 2, xo), xo + pw - lw_)
            blend_rect(img, lx, ly, lx + lw_, ly + 22, C["ink_deep"], 0.85)
            TXT.add(label, (lx + lw_ // 2, ly + 11), 14, col, True, "mm")
    else:
        blend_rect(img, xo, y0 + pad, xo + pw, y0 + pad + ph, C["ink_deep"], 0.9)
        TXT.add("Waiting for the camera...", (xo + pw // 2, y0 + pad + ph // 2), 16, C["muted"], False, "mm")
    ty = y0 + pad + ph + 14
    cv2.circle(img, (x0 + pad + 6, ty + 10), 5, C["teal"] if seen else C["coral"], -1, cv2.LINE_AA)
    TXT.add("Hand tracked" if seen else "Show your hand to the camera", (x0 + pad + 18, ty + 1), 16,
            C["paper"], True)
    if seen:
        TXT.add("grabbing" if ctrl.closed else POSE_TEXT.get(ctrl.pose, ""),
                (x0 + bw + pad, ty + 2), 14, C["teal"] if ctrl.closed else C["muted"], False, "rt")
    if cam_info:
        name, _, mirror, rot = cam_info
        extra = ("" if mirror else ", not mirrored") + (f", rotated {rot}°" if rot else "")
        TXT.add(f"{name}{extra}", (x0 + pad, ty + 26), 14, C["muted"])
        TXT.add("K camera · P phone · M mirror · O rotate", (x0 + pad, ty + 48), 13, C["muted"])
        for i, line in enumerate(lines):
            TXT.add(line, (x0 + pad, ty + 112 + 20 * i), 14, C["yellow"])
    TXT.add("Rock sign: drive / arm mode \u00b7 Point: pause", (x0 + pad, ty + 70), 13, C["paper"])
    TXT.add("Thumbs up: new round \u00b7 Peace: auto mode", (x0 + pad, ty + 90), 13, C["paper"])

    # status card (bottom left)
    sx0, sy1 = 16, WIN_H - 58
    sy0 = sy1 - 150
    panel(img, sx0, sy0, sx0 + 372, sy1)
    grip_gauge(img, sx0 + 62, sy0 + 76, 40, ctrl.grip, ctrl.closed)
    holding = world.held is not None
    TXT.add("Holding a cube" if holding else "Gripper empty", (sx0 + 124, sy0 + 20), 18,
            C["teal"] if holding else C["paper"], True)
    if g.pilot is not None:
        l1, l2 = "Auto mode: the robot is playing", "Peace sign or A to take over"
    elif g.drive:
        l1, l2 = "Drive: hand up = go, sides = turn", "Rock sign or D for arm mode"
    elif g.replaying:
        l1, l2 = "Replaying the moves you taught", "Y to stop the replay"
    elif g.rec.recording:
        l1, l2 = "Teaching: your moves are recorded", "T to finish, then Y to replay"
    else:
        l1, l2 = "Move your hand to steer", "Pinch or fist to grab, open to drop"
    TXT.add(l1, (sx0 + 124, sy0 + 50), 14, C["paper"] if g.pilot or g.replaying or g.rec.recording or g.drive else C["muted"])
    TXT.add(l2, (sx0 + 124, sy0 + 70), 14, C["muted"])
    TXT.add(f"Auto view {'on' if g.auto_view else 'off'} · Sound {'on' if g.sound.on else 'off'}",
            (sx0 + 124, sy0 + 112), 13, C["muted"])

    # keycaps along the bottom
    x = 16
    for key, label in [("D", "Drive"), ("G", "Mode"), ("A", "Auto"), ("T", "Teach"), ("Y", "Replay"), ("Space", "Pause"),
                       ("H", "Help"), ("R", "New round"), ("Q", "Quit")]:
        x = keycap(img, x, WIN_H - 40, key, label)
    TXT.add(f"{fps:.0f} fps • tracking {ai_fps:.0f} fps", (WIN_W - 18, WIN_H - 28), 13, C["muted"], False, "rt")

    draw_badges(img, g, t)

    # toast message (top centre)
    msg = g.toast_text if t < g.toast_until else ""
    if msg:
        w = TXT.width(msg, 20, True) + 48
        tx0 = WIN_W // 2 - w // 2
        blend_rect(img, tx0, 22, tx0 + w, 66, C["ink"], 0.9)
        cv2.rectangle(img, (tx0, 22), (tx0 + w, 66), C["orange"], 1, cv2.LINE_AA)
        cv2.line(img, (tx0, 66), (tx0 + w, 66), C["orange"], 3)
        TXT.add(msg, (WIN_W // 2, 44), 20, C["paper"], True, "mm")

    # pop-ups on top of everything (render the text so far first, so the pop-up covers it)
    if g.help:
        TXT.render(img)
        draw_help(img)
    elif g.round.done and g.round.result is not None and not g.tut.active:
        TXT.render(img)
        draw_results(img, g)


# =====================================================================
# 8. MAIN LOOP
# =====================================================================
def load_settings():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(**kw):
    try:
        data = load_settings()
        data.update(kw)
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def draw_text_box(img, title, text, hint):
    x0, y0, w, h = WIN_W // 2 - 310, WIN_H // 2 - 90, 620, 180
    blend_rect(img, 0, 0, WIN_W, WIN_H, C["ink_deep"], 0.55)       # dim everything behind
    panel(img, x0, y0, x0 + w, y0 + h, C["orange"])
    TXT.add(title, (x0 + 24, y0 + 20), 22, C["paper"], True)
    cv2.rectangle(img, (x0 + 24, y0 + 62), (x0 + w - 24, y0 + 112), C["ink_deep"], -1)
    cv2.rectangle(img, (x0 + 24, y0 + 62), (x0 + w - 24, y0 + 112), C["orange"], 1, cv2.LINE_AA)
    TXT.add(text, (x0 + 40, y0 + 87), 24, C["paper"], True, "lm")
    if int(time.time() * 2) % 2:
        cx = x0 + 42 + TXT.width(text, 24, True)
        cv2.line(img, (cx, y0 + 74), (cx, y0 + 100), C["orange"], 2)
    TXT.add(hint, (x0 + 24, y0 + 132), 14, C["muted"])


def type_key(buf, key):
    """Text box typing. Returns (new_text, 'done' | 'cancel' | None)."""
    if key in (13, 10):
        return buf, "done"
    if key == 27:
        return buf, "cancel"
    if key in (8, 127):
        return buf[:-1], None
    if 32 < key < 127 and (chr(key).isalnum() or chr(key) in ".:/-_"):
        return (buf + chr(key))[:60], None
    return buf, None



class Game:
    """Everything that happens in the app, without the camera and window (so it can be tested)."""

    def __init__(self, cfg=None, seed=None):
        self.cfg = cfg if cfg is not None else {}
        self.stats = load_json(STATS_FILE)
        self.world = World(self.cfg.get("mode", START_MODE), seed)
        self.ctrl, self.auto, self.view, self.base = HandControl(), AutoGrip(), View(), Base()
        self.pilot, self.rec, self.tut = None, Recorder(), Tutorial()
        self.sound = Sound(self.cfg.get("sound", SOUND))
        self.round = Round(self.world.mode)
        self.auto_view, self.paused, self.help, self.replaying = AUTO_VIEW, False, False, False
        self.drive = False                                  # drive mode (hand = wheels) or arm mode (hand = arm)
        self.sim_t = 0.0                                    # game clock: stops while paused
        self.q = solve_ik(*self.auto.target(self.ctrl.xy, self.world))
        self.grip_gap, self.was_closed, self.closed_now, self.lock = 0.035, False, False, None
        self.toast_text, self.toast_until, self.blocked_said = "", 0.0, -9.0
        if not self.cfg.get("tutorial_done"):
            self.tut.start()
            self.round.practice = True

    # ---------- small helpers ----------
    def say(self, text, now, secs=1.4):
        self.toast_text, self.toast_until = text, now + secs

    def remember(self, **kw):
        self.cfg.update(kw)
        save_settings(**kw)

    def grasp(self):
        """Gripper position in the world."""
        return self.base.to_world3(forward_kinematics(self.q)["grasp"])

    def hand_over(self):
        """Auto mode / replay ends: give control back to your hand (in arm mode) without surprises."""
        self.drive = False
        self.base.stop()
        if self.auto.phase == "down_pick":                  # it was about to grab: cancel
            self.auto.set("up", self.sim_t)
        elif self.world.held is not None and self.auto.phase in ("travel", "up"):
            self.auto.set("down_place", self.sim_t, self.grasp()[:2], ("place_start", self.world.held))  # put it down here
        self.was_closed = self.ctrl.closed

    def stop_pilot(self, now, msg):
        self.pilot = None
        self.hand_over()
        self.say(msg, now, 2.0)

    # ---------- actions (keys and gestures) ----------
    def new_round(self, now, msg="New round"):
        if self.rec.recording:
            self.rec.stop()
        self.replaying = False
        self.world.reset()
        self.base.reset()
        self.auto.set("travel", self.sim_t)
        self.auto.events.clear()
        finished = self.tut.on_event("new_round") == "finished"
        self.round = Round(self.world.mode, practice=self.tut.active)
        if self.pilot is not None:
            self.pilot = AutoPilot(self.grasp())
            self.round.assisted = "auto mode"
        else:
            self.drive = False
        self.was_closed = self.pilot.closed if self.pilot is not None else self.ctrl.closed
        self.paused = False
        self.sound.play("start")
        if finished:
            self.remember(tutorial_done=True)
            msg = "Tutorial complete - go!"
        if msg:
            self.say(msg, now)

    def set_mode(self, mode, now):
        self.world.set_mode(mode)
        self.remember(mode=self.world.mode)
        self.new_round(now, f"Mode: {MODES[self.world.mode][0]}")

    def toggle_drive(self, now):
        if self.pilot is not None or self.replaying:
            self.say("Auto mode / replay is driving", now)
            return
        if not self.drive:
            if self.auto.phase != "travel":
                self.say("Wait for the arm to finish", now)
                return
            self.drive = True
            self.sound.play("click")
            self.say("Drive mode - hand up = go, sideways = turn", now, 2.0)
            if self.tut.on_event("drive_on") == "next":
                self.say("Nice! Now drive to the cubes", now, 2.0)
        else:
            self.drive = False
            self.base.stop()
            self.was_closed = self.ctrl.closed
            self.sound.play("click")
            holding = self.world.held is not None
            self.say("Arm mode - pinch, then open to drop" if holding and not self.ctrl.closed else "Arm mode", now, 2.0)
            if self.tut.on_event("drive_off") == "next":
                self.say("Nice! Now pick up a cube", now, 2.0)

    def toggle_auto(self, now):
        if self.pilot is not None:
            self.pilot = None
            self.hand_over()
            self.say("Auto mode off - your turn", now)
            return
        if self.replaying:
            self.replaying = False
        if self.rec.recording:
            self.rec.stop()
        if self.round.done:
            self.new_round(now, None)
        self.pilot = AutoPilot(self.grasp(), holding=self.world.held is not None)
        self.was_closed = self.pilot.closed
        if self.auto.phase == "down_pick":
            self.auto.set("up", self.sim_t)
        self.round.assisted = "auto mode"
        self.paused = False
        self.sound.play("gesture")
        self.say("Auto mode on - the robot plays", now)

    def toggle_pause(self, now):
        self.paused = not self.paused
        self.sound.play("click")
        if not self.paused:
            self.say("Go!", now, 1.0)

    def toggle_teach(self, now):
        if self.rec.recording:
            self.rec.stop()
            if self.rec.length() < 1.0:
                self.rec.samples = []
                self.say("Too short - press T and show some moves", now, 2.0)
            else:
                self.say(f"Taught {self.rec.length():.0f} s of moves - press Y to replay", now, 2.5)
            return
        if self.pilot is not None:
            self.pilot = None
            self.hand_over()
        if self.replaying:
            self.replaying = False
            self.hand_over()
        self.rec.start(dict(world=self.world.snapshot(), q=self.q.copy(), base=self.base.snapshot(),
                            drive=self.drive, phase=self.auto.phase,
                            lock=None if self.auto.lock_xy is None else self.auto.lock_xy.copy(),
                            closed=self.was_closed))
        self.paused = False
        self.sound.play("click")
        self.say("Teaching - do your moves, press T to finish", now, 2.5)

    def toggle_replay(self, now):
        if self.replaying:
            self.replaying = False
            self.hand_over()
            self.say("Replay stopped", now)
            return
        if self.rec.recording:
            self.rec.stop()
        if self.rec.length() < 1.0:
            self.say("Nothing taught yet - press T first", now, 2.0)
            return
        self.pilot = None
        s = self.rec.snap
        self.world.restore(s["world"])
        self.base.restore(s["base"])
        self.q, self.drive = s["q"].copy(), s["drive"]
        self.auto.set(s["phase"], self.sim_t, s["lock"])
        self.auto.events.clear()
        self.was_closed = s["closed"]
        self.round = Round(self.world.mode, practice=self.tut.active)
        self.round.assisted = "replay"
        self.rec.start_play()
        self.replaying, self.paused = True, False
        self.sound.play("gesture")
        self.say("Replaying what you taught", now)

    def toggle_tutorial(self, now):
        if self.tut.active:
            self.tut.active = False
            self.remember(tutorial_done=True)
            if not self.round.running:                      # nothing done yet: this round counts
                self.round.practice = False
            self.say("Tutorial skipped - U starts it again", now, 2.0)
        else:
            self.tut.start()
            self.new_round(now, "Tutorial")

    def key(self, k, now):
        """Keys the game handles (the window handles K, P, M, O and Q)."""
        if self.help:
            if k in (27, ord('h')):
                self.help = False
            return
        if k == ord('h'):
            self.help = True
        elif k == ord('d'):
            self.toggle_drive(now)
        elif k == ord('g'):
            self.set_mode(MODE_ORDER[(MODE_ORDER.index(self.world.mode) + 1) % len(MODE_ORDER)], now)
        elif k == ord('a'):
            self.toggle_auto(now)
        elif k == ord('t'):
            self.toggle_teach(now)
        elif k == ord('y'):
            self.toggle_replay(now)
        elif k == 32:
            self.toggle_pause(now)
        elif k == ord('r'):
            self.new_round(now)
        elif k == ord('u'):
            self.toggle_tutorial(now)
        elif k == ord('n'):
            self.sound.on = not self.sound.on
            self.remember(sound=self.sound.on)
            self.say(f"Sound {'on' if self.sound.on else 'off'}", now, 1.0)
        elif k == ord('v'):
            self.auto_view = not self.auto_view
            self.say(f"Auto view {'on' if self.auto_view else 'off'}", now, 1.0)

    # ---------- every frame ----------
    def on_hands(self, hands, fw, fh, now):
        self.ctrl.update(hands, fw, fh, now)
        for gesture in self.ctrl.events:
            if self.help:
                continue
            if gesture == "thumbs_up":
                self.new_round(now)
            elif gesture == "peace":
                self.toggle_auto(now)
            elif gesture == "point":
                self.toggle_pause(now)
            elif gesture == "rock":
                self.toggle_drive(now)
        self.ctrl.events.clear()

    def update(self, dt, now):
        if self.paused or self.help:
            return
        self.sim_t += dt
        t, world, auto, base = self.sim_t, self.world, self.auto, self.base
        replay_over, v_cmd, w_cmd, xy = False, 0.0, 0.0, None

        # who steers: replay, auto mode or your hand (arm mode or drive mode)
        if self.replaying:
            pos, th, xy, closed, self.drive, replay_over = self.rec.sample(dt)
            base.pos, base.th = np.array(pos, float), th
        elif self.pilot is not None:
            v_cmd, w_cmd, status = self.pilot.step(world, base, auto, self.grasp(), dt)
            if status:
                self.stop_pilot(now, "Auto mode: all done" if status == "done" else "Auto mode is stuck - your turn")
                v_cmd = w_cmd = 0.0
                closed = self.ctrl.closed
            else:
                xy, closed, self.drive = self.pilot.xy, self.pilot.closed, self.pilot.drive
        elif self.drive:
            jx, jy = self.ctrl.joy if self.ctrl.visible(now) else (0.0, 0.0)
            v_cmd = jy * DRIVE_SPEED * (1.0 if jy > 0 else 0.6)        # reverse a bit slower
            w_cmd = -jx * TURN_SPEED                                    # hand right = turn right
            closed = self.was_closed                                     # the gripper keeps what it has
        else:
            closed = self.ctrl.closed

        # move the base (only between grabs), stopped by walls and cubes
        if not self.replaying:
            if auto.phase != "travel":
                v_cmd = w_cmd = 0.0
            base.drive(v_cmd, w_cmd, dt, world)
            if base.blocked and self.drive and self.pilot is None and now - self.blocked_said > 2.0:
                self.blocked_said = now
                self.say("Blocked - turn or back up", now, 1.2)

        # where the gripper should go (world): from auto mode / replay, tucked in front while driving, or your hand
        if xy is None:
            xy = base.to_world(CARRY_XY if self.drive else self.ctrl.xy)
            if not self.drive:
                xy, self.lock = aim_assist(xy, world)
        if self.pilot is not None or self.replaying:
            self.lock = aim_assist(xy, world)[1]
        if self.drive or (self.lock is not None and np.linalg.norm(self.lock - xy) > 0.012):
            self.lock = None                                  # only show "locked" when really on it

        self.closed_now = closed
        if closed != self.was_closed:
            auto.on_pinch_change(closed, xy, world, t)
            self.was_closed = closed
        if self.rec.recording and not self.rec.add(dt, base.pos, base.th, xy, closed, self.drive):
            self.say("Teaching full (3 min) - press Y to replay", now, 2.5)

        # move the arm: IK in the base's own frame, eased like servo motors
        local = clamp_workspace(base.to_local3(auto.target(xy, world)))
        self.q = self.q + np.clip((solve_ik(*local) - self.q) * min(1.0, JOINT_RESPONSE * dt),
                                  -MAX_JOINT_SPEED * dt, MAX_JOINT_SPEED * dt)
        auto.step(self.grasp(), base.to_world3(local), world, t)
        want = CUBE + 0.004 if world.held is not None else (0.006 if auto.gripper_closed(world, closed) else 0.035)
        self.grip_gap += (want - self.grip_gap) * min(1.0, 12 * dt)
        world.step(self.grasp(), dt)
        self.view.follow(base.th if self.drive else base.th + self.q[0], base.pos, dt, self.drive, self.auto_view)

        self.handle_events(now)
        r = self.round
        if r.running and not r.done:
            r.time += dt
        if not r.done and world.complete():
            self.finish_round(now)
        if self.tut.check(self.ctrl, world, base, self.drive, now, dt):
            self.sound.play("click")
            self.say("Nice!", now, 1.0)
        if replay_over:
            self.replaying = False
            self.hand_over()
            self.say("Replay finished", now)

    def handle_events(self, now):
        texts = {"pick_start": "Picking up...", "picked": "Got it!", "missed": "Nothing there",
                 "place_start": "Putting down...", "placed": "Placed", "kept": "Kept the cube"}
        events, self.auto.events = self.auto.events, []     # new events made while handling these wait a frame
        for kind, _cube in events:
            self.say(texts[kind], now, 1.2)
            if kind == "pick_start":
                self.round.picks += 1
                if self.world.mode != "free":
                    self.round.running = True
                if self.pilot is not None:                  # auto mode gives up if it stops making progress
                    s = self.world.score()
                    self.pilot.idle = 0 if s > self.pilot.best else self.pilot.idle + 1
                    self.pilot.best = max(self.pilot.best, s)
                    if self.pilot.idle > 6:
                        self.stop_pilot(now, "Auto mode stopped - your turn")
            elif kind == "picked":
                self.sound.play("pick")
            elif kind == "missed":
                self.sound.play("miss")
                if self.pilot is not None:
                    self.pilot.misses += 1
                    if self.pilot.misses >= 3:
                        self.stop_pilot(now, "Auto mode stopped - your turn")
            elif kind == "placed":
                self.sound.play("place")
                if self.tut.on_event("placed") == "next":
                    self.say("Nice! Now the last step", now, 1.5)

    def finish_round(self, now):
        r = self.round
        r.done, r.running = True, False
        why = "the tutorial" if r.practice else r.assisted
        counts = not why
        best = self.stats.get("best", {}).get(r.mode)
        new_best = counts and (best is None or r.time < best)
        if counts:
            self.stats.setdefault("best", {})
            self.stats["rounds"] = self.stats.get("rounds", 0) + 1
            if new_best:
                self.stats["best"][r.mode] = round(r.time, 2)
            save_json(STATS_FILE, self.stats)
        r.result = dict(time=r.time, best=best, new_best=new_best, counts=counts, why=why, picks=r.picks)
        self.sound.play("win")
        self.say("New best time!" if new_best else "Round complete!", now, 2.0)
        if self.pilot is not None:
            self.pilot = None
            self.drive = False
            self.base.stop()
            self.was_closed = self.ctrl.closed

    def draw(self, cam, fps, ai_fps, now, cam_info=None):
        img = draw_scene(self.view, self.q, self.grip_gap, self.world, self.lock, self.base, self.drive)
        TXT.render(img)                                      # scene labels go under the HUD cards
        draw_hud(img, self, cam, fps, ai_fps, now, cam_info)
        return img


def main():
    ap = argparse.ArgumentParser(description="Arm Lab - hand-controlled robot arm")
    ap.add_argument("--cam", type=int, help="camera number to use (0 = laptop)")
    ap.add_argument("--phone", help="phone camera address, e.g. 192.168.1.5:8080")
    args = ap.parse_args()

    cfg = load_settings()
    phone = args.phone or cfg.get("phone", PHONE_ADDRESS)
    if args.phone:
        start = phone_url(args.phone)
    elif args.cam is not None:
        start = args.cam
    else:
        start = cfg.get("source", CAM_INDEX)
        if isinstance(start, str) and not start:
            start = CAM_INDEX

    print("Looking for cameras...")
    local = find_local_cameras()
    print("Cameras found:", local or "none", "| phone:", phone or "not set (press P in the app)")
    if not local and not phone and isinstance(start, int):
        print("No camera found. Connect a camera, or start with:  py -3.14 hand_arm.py --phone 192.168.1.5:8080")
        return

    if isinstance(start, int) and start not in local:
        start = local[0] if local else phone_url(phone)

    def sources():
        return list(local) + ([phone_url(phone)] if phone else [])

    cam = Camera(start)
    tracker = HandTracker(cam)
    tracker.mirror = cfg.get("mirror", MIRROR)
    tracker.rotate = cfg.get("rotate", ROTATE)
    game = Game(cfg)
    typing, buf = False, phone
    last_t, fps, last = 0.0, 0.0, time.perf_counter()

    cv2.namedWindow("Robot Arm", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Arm", WIN_W, WIN_H)
    cv2.setWindowTitle("Robot Arm", "Arm Lab - hand-controlled robot arm")

    while True:
        now = time.perf_counter()
        dt, last = min(now - last, 0.05), now
        fps = 0.9 * fps + 0.1 / max(dt, 1e-3)
        res, frame = tracker.get()
        if res is not None and res[1] != last_t:             # new tracking result
            hands, last_t = res
            game.on_hands(hands, frame.shape[1], frame.shape[0], now)
        if frame is not None:
            frame = frame.copy()
            draw_camera_overlay(frame, game.ctrl, now, game.drive)
        game.update(dt, now)

        cam_info = (source_name(cam.source if cam.source is not None else start), cam.status,
                    tracker.mirror, tracker.rotate)
        img = game.draw(frame, fps, tracker.fps, now, cam_info)
        if typing:
            TXT.render(img)                                  # everything else first, so the box covers it
            draw_text_box(img, "Connect your phone camera", buf,
                          "Type the address the phone app shows, like 192.168.1.5:8080. Enter connects, Esc cancels.")
        TXT.render(img)                                      # all text, in one smooth pass
        cv2.imshow("Robot Arm", img)

        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty("Robot Arm", cv2.WND_PROP_VISIBLE) < 1:
            break
        if typing:                                          # phone address box takes all keys
            if key != 255:
                buf, done = type_key(buf, key)
                if done == "done" and buf.strip():
                    phone, typing = buf.strip(), False
                    cam.switch(phone_url(phone))
                    save_settings(phone=phone, source=phone_url(phone))
                    game.say("Connecting to your phone...", now, 2.0)
                elif done:
                    typing = False
            continue
        if key == 255:
            continue
        if key == ord('q'):
            break
        elif key == ord('k'):
            srcs = sources()
            if srcs:
                cur = cam.source if cam.source in srcs else None
                nxt = srcs[(srcs.index(cur) + 1) % len(srcs)] if cur is not None else srcs[0]
                cam.switch(nxt)
                save_settings(source=nxt)
                game.say(f"Switching to {source_name(nxt)}", now, 1.5)
            else:
                game.say("No other camera found - press P to add your phone", now, 2.0)
        elif key == ord('p'):
            typing, buf = True, phone
        elif key == ord('m'):
            tracker.mirror = not tracker.mirror
            save_settings(mirror=tracker.mirror)
        elif key == ord('o'):
            tracker.rotate = (tracker.rotate + 90) % 360
            save_settings(rotate=tracker.rotate)
        else:
            game.key(key, now)

    tracker.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
