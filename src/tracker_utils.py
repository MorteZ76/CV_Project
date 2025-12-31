"""
tracker_utils.py
----------------
Core utilities, classes, and algorithms for the Object Tracker.

This module encapsulates the foundational logic required for the tracking pipeline, including:
1. Hyperparameters and Configuration: Centralized tuning for tracking dynamics.
2. Geometry & Math: Helper functions for bounding box operations and IoU calculations.
3. Visualization: Tools for drawing annotations, trajectories, and velocity vectors.
4. Core Tracking Logic: Implementation of the Kalman Filter, Track state management, 
   and the ByteTrack-like association algorithm extended for drone footage.
5. Evaluation Metrics: Implementation of HOTA (Higher Order Tracking Accuracy) and MSE metrics.
"""

import os
import cv2
import math
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Any, Set

# Local project utils
import utils

# --- Third-Party Dependencies ---
# Scipy is required for the Hungarian Algorithm (linear assignment)
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("scipy not found. Please install: pip install scipy")

# FilterPy is optional but recommended for Kalman Filtering. 
# A dummy fallback is provided if missing.
HAS_FILTERPY = True
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    HAS_FILTERPY = False


# ==============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==============================================================================

# --- Detection / NMS ---
DET_CONF_THRES = 0.55  # Minimum confidence to consider a detection valid
DET_IOU_NMS = 0.50     # IoU threshold for Non-Maximum Suppression
AGNOSTIC_NMS = True    # If True, NMS is applied across all classes (class-agnostic)

# --- Revival / Gating ---
# Parameters for recovering "lost" tracks based on motion predictions
USE_IOU_GATES = False
REVIVE_REQUIRE_SAME_CLASS = False
REVIVE_USE_IOU = False
REVIVE_IOU_THRES = 0.12
REVIVE_CENTER_MULT = 0.9
REVIVE_CENTER_MIN = 50.0

# --- Tracking Dynamics ---
PRED_EXPAND = 1.05        # Expansion factor for predicted boxes during visualization
PRED_HORIZON_LOST = 100   # How many frames to project a lost track into the future
LOST_SIZE_INFLATE = 1.05  # Inflation factor for lost track matching to account for uncertainty
SIZE_EMA = 0.25           # Exponential Moving Average factor for smooth box size updates
SPEED_CAP_ABS = 4.0       # Absolute max speed (pixels/frame) to prevent exploding Kalman states
SPEED_CAP_DIAG_FRAC = 0.6 # Max speed as a fraction of the object's diagonal size
MIN_SPEED = 0.2           # Minimum speed to consider an object moving
LOST_KEEP = 150           # Max frames to keep a track in "lost" state before deletion
DISABLE_LOW_WHEN_LOST = True
CENTER_DIST_GATE = 70     # Distance threshold (pixels) for spatial gating

# --- Data Association (ByteTrack Logic) ---
BYTE_HIGH_THRES = 0.68    # Threshold for "High confidence" detections
BYTE_LOW_THRES = 0.45     # Threshold for "Low confidence" detections
IOU_GATE = 0.08           # Minimum IoU required to match a track to a detection
MIN_HITS = 3              # Minimum consecutive hits to confirm a new track

# --- Class Locking ---
# Logic to stabilize class prediction over time (majority voting)
CLASS_LOCK_CONF = 0.80
CLASS_LOCK_COUNT = 10
CLASS_HIST_MAX = 50

# --- Kinematics & Trajectory ---
KINEMA_WINDOW = 10              # Window size for velocity smoothing
AREA_WINDOW = 10                # Window size for area smoothing
AREA_W_MEAN = 0.7               # Weight for mean area in smoothing
AREA_W_LAST = 0.3               # Weight for most recent area
TRAJ_MAX_LEN = 2000             # Maximum history length for drawing trajectories
MISS_FRAMES_TO_DROP_PATH = 100  # Drop visualization path if object lost for this long

# --- Border Logic ---
# Special handling for objects entering/leaving the frame
BORDER_RATIO = 0.1              # Margin size relative to frame dimensions
BORDER_REVIVE_PENALTY = 2.0     # Cost penalty for reviving tracks near the border
ANGLE_CHANGE_THRES = 45.0       # Max allowed angle change for border revival
PREFER_NEW_AT_BORDER = True     # If True, prefer creating new tracks over matching lost ones at borders

# --- Anti-Jerk (Size Consistency) ---
SIZE_CHANGE_MAX = 2.0   # Max allowed relative size change between frames
SIZE_CHANGE_MIN = 0.3   # Min allowed relative size change

# --- Adaptive Relaxation Scales ---
# Dynamic thresholds that relax as a track remains lost longer
CENTER_RELAX_SCALES = [0.2, 0.6, 1.0]
IOU_RELAX_SCALES = [0.2, 0.5, 0.8]
SIZE_MIN_FLOORS = [0.3, 0.2, 0.1]
SIZE_MAX_INCREASES = [1.0, 2.0, 3.0]
MAHA_RELAX_SCALES = [0.5, 1.2, 2.0]
CHI2_POS_99 = 9.21  # Chi-squared threshold for Mahalanobis gating (99% confidence)

# --- Visualization Constants ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
ARROW_LEN = 35
SHOW_UNITS = "px/frame"


# ==============================================================================
# 2. GEOMETRY & MATH HELPERS
# ==============================================================================

def calculate_box_area(box: np.ndarray) -> float:
    """Computes area of a box [x1, y1, x2, y2]."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h

def get_blended_ref_area(hist: deque, w_mean: float = AREA_W_MEAN, w_last: float = AREA_W_LAST) -> Optional[float]:
    """Computes weighted average area from history to stabilize tracking size."""
    if not hist: return None
    a_mean = float(np.mean(hist))
    a_last = float(hist[-1])
    return w_mean * a_mean + w_last * a_last

def get_track_box_current(t: 'Track') -> np.ndarray:
    """Retrieves current bounding box [x1, y1, x2, y2] from Kalman Filter state."""
    if HAS_FILTERPY:
        return cxcywh_to_xyxy(t.kf.x[:4, 0])
    return cxcywh_to_xyxy(t.kf.state)

def compute_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the Intersection over Union (IoU) matrix between two sets of boxes.
    
    Args:
        a: Array of shape (N, 4) containing [x1, y1, x2, y2].
        b: Array of shape (M, 4) containing [x1, y1, x2, y2].
        
    Returns:
        np.ndarray: IoU matrix of shape (N, M).
    """
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
        
    # Broadcast coordinates to shape (N, M)
    x11, y11 = a[:, 0][:, None], a[:, 1][:, None]
    x12, y12 = a[:, 2][:, None], a[:, 3][:, None]
    x21, y21 = b[:, 0][None, :], b[:, 1][None, :]
    x22, y22 = b[:, 2][None, :], b[:, 3][None, :]
    
    # Calculate overlap
    inter_w = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
    inter_h = np.maximum(0, np.minimum(y12, y22) - np.maximum(y11, y21))
    inter = inter_w * inter_h
    
    # Calculate union
    area_a = np.maximum(0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = np.maximum(0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    
    union = area_a + area_b - inter
    # Avoid division by zero
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)

def is_near_border(bbox: np.ndarray, frame_shape: Tuple[int, int], border_ratio: float = 0.04) -> bool:
    """
    Checks if a bounding box is within the defined border margin of the frame.
    Useful for determining if an object is entering or leaving the scene.
    """
    x1, y1, x2, y2 = bbox
    H, W = frame_shape[:2]
    bx, by = W * border_ratio, H * border_ratio
    return (x1 <= bx or y1 <= by or x2 >= (W - bx) or y2 >= (H - by))

def xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    """Converts [x1, y1, x2, y2] (Top-Left/Bottom-Right) to [cx, cy, w, h] (Center/Size)."""
    x1, y1, x2, y2 = box
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return np.array([x1 + w/2.0, y1 + h/2.0, w, h], dtype=np.float32)

def cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Converts [cx, cy, w, h] (Center/Size) to [x1, y1, x2, y2] (Top-Left/Bottom-Right)."""
    cx, cy, w, h = box
    return np.array([cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0], dtype=np.float32)

def expand_box(box: np.ndarray, scale: float) -> np.ndarray:
    """Inflates a box from its center by a scale factor."""
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)

def angle_to_compass(deg: float) -> str:
    """Converts degrees (0-360) to 8-point compass direction string (N, NE, E, ...)."""
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((deg % 360) + 22.5) // 45) % 8
    return dirs[idx]


# ==============================================================================
# 3. VISUALIZATION HELPERS
# ==============================================================================

def draw_box_with_id(frame: np.ndarray, bbox: np.ndarray, cls: int, tid: int, 
                     conf: Optional[float] = None, color: Tuple[int, int, int] = (0, 255, 0), 
                     label_map: Optional[Dict] = None) -> None:
    """Draws a bounding box with Class Name, ID, and optionally Confidence."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    cls_name = label_map.get(cls, str(cls)) if label_map else str(cls)
    text = f"{cls_name} ID:{tid}" + (f" {conf:.2f}" if conf is not None else "")
    
    # Draw label above the box
    cv2.putText(frame, text, (x1, max(0, y1 - 5)), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

def draw_trajectory(frame: np.ndarray, pts: List[Tuple[int, int]], color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """Draws the track history as a polyline connected segments."""
    if len(pts) < 2: return
    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_arr], False, color, 2)

def draw_velocity_arrow(frame: np.ndarray, cx: float, cy: float, deg: float, 
                        length: int = ARROW_LEN, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """Draws an arrow indicating motion direction based on angle and center point."""
    rad = math.radians(deg)
    ex = int(cx + length * math.cos(rad))
    ey = int(cy + length * math.sin(rad))
    cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey), color, 2, tipLength=0.3)


# ==============================================================================
# 4. NMS & DATA LOADING
# ==============================================================================

def perform_iou_nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Standard IoU-based Non-Maximum Suppression (NMS) implementation."""
    if len(boxes) == 0: return []
    # Sort boxes by confidence score in descending order
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        
        # Calculate intersection with remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-9)
        
        # Keep boxes that have IoU less than threshold
        order = order[1:][iou <= iou_thres]
    return keep

def apply_nms(rows: np.ndarray, conf_thres: float, iou_thres: float, agnostic: bool = False) -> List[Dict]:
    """
    Filters detections by confidence and runs NMS.
    
    Args:
        rows: Raw detection array [frame, x1, y1, x2, y2, score, cls].
        conf_thres: Minimum confidence to retain a box.
        iou_thres: IoU threshold for overlap removal.
        agnostic: If True, suppresses overlap regardless of class ID.
    """
    if rows.size == 0: return []
    # Filter by confidence
    fr = rows[rows[:, 5] >= conf_thres]
    if fr.size == 0: return []
    
    out = []
    if agnostic:
        boxes = fr[:, 1:5].astype(np.float32)
        scores = fr[:, 5].astype(np.float32)
        keep = perform_iou_nms(boxes, scores, iou_thres)
        for i in keep:
            out.append({"bbox": boxes[i], "conf": float(scores[i]), "cls": int(fr[i, 6])})
    else:
        # Per-class NMS
        for c in np.unique(fr[:, 6]).astype(int):
            sc = fr[fr[:, 6] == c]
            boxes = sc[:, 1:5].astype(np.float32)
            scores = sc[:, 5].astype(np.float32)
            keep = perform_iou_nms(boxes, scores, iou_thres)
            for i in keep:
                out.append({"bbox": boxes[i], "conf": float(scores[i]), "cls": int(c)})
    return out

def load_cached_detections(path: str) -> Dict[int, np.ndarray]:
    """
    Loads detections from Parquet or CSV into a memory-efficient dictionary.
    Keys are frame indices, values are numpy arrays of detections.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Detection file not found: {path}")

    try:
        df = pd.read_parquet(path)
    except Exception:
        df = pd.read_csv(path)
    
    cols = ["frame", "x1", "y1", "x2", "y2", "score", "cls"]
    if any(c not in df.columns for c in cols):
        raise ValueError(f"Detection file missing required columns: {cols}")
        
    per_frame = defaultdict(list)
    for row in df.itertuples():
        # Store as list first for speed
        per_frame[int(row.frame)].append([
            int(row.frame), float(row.x1), float(row.y1),
            float(row.x2), float(row.y2), float(row.score), int(row.cls)
        ])
    # Convert lists to numpy arrays
    for k in list(per_frame.keys()):
        per_frame[k] = np.array(per_frame[k], dtype=np.float32)
    return per_frame


# ==============================================================================
# 5. CORE TRACKER CLASSES
# ==============================================================================

def create_kalman_filter(initial_cxcywh: np.ndarray) -> Any:
    """
    Initializes a Kalman Filter for constant velocity motion model.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Meas:  [cx, cy, w, h]
    """
    if HAS_FILTERPY:
        kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        # Transition matrix (F)
        kf.F = np.eye(8, dtype=np.float32)
        for i in range(4): kf.F[i, i + 4] = dt
        # Measurement matrix (H)
        kf.H = np.zeros((4, 8), dtype=np.float32)
        kf.H[0, 0] = kf.H[1, 1] = kf.H[2, 2] = kf.H[3, 3] = 1
        # Covariance setup
        kf.P *= 10.0
        kf.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        kf.Q = np.eye(8, dtype=np.float32) * 1.0
        kf.x[:4, 0] = initial_cxcywh.reshape(4)
        return kf
    else:
        # Dummy fallback if filterpy is not installed
        class DummyKF:
            def __init__(self, s): self.state = s.copy()
            def predict(self): return self.state
            def update(self, z): self.state = z.copy()
            @property
            def x(self): return np.concatenate([self.state, np.zeros(4, np.float32)])[:, None]
        return DummyKF(initial_cxcywh.astype(np.float32))

class Track:
    """
    Represents a single tracked object. 
    Maintains state, history, and the Kalman Filter instance.
    """
    _next_id = 1
    def __init__(self, bbox_xyxy, cls_id, conf, frame_idx):
        self.id = Track._next_id
        Track._next_id += 1
        self.cls = int(cls_id)
        self.conf = float(conf)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.matched_this_frame = False
        self.high_conf_match = False
        
        # Initialize KF
        cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, np.float32))
        self.kf = create_kalman_filter(cxcywh)
        if HAS_FILTERPY: self.kf.predict()
        
        # History buffers
        self.history = deque(maxlen=TRAJ_MAX_LEN)
        cx, cy, _, _ = cxcywh
        self.history.append((int(cx), int(cy)))
        self.det_hist = deque(maxlen=KINEMA_WINDOW)
        self.det_hist.append((float(cx), float(cy), int(frame_idx)))

    def predict(self):
        """Advances the Kalman Filter state."""
        if HAS_FILTERPY: self.kf.predict()
        return cxcywh_to_xyxy(self.kf.x[:4, 0]) if HAS_FILTERPY else cxcywh_to_xyxy(self.kf.state)

    def update(self, bbox_xyxy, cls_id, conf, frame_idx):
        """Updates the track with a matched detection."""
        cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, np.float32))
        self.kf.update(cxcywh)
        self.conf = float(conf)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        cx, cy, _, _ = (self.kf.x[:4, 0] if HAS_FILTERPY else self.kf.state)
        self.history.append((int(cx), int(cy)))
        self.det_hist.append((float(cx), float(cy), int(frame_idx)))

    def mark_missed(self):
        """Increments the counter for missed frames."""
        self.time_since_update += 1
        self.age += 1

class ByteTrackLike:
    """
    ByteTrack-based tracker with motion extensions for drone footage.
    
    Key Features:
    - Two-stage matching: High confidence first, then Low confidence.
    - Revival Logic: Uses future trajectory predictions to recover lost tracks.
    - Adaptive Gating: Relaxes constraints for tracks that have been lost longer.
    """
    def __init__(self, iou_gate=IOU_GATE, min_hits=MIN_HITS):
        self.iou_gate = iou_gate
        self.min_hits = min_hits
        self.tracks = []
        self.lost_ids = set()
        self.area_hist = {}
        self.future = defaultdict(dict) # Stores predicted positions for lost tracks
        self.vel_hist = {}
        self.size_ref = {}
        self.frame_shape = None
        self.class_hist = {}
        self.locked_classes = set()

    def _get_relax_stage(self, lost_age):
        """Determines relaxation stage based on how long track has been lost."""
        if lost_age < 3: return 0
        elif lost_age < 10: return 1
        else: return 2

    def _relax_factor(self, t):
        lost_age = float(getattr(t, "time_since_update", 0))
        return self._get_relax_stage(lost_age)

    def _update_class_history(self, track_id, cls, conf):
        """Maintains a history of class predictions to smooth out flicker."""
        if track_id not in self.class_hist: self.class_hist[track_id] = []
        self.class_hist[track_id].append((cls, conf))
        if len(self.class_hist[track_id]) > CLASS_HIST_MAX:
            self.class_hist[track_id] = self.class_hist[track_id][-CLASS_HIST_MAX:]
        
        # Check if class can be locked
        if track_id not in self.locked_classes:
            class_counts = defaultdict(int)
            for c, conf in self.class_hist[track_id]:
                if conf >= CLASS_LOCK_CONF: class_counts[c] += 1
            for c, count in class_counts.items():
                if count >= CLASS_LOCK_COUNT:
                    self.locked_classes.add(track_id)
                    return c
        return None
        
    def _determine_class(self, track_id, cls, conf):
        """Returns the most likely class for a track based on history."""
        if track_id in self.locked_classes and track_id in self.class_hist and len(self.class_hist[track_id]) > 0:
            return self.class_hist[track_id][-1][0]
        locked_cls = self._update_class_history(track_id, cls, conf)
        if locked_cls is not None: return locked_cls
        
        # Simple voting if not locked
        if track_id in self.class_hist:
            class_confs = defaultdict(list)
            for c, conf in self.class_hist[track_id]:
                if conf >= BYTE_HIGH_THRES: class_confs[c].append(conf)
            best_cls = None; best_avg = -1
            for c, confs in class_confs.items():
                if len(confs) > 0:
                    avg = sum(confs) / len(confs)
                    if avg > best_avg: best_avg = avg; best_cls = c
            if best_cls is not None: return best_cls
        return cls

    def _relaxed_params(self, tid, pbox=None):
        """Calculates dynamic gating parameters based on track state."""
        trk = tid if hasattr(tid, "time_since_update") else next((tt for tt in self.tracks if tt.id == tid), None)
        if trk is None:
            base_center = CENTER_DIST_GATE
            diag = (self._diag_len(pbox) if pbox is not None else 1.0)
        else:
            base_center = max(CENTER_DIST_GATE, 0.35 * self._diag_len(get_track_box_current(trk)))
            diag = self._diag_len(get_track_box_current(trk))

        stage = self._relax_factor(trk) if trk is not None else 0
        return {
            "center_gate": float(base_center * (1.0 + CENTER_RELAX_SCALES[stage])),
            "iou_thres": float(max(0.01, REVIVE_IOU_THRES * max(0.0, (1.0 - IOU_RELAX_SCALES[stage])))),
            "size_min": float(SIZE_MIN_FLOORS[stage]),
            "size_max": float(SIZE_MAX_INCREASES[stage]),
            "maha_thresh": float(CHI2_POS_99 * (1.0 + MAHA_RELAX_SCALES[stage])),
            "relax_stage": stage
        }

    def _update_size_ref(self, t, det_bbox):
        """Updates the EMA reference size for the track."""
        cx, cy, w, h = xyxy_to_cxcywh(np.array(det_bbox, np.float32))
        if t.id not in self.size_ref:
            self.size_ref[t.id] = (float(w), float(h))
        else:
            pw, ph = self.size_ref[t.id]
            self.size_ref[t.id] = (
                (1.0 - SIZE_EMA)*pw + SIZE_EMA*float(w),
                (1.0 - SIZE_EMA)*ph + SIZE_EMA*float(h)
            )

    def _diag_len(self, box):
        w = max(0.0, box[2]-box[0]); h = max(0.0, box[3]-box[1])
        return math.hypot(w, h)

    def _cap_velocity(self, t, vx, vy):
        """Caps velocity to prevent unreasonable jumps in tracking."""
        diag = self._diag_len(get_track_box_current(t))
        cap = min(SPEED_CAP_ABS, SPEED_CAP_DIAG_FRAC * diag)
        speed = math.hypot(vx, vy)
        if speed <= MIN_SPEED: return 0.0, 0.0
        if speed > cap:
            s = cap / (speed + 1e-6)
            return vx * s, vy * s
        return vx, vy

    def _append_obs_velocity(self, t):
        """Calculates observed velocity from detection history and adds to buffer."""
        if len(t.det_hist) < 2: return
        (x0, y0, f0), (x1, y1, f1) = t.det_hist[-2], t.det_hist[-1]
        dt = max(1, int(f1 - f0))
        vx = (x1 - x0) / dt; vy = (y1 - y0) / dt
        cap = min(SPEED_CAP_ABS, SPEED_CAP_DIAG_FRAC * self._diag_len(get_track_box_current(t)))
        if math.hypot(vx, vy) > 1.2 * cap: return
        vx, vy = self._cap_velocity(t, vx, vy)
        dqv = self.vel_hist.get(t.id)
        if dqv is None: dqv = deque(maxlen=KINEMA_WINDOW); self.vel_hist[t.id] = dqv
        dqv.append((vx, vy))
    
    def _clear_future_for(self, tid, from_frame):
        """Removes future predictions for a track once it has been matched."""
        for f in list(self.future.keys()):
            if f < from_frame: continue
            self.future[f].pop(tid, None)
            if not self.future[f]: self.future.pop(f, None)

    def _get_mean_velocity(self, t):
        """Computes mean velocity from history or Kalman state."""
        hist = self.vel_hist.get(t.id)
        if hist and len(hist) > 0:
            vx = float(np.mean([v[0] for v in hist])); vy = float(np.mean([v[1] for v in hist]))
            return self._cap_velocity(t, vx, vy)
        if len(t.det_hist) >= 2:
            (x0, y0, f0), (x1, y1, f1) = t.det_hist[-2], t.det_hist[-1]
            dt = max(1, int(f1 - f0))
            vx = float((x1 - x0) / dt); vy = float((y1 - y0) / dt)
            return self._cap_velocity(t, vx, vy)
        if HAS_FILTERPY:
            vx, vy = float(t.kf.x[4, 0]), float(t.kf.x[5, 0])
            return self._cap_velocity(t, vx, vy)
        return 0.0, 0.0

    def _schedule_future(self, t, cur_frame, horizon=PRED_HORIZON_LOST):
        """Projects the track into the future for retrieval if lost."""
        if HAS_FILTERPY: cx, cy, w, h = [float(v) for v in t.kf.x[:4, 0]]
        else: cx, cy, w, h = [float(v) for v in t.kf.state]
        vx, vy = self._get_mean_velocity(t)
        if t.id in self.size_ref: w, h = self.size_ref[t.id]
        w = max(1.0, w * LOST_SIZE_INFLATE); h = max(1.0, h * LOST_SIZE_INFLATE)
        for k in range(1, horizon + 1):
            cxk = cx + k * vx; cyk = cy + k * vy
            vbox = cxcywh_to_xyxy(np.array([cxk, cyk, w, h], np.float32))
            self.future[cur_frame + k][t.id] = vbox

    def _center_dists(self, boxes_a, boxes_b):
        """Computes Euclidean distance matrix between box centers."""
        acx = (boxes_a[:, 0:1] + boxes_a[:, 2:3]) * 0.5
        acy = (boxes_a[:, 1:2] + boxes_a[:, 3:4]) * 0.5
        bcx = (boxes_b[:, 0:1] + boxes_b[:, 2:3]) * 0.5
        bcy = (boxes_b[:, 1:2] + boxes_b[:, 3:4]) * 0.5
        return np.hypot(acx - bcx.T, acy - bcy.T)

    def _dyn_center_gate(self, track_box):
        """Calculates dynamic spatial gate based on box size."""
        return max(CENTER_DIST_GATE, 0.35 * self._diag_len(track_box))

    def _calculate_mahalanobis(self, track, det_cx, det_cy):
        """Computes Mahalanobis distance between track prediction and detection."""
        if not HAS_FILTERPY: return None
        H = np.zeros((2,8), np.float32); H[0,0]=1; H[1,1]=1
        x = track.kf.x.reshape(-1,1)
        z = np.array([[det_cx],[det_cy]], np.float32)
        y = z - H @ x
        S = H @ track.kf.P @ H.T + np.eye(2, dtype=np.float32)
        try: Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError: return None
        return float((y.T @ Sinv @ y)[0,0])

    def _compute_association(self, tracks, dets):
        """
        Associates tracks with detections using the Hungarian Algorithm.
        Cost includes IoU, spatial distance, and Mahalanobis distance.
        """
        if len(tracks)==0 or len(dets)==0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        
        # Prepare matrices
        t_boxes = np.array([get_track_box_current(t) for t in tracks], np.float32)
        d_boxes = np.array([d["bbox"] for d in dets], np.float32)
        iou = compute_iou_matrix(t_boxes, d_boxes)
        cost = 1.0 - iou
        
        t_cx = (t_boxes[:,0] + t_boxes[:,2]) * 0.5
        t_cy = (t_boxes[:,1] + t_boxes[:,3]) * 0.5
        d_cx = (d_boxes[:,0] + d_boxes[:,2]) * 0.5
        d_cy = (d_boxes[:,1] + d_boxes[:,3]) * 0.5
        
        # Apply Gating
        if USE_IOU_GATES: cost[iou < self.iou_gate] = 1e6
        for i, t in enumerate(tracks):
            gate_px = self._dyn_center_gate(t_boxes[i])
            for j in range(len(dets)):
                dx = float(abs(t_cx[i] - d_cx[j]))
                dy = float(abs(t_cy[i] - d_cy[j]))
                if dx*dx + dy*dy > gate_px*gate_px:
                    cost[i, j] = 1e6
                    continue
                m2 = self._calculate_mahalanobis(t, d_cx[j], d_cy[j])
                if m2 is not None and m2 > CHI2_POS_99:
                    cost[i, j] = 1e6
        
        # Solve Assignment
        row, col = linear_sum_assignment(cost)
        matches, un_t, un_d = [], [], []
        rset, cset = set(row.tolist()), set(col.tolist())
        
        for i in range(len(tracks)):
            if i not in rset: un_t.append(i)
        for j in range(len(dets)):
            if j not in cset: un_d.append(j)
            
        for i, j in zip(row, col):
            if cost[i, j] < 1e5: matches.append((i, j))
            else: un_t.append(i); un_d.append(j)
        return matches, un_t, un_d

    def _attempt_revival_from_predictions(self, dets_high, frame_idx):
        """
        Tries to match high-confidence detections to "future" predictions 
        of currently lost tracks. This is a recovery mechanism.
        """
        if frame_idx not in self.future or len(self.future[frame_idx]) == 0:
            return set(), set()
        if len(dets_high) == 0:
            return set(), set()
            
        # Get predictions for this frame
        pred_items = list(self.future[frame_idx].items())
        tids = [tid for tid, _ in pred_items]
        pboxes = np.stack([box for _, box in pred_items], axis=0).astype(np.float32)
        dboxes = np.stack([d["bbox"] for d in dets_high], axis=0).astype(np.float32)
        
        dist = self._center_dists(pboxes, dboxes)
        iou = compute_iou_matrix(pboxes, dboxes)
        R, C = dist.shape
        gated = np.full((R, C), 1e6, dtype=np.float32)
        
        # Build Cost Matrix with Relaxed Gating
        for r in range(R):
            tid = tids[r]; pbox = pboxes[r]
            params = self._relaxed_params(tid, pbox=pbox)
            center_gate = params["center_gate"]; iou_thres = params["iou_thres"]
            size_min = params["size_min"]; size_max = params["size_max"]; maha_thresh = params["maha_thresh"]
            trk = next((tt for tt in self.tracks if tt.id == tid), None)
            
            for c in range(C):
                if dist[r, c] > center_gate: continue
                if REVIVE_REQUIRE_SAME_CLASS and trk is not None:
                    if dets_high[c]["cls"] != trk.cls: continue
                if REVIVE_USE_IOU and iou[r, c] < iou_thres: continue
                if trk is not None and HAS_FILTERPY:
                    db = dets_high[c]["bbox"]
                    dcx = (db[0] + db[2]) * 0.5; dcy = (db[1] + db[3]) * 0.5
                    m2 = self._calculate_mahalanobis(trk, dcx, dcy)
                    if m2 is not None and m2 > maha_thresh: continue
                    
                # Size consistency check
                det_area = calculate_box_area(dets_high[c]["bbox"])
                pred_area = calculate_box_area(pbox)
                size_ratio = det_area / max(1.0, pred_area)
                if not (size_min <= size_ratio <= size_max): continue
                
                # Border Logic: penalize revival if angle changes abruptly at border
                det_near_border = is_near_border(dets_high[c]["bbox"], self.frame_shape, border_ratio=BORDER_RATIO)
                if det_near_border:
                    angle_change = 0
                    if trk is not None and trk.id in self.vel_hist and len(self.vel_hist[trk.id]) > 0:
                        curr_vx, curr_vy = self.vel_hist[trk.id][-1]
                        t_box = get_track_box_current(trk)
                        t_cx = (t_box[0] + t_box[2]) / 2; t_cy = (t_box[1] + t_box[3]) / 2
                        d_cx = (dboxes[c,0]+dboxes[c,2])/2; d_cy = (dboxes[c,1]+dboxes[c,3])/2
                        curr_angle = math.degrees(math.atan2(curr_vy, curr_vx))
                        new_angle = math.degrees(math.atan2(d_cy - t_cy, d_cx - t_cx))
                        angle_change = abs(new_angle - curr_angle)
                        angle_change = min(angle_change, 360 - angle_change)
                        if angle_change > ANGLE_CHANGE_THRES: continue
                    gated[r, c] = float(dist[r, c] * BORDER_REVIVE_PENALTY)
                else:
                    gated[r, c] = float(dist[r, c])
                    
        # Assignment
        rows, cols = linear_sum_assignment(gated)
        used_det_idx = set(); revived_tids = set()
        for ri, ci in zip(rows, cols):
            if gated[ri, ci] >= 1e5: continue
            tid = tids[ri]; revived_tids.add(tid); used_det_idx.add(ci)
            trk = next((tt for tt in self.tracks if tt.id == tid), None)
            if trk is None: continue
            
            # Reactivate track
            d = dets_high[ci]
            trk.update(d["bbox"], d["cls"], d["conf"], frame_idx)
            self._append_obs_velocity(trk)
            self._update_size_ref(trk, d["bbox"])
            trk.matched_this_frame = True
            trk.high_conf_match = True
            a = calculate_box_area(d["bbox"])
            dq = self.area_hist.get(trk.id)
            if dq is None: dq = deque(maxlen=AREA_WINDOW); self.area_hist[trk.id] = dq
            dq.append(a)
            
        # Clean up futures for revived tracks
        for tid in revived_tids: self._clear_future_for(tid, frame_idx)
        return revived_tids, used_det_idx

    def update(self, detections, frame_idx, frame_shape=None):
        """
        Main update loop for the tracker.
        
        Steps:
        1. Predict Kalman states.
        2. Divide detections into High and Low confidence.
        3. Associate High confidence detections with Active tracks.
        4. Attempt to revive lost tracks using predictions.
        5. Associate Low confidence detections with remaining active tracks.
        6. Create new tracks for unmatched High detections.
        7. Update track states (mark lost/missed) and clean up.
        """
        self.frame_shape = frame_shape
        for t in self.tracks:
            if HAS_FILTERPY: t.kf.predict()
            
        # Split detections
        high = [d for d in detections if d["conf"] >= BYTE_HIGH_THRES]
        low  = [d for d in detections if BYTE_LOW_THRES <= d["conf"] < BYTE_HIGH_THRES]
        
        for t in self.tracks: t.matched_this_frame = False; t.high_conf_match = False
        used_high = set(); matched_global = set()
        
        # 1. Match High Conf Dets to Active Tracks
        active_idx = [i for i,t in enumerate(self.tracks) if t.time_since_update == 0]
        active_trs = [self.tracks[i] for i in active_idx]
        
        if active_trs and high:
            mA, un_act, un_hA = self._compute_association(active_trs, high)
            for ai, dj in mA:
                ti = active_idx[ai]; t = self.tracks[ti]; d = high[dj]
                determined_cls = self._determine_class(t.id, d["cls"], d["conf"])
                t.update(d["bbox"], determined_cls, d["conf"], frame_idx)
                t.cls = determined_cls
                self._append_obs_velocity(t)
                a = calculate_box_area(d["bbox"])
                dq = self.area_hist.get(t.id)
                if dq is None: dq = deque(maxlen=AREA_WINDOW); self.area_hist[t.id] = dq
                dq.append(a)
                self._update_size_ref(t, d["bbox"])
                t.matched_this_frame = True; t.high_conf_match = True
                self._clear_future_for(t.id, frame_idx)
                used_high.add(dj); matched_global.add(ti)
                
        # Prepare unmatched high detections for revival
        if len(high) > 0 and len(used_high) > 0:
            idx_map = [i for i in range(len(high)) if i not in used_high]
            high_left = [high[i] for i in idx_map]
        else:
            idx_map = list(range(len(high))); high_left = high
            
        # 2. Attempt Revival (Prediction matching)
        revived_tids, used_det_idx = self._attempt_revival_from_predictions(high_left, frame_idx)
        if used_det_idx:
            for k in used_det_idx: used_high.add(idx_map[k])
        if revived_tids:
            for i, t in enumerate(self.tracks):
                if t.id in revived_tids: matched_global.add(i)
        
        # 3. Match Low Conf Dets to Remaining Active Tracks
        if low:
            still_unmatched_act = [i for i in active_idx if i not in matched_global]
            if still_unmatched_act:
                trs_lo = [self.tracks[i] for i in still_unmatched_act]
                mC, _, _ = self._compute_association(trs_lo, low)
                accepted = []
                for li, dj in mC:
                    t = trs_lo[li]; d = low[dj]
                    # Size check to prevent matching noise
                    det_area = calculate_box_area(d["bbox"])
                    ref_hist = self.area_hist.get(t.id)
                    ref_area = get_blended_ref_area(ref_hist)
                    if ref_area is None: ref_area = calculate_box_area(get_track_box_current(t))
                    ratio = det_area / max(1.0, ref_area)
                    if SIZE_CHANGE_MIN <= ratio <= SIZE_CHANGE_MAX: accepted.append((li, dj))
                    
                for li, dj in accepted:
                    ti = still_unmatched_act[li]; t = self.tracks[ti]; d = low[dj]
                    t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                    self._append_obs_velocity(t)
                    t.matched_this_frame = True; t.high_conf_match = False
                    self._clear_future_for(t.id, frame_idx)
                    matched_global.add(ti)
                    
        # 4. Final Revival Attempt (Optional second pass logic)
        if frame_idx in self.future and self.future[frame_idx]:
            remaining_high_idx = [i for i in range(len(high)) if i not in used_high]
            if remaining_high_idx:
                high_left = [high[i] for i in remaining_high_idx]
                revived_tids2, used_det_idx2 = self._attempt_revival_from_predictions(high_left, frame_idx)
                if used_det_idx2:
                    for k in used_det_idx2: used_high.add(remaining_high_idx[k])
                if revived_tids2:
                    for i, t in enumerate(self.tracks):
                        if t.id in revived_tids2: matched_global.add(i)
        
        # 5. Initialize New Tracks
        for j in range(len(high)):
            if j not in used_high:
                d = high[j]; det_near_border = False
                if self.frame_shape is not None: det_near_border = is_near_border(d["bbox"], self.frame_shape, border_ratio=BORDER_RATIO)
                
                # Check if this new detection is actually just a lost track nearby (duplicate avoidance)
                if not det_near_border and PREFER_NEW_AT_BORDER:
                    nearby_lost = False
                    for t in self.tracks:
                        if t.time_since_update > 0:
                            t_box = get_track_box_current(t)
                            t_cx = (t_box[0] + t_box[2]) / 2; t_cy = (t_box[1] + t_box[3]) / 2
                            d_cx = (d["bbox"][0] + d["bbox"][2]) / 2; d_cy = (d["bbox"][1] + d["bbox"][3]) / 2
                            dist = math.sqrt((d_cx - t_cx)**2 + (d_cy - t_cy)**2)
                            if dist < CENTER_DIST_GATE * 1.5: nearby_lost = True; break
                    if nearby_lost: continue
                self.tracks.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))
        
        # 6. Update State for Unmatched Tracks
        for i, t in enumerate(self.tracks):
            if i not in matched_global:
                prev_tsu = t.time_since_update; t.mark_missed()
                # If just lost, initialize future prediction
                if prev_tsu == 0:
                    vx_mean, vy_mean = self._get_mean_velocity(t)
                    if HAS_FILTERPY: t.kf.x[4, 0] = vx_mean; t.kf.x[5, 0] = vy_mean
                    self._schedule_future(t, frame_idx, horizon=PRED_HORIZON_LOST)
                    
        # 7. Identify "Lost" vs "Active" and Clean Up
        new_lost = set()
        for t in self.tracks:
            near_b = is_near_border(get_track_box_current(t), frame_shape, border_ratio=0.04) if frame_shape is not None else False
            if (t.hits >= self.min_hits) and (t.time_since_update >= 2) and (not near_b): new_lost.add(t.id)
        self.lost_ids = new_lost
        
        survivors = []
        for t in self.tracks:
            if t.time_since_update <= LOST_KEEP: survivors.append(t)
            else:
                self._clear_future_for(t.id, frame_idx + 1)
                self.vel_hist.pop(t.id, None)
        self.tracks = survivors
        return self.tracks


# ==============================================================================
# 6. EVALUATION METRICS
# ==============================================================================

def calculate_mse_per_frame(gt_centers, pred_centers, max_dist=60.0):
    """Calculates Mean Squared Error between matched Ground Truth and Prediction centers."""
    if len(gt_centers) == 0 or len(pred_centers) == 0: return None
    G = np.array(gt_centers, np.float32); P = np.array(pred_centers, np.float32)
    
    # Pairwise squared distances
    diff = G[:, None, :] - P[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    
    # Hungarian match
    r, c = linear_sum_assignment(cost)
    if len(r) == 0: return None
    
    # Filter valid matches by distance threshold
    sel = cost[r, c] <= (max_dist * max_dist)
    if not np.any(sel): return None
    
    return float(np.mean(cost[r[sel], c[sel]]))

def build_gt_by_frame(frames_gt, scale_x, scale_y, skip_occ=True):
    """Reorganizes Ground Truth data by frame index for quick lookup."""
    out = {}
    for f, anns in frames_gt.items():
        cur = []
        for ann in anns:
            if skip_occ and (ann["lost"]==1 or ann["occluded"]==1): continue
            x1,y1,x2,y2 = ann["bbox"]
            bb = [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]
            cur.append((int(ann["id"]), np.array(bb, np.float32)))
        if cur: out[int(f)] = cur
    return out

def _match_frame_hota(g_ids, g_boxes, t_ids, t_boxes, tau):
    """Helper for HOTA: Matches GT and Predicted boxes for a specific IoU threshold (tau)."""
    if len(g_ids)==0 or len(t_ids)==0:
        return [], list(range(len(g_ids))), list(range(len(t_ids)))
    iou = compute_iou_matrix(g_boxes, t_boxes)
    cost = 1.0 - iou; cost[iou < tau] = 1e6
    r, c = linear_sum_assignment(cost)
    
    matches, un_g, un_t = [], [], []
    rset, cset = set(r.tolist()), set(c.tolist())
    for i in range(len(g_ids)):
        if i not in rset: un_g.append(i)
    for j in range(len(t_ids)):
        if j not in cset: un_t.append(j)
    for i, j in zip(r, c):
        if iou[i, j] >= tau: matches.append((i, j))
        else: un_g.append(i); un_t.append(j)
    return matches, un_g, un_t

def eval_hota(gt_by_frame, pred_by_frame, total_frames, taus):
    """
    Computes HOTA metrics (DetA, AssA, HOTA) across multiple IoU thresholds.
    HOTA decomposes into Detection Accuracy (DetA) and Association Accuracy (AssA).
    """
    rows = []
    for tau in taus:
        TP=FP=FN=0; g2t={}
        # 1. Detection Evaluation
        for f in range(total_frames):
            g_list = gt_by_frame.get(f, []); t_list = pred_by_frame.get(f, [])
            g_ids = [gid for gid,_ in g_list]; t_ids = [tid for tid,_ in t_list]
            g_boxes = np.array([b for _,b in g_list], np.float32) if g_list else np.zeros((0,4),np.float32)
            t_boxes = np.array([b for _,b in t_list], np.float32) if t_list else np.zeros((0,4),np.float32)
            
            m, ug, ut = _match_frame_hota(g_ids, g_boxes, t_ids, t_boxes, tau)
            TP += len(m); FP += len(ut); FN += len(ug)
            g2t[f] = { g_ids[i]: t_ids[j] for i,j in m }
            
        det_den = TP + 0.5*(FP+FN)
        DetA = (TP/det_den) if det_den>0 else 0.0
        
        # 2. Association Evaluation
        pairs = {(gid,tid) for f,mm in g2t.items() for gid,tid in mm.items()}
        accs = []
        for gid, tid in pairs:
            IDTP=IDFP=IDFN=0
            for f in range(total_frames):
                g_present = any(g==gid for g,_ in gt_by_frame.get(f, []))
                t_present = any(t==tid for t,_ in pred_by_frame.get(f, []))
                if g_present and t_present:
                    if g2t.get(f, {}).get(gid, None) == tid: IDTP += 1
                    else: IDFP += 1; IDFN += 1
                elif g_present and not t_present: IDFN += 1
                elif t_present and not g_present: IDFP += 1
            denom = IDTP + 0.5*(IDFP+IDFN)
            if denom>0: accs.append(IDTP/denom)
            
        AssA = float(np.mean(accs)) if accs else 0.0
        HOTA = math.sqrt(max(0.0, DetA)*max(0.0, AssA))
        rows.append({"tau":tau, "DetA":DetA, "AssA":AssA, "HOTA":HOTA})
        
    df = pd.DataFrame(rows)
    return df, float(df["DetA"].mean()), float(df["AssA"].mean()), float(df["HOTA"].mean())

def parse_sdd_annotations(path):
    """Parses Stanford Drone Dataset annotation text file into a dictionary."""
    frames = defaultdict(list); max_x = 0; max_y = 0
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            p = s.split()
            if len(p) < 10: continue
            # Format: TrackID, x1, y1, x2, y2, Frame, Lost, Occluded, Generated, Label
            tid = int(p[0]); x1=float(p[1]); y1=float(p[2]); x2=float(p[3]); y2=float(p[4])
            fr  = int(p[5]); lost=int(p[6]); occl=int(p[7])
            label = " ".join(p[9:]).strip().strip('"')
            frames[fr].append({"id":tid,"bbox":[x1,y1,x2,y2],"lost":lost,"occluded":occl,"label":label})
            max_x = max(max_x, x2); max_y = max(max_y, y2)
    return frames, (int(math.ceil(max_x)), int(math.ceil(max_y)))

def scale_bbox(bbox, sx, sy):
    """Rescales a bounding box by X and Y factors."""
    x1,y1,x2,y2 = bbox
    return [x1*sx, y1*sy, x2*sx, y2*sy]