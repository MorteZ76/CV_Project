"""
tracker.py
----------
Main execution script for the Advanced Object Tracker.
Handles video loading, main loop execution, visualization, and result saving.
"""

import os
import cv2
import sys
import math
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from pathlib import Path
import utils  # Local project utils
import tracker_utils as tu  # Tracker specific utils

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Paths ---
video_num = 0  # 0 or 3 to switch videos
VIDEO_PATH, ANNOT_PATH = utils.get_video_paths(video_num)

# Resolve Base Directory (Go back one level from src if script is in src/)
BASE_DIR = Path(__file__).parent.parent if "__file__" in locals() else Path.cwd().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

# Relative Detection Path
DET_PATH = str(OUTPUTS_DIR / "detections_cache" / f"new_model_video{video_num}_detections.parquet")

# --- Evaluation & Output Directories ---
TRAJ_DIR = OUTPUTS_DIR / "trajectories"
HOTA_DIR = OUTPUTS_DIR / "hota"
utils.ensure_dir(str(TRAJ_DIR))
utils.ensure_dir(str(HOTA_DIR))

TRAJ_CSV = str(TRAJ_DIR / f"video{video_num}_trajectoriesNEW.csv")
HOTA_CSV = str(HOTA_DIR / f"video{video_num}_hota_breakdownNEW.csv")

# --- Visual Settings ---
COMPUTE_MSE = True
HOTA_TAUS = [i / 20 for i in range(1, 20)]  # 0.05 to 0.95
SKIP_GT_OCCLUDED = True
PAUSE_ON_START = False

# Class Definitions (Dynamic from utils)
CLASS_NAMES = {v: k for k, v in utils.CLASS_MAP.items()}
LABEL_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
CLASS_COLORS = {v: utils.CLASS_COLORS[k] for k, v in utils.CLASS_MAP.items()}
LOST_COLOR = (0, 165, 255)


# ==============================================================================
# MAIN EXECUTION LOOP
# ==============================================================================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Load Ground Truth ---
    frames_gt, (W_ref, H_ref) = tu.parse_sdd_annotations(ANNOT_PATH)
    scale_x = W / float(W_ref if W_ref > 0 else W)
    scale_y = H / float(H_ref if H_ref > 0 else H)
    gt_by_frame = tu.build_gt_by_frame(frames_gt, scale_x, scale_y, SKIP_GT_OCCLUDED)
    pred_by_frame = defaultdict(list)

    # --- Load Detections ---
    dets_by_frame = tu.load_cached_detections(DET_PATH)

    # --- Initialize Tracker ---
    tracker = tu.ByteTrackLike(iou_gate=tu.IOU_GATE, min_hits=tu.MIN_HITS)

    # State Containers
    track_paths = {}         # stores high-conf centers
    track_last_seen = {}     # last frame a high-conf point was added
    track_cls = {}
    gt_paths = defaultdict(deque)
    
    traj_rows = []
    mse_values = []
    
    frame_idx = 0
    paused = PAUSE_ON_START
    did_seek = False

    # --- Helper Functions ---
    def _clamp(i: int) -> int: 
        return max(0, min(total_frames - 1, i))

    def _reset_tracking_state():
        nonlocal tracker, track_paths, track_last_seen, track_cls
        tracker = tu.ByteTrackLike(iou_gate=tu.IOU_GATE, min_hits=tu.MIN_HITS)
        track_paths = {}
        track_last_seen = {}
        track_cls = {}

    def _seek_to(target_idx: int):
        nonlocal frame_idx, did_seek
        frame_idx = _clamp(target_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _reset_tracking_state()
        did_seek = True

    if PAUSE_ON_START: 
        print("Paused. Press 'r' to resume, 'p' to pause again.")

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret: break

        vis_gt = frame.copy()
        vis_det = frame.copy()

        # ---------------------------
        # 1. Ground Truth Visualization
        # ---------------------------
        gt_centers = []
        if frame_idx in frames_gt:
            for ann in frames_gt[frame_idx]:
                if SKIP_GT_OCCLUDED and (ann["lost"] == 1 or ann["occluded"] == 1): continue
                
                bb = tu.scale_bbox(ann["bbox"], scale_x, scale_y)
                cx, cy, _, _ = tu.xyxy_to_cxcywh(np.array(bb, np.float32))
                gt_centers.append((cx, cy))
                
                gt_paths[ann["id"]].append((int(cx), int(cy)))
                if len(gt_paths[ann["id"]]) > tu.TRAJ_MAX_LEN: 
                    gt_paths[ann["id"]].popleft()
                
                gt_cls_id = LABEL_TO_ID.get(ann["label"], 0)
                color = CLASS_COLORS.get(gt_cls_id, (0, 255, 0))
                
                tu.draw_box_with_id(vis_gt, bb, cls=gt_cls_id, tid=ann["id"], conf=None, color=color, label_map=CLASS_NAMES)
                tu.draw_trajectory(vis_gt, list(gt_paths[ann["id"]]), color=color)

        # ---------------------------
        # 2. Tracking Logic
        # ---------------------------
        rows = dets_by_frame.get(frame_idx, np.empty((0, 7), np.float32))
        detections = tu.apply_nms(rows, tu.DET_CONF_THRES, tu.DET_IOU_NMS, tu.AGNOSTIC_NMS)

        tracks = tracker.update(detections, frame_idx, frame.shape)

        pred_centers = []
        
        for t in tracks:
            # Skip ephemeral tracks unless predicting lost ones
            if t.hits < tu.MIN_HITS and t.time_since_update != 0:
                continue

            box = tu.get_track_box_current(t)
            cx, cy, w, h = (t.kf.x[:4, 0] if tu.HAS_FILTERPY else t.kf.state)

            # Store data for HOTA
            if t.matched_this_frame:
                pred_by_frame[frame_idx].append((int(t.id), box.astype(np.float32).copy()))

            # Store data for MSE
            if t.matched_this_frame and t.time_since_update == 0:
                pred_centers.append((float(cx), float(cy)))

            # --- Trajectory Storage (High-Conf Only) ---
            if t.high_conf_match:
                if t.id not in track_paths:
                    track_paths[t.id] = deque(maxlen=tu.TRAJ_MAX_LEN)
                track_paths[t.id].append((int(cx), int(cy)))
                track_last_seen[t.id] = frame_idx
                track_cls[t.id] = t.cls
                
                traj_rows.append(["video0", t.id, frame_idx, float(cx), float(cy), t.cls, CLASS_NAMES.get(t.cls, "Unknown")])

            # --- Visualization ---
            # Determine drawing box (inflate if lost)
            if t.id in tracker.lost_ids and t.id in tracker.size_ref:
                sw, sh = tracker.size_ref[t.id]
                draw_box = tu.cxcywh_to_xyxy(np.array([cx, cy, sw * tu.LOST_SIZE_INFLATE, sh * tu.LOST_SIZE_INFLATE], np.float32))
            elif t.id in tracker.lost_ids:
                draw_box = tu.expand_box(box, tu.PRED_EXPAND)
            else:
                draw_box = box

            color = LOST_COLOR if (t.id in tracker.lost_ids) else CLASS_COLORS.get(t.cls, (200, 200, 200))
            tag = " LOST" if (t.id in tracker.lost_ids) else ""
            
            tu.draw_box_with_id(vis_det, draw_box, cls=t.cls, tid=t.id, conf=t.conf, color=color, label_map=CLASS_NAMES)
            
            if tag:
                cv2.putText(vis_det, tag, (int(draw_box[0]), max(0, int(draw_box[1]) - 20)), tu.FONT, 0.5, color, 2, cv2.LINE_AA)

            if t.id in track_paths:
                tu.draw_trajectory(vis_det, list(track_paths[t.id]), color=color)

            # Velocity Arrow
            vx, vy = tracker._get_mean_velocity(t)
            speed = math.hypot(vx, vy)
            if speed > tu.MIN_SPEED:
                cx_draw, cy_draw = (draw_box[0] + draw_box[2]) / 2, (draw_box[1] + draw_box[3]) / 2
                angle = math.degrees(math.atan2(vy, vx))
                
                tu.draw_velocity_arrow(vis_det, cx_draw, cy_draw, angle, length=tu.ARROW_LEN, color=color)
                comp = tu.angle_to_compass((angle + 360) % 360)
                
                cv2.putText(vis_det, f"v:{speed:.1f} {tu.SHOW_UNITS}  dir:{comp}",
                            (int(cx) + 5, int(cy) + 15), tu.FONT, 0.5, color, 2, cv2.LINE_AA)

        # Cleanup stale UI tracks
        stale_ids = [tid for tid, last in list(track_last_seen.items())
                     if frame_idx - last > tu.MISS_FRAMES_TO_DROP_PATH]
        for tid in stale_ids:
            track_paths.pop(tid, None)
            track_last_seen.pop(tid, None)
            track_cls.pop(tid, None)

        # MSE Overlay
        if COMPUTE_MSE:
            frame_mse = tu.calculate_mse_per_frame(gt_centers, pred_centers)
            if frame_mse is not None:
                mse_values.append(frame_mse)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), tu.FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), tu.FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # ----------------------------
        # 2.5. Display
        # ----------------------------
        # Resize images to fit screen (height = 960px)
        target_h = 960
        scale = target_h / float(vis_gt.shape[0])
        target_w = int(vis_gt.shape[1] * scale)

        vis_gt_resized = cv2.resize(vis_gt, (target_w, target_h))
        vis_det_resized = cv2.resize(vis_det, (target_w, target_h))

        cv2.imshow("GT (rescaled annotations + trajectories)", vis_gt_resized)
        cv2.imshow("Cached dets + ByteTrack + trajectories", vis_det_resized)
        # # Display Windows
        # cv2.imshow("GT (rescaled annotations + trajectories)", vis_gt)
        # cv2.imshow("Cached dets + ByteTrack + trajectories", vis_det)

        # ---------------------------
        # 3. Controls
        # ---------------------------
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('r'): paused = False
        
        if paused:
            if key == ord('o'): _seek_to(frame_idx + 1)
            elif key == ord('i'): _seek_to(frame_idx - 1)
            elif key == ord('l'): _seek_to(frame_idx + 100)
            elif key == ord('k'): _seek_to(frame_idx - 100)
            
        if not paused and not did_seek: 
            frame_idx += 1
        did_seek = False

    cap.release()
    cv2.destroyAllWindows()

    # ==============================================================================
    # EXPORT RESULTS
    # ==============================================================================
    
    # Save Trajectories
    if traj_rows:
        df_traj = pd.DataFrame(traj_rows, columns=["video_id", "track_id", "frame", "x", "y", "class_id", "class_name"])
        df_traj.to_csv(TRAJ_CSV, index=False)
        print(f"Trajectories saved to {TRAJ_CSV}")

    # Calculate and Save HOTA
    processed_frames = frame_idx + 1
    if gt_by_frame:
        df_hota, mean_DetA, mean_AssA, mean_HOTA = tu.eval_hota(
            gt_by_frame, pred_by_frame, processed_frames, HOTA_TAUS
        )
        df_hota.to_csv(HOTA_CSV, index=False)
        print(f"HOTA saved to {HOTA_CSV}")
        print(f"Mean DetA: {mean_DetA:.4f}  Mean AssA: {mean_AssA:.4f}  Mean HOTA: {mean_HOTA:.4f}")
    else:
        print("HOTA skipped: no GT.")

    # Report MSE
    if COMPUTE_MSE and len(mse_values) > 0:
        print(f"Overall MSE: {np.mean(mse_values):.3f}")


if __name__ == "__main__":
    if not hasattr(cv2, "imshow"):
        print("OpenCV built without HighGUI. Install opencv-python.")
        sys.exit(1)
    main()