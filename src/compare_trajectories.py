"""
compare_trajectories.py
-----------------------
Visualizes and compares tracking trajectories from two different models (New vs Old)
side-by-side.

Key Features:
- Loads trajectory data from CSVs.
- Resizes video to fixed height (960px) while STRICTLY maintaining aspect ratio.
- Displays "New Model" and "Old Model" results side-by-side.
- Uses track-specific color variations for better distinction.
"""

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Local imports
try:
    import utils
except ImportError:
    raise ImportError("utils.py not found. Ensure it is in the same directory.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Select Video to Compare (0 or 3)
VIDEO_NUM = 0

# --- Visual Settings ---
TARGET_HEIGHT = 960   # Fixed height
THICKNESS = 2
POINT_RADIUS = 3
TRAJ_MAX_LEN = 10000  # Max history points to draw
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Paths ---
# Resolve paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "trajectories"

# CSV Files (Auto-select based on VIDEO_NUM)
NEW_CSV_PATH = OUTPUTS_DIR / f"video{VIDEO_NUM}_trajectoriesNEW.csv"
OLD_CSV_PATH = OUTPUTS_DIR / f"video{VIDEO_NUM}_trajectoriesOLD.csv"

# Get Video Path from utils
try:
    VIDEO_PATH, _ = utils.get_video_paths(VIDEO_NUM)
except ValueError as e:
    print(f"[Error] {e}")
    VIDEO_PATH = ""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_shaded_color(class_id: int, track_id: int) -> Tuple[int, int, int]:
    """
    Generates a color for a track based on its class, adding slight
    variations based on track_id to distinguish individual instances.
    """
    # 1. Get base name and color from utils
    class_name = next((k for k, v in utils.CLASS_MAP.items() if v == class_id), "Unknown")
    base_bgr = utils.CLASS_COLORS.get(class_name, (200, 200, 200))

    # 2. Convert to HSV to shift Saturation/Value
    base_hsv = cv2.cvtColor(np.uint8([[base_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    h = int(base_hsv[0])
    s = max(100, min(255, int(base_hsv[1]) - (track_id * 7) % 60))
    v = max(100, min(255, int(base_hsv[2]) - (track_id * 11) % 40))
    
    # 3. Convert back to BGR
    final_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, final_bgr))


def load_trajectories(csv_path: Path) -> Tuple[Dict[int, List], Dict[int, Dict]]:
    """
    Efficiently loads trajectory data from a CSV file.
    
    Returns:
        frames_map: {frame_idx: [(track_id, (x, y), class_id), ...]}
        track_info: {track_id: {'class_name': str, 'last_frame': int}}
    """
    if not csv_path.exists():
        print(f"[Warning] File not found: {csv_path}")
        return {}, {}

    print(f"Loading: {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Could not read CSV: {e}")
        return {}, {}

    # Auto-correct 1-based frames if detected
    if not df.empty and df["frame"].min() > 0:
        df["frame"] -= 1

    frames_map = defaultdict(list)
    track_info = {}

    # Group by frame for fast iteration
    for frame_idx, group in df.groupby("frame"):
        for row in group.itertuples():
            tid = int(row.track_id)
            pt = (int(round(row.x)), int(round(row.y)))
            cid = int(row.class_id)
            
            frames_map[frame_idx].append((tid, pt, cid))
            
            # Store metadata
            if tid not in track_info:
                track_info[tid] = {
                    'class_name': str(row.class_name),
                    'last_frame': frame_idx
                }
            else:
                track_info[tid]['last_frame'] = max(track_info[tid]['last_frame'], frame_idx)

    print(f" -> Loaded {len(track_info)} unique tracks.")
    return frames_map, track_info


def draw_paths(frame: np.ndarray, paths: Dict[int, deque], colors: Dict[int, Tuple], track_info: Dict) -> None:
    """Draws trajectory lines, endpoints, and labels on the frame."""
    for tid, pts in paths.items():
        if len(pts) < 2: 
            continue
            
        # Draw Polyline
        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts_np], isClosed=False, color=colors[tid], thickness=THICKNESS)
        
        # Draw Endpoint
        last_pt = pts[-1]
        cv2.circle(frame, last_pt, POINT_RADIUS, colors[tid], -1)
        
        # Draw Label (Class Name + ID)
        name = track_info.get(tid, {}).get('class_name', '?')
        label = f"{name} ({tid})"
        cv2.putText(frame, label, (last_pt[0] + 5, last_pt[1] - 5),
                    FONT, 0.4, colors[tid], 1, cv2.LINE_AA)


def draw_legend(frame: np.ndarray, title: str) -> None:
    """Overlays a semi-transparent title and class legend."""
    # Title Bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, title, (10, 28), FONT, 0.8, (255, 255, 255), 2)

    # Class Legend (Vertical)
    y = 70
    for name, color in utils.CLASS_COLORS.items():
        # Text shadow for visibility
        cv2.putText(frame, name, (11, y+1), FONT, 0.5, (0,0,0), 2)
        cv2.putText(frame, name, (10, y), FONT, 0.5, color, 1)
        y += 20


# ==============================================================================
# MAIN LOOP
# ==============================================================================

def main():
    print(f"Initializing Comparison for Video {VIDEO_NUM}...")
    print(f"Video Source: {VIDEO_PATH}")

    # 1. Load Data
    new_map, new_info = load_trajectories(NEW_CSV_PATH)
    old_map, old_info = load_trajectories(OLD_CSV_PATH)

    # 2. Setup Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] Cannot open video file: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read first frame to calculate proper resize dimensions
    ret, initial_frame = cap.read()
    if not ret:
        print("[Error] Failed to read first frame.")
        return
    
    # Calculate Resize based on Actual Frame Shape
    orig_h, orig_w = initial_frame.shape[:2]
    
    # Fixed Height = 960, Width = Scaled proportionally
    scale_factor = TARGET_HEIGHT / orig_h
    new_width = int(orig_w * scale_factor)
    resize_dim = (new_width, TARGET_HEIGHT)  # (Width, Height) for cv2.resize
    
    print(f"Original: {orig_w}x{orig_h} -> Resized: {new_width}x{TARGET_HEIGHT}")
    
    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 3. State Setup
    frame_idx = 0
    paused = False
    
    # Trajectory buffers
    new_paths = defaultdict(lambda: deque(maxlen=TRAJ_MAX_LEN))
    old_paths = defaultdict(lambda: deque(maxlen=TRAJ_MAX_LEN))
    new_colors = {}
    old_colors = {}

    # Helper to update paths for current frame
    def update_frame_data(frame_data, paths_dict, colors_dict):
        for tid, pt, cid in frame_data.get(frame_idx, []):
            paths_dict[tid].append(pt)
            if tid not in colors_dict:
                colors_dict[tid] = get_shaded_color(cid, tid)

    # Helper to clean up finished tracks
    def cleanup_tracks(paths_dict, colors_dict, info_dict):
        for tid in list(paths_dict.keys()):
            if frame_idx > info_dict[tid]['last_frame']:
                paths_dict.pop(tid, None)
                colors_dict.pop(tid, None)

    # Pre-load frame 0 data
    update_frame_data(new_map, new_paths, new_colors)
    update_frame_data(old_map, old_paths, old_colors)

    cv2.namedWindow("Trajectory Comparison", cv2.WINDOW_NORMAL)
    
    print("\n--- Controls ---")
    print(" [P] Play/Pause")
    print(" [R] Restart")
    print(" [Q] Quit")

    while True:
        # Read Frame
        if not paused:
            ret, base_frame = cap.read()
            if not ret:
                print("End of video.")
                break
        
        # Create canvases
        frame_new = base_frame.copy()
        frame_old = base_frame.copy()

        # Update & Draw NEW Model
        cleanup_tracks(new_paths, new_colors, new_info)
        draw_paths(frame_new, new_paths, new_colors, new_info)
        draw_legend(frame_new, f"NEW Model | Frame {frame_idx}")

        # Update & Draw OLD Model
        cleanup_tracks(old_paths, old_colors, old_info)
        draw_paths(frame_old, old_paths, old_colors, old_info)
        draw_legend(frame_old, f"OLD Model | Frame {frame_idx}")

        # Resize both frames using pre-calculated dimensions
        frame_new_res = cv2.resize(frame_new, resize_dim)
        frame_old_res = cv2.resize(frame_old, resize_dim)

        # Stack Side-by-Side
        combined = np.hstack((frame_new_res, frame_old_res))

        cv2.imshow("Trajectory Comparison", combined)

        # Handle Input
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') or key == 32: # Space or P
            paused = not paused
        elif key == ord('r'):
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            new_paths.clear(); new_colors.clear()
            old_paths.clear(); old_colors.clear()
            update_frame_data(new_map, new_paths, new_colors)
            update_frame_data(old_map, old_paths, old_colors)
            paused = False

        # Step Forward
        if not paused:
            frame_idx += 1
            if frame_idx >= total_frames: break
            update_frame_data(new_map, new_paths, new_colors)
            update_frame_data(old_map, old_paths, old_colors)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()