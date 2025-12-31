import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

# Target height for resizing video frames and images for display
TARGET_DISPLAY_HEIGHT: int = 960

# Mapping of class names to integer IDs (compatible with YOLO format)
CLASS_MAP: Dict[str, int] = {
    "Pedestrian": 0,
    "Biker": 1,
    "Car": 2,
    "Bus": 3,
    "Skater": 4,
    "Cart": 5,
}

# Visualization colors in (B, G, R) format for OpenCV
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Pedestrian": (0, 255, 0),    # Green
    "Biker":      (0, 255, 255),  # Yellow
    "Car":        (255, 0, 0),    # Blue
    "Bus":        (0, 128, 255),  # Orange
    "Skater":     (255, 0, 255),  # Magenta
    "Cart":       (255, 128, 0),  # Light Blue
}

# ==========================================
# FILE I/O UTILITIES
# ==========================================

def ensure_dir(path: str) -> None:
    """Creates a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def get_video_paths(video_num: int, base_dir: str = "data") -> Tuple[str, str]:
    """
    Retrieves the video and annotation file paths for a specific dataset scene.

    Args:
        video_num (int): Identifier for the video (0 for 'video0', 3 for 'video3').
        base_dir (str): Root directory containing the dataset videos.

    Returns:
        Tuple[str, str]: A tuple containing (video_file_path, annotation_file_path).

    Raises:
        ValueError: If an invalid video_num is provided.
    """
    if video_num == 0:
        video_dir = os.path.join(base_dir, "video0")
    elif video_num == 3:
        video_dir = os.path.join(base_dir, "video3")
    else:
        raise ValueError("Video number must be 0 (video0) or 3 (video3)")
    
    return os.path.join(video_dir, "video.mp4"), os.path.join(video_dir, "annotations.txt")


def load_annotations(annotation_path: str) -> pd.DataFrame:
    """
    Loads Ground Truth annotations from the Stanford Drone Dataset text format.

    Args:
        annotation_path (str): Path to the annotation text file.

    Returns:
        pd.DataFrame: DataFrame containing columns ['track_id', 'xmin', 'ymin', ..., 'label'].
                      Returns an empty DataFrame if the file is not found.
    """
    # Define expected column names for the dataset
    columns = ['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 
               'lost', 'occluded', 'generated', 'label']
    
    if not os.path.exists(annotation_path):
        print(f"[Error] Annotation file not found: {annotation_path}")
        return pd.DataFrame(columns=columns)

    try:
        # Load whitespace-separated values
        df = pd.read_csv(annotation_path, header=None, names=columns, sep=r'\s+')
        # Clean label strings (remove quotes)
        df['label'] = df['label'].str.replace('"', '') 
        return df
    except Exception as e:
        print(f"[Error] Failed to read annotations: {e}")
        return pd.DataFrame(columns=columns)

# ==========================================
# METRICS & MATH UTILITIES
# ==========================================

def calculate_mse(predicted_traj: List[float], gt_traj: List[float]) -> float:
    """
    Calculates the Mean Squared Error (MSE) between two trajectory arrays.

    Args:
        predicted_traj (List[float]): List or array of predicted points.
        gt_traj (List[float]): List or array of ground truth points.

    Returns:
        float: The calculated MSE, or infinity if input lists are empty.
    """
    if len(predicted_traj) == 0 or len(gt_traj) == 0:
        return float('inf')

    # Truncate to the length of the shorter sequence to allow comparison
    min_len = min(len(predicted_traj), len(gt_traj))
    traj_a = np.array(predicted_traj[:min_len])
    traj_b = np.array(gt_traj[:min_len])
    
    return float(np.mean((traj_a - traj_b) ** 2))


def calculate_scaling(orig_w: int, orig_h: int, annotations: pd.DataFrame) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """
    Calculates dimensions and scaling factors to fit video within the target display height.
    
    It adjusts scaling based on whether annotations fall outside the video frame boundaries,
    which can happen in some datasets where coordinate systems differ.

    Args:
        orig_w (int): Original width of the video.
        orig_h (int): Original height of the video.
        annotations (pd.DataFrame): The loaded annotations dataframe.

    Returns:
        Tuple:
            - (new_w, new_h): The target display resolution.
            - (scale_x, scale_y): Scaling factors for coordinates.
    """
    # Determine base resize factor for display purposes
    resize_factor = TARGET_DISPLAY_HEIGHT / orig_h
    new_w = int(orig_w * resize_factor)
    new_h = int(orig_h * resize_factor)

    # Check if annotations exceed video bounds and adjust scaling factors accordingly
    if not annotations.empty:
        max_ann_x = annotations[['xmin', 'xmax']].max().max()
        max_ann_y = annotations[['ymin', 'ymax']].max().max()
        
        # Calculate scale based on the larger of the video or annotation bounds
        scale_x = (orig_w / max_ann_x) * resize_factor if max_ann_x > orig_w else resize_factor
        scale_y = (orig_h / max_ann_y) * resize_factor if max_ann_y > orig_h else resize_factor
    else:
        scale_x, scale_y = resize_factor, resize_factor
        
    return (new_w, new_h), (scale_x, scale_y)


def bbox_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    Converts standard bounding box coordinates to normalized YOLO format.

    Args:
        xmin, ymin, xmax, ymax (float): Bounding box coordinates (absolute).
        img_w, img_h (int): Image dimensions.

    Returns:
        Tuple[float, float, float, float]: Normalized (center_x, center_y, width, height).
    """
    # Calculate center point
    x_c = (xmin + xmax) / 2.0
    y_c = (ymin + ymax) / 2.0
    
    # Calculate dimensions
    w = xmax - xmin
    h = ymax - ymin
    
    # Normalize by image size
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h

# ==========================================
# VISUALIZATION UTILITIES
# ==========================================

def get_class_color(label: str) -> Tuple[int, int, int]:
    """Returns the (B, G, R) color tuple for a given class label string."""
    clean_label = label.strip('"')
    return CLASS_COLORS.get(clean_label, (255, 255, 255))


def get_state_color(lost: int, occluded: int, generated: int) -> Tuple[int, int, int]:
    """
    Returns a color code representing the tracking state.
    
    Priority: Lost > Occluded & Generated > Occluded > Generated > Normal.
    Used to visually distinguish annotation reliability.
    """
    if lost == 1:
        return (128, 128, 128)  # Gray: Object is lost
    elif occluded == 1 and generated == 1:
        return (255, 165, 0)    # Orange: Occluded and inferred
    elif occluded == 1:
        return (0, 0, 255)      # Red: Occluded
    elif generated == 1:
        return (255, 0, 255)    # Purple: Generated/Interpolated
    else:
        return (0, 255, 0)      # Green: Normal visibility


def draw_text_with_background(frame: np.ndarray, text: str, y_pos: int = 30) -> None:
    """Draws centered text with a black background box at the specified Y position."""
    h, w = frame.shape[:2]
    # Calculate text size to center it
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    text_x = (w - text_w) // 2
    
    # Draw background rectangle
    cv2.rectangle(frame, 
                 (text_x - 10, y_pos - text_h - 10), 
                 (text_x + text_w + 10, y_pos + baseline + 5), 
                 (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (text_x, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def draw_timestamp(frame: np.ndarray, frame_number: int, fps: int) -> None:
    """Calculates time from frame number and overlays a timestamp on the frame."""
    time_sec = frame_number / fps
    time_text = f"Frame: {frame_number} | Time: {int(time_sec // 60):02d}:{int(time_sec % 60):02d}"
    draw_text_with_background(frame, time_text, y_pos=30)


def draw_annotations_on_frame(frame: np.ndarray, frame_anns: pd.DataFrame, scale_x: float, scale_y: float) -> None:
    """
    Draws bounding boxes, IDs, and labels for all objects in the provided dataframe slice.
    """
    h, w = frame.shape[:2]

    for _, box in frame_anns.iterrows():
        # Scale coordinates to display resolution
        xmin = int(box['xmin'] * scale_x)
        ymin = int(box['ymin'] * scale_y)
        xmax = int(box['xmax'] * scale_x)
        ymax = int(box['ymax'] * scale_y)

        # Clamp to boundaries to prevent drawing errors
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)

        state_color = get_state_color(box['lost'], box['occluded'], box['generated'])
        class_color = get_class_color(box['label'])

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), state_color, 2)

        # Draw ID and Label
        label_y = max(ymin - 20, 20)
        cv2.putText(frame, f"ID: {int(box['track_id'])}", 
                   (xmin, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, box['label'].strip(), 
                   (xmin, label_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)


def draw_yolo_labels(frame: np.ndarray, label_path: str) -> None:
    """
    Reads a YOLO format label file and draws denormalized bounding boxes on the frame.
    Useful for visualizing the generated dataset.
    """
    if not os.path.exists(label_path):
        return

    # Create reverse mapping for IDs to names
    id_to_class = {v: k for k, v in CLASS_MAP.items()}
    h, w = frame.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5: 
            continue
        
        cls_id = int(parts[0])
        n_xc, n_yc, n_bw, n_bh = map(float, parts[1:])

        # Denormalize coordinates (0-1 range to pixel values)
        xc, yc = n_xc * w, n_yc * h
        bw, bh = n_bw * w, n_bh * h

        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)

        class_name = id_to_class.get(cls_id, str(cls_id))
        color = CLASS_COLORS.get(class_name, (0, 255, 0))

        # Draw Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label with background for readability
        label_text = f"{class_name}"
        (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Position label text above or inside box
        box_y = max(y1 - 20, 5)
        text_y = max(y1 - 5, 20)
        
        cv2.rectangle(frame, (x1, box_y), (x1 + txt_w, box_y + 20), color, -1)
        cv2.putText(frame, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_legends(frame: np.ndarray) -> None:
    """Wrapper to draw state, class, and control legends on the frame."""
    draw_state_legend(frame)
    draw_class_legend(frame)
    draw_controls_legend(frame)


def draw_state_legend(frame: np.ndarray) -> None:
    """Draws the tracking state color legend (e.g., Occluded, Lost) at top right."""
    start_x = frame.shape[1] - 150
    start_y = 30
    line_height = 20
    
    legend_items = [
        ("Normal", (0, 255, 0)),
        ("Lost", (128, 128, 128)),
        ("Occluded", (0, 0, 255)),
        ("Generated", (255, 0, 255)),
        ("Oc + Generated", (255, 165, 0))
    ]
    
    # Draw background
    cv2.rectangle(frame, (start_x - 10, start_y - 20), 
                 (frame.shape[1] - 10, start_y + len(legend_items) * line_height),
                 (0, 0, 0), -1)
    
    cv2.putText(frame, "State:", (start_x - 10, start_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw items
    for i, (text, color) in enumerate(legend_items):
        y = start_y + i * line_height + 15
        cv2.rectangle(frame, (start_x, y - 10), (start_x + 15, y + 2), color, -1)
        cv2.putText(frame, text, (start_x + 25, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_class_legend(frame: np.ndarray) -> None:
    """Draws the object class color legend at top left."""
    start_x = 30
    start_y = 30
    line_height = 20
    
    # Draw background
    cv2.rectangle(frame, (start_x - 10, start_y - 20), 
                 (start_x + 130, start_y + len(CLASS_COLORS) * line_height),
                 (0, 0, 0), -1)
    
    cv2.putText(frame, "Classes:", (start_x - 10, start_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw items
    for i, (text, color) in enumerate(CLASS_COLORS.items()):
        y = start_y + i * line_height + 15
        cv2.rectangle(frame, (start_x, y - 10), (start_x + 15, y + 2), color, -1)
        cv2.putText(frame, text, (start_x + 25, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_controls_legend(frame: np.ndarray) -> None:
    """Draws the keyboard control instructions at the bottom left."""
    start_x = 30
    start_y = frame.shape[0] - 130
    line_height = 20
    
    controls = [
        ("P", "Play/Pause"),
        ("J/K", "Prev/Next Frame"),
        ("I/O", "-/+ 10 Sec"),
        ("Q", "Quit"),
    ]
    
    # Draw background
    cv2.rectangle(frame, (start_x - 10, start_y - 20), 
                 (start_x + 200, start_y + len(controls) * line_height),
                 (0, 0, 0), -1)
    
    cv2.putText(frame, "Controls:", (start_x - 10, start_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw items
    for i, (key, desc) in enumerate(controls):
        y = start_y + i * line_height + 15
        cv2.putText(frame, f"[{key}] {desc}", (start_x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# ==========================================
# INTERACTIVE CONTROL UTILITIES
# ==========================================

def handle_playback_controls(key: int, frame_number: int, total_frames: int, paused: bool, fps: int) -> Tuple[int, bool, bool]:
    """
    Processes keyboard input to update playback state and frame position.

    Args:
        key (int): Key code from cv2.waitKey().
        frame_number (int): Current frame index.
        total_frames (int): Total frames in the video/dataset.
        paused (bool): Current pause state.
        fps (int): Frames per second (used for time jumping).

    Returns:
        Tuple[int, bool, bool]: (updated_frame_number, updated_paused_state, should_quit)
    """
    should_quit = False

    if key == ord('q'): 
        should_quit = True
    elif key == ord('p'): 
        paused = not paused
    elif key == ord('k'): # Next Frame
        frame_number = min(frame_number + 1, total_frames - 1)
        paused = True
    elif key == ord('j'): # Previous Frame
        frame_number = max(frame_number - 1, 0)
        paused = True
    elif key == ord('o'): # Jump Forward (approx 10 sec)
        frame_number = min(frame_number + (fps * 10), total_frames - 1)
    elif key == ord('i'): # Jump Backward (approx 10 sec)
        frame_number = max(frame_number - (fps * 10), 0)
    else:
        # Auto-advance if not paused
        if not paused:
            frame_number += 1
            if frame_number >= total_frames: 
                frame_number = 0 
                
    return frame_number, paused, should_quit

# ==========================================
# DATASET PREPARATION UTILITIES
# ==========================================

def extract_dataset_frames(sources: List[Dict], images_dir: str, frame_step: int = 1, img_ext: str = ".jpg") -> Tuple[Dict, Dict]:
    """
    Extracts frames from video files and saves them as images to disk.

    Args:
        sources (List[Dict]): List of dicts containing 'name' and 'id' of videos.
        images_dir (str): Destination directory for images.
        frame_step (int): Save every k-th frame (1 = all frames).
        img_ext (str): Image extension (e.g., '.jpg').

    Returns:
        Tuple[Dict, Dict]: 
            - mapping: (video_name, original_frame_idx) -> absolute_image_path
            - video_info: Metadata per video {name: {width, height, ann_path}}
    """
    ensure_dir(images_dir)
    mapping = {}    
    video_info = {} 
    global_img_idx = 1 

    for source in sources:
        name = source["name"]
        video_id = source["id"]
        
        # Resolve paths
        video_path, ann_path = get_video_paths(video_id)
        print(f"Processing {name} from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames_saved = 0
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            # Save frame if it matches the step interval
            if current_frame % frame_step == 0:
                filename = f"frame_{global_img_idx:06d}{img_ext}"
                out_path = os.path.join(images_dir, filename)
                
                cv2.imwrite(out_path, frame)
                mapping[(name, current_frame)] = out_path
                global_img_idx += 1
                frames_saved += 1
            
            current_frame += 1

        cap.release()
        
        video_info[name] = {"width": width, "height": height, "ann_path": ann_path}
        print(f"[{name}] Extracted {frames_saved} frames.")

    return mapping, video_info


def generate_yolo_labels(mapping: Dict, video_info: Dict, labels_dir: str) -> None:
    """
    Converts internal annotations to YOLO format (normalized cx, cy, w, h) and saves .txt files.
    
    Iterates through extracted frames and matches them with Ground Truth data.
    """
    ensure_dir(labels_dir)
    labels_written = 0

    for name, info in video_info.items():
        ann_path = info["ann_path"]
        vid_w = info["width"]
        vid_h = info["height"]
        
        # Load raw annotations
        df = load_annotations(ann_path)
        if df.empty: 
            continue

        # Calculate scaling (handles coordinate mismatch in dataset if any)
        max_x = df[['xmin', 'xmax']].max().max()
        max_y = df[['ymin', 'ymax']].max().max()
        
        scale_x = vid_w / max_x if max_x > vid_w else 1.0
        scale_y = vid_h / max_y if max_y > vid_h else 1.0

        # Filter only active (non-lost) tracks
        active_tracks = df[df['lost'] == 0]
        grouped = active_tracks.groupby('frame')

        # Process each frame for which we have an image
        for frame_idx, group in grouped:
            if (name, frame_idx) not in mapping: 
                continue

            img_path = mapping[(name, frame_idx)]
            # Match label filename to image filename
            label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)

            yolo_lines = []
            for _, row in group.iterrows():
                label_name = row['label']
                if label_name not in CLASS_MAP: 
                    continue

                class_id = CLASS_MAP[label_name]
                
                # Scale and clamp coordinates to image dimensions
                x1 = max(0, min(vid_w, row['xmin'] * scale_x))
                y1 = max(0, min(vid_h, row['ymin'] * scale_y))
                x2 = max(0, min(vid_w, row['xmax'] * scale_x))
                y2 = max(0, min(vid_h, row['ymax'] * scale_y))

                # Convert to normalized YOLO format
                cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, vid_w, vid_h)
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Write label file if valid objects exist
            if yolo_lines:
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                labels_written += 1

    print(f"Total Label Files Created: {labels_written}")