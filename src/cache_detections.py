import os
import cv2
import pandas as pd
import torch  # Added to check for GPU availability
from ultralytics import YOLO
from tqdm import tqdm
import utils

# ==========================================
# CONFIGURATION
# ==========================================

# Video Identifiers to process (0 for video0, 1 for video3)
VIDEO_IDS = [0, 1]

# Model Weights
# Ensure these paths are correct relative to where you run the script
YOLO_MODELS = {
    "new_model": os.path.join("models", "sdd_yolov8s", "weights", "best.pt"),
    "old_model": os.path.join("old_model", "models", "sdd_yolov8s_resume", "weights", "best.pt")
}

# Output Configuration
OUTPUT_DIR = os.path.join("outputs", "detections_cache")

# Inference Hyperparameters
CONF_THRESHOLD = 0.25     # Minimum confidence score
MAX_DETECTIONS = 3000     # Maximum detections per frame
FRAME_STEP = 1            # Process every Nth frame
IOU_THRESHOLD = 1.0       # IoU threshold (1.0 disables NMS filtering)
AGNOSTIC_NMS = True       # Class-agnostic NMS

# Automatic Device Selection
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==========================================
# CORE LOGIC
# ==========================================

def cache_video_detections(video_path: str, model_path: str, output_parquet: str) -> None:
    """
    Runs YOLOv8 inference on a video and saves detections to a Parquet file.
    """
    utils.ensure_dir(os.path.dirname(output_parquet))
    
    # Validation
    if not os.path.exists(model_path):
        print(f"[Error] Model weights not found: {model_path}")
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[Error] Failed to initialize model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    detection_data = []

    print(f"\n[Processing] Video: {os.path.basename(video_path)}")
    print(f" - Model: {os.path.basename(model_path)}")
    print(f" - Device: {DEVICE}")
    print(f" - Total Frames: {total_frames}")

    # Inference Loop
    for frame_idx in tqdm(range(total_frames), desc="Inference", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % FRAME_STEP != 0:
            continue

        # Run Prediction
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            agnostic_nms=AGNOSTIC_NMS,
            max_det=MAX_DETECTIONS,
            verbose=False,
            device=DEVICE  # Dynamically set to 'cpu' or 0
        )[0]

        # Extract Boxes
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls_id in zip(boxes, confs, classes):
                detection_data.append({
                    "frame": frame_idx,
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "score": float(score),
                    "cls": int(cls_id)
                })

    cap.release()

    # Serialization
    if not detection_data:
        print(f"[Warning] No detections found for {os.path.basename(video_path)}.")
        return

    df = pd.DataFrame(detection_data)
    cols = ["frame", "x1", "y1", "x2", "y2", "score", "cls"]
    df = df[cols]
    
    df.to_parquet(output_parquet, index=False)
    print(f"[Success] Saved {len(df)} detections to: {output_parquet}")

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_caching():
    """
    Iterates through all defined videos and models, caching detection results.
    """
    for video_id in VIDEO_IDS:
        # Retrieve video path dynamically using utils
        try:
            video_path, _ = utils.get_video_paths(video_id)
        except Exception as e:
            print(f"[Error] Could not resolve path for video ID {video_id}: {e}")
            continue

        if not os.path.exists(video_path):
            print(f"[Error] Video file does not exist: {video_path}")
            continue

        # Process all models for this video
        for model_name, weight_path in YOLO_MODELS.items():
            # Output filename includes both model name and video ID to avoid overwrites
            output_filename = f"{model_name}_video{video_id}_detections.parquet"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            cache_video_detections(video_path, weight_path, output_path)

if __name__ == "__main__":
    run_caching()