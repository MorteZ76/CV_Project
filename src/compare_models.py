"""
YOLO Model Comparison Tool
--------------------------
This script evaluates and visualizes the performance of two YOLO models (Old vs. New)
against a Ground Truth dataset. It computes micro-averaged metrics (Precision, Recall, F1)
and provides an interactive side-by-side visualization of detections.

Features:
- Side-by-side visualization: Ground Truth | Old Model | New Model
- Interactive controls: Pause (p), Next (d), Prev (a), Quit (q)
- Class-aware metric calculation
- Dynamic class name resolution from model weights
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Define base directory relative to this script location
BASE_DIR = Path(__file__).parent.parent if "__file__" in locals() else Path.cwd()

# Dataset Paths
IMAGES_PATH = BASE_DIR / "dataset_yolo" / "images"
LABELS_PATH = BASE_DIR / "dataset_yolo" / "labels"

# Model Weights Paths
# Defines the specific model checkpoints to compare
MODELS = {
    "old_model": str(BASE_DIR / "old_model" / "models" / "sdd_yolov8s" / "weights" / "best.pt"),
    "new_model": str(BASE_DIR / "models" / "sdd_yolov8s" / "weights" / "best.pt")
}

# Evaluation Hyperparameters
IOU_THRESHOLD = 0.50      # Minimum overlap required to consider a detection a True Positive
CONF_THRESHOLD = 0.70     # Confidence threshold for filtering weak detections
IMGSZ = 960               # Input resolution size for model inference
MAX_FRAMES = 300          # Limit total frames to process for faster iteration
FRAME_STEP = 50           # Process every Nth frame (subsampling)

# Visualization Settings
DISPLAY_SCALE = 0.8       # Resize factor for the GUI window (0.8 = 80% of original size)
TEXT_FONT_SCALE = 0.6     # Font size for bounding box labels
TEXT_THICKNESS = 2        # Thickness for text drawing
# ==============================================================================


def load_gt_yolo(label_file, img_w, img_h):
    """
    Parses a standard YOLO format text file into bounding boxes.
    
    Args:
        label_file (str): Path to the .txt label file.
        img_w (int): Image width.
        img_h (int): Image height.
        
    Returns:
        np.array: Array of boxes in format [[x1, y1, x2, y2, cls], ...]
    """
    gt_boxes = []
    if not os.path.isfile(label_file):
        return np.array(gt_boxes, dtype=float)

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            # YOLO format: class x_center y_center width height (normalized 0-1)
            cls, xc, yc, w, h = map(float, parts)
            
            # Convert normalized center-coordinates to absolute corner-coordinates (x1, y1, x2, y2)
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            
            gt_boxes.append([x1, y1, x2, y2, int(cls)])
            
    return np.array(gt_boxes, dtype=float)


def draw_boxes(img, boxes, class_names, color=(0, 255, 0)):
    """
    Draws bounding boxes and class labels on an image.
    
    Style:
    - No background box behind text.
    - Text color matches the bounding box color.
    - Text is placed slightly above the top-left corner.
    """
    for b in boxes:
        x1, y1, x2, y2, cls = map(int, b[:5])
        
        # Resolve class name from ID, fallback to ID if missing
        cls_name = class_names[cls] if cls in class_names else str(cls)
        
        # 1. Draw Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 2. Draw Text (Color matches box)
        # Ensure text doesn't go off the top edge of the image
        y_text = max(y1 - 5, 15)
        cv2.putText(img, cls_name, (x1, y_text), 
                    cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, color, TEXT_THICKNESS)


def evaluate_and_visualize(models_dict, images_dir, labels_dir, 
                           conf_thres=0.25, iou_thres=0.5, imgsz=960, 
                           max_frames=300, frame_step=50):
    """
    Main evaluation loop. Processes images, runs inference, calculates metrics,
    and handles the interactive visualization loop.
    """
    # 1. Prepare Image List (Subsampled)
    all_images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    image_files = all_images[::frame_step][:max_frames]

    # 2. Load Models & Extract Class Names
    # Assuming both models share the same class mapping (names)
    print(f"Loading models...")
    loaded_models = {name: YOLO(path) for name, path in models_dict.items()}
    class_names = list(loaded_models.values())[0].names 

    # 3. Initialize Metric Accumulators
    metrics_data = {
        name: {"tp": 0, "fp": 0, "fn": 0} 
        for name in models_dict
    }

    # 4. Setup Visualization Window
    window_name = "Comparison: GT | Old | New"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    i = 0
    paused = False

    print(f"Starting evaluation on {len(image_files)} frames...")
    print("Controls: [P]ause/Play | [D] Next | [A] Prev | [Q] Quit")

    while i < len(image_files):
        img_name = image_files[i]
        img_path = os.path.join(images_dir, img_name)
        # Infer corresponding label file path
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        
        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            i += 1
            continue
        h, w = img.shape[:2]

        # Load Ground Truth boxes for this frame
        gt_boxes = load_gt_yolo(label_path, w, h)

        # Run Inference & Process Detections
        dets_per_model = {}
        
        for name, model in loaded_models.items():
            # Run YOLO inference
            results = model.predict(img, imgsz=imgsz, conf=conf_thres, verbose=False)[0]
            det_boxes = []
            
            if results.boxes is not None:
                for b in results.boxes:
                    cls = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    if conf < conf_thres:
                        continue
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    det_boxes.append([x1, y1, x2, y2, cls])
            
            dets_per_model[name] = det_boxes

            # --- Calculate Frame-Level Metrics ---
            # Prepare tensors for vectorized IoU calculation
            gt_t = torch.tensor(gt_boxes[:, :4]) if len(gt_boxes) > 0 else torch.empty((0,4))
            det_t = torch.tensor([b[:4] for b in det_boxes]) if len(det_boxes) > 0 else torch.empty((0,4))

            # Handle edge cases (No GT or No Detections)
            if len(gt_boxes) == 0:
                metrics_data[name]["fp"] += len(det_boxes)
                continue
            if len(det_boxes) == 0:
                metrics_data[name]["fn"] += len(gt_boxes)
                continue

            # Compute IoU Matrix between all Dets and all GTs
            ious = box_iou(det_t, gt_t).numpy()
            
            # Greedy Matching: Match detections to GTs based on highest IoU
            matched_gt = set()
            for k, det_box in enumerate(det_boxes):
                det_cls = int(det_box[4])
                det_iou = ious[k]
                
                # Filter GTs to only consider those with the same class
                same_cls_idxs = [j for j, gt in enumerate(gt_boxes) if int(gt[4]) == det_cls]
                
                if not same_cls_idxs:
                    metrics_data[name]["fp"] += 1
                    continue
                
                # Find best matching GT among those with the same class
                same_cls_ious = [(j, det_iou[j]) for j in same_cls_idxs]
                best_match_idx, max_iou = max(same_cls_ious, key=lambda x:x[1])
                
                # Check overlap threshold and ensure GT hasn't been matched yet (1-to-1 matching)
                if max_iou >= iou_thres and best_match_idx not in matched_gt:
                    metrics_data[name]["tp"] += 1
                    matched_gt.add(best_match_idx)
                else:
                    metrics_data[name]["fp"] += 1
            
            # Unmatched GTs count as False Negatives
            metrics_data[name]["fn"] += len(gt_boxes) - len(matched_gt)

        # ----------------------------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------------------------
        img_gt = img.copy()
        img_old = img.copy()
        img_new = img.copy()

        # Draw Boxes (GT: Green, Old: Blue, New: Red)
        draw_boxes(img_gt, gt_boxes, class_names, color=(0, 255, 0))
        draw_boxes(img_old, dets_per_model["old_model"], class_names, color=(0, 0, 255))
        draw_boxes(img_new, dets_per_model["new_model"], class_names, color=(255, 0, 0))

        # Add Titles for clarity
        title_font_scale = 1.2
        title_thickness = 3
        cv2.putText(img_gt, "Ground Truth", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (0, 255, 0), title_thickness)
        cv2.putText(img_old, "Old Model", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (0, 0, 255), title_thickness)
        cv2.putText(img_new, "New Model", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (255, 0, 0), title_thickness)

        # Resize for display and Concatenate horizontally
        h_disp, w_disp = int(h * DISPLAY_SCALE), int(w * DISPLAY_SCALE)
        combined_view = np.hstack((
            cv2.resize(img_gt, (w_disp, h_disp)),
            cv2.resize(img_old, (w_disp, h_disp)),
            cv2.resize(img_new, (w_disp, h_disp))
        ))

        cv2.imshow(window_name, combined_view)

        # ----------------------------------------------------------------------
        # PLAYBACK CONTROLS
        # ----------------------------------------------------------------------
        wait_time = 0 if paused else 1
        key = cv2.waitKey(wait_time) & 0xFF

        if key == 27 or key == ord('q'):    # ESC or q to Quit
            break
        elif key == ord('p'):               # P to Pause/Play
            paused = not paused
        elif key == ord('d') or key == 83:  # D / Right Arrow for Next
            i += 1
            paused = True 
        elif key == ord('a') or key == 81:  # A / Left Arrow for Prev
            i = max(0, i - 1)
            paused = True 
        else:
            if not paused:
                i += 1

        print(f"Processed: [{i}/{len(image_files)}] | Status: {'PAUSED' if paused else 'PLAYING'}")

    cv2.destroyAllWindows()

    # --------------------------------------------------------------------------
    # METRICS SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("📊 MICRO-AVERAGED (CLASS-AWARE) RESULTS")
    print("="*50)
    print(f"{'Metric':<12} {'Old Model':>12} {'New Model':>12}")
    print("-" * 50)

    final_metrics = {}
    for name in models_dict:
        tp, fp, fn = metrics_data[name]["tp"], metrics_data[name]["fp"], metrics_data[name]["fn"]
        
        # Avoid division by zero with epsilon
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy = tp / (tp + fp + fn + 1e-9)
        
        final_metrics[name] = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    # Print comparative table
    for key in ["precision", "recall", "f1", "accuracy"]:
        print(f"{key.capitalize():<12} {final_metrics['old_model'][key]:>12.4f} {final_metrics['new_model'][key]:>12.4f}")
    
    print("-" * 50)
    for key in ["tp", "fp", "fn"]:
        print(f"{key.upper():<12} {metrics_data['old_model'][key]:>12} {metrics_data['new_model'][key]:>12}")
    print("="*50)


if __name__ == "__main__":
    evaluate_and_visualize(
        MODELS, 
        str(IMAGES_PATH), 
        str(LABELS_PATH),
        conf_thres=CONF_THRESHOLD, 
        iou_thres=IOU_THRESHOLD,
        imgsz=IMGSZ, 
        max_frames=MAX_FRAMES, 
        frame_step=FRAME_STEP
    )