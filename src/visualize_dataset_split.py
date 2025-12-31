import os
import cv2
import yaml
import random
import utils
from typing import List

# =========================
# CONFIGURATION
# =========================

# Path to the YOLO dataset configuration file
DATA_YAML_PATH = os.path.join("dataset_yolo", "data.yaml")

# Dataset split to visualize (e.g., 'train', 'val', 'test')
SPLIT = "train"
SHUFFLE = False
LIMIT = None
WINDOW_NAME = "YOLO Labels Viewer"

# =========================
# HELPER FUNCTIONS
# =========================

def load_image_list(yaml_path: str, split_name: str) -> List[str]:
    """
    Retrieves the list of image paths for a specific split from the dataset configuration.

    Args:
        yaml_path (str): Path to the YOLO data.yaml file.
        split_name (str): The dataset split key (e.g., 'train', 'val', 'test').

    Returns:
        List[str]: A list of file paths.
    """
    if not os.path.exists(yaml_path):
        return []

    # Parse the YAML configuration to find the text file listing image paths
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    txt_path = data.get(split_name)
    if not txt_path:
        return []
    
    # Resolve relative paths against the YAML file location
    if not os.path.isabs(txt_path):
        txt_path = os.path.join(os.path.dirname(yaml_path), txt_path)

    if not os.path.exists(txt_path):
        return []

    # Read all non-empty lines from the split text file
    with open(txt_path, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
        
    return paths

def infer_label_path(img_path: str) -> str:
    """
    Derives the expected label file path from the image file path.
    Assumes standard YOLO directory structure (images/ -> labels/).
    
    Args:
        img_path (str): The full path to the image file.
        
    Returns:
        str: The corresponding path to the .txt label file.
    """
    # Normalize path separators
    path = img_path.replace("\\", "/")
    
    # Swap directory convention from 'images' to 'labels'
    if "/images/" in path:
        path = path.replace("/images/", "/labels/")
    
    # Swap extension to .txt
    base, _ = os.path.splitext(path)
    return base + ".txt"

def count_labels_in_file(label_path: str) -> int:
    """Counts the number of object entries in a label file."""
    if not os.path.exists(label_path):
        return 0
    with open(label_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

# =========================
# MAIN VIEWER
# =========================

def run_viewer():
    """
    Iterates through the dataset split and displays images with overlayed ground truth labels.
    Uses consistent scaling and controls from utils.py.
    """
    # Load dataset index based on configuration
    images = load_image_list(DATA_YAML_PATH, SPLIT)
    
    if not images:
        print(f"No images found for split: {SPLIT}")
        return

    # Apply processing limits or shuffling
    if SHUFFLE:
        random.shuffle(images)
    
    if LIMIT:
        images = images[:LIMIT]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    idx = 0
    total = len(images)

    # Navigation Loop
    while 0 <= idx < total:
        img_path = images[idx]
        
        # Verify file existence before attempting load
        if not os.path.exists(img_path) and not os.path.exists(os.path.abspath(img_path)):
            idx += 1
            continue
        
        frame = cv2.imread(img_path)
        if frame is None:
            idx += 1
            continue

        # Resize to match target display height (consistent with visualize_dataset.py)
        # This ensures the window fits comfortably on standard screens
        orig_h, orig_w = frame.shape[:2]
        factor = utils.TARGET_DISPLAY_HEIGHT / orig_h
        new_w = int(orig_w * factor)
        new_h = int(orig_h * factor)
        frame_resized = cv2.resize(frame, (new_w, new_h))

        # Retrieve annotation data
        label_path = infer_label_path(img_path)
        box_count = count_labels_in_file(label_path)
        
        # Render visualization (bounding boxes)
        utils.draw_yolo_labels(frame_resized, label_path)

        # Overlay metadata (Split info, progress, file name)
        info_text = f"{SPLIT.upper()} [{idx+1}/{total}] | Boxes: {box_count} | {os.path.basename(img_path)}"
        utils.draw_text_with_background(frame_resized, info_text, y_pos=30)
        
        # Overlay Controls (Matching utils.handle_playback_controls)
        controls_text = "[K] Next | [J] Previous | [Q] Quit"
        utils.draw_text_with_background(frame_resized, controls_text, y_pos=70)

        cv2.imshow(WINDOW_NAME, frame_resized)

        # Input handling: Wait indefinitely for a key press
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('j'): # Previous Image
            idx = max(0, idx - 1)
        elif key == ord('k'): # Next Image
            idx += 1
        else:
            # Default to next on other keys (optional, matches standard viewers)
            idx += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_viewer()