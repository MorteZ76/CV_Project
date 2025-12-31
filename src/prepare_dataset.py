import os
import utils
from typing import List, Dict

# =========================
# CONFIGURATION
# =========================

# Define source videos to process
# 'id' corresponds to the specific scene identifier in utils.get_video_paths
SOURCES: List[Dict[str, object]] = [
    {"name": "video0", "id": 0},
    {"name": "video3", "id": 3},
]

# Output Configuration
DATASET_ROOT: str = "dataset_yolo"  
IMAGES_DIR: str = os.path.join(DATASET_ROOT, "images")
LABELS_DIR: str = os.path.join(DATASET_ROOT, "labels")

# Sampling parameters
FRAME_STEP: int = 1      # Save every k-th frame (1 = all frames, 10 = decimate by 10)
IMG_EXT: str = ".jpg"    # Output image format

def run_dataset_preparation() -> None:
    """
    Orchestrates the creation of a YOLO-formatted dataset.
    
    This process involves two main steps:
    1. Extracting individual frames from source videos.
    2. Generating normalized YOLO label files (.txt) matching the extracted frames.
    """
    
    # --- Step 1: Extract Frames ---
    # Decodes video files and saves frames to disk.
    # Returns a mapping dict to link specific video frames to their saved file paths.
    print("Step 1: Extracting Frames...")
    mapping, video_info = utils.extract_dataset_frames(
        SOURCES, IMAGES_DIR, FRAME_STEP, IMG_EXT
    )
    
    # --- Step 2: Generate Labels ---
    # specific YOLO format (class_id center_x center_y width height)
    # Uses the mapping from Step 1 to ensure labels correspond exactly to saved images.
    print("\nStep 2: Generating YOLO Labels...")
    utils.generate_yolo_labels(mapping, video_info, LABELS_DIR)
    
    # --- Completion Summary ---
    print("\n[Done] Dataset preparation complete.")
    print(f"Images stored in: {os.path.abspath(IMAGES_DIR)}")
    print(f"Labels stored in: {os.path.abspath(LABELS_DIR)}")

if __name__ == "__main__":
    run_dataset_preparation()