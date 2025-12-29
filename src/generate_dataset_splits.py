import os
import re
import yaml
import random
import utils
from typing import List, Tuple, Set

# ==========================================
# CONFIGURATION
# ==========================================

# Paths relative to the project root
DATASET_ROOT = "dataset_yolo"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
OUTPUT_YAML = os.path.join(DATASET_ROOT, "data.yaml")

# Dataset Split Parameters
MIN_FRAME_GAP = 5         # Minimum gap between selected frames to reduce redundancy
TRAIN_SIZE = 1000         # Number of frames for the training set
VAL_SIZE = 150            # Number of frames for the validation set
TEST_SIZE = 200           # Number of frames for the test set
RANDOM_SEED = 42

# Selection Weights
RARE_CLASS_WEIGHT = 3     # Score multiplier for less common classes

# File Extensions
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Class IDs retrieved from utils
PED_ID = utils.CLASS_MAP.get("Pedestrian", 0)
BIKER_ID = utils.CLASS_MAP.get("Biker", 1)

# Set of rare class IDs for scoring logic
RARE_CLASSES = {
    utils.CLASS_MAP[k] for k in ["Car", "Bus", "Skater", "Cart"] 
    if k in utils.CLASS_MAP
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_frame_idx(filename: str) -> int:
    """
    Parses the numeric frame index from a filename.
    
    Args:
        filename (str): The name of the file (e.g., 'frame_000123.jpg').

    Returns:
        int: The extracted index, or -1 if no digits are found.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    matches = list(re.finditer(r'(\d+)', name))
    return int(matches[-1].group(1)) if matches else -1

def get_label_classes(label_path: str) -> List[int]:
    """
    Retrieves all class IDs present in a YOLO format label file.

    Args:
        label_path (str): Path to the .txt label file.

    Returns:
        List[int]: A list of integer class IDs found in the file.
    """
    if not os.path.isfile(label_path):
        return []
    
    class_ids = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.append(int(parts[0]))
    except ValueError:
        pass
    return class_ids

def write_split_file(file_path: str, paths: List[str]):
    """
    Writes a list of image paths to a text file.

    Args:
        file_path (str): Destination path for the text file.
        paths (List[str]): List of image file paths to write.
    """
    utils.ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(paths))

# ==========================================
# SELECTION LOGIC
# ==========================================

def select_training_candidates(all_images: List[str]) -> Tuple[List[str], Set[str]]:
    """
    Identifies optimal frames for training based on content and diversity.

    Selection Criteria:
    1. Frame must contain both 'Pedestrian' and 'Biker' classes.
    2. Frames must be separated by at least MIN_FRAME_GAP.
    3. Frames are ranked by a score derived from object count and class rarity.

    Args:
        all_images (List[str]): List of all available image paths.

    Returns:
        Tuple[List[str], Set[str]]: A tuple containing the selected list of paths
        and a set of those paths for efficient exclusion lookup.
    """
    candidates = []
    last_kept_idx = None

    print(f"Scanning {len(all_images)} images for training candidates...")

    for img_path in all_images:
        # Resolve label path
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS_DIR, base_name + ".txt")
        
        # Check class content
        classes = get_label_classes(label_path)
        if not classes:
            continue

        if not (PED_ID in classes and BIKER_ID in classes):
            continue

        # Enforce frame gap
        curr_idx = extract_frame_idx(img_path)
        if last_kept_idx is not None and curr_idx < last_kept_idx + MIN_FRAME_GAP:
            continue
        
        last_kept_idx = curr_idx

        # Calculate score
        score = len(classes)
        for cls_id in classes:
            if cls_id in RARE_CLASSES:
                score += RARE_CLASS_WEIGHT
        
        candidates.append((img_path, score))

    # Rank by score and select top K
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_paths = [x[0] for x in candidates[:TRAIN_SIZE]]
    
    return selected_paths, set(selected_paths)

def select_random_subset(pool: List[str], excluded: Set[str], count: int) -> List[str]:
    """
    Selects a random subset of images that are not present in the excluded set.

    Args:
        pool (List[str]): The total pool of available images.
        excluded (Set[str]): Images to exclude from selection.
        count (int): The number of images to select.

    Returns:
        List[str]: A list of randomly selected image paths.
    """
    available = [img for img in pool if img not in excluded]
    
    if len(available) < count:
        print(f"[Warning] Requested {count} images, but only {len(available)} are available.")
        count = len(available)
        
    random.shuffle(available)
    return available[:count]

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    random.seed(RANDOM_SEED)
    
    if not os.path.exists(IMAGES_DIR):
        print(f"[Error] Images directory not found: {IMAGES_DIR}")
        return

    # Aggregate all valid image files
    all_images = [
        os.path.join(IMAGES_DIR, f) 
        for f in os.listdir(IMAGES_DIR) 
        if f.lower().endswith(IMG_EXTS)
    ]
    all_images.sort(key=extract_frame_idx)

    # Generate Splits
    train_imgs, train_set = select_training_candidates(all_images)
    val_imgs = select_random_subset(all_images, train_set, VAL_SIZE)
    
    # Exclude both training and validation sets from test selection
    excluded_from_test = train_set.union(set(val_imgs))
    test_imgs = select_random_subset(all_images, excluded_from_test, TEST_SIZE)

    # Convert paths to relative format for portability
    # Assumes execution from project root; converts paths to match YOLO requirements
    normalize = lambda p: os.path.relpath(p, os.getcwd()).replace("\\", "/")
    
    train_txt = os.path.join(DATASET_ROOT, "train.txt")
    val_txt = os.path.join(DATASET_ROOT, "val.txt")
    test_txt = os.path.join(DATASET_ROOT, "test.txt")

    write_split_file(train_txt, [normalize(p) for p in train_imgs])
    write_split_file(val_txt, [normalize(p) for p in val_imgs])
    write_split_file(test_txt, [normalize(p) for p in test_imgs])

    # Generate Data YAML
    data_config = {
        "path": os.path.abspath(DATASET_ROOT).replace("\\", "/"), # Absolute path to dataset root
        "train": "train.txt", # Relative to 'path'
        "val": "val.txt",     # Relative to 'path'
        "test": "test.txt",   # Relative to 'path'
        "names": {v: k for k, v in utils.CLASS_MAP.items()}, # Dictionary map for YOLOv5/v8
        "nc": len(utils.CLASS_MAP)
    }

    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print("-" * 30)
    print(f"Train Set : {len(train_imgs)} images")
    print(f"Val Set   : {len(val_imgs)} images")
    print(f"Test Set  : {len(test_imgs)} images")
    print(f"Config    : {OUTPUT_YAML}")
    print("-" * 30)

if __name__ == "__main__":
    main()