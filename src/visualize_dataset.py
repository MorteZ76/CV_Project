import os
import cv2
import utils
from typing import Optional

# =========================
# CONFIGURATION
# =========================

DATASET_ROOT: str = "dataset_yolo"
IMAGES_DIR: str = os.path.join(DATASET_ROOT, "images")
LABELS_DIR: str = os.path.join(DATASET_ROOT, "labels")
FPS: int = 32

def get_processed_frame(img_path: str, label_path: str, idx: int, total: int, img_name: str) -> Optional[cv2.Mat]:
    """
    Loads an image, resizes it, and overlays YOLO labels and legends.

    Args:
        img_path (str): Path to the source image.
        label_path (str): Path to the corresponding YOLO label file.
        idx (int): Current index in the image sequence.
        total (int): Total number of images.
        img_name (str): Filename for display purposes.

    Returns:
        Optional[cv2.Mat]: The processed image ready for display, or None if loading fails.
    """
    # 1. Load Image
    img = cv2.imread(img_path)
    if img is None:
        return None

    # 2. Resize Image
    orig_h, orig_w = img.shape[:2]
    factor = utils.TARGET_DISPLAY_HEIGHT / orig_h
    new_w = int(orig_w * factor)
    new_h = int(orig_h * factor)
    img_resized = cv2.resize(img, (new_w, new_h))

    # 3. Draw Elements using Utils
    utils.draw_yolo_labels(img_resized, label_path)
    utils.draw_class_legend(img_resized)
    utils.draw_controls_legend(img_resized)
    
    # 4. Draw Info Overlay
    info_text = f"Image: {idx + 1}/{total} | {img_name}"
    utils.draw_text_with_background(img_resized, info_text, y_pos=30)

    return img_resized

def run_dataset_visualization() -> None:
    """
    Runs the interactive viewer for the generated YOLO dataset.
    """
    if not os.path.exists(IMAGES_DIR):
        print(f"[Error] Dataset directory not found: {IMAGES_DIR}")
        return

    # Filter for valid image files only
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".png"))])
    if not image_files:
        print("[Error] No images found.")
        return

    print(f"Loaded {len(image_files)} images.")
    print("Press 'q' to quit, 'p' to pause/play, 'j/k' to seek.")
    
    idx = 0
    total = len(image_files)
    last_idx = -1
    paused = True
    display_frame = None
    
    window_name = "Dataset Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        # Optimization: Only process frame if index changed
        if idx != last_idx:
            img_name = image_files[idx]
            img_path = os.path.join(IMAGES_DIR, img_name)
            # Infer label path from image name
            label_path = os.path.join(LABELS_DIR, os.path.splitext(img_name)[0] + ".txt")

            processed = get_processed_frame(img_path, label_path, idx, total, img_name)
            
            if processed is not None:
                display_frame = processed
                last_idx = idx
            else:
                # Skip corrupt images automatically
                idx = (idx + 1) % total
                continue

        if display_frame is not None:
            cv2.imshow(window_name, display_frame)

        # Handle Playback Control
        delay = 0 if paused else int(1000 / FPS)
        key = cv2.waitKey(delay) & 0xFF
        
        # Reuse logic by treating image index as frame number
        idx, paused, should_quit = utils.handle_playback_controls(
            key, idx, total, paused, FPS
        )

        if should_quit:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dataset_visualization()