"""
extract_paths.py
----------------
Analyzes object trajectories to determine their movement patterns across specific 
regions of the video frame (e.g., Top-Left -> Middle-Center -> Bottom-Right).

Features:
- Divides the video frame into a 3x3 grid (9 regions).
- Maps each trajectory point to a specific region.
- Compresses trajectory data into a sequence of visited regions (path steps).
- Supports batch processing of multiple models (e.g., NEW vs OLD).
"""

import cv2
import csv
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from enum import Enum
from typing import Dict, Tuple, List, Optional

# Local imports
try:
    import utils
except ImportError:
    raise ImportError("utils.py not found. Ensure it is in the same directory.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

VIDEO_NUM = 0  # Select Video: 0 or 3

# Resolve Base Directory (Go back one level from src if script is in src/)
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
TRAJ_DIR = OUTPUTS_DIR / "trajectories"
PATHS_DIR = OUTPUTS_DIR / "paths"

# Ensure output directory exists
utils.ensure_dir(str(PATHS_DIR))

# ==============================================================================
# REGION DEFINITIONS
# ==============================================================================

class Region(Enum):
    """Enumeration of the 9 grid regions in the video frame."""
    TOP_LEFT = 0
    TOP_MIDDLE = 1
    TOP_RIGHT = 2
    MIDDLE_LEFT = 3
    MIDDLE_MIDDLE = 4
    MIDDLE_RIGHT = 5
    BOTTOM_LEFT = 6
    BOTTOM_MIDDLE = 7
    BOTTOM_RIGHT = 8

def get_region_display_name(region: Region) -> str:
    """Returns a formatted string name for the region (e.g., 'Top-Left')."""
    return region.name.replace('_', '-').title()


class PathExtractor:
    """
    Handles the logic of mapping trajectory coordinates to defined regions
    and extracting sequential movement paths.
    """

    def __init__(self, video_path: str, traj_csv_path: Path, output_csv_path: Path):
        """
        Initialize the PathExtractor.

        Args:
            video_path (str): Path to the source video file (for dimensions).
            traj_csv_path (Path): Path to the input trajectories CSV.
            output_csv_path (Path): Path where the extracted paths CSV will be saved.
        """
        self.video_path = video_path
        self.traj_csv_path = traj_csv_path
        self.output_csv_path = output_csv_path
        
        # Frame dimensions
        self.frame_width = 0
        self.frame_height = 0
        self.region_w = 0
        self.region_h = 0
        
        # Data storage
        # {track_id: [{'region': Region, 'frame': int, 'x': int, 'y': int}, ...]}
        self.track_paths = defaultdict(list)
        self.track_classes = {}  # {track_id: class_name}

    def initialize_dimensions(self) -> None:
        """Opens the video to retrieve dimensions and calculate grid size."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Failed to read the first frame from video.")
        
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Calculate 3x3 grid dimensions
        self.region_w = self.frame_width // 3
        self.region_h = self.frame_height // 3
        
        print(f"Video Dimensions: {self.frame_width}x{self.frame_height}")
        print(f"Region Grid Size: {self.region_w}x{self.region_h}")

    def get_region_from_coords(self, x: int, y: int) -> Region:
        """Maps an (x, y) coordinate to a specific Region enum."""
        # Clamp coordinates to frame boundaries to avoid index errors
        x = max(0, min(x, self.frame_width - 1))
        y = max(0, min(y, self.frame_height - 1))
        
        col = min(2, x // self.region_w)
        row = min(2, y // self.region_h)
        return Region(row * 3 + col)

    def get_region_bounds(self, region: Region) -> Tuple[int, int, int, int]:
        """Returns the (x1, y1, x2, y2) boundaries for a given region."""
        row = region.value // 3
        col = region.value % 3
        
        x1 = col * self.region_w
        y1 = row * self.region_h
        x2 = x1 + self.region_w
        y2 = y1 + self.region_h
        return x1, y1, x2, y2

    def process(self) -> None:
        """Reads the trajectory CSV and builds the region path for each track."""
        if not self.traj_csv_path.exists():
            print(f"[Warning] Trajectory file not found: {self.traj_csv_path}")
            return

        print(f"Processing trajectories: {self.traj_csv_path.name}...")
        df = pd.read_csv(self.traj_csv_path)
        
        # Sort to ensure we process frames sequentially
        if "frame" in df.columns and "track_id" in df.columns:
            df = df.sort_values(['track_id', 'frame'])
        
        # Group by track to process individual paths
        grouped = df.groupby('track_id')
        
        for track_id, group in grouped:
            prev_region = None
            
            # Store class name (assuming constant per track)
            class_name = group['class_name'].iloc[0] if 'class_name' in group.columns else "Unknown"
            self.track_classes[track_id] = class_name
            
            for _, row in group.iterrows():
                x = int(round(float(row['x'])))
                y = int(round(float(row['y'])))
                frame_idx = int(row['frame'])
                
                curr_region = self.get_region_from_coords(x, y)
                
                # Record only unique steps (when region changes or it's the start)
                if curr_region != prev_region:
                    self.track_paths[track_id].append({
                        'region': curr_region,
                        'frame': frame_idx,
                        'x': x,
                        'y': y
                    })
                    prev_region = curr_region

    def save(self) -> None:
        """Saves the extracted paths to the output CSV."""
        if not self.track_paths:
            print("[Info] No paths extracted to save.")
            return

        print(f"Saving extracted paths to: {self.output_csv_path}")
        
        with open(self.output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # CSV Header
            writer.writerow(['track_id', 'class_name', 'path_step', 'region', 'frame', 'x', 'y'])
            
            for track_id, path in self.track_paths.items():
                class_name = self.track_classes.get(track_id, "Unknown")
                
                for step_idx, point in enumerate(path):
                    writer.writerow([
                        track_id,
                        class_name,
                        step_idx + 1,  # 1-based step index
                        point['region'].name,
                        point['frame'],
                        point['x'],
                        point['y']
                    ])
        
        print(f" -> Saved {len(self.track_paths)} unique paths.")

    def visualize_grid(self) -> None:
        """
         displays the first frame of the video overlaid with the region grid.
         useful for verifying region definitions.
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[Warning] Could not read frame for visualization.")
            return

        # Draw Grid Lines
        # Vertical
        for i in range(1, 3):
            x = i * self.region_w
            cv2.line(frame, (x, 0), (x, self.frame_height), (255, 255, 255), 2)
        
        # Horizontal
        for i in range(1, 3):
            y = i * self.region_h
            cv2.line(frame, (0, y), (self.frame_width, y), (255, 255, 255), 2)
            
        # Draw Labels
        for region in Region:
            x1, y1, x2, y2 = self.get_region_bounds(region)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            label = get_region_display_name(region)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Text Background
            cv2.rectangle(frame, (cx - w//2 - 5, cy - h - 5), (cx + w//2 + 5, cy + 5), (0,0,0), -1)
            cv2.putText(frame, label, (cx - w//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Resize for display if too large
        target_h = 800
        scale = target_h / self.frame_height
        target_w = int(self.frame_width * scale)
        frame_resized = cv2.resize(frame, (target_w, target_h))

        cv2.imshow(f"Region Grid Visualization (Video {VIDEO_NUM})", frame_resized)
        print("Press any key to close the grid visualization...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    # 1. Setup Video Path
    try:
        video_path, _ = utils.get_video_paths(VIDEO_NUM)
        print(f"Target Video: {video_path}")
    except ValueError as e:
        print(f"[Error] {e}")
        return

    # 2. Define Models to Process
    models = ["NEW", "OLD"]

    for model_type in models:
        print(f"\n--- Processing {model_type} Model ---")
        
        # Construct dynamic file paths
        traj_filename = f"video{VIDEO_NUM}_trajectories{model_type}.csv"
        traj_path = TRAJ_DIR / traj_filename
        
        output_filename = f"video{VIDEO_NUM}_paths{model_type}.csv"
        output_path = PATHS_DIR / output_filename

        # Initialize Extractor
        try:
            extractor = PathExtractor(video_path, traj_path, output_path)
            extractor.initialize_dimensions()
            
            # Run Extraction
            extractor.process()
            extractor.save()
            
            # Show grid only once (for the first model processed)
            if model_type == models[0]:
                print("\nShowing region grid for verification...")
                extractor.visualize_grid()
                
        except Exception as e:
            print(f"[Error] Failed processing {model_type} model: {e}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()