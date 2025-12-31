"""
extract_paths.py
----------------
Analyzes object trajectories to determine their movement patterns across specific 
regions of the video frame (e.g., Top-Left -> Middle-Center -> Bottom-Right).

Features:
- Unified processing for 'NEW', 'OLD', and 'GT' (Ground Truth) data sources.
- Automatically scales GT annotations to match the video resolution.
- Maps each trajectory point to a 3x3 grid region.
- Compresses trajectory data into a sequence of visited regions (path steps).
- Outputs standardized path CSVs for analysis.
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

# Videos to process: 0 (video0) and 3 (video3)
VIDEOS_TO_PROCESS = [0, 3]
MODELS_TO_PROCESS = ["GT", "NEW", "OLD"]

# Path Resolution
# Establish base directory relative to this script's location
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
TRAJ_DIR = OUTPUTS_DIR / "trajectories"
PATHS_DIR = OUTPUTS_DIR / "paths"

# Ensure output directory exists to prevent IO errors
utils.ensure_dir(str(PATHS_DIR))

# ==============================================================================
# REGION DEFINITIONS
# ==============================================================================

class Region(Enum):
    """
    Enumeration of the 9 grid regions in the video frame.
    Layout:
    0 | 1 | 2
    --+---+--
    3 | 4 | 5
    --+---+--
    6 | 7 | 8
    """
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

    def __init__(self, video_path: str):
        """
        Initialize the PathExtractor with video dimensions.
        """
        self.video_path = video_path
        self.frame_w, self.frame_h = self._get_video_dimensions()
        
        # Calculate 3x3 grid dimensions based on frame size
        self.region_w = self.frame_w // 3
        self.region_h = self.frame_h // 3
        
        # Data storage
        # {track_id: [{'region': Region, 'frame': int, 'x': int, 'y': int}, ...]}
        self.track_paths = defaultdict(list)
        self.track_classes = {}  # {track_id: class_name}

    def _get_video_dimensions(self) -> Tuple[int, int]:
        """Opens the video file to retrieve its width and height."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if w == 0 or h == 0:
            raise ValueError("Failed to read video dimensions.")
            
        return w, h

    def get_region_from_coords(self, x: int, y: int) -> Region:
        """
        Maps an (x, y) coordinate to a specific Region enum.
        Clamps coordinates to frame boundaries to avoid index errors.
        """
        # Clamp coordinates to frame boundaries
        x = max(0, min(x, self.frame_w - 1))
        y = max(0, min(y, self.frame_h - 1))
        
        # Determine grid column (0, 1, or 2) and row (0, 1, or 2)
        col = min(2, x // self.region_w)
        row = min(2, y // self.region_h)
        return Region(row * 3 + col)

    def get_region_bounds(self, region: Region) -> Tuple[int, int, int, int]:
        """Returns the (x1, y1, x2, y2) pixel boundaries for a given region."""
        row = region.value // 3
        col = region.value % 3
        
        x1 = col * self.region_w
        y1 = row * self.region_h
        x2 = x1 + self.region_w
        y2 = y1 + self.region_h
        return x1, y1, x2, y2

    def process_dataframe(self, df: pd.DataFrame) -> None:
        """
        Reads a standardized trajectory DataFrame and builds region paths.
        Expected columns: ['track_id', 'frame', 'x', 'y', 'class_name']
        
        Compresses the path by only recording changes in region (key steps).
        """
        # Sort to ensure we process frames sequentially per track
        df = df.sort_values(['track_id', 'frame'])
        
        # Group by track to process individual paths
        grouped = df.groupby('track_id')
        
        for track_id, group in grouped:
            prev_region = None
            
            # Store class name (handle potential missing values/nan)
            cls_val = group.iloc[0]['class_name']
            class_name = str(cls_val).replace('"', '').strip() if pd.notna(cls_val) else "Unknown"
            self.track_classes[track_id] = class_name
            
            for _, row in group.iterrows():
                x = int(row['x'])
                y = int(row['y'])
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

    def save(self, output_path: Path) -> None:
        """Saves the extracted path steps to the output CSV."""
        if not self.track_paths:
            print(f"   [Info] No paths extracted for {output_path.name}")
            return

        with open(output_path, 'w', newline='') as f:
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
        
        print(f"   -> Saved {len(self.track_paths)} unique paths to {output_path.name}")

    def visualize_grid(self) -> None:
        """
        Displays the first frame of the video overlaid with the region grid.
        Useful for visually verifying region boundaries.
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[Warning] Could not read frame for visualization.")
            return

        # Draw Grid Lines
        for i in range(1, 3):
            # Vertical Lines
            cv2.line(frame, (i * self.region_w, 0), (i * self.region_w, self.frame_h), (255, 255, 255), 2)
            # Horizontal Lines
            cv2.line(frame, (0, i * self.region_h), (self.frame_w, i * self.region_h), (255, 255, 255), 2)
            
        # Draw Region Labels
        for r in Region:
            x1, y1, x2, y2 = self.get_region_bounds(r)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            label = get_region_display_name(r)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Text Background for readability
            cv2.rectangle(frame, (cx - w//2 - 5, cy - h - 5), (cx + w//2 + 5, cy + 5), (0,0,0), -1)
            cv2.putText(frame, label, (cx - w//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Resize for display (fit to screen height 800px)
        target_h = 800
        scale = target_h / self.frame_h
        target_w = int(self.frame_w * scale)
        frame_resized = cv2.resize(frame, (target_w, target_h))

        cv2.imshow("Region Grid Visualization", frame_resized)
        print("   [UI] Press any key to close the grid visualization...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ==============================================================================
# DATA LOADING HELPERS
# ==============================================================================

def load_gt_data(video_num: int, video_w: int, video_h: int) -> pd.DataFrame:
    """
    Loads Ground Truth annotations and scales them to the video resolution.
    Returns a normalized DataFrame matching the tracker output format.
    """
    _, ann_path = utils.get_video_paths(video_num)
    df = utils.load_annotations(ann_path)
    
    if df.empty:
        return pd.DataFrame()

    # Calculate scaling (Annotation Coords -> Video Pixel Coords)
    # SDD annotations often use a coordinate space different from the video resolution
    max_x = max(df['xmax'].max(), 1.0)
    max_y = max(df['ymax'].max(), 1.0)
    
    scale_x = video_w / max_x
    scale_y = video_h / max_y

    # Calculate Center Points and Scale
    df['x'] = ((df['xmin'] + df['xmax']) / 2) * scale_x
    df['y'] = ((df['ymin'] + df['ymax']) / 2) * scale_y
    
    # Standardize Class Name column
    df = df.rename(columns={'label': 'class_name'})
    
    return df[['track_id', 'frame', 'x', 'y', 'class_name']]


def load_tracker_data(video_num: int, model: str) -> pd.DataFrame:
    """
    Loads trajectory data from a tracker output CSV.
    """
    filename = f"video{video_num}_trajectories{model}.csv"
    path = TRAJ_DIR / filename
    
    if not path.exists():
        print(f"   [Warning] Tracker file missing: {path}")
        return pd.DataFrame()
        
    return pd.read_csv(path)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main batch processing loop.
    Iterates through configured videos and models to generate path analysis CSVs.
    """
    for video_num in VIDEOS_TO_PROCESS:
        print(f"\n=== Processing Video {video_num} ===")
        
        try:
            # 1. Setup Video and Extractor
            video_path, _ = utils.get_video_paths(video_num)
            print(f"   Target Video: {video_path}")
            
            extractor = PathExtractor(video_path)
            
            # Show grid once per video for visual verification of regions 
            extractor.visualize_grid()

            # 2. Process Each Model (GT, NEW, OLD)
            for model in MODELS_TO_PROCESS:
                print(f"\n   > Extracting {model} paths...")
                
                # Load appropriate data
                if model == "GT":
                    df = load_gt_data(video_num, extractor.frame_w, extractor.frame_h)
                else:
                    df = load_tracker_data(video_num, model)

                if df.empty:
                    print(f"     [Skip] No data available for {model}")
                    continue

                # Run Extraction
                # Clear previous data to reuse the extractor instance
                extractor.track_paths.clear()
                extractor.track_classes.clear()
                
                extractor.process_dataframe(df)
                
                # Save Result
                out_filename = f"video{video_num}_paths{model}.csv"
                extractor.save(PATHS_DIR / out_filename)

        except Exception as e:
            print(f"[Error] Failed processing Video {video_num}: {e}")

    print("\nBatch path extraction complete.")

if __name__ == "__main__":
    main()