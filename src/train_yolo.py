import numpy as np
import os
import shutil
from ultralytics import YOLO
import torch
from pathlib import Path
import utils

# ==========================================
# CONFIGURATION
# ==========================================

# File Paths (Relative to Project Root)
DATA_YAML_PATH = Path("dataset_yolo/data.yaml")
MODELS_DIR = Path("models")
CHECKPOINT_DIR = Path("checkpoints")

# Model Parameters
BASE_MODEL = "yolov8s.pt"
RUN_NAME = "sdd_yolov8s"
IMG_SIZE = 960
BATCH_SIZE = 8
OPTIMIZER = "AdamW"
WORKERS = 0

# Training Schedule
EPOCHS_FRESH = 150
EPOCHS_RESUME = 0        # Epoch count for fine-tuning/resuming
RESUME_TRAINING = False  # Flag to trigger resume logic

# Derived Constants
LAST_CHECKPOINT = CHECKPOINT_DIR / "last.pt"

# ==========================================
# CALLBACKS
# ==========================================

def backup_checkpoint_callback(trainer):
    """
    Copies the current epoch's weights to a persistent checkpoint directory.
    Ensures 'last.pt' is preserved outside the auto-generated run folder.
    """
    source_path = Path(trainer.save_dir) / "weights" / "last.pt"
    
    if source_path.exists():
        target_path = CHECKPOINT_DIR / "last.pt"
        try:
            shutil.copy2(source_path, target_path)
        except Exception as e:
            print(f"[Warning] Checkpoint backup failed: {e}")

# ==========================================
# TRAINING EXECUTION
# ==========================================

def run_training():
    """
    Configures and executes the YOLOv8 training pipeline.
    Manages device selection, model initialization, and resume logic.
    """
    # 1. Environment Setup
    utils.ensure_dir(str(MODELS_DIR))
    utils.ensure_dir(str(CHECKPOINT_DIR))
    
    device = 0 if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 1)

    print(f"Device: {device}")
    
    # 2. Model Initialization
    if RESUME_TRAINING and LAST_CHECKPOINT.exists():
        print(f"[Info] Resuming from checkpoint: {LAST_CHECKPOINT}")
        
        # Reset model state for fine-tuning
        model = YOLO(str(LAST_CHECKPOINT)).reset()
        current_run_name = f"{RUN_NAME}_resume"
        total_epochs = EPOCHS_RESUME
    else:
        print(f"[Info] Initializing base model: {BASE_MODEL}")
        model = YOLO(BASE_MODEL)
        current_run_name = RUN_NAME
        total_epochs = EPOCHS_FRESH

    # 3. Register Callbacks
    model.add_callback("on_fit_epoch_end", backup_checkpoint_callback)

    # 4. Start Training
    try:
        model.train(
            data=str(DATA_YAML_PATH),
            epochs=total_epochs,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=str(MODELS_DIR),
            name=current_run_name,
            
            # Optimization
            optimizer=OPTIMIZER,
            cos_lr=True,
            lr0=0.001,
            patience=20,
            
            # Persistence
            save=True,
            save_period=5,
            exist_ok=True,
            
            # Hardware & Logging
            device=device,
            workers=WORKERS,
            verbose=True,
            plots=True,
            val=True,
            cache=True
        )
        print("\n[Success] Training process complete.")
        print(f"Output Directory: {MODELS_DIR}/{current_run_name}")

    except Exception as e:
        print(f"\n[Error] Training interrupted: {e}")

if __name__ == "__main__":
    run_training()