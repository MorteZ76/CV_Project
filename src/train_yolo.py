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
# Defines the location of the dataset configuration and output directories
DATA_YAML_PATH = Path("dataset_yolo/data.yaml")
MODELS_DIR = Path("models")
CHECKPOINT_DIR = Path("checkpoints")

# Model Parameters
BASE_MODEL = "yolov8s.pt"  # Initial pre-trained weights (YOLOv8 Small)
RUN_NAME = "sdd_yolov8s"   # Identifier for the training run
IMG_SIZE = 960             # Input image resolution (must be a multiple of 32)
BATCH_SIZE = 8             # Batch size; reduce if CUDA out-of-memory occurs
OPTIMIZER = "AdamW"        # Optimizer algorithm (AdamW is generally robust for object detection)
WORKERS = 0                # Number of data loader workers (0 runs in the main process)

# Training Schedule
EPOCHS_FRESH = 150       # Total epochs for a new training session
EPOCHS_RESUME = 0        # Total epochs if resuming (fine-tuning)
RESUME_TRAINING = False  # Set to True to load weights from the last checkpoint

# Derived Constants
LAST_CHECKPOINT = CHECKPOINT_DIR / "last.pt"

# ==========================================
# CALLBACKS
# ==========================================

def backup_checkpoint_callback(trainer):
    """
    Callback function executed at the end of every training epoch.
    
    It copies the 'last.pt' checkpoint from the dynamic run directory to a 
    fixed location. This ensures the latest model state is always accessible 
    at a known path, facilitating easier resumption of training.
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
    Orchestrates the YOLOv8 training process.
    
    Key responsibilities:
    1. Prepares necessary directories.
    2. Detects available hardware (GPU vs CPU).
    3. Initializes the model (either fresh or from a checkpoint).
    4. Executes the training loop with specified hyperparameters.
    """
    # 1. Environment Setup
    utils.ensure_dir(str(MODELS_DIR))
    utils.ensure_dir(str(CHECKPOINT_DIR))
    
    # Select device: 0 for GPU, 'cpu' otherwise
    device = 0 if torch.cuda.is_available() else "cpu"
    # Optimize CPU thread allocation for PyTorch
    torch.set_num_threads(os.cpu_count() or 1)

    print(f"Device: {device}")
    
    # 2. Model Initialization
    # Handle logic for resuming from a previous state or starting fresh
    if RESUME_TRAINING and LAST_CHECKPOINT.exists():
        print(f"[Info] Resuming from checkpoint: {LAST_CHECKPOINT}")
        
        # Load the existing model and reset internal state for fine-tuning
        model = YOLO(str(LAST_CHECKPOINT)).reset()
        current_run_name = f"{RUN_NAME}_resume"
        total_epochs = EPOCHS_RESUME
    else:
        print(f"[Info] Initializing base model: {BASE_MODEL}")
        model = YOLO(BASE_MODEL)
        current_run_name = RUN_NAME
        total_epochs = EPOCHS_FRESH

    # 3. Register Callbacks
    # Attach custom behavior to the training loop
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
            
            # Optimization Hyperparameters
            optimizer=OPTIMIZER,
            cos_lr=True,      # Use cosine annealing learning rate scheduler
            lr0=0.001,        # Initial learning rate
            patience=20,      # Early stopping patience (epochs with no improvement)
            
            # Persistence Settings
            save=True,        # Save model checkpoints
            save_period=5,    # interval (in epochs) for saving checkpoints
            exist_ok=True,    # Allow overwriting existing project folder
            
            # Hardware & Logging
            device=device,
            workers=WORKERS,
            verbose=True,     # Enable detailed logging
            plots=True,       # Generate training metric plots
            val=True,         # Run validation after each epoch
            cache=True        # Cache images in RAM to speed up training
        )
        print("\n[Success] Training process complete.")
        print(f"Output Directory: {MODELS_DIR}/{current_run_name}")

    except Exception as e:
        print(f"\n[Error] Training interrupted: {e}")

if __name__ == "__main__":
    run_training()