# CV_Project
this repo is dedicated to Computer Vision's project


# UAV Object Tracking & Trajectory Analysis Pipeline

## 📌 Project Overview
This project is a computer vision pipeline focused on **Object Tracking**, **Trajectory Analysis**, and **YOLO Model Training** on the Stanford Drone Dataset (SDD).

It addresses specific challenges of aerial surveillance (small objects, occlusions) by implementing:
* [cite_start]**High-Resolution Training (960px)** to detect small targets[cite: 11, 22].
* [cite_start]**Heuristic Data Curation** to prioritize complex, dense scenes[cite: 23, 47, 48].
* [cite_start]**ByteTrack-based Logic** with Kalman Filtering and Class Locking[cite: 9, 24, 63, 67].
* [cite_start]**Semantic Path Analysis** to map crowd movements to Origin-Destination flows[cite: 14, 25, 69, 71].

---

## 🔄 Data Flow Pipeline
The pipeline processes raw drone footage into actionable semantic insights following this flow:

```mermaid
graph TD;
    A[Raw Video & GT] -->|prepare_dataset.py| B(YOLO Images & Labels);
    B -->|train_yolo.py| C[Model Weights .pt];
    A & C -->|cache_detections.py| D[Cached Detections .parquet];
    D -->|tracker.py| E[Trajectories .csv];
    E -->|extract_paths.py| F(Regional Paths .csv);
    F -->|analyze_path_frequencies.py| G[Final Stats .csv];
