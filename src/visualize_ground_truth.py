import cv2
import argparse
import utils

def run_visualization(video_num: int, fps: int = 30) -> None:
    """
    Runs the ground truth visualization for a specific video.

    Args:
        video_num (int): The ID of the video to visualize (0 or 3).
        fps (int): Playback speed in frames per second.
    """
    # Setup Paths & Load Data
    try:
        video_path, annotation_path = utils.get_video_paths(video_num)
    except ValueError as e:
        print(f"Error: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    annotations = utils.load_annotations(annotation_path)

    # Video Properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate Dimensions and Scaling
    (new_w, new_h), (scale_x, scale_y) = utils.calculate_scaling(orig_w, orig_h, annotations)
    
    print(f"Original Resolution: {orig_w}x{orig_h}")
    print(f"Display Resolution:  {new_w}x{new_h}")
    print("Press 'q' to quit, 'p' to pause/play, 'j/k' to seek.")

    # Initialize playback
    frame_number = 0
    paused = False
    window_name = 'Ground Truth Visualization'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    while True:
        # Sync video reader
        if frame_number != int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret: 
            break

        # Resize and Draw
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        if not annotations.empty:
            frame_anns = annotations[annotations['frame'] == frame_number]
            utils.draw_annotations_on_frame(frame_resized, frame_anns, scale_x, scale_y)

        # Draw Overlays
        utils.draw_legends(frame_resized)
        utils.draw_timestamp(frame_resized, frame_number, fps)
        
        cv2.imshow(window_name, frame_resized)

        # Handle Playback Control
        delay = 0 if paused else int(1000 / fps)
        key = cv2.waitKey(delay) & 0xFF

        frame_number, paused, should_quit = utils.handle_playback_controls(
            key, frame_number, total_frames, paused, fps
        )

        if should_quit:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Allows running via command line: python visualize_ground_truth.py --video 0 --fps 60
    parser = argparse.ArgumentParser(description="Visualize Ground Truth Annotations.")
    parser.add_argument("--video", type=int, default=1, choices=[0, 3], help="Video ID (0 or 3)")
    parser.add_argument("--fps", type=int, default=32, help="Playback FPS")
    
    args = parser.parse_args()
    
    run_visualization(args.video, args.fps)