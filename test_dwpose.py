import os
from pathlib import Path

import cv2

from dwpose import DWposeDetector


INPUT_PATH = r"/opt/code/720p_frames/frame_000080.jpg"

# Model pathsr
DET_MODEL_PATH = "models/yolo11m.onnx"
POSE_MODEL_PATH = "models/dw-ll_ucoco_384.onnx"

# ID of GPU or "cpu"
DEVICE = "cpu"
# Whether to visualize results (imshow). When False, only run inference.
SHOW = True
# SAVE_PATH = "output/720p_pose.mp4"
SAVE_PATH = None
# ====================================


def process_video(
    input_path,
    det_model_path,
    pose_model_path,
    device="cuda:0",
    show=False,
    save_path=None,
):
    """
    Process video file for pose estimation and display results.

    Args:
        input_path: Path to input video file
        det_model_path: Path to detection model
        pose_model_path: Path to pose estimation model
        device: Device to use (cuda:0 or cpu)
        show: Whether to display frames during processing
        save_path: Optional path to save the processed video; disabled when None
    """
    # Initialize detector
    pose = DWposeDetector(
        det_model_path=det_model_path,
        pose_model_path=pose_model_path,
        device=device,
    )

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Failed to open VideoWriter for: {save_path}")

    frame_count = 0
    print(f"Processing video: {input_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    print("Press 'q' to quit, 'p' to pause/resume")

    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                result = pose(frame, draw_on_image=show, alpha=0.6)

                if writer:
                    writer.write(result)

                if show:
                    cv2.imshow("DWPose Video", result)

                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)"
                    )

            # Handle keyboard input
            if show:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    paused = not paused
                    print("Paused" if paused else "Resumed")

    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        print(f"Video processing completed. Processed {frame_count} frames.")


def process_image(
    input_path, det_model_path, pose_model_path, device="cuda:0", show=False
):
    """
    Process single image file for pose estimation and display result.

    Args:
        input_path: Path to input image file
        det_model_path: Path to detection model
        pose_model_path: Path to pose estimation model
        device: Device to use (cuda:0 or cpu)
    """
    # Initialize detector
    pose = DWposeDetector(
        det_model_path=det_model_path,
        pose_model_path=pose_model_path,
        device=device,
    )

    # Process image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    result = pose(img, draw_on_image=show, alpha=0.6)

    if show:
        cv2.imshow("DWPose Image", result)
        print("Image processing completed. Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # Validate input file
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # Determine if input is video or image
    input_ext = Path(INPUT_PATH).suffix.lower()
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

    if input_ext in video_extensions:
        process_video(
            INPUT_PATH, DET_MODEL_PATH, POSE_MODEL_PATH, DEVICE, SHOW, SAVE_PATH
        )
    else:
        process_image(INPUT_PATH, DET_MODEL_PATH, POSE_MODEL_PATH, DEVICE, SHOW)


if __name__ == "__main__":
    main()
