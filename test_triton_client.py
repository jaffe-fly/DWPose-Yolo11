"""
Test script for DWPose Triton Server deployment.

Supports both single image and video file processing with visualization.

Usage:
    # Process single image
    python test_triton_client.py --input path/to/image.jpg --url localhost:8000 --visualize
    
    # Process video
    python test_triton_client.py --input path/to/video.mp4 --url localhost:8000 --show --save output/result.mp4
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from draw_utils import draw_triton_result
from ensemble_client import infer_python_backend


def process_image(
    input_path: str, 
    server_url: str = "localhost:8000", 
    show: bool = False,
    save_path: str = None
):
    """
    Process single image file for pose estimation and display result.

    Args:
        input_path: Path to input image file
        server_url: Triton Server URL
        show: Whether to display the result
        save_path: Optional path to save visualization
    """
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")
    
    # Triton detects using RGB usually, check if local implementation feeds BGR?
    # Standard DWPose/RTMPose expects RGB.
    # Convert BGR to RGB for inference
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Loaded image: {image.shape}")
    print(f"Running inference on Triton Server: {server_url}")

    # Run inference using ensemble_client
    # Pass RGB image to inference
    bboxes, keypoints, scores = infer_python_backend(image_rgb, server_url)

    print("\nInference completed successfully!")
    print(f"Detected {len(keypoints)} person(s)")

    # Print keypoint information
    for i, (kpts, scrs) in enumerate(zip(keypoints, scores)):
        print(f"\nPerson {i + 1}:")
        print(f"  Valid keypoints: {np.sum(scrs > 0.3)} / {len(scrs)}")
        if len(bboxes) > i:
            print(f"  BBox: {bboxes[i]}")
            print(f"  Average score: {scrs.mean():.3f}")

    # Visualize results
    result = draw_triton_result(image, keypoints, scores, score_thr=0.3, alpha=0.6)

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        cv2.imwrite(save_path, result)
        print(f"\nVisualization saved to: {save_path}")

    if show:
        cv2.imshow("DWPose Triton - Image", result)
        print("\nImage processing completed. Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return keypoints, scores, bboxes


def process_video(
    input_path: str,
    server_url: str = "localhost:8000",
    show: bool = False,
    save_path: str = None,
):
    """
    Process video file for pose estimation and display results.

    Args:
        input_path: Path to input video file
        server_url: Triton Server URL
        show: Whether to display frames during processing
        save_path: Optional path to save the processed video; disabled when None
    """
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
    print(f"\nProcessing video: {input_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"Triton Server: {server_url}")
    print("Press 'q' to quit, 'p' to pause/resume")

    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame via Triton Server
                try:
                    # Convert to RGB for inference
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bboxes, keypoints, scores = infer_python_backend(frame_rgb, server_url)
                    result = draw_triton_result(frame, keypoints, scores, score_thr=0.3, alpha=0.6)
                except Exception as e:
                    print(f"\nWarning: Frame {frame_count} inference failed: {e}")
                    result = frame  # Use original frame if inference fails

                if writer:
                    writer.write(result)

                if show:
                    cv2.imshow("DWPose Triton - Video", result)

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
        print(f"\nVideo processing completed. Processed {frame_count} frames.")
        if save_path:
            print(f"Output saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test DWPose Triton Server deployment with image/video support"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to input image or video file"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8000",
        help="Triton Server URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Optional path to save visualization (image or video)"
    )
    parser.add_argument(
        "--show", 
        action="store_true", 
        help="Display visualization during processing"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Determine if input is video or image
    input_ext = Path(args.input).suffix.lower()
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

    try:
        if input_ext in video_extensions:
            process_video(args.input, args.url, args.show, args.output)
        else:
            process_image(args.input, args.url, args.show, args.output)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
