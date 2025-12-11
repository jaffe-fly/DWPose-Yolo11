"""
Test script comparing different Triton communication methods:
1. HTTP (standard REST API)
2. gRPC (binary protocol, faster than HTTP)
3. System Shared Memory (zero-copy, fastest)

Usage:
    python test_triton_client_shm.py --input image.jpg --method http
    python test_triton_client_shm.py --input image.jpg --method grpc
    python test_triton_client_shm.py --input image.jpg --method shm
    python test_triton_client_shm.py --input image.jpg --method all
"""

import argparse
import time

import cv2
import numpy as np
from draw_utils import draw_triton_result


def get_server_url(base_url: str, method: str) -> str:
    """
    Get the correct server URL based on the method.

    Intelligently handles port mapping between HTTP and gRPC:
    - Standard ports: 8000 (HTTP) ↔ 8001 (gRPC)
    - Custom ports: 8800 (HTTP) ↔ 8801 (gRPC), etc.

    Args:
        base_url: Base URL (e.g., "localhost:8001" or "192.168.1.100")
        method: Communication method ('http', 'grpc', 'shm')

    Returns:
        Correct URL for the method with appropriate port

    Examples:
        >>> get_server_url("localhost:8001", "http")
        'localhost:8000'
        >>> get_server_url("localhost:8801", "http")
        'localhost:8800'
        >>> get_server_url("192.168.1.100", "grpc")
        '192.168.1.100:8001'
    """
    # Extract host and port
    if ":" in base_url:
        host, port_str = base_url.rsplit(":", 1)
        port = int(port_str)
    else:
        host = base_url
        port = 8801  # Default gRPC port

    # Determine the target port based on method
    if method == "http":
        http_port = 8000  # Fallback to standard HTTP port
        return f"{host}:{http_port}"
    else:  # grpc or shm
        # Use the provided port or default to 8001
        return f"{host}:{port}"


def benchmark_inference(
    method: str, image: np.ndarray, server_url: str, num_runs: int = 10
):
    """
    Benchmark inference performance with different methods.

    Args:
        method: 'http', 'grpc', or 'shm'
        image: Input image
        server_url: Triton server URL
        num_runs: Number of runs for averaging
    """
    if method == "http":
        from ensemble_client import infer_python_backend as infer_fn
    elif method == "grpc":
        from ensemble_client_grpc import infer_python_backend as infer_fn
    elif method == "shm":
        from ensemble_client_shm import infer_python_backend_shm as infer_fn
    else:
        raise ValueError(f"Unknown method: {method}. Use 'http', 'grpc', or 'shm'")

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {method.upper()}")
    print(f"{'=' * 60}")

    # Get correct URL for the method
    actual_url = get_server_url(server_url, method)
    print(f"Server URL: {actual_url}")

    # Warmup
    print("Warming up...")
    for _ in range(3):
        infer_fn(image, actual_url)

    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        bboxes, keypoints, scores = infer_fn(image, actual_url)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)
        print(f"  Run {i + 1}/{num_runs}: {elapsed:.2f} ms")

    # Statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")
    print(f"  FPS:     {1000 / avg_time:.1f}")

    return bboxes, keypoints, scores


def main():
    parser = argparse.ArgumentParser(
        description="Test Triton inference with different communication methods"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        help="Triton Server base URL (gRPC port, HTTP will be auto-adjusted). Examples: localhost:8001, localhost:8801, 192.168.1.100:8001",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["http", "grpc", "shm", "all"],
        default="all",
        help="Communication method: http, grpc, shm, or all (benchmark all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualization",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save visualization",
    )

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.input)
    if image is None:
        raise ValueError(f"Failed to load image: {args.input}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\nImage: {args.input}")
    print(f"Shape: {image.shape}")
    print(f"Size:  {image.nbytes / 1024 / 1024:.2f} MB")
    print(f"Base Server URL: {args.url}")
    print(f"  → HTTP will use: {get_server_url(args.url, 'http')}")
    print(f"  → gRPC/SHM will use: {get_server_url(args.url, 'grpc')}")

    # Run benchmark(s)
    results = {}

    if args.method == "all":
        methods = ["http", "grpc", "shm"]
    else:
        methods = [args.method]

    for method in methods:
        bboxes, keypoints, scores = benchmark_inference(
            method, image_rgb, args.url, args.runs
        )
        if bboxes is not None:
            results[method] = {
                "bboxes": bboxes,
                "keypoints": keypoints,
                "scores": scores,
            }

    # Performance comparison
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'=' * 60}")

    # Visualize results from the last method
    if results:
        last_method = (
            methods[-1] if methods[-1] in results else list(results.keys())[-1]
        )
        result_data = results[last_method]

        print(f"\nDetected {len(result_data['keypoints'])} person(s)")

        # Draw results
        result_img = draw_triton_result(
            image,
            result_data["keypoints"],
            result_data["scores"],
            score_thr=0.3,
            alpha=0.6,
        )

        if args.output:
            cv2.imwrite(args.output, result_img)
            print(f"Saved visualization to: {args.output}")

        if args.show:
            cv2.imshow(f"DWPose Triton - {last_method.upper()}", result_img)
            print("\nPress any key to close the window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # python test_triton_client_shm.py --input ../asserts/frame_000080.jpg --url localhost
