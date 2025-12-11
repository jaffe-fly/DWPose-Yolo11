from __future__ import annotations

import cupy as cp
import numpy as np
import tritonclient.http as httpclient


def infer_python_backend(image: np.ndarray, server_url: str = "localhost:8000"):
    """
    Run DWPose inference using Triton Python Backend (dwpose_ensemble).

    All preprocessing (letterbox, normalization, cropping) and postprocessing
    (NMS, coordinate restoration, SimCC decoding) are handled GPU-accelerated
    on the server side.

    Args:
        image: Input image in BGR format (H, W, 3) as uint8
        server_url: Triton server URL

    Returns:
        Tuple of (bboxes, keypoints, scores)
        - bboxes: (N, 4) detection boxes in [x1, y1, x2, y2] format
        - keypoints: (N, 133, 2) keypoint coordinates in original image space
        - scores: (N, 133) keypoint confidence scores
    """

    def safe_as_numpy(output_name: str):
        """Safely convert Triton output to NumPy, handling GPU tensors."""
        try:
            tensor = response.as_numpy(output_name)
            if isinstance(tensor, cp.ndarray):
                return cp.asnumpy(tensor)
            return tensor
        except Exception as e:
            raise RuntimeError(f"Failed to get output '{output_name}': {e}")

    # Connect to Triton
    try:
        triton_client = httpclient.InferenceServerClient(url=server_url)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Triton at {server_url}: {e}")

    # Prepare input
    # Triton automatically handles batch dimension when max_batch_size > 0
    # We send (H, W, 3) and Triton treats it as batch_size=1
    input_data = httpclient.InferInput("IMAGE", image.shape, "UINT8")
    input_data.set_data_from_numpy(image)

    # Request outputs
    outputs = [
        httpclient.InferRequestedOutput("KEYPOINTS"),
        httpclient.InferRequestedOutput("SCORES"),
        httpclient.InferRequestedOutput("BBOXES"),
    ]

    # Run inference
    try:
        response = triton_client.infer(
            "dwpose_ensemble", model_version="1", inputs=[input_data], outputs=outputs
        )
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    # Get results (handle potential CuPy arrays from GPU)
    keypoints = safe_as_numpy("KEYPOINTS")
    scores = safe_as_numpy("SCORES")
    bboxes = safe_as_numpy("BBOXES")

    return bboxes, keypoints, scores
