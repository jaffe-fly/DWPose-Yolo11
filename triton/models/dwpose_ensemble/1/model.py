"""
GPU-optimized Python Backend for DWPose using CuPy.

CuPy is NVIDIA's GPU array library with NumPy-compatible API.
It's typically pre-installed in Triton Server Python Backend.

Keeps tensors on GPU throughout the pipeline using CuPy arrays.
"""

from __future__ import annotations  # Fix type annotation compatibility

import json
from typing import Tuple

import cupy as cp
import numpy as np
import triton_python_backend_utils as pb_utils


def tensor_to_cupy(tensor: pb_utils.Tensor) -> cp.ndarray:
    """Convert Triton tensor to CuPy array using DLPack (zero-copy GPU transfer)."""
    return cp.from_dlpack(tensor.to_dlpack())


class TritonPythonModel:
    """GPU-Optimized Python Backend using CuPy."""

    def initialize(self, args):
        # Set CuPy to use GPU device 0 (avoid multi-GPU device mismatch)
        cp.cuda.Device(0).use()

        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        self.det_model = params.get("det_model", {"string_value": "yolo11m"})[
            "string_value"
        ]
        self.pose_model = params.get("pose_model", {"string_value": "dwpose"})[
            "string_value"
        ]
        self.det_conf = float(
            params.get("det_conf", {"string_value": "0.25"})["string_value"]
        )
        self.det_iou = float(
            params.get("det_iou", {"string_value": "0.45"})["string_value"]
        )
        self.pose_score_thr = float(
            params.get("pose_score_thr", {"string_value": "0.3"})["string_value"]
        )
        self.simcc_split_ratio = float(
            params.get("simcc_split_ratio", {"string_value": "2.0"})["string_value"]
        )

        self.det_input_size = self._parse_hw(
            params.get("det_input_size", {"string_value": "640,640"})
        )
        self.pose_input_size = self._parse_hw(
            params.get("pose_input_size", {"string_value": "288,384"})
        )
        self.det_input_dtype = params.get("det_input_dtype", {"string_value": "FP16"})[
            "string_value"
        ].upper()

        self.pose_mean_gpu = cp.array([123.675, 116.28, 103.53], dtype=cp.float32)
        self.pose_std_gpu = cp.array([58.395, 57.12, 57.375], dtype=cp.float32)
        self.pose_model_size_gpu = cp.array(
            [self.pose_input_size[1], self.pose_input_size[0]], dtype=cp.float32
        )  # (W, H)

    def _parse_hw(self, param) -> Tuple[int, int]:
        val = param["string_value"]
        parts = val.split(",")
        return (int(parts[0]), int(parts[1])) if len(parts) == 2 else (640, 640)

    def _preprocess_det(self, img: np.ndarray) -> Tuple[cp.ndarray, float]:
        """Letterbox resize on GPU using CuPy."""
        h, w = self.det_input_size

        # Move image to GPU
        img_gpu = cp.asarray(img, dtype=cp.float32)

        # Calculate resize ratio
        r = min(h / img.shape[0], w / img.shape[1])
        new_h, new_w = int(img.shape[0] * r), int(img.shape[1] * r)

        # Resize on GPU
        resized = self._resize_bilinear_cupy(img_gpu, (new_h, new_w))

        # Create padded image on GPU (filled with 114)
        padded_img = cp.ones((h, w, 3), dtype=cp.float32) * 114
        padded_img[:new_h, :new_w] = resized

        # HWC -> CHW
        padded_img = padded_img.transpose(2, 0, 1)

        # Normalize and convert dtype in one step (optimization)
        padded_img = padded_img / 255.0
        if self.det_input_dtype == "FP16":
            padded_img = padded_img.astype(cp.float16)

        return padded_img, r

    def _postprocess_det_cupy(
        self, outputs: cp.ndarray, img_size: Tuple[int, int]
    ) -> cp.ndarray:
        """Process YOLO11 ONNX outputs.

        YOLO11 ONNX output format: [8400, 84]
        - First 4 values: [cx, cy, w, h] in pixels relative to model input size (640x640)
        - Next 80 values: class probabilities

        No transformation needed - the output is already in the correct format.
        The coordinate values are absolute pixels relative to the 640x640 input.
        """
        # No processing needed for YOLO11 outputs
        return outputs

    def _nms_cupy(
        self, boxes: cp.ndarray, scores: cp.ndarray, iou_thr: float
    ) -> cp.ndarray:
        """NMS implementation using CuPy (on GPU)."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        # Fix: align with original implementation - add +1 for area calculation
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort by scores (descending)
        order = cp.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            if order.size == 1:
                break

            # Compute IoU
            xx1 = cp.maximum(x1[i], x1[order[1:]])
            yy1 = cp.maximum(y1[i], y1[order[1:]])
            xx2 = cp.minimum(x2[i], x2[order[1:]])
            yy2 = cp.minimum(y2[i], y2[order[1:]])

            w = cp.maximum(0.0, xx2 - xx1)
            h = cp.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU <= threshold
            inds = cp.where(iou <= iou_thr)[0]
            order = order[inds + 1]

        return cp.array(keep, dtype=cp.int32)

    def _run_detection(self, image: np.ndarray) -> cp.ndarray:
        """Run detection, return boxes on GPU (CuPy array)."""
        det_input, ratio = self._preprocess_det(image)

        det_input_batched = det_input[None, ...]  # (1, 3, H, W)
        det_input_tensor = pb_utils.Tensor.from_dlpack(
            "images", det_input_batched.toDlpack()
        )

        response = pb_utils.InferenceRequest(
            model_name=self.det_model,
            requested_output_names=["output0"],
            inputs=[det_input_tensor],
        ).exec()

        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())

        det_output = pb_utils.get_output_tensor_by_name(response, "output0")
        if det_output is None:
            return cp.zeros((0, 4), dtype=cp.float32)

        # Convert to CuPy (keep on GPU) and ensure FP32 for calculations
        raw = tensor_to_cupy(det_output)
        if raw.dtype == cp.float16:
            # Optimized: convert FP16 → FP32 (from YOLO FP16 output)
            raw = raw.astype(cp.float32)

        # YOLO11 output format: (1, 84, 8400) -> transpose to (1, 8400, 84)
        # 84 = 4 (bbox coords) + 80 (class scores)
        if raw.shape[1] < raw.shape[2]:
            raw = raw.transpose(0, 2, 1)  # (1, 84, 8400) -> (1, 8400, 84)

        # Post-process on GPU
        predictions = self._postprocess_det_cupy(raw, self.det_input_size)[0]

        # Extract class scores (YOLO11 format: no separate objectness)
        class_scores = predictions[:, 4:]  # All class scores
        boxes = predictions[:, :4]

        # OPTIMIZATION: Filter for person class (index 0) BEFORE bbox conversion and NMS
        # This reduces computation by eliminating non-person detections early
        person_scores = class_scores[:, 0]
        valid_mask = person_scores > self.det_conf

        if cp.sum(valid_mask) == 0:
            return cp.zeros((0, 4), dtype=cp.float32)

        # Only process boxes that passed confidence threshold
        boxes = boxes[valid_mask]
        person_scores = person_scores[valid_mask]

        # Convert boxes from cxcywh to xyxy
        boxes_xyxy = cp.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # Protect against division by zero or very small ratio
        if ratio < 0.001:
            pb_utils.Logger.log_warn(f"Ratio too small: {ratio}, using 1.0 instead")
            ratio = 1.0

        boxes_xyxy /= ratio

        # Check for inf/nan after division
        if not cp.all(cp.isfinite(boxes_xyxy)):
            pb_utils.Logger.log_warn("bbox_xyxy contains inf/nan after ratio division")
            # Replace inf/nan with 0
            boxes_xyxy = cp.nan_to_num(boxes_xyxy, nan=0.0, posinf=0.0, neginf=0.0)

        # NMS on GPU (now with reduced input size from early filtering)
        keep_indices = self._nms_cupy(boxes_xyxy, person_scores, self.det_iou)

        return boxes_xyxy[keep_indices]

    def _resize_bilinear_cupy(
        self, img: cp.ndarray, new_size: Tuple[int, int]
    ) -> cp.ndarray:
        """Bilinear resize on GPU using CuPy."""
        h_out, w_out = new_size
        h_in, w_in = img.shape[:2]

        if h_in == h_out and w_in == w_out:
            return img

        # Create coordinate grids
        y_out = cp.arange(h_out, dtype=cp.float32)
        x_out = cp.arange(w_out, dtype=cp.float32)

        # Map output coordinates to input coordinates
        y_in = y_out * (h_in / h_out)
        x_in = x_out * (w_in / w_out)

        # Get integer parts and fractional parts
        y0 = cp.floor(y_in).astype(cp.int32)
        x0 = cp.floor(x_in).astype(cp.int32)
        y1 = cp.minimum(y0 + 1, h_in - 1)
        x1 = cp.minimum(x0 + 1, w_in - 1)

        # Clamp to image bounds
        y0 = cp.clip(y0, 0, h_in - 1)
        x0 = cp.clip(x0, 0, w_in - 1)

        wy = (y_in - y0)[:, None, None]
        wx = (x_in - x0)[None, :, None]

        # Bilinear interpolation
        if img.ndim == 3:  # Color image
            result = (
                img[y0[:, None], x0[None, :]] * (1 - wy) * (1 - wx)
                + img[y1[:, None], x0[None, :]] * wy * (1 - wx)
                + img[y0[:, None], x1[None, :]] * (1 - wy) * wx
                + img[y1[:, None], x1[None, :]] * wy * wx
            )
        else:  # Grayscale
            result = (
                img[y0[:, None], x0[None, :]] * (1 - wy) * (1 - wx)
                + img[y1[:, None], x0[None, :]] * wy * (1 - wx)
                + img[y0[:, None], x1[None, :]] * (1 - wy) * wx
                + img[y1[:, None], x1[None, :]] * wy * wx
            )

        return result

    def _get_center_scale_cupy(self, bbox: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """Calculate center and scale with padding and aspect ratio fix.

        Matches logic in onnxpose.py: bbox_xyxy2cs -> top_down_affine
        """
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # 1. Padding 1.25x (bbox_xyxy2cs)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1

        scale_w = w * 1.25
        scale_h = h * 1.25

        # 2. Fix Aspect Ratio (top_down_affine -> _fix_aspect_ratio)
        # Model input size: 288x384 (W,H) -> Ratio 0.75
        model_w, model_h = self.pose_input_size
        aspect_ratio = model_w / model_h

        if scale_w > scale_h * aspect_ratio:
            scale_h = scale_w / aspect_ratio
        else:
            scale_w = scale_h * aspect_ratio

        return cp.array([center_x, center_y], dtype=cp.float32), cp.array(
            [scale_w, scale_h], dtype=cp.float32
        )

    def _crop_and_resize_cupy(
        self,
        img: cp.ndarray,
        center: cp.ndarray,
        scale: cp.ndarray,
        out_size: Tuple[int, int],
    ) -> cp.ndarray:
        """Crop and resize using affine transformation logic on GPU.

        Simulates cv2.warpAffine used in top_down_affine.
        Can handle cropping outside image bounds (padding).
        """
        h_out, w_out = out_size
        h_img, w_img = img.shape[:2]

        cx, cy = center[0], center[1]
        sw, sh = scale[0], scale[1]

        # Generate destination grid coordinates
        # Pixel centered coordinates
        x_out = cp.arange(w_out, dtype=cp.float32) + 0.5
        y_out = cp.arange(h_out, dtype=cp.float32) + 0.5

        # Map grid back to source coordinates
        # source_x = (dest_x / dest_w - 0.5) * scale_w + center_x
        # Simplified: stride = scale / output_size
        stride_w = sw / w_out
        stride_h = sh / h_out

        # Start position (top-left of the crop region)
        start_x = cx - sw * 0.5
        start_y = cy - sh * 0.5

        # Source coordinates grid
        x_in = start_x + x_out * stride_w
        y_in = start_y + y_out * stride_h

        # Bilinear Sampling similar to _resize_bilinear_cupy
        # Get integer parts
        x0 = cp.floor(x_in - 0.5).astype(cp.int32)
        y0 = cp.floor(y_in - 0.5).astype(cp.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # Compute weights
        # dx = x_in - (x0 + 0.5)
        wx = x_in - (x0.astype(cp.float32) + 0.5)
        wy = y_in - (y0.astype(cp.float32) + 0.5)

        # Sample with boundary check (padding with 114 or 0 if out of bounds)
        # Using 0 (grayish/black) for padding is common, or mean value
        # dwpose usually uses [123.675, 116.28, 103.53] for mean, but padding value is usually 0 or 127
        # Let's use 0 to be safe/neutral
        pad_val = 0.0

        # Helper to sample safely
        def sample_at(y, x):
            # Mask for valid coordinates
            valid_x = (x >= 0) & (x < w_img)
            valid_y = (y >= 0) & (y < h_img)
            valid = valid_x[None, :] & valid_y[:, None]  # Outer product mask

            # Clamp for sampling (values won't be used where invalid, but need valid indices)
            xc = cp.clip(x, 0, w_img - 1)
            yc = cp.clip(y, 0, h_img - 1)

            val = img[yc[:, None], xc[None, :]]  # (H_out, W_out, 3)

            # Apply mask (broadcasting over channels)
            return cp.where(valid[..., None], val, pad_val)

        # Sample 4 points
        # x0, x1 are (W_out,), y0, y1 are (H_out,)
        # We need grid sampling: img[y_indices[:, None], x_indices[None, :]]

        v00 = sample_at(y0, x0)
        v01 = sample_at(y0, x1)
        v10 = sample_at(y1, x0)
        v11 = sample_at(y1, x1)

        # Interpolate
        # wx is (W_out,), wy is (H_out,)
        # Broadcasting: (H, 1, 1) * (1, W, 1)
        wx = wx[None, :, None]
        wy = wy[:, None, None]

        result = (
            v00 * (1 - wy) * (1 - wx)
            + v01 * (1 - wy) * wx
            + v10 * wy * (1 - wx)
            + v11 * wy * wx
        )

        return result

    def _preprocess_pose(
        self, img: np.ndarray, bboxes: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Preprocess for pose - ALL operations on GPU using CuPy.

        Returns:
            crops: (N, 3, H, W) preprocessed crops
            centers: (N, 2) bbox centers
            scales: (N, 2) bbox scales
        """
        if bboxes.shape[0] == 0:
            return (
                cp.zeros((0, 3, *self.pose_input_size), dtype=cp.float32),
                cp.zeros((0, 2), dtype=cp.float32),
                cp.zeros((0, 2), dtype=cp.float32),
            )

        # Move image to GPU once
        img_gpu = cp.asarray(img, dtype=cp.float32)

        # Use pre-allocated constants (optimization)
        mean = self.pose_mean_gpu
        std = self.pose_std_gpu

        h_out, w_out = self.pose_input_size
        num_boxes = bboxes.shape[0]

        # OPTIMIZATION: Pre-allocate buffers instead of dynamic append
        crops = cp.empty((num_boxes, 3, h_out, w_out), dtype=cp.float32)
        centers = cp.empty((num_boxes, 2), dtype=cp.float32)
        scales = cp.empty((num_boxes, 2), dtype=cp.float32)

        for i in range(num_boxes):
            bbox = bboxes[i]

            # Check for inf/nan values
            if not cp.all(cp.isfinite(bbox)):
                pb_utils.Logger.log_warn(
                    f"Skipping bbox {i} with inf/nan values: {bbox}"
                )
                crops[i] = 0.0
                centers[i] = 0.0
                scales[i] = 1.0
                continue

            # Clip bbox values to reasonable range
            bbox_clipped = cp.clip(bbox, -1e6, 1e6)

            # Calculate Center and Scale (with Padding + Aspect Ratio Fix)
            center, scale = self._get_center_scale_cupy(bbox_clipped)

            centers[i] = center
            scales[i] = scale

            # Crop and Resize using Affine Logic
            crop_resized = self._crop_and_resize_cupy(
                img_gpu, center, scale, (h_out, w_out)
            )

            # Normalize on GPU
            crop_normalized = (crop_resized - mean) / std
            crop_normalized = crop_normalized.astype(cp.float32)

            # HWC -> CHW
            crop_chw = crop_normalized.transpose(2, 0, 1)
            crops[i] = crop_chw

        return crops, centers, scales

    def _run_pose(
        self, image: np.ndarray, det_boxes: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Run pose estimation, keep on GPU."""
        if det_boxes.shape[0] == 0:
            return cp.zeros((0, 133, 2), dtype=cp.float32), cp.zeros(
                (0, 133), dtype=cp.float32
            )

        batch, centers, scales = self._preprocess_pose(image, det_boxes)

        # OPTIMIZATION: Use DLPack for zero-copy GPU tensor transfer
        batch_tensor = pb_utils.Tensor.from_dlpack("input", batch.toDlpack())

        response = pb_utils.InferenceRequest(
            model_name=self.pose_model,
            requested_output_names=["simcc_x", "simcc_y"],
            inputs=[batch_tensor],
        ).exec()

        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())

        # Convert to CuPy (keep on GPU)
        simcc_x = tensor_to_cupy(
            pb_utils.get_output_tensor_by_name(response, "simcc_x")
        )
        simcc_y = tensor_to_cupy(
            pb_utils.get_output_tensor_by_name(response, "simcc_y")
        )

        # Decode on GPU and restore coordinates to original image space
        keypoints, scores = self._decode_simcc_cupy(simcc_x, simcc_y, centers, scales)
        return keypoints, scores

    def _decode_simcc_cupy(
        self,
        simcc_x: cp.ndarray,
        simcc_y: cp.ndarray,
        centers: cp.ndarray,
        scales: cp.ndarray,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Decode SimCC on GPU using CuPy and restore coordinates to original image."""
        n_inst, n_kpt, _ = simcc_x.shape
        simcc_x_flat = simcc_x.reshape(n_inst * n_kpt, -1)
        simcc_y_flat = simcc_y.reshape(n_inst * n_kpt, -1)

        # Get argmax
        x_locs = cp.argmax(simcc_x_flat, axis=1)
        y_locs = cp.argmax(simcc_y_flat, axis=1)
        locs = cp.stack((x_locs, y_locs), axis=-1).astype(cp.float32)

        # Get max values
        max_val_x = cp.max(simcc_x_flat, axis=1)
        max_val_y = cp.max(simcc_y_flat, axis=1)
        scores = cp.minimum(max_val_x, max_val_y)

        # Filter invalid keypoints
        locs[scores <= 0] = -1

        # Reshape
        locs = locs.reshape(n_inst, n_kpt, 2)
        scores = scores.reshape(n_inst, n_kpt)

        # Apply split ratio to get coordinates in model input space (384x288)
        locs /= self.simcc_split_ratio

        # Restore coordinates to original image space
        # Formula: keypoints / model_input_size * scale + center - scale / 2
        # Used by onnxpose.py: keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        model_size = self.pose_model_size_gpu  # Pre-allocated constant

        # Reshape for broadcasting
        for i in range(locs.shape[0]):
            center = centers[i]  # (2,) [cx, cy]
            scale = scales[i]  # (2,) [w, h]

            # Since scale is now the "padded" and "aspect-ratio fixed" scale, this matches onnxpose.py logic
            top_left = center - scale * 0.5
            locs[i] = (locs[i] / model_size) * scale + top_left

        return locs, scores

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                in_image = pb_utils.get_input_tensor_by_name(request, "IMAGE")
                if in_image is None:
                    raise ValueError("IMAGE input not found")

                image = in_image.as_numpy()
                if image.ndim == 4:
                    image = image[0]
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                # Run pipeline on GPU
                det_boxes = self._run_detection(image)
                keypoints, scores = self._run_pose(image, det_boxes)

                # Only convert to CPU/NumPy at the very end
                bboxes_np = (
                    cp.asnumpy(det_boxes).astype(np.float32)
                    if det_boxes.size > 0
                    else np.zeros((0, 4), np.float32)
                )
                keypoints_np = cp.asnumpy(keypoints).astype(np.float32)
                scores_np = cp.asnumpy(scores).astype(np.float32)

                # Create output tensors
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("KEYPOINTS", keypoints_np),
                            pb_utils.Tensor("SCORES", scores_np),
                            pb_utils.Tensor("BBOXES", bboxes_np),
                        ]
                    )
                )
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
                )

        return responses

    def finalize(self):
        pass
