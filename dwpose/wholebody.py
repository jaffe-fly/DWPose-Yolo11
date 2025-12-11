import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from .onnxpose import inference_pose


class Wholebody:
    def __init__(
        self,
        det_model_path=None,
        pose_model_path=None,
        device="cuda:0",
        det_conf=0.45,
        det_iou=0.25,
        pose_score_thr=0.4,
    ):
        providers = (
            ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        if det_model_path is None or pose_model_path is None:
            raise ValueError("model path must be provided and cannot be None.")

        self.session_pose = ort.InferenceSession(
            path_or_bytes=pose_model_path, providers=providers
        )
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.pose_score_thr = pose_score_thr
        self.device = device
        self.model = YOLO(det_model_path, task="detect")
        self.use_half = True if torch.cuda.is_available() else False

    def __call__(self, oriImg):
        results = self.model.predict(
            source=oriImg,
            imgsz=640,
            device=self.device,
            classes=[0],
            half=self.use_half,
            conf=self.det_conf,
            iou=self.det_iou,
            verbose=False,
            max_det=10,
            show=False,
        )

        if not results or results[0].boxes is None:
            det_result = np.array([])
        else:
            det_result = results[0].boxes.xyxy.cpu().numpy()

        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > self.pose_score_thr,
            keypoints_info[:, 6, 2:4] > self.pose_score_thr,
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        return keypoints, scores
