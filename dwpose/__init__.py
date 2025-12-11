import os
from pathlib import Path
from typing import Any

from numpy import dtype, ndarray
from numpy._typing._array_like import NDArray
from ruamel.yaml import YAML

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from . import util
from .wholebody import Wholebody


def load_dwpose_config():
    try:
        # Assuming config is in ../../config/dwpose.yaml relative to this file
        config_path = (
            Path(__file__).resolve().parent.parent.parent / "config" / "dwpose.yaml"
        )
        if not config_path.exists():
            print(f"Warning: Config file not found at {config_path}")
            return {}

        yaml = YAML()
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.load(f)
            return data.get("dwpose", {})
    except Exception as e:
        print(f"Warning: Failed to load dwpose config: {e}")
        return {}


CONFIG = load_dwpose_config()
DET_CONF = CONFIG.get("det_conf", 0.45)
DET_IOU = CONFIG.get("det_iou", 0.25)
POSE_SCORE_THR = CONFIG.get("pose_score_thr", 0.4)


def draw_pose(pose, H, W, background_img=None, alpha=0.6):
    """
    绘制姿态骨架

    Args:
        pose: 姿态字典
        H: 图像高度
        W: 图像宽度
        background_img: 背景图像（可选），如果提供则在原图上绘制
        alpha: 骨架透明度（0-1），仅在 background_img 不为 None 时有效

    Returns:
        绘制了姿态的图像
    """
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    if background_img is not None:
        canvas = background_img.copy().astype(np.float32)
        skeleton_canvas = np.zeros_like(background_img, dtype=np.uint8)
    else:
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        skeleton_canvas = canvas.copy()

    skeleton_canvas = util.draw_bodypose(skeleton_canvas, candidate, subset)
    skeleton_canvas = util.draw_handpose(skeleton_canvas, hands)
    # skeleton_canvas = util.draw_facepose(skeleton_canvas, faces)
    if background_img is not None:
        canvas = np.clip(
            canvas + skeleton_canvas.astype(np.float32) * alpha, 0, 255
        ).astype(np.uint8)
    else:
        canvas = skeleton_canvas

    return canvas


class DWposeDetector:
    def __init__(
        self,
        det_model_path=None,
        pose_model_path=None,
        device="cuda:0",
    ):
        self.det_conf = DET_CONF
        self.det_iou = DET_IOU
        self.pose_score_thr = POSE_SCORE_THR

        self.pose_estimation = Wholebody(
            det_model_path=det_model_path,
            pose_model_path=pose_model_path,
            device=device,
            det_conf=self.det_conf,
            det_iou=self.det_iou,
            pose_score_thr=self.pose_score_thr,
        )

    def __call__(self, oriImg, return_pose_dict=False, draw_on_image=False, alpha=0.6):
        """
        检测姿态

        Args:
            oriImg: 输入图像 (H, W, 3) BGR 格式
            return_pose_dict: 是否返回原始姿态字典
            draw_on_image: 是否在原图上绘制（而不是黑色背景）
            alpha: 骨架透明度（0-1），仅在 draw_on_image=True 时有效

        Returns:
            如果 return_pose_dict=True: 返回 (pose_dict, visualized_image)
            否则: 返回可视化图像
        """
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > self.pose_score_thr:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < self.pose_score_thr
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict[str, dict[str, ndarray[Any, dtype]] | NDArray](
                bodies=bodies, hands=hands, faces=faces
            )

            background = oriImg if draw_on_image else None

            visualized = draw_pose(pose, H, W, background_img=background, alpha=alpha)

            if return_pose_dict:
                return pose, visualized
            else:
                return visualized
