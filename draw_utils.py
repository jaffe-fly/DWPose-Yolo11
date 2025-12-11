import numpy as np

from dwpose import draw_pose


def convert_to_pose_dict(keypoints, scores, img_width, img_height, score_thr=0.3):
    nums = len(keypoints)
    if nums == 0:
        return dict(
            bodies=dict(candidate=np.zeros((0, 2)), subset=np.zeros((0, 18))),
            hands=np.array([]),
            faces=np.array([]),
        )

    coco_to_openpose = {
        0: 0,  # Nose
        # 1: Neck (需计算)
        2: 6,  # RShoulder
        3: 8,  # RElbow
        4: 10,  # RWrist
        5: 5,  # LShoulder
        6: 7,  # LElbow
        7: 9,  # LWrist
        8: 12,  # RHip
        9: 14,  # RKnee
        10: 16,  # RAnkle
        11: 11,  # LHip
        12: 13,  # LKnee
        13: 15,  # LAnkle
        14: 2,  # REye
        15: 1,  # LEye
        16: 4,  # REar
        17: 3,  # LEar
    }

    openpose_kpts = []
    openpose_scores = []

    for person_kpts, person_scores in zip(keypoints, scores):
        person_18kpts = []
        person_18scores = []

        for op_idx in range(18):
            if op_idx == 1:  # Neck
                l_sh = person_kpts[5]  # COCO left shoulder
                r_sh = person_kpts[6]  # COCO right shoulder
                l_sc = person_scores[5]
                r_sc = person_scores[6]

                neck_x = (l_sh[0] + r_sh[0]) / 2.0
                neck_y = (l_sh[1] + r_sh[1]) / 2.0
                neck_score = (
                    min(l_sc, r_sc) if (l_sc > score_thr and r_sc > score_thr) else 0
                )
                person_18kpts.append([neck_x, neck_y])
                person_18scores.append(neck_score)
            else:
                coco_idx = coco_to_openpose[op_idx]
                kpt = person_kpts[coco_idx]
                score = person_scores[coco_idx]
                person_18kpts.append([kpt[0], kpt[1]])
                person_18scores.append(score)

        openpose_kpts.append(person_18kpts)
        openpose_scores.append(person_18scores)

    openpose_kpts = np.array(openpose_kpts)  # (N, 18, 2)
    openpose_scores = np.array(openpose_scores)  # (N, 18)

    openpose_kpts[..., 0] /= float(img_width)
    openpose_kpts[..., 1] /= float(img_height)

    body = openpose_kpts[:, :18].copy()  # (N, 18, 2)
    candidate = body.reshape(nums * 18, 2)  # (N*18, 2)

    subset = []
    for person_idx in range(nums):
        person_subset = []
        for kpt_idx in range(18):
            score = openpose_scores[person_idx, kpt_idx]
            if score > score_thr:
                person_subset.append(18 * person_idx + kpt_idx)
            else:
                person_subset.append(-1)
        subset.append(person_subset)

    subset = np.array(subset, dtype=np.float32)  # (N, 18)

    for person_idx in range(nums):
        for kpt_idx in range(18):
            score = openpose_scores[person_idx, kpt_idx]
            if score <= score_thr:
                global_idx = 18 * person_idx + kpt_idx
                candidate[global_idx] = -1

    hands_list = []
    for person_kpts, person_scores in zip(keypoints, scores):
        left_hand = []
        for i in range(91, 112):
            kpt = person_kpts[i]
            score = person_scores[i]
            if score > score_thr and kpt[0] >= 0:
                left_hand.append(
                    [kpt[0] / float(img_width), kpt[1] / float(img_height)]
                )
            else:
                left_hand.append([-1, -1])

        right_hand = []
        for i in range(112, 133):
            kpt = person_kpts[i]
            score = person_scores[i]
            if score > score_thr and kpt[0] >= 0:
                right_hand.append(
                    [kpt[0] / float(img_width), kpt[1] / float(img_height)]
                )
            else:
                right_hand.append([-1, -1])

        hands_list.append(left_hand)
        hands_list.append(right_hand)

    hands = np.array(hands_list) if hands_list else np.array([])

    bodies = dict(candidate=candidate, subset=subset)
    pose = dict(bodies=bodies, hands=hands, faces=np.array([]))

    return pose


def draw_triton_result(img, keypoints, scores, score_thr=0.3, alpha=0.6):
    H, W = img.shape[:2]
    pose = convert_to_pose_dict(keypoints, scores, W, H, score_thr)
    return draw_pose(pose, H, W, background_img=img, alpha=alpha)
