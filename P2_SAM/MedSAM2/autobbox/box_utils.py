"""
box_utils.py
============
IoU + merging
"""

import numpy as np


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


def merge_boxes(boxes, iou_thr=0.3):
    merged = []

    for b in boxes:
        keep = True
        for m in merged:
            if iou(b, m) > iou_thr:
                keep = False
                break
        if keep:
            merged.append(b)

    return merged
