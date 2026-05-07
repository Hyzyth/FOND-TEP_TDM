"""
auto_prompting/box_utils.py
============================
3-D bounding-box IoU and NMS utilities.
"""

from __future__ import annotations

import numpy as np


def iou_3d(a: list, b: list) -> float:
    """Intersection-over-Union for two 3-D boxes.

    Each box is [z0, z1, y0, y1, x0, x1].
    """
    z0 = max(a[0], b[0]); z1 = min(a[1], b[1])
    y0 = max(a[2], b[2]); y1 = min(a[3], b[3])
    x0 = max(a[4], b[4]); x1 = min(a[5], b[5])

    inter = max(0, z1 - z0) * max(0, y1 - y0) * max(0, x1 - x0)
    if inter == 0:
        return 0.0

    vol_a = (a[1]-a[0]) * (a[3]-a[2]) * (a[5]-a[4])
    vol_b = (b[1]-b[0]) * (b[3]-b[2]) * (b[5]-b[4])
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def nms_3d(proposals: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """Non-maximum suppression on 3-D proposal boxes.

    Assumes proposals are sorted by voxel_count descending.
    """
    kept: list[dict] = []
    for p in proposals:
        dominated = any(
            iou_3d(p["bbox_3d"], k["bbox_3d"]) > iou_threshold
            for k in kept
        )
        if not dominated:
            kept.append(p)
    return kept


def intersects_3d(a: list, b: list) -> bool:
    """Return True if two 3-D boxes have any overlap."""
    return (a[0] < b[1] and a[1] > b[0] and
            a[2] < b[3] and a[3] > b[2] and
            a[4] < b[5] and a[5] > b[4])
