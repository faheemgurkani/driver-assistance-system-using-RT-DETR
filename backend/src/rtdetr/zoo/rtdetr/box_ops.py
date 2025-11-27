'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
'''

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    # Replace NaN and inf with 0
    x_c = torch.where(torch.isfinite(x_c), x_c, torch.zeros_like(x_c))
    y_c = torch.where(torch.isfinite(y_c), y_c, torch.zeros_like(y_c))
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    h = torch.where(torch.isfinite(h), h, torch.zeros_like(h))
    
    # Ensure width and height are non-negative (clamp to avoid invalid boxes)
    w = torch.clamp(w, min=0)
    h = torch.clamp(h, min=0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    boxes = torch.stack(b, dim=-1)
    # Sanitize: ensure x1 >= x0 and y1 >= y0 using torch.maximum (non-inplace)
    if boxes.numel() > 0:
        # Create new tensor instead of inplace modification to avoid MPS issues
        x0, y0, x1, y1 = boxes.unbind(-1)
        x1 = torch.maximum(x1, x0)
        y1 = torch.maximum(y1, y0)
        boxes = torch.stack([x0, y0, x1, y1], dim=-1)
    return boxes


def sanitize_boxes(boxes):
    """
    Sanitize boxes to ensure they are valid (x1 >= x0, y1 >= y0).
    Boxes should be in [x0, y0, x1, y1] format.
    Handles NaN, inf, and invalid coordinates.
    """
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.clone()
    
    # Replace NaN and inf with 0
    boxes = torch.where(torch.isfinite(boxes), boxes, torch.zeros_like(boxes))
    
    # Ensure x1 >= x0 and y1 >= y0 using torch.maximum (non-inplace)
    x0, y0, x1, y1 = boxes.unbind(-1)
    x1 = torch.maximum(x1, x0)
    y1 = torch.maximum(y1, y0)
    boxes = torch.stack([x0, y0, x1, y1], dim=-1)
    
    # Clamp to reasonable ranges (assuming normalized coordinates [0, 1])
    # This prevents extreme values that might cause issues
    boxes = torch.clamp(boxes, min=-10.0, max=10.0)
    
    # Final check: ensure x1 >= x0 and y1 >= y0 again after clamping (non-inplace)
    x0, y0, x1, y1 = boxes.unbind(-1)
    x1 = torch.maximum(x1, x0)
    y1 = torch.maximum(y1, y0)
    boxes = torch.stack([x0, y0, x1, y1], dim=-1)
    
    return boxes


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # Sanitize boxes to ensure they are valid before computing GIoU
    boxes1 = sanitize_boxes(boxes1)
    boxes2 = sanitize_boxes(boxes2)
    
    # Final validation and fix: ensure boxes are valid (non-inplace)
    if boxes1.numel() > 0:
        invalid_x = boxes1[:, 2] < boxes1[:, 0]
        invalid_y = boxes1[:, 3] < boxes1[:, 1]
        if invalid_x.any() or invalid_y.any():
            # Create new tensor instead of inplace modification
            x0, y0, x1, y1 = boxes1.unbind(-1)
            x1 = torch.where(invalid_x, x0, x1)
            y1 = torch.where(invalid_y, y0, y1)
            boxes1 = torch.stack([x0, y0, x1, y1], dim=-1)
    
    if boxes2.numel() > 0:
        invalid_x = boxes2[:, 2] < boxes2[:, 0]
        invalid_y = boxes2[:, 3] < boxes2[:, 1]
        if invalid_x.any() or invalid_y.any():
            # Create new tensor instead of inplace modification
            x0, y0, x1, y1 = boxes2.unbind(-1)
            x1 = torch.where(invalid_x, x0, x1)
            y1 = torch.where(invalid_y, y0, y1)
            boxes2 = torch.stack([x0, y0, x1, y1], dim=-1)
    
    # degenerate boxes gives inf / nan results
    # so do an early check (now should always pass after all fixes)
    if boxes1.numel() > 0:
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), f"boxes1 still invalid after sanitization"
    if boxes2.numel() > 0:
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), f"boxes2 still invalid after sanitization"
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)