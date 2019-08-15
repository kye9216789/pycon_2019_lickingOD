import torch

from src.core.bbox.assigners import AssignResult


def assign_wrt_overlaps(overlaps, cfg_assigner):
    """gt와 bbox의 overlap에 적절한 label을 할당합니다.

    Args:
        overlaps (Tensor): k번째 gt_bbox 와 n 번째 bbox의 overlap, shape = (k, n).

    Returns:
        :obj:`AssignResult`: 각 anchor에 할당된 label 정보를 handling하는 Data class의 객체.
    """

    positive_threshold = cfg_assigner.pos_iou_thr
    negative_threshold = cfg_assigner.neg_iou_thr
    mininum_positive_threshold = cfg_assigner.min_pos_iou
    if overlaps.numel() == 0:
        raise ValueError('No gt or proposals')

    num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

    # 1. 모든 overlap 관계에 기본적으로 -1을 assign합니다.
    #    -1은 not attribute 입니다.
    assigned_gt_inds = overlaps.new_full(
        (num_bboxes, ), -1, dtype=torch.long)

    # 1-1. 각 Anchor마다 가장 높은 overlap을 갖는 gt의 index와 그 값을 구합니다.
    max_overlaps, argmax_overlaps = overlaps.max(dim=0)
    # 1-2. 1-1에서 수행한 작업을 gt기준으로 한번 더 수행합니다.
    gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

    # 2. negative IoU threshold에 못 미치는 IoU를 갖는 anchor에 negative label을 할당합니다.
    #    1-1의 결과를 활용합니다.
    assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < negative_threshold)] = 0 #TODO

    # 3. positive IoU threshold보다 높은 IoU를 갖는 anchor에 positive label을 할당합니다.
    #    1-1의 결과를 활용합니다.
    pos_inds = max_overlaps >= positive_threshold #TODO
    assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1 #TODO

    # 4. 각 gt에 대하여 최소한의 IoU threshold를 넘는 anchor중 가장 높은 IoU를 갖는 anchor에 대하여 positive label을 할당합니다.
    #    1-2의 결과를 활용합니다.
    for i in range(num_gts):
        if gt_max_overlaps[i] >= mininum_positive_threshold: #TODO
            max_iou_inds = overlaps[i, :] == gt_max_overlaps[i] #TODO
            assigned_gt_inds[max_iou_inds] = i + 1 #TODO

    return AssignResult(
        num_gts, assigned_gt_inds, max_overlaps)