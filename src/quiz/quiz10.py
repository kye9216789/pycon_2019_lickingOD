import torch

def nms(boxes, scores, overlap_thr=0.5, top_k=200):
    """유효한 Bounding Box만을 남겨놓기 위하여 Non-maximum suppression 을 수행합니다.
    Args:
        boxes: (tensor) bbox의 position 예측값, shape=(n, 4)
        scores: (tensor) bbox의 class 예측값, shape=(n, )
        overlap_thr: (float) threshold
        top_k: (int) 한 이미지당 bbox의 최대 개수
    Return:
        keep: (Tensor) 결과로 사용할 bbox의 index
        count: (int) 결과로 사용할 bbox의 개수
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1 + 1, y2 - y1 + 1)
    v, idx =   #TODO : score를 오름차순으로 정렬합니다.
    idx =   #TODO : score가 높은 순서대로 top_k만큼만을 선택합니다.

    count = 0
    while idx.numel() > 0:
        i =   #TODO : 현재 가장 score가 높은 bbox의 index를 선택합니다.
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx =   #TODO 결과로 선택된 bbox의 index를 목록에서 제거합니다.

        # 현재 가장 score가 높은 bbox의 좌표를 획득합니다.
        # bbox들의 좌표는 각각 x1, y1, x2, y2에 저장되어 있으며,
        # 선택된 bbox의 좌표는 각각 xx1, yy1, xx2, yy2에 저장됩니다.
        # torch.index_select() 참조 : https://pytorch.org/docs/stable/torch.html#torch.index_select
        xx1 =   #TODO
        yy1 =   #TODO
        xx2 =   #TODO
        yy2 =   #TODO
        w =   #TODO
        h =   #TODO
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter_area = w * h
        # 가장 높은 score의 bbox와 비교중인 2인자 bbox의 넓이를 가져옵니다.
        submax_areas =  #TODO
        # 두 bbox의 iou를 계산합니다.
        # threshold 이상의 iou를 가질 경우 2인자 bbox는 제거됩니다.
        union_area = (submax_areas - inter_area) + area[i]
        iou = inter_area / union_area
        # keep only elements with an iou <= overlap
        idx = idx[iou.le(overlap_thr)]
    keep = keep[keep > 0]
    return keep, count