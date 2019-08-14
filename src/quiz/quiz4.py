import torch

def bbox_overlaps(bboxes1, bboxes2):
    """두 bbox set의 overlap을 계산합니다.
       여기서 두 bbox set이란 gt bbox와 anchor box를 뜻 합니다.
       결과물은 2차원 배열 형태가 되어야 합니다.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4)

    Returns:
        ious(Tensor): shape (m, n)
    """

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    #TODO lt =  # [rows, cols, 2]
    #TODO rb =  # [rows, cols, 2]

    #TODO wh =  # [rows, cols, 2]
    #TODO overlap =
    #TODO area1 =

    #TODO area2 =
    ious = overlap / (area1[:, None] + area2 - overlap)

    return ious