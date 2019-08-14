from src.quiz.quiz4 import bbox_overlaps
from src.quiz.quiz5 import assign_wrt_overlaps


def assign(bboxes, gt_bboxes, cfg_assigner):
    """Assign gt to bboxes.
    본 method는 각각의 anchor에 ground truth(이하 gt)를 할당하는 역할을 합니다.
    해당 작업이 완료되면 각각의 anchor는 -1, 0, 또는 어떠한 양수값을 갖게 됩니다.
    -1은 학습에 사용되지 않는 anchor를 뜻하며
    0은 negative sample을 뜻하고
    나머지 어떠한 양수값들은 할당된 gt bounding box의 index 번호를 뜻합니다.

    위 작업은 아래 정의된 assign_wrt_overlaps 함수에서 아래 순서대로 이뤄집니다.

    1. 모든 anchor box에 -1을 할당합니다.
    2. 모든 gt들과의 iou가 negative threshold값을 넘지 못하는 anchor box에 0을 할당합니다.
       다시 말해 어떠한 gt와도 유의미한 상관관계를 갖지 않는 anchor box를 negative sample로 사용한다는 말 입니다.
    3. gt와의 iou가 positive threshold를 넘는 anchor box에 대하여, 가장 가까운(iou가 가장 높은) gt의 index를 부여합니다.
    4. 각 gt에 대하여 가장 iou가 높은 anchor box에 해당 gt의 index값을 할당합니다.
       3번 과정이 anchor box를 기준으로 이루어졌다면, 본 과정은 gt를 기준으로 이뤄집니다.
       만일 어떠한 gt를 기준으로 각 anchor들과의 iou가 모두 positive threshold보다 낮다면 해당 gt에 대해 학습할 positive sample이 없는 상황이 발생합니다.
       이럴 때에는 비록 iou가 충분치 않더라도 가장 가까운 anchor를 positive sample로 삼아 학습하게 됩니다.


    Args:
        bboxes (Tensor): 적절한 label을 할당해줘야 할 anchor box.
                         shape = (n, 4).
        gt_bboxes (Tensor): Ground truth box.
                            shape = (k, 4).

    Returns:
        :obj:`AssignResult`: 각 anchor에 할당된 label 정보를 handling하는 Data class의 객체.
    """

    if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
        raise ValueError('No gt or bboxes')
    bboxes = bboxes[:, :4]
    overlaps = bbox_overlaps(gt_bboxes, bboxes)

    assign_result = assign_wrt_overlaps(overlaps, cfg_assigner)
    return assign_result
