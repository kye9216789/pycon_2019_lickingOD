import mmcv
import numpy as np
import torch


def bbox2result(bboxes, labels, num_classes):
    """detection result를 numpy array형태로 바꿉니다.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): background를 포함한 class의 수

    Returns:
        list(ndarray): 각 클래스별로 정리된 detection 결과
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]
