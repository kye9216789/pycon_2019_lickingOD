import copy
from collections import Sequence
from math import floor

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.runner import obj_from_dict

from .. import datasets

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()


def get_dataset(data_cfg):

    data_info = copy.deepcopy(data_cfg)
    data_info['ann_file'] = data_cfg['ann_file']
    data_info['proposal_file'] = None
    data_info['img_prefix'] = data_cfg['img_prefix']
    dset = obj_from_dict(data_info, datasets)
    return dset


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(
        d_type
    )


def generate_heatmap(
    bbox_shape, kx_bbox, ky_bbox, heatmap_size=(64, 48), gaussian_size=5, gaussian_std=2
):
    """
    Args:
        bbox_shape (list): h, w of bbox
        kx_bbox, ky_bbox (int): keypoint coordination in bbox
        heatmap_size (tuple): default value is (64, 48)
        gaussian_size (int): gaussian size
        gaussian_std (int): gaussian std

    Returns:
        heatmap (np.darray): heatmap w/wo gaussian
        target_weight (np.darray): if heatmap has gaussian, 1 else 0
    """
    assert gaussian_size % 2 != 0

    h, w = bbox_shape
    assert 0 <= kx_bbox < w and 0 <= ky_bbox < h

    heatmap = np.zeros(heatmap_size)
    ratio_w = heatmap_size[1] / w
    ratio_h = heatmap_size[0] / h

    resized_kx_bbox = floor(kx_bbox * ratio_w)
    resized_ky_bbox = floor(ky_bbox * ratio_h)

    # Add temp pad for gaussian addition
    pad_size = gaussian_size // 2
    heatmap_pad = np.pad(heatmap, (pad_size, pad_size), mode="constant")

    gaussian = make_gaussian([gaussian_size, gaussian_size], gaussian_std)

    # heatmap coords adjusting gaussian
    gx1 = resized_kx_bbox
    gy1 = resized_ky_bbox

    gx2 = resized_kx_bbox + gaussian_size
    gy2 = resized_ky_bbox + gaussian_size

    heatmap_pad[gy1:gy2, gx1:gx2] = gaussian
    heatmap = heatmap_pad[pad_size:-pad_size, pad_size:-pad_size]
    target_weight = np.ones([1, 1])
    return heatmap, target_weight


def add_boundary_noise(img_shape, x1, y1, x2, y2, kx_bbox=None, ky_bbox=None):
    """Add noise when cropping bbox

    Args:
        img_shape (list): [h, w]
        x1, y1, x2, y2 (int): bbox coordination
        kx_bbox, ky_bbox (int): keypoint coordination, default is None

    Returns:
        new_x1, new_x2, new_y1, new_y2 (int): new bbox coordination
    """
    img_h, img_w = img_shape

    randint_w = sorted((0, (x2 - x1) // 3 + 1))
    randint_h = sorted((0, (y2 - y1) // 3 + 1))

    dx1 = min(x1, np.random.randint(*randint_w))
    dx2 = min(img_w - x2, np.random.randint(*randint_w))
    dy1 = min(y1, np.random.randint(*randint_h))
    dy2 = min(img_h - y2, np.random.randint(*randint_h))

    new_x1 = x1 - dx1
    new_x2 = x2 + dx2
    new_y1 = y1 - dy1
    new_y2 = y2 + dy2

    if kx_bbox is not None and ky_bbox is not None:
        new_kx_bbox = kx_bbox + dx1
        new_ky_bbox = ky_bbox + dy1
        return new_x1, new_x2, new_y1, new_y2, new_kx_bbox, new_ky_bbox
    return new_x1, new_x2, new_y1, new_y2


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
