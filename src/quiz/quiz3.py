import torch
import torch.nn.functional as F
from src.core.utils import describe


def loss_single(cls_score, bbox_pred, labels, label_weights,
                bbox_targets, bbox_weights, num_total_samples, cfg, cls_out_channels):
    # classification loss로는 focal loss를 사용합니다.
    labels = labels.reshape(-1, cls_out_channels)
    print("##################################")
    print("label shape : ", describe(labels))
    label_weights = label_weights.reshape(-1, cls_out_channels)
    print("class score shape before reshape : ", describe(cls_score))
    cls_score = cls_score. #TODO

    assert (describe(labels) == describe(cls_score))
    loss_cls = weighted_sigmoid_focal_loss(
        cls_score,
        labels,
        label_weights,
        gamma=cfg.gamma,
        alpha=cfg.alpha,
        avg_factor=num_total_samples)

    # regression loss로는 smooth L1 loss를 사용합니다.
    bbox_targets = bbox_targets.reshape(-1, 4)
    print("bbox target shape : ", describe(bbox_targets))
    bbox_weights = bbox_weights.reshape(-1, 4)
    print("bbox pred shape before reshape : ", describe(bbox_pred))
    bbox_pred = bbox_pred. #TODO
    assert (describe(bbox_targets) == describe(bbox_pred))
    loss_reg = weighted_smoothl1(
        bbox_pred,
        bbox_targets,
        bbox_weights,
        beta=cfg.smoothl1_beta,
        avg_factor=num_total_samples)
    return loss_cls, loss_reg


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta)
    return torch.sum(loss * weight)[None] / avg_factor


def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha)[None] / avg_factor


def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25):
    #FL(pt) = -alpha * (1-pt) ^ gamma * log(pt)
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    return loss.sum()