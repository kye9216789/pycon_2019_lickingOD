import torch

from src.quiz.quiz9 import delta2bbox
from src.core import multiclass_nms
from src.core.utils import describe

def get_bboxes(cls_scores, bbox_preds, img_metas, cfg,
                anchor_generators, anchor_strides, cls_out_channels):
    """anchor, cls_score, bbox_pred를 통해 bbox coordination과 score의 prediction값을 구합니다.
    """
    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)

    mlvl_anchors = [
        anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                            anchor_strides[i], device='cpu')
        for i in range(num_levels)
    ]
    result_list = []
    for img_id in range(len(img_metas)):
        cls_score_list = [ #TODO
            cls_scores[i][img_id].detach() for i in range(num_levels) #TODO
        ] #TODO
        bbox_pred_list = [ #TODO
            bbox_preds[i][img_id].detach() for i in range(num_levels) #TODO
        ] #TODO
        img_shape = img_metas[img_id]['img_shape']
        scale_factor = img_metas[img_id]['scale_factor']
        proposals = get_bboxes_single(cls_score_list, bbox_pred_list,
                                        mlvl_anchors, img_shape,
                                        scale_factor, cfg, cls_out_channels)
        result_list.append(proposals)
    return result_list


def get_bboxes_single(cls_scores,
                      bbox_preds,
                      mlvl_anchors,
                      img_shape,
                      scale_factor,
                      cfg,
                      cls_out_channels,
                      target_means=[.0, .0, .0, .0],
                      target_stds=[1.0, 1.0, 1.0, 1.0]):
    assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
    mlvl_bboxes = []
    mlvl_scores = []
    for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        cls_score = cls_score.permute(1, 2, 0).reshape(
            -1, cls_out_channels)

        scores = cls_score.sigmoid()

        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:

            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
        bboxes = delta2bbox(anchors, bbox_pred, target_means,
                            target_stds, img_shape)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
    mlvl_bboxes = torch.cat(mlvl_bboxes)
    mlvl_scores = torch.cat(mlvl_scores)

    padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
    mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
    det_bboxes, det_labels = multiclass_nms(
        mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
    return det_bboxes, det_labels