import numpy as np


def get_anchors(anchor_generators, anchor_strides, featmap_sizes, img_metas):
    """feature map의 크기에 따라 anchor 정보를 계산합니다.

    Args:
        featmap_sizes (list[tuple]): Multi-level feature map size를 뜻 합니다. 리스트의 길이는 Multi-level의 크기와 같으며, FPN에 의해 결정됩니다.
        img_metas (list[dict]): Image의 meta 정보 입니다. 각 이미지마다 하나씩 존재합니다.

    Returns:
        tuple: anchors of each image, valid flags of each image
    """
    num_imgs = len(img_metas)
    num_levels = len(featmap_sizes)

    # 배치 내의 모든 이미지의 크기는 같아야 하며, 따라서 feature map의 크기도 모든 이미지에 대하여 동일하게 정해집니다.
    # anchor는 feature map의 크기에 따라 정해지므로 단 한 번만 anchor를 계산하면 됩니다.
    multi_level_anchors = []
    for i in range(num_levels):
        anchors = anchor_generators[i].grid_anchors(
            featmap_sizes[i], anchor_strides[i], device='cpu')
        multi_level_anchors.append(anchors)
    anchor_list = [multi_level_anchors for _ in range(num_imgs)]

    #TODO anchor의 valid flag를 구합니다.
    #TODO 각 level마다 feature map의 크기를 바탕으로 anchor가 유효한 영역 내에 있는지 확인합니다.
    valid_flag_list = []
    for img_id, img_meta in enumerate(img_metas):
        multi_level_flags = []
        for i in range(num_levels):
            anchor_stride = anchor_strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w, _ = img_meta['pad_shape']
            #TODO valid_feat_h =
            #TODO valid_feat_w =
            flags = anchor_generators[i].valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w))
            multi_level_flags.append(flags)
        valid_flag_list.append(multi_level_flags)

    return anchor_list, valid_flag_list