def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    """각 anchor의 모든 모서리가 이미지 내에 들어가는지 여부를 확인합니다.

    Args:
        flat_anchors (Tensor): anchor의 list. shape=(n, 4)
        valid_flags (Tensor): anchor의 valid flags. shape=(n, 1)

    Returns:
        : Tensor: anchor의 inside flags.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
         #TODO
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags