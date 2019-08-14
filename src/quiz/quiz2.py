import torch

def images_to_levels(target, num_level_anchors):
    """이미지별로 구성된 target의 형태를 level별로 구성하도록 변경합니다.
    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        #TODO level_targets.append()
        start = end
    return level_targets