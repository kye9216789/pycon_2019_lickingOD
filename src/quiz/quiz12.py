import torch


#TODO base_size를 기준으로 scale, ratio에 따라 base_anchors를 구합니다.
def gen_base_anchors(base_size, scales, ratios):
    """ scales와 ratios를 통해 base_anchors를 생성합니다.
        생성된 base_anchors는 feature map의 각 기준점마다 한 묶음씩 생성됩니다.
    """
    w = base_size
    h = base_size

    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    #TODO h_ratios =
    #TODO w_ratios =

    #TODO ws = (w * ).view(-1)
    #TODO hs = (h * ).view(-1)

    #base_anchors : list([x1, y1, x2, y2])
    base_anchors = torch.stack(
        [
            x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
        ],
        dim=-1).round()

    return base_anchors