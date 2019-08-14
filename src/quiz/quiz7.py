import torch


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    """bbox를 delta형태로 변경합니다.
    [x1, y1, x2, y2] -> [x, y, w, h]
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px =  * 0.5 #TODO
    py =  * 0.5 #TODO
    pw =  + 1.0 #TODO
    ph =  + 1.0 #TODO

    gx =  * 0.5 #TODO
    gy =  * 0.5 #TODO
    gw =  + 1.0 #TODO
    gh =  + 1.0 #TODO

    dx =  #TODO
    dy =  #TODO
    dw =  #TODO
    dh =  #TODO
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas