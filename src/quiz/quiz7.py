import torch


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    """bbox를 delta형태로 변경합니다.
    [x1, y1, x2, y2] -> [x, y, w, h]
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5 #TODO
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5 #TODO
    pw = proposals[..., 2] - proposals[..., 0] + 1.0 #TODO
    ph = proposals[..., 3] - proposals[..., 1] + 1.0 #TODO

    gx = (gt[..., 0] + gt[..., 2]) * 0.5 #TODO
    gy = (gt[..., 1] + gt[..., 3]) * 0.5 #TODO
    gw = gt[..., 2] - gt[..., 0] + 1.0 #TODO
    gh = gt[..., 3] - gt[..., 1] + 1.0 #TODO

    dx = (gx - px) / pw #TODO
    dy = (gy - py) / ph #TODO
    dw = torch.log(gw / pw) #TODO
    dh = torch.log(gh / ph) #TODO
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas