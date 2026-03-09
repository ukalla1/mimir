import torch
import torch.nn.functional as F


def infonce_loss(h_food, anchors, positives, negatives, tau=0.2):

    a = F.normalize(h_food[anchors], dim=-1)
    p = F.normalize(h_food[positives], dim=-1)
    n = F.normalize(h_food[negatives], dim=-1)

    pos = (a * p).sum(dim=-1, keepdim=True) / tau
    neg = (a.unsqueeze(1) * n).sum(dim=-1) / tau

    logits = torch.cat([pos, neg], dim=1)

    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)