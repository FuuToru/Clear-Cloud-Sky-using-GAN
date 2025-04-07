import torch.nn as nn

bce_loss = nn.BCELoss()

def gan_loss(pred, target_is_real):
    targets = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return bce_loss(pred, targets)