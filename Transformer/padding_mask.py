import numpy as np
import torch


def padding_mask(QK, attn_mask=None):
    QK = QK * attn_mask
    return torch.tensor(QK)
