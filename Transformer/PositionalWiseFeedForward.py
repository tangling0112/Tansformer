import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        """
        :param model_dim:512
        :param ffn_dim: 2048
        :param dropout: Dropout的概率
        """
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        """
        :param x: [Batch,SeqLen,512]
        :return: [Batch,SeqLen,512]
        """
        output = x.transpose(1, 2)#[Batch,512,SeqLen]
        output = self.w2(F.relu(self.w1(output)))#w1==>[Batch,2048,SeqLen],w2==>[Batch,512,SeqLen]
        output = self.dropout(output.transpose(1, 2))#[Batch,SeqLen,512]

        # ADD & Norm
        output = self.layer_norm(x + output)
        return output
