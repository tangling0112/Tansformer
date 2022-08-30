import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        """
        :param d_model: 一个标量。每个词语的词向量长度，论文默认是512
        :param max_len: 指定一个句子的最大长度，在我们的数据集里为64
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        self.position_encoding = torch.tensor(np.array([
            [pos / torch.pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_len)]))
        # 偶数列使用sin，奇数列使用cos
        self.position_encoding[:, 0::2] = torch.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = torch.cos(self.position_encoding[:, 1::2])

        """
        self.position_encoding = nn.Embedding(max_seq_len, d_model)
        self.position_encoding.weight = nn.Parameter(self.position_encoding,
                                                     requires_grad=False)
        """

    def forward(self, Batch):
        """
        :param Batch: [1]
        :return:[Batch,SeqLen,512]
        """
        out = [self.position_encoding for i in range(Batch)]

        return torch.tensor(out)
