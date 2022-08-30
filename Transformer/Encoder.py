import torch
import torch.nn as nn
import EncoderLayer
import PositionEncoding


class Encoder(nn.Module):
    # 多层EncoderLayer组成Encoder。
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        """
        :param vocab_size: 输入数据集的词典的大小
        :param max_seq_len: 输入数据集的每一个句子的单词数
        :param num_layers: Encoder小模块的个数
        :param model_dim: Embedding的维度
        :param num_heads: 多头自注意的头数
        :param ffn_dim:？？？
        :param dropout: dropout的概率
        """
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer.EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        # word embedding
        self.seq_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        # positional embedding
        self.pos_embedding = PositionEncoding.PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, Batch):
        count = 1
        mask = torch.where(inputs != 0, 1, 0).unsqueeze(1).expand(-1, inputs.shape[1], -1)
        # 获取word embedding [Batch,SeqLen,512]
        output = self.seq_embedding(inputs)
        # 获取positional embedding [Batch,SeqLen,512]
        output += self.pos_embedding(Batch)
        for encoder in self.encoder_layers:
            if count == 1:
                output = encoder(output, mask)
        return output
