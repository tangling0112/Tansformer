import torch.nn as nn
import torch
import Encoder
import Decoder


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 src_max_len,
                 tgt_vocab_size,
                 tgt_max_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2):
        """
        :param src_vocab_size: 输入数据集的词典包含的词语的个数
        :param src_max_len: 输入数据集单个句子的最大长度
        :param tgt_vocab_size: 输出数据集的词典包含的词语的个数
        :param tgt_max_len: 输出数据集的词典包含的单个句子的最大长度
        :param num_layers: 定义使用多少层编码器解码器叠加，在Transformer的论文中使用了6层
        :param model_dim: 指定单个词的词向量的长度
        :param num_heads: MultiHeadAttention的头数
        :param ffn_dim:FeedForward的中间状态的词向量维度
        :param dropout: Dropout机制的发生概率
        """

        super(Transformer, self).__init__()

        self.encoder = Encoder.Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                                       num_heads, ffn_dim, dropout)
        self.decoder = Decoder.Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                                       num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, tgt_seq):
        """
        :param src_seq: [Batch,SeqLen]
        :param tgt_seq: [Batch,SeqLen]
        :return: [Batch,SeqLen,Vocabsize]
        """
        Batch = src_seq.shape[0]
        tgt_mask = torch.where(tgt_seq != 0, 1, 0).unsqueeze(1).expand(-1, tgt_seq.shape[1], -1)
        # 对目标序列[Batch,Target_seq]做Padding Mask
        # output = [Batch,SeqLen,512]
        output = self.encoder(src_seq, Batch)

        output = self.decoder(
            tgt_seq, output, tgt_mask)

        output = self.linear(output)
        output = self.softmax(output)

        return output
