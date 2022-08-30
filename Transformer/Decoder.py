import torch.nn as nn
import DecoderLayer
import PositionEncoding
import padding_mask
import sequence_mask
import torch


class Decoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        """
        :param vocab_size: 输入数据集的词典的大小
        :param max_len: 输入数据集的每一个句子的单词数
        :param num_layers: Encoder小模块的个数
        :param model_dim: Embedding的维度
        :param num_heads: 多头自注意的头数
        :param ffn_dim:FeedForWard的中间状态的词向量维度
        :param dropout: dropout的概率
        """
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer.DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.pos_embedding = PositionEncoding.PositionalEncoding(model_dim, max_len)

    def forward(self, inputs, enc_output, Firstmask, context_attn_mask=None):
        """
        :param inputs: tgt_seq = [Batch,SeqLen]
        :param enc_output: 解码器的输出 [Batch,SeqLen,512]
        :param Firstmask: Mask 矩阵[Batch,SeqLen,SeqLen]
        :param context_attn_mask: None
        :return:[Batch,SeqLen,512]
        """
        # word embedding
        output = self.seq_embedding(inputs)
        # Positional embedding
        output += self.pos_embedding(inputs.shape[1])
        # 获取Padding Mask
        self_attention_padding_mask = padding_mask.padding_mask(inputs, Firstmask)
        # 获取Sentence Mask
        seq_mask = sequence_mask.sequence_mask(inputs)

        # 获得Sentence Mask与Padding Mask的组合Mask
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        for decoder in self.decoder_layers:
            output = decoder(
                output, enc_output, self_attn_mask, context_attn_mask)

        return output
