import torch.nn as nn
import MultiHeadSelfAttention
import PositionalWiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadSelfAttention.MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward.PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
                dec_inputs,
                enc_outputs,
                self_attn_mask=None,
                context_attn_mask=None):
        """

        :param dec_inputs: Decoder的输入 [Batch,SeqLen,512]
        :param enc_outputs: Encoder的最终输出 [Batch,SeqLen,512]
        :param self_attn_mask: 第一个Mask MultiHeadAttention所需的Mask矩阵
        :param context_attn_mask: None
        :return: [Batch,SeqLen,512]
        """
        # Mask MultiHeadAttention
        dec_output = self.attention(
            dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # MultiHeadAttention
        dec_output = self.attention(
            enc_outputs, enc_outputs, dec_output, context_attn_mask)
        # FeedForWard
        dec_output = self.feed_forward(dec_output)

        return dec_output  # , self_attention, context_attention
