import torch.nn.functional as F
import torch
from torch import dot as dot
from torch.nn import Parameter
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import math
class NaiveLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(NaiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #注意每个权重都是具备三个维度的，第一个维度用于表示Batch
        # 输入门的权重矩阵和bias矩阵
        self.w_ii = Parameter(Tensor(1,input_size,hidden_size))
        self.w_hi = Parameter(Tensor(1,hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(1,1,hidden_size))
        self.b_hi = Parameter(Tensor(1,1,hidden_size))
        # 遗忘门的权重矩阵和bias矩阵
        self.w_if = Parameter(Tensor(1,input_size,hidden_size))
        self.w_hf = Parameter(Tensor(1,hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(1,hidden_size))
        self.b_hf = Parameter(Tensor(1,hidden_size))
        # 输出门的权重矩阵和bias矩阵
        self.w_io = Parameter(Tensor(1,input_size,hidden_size))
        self.w_ho = Parameter(Tensor(1,hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(1,1,hidden_size))
        self.b_ho = Parameter(Tensor(1,1,hidden_size))
        # cell的的权重矩阵和bias矩阵
        self.w_ig = Parameter(Tensor(1,input_size,hidden_size))
        self.w_hg = Parameter(Tensor(1,hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(1,1,hidden_size))
        self.b_hg = Parameter(Tensor(1,1,hidden_size))

        #self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, state):
        if state is None:
            h_t = torch.zeros(1, self.hidden_size)
            c_t = torch.zeros(1, self.hidden_size)
        else:
            (h, c) = state
            h_t = h.squeeze(0)
            c_t = c
        #用于存储每一个隐藏状态
        hidden_seq = []
        #获取SeqLen，如一个句子有10个词，那么seq_size=10
        seq_size = inputs.shape[1]
        #按照词进行迭代
        for t in range(seq_size):
            #迭代第t个词
            x = inputs[:, t, :].unsqueeze(1)
            # 输入门
            #[batch,1,wordvector]@[1,wordvector,hiddensize]+
            #[1,1,hiddensize]+
            #[1,1,hiddensize]@[batch,hiddensize,hiddensize]+
            #[1,1,hiddensize]
            #=[batch,1,hiddensize]
            i = torch.sigmoid(x@self.w_ii + self.b_ii + h_t@self.w_hi  + self.b_hi)
            # 遗忘门
            f = torch.sigmoid(x@self.w_if + self.b_if + h_t@self.w_hf + self.b_hf)
            # 记忆细胞
            g = torch.tanh(x@self.w_ig + self.b_ig + h_t@self.w_hg + self.b_hg)
            # 输出门
            o = torch.sigmoid(x@self.w_io + self.b_io + h_t@self.w_ho + self.b_ho)
            #记忆细胞更新
            #[batch,1,hiddensize]*[batch,1,hiddensize]+
            #[batch,1,hiddensize]*[batch,1,hiddensize]+
            #=[batch,1,hiddensize]
            c_next = f * c_t + i * g
            #隐藏状态更新
            #[batch,1,hiddensize]*[batch,1,hiddensize]==>[batch,1,hiddensize]
            h_next = o * torch.tanh(c_next)
            #记忆细胞与隐藏细胞维度填充
            c_next_t = c_next
            h_next_t = h_next
            hidden_seq.append(h_next_t)
            #print(len(hidden_seq))
        #存储下所有隐藏状态，在NLP中则相当于存储每一个词的隐藏状态
        #stack([batch,1,wordvector])=>[seqlen,batch,1,wordvector]
        hidden_seq = torch.stack(hidden_seq, dim=0)
        #return [seqlen,batch,1,wordvector],([batch,1,hiddensize],[batch,1,hiddensize])
        return hidden_seq, (h_next_t, c_next_t)
#模型参数初始化
def reset_weigths(model):
    for weight in model.parameters():
        init.constant_(weight,0.4)
#参数的三个维度为[BatchSize,SeqLen,WordVectorLen]
inputs = torch.ones(24, 10, 10)
h0 = torch.ones(24, 1, 30)
c0 = torch.ones(24, 1, 30)
#实例化一个inputsize=10，hiddensize=20的LSTM类
model = NaiveLSTM(10, 30)
#初始化模型权重
#reset_weigths(naive_lstm)
#output1, (hn1, cn1) = naive_lstm(inputs, (h0, c0))
#print(hn1.shape, cn1.shape, output1.shape)
#print(output1)
device=torch.device('cuda')
model = NaiveLSTM(10, 30).to(device)
reset_weigths(model)
LossFunction=nn.MSELoss(model.parameters()).to(device)
print(model.parameters().__next__())

