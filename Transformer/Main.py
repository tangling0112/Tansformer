import torch
import Transformer
import torch.nn as nn
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import Transformer
from torch.optim import SGD

import MyDataset.MyDataset as MyDataset
import jieba
embedding_size = 512
path_en = '../TrainData/train.en'
path_zh = '../TrainData/train.zh'
max_size = 3000
max_len = 64
min_freq = 3
trainnig = 'train'
Dataset = MyDataset.MyDataset(path_en,path_zh,max_len,max_size,min_freq,trainnig)
vocab_size_en,vocab_size_zh = Dataset.get_vocab_size()
#embedding_en = nn.Embedding(vocab_size_en, embedding_size, padding_idx=0)
#embedding_zh = nn.Embedding(vocab_size_zh,embedding_size,padding_idx=0)
device = torch.device('cpu')
model = Transformer.Transformer(vocab_size_en,max_len,vocab_size_zh,max_len).to(device)
optimizer = SGD(model.parameters(), lr = 0.01, momentum=0.9)
loss = nn.CrossEntropyLoss().cuda()
model.train()

# 获得输入的词嵌入编码
DataLoader = DataLoader(Dataset, batch_size=1, shuffle=False)

for epoch in range(1,51):
    for idx,(en,zh) in enumerate(DataLoader):
        en = [aa.tolist() for aa in en]
        en = torch.tensor(en)
        en = en.t().to(device)
        zh = [aa.tolist() for aa in zh]
        zh = torch.tensor(zh)
        zh = zh.t().to(device)
        #print(zh.shape)
        output = model(en,torch.tensor(en.shape[1]).to(device),zh,torch.tensor(zh.shape[1]).to(device))
        #print(vocab_size_zh)
        #print(output)
        los = loss(torch.tensor(output[0]),zh)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(los)




