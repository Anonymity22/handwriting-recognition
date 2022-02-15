import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from myresnet import ResNet18
from torchvision import models
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, encoder_hidden = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False,pt_res=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'


        self.resnet = ResNet18(pt_res)
        #self.resnet.conv1=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def attention_net(self, x, query, mask=None):      #软性注意力机制（key=value=x）

        d_k = query.size(-1)                                              #d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  #打分机制  scores:[batch, seq_len, seq_len]

        p_attn = F.softmax(scores, dim = -1)                              #对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)       #对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn

    def forward(self, input):
        conv=self.resnet(input)
        # print(conv.shape)
        # b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)   #因为h=1，所以摊平无影响？
        conv = conv.permute(2, 0, 1)  # [w, b, c]  换位

        # rnn features
        output = self.rnn(conv)

        return output
