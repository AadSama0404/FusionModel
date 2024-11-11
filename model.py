'''
coding:utf-8
@Software:PyCharm
@Time:2024/7/31 11:45
@Author:tianyi.zhu
'''

import torch
import torch.nn as nn

class Self_Attention(nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()
        self.input = 3
        self.encoder = 64
        self.decoder = 32
        self.K = 16
        self.V = 16
        self.query_layer = nn.Linear(self.decoder, self.K)
        self.key_layer = nn.Linear(self.decoder, self.K)
        self.value_layer = nn.Linear(self.decoder, self.V)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input, self.encoder),
            nn.ReLU(),
            nn.Linear(self.encoder, self.decoder),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.V, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor(x)
        H = H.view(-1, self.decoder)

        Query = self.query_layer(H)
        Key = self.key_layer(H)
        Value = self.value_layer(H)
        scores = torch.mm(Query, Key.transpose(-2, -1)) / (self.K ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        A = torch.mean(attention_weights, dim=0)
        A = A.view(1, -1)

        Z = torch.mm(A, Value)
        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A

    def calculate(self, X, Y, pos_weight = 1):
        Y = Y.float()
        Y_prob, Y_hat, A = self.forward(X)
        # Y_prob: grad_fn=<ClampBackward1>
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (pos_weight * Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return neg_log_likelihood, Y_prob, error, Y_hat, A