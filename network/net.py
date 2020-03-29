import torch.nn as nn


class LogisticRegression(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
