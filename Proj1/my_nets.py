import torch
from torch import nn
from torch.nn import functional as F
from model import Net


class NetWithDropout(Net):
    def __init__(self):
        super(NetWithDropout, self).__init__()
        self.dropout = nn.Dropout2d(p=0.2)
        self.nb_epoch = 100

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), kernel_size=5, stride=5))
        x = self.dropout(x)
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NetWithBatchNorm(Net):
    def __init__(self):
        super(NetWithBatchNorm, self).__init__()
        self.nb_epoch = 25
        self.batch_norm = nn.BatchNorm2d(64)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        with torch.no_grad():
            for p in self.parameters() : p.normal_(0, 0.01)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), kernel_size=5, stride=5))
        x = self.batch_norm(x)
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
