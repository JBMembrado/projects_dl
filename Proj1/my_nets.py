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


class NetNumber(Net):
    def __init__(self):
        super(NetNumber, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 10)
        self.nb_epoch = 25
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

    def trainer_nb(self, train_input, train_classes):
        """
        """
        train_input = train_input.view(2000, 1, 14, 14)
        train_classes = train_classes.view(2000)

        self.trainer(train_input, train_classes)

    # Test error
    def nb_errors_nb(self, input_data, target):
        """
        Compute the number of error of the model on a test set
        :param input_data: test features
        :param target: test target
        :return: number of errors
        """
        input_data = input_data.view(2000, 1, 14, 14)
        number_output = self(input_data)
        number_output = number_output.view(1000, 2, 10)
        _, predicted_classes = number_output.data.max(2)
        predictions = predicted_classes[:, 0] <= predicted_classes[:, 1]
        target_labels = target.byte()
        nb_errors = torch.sum(predictions != target_labels)
        return float(nb_errors) * 100 / input_data.size(0)
