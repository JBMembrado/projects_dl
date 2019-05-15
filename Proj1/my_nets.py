import torch
from torch import nn
from torch.nn import functional as F
from model import Net

class NetWithDropout(Net):
    def __init__(self):
        super(NetWithDropout, self).__init__()
        self.dropout = nn.Dropout2d(p=0.15)
        self.nb_epoch = 25

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
            for p in self.parameters(): p.normal_(0, 0.01)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), kernel_size=5, stride=5))
        x = self.batch_norm(x)
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)#pas de batchnorm ici, alors qu'on avait un dropout a cet endroit
        return x

class NetWithWeightSharing(Net):
    def __init__(self):
        super(NetWithWeightSharing, self).__init__()
        self.nb_epoch = 25
        self.sub_net = SubNetForSharing()
        self.fc1 = nn.Linear(20, 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        # we take the slices 0:1 and 1:2 so that the vector is still of dimension 4 and does not go to dimension 3.
        x1 = self.sub_net(x[:, 0:1, :, :])
        x2 = self.sub_net(x[:, 1:2, :, :])
        x = torch.cat((x1, x2), 1)
        x = self.fc1(x)
        return x


class NetWithWeightSharingAndAuxiliaryLoss(Net):

    def __init__(self):
        super(NetWithWeightSharingAndAuxiliaryLoss, self).__init__()
        self.nb_epoch = 25
        self.sub_net = SubNetForSharing()
        #self.fc1 = nn.Linear(20, 20)
        #self.fc2 = nn.Linear(20, 2)
        self.fc1 = nn.Linear(20, 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        self.x1 = None
        self.x2 = None
        self.auxfactor = 0.8

    def forward(self, x):
        # we take the slices 0:1 and 1:2 so that the vector is still of dimension 4 and does not go to dimension 3.
        self.x1 = self.sub_net(x[:, 0:1, :, :])
        self.x2 = self.sub_net(x[:, 1:2, :, :])
        x = torch.cat((self.x1, self.x2), 1)
        x = self.fc1(x)
        return x

    def trainer(self, train_input, train_target, train_classes):

        self.train()  # mode to tell dropout/batchnorm

        for e in range(self.nb_epoch):
            sum_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self(train_input.narrow(0, b, self.mini_batch_size))
                loss1 = self.criterion(self.x1, train_classes[:, 0].narrow(0, b, self.mini_batch_size))
                loss2 = self.criterion(self.x2, train_classes[:, 1].narrow(0, b, self.mini_batch_size))
                losstot = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                loss = losstot + (loss1 + loss2) * self.auxfactor
                self.optimizer.zero_grad()
                loss.backward()
                sum_loss = sum_loss + loss.item()
                self.optimizer.step()
            print("Step %d : %f" % (e, sum_loss))


class SubNetForSharing(Net):
    def __init__(self):
        super(SubNetForSharing, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16,32,kernel_size=2)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 128)))
        x = self.fc2(x)

        return x

class NetWithDropoutWSAL(Net):
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

class SubNetForSharingDropout(Net):
    def __init__(self):
        super(SubNetForSharing, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x


class NetWithBatchNormWSAL(Net):
    def __init__(self):
        super(NetWithBatchNorm, self).__init__()
        self.nb_epoch = 25
        self.batch_norm = nn.BatchNorm2d(64)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        with torch.no_grad():
            for p in self.parameters(): p.normal_(0, 0.01)

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
        self.nb_epoch = 5
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
        input_data_resize = input_data.view(2000, 1, 14, 14)
        number_output = self(input_data_resize)
        number_output = number_output.view(1000, 2, 10)
        predicted_classes = number_output.argmax(2)
        predictions = predicted_classes[:, 0] <= predicted_classes[:, 1]
        target_labels = target
        nb_errors = torch.sum(predictions.type(torch.LongTensor) != target_labels)
        return float(nb_errors) * 100 / input_data.size(0)


class SmallNet(Net):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 10)

        self.mini_batch_size = 100
        self.criterion = nn.CrossEntropyLoss()
        self.nb_epoch = 25
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), kernel_size=5, stride=5))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

    def Smalltrainer(self, train_input, train_classes):
        train_input = train_input.view(2000, 1, 14, 14)
        train_classes = train_classes.view(2000)

        self.trainer(train_input, train_classes)


class BigNet(Net):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 2)

        self.mini_batch_size = 100
        self.criterion = nn.CrossEntropyLoss()
        self.nb_epoch = 25
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), kernel_size=5, stride=5))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

    def BigTrainer(self, train_input, train_target):
        self.trainer(train_input, train_target)

    def nb_errors(self, input_data, target):
        """
        Compute the number of error of the model on a test set
        :param input_data: test features
        :param target: test target
        :return: number of errors
        """
        self.eval()  # mode to tell dropout/batchnorm

        nb_errors = 0
        for b in range(0, input_data.size(0), self.mini_batch_size):
            output = self(input_data.narrow(0, b, self.mini_batch_size))
            predictions = output.argmax(1)
            target_labels = target.narrow(0, b, self.mini_batch_size)
            nb_errors += torch.sum(predictions != target_labels)
        return float(nb_errors) * 100 / input_data.size(0)

