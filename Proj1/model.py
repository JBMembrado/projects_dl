import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(2, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 2)

        self.mini_batch_size = 100
        self.criterion = nn.CrossEntropyLoss()
        self.nb_epoch = 25
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), kernel_size=5, stride=5))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
        # Training Function
    def trainer(self, train_input, train_target):
        """
        Train the model on a training set
        :param train_input: Training features
        :param train_target: Training labels
        """
        for e in range(self.nb_epoch):
            sum_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                self.optimizer.zero_grad()
                loss.backward()
                sum_loss = sum_loss + loss.item()
                self.optimizer.step()
            print("Step %d : %f" % (e, sum_loss))
            
        # Test error
    def nb_errors(self, input_data, target):
        """
        Compute the number of error of the model on a test set
        :param input_data: test features
        :param target: test target
        :return: number of errors
        """
        nb_errors = 0
        for b in range(0, input_data.size(0), self.mini_batch_size):
            output = self(input_data.narrow(0, b, self.mini_batch_size))
            predictions = output.argmax(1)
            target_labels = target.narrow(0, b, self.mini_batch_size)
            nb_errors += torch.sum(predictions != target_labels)
        return float(nb_errors) * 100 / input_data.size(0)

class Net_number(nn.Module):
    def __init__(self):
        super(Net_number, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 10)
        self.mini_batch_size = 100
        self.criterion = nn.CrossEntropyLoss()
        self.nb_epoch = 25
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
    def trainer_number(self, train_input, train_classes):
        """
        """
        train_input=train_input.view(2000,1,14,14)
        train_target=torch.empty(2000,10)
        for i in range(2000):
            for j in range(10):
                if j == train_classes.view(2000,1)[i,0]:
                    train_target[i,j]=1
                else:
                    train_target[i,j]=0
        #c'est moche mais je crois que ca marche. à changer.
        print(train_target)
#        for e in range(self.nb_epoch):
#            sum_loss = 0
#            for b in range(0, train_input.size(0), self.mini_batch_size):
#                output = self(train_input.narrow(0, b, self.mini_batch_size))
#                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
#                self.optimizer.zero_grad()
#                loss.backward()
#                sum_loss = sum_loss + loss.item()
#                self.optimizer.step()
#            print("Step %d : %f" % (e, sum_loss))

    # Test error
    def nb_errors_number(self, input_data, target):
        """
        Compute the number of error of the model on a test set
        :param input_data: test features
        :param target: test target
        :return: number of errors
        """
        nb_errors = 0
        input_data=input_data.view(2000,1,14,14)
        print(input_data.size())
        for b in range(0, input_data.size(0), self.mini_batch_size):
            output = self(input_data.narrow(0, b, self.mini_batch_size))
            _, predicted_classes = output.data.max(1)
            print(predicted_classes)
            print(predicted_classes.size())
#            for k in range(self.mini_batch_size):
#                nb_errors += torch.sum()
#                    
#                    
#        nb_errors = 0
#        for b in range(0, input_data.size(0), self.mini_batch_size):
#            output = self(input_data.narrow(0, b, self.mini_batch_size))
#            predictions = 0
#            target_labels = target.narrow(0, b, self.mini_batch_size)
#            nb_errors += torch.sum(predictions != target_labels)
        return float(nb_errors) * 100 / input_data.size(0)