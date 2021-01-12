import torch
import torch.nn as nn

from zoo.common import * 
# class Flatten(nn.Module):
    # def forward(self, input):
        # return input.view(input.size(0), -1)

class baselineModel(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.num_classes = 10

        # (b, 3, 32, 32)
        self.add_module("Conv2dx1", nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1))
        self.add_module("BatchNorm2dx1", nn.BatchNorm2d(32))
        self.add_module("ReLUx1", nn.ReLU())
        self.add_module("MaxPool2dx1", nn.MaxPool2d(kernel_size=2, stride=2))

        # (b, 32. 16, 16)
        self.add_module("Conv2dx2", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        self.add_module("BatchNorm2dx2", nn.BatchNorm2d(64))
        self.add_module("ReLUx2", nn.ReLU())
        self.add_module("MaxPool2dx2", nn.MaxPool2d(kernel_size=2, stride=2))

        self.add_module("Conv2dx3", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.add_module("BatchNorm2dx3", nn.BatchNorm2d(128))
        self.add_module("ReLUx3", nn.ReLU())
        self.add_module("MaxPool2dx3", nn.MaxPool2d(kernel_size=2, stride=2))

        self.add_module("Conv2dx4", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.add_module("BatchNorm2dx4", nn.BatchNorm2d(256))
        self.add_module("ReLUx4", nn.ReLU())
        self.add_module("MaxPool2dx4", nn.MaxPool2d(kernel_size=2, stride=2))

        self.add_module("Flatten", Flatten())
        self.add_module("Dropout", nn.Dropout(p=0.5))
        self.add_module("Linear", nn.Linear(2*2*256, self.num_classes))
    
    def forward(self, V):
        return super().forward(V).squeeze()

class leakyModel(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.num_classes = 10

        self.add_module("Conv2dx1", nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1))
        self.add_module("BatchNorm2dx1", nn.BatchNorm2d(32))
        self.add_module("LeakyReLUx1", nn.LeakyReLU())
        self.add_module("MaxPool2dx1", nn.MaxPool2d(kernel_size=2, stride=2))

        self.add_module("Conv2dx2", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        self.add_module("BatchNorm2dx2", nn.BatchNorm2d(64))
        self.add_module("LeakyReLUx2", nn.LeakyReLU())
        self.add_module("MaxPool2dx2", nn.MaxPool2d(kernel_size=2, stride=2))

        self.add_module("Conv2dx3", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.add_module("BatchNorm2dx3", nn.BatchNorm2d(128))
        self.add_module("LeakyReLUx3", nn.LeakyReLU())
        self.add_module("MaxPool2dx3", nn.MaxPool2d(kernel_size=2, stride=2))

        self.add_module("Conv2dx4", nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.add_module("BatchNorm2dx4", nn.BatchNorm2d(256))
        self.add_module("LeakyReLUx4", nn.LeakyReLU())
        self.add_module("MaxPool2dx4", nn.MaxPool2d(kernel_size=2, stride=2))

        self.add_module("Flatten", Flatten())
        self.add_module("Dropout", nn.Dropout(p=0.3))
        self.add_module("Linear", nn.Linear(2*2*256, self.num_classes))
    
    def forward(self, V):
        return super().forward(V).squeeze()