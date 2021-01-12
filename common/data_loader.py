import torch
import torchvision
import torchvision.transforms as transforms

class DataLoader():
    def __init__(self):
        self.dataset_type = "CIFAR10"
        self.dataset = torchvision.datasets.__dict__[self.dataset_type]
        self.transform = None
    
    def set_transform(self, transform):
        self.transform = transform


    def set_dataset(self, dataset:str):
        self.dataset_type = dataset
        self.dataset = torchvision.datasets.__dict__[self.dataset_type]
    
    def __get_train_dataset(self):
        return self.dataset(root='data/',
                            train=True,
                            transform=self.transform,
                            download=True)
    
    def __get_test_dataset(self):
        return self.dataset(root='data/',
                            train=False,
                            transform=transforms.ToTensor())
    
    def get_train_data(self, batch_size):
        return torch.utils.data.DataLoader(dataset=self.__get_train_dataset(),
                                           batch_size=batch_size, 
                                           shuffle=True)
    
    def get_test_data(self, batch_size):
        return torch.utils.data.DataLoader(dataset=self.__get_test_dataset(),
                                           batch_size=batch_size, 
                                           shuffle=False)
