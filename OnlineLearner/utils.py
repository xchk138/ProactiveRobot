from crypt import methods
import numpy as np
import os
import pickle
from fashion_mnist.utils import mnist_reader
import torchvision
from torchvision import transforms
import torch


def _load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("uint8")
        Y = np.array(Y)
        return X, Y


def _load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = _load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = _load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte



def LoadCIFAR10():
    return _load_CIFAR10('cifar10/data/')


def LoadFashionMNIST():
    xtr, ytr = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
    xts, yts = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')
    xtr = xtr.reshape(-1, 28, 28)
    xts = xts.reshape(-1, 28, 28)
    return xtr, ytr, xts, yts

# abstract class
class Dataset(object):
    def __init__(self) -> None:
        pass
    def train_loader(self):
        raise NotImplementedError()
    def test_loader(self):
        raise NotImplementedError()

class CIFAR10(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        # setup training dataset
        self.tr_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010))
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=path, 
            train=True, 
            download=False, 
            transform=self.tr_train)
        # setup testing dataset
        self.tr_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010))
        ])
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=path, 
            train=False, 
            download=False, 
            transform=self.tr_test)
        
    def train_loader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=128, 
            shuffle=True, 
            num_workers=2)
        return train_loader

    def test_loader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=100, 
            shuffle=False, 
            num_workers=2)
        return test_loader