from utils import CIFAR10, LoadCIFAR10
from utils import LoadFashionMNIST
import cv2
import matplotlib.pyplot as plt

# models
from models import ResNet18


def main():
    xtr, ytr, xts, yts = LoadCIFAR10()
    print('training samples in cifar-10: %d' % len(xtr))
    print('testing samples in cifar-10: %d' % len(xts))
    # show a demo from CIFAR-10 dataset
    print(xtr[0].shape)
    print(xtr[0].dtype)
    print(ytr[0].dtype)
    cv2.imshow('CIFAR-10 demo', xtr[0])
    cv2.waitKey(0)
    # load fashion-mnist dataset
    xtr, ytr, xts, yts = LoadFashionMNIST()
    print('training samples in fashion-mnist: %d' % len(xtr))
    print('testing samples in fashion-mnist: %d' % len(xts))
    # show a demo from Fashion-MNIST dataset
    print(xtr[0].shape)
    print(xtr[0].dtype)
    print(ytr[0].dtype)
    cv2.imshow('Fashion-MNIST demo', xtr[0])
    cv2.waitKey(0)

    net = ResNet18(10, 'models/resnet18-cifar10/', 'logs/resnet18-cifar10/')
    print(net)
    # get custom cifar-10 dataset loader
    cifar10 = CIFAR10('cifar10/cifar-10-python')
    net.train_online(xtr, ytr)
    #net.test_online(xts, yts)




if __name__ == '__main__':
    print('================ ONLINE LEARNER TEST BEGIN ==============')
    main()
    print('================ ONLINE LEARNER TEST DONE ===============')
