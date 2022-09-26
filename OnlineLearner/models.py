import torch
import json
from torch.utils.tensorboard.summary import image
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

def FormatJson(dict_):
    return json.dumps(dict_, sort_keys=True, indent=4, separators=(',', ': '))


def GetModelInfo(params):
    _dict = {}
    for name, param in params:
        _dict[name] = list(param.shape)
    return _dict


class ResNet18(object):
    def __init__(self, num_classes, save_path, log_path):
        # create specific path if they dont exist
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        self.config = {}
        self.config['save_path'] = save_path
        print('model will be saved into %s' % self.config['save_path'])
        self.net = torchvision.models.resnet18(num_classes=num_classes)
        self.config['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.config['model'] = self.net.__str__()
        self.device = torch.device(self.config['device'])
        self.net = self.net.to(self.device)
        self.config['learning_rate'] = 1e-2
        self.config['weights'] = GetModelInfo(self.net.named_parameters())
        self.config['momentum'] = 0.9
        self.config['log_path'] = log_path
        self.sw = SummaryWriter(log_path)
        # load the model and reset the learning rate
        files = os.listdir(save_path)
        model_files = [os.path.join(save_path, f) for f in files if f.endswith('.pth')]
        print(model_files)
        if len(model_files) >0:
            model_files.sort() # in place sort
            self.net = torch.load(model_files[-1])
            print('pretrained model %s is loaded!' % model_files[-1])

    def train(self, train_loader, num_epoc=300, test_loader=None):
        SAVE_FREQ = 1
        PRINT_FREQ = 100
        _opt = optim.SGD(
            self.net.parameters(), 
            lr = self.config['learning_rate'], 
            momentum=self.config['momentum'])
        _loss = torch.nn.CrossEntropyLoss()
        for _epoch in range(num_epoc):
            training_loss = 0.0
            for _step, input_data in enumerate(train_loader):
                #continue
                image, label = input_data[0].to(self.device), input_data[1].to(self.device)
                predict_label = self.net.forward(image)
                loss = _loss(predict_label, label)
                self.sw.add_scalar('training loss', loss, global_step = _epoch*len(train_loader) + _step)
                _opt.zero_grad()
                loss.backward()
                _opt.step()
                training_loss = training_loss + loss.item()  
                if _step % PRINT_FREQ == PRINT_FREQ-1:
                    print('[iteration - %03d] training loss: %.3f' % (_epoch*len(train_loader) + _step, training_loss/100))
                    training_loss = 0.0
            if _epoch % SAVE_FREQ == SAVE_FREQ-1:
                torch.save(self.net, self.config['save_path'] + '/%03d.pth' % _epoch)
            if test_loader is not None:
                self.test(test_loader, _epoch)
            print('epoc#%03d done.' % _epoch)
        torch.save(self.net, self.config['save_path'] + '/last.pth')

    def train_online(self, train_x, train_y):
        # randomly select N initial solution(parameter-setting) to start with
        # for each solution S0, we train with single instance till converged
        SPLITS = 1
        for i in range(SPLITS):
            # randomly initialize the parameters
            params = self.net.named_parameters()
            for _name, _val in params:
                print(_name)
                _val.set

    def test_online(self, test_x, test_y):
        self.net.eval()
        correct = 0
        total = 0   
        with torch.no_grad():
            for i in range(len(test_x)):
                images, labels = test_x[i], test_y[i]
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
            print('Testing Accuracy : %.3f %%' % ( 100 * correct / total))

    def test(self, test_loader, _epoch):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images,labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
            print('Testing Accuracy : %.3f %%' % ( 100 * correct / total))
            self.sw.add_scalar('test_Accuracy', 100 * correct / total,  global_step=_epoch)

    def __str__(self):
        return 'ResNet-18: \n' + FormatJson(self.config)

