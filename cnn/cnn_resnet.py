'''
本篇主要为实现卷积网络的残差连接
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


torch.manual_seed(1024)


dataset = datasets.MNIST('./dl/cnn/cnn_mnist', train=True, download=True, transform=transforms.ToTensor())

# print(dataset[0]) #格式是一个tuple，第一个元素是图片，第二个元素是标签 索引不是简单的按顺序排列的
# print(dataset[21][0].shape) #图片是一个三维张量，第一个维度是通道数，第二三个维度是图片的高和宽

x, y = dataset[21]

#plt.imshow(x.squeeze(0).numpy(), cmap='gray') 注意这里numpy一定要加括号，否则会报错
train_set, val_set = random_split(dataset, [50000, 10000])
test_set = datasets.MNIST('./dl/cnn/cnn_mnist', train=False, download=True, transform=transforms.ToTensor())

#使用data loader 加载数据 进行封装
#shuffle=True表示每个epoch都会打乱数据的顺序，取batch_size时是一个一个取的，所以每次取的数据都是不一样的
train_loader = DataLoader(train_set, batch_size=500, shuffle=True)
val_loader = DataLoader(val_set, batch_size=500, shuffle=True)
test_loader = DataLoader(test_set, batch_size=500, shuffle=True)

eval_iters = 10

def estimate_loss(model):
    pass

def _loss(model, dataloader):
    pass

def train_model(model, optimizer, epochs=10, penalty=False):
    pass


#卷积层输入输出一样



