'''
本篇内容将完成卷机神经网络的代码
主要内容包括：
1.卷积层的设计
# 本地感受野（Local receptive field）： 卷积核的大小
# 权值共享（Weight sharing）： 减少参数方便训练//特征提取器
# 多通道（Multi-channel）
# 步长（Stride）
# 填充（Padding）
# 池化（Pooling）
# 激活函数（Activation function）

2. 对同一个输出通道，不同的输入通道的kernal的参数不同
对同一个输入通道，不同的输出通道也不同，因为提取不同的特征

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

class CNN(nn.Module):
    '''
    构建一个普通的CNN模型
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, (5, 5))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, (5, 5))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear( 40*4*4, 120) # 此处的40*4*4是根据卷积核的大小和步长计算出来的
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # x : (B, 1, 28, 28) 数据集是黑白，只有一个通道
        B = x.shape[0]               # (B,  1, 28, 28)
        x = F.relu(self.conv1(x))    # (B, 20, 24, 24)
        x = self.pool1(x)            # (B, 20, 12, 12)
        x = F.relu(self.conv2(x))    # (B, 40,  8,  8)
        x = self.pool2(x)            # (B, 40,  4,  4)
        x = F.relu(self.fc1(x.view(B, -1))) # (B, 120)
        x = self.fc2(x)                     # (B,  10)
        return x
    
    def parameters(self):
        pass

class CNN2(nn.Module):
    '''
    构建一个CNN模型
    加入层归一化和随机失活函数
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, (5, 5))
        self.ln1 = nn.LayerNorm([20, 24, 24])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, (5, 5))
        self.ln2 = nn.LayerNorm([40, 8, 8])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear( 40*4*4, 120)
        self.dp = nn.Dropout(0.2)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # x : (B, 1, 28, 28) 数据集是黑白，只有一个通道
        B = x.shape[0]                         # (B,  1, 28, 28)
        x = F.relu(self.ln1(self.conv1(x)))    # (B, 20, 24, 24)
        x = self.pool1(x)                      # (B, 20, 12, 12)
        x = F.relu(self.ln2(self.conv2(x)))    # (B, 40,  8,  8)
        x = self.pool2(x)                      # (B, 40,  4,  4)
        x = F.relu(self.fc1(x.view(B, -1)))    # (B, 120)
        x = self.dp(x)
        x = self.fc2(x)                         # (B,  10)
        return 
    
    def parameters(self):
        pass


eval_iters = 10

def estimate_loss(model):
    pass

def _loss(model, dataloader):
    pass

def train_model(model, optimizer, epochs=10, penalty=False):
    pass















