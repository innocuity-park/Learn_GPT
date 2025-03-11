'''
本篇内容为使用多层感知器实现图像识别，具体的内容有如下几部分
1.加载数据
2.定义模型
3.定义损失函数
4.定义优化器
5.训练模型
6.评估模型

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(12046)

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


#next()函数返回迭代器的下一个项目, iter()函数用来生成迭代器, next(iter(train_loader))返回的是一个batch的数据
x, y = next(iter(train_loader)) 
#为了将x.shape从[500, 1, 28, 28]转换成[500, 784]，使用view函数
x = x.view(x.shape[0], -1) #view函数的作用是将一个多行的Tensor拼接成一行

#定义模型有两种方式，第一种是继承nn.Module类，第二种是使用nn.Sequential
#第一种方式可以定义更加复杂的模型，第二种方式只能定义简单的模型
#第一种方式代码量更多，但是更加灵活，第二种方式代码量少，但是不够灵活
'''
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
'''
#第二种方式
#代码量少，但是不够灵活
# 注意这里是继承的nn.sequential 类，所以只用小括号即可
# 在MLP.py代码中，是自定义的Sequential类，所以小括号里套入[]类。
# model = nn.Sequential(
#     nn.Linear(784, 30), nn.Sigmoid(),
#     nn.Linear( 30, 20), nn.Sigmoid(),
#     nn.Linear( 20, 10)
# )
  
#如何评估模型，借助评估的过程，验证模型建立的代码是没有问题的
#评估模型的过程 是使用若干个批次的数据，然后计算模型的输出和标签之间的差距
#并不需要计算梯度，之间的依赖关系
#定义超参数，使用多少个批次的数据来评估模型的效果
eval_iters = 10


#类在定义时，前有单下划线，表示这个方法是私有的，不应该被外部调用，但是可以被子类调用，
#from module import * 时，不会导入以单下划线开头的方法
#双下划线开头的方法，表示这个方法是私有的，不应该被外部调用，也不应该被子类调用
@torch.no_grad()
def _loss(model, dataloader):
    #估计模型效果
    loss = []
    acc = []
    data_iter = iter(dataloader) # #将dataloader转换成迭代器是怎么做的？
    for t in range(eval_iters):
        inputs, labels = next(data_iter)
        # inputs: [500, 1, 28, 28]
        # outputs: [500]
        # B: batch size, C: channel, H: height, W: width
        B, C, H, W = inputs.shape 
        #inputs.view(B, -1)的作用是将inputs从[500, 1, 28, 28]转换成[500, 784]
        logits = model(inputs.view(B, -1)) #logits是模型的输出
        #logits.shape是[500, 10]，labels.shape是[500]
        loss.append(F.cross_entropy(logits, labels))
        #prds = torch.argmax(F.softmax(logits, dim=-1),dim=-1) 
        '''prds = torch.argmax(logits, dim=-1)和prds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        计算的输出结果是一样的。原因如下：
        1.logits是模型的输出，通常是未经过归一化的分数。
        2.F.softmax(logits, dim=-1)将logits转换为概率分布，但不会改变每个元素的相对大小。
        torch.argmax函数会返回指定维度上最大值的索引。由于 `softmax` 函数只是对logits进行指数运算和归一化处理，
        并不会改变最大值的位置，因此在 logits上直接使用 torch.argmax和在 `softmax(logits)` 上使用 torch.argmax
        得到的结果是相同的。简而言之，`softmax` 不会改变最大值的位置，所以两者的输出结果是一样的。
        '''
        prds = torch.argmax(logits, dim=-1)
        #prds == labels是一个bool张量，生成一个tensor
        #sum()函数的作用是将bool张量中的True转换成1，False转换成0，然后求和
        acc.append((prds == labels).sum() / B)
    re = {
        #.item()函数的作用是将张量中的值取出来,转换成python的标量
        'loss': torch.tensor(loss).mean().item(),
        'acc': torch.tensor(acc).mean().item()
    }
    return re


def estimate_loss(model):
    re = {}
    #将模型切换为评估模式
    #在评估模式下，有些函数会不可用，比如随机失活
    model.eval()
    re['train'] = _loss(model, train_loader)
    re['val'] = _loss(model, val_loader)
    re['test'] = _loss(model, test_loader)
    #评估完成后，需要切换回训练模式
    model.train()
    return re


#print(estimate_loss(model))

#定义训练模型的函数
def train_model(model, optimizer, epochs=10):
    lossi = []
    #epochs 表示训练模型的次数, 每次使用全部数据训练一次为一轮
    for e in range(epochs):
        for data in train_loader:
            inputs, labels = data
            B, C, H, W = inputs.shape
            logits = model(inputs.view(B, -1))
            loss = F.cross_entropy(logits, labels)
            lossi.append(loss.item())
            optimizer.zero_grad() #梯度清零
            loss.backward() #反向传播，计算梯度
            optimizer.step()#获得梯度后，使用优化器更新模型参数
        states = estimate_loss(model)
        train_loss = f"{states['train']['loss']:.3f}"
        #train_loss = f'{states['train']['loss']:.3f}'
        #f-string 要使用双引号，才能用[]，否则会报错，因为f-string中的{}是用来表示变量的
        val_loss = f"{states['val']['loss']:.3f}"
        test_loss = f"{states['test']['loss']:.3f}"        
        print(f"epoch {e} train {train_loss} val {val_loss} test {test_loss}")
    return lossi

loss = {}
#定义模型
#sequential 使用小括号建立，而不用花括号

model_sigmoid = nn.Sequential(
    nn.Linear(784, 30), nn.Sigmoid(),
    nn.Linear( 30, 20), nn.Sigmoid(),
    nn.Linear( 20, 10)
)

model_relu = nn.Sequential(
    nn.Linear(784, 30), nn.ReLU(),
    nn.Linear( 30, 20), nn.ReLU(),
    nn.Linear( 20, 10)
)
# 加入层归一化层
model = nn.Sequential(
    nn.Linear(784, 30, bias=False), nn.LayerNorm(30),nn.ReLU(),
    nn.Linear( 30, 20, bias=False), nn.LayerNorm(20), nn.ReLU(),
    nn.Linear( 20, 10)
)

#定义优化器 并进行训练
loss['mlp_sigmoid'] = train_model(model_sigmoid, optim.SGD(model_sigmoid.parameters(), lr = 0.01))
loss['mlp_relu'] = train_model(model_relu, optim.SGD(model_relu.parameters(), lr = 0.01))
loss['mlp_layer_norm'] = train_model(model, optim.SGD(model.parameters(), lr = 0.01))

# for i in ['mlp_sigmoid', 'mlp_relu', 'mlp_layer_norm']:
#     plt.plot(torch.tensor(loss[i]).view(-1, 10).mean(dim=-1), label=i)
# plt.legend()
# plt.savefig('./dl/cnn/cnn_mlp_loss.png')


#使用随机失活函数，对被选中的点置零，对没被选中的点乘失活概率的倒数
#nn.Dropout函数只在训练模式下起作用，在eval模式下不起作 

 
model = nn.Sequential(
    nn.Linear(784, 30, bias=False), nn.LayerNorm(30),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear( 30, 20, bias=False), nn.LayerNorm(20), nn.ReLU(),nn.Dropout(0.2),
    nn.Linear( 20, 10)
)

loss['mlp_relu_layer_dropout'] = train_model(model, optim.SGD(model.parameters(), lr = 0.01))

for i in ['mlp_sigmoid', 'mlp_relu', 'mlp_layer_norm', 'mlp_relu_layer_dropout']:
    plt.plot(torch.tensor(loss[i]).view(-1, 10).mean(dim=-1), label=i)
plt.legend()
plt.savefig('./dl/cnn/cnn_mlp_loss_dropout.png')


def train_model_penalty(model, optimizer, epochs=10, penalty=False):
    '''带有惩罚项的train 函数'''
    lossi = []
    #epochs 表示训练模型的次数, 每次使用全部数据训练一次为一轮
    for e in range(epochs):
        for data in train_loader:
            inputs, labels = data
            B, C, H, W = inputs.shape
            logits = model(inputs.view(B, -1))
            loss = F.cross_entropy(logits, labels)
            lossi.append(loss.item())
            if penalty:
                w = torch.cat([p.view(-1) for p in model.parameters()])
                loss += 0.001 * w.abs().sum() + 0.002 * w.square().sum()
            optimizer.zero_grad() #梯度清零
            loss.backward() #反向传播，计算梯度
            optimizer.step()#获得梯度后，使用优化器更新模型参数
        states = estimate_loss(model)
        train_loss = f"{states['train']['loss']:.3f}"
        #train_loss = f'{states['train']['loss']:.3f}'
        #f-string 要使用双引号，才能用[]，否则会报错，因为f-string中的{}是用来表示变量的
        val_loss = f"{states['val']['loss']:.3f}"
        test_loss = f"{states['test']['loss']:.3f}"        
        print(f"epoch {e} train {train_loss} val {val_loss} test {test_loss}")
    return lossi



