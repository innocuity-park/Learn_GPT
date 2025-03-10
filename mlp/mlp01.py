'''
本篇代码为手动实现线性模型和sigomoid函数
并将二者结合生成一个二分类感知器模型
'''



import torch
import torch.nn.functional as F
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


torch.manual_seed(1024)

#定义线性模型和 sigmoid 函数

class Linear:
    # input: (B, in_features) B是batch size，是数据点，是人为给定的
    # n_features是输入特征数，是根据具体的数据情况确定的。用多少个特征来描述一个数据点
    # output: (B, out_features) out_features是输出的线性函数的个数，是人为给定的，是根据具体的问题确定的
    # 现在这个线性函数是一个全连接层，输入的每个特征都与输出的每个特征相连
    # 并没有开始训练其参数，只是定义了一个线性函数
    def __init__(self, in_features, out_features, bias=True):
        # 初始化权重
        self.weight = torch.randn(in_features, out_features, requires_grad=True)
        # 初始化偏置
        if bias:
            self.bias = torch.randn(out_features, requires_grad=True) if bias else None   #（ ， out_features) 广播机制
        else:
            self.bias = None

    def __call__(self, x):
        # 计算线性函数的输出
        # x : (B, in_features)
        #self.weight: (in_features, out_features)
        self.output = x @ self.weight
        if self.bias is not None:
            self.output += self.bias
        return self.output
    
    def parameters(self):
        # 返回权重和偏置
        if self.bias is not None:
            return [self.weight, self.bias]
        else:
            return [self.weight]

l = Linear(3, 4) # 输入特征数为3，输出的描述y的特征数为4
x = torch.randn(5, 3) # 五个数据点，每个数据点有3个特征

# y = l(x)
# print(l.parameters())

class Sigmoid:
    def __call__(self, x):
        self.out = torch.sigmoid(x)
        return self.out

    def parameters(self):
        return []


class Perception:
    def __init__(self, in_features):
        self.ln = Linear(in_features, 1) #意味着只有一个神经元模型
        self.f = Sigmoid()
        
    def __call__(self, x):
        #一个神经元，单层的神经元模型s
        self.out = self.f(self.ln(x)) #
        return self.out

    def parameters(self):
        return self.ln.parameters()+self.f.parameters()


#实际在使用时，需要使用softmax函数，来将激活函数的输出转换为概率分布
#进而完成逻辑分类的任务
class LogitRegression:
    # 定义逻辑回归模型，现在有两个神经元，解决二分类任务
    # input: (B, in_features)
    # output: (B, 2) 二分类问题，输出的特征数为2
    def __init__(self, in_features):
        self.pos = Linear(in_features, 1)
        self.neg = Linear(in_features, 1)

    def __call__(self, x):
        # x: (B, in_features)
        # torch.concat()函数，将两个张量拼接在一起,(B, 1)和(B, 1)拼接在一起，得到(B, 2)
        self.out = torch.concat((self.pos(x), self.neg(x)), dim=-1) 
        return self.out
    
    def parameters(self):
        return self.pos.parameters() + self.neg.parameters()
    
#test
# lr = LogitRegression(3)
# x = torch.randn(5, 3)
# print(lr(x).shape)
# print(lr.pos(x))
# logits = LogitRegression(x) 

# probs = F.softmax(logits,dim=-1)
# pred = torch.argmax(probs, dim=-1)
# print(pred)

# 定义逻辑回归的损失，在此基础上，使用最优化算法进行梯度更新优化模型参数
# 分类问题的损失是交叉熵



data = make_blobs(200, centers=[[-2, -2], [2, 2]])
# print(data[0])
# print(data[0][:,0])
batch_size = 20 
max_steps = 2000
learning_rate = 0.1

x, y = torch.tensor(data[0]).float(), torch.tensor(data[1])
lr = LogitRegression(2) # 生成二分类模型
lossi = []
# 提取数据
for t in range(max_steps):
    # 随机抽取数据
    ix = (t * batch_size) % len(x)
    xx = x[ix:ix+batch_size]
    yy = y[ix:ix+batch_size]
    logits = lr(xx)
    loss = F.cross_entropy(logits, yy)
    loss.backward()
    with torch.no_grad():
        for p in lr.parameters():
            p -= learning_rate * p.grad
            p.grad = None
    if t % 200 == 0:
        print(f'step: {t}, loss: {loss.item()}')
    lossi.append(loss.item())
plt.plot(lossi)
plt.savefig('loss.png')






    






