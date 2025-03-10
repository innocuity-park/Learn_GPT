'''
本篇内容为搭建多层感知器模型
第一层为输入数据的特征
中间为隐藏层，线性加总和激活函数，列数为隐藏层的神经元个数
[B, in_features] @ [in_features, hidden_features] = [B, hidden_features]
最后为输出层
'''
import torch 
import torch.nn.functional as F
from sklearn.datasets import make_moons #生成月牙形数据
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1024)

# 手写线性模型和sigmoid 函数
class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = torch.randn((in_features, out_features), requires_grad=True)
        if bias:
            self.bias = torch.randn(out_features, requires_grad=True)
        else:
            self.bias = None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        # 返回线性模型的参数
        #由于pytorch的计算单元就是张量
        #所以此次只需将不同的参数简单合并成列表即可
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

class Sigmoid:
    def __call__(self, x):
        #self.out = 1 / (1 + torch.exp(-x))
        self.out = torch.sigmoid(x)
        return self.out
    
    # def backward(self, grad):
    #     # Sigmoid derivative: σ'(x) = σ(x)*(1-σ(x))
    #     return grad * (self.out * (1 - self.out))
    
    def parameters(self):
        # 激活函数没有可训练参数
        return []
    
class Sequential:
    
    def __init__(self, layers):
        # layers 表示模型组件
        self.layers = layers

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def predict_proba(self, x):
        # 需要判断 x 是张量还是numpy
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        logits = self(x)
        # detach() 函数将张量从计算图中分离出来，并转换为numpy
        # 使用self.prob 而不是 prob 是因为在predict_proba 函数中，self.prob 是类属性，而不是局部变量
        self.prob = F.softmax(logits, dim=-1).detach().numpy()
        return self.prob

#定义感知器模型，模型的输入为x，有两个隐藏层，每个隐藏层的神经元个数为4，最终输出为二分类
#x ： [B, 2] ; mlp : [4, 4, 2]
# model = Sequential([
#     Linear(2, 4), Sigmoid(), #输出形状为 （B， 4）
#     Linear(4, 4), Sigmoid(), #输出形状为（B， 4）
#     Linear(4, 2)
# ])

# 进行简单的测试
# x  = torch.randn(3, 2)
# print(model(x))
# print(model.predict_proba(x))

# 生成月牙形数据
data = make_moons(200, noise=0.05)

batch_size = 20
max_steps = 40000

x, y = torch.tensor(data[0]).float(), torch.tensor(data[1])
# print(len(x)) = 200
learning_rate = 0.1
model = Sequential([
    Linear(2, 4), Sigmoid(), #输出形状为 （B， 4）
    Linear(4, 4), Sigmoid(), #输出形状为（B， 4）
    Linear(4, 2)
])
lossi = []
for t in range(max_steps):
    ix = (t * batch_size) % len(x) # 取模运算，确保索引在有效范围内 bs = 20 时，每10步 会更新一次 ix
    xx = x[ix: ix+batch_size]
    yy = y[ix: ix+batch_size]
    logits = model(xx)
    loss = F.cross_entropy(logits, yy)
    loss.backward() # 梯度的反向传播，传播到每一个参数上
    with torch.no_grad():          # 在更新参数时，不需要计算梯度，手动计算
        for p in model.parameters():
            p -= learning_rate * p.grad
            p.grad = None
    lossi.append(loss.item())

# 绘制损失函数
plt.plot(torch.tensor(lossi).view(-1, 100).mean(dim=-1))
plt.savefig('loss_mlp02_mean.png')

# # 绘制决策边界
# h = 0.25
# x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
# y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))    

# # 将网格点展平
# xx_flat = xx.reshape(-1)
# yy_flat = yy.reshape(-1)

# # 预测概率
# probs = model.predict_proba(torch.tensor(np.c_[xx_flat, yy_flat]))

# # 将概率转换为类别
# preds = np.argmax(probs, axis=-1)

# # 绘制决策边界
# plt.figure(figsize=(10, 8))
# plt.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.3, cmap='viridis')
# plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
# plt.savefig('decision_boundary_mlp02.png')

















