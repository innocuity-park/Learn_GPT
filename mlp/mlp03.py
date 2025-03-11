'''
本篇内容为mlp模型的优化训练
主要解决梯度消失
'''
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
torch.manual_seed(1024)

#依然先定义 线性模型和sigmoid 函数
class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = torch.randn((in_features, out_features),requires_grad=True)
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
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
class Sigmoid:
    def __call__(self, x):
        self.out = torch.sigmoid(x)
        return self.out
    
    def parameters(self):
        return []
    
class Sequential:
    def __init__(self, layers):
        # layers 可以一次性传入很多层
        self.layers = layers

    def __call__(self, x):
        for layers in self.layers:
            x = layers(x)
        self.out = x
        return self.out 
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
def train_model(model, data, max_steps):
    lossi = []
    learning_rate = 0.1
    #记录各层的参数的更新幅度
    # {1: [...], 2: [...]}
    udi = {}
    # data 是根据sklearn 生成的数据集
    x, y = torch.tensor(data[0]).float(), torch.tensor(data[1])
    for i in range(max_steps):
        # 向前传播
        logits = model(x)
        # 计算损失
        loss = F.cross_entropy(logits, y)
        # 保留中间节点的梯度，以便观察
        for layer in model.layers:
            layer.out.retain_grad()
        for p in model.parameters():
            p.grad = None
        
        # 方向传播
        loss.backward()
        with torch.no_grad(): # torch.no_grad() 上下文管理器，用于关闭梯度计算
            for i, p in enumerate(model.parameters()):
                p -= learning_rate * p.grad
                # 计算参数的更新幅度 .get(i, []) 如果i不在字典中，返回空列表
                udi[i] = udi.get(i, []) + [(learning_rate * p.grad).std()/p.std()]
        lossi.append(loss.item())
    return lossi, udi

data = make_moons(2000, noise = 0.05)
n_hidden = 100

model = Sequential([
    Linear(2       , n_hidden), Sigmoid(),
    Linear(n_hidden, n_hidden), Sigmoid(),
    Linear(n_hidden, 2)

])
train_model(model, data, 2)
# print(train_model(model , data, 2))

# 监控激活函数的输出
def saturation_stats(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Sigmoid):
            # 首先取出每一层激活函数的输出
            t = layer.out
            # 当激活函数的输出大于0.99 或者 小于 0.01 时，认为激活函数过热
            # 同时计算过热比例
            saturation = ((t - 0.5).abs() > 0.49).float().mean()
            # 激活函数的输出分布情况
            hy , hx = torch.histogram(t, density=True)
            a = plt.figure(1)
            plt.plot(hx[:-1].detach(), hy.detach())
            print(f'layer {i}  mean{t.mean():.2f} std{t.std():.2f}saturation: {saturation:.2f}')
        if isinstance(layer, Linear):
            w = layer.parameters()[0]
            g = w.grad
            grad_ratio = g.std() / w.std()
            hy , hx = torch.histogram(g, density=True)
            b = plt.figure(2)
            plt.plot(hx[:-1].detach(), hy.detach())
            print(f'layer {i}  mean{g.mean():.2f} std{g.std():.2f} grad_ratio {grad_ratio:.2f} ')
    # 保存图片
    a.savefig('saturation_stats.png')
    b.savefig('grad_stats.png')
saturation_stats(model) # 执行此项函数 需先完成train_model 的步骤


class BatchNormld:
    '''
    对每一批数据的 features 进行归一化处理 批归一化
    输入 x ： （B， f）
    需要区分是训练状态还是运行预测状态
    '''
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta = torch.zeros(dim, requires_grad=True)

    def __call__(self, x):
        # x ： （B， f）
        xmean = x.mean(0, keepdim=True) # 0 表示按列求均值，1表示按行求均值，keepdim 保持维度不变
        xvar = x.var(0, keepdim=True) # 按列求方差
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

model2 = Sequential([
    Linear(2       , n_hidden, bias=False), BatchNormld(n_hidden), Sigmoid(),
    Linear(n_hidden, n_hidden, bias=False), BatchNormld(n_hidden), Sigmoid(),
    Linear(n_hidden, 2, bias=False)

])

class LayerNormld:
    '''
    对每一批数据的 features 进行归一化处理 批归一化
    输入 x ： （B， f）
    需要区分是训练状态还是运行预测状态
    '''
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta = torch.zeros(dim, requires_grad=True)

    def __call__(self, x):
        # x ： （B， f）
        xmean = x.mean(1, keepdim=True) # 0 表示按列求均值，1表示按行求均值，keepdim 保持维度不变
        xvar = x.var(1, keepdim=True) # 按列求方差
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]










