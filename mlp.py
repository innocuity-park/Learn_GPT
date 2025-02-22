import torch

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
            self.bias = torch.randn(out_features) if bias else None
        else:
            self.bias = None

    def __call__(self, x):
        # 计算线性函数的输出
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


