import sklearn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self):
        super().__init__()#继承Module的init方法
        #定义模型参数
        self.a = nn.Parameter(torch.zeros())
        self.b = nn.Parameter(torch.zeros())

        
    def forward(self, x):
        #向前传播
        return self.a * x + self.b
    
    def string(self):
        return f'y = {self.a.item():.2f} * x + {self.b.item():.2f}'

