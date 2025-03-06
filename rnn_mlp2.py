'''
本内容主要为训练自然语言模型，使用MLP模型
生成python自动写脚本的模型
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import string


#定义字典
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
char2indx = {s: i for i, s in enumerate(string.ascii_lowercase)} 

#所谓文本嵌入（embedding），就是将文本的热向量（one-hot vector）转换为一个低维的实数向量
#具体的操作就是将一个矩阵乘以一个向量，得到一个向量
#nn.Embedding()函数就是用来实现这个功能的



























