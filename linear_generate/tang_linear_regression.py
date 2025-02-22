import os
import pandas as pd
import numpy as np

from sklearn import linear_model

#使用sklearn 的模型，要求输入为n*m的矩阵，n为样本数，m为特征数
#data.x 是一个1*n的矩阵
#data[['x']] 是一个n*1的矩阵

def read_data(path):
    data = pd.read_csv(path)
    return data

def train_model(x, y):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    return model

def evaluate_model(model, x, y):
    mse = np.mean((model.predict(x) - y) ** 2)
    score = model.score(x, y)
    return mse, score

def visualize_model(model, x, y):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_xticks(range(10, 31, 5))
    ax.set_ylabel('y')
    ax.set_yticks(range(10, 31, 5))
    ax.scatter(x, y, color='blue',label='data')
    ax.plot(x, model.predict(x), color='red', label='model')
    plt.savefig('tang_linear_plot.png')
    plt.legend(shadow=True)
    plt.show()

def run_model(data):
    features = ['x']
    label = ['y']

    model = train_model(data[features], data[label])
    mse, score = evaluate_model(model, data[features], data[label])
    print('MSE: %f' % mse)
    print('Score: %f' % score)
    visualize_model(model, data[features], data[label])

if __name__ == '__main__':
    home_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(home_path, 'tang_linear.csv')
    data = read_data(data_path)
    
    run_model(data)
