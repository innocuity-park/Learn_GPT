import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_data():

    np.random.seed(4889)

    x = np.array([10]+ list(range(10, 29)))
    error = np.round(np.random.randn(20), 2)
    y = x + error
    return pd.DataFrame({'x': x, 'y': y}) 
    #生成了一个DataFrame列表，只需要一个变量

def visualize_data(data):
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_xticks(range(10, 31, 5))
    ax.set_ylabel('y')
    ax.set_yticks(range(10, 31, 5))
    ax.scatter(data.x, data.y, color='blue',label='data')
    plt.legend(shadow=True) #展示出label 的内容
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tang_linear_plot.png'))
    plt.show()

if __name__ == '__main__':
    data = generate_data()
    home_path = os.path.dirname(os.path.abspath(__file__))
    #print(home_path)
    data.to_csv("%s/tang_linear.csv" % home_path, index=False)
    visualize_data(data)