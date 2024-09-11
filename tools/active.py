import numpy as np
import matplotlib.pyplot as plt

# 定义Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义Tanh函数
def tanh(x):
    return (np.tanh(x)+1)/2

# 生成数据点
x = np.linspace(-10, 10, 100)
sigmoid_y = sigmoid(x)
tanh_y = tanh(x)

# 绘制Sigmoid函数曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 1行2列的第一个位置
plt.plot(x, sigmoid_y, label='Sigmoid')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.legend()

# 绘制Tanh函数曲线
plt.subplot(1, 2, 2)  # 1行2列的第二个位置
plt.plot(x, tanh_y, label='Tanh')
plt.title('Tanh Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True)
plt.legend()

# 展示图形
plt.tight_layout()
plt.show()
