import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# 入参以横向展示,多个参数可以以二维传入
class NeuralNetwork:
    def __init__(self, layers):  # """参数sizes表示每一层神经元的个数，如[2,3,1],表示第一层有2个神经元，第二层有3个神经元，第三层有1个神经元."""
        self.activation = sigmoid
        self.activation_deriv = sigmoid_derivative

        self.weights = []  # 权重 list
        for i in range(len(layers) - 1):  # 初始化权重
            self.weights.append(np.random.random((layers[i], layers[i + 1])))  # 产生矩阵

        self.biases = []
        for i in range(len(layers) - 1):
            self.biases.append(np.random.random((layers[i + 1])))  # 产生矩阵

    def fit(self, X, y, learning_rate=0.2, epochs=10000):  # X二维的,一行代表一个实例  Y 输出值
        #  learning_rate 学习率,, epochs,避免计算量大,使用抽样数据,终止三条件之一,循环次数
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, :] = X  # 添加 偏差??
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # 从X中选取任一行
            outList = []  # 输出集
            outList.append(X[i])  # 入参当做第一个输出集
            for j in range(len(self.weights)):  # going forward network, for each layer
                out = np.dot(outList[j], self.weights[j]) + self.biases[j]
                outList.append(self.activation(out))

            #             out = outList[-1]
            deltas = [outList[-1] * (1 - outList[-1]) * (y[i] - outList[-1])]
            length = len(self.weights)
            length2 = len(outList)
            for j in range(length):  #
                temp = outList[j].dot(1 - outList[j]).dot(deltas[j].dot(self.weights[length - j - 1].T))
                deltas.append(temp)
            #                 deltas.append((deltas[j].dot(self.weights[length - j - 1].T)) * self.activation_deriv(outList[length2 - j - 2]) * (1 - self.activation_deriv(outList[length2 - j - 2])))
            # deltas.append(  deltas[-1].dot(self.weights[j].T) * self.activation_deriv(a[j]))  # 往回走使用反函数activation_deriv
            deltas.reverse()  # reverse 颠倒顺序
            for i in range(len(self.weights)):
                # layer = np.atleast_2d(a[i+1])
                aaa = np.array(outList[i + 1])
                bbb = np.array(deltas[i])
                # delta = np.atleast_2d(deltas[i])
                # self.weights[i] += learning_rate * layer.T.dot(delta)  # T  转置
                self.weights[i] += bbb.T.dot(aaa) * learning_rate  # T  转置

                # 偏向更新
                self.biases[i] += learning_rate * deltas[i + 1]

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0])
        temp[:] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
