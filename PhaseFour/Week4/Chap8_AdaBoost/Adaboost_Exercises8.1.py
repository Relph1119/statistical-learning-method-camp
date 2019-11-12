# 自编程求解习题8.1
import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, x=None, label=None, y=None, data=None):
        # 特征值序号
        self.label = label
        # x为特征值的类别值
        self.x = x
        # 子节点
        self.child = []
        # 分类类别
        self.y = y
        # 分类类别数据
        self.data = data

        # 添加子节点

    def append(self, node):
        self.child.append(node)

    # 根据决策树进行预测
    def predict(self, features):
        if self.y is not None:
            # 返回预测结果
            return self.y
        # 对子节点进行遍历
        for c in self.child:
            # 特征值x为待预测数据的特征值
            if c.x == features[self.label]:
                return c.predict(features)


class DecisionTreeStump:
    def __init__(self, datasets, epsilon=0, alpha=0):
        # 阈值
        self.epsilon = epsilon
        # alpha值
        self.alpha = alpha
        # 初始化为单节点树
        self.tree = Node()
        self.datasets = datasets

    # 计算概率值
    @staticmethod
    def prob(datasets):
        # 得到数据集的样本总数
        datalen = len(datasets)
        # 得到样本的分类类别数
        labelx = set(datasets)
        # 初始化频值
        p = {l: 0 for l in labelx}
        # 统计各个频值数
        for d in datasets:
            p[d] += 1
        # 得到每个类别的概率值
        for i in p.items():
            p[i[0]] /= datalen
        return p

    # 计算熵
    def calc_ent(self, datasets):
        p = self.prob(datasets)
        ent = sum([-v * math.log(v, 2) for v in p.values()])
        return ent

    # 计算经验条件熵
    def cond_ent(self, datasets, col):
        # 取出第col行的数据（即取出第几个特征值对应的行数据）
        labelx = set(datasets.iloc[col])
        # 初始化特征值对应的频值字典
        p = {x: [] for x in labelx}
        # 对i(第i条数据)，d(分类类别)迭代
        # 统计特征值下的各个类别的标签分类
        for i, d in enumerate(datasets.iloc[-1]):
            p[datasets.iloc[col][i]].append(d)
        # 计算经验条件熵
        return sum([self.prob(datasets.iloc[col])[k] * self.calc_ent(p[k]) for k in p.keys()])

    def info_gain_train(self, datasets, datalabels):
        # 将数据集转置，即特征值是表的RowName，数据集是0-14列
        datasets = datasets.T
        # iloc[num]：通过行号来取行数据
        # 取最后一行数据（即类别行）计算经验熵
        ent = self.calc_ent(datasets.iloc[-1])
        # 初始化信息增益字典
        gainmax = {}
        # 计算信息增益，并将信息增值作为字典的key，而value为数据序号
        # len(datasets) - 1 除去分类列
        for i in range(len(datasets) - 1):
            cond = self.cond_ent(datasets, i)
            gainmax[ent - cond] = i
        # 取最大信息增益
        m = max(gainmax.keys())
        # gainmax[m]：最大信息增益对应的特征值序号，m：信息增益
        return gainmax[m], m

    # 训练函数（节点划分函数）
    # datasets：数据集，node：节点树
    def train(self, datasets, node):
        # 取标签列
        labely = datasets.columns[-1]
        # value_counts() 以Series形式返回指定列的不同取值的频率
        # 如果都为同一标签分类的数据
        if len(datasets[labely].value_counts()) == 1:
            # 该子节点的数据集为该分类列的数据
            node.data = datasets[labely]
            # y为标签分类类别
            node.y = datasets[labely][0]
            return
        # 如果没有了需要分类的特征值
        if len(datasets.columns[:-1]) == 0:
            # 该节点的数据集为该分类列的数据
            node.data = datasets[labely]
            # y为该标签分类的类别
            node.y = datasets[labely].value_counts().index[0]
            return

        # 得到gainmaxi：最优的特征，gainmax：最大信息增益
        gainmaxi, gainmax = self.info_gain_train(datasets, datasets.columns)
        # 小于阈值，不需要再划分了
        if gainmax <= self.epsilon:
            node.data = datasets[labely]
            node.y = datasets[labely].value_counts().index[0]
            return

        # 最优特征下的类别频值
        vc = datasets[datasets.columns[gainmaxi]].value_counts()
        # 对类别频值进行遍历
        for Di in vc.index:
            # label赋值为最优特征值序号
            node.label = gainmaxi
            # 创建类别子节点，x为类别值
            child = Node(Di)
            # 添加子节点列表到类别节点
            node.append(child)
            # 得到该特征值下为Di类别的数据集
            new_datasets = pd.DataFrame([list(i) for i in datasets.values if i[gainmaxi] == Di],
                                        columns=datasets.columns)
            # 对数据集进行再分类
            self.train(new_datasets, child)

    # 匹配函数
    def fit(self):
        self.train(self.datasets, self.tree)

    # 得到所有的叶子节点集合
    def findleaf(self, node, leaf):
        # 对该节点下的子节点进行遍历
        for t in node.child:
            # 判断分类值是否为空，如果为空，则表示这是一个特征，如果不为空，则表示这是一个叶子节点
            if t.y is not None:
                # 添加数据到叶子节点集合中
                leaf.append(t.data)
            else:
                for c in node.child:
                    self.findleaf(c, leaf)

    def findfather(self, node, errormin):
        if node.label is not None:
            cy = [c.y for c in node.child]
            if None not in cy:
                childdata = []
                for c in node.child:
                    for d in list(c.data):
                        childdata.append(d)
                # 统计childdata的频值
                childcounter = Counter(childdata)

                old_child = node.child
                old_label = node.label
                old_y = node.y
                old_data = node.data

                node.label = None
                # 根据奥卡姆剃刀准则，进行简化处理
                node.y = childcounter.most_common(1)[0][0]
                node.data = childdata

                # 再次计算预测误差
                error = self.c_error()
                # 获得最小的预测误差
                if error <= errormin:
                    errormin = error
                    return 1
                else:
                    node.child = old_child
                    node.label = old_label
                    node.y = old_y
                    node.data = old_data
            else:
                re = 0
                i = 0
                while i < len(node.child):
                    if_re = self.findfather(node.child[i], errormin)
                    if if_re == 1:
                        re = 1
                    elif if_re == 2:
                        i -= 1
                    i += 1
                if re:
                    return 2
        return 0

    # 求模型对训练数据的预测误差
    def c_error(self):
        leaf = []
        # 找到叶子节点
        self.findleaf(self.tree, leaf)
        # 计算每个特征值下的总数
        leafnum = [len(l) for l in leaf]
        # 计算每一类的信息熵
        ent = [self.calc_ent(l) for l in leaf]
        print("Ent:", ent)
        # 求偏差alpha*|T|
        error = self.alpha * len(leafnum)
        # 求损失函数的C(T)值
        for l, e in zip(leafnum, ent):
            error += l * e
        print("C(T):", error)
        return error


class AdaBoost:
    def __init__(self, X, y, train_data,base_estimator=DecisionTreeStump, max_iter=5):
        self.X = np.array(X)
        self.y = np.array(y).flatten(1)
        self.train_data = train_data
        self.base_estimator = base_estimator
        self.sums = np.zeros(self.y.shape)
        self.max_iter = max_iter

        # W为权值，初试情况为均匀分布，即所有样本都为1/n
        self.W = np.ones((self.X.shape[1], 1)).flatten(1) / self.X.shape[1]
        # 弱分类器的实际个数
        self.Q = 0

    # M 为弱分类器的最大数量，可以在main函数中修改
    def fit(self):
        self.G = {}  # 表示弱分类器的字典
        self.alpha = {}  # 每个弱分类器的参数
        for i in range(self.max_iter):
            self.G.setdefault(i)
            self.alpha.setdefault(i)
        for i in range(self.max_iter):  # self.G[i]为第i个弱分类器
            # TODO:
            train_data = self.train_data , self.W
            self.G[i] = self.base_estimator(train_data)
            self.G[i].fit()  # 根据当前权值进行该个弱分类器训练
            e = self.G[i].c_error()
            self.alpha[i] = 1.0 / 2 * np.log((1 - e) / e)  # 计算该分类器的系数
            res = self.G[i].tree.predict(self.X)  # res表示该分类器得出的输出

            # 计算当前次数训练精确度
            print("weak classfier acc", accuracy_score(self.y, res),
                  "\n======================================================")

            # Z表示规范化因子
            Z = self.W * np.exp(-self.alpha[i] * self.y * res.transpose())
            self.W = (Z / Z.sum()).flatten(1)  # 更新权值
            self.Q = i
            # errorcnt返回分错的点的数量，为0则表示perfect
            if self.errorcnt(i) == 0:
                print("%d个弱分类器可以将错误率降到0" % (i + 1))
                break

    def errorcnt(self, t):  # 返回错误分类的点
        self.sums = self.sums + self.G[t].tree.predict(self.X).flatten(1) * self.alpha[t]

        pre_y = np.zeros(np.array(self.sums).shape)
        pre_y[self.sums >= 0] = 1
        pre_y[self.sums < 0] = -1

        t = (pre_y != self.y).sum()
        return t

    def predict(self, test_X):  # 测试最终的分类器
        test_X = np.array(test_X)
        sums = np.zeros(test_X.shape[1])
        for i in range(self.Q + 1):
            sums = sums + self.G[i].tree.predict(test_X).flatten(1) * self.alpha[i]
        pre_y = np.zeros(np.array(sums).shape)
        pre_y[sums >= 0] = 1
        pre_y[sums < 0] = -1
        return pre_y

    def score(self, X, y):
        """对训练效果进行评价"""
        y_predict = self.predict(X)
        error_rate = np.sum(np.abs(y_predict - y)) / 2 / y.shape[0]
        return 1 - error_rate


def main():
    datasets = np.array([[0, 1, 3, -1],
                         [0, 3, 1, -1],
                         [1, 2, 2, -1],
                         [1, 1, 3, -1],
                         [1, 2, 3, -1],
                         [0, 1, 2, -1],
                         [1, 1, 2, 1],
                         [1, 1, 1, 1],
                         [1, 3, 1, -1],
                         [0, 2, 1, -1]
                         ])
    datalabels = np.array(['身体', '业务', '潜力', '分类'])
    train_data = pd.DataFrame(datasets, columns=datalabels)
    X = np.array(train_data[train_data.columns[:-1]].values)
    y = np.array(train_data[train_data.columns[-1]].values)
    clf = AdaBoost(X, y, train_data, max_iter=50)
    clf.fit()

    y_predict = clf.predict(X)
    score = clf.score(X, y)

    print("原始输出:", y)
    print("预测输出:", y_predict)
    print("预测正确率：{:.2%}".format(score))


if __name__ == "__main__":
    main()
