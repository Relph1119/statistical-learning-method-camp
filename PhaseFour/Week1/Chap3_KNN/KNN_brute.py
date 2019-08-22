import numpy as np
from collections import Counter
from Week1.Chap3_KNN.draw import draw


class KNN:
    def __init__(self, X_train, y_train, k=3):
        # 所需参数初始化
        self.k = k  # 所取k值
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_new):
        # 计算欧氏距离
        dist_list = [(np.linalg.norm(X_new - self.X_train[i], ord=2), self.y_train[i])
                     for i in range(self.X_train.shape[0])]
        # [(d0,-1),(d1,1)...]
        # 对所有距离进行排序
        dist_list.sort(key=lambda x: x[0])
        # 取前k个最小距离对应的类别（也就是y值）
        y_list = [dist_list[i][-1] for i in range(self.k)]
        # [-1,1,1,-1...]
        # 对上述k个点的分类进行统计
        y_count = Counter(y_list).most_common()
        # [(-1, 3), (1, 2)]
        return y_count[0][0]


def main():
    # 训练数据
    X_train = np.array([[5, 4],
                        [9, 6],
                        [4, 7],
                        [2, 3],
                        [8, 1],
                        [7, 2]])
    y_train = np.array([1, 1, 1, -1, -1, -1])
    # 测试数据
    X_new = np.array([[5, 3]])
    # 绘图
    draw(X_train, y_train, X_new)
    # 不同的k(取奇数）对分类结果的影响
    for k in range(1, 6, 2):
        # 构建KNN实例
        clf = KNN(X_train, y_train, k=k)
        # 对测试数据进行分类预测
        y_predict = clf.predict(X_new)
        print("k={}，被分类为：{}".format(k, y_predict))


if __name__ == "__main__":
    main()
