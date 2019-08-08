import numpy as np
from collections import Counter
from .draw import draw
from concurrent import futures
import heapq
import time


class KNN:
    def __init__(self, X_train, y_train, k=3):
        # 所需参数初始化
        self.k = k  # 所取k值
        self.X_train = X_train
        self.y_train = y_train

    def predict_single(self, x_new):
        # 计算与前k个样本点欧氏距离，距离取负值是把原问题转化为取前k个最大的距离
        dist_list = [(-np.linalg.norm(x_new - self.X_train[i], ord=2), self.y_train[i], i)
                     for i in range(self.k)]
        # 利用前k个距离构建堆
        heapq.heapify(dist_list)
        # 遍历计算与剩下样本点的欧式距离
        for i in range(self.k, self.X_train.shape[0]):
            dist_i = (-np.linalg.norm(x_new - self.X_train[i], ord=2), self.y_train[i], i)
            # 若dist_i 比 dis_list的最小值大，则用dis_i替换该最小值，执行下堆操作
            if dist_i[0] > dist_list[0][0]:
                heapq.heappushpop(dist_list, dist_i)
            # 若dist_i 比 dis_list的最小值小，堆保持不变，继续遍历
            else:
                continue
        y_list = [dist_list[i][1] for i in range(self.k)]
        # [-1,1,1,-1...]
        # 对上述k个点的分类进行统计
        y_count = Counter(y_list).most_common()
        # {1:n,-1:m}
        return y_count[0][0]

    # 用多进程实现并行，处理多个值的搜索
    def predict_many(self, X_new):
        # 导入多进程
        with futures.ProcessPoolExecutor(max_workers=4) as executor:
            # 建立多进程任务
            tasks = [executor.submit(self.predict_single, X_new[i]) for i in range(X_new.shape[0])]
            # 驱动多进程运行
            done_iter = futures.as_completed(tasks)
            # 提取运行结果
            res = [future.result() for future in done_iter]
        return res


def main():
    t0 = time.time()
    # 训练数据
    X_train = np.array([[5, 4],
                        [9, 6],
                        [4, 7],
                        [2, 3],
                        [8, 1],
                        [7, 2]])
    y_train = np.array([1, 1, 1, -1, -1, -1])
    # 测试数据
    X_new = np.array([[5, 3], [9, 2]])
    # 绘图
    draw(X_train, y_train, X_new)
    # 不同的k(取奇数）对分类结果的影响
    for k in range(1, 6, 2):
        # 构建KNN实例
        clf = KNN(X_train, y_train, k=k)
        # 对测试数据进行分类预测
        y_predict = clf.predict_many(X_new)
        print("k={},被分类为：{}".format(k, y_predict))
    print("用时:{}s".format(round(time.time() - t0), 2))


if __name__ == "__main__":
    main()
