import  matplotlib.pyplot as plt
import numpy as np

def draw(X_train,y_train,X_new):
    # 正负实例点初始化
    X_po=np.zeros(X_train.shape[1])
    X_ne=np.zeros(X_train.shape[1])
    # 区分正、负实例点
    for i in range(y_train.shape[0]):
        if y_train[i]==1:
            X_po=np.vstack((X_po,X_train[i]))
        else:
            X_ne=np.vstack((X_ne,X_train[i]))
    # 实例点绘图
    plt.plot(X_po[1:,0],X_po[1:,1],"g*",label="1")
    plt.plot(X_ne[1:, 0], X_ne[1:, 1], "rx", label="-1")
    plt.plot(X_new[:, 0], X_new[:, 1], "bo", label="test_points")
    # 测试点坐标值标注
    for xy in zip(X_new[:, 0], X_new[:, 1]):
        plt.annotate("test{}".format(xy),xy)
    # 设置坐标轴
    plt.axis([0,10,0,10])
    plt.xlabel("x1")
    plt.ylabel("x2")
    # 显示图例
    plt.legend()
    # 显示图像
    plt.show()

