
"""逻辑斯蒂回归算法实现-调用sklearn模块"""

from sklearn.linear_model import LogisticRegression
import numpy as np

def main():
    # 训练数据集
    X_train=np.array([[3,3,3],[4,3,2],[2,1,2],[1,1,1],[-1,0,1],[2,-2,1]])
    y_train=np.array([1,1,1,0,0,0])
    # 选择不同solver,构建实例，进行训练、测试
    methodes=["liblinear","newton-cg","lbfgs","sag","saga"]
    res=[]
    X_new = np.array([[1, 2, -2]])
    for method in methodes:
        clf=LogisticRegression(solver=method,intercept_scaling=2,max_iter=1000)
        clf.fit(X_train,y_train)
        # 预测新数据
        y_predict=clf.predict(X_new)
        #利用已有数据对训练模型进行评价
        X_test=X_train
        y_test=y_train
        correct_rate=clf.score(X_test,y_test)
        res.append((y_predict,correct_rate))

    # 格式化输出
    methodes=["liblinear","newton-cg","lbfgs    ","sag      ","saga      "]
    print("solver选择：          {}".format("  ".join(method for method in methodes)))
    print("{}被分类为：  {}".format(X_new[0],"        ".join(str(re[0]) for re in res)))
    print("测试{}组数据，正确率： {}".format(X_train.shape[0],"        ".join(str(round(re[1],1)) for re in res)))

if __name__=="__main__":
    main()