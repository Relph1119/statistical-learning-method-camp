import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def main():
    # 训练数据
    X_train=np.array([[5,4],
                      [9,6],
                      [4,7],
                      [2,3],
                      [8,1],
                      [7,2]])
    y_train=np.array([1,1,1,-1,-1,-1])
    # 待预测数据
    X_new = np.array([[5, 3]])
    # 不同k值对结果的影响
    for k in range(1,6,2):
        # 构建实例
        clf = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
        # 选择合适算法
        clf.fit(X_train, y_train)
        # print(clf.kneighbors(X_new))
        # 预测
        y_predict=clf.predict(X_new)
        #print(clf.predict_proba(X_new))
        print("预测正确率:{:.0%}".format(clf.score([[5,3]],[[1]])))
        print("k={},被分类为：{}".format(k,y_predict))

if __name__=="__main__":
    main()