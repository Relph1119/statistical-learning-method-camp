# 自编程求解例8.1
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

def main():
    # X=np.array([0,1,2,3,4,5,6,7,8,9]).reshape(10,1)
    # y=np.array([1,1,1,-1,-1,-1,1,1,1,-1])
    X=np.array([[0,1,3],
                [0,3,1],
                [1,2,2],
                [1,1,3],
                [1,2,3],
                [0,1,2],
                [1,1,2],
                [1,1,1],
                [1,3,1],
                [0,2,1]
               ])
    y=np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1])
    clf=AdaBoostClassifier()
    clf.fit(X,y)
    y_predict=clf.predict(X)
    score=clf.score(X,y)
    print("原始输出:",y)
    print("预测输出:",y_predict)
    print("预测正确率：{:.2%}".format(score))

if __name__=="__main__":
    main()