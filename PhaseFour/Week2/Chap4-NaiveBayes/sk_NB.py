"""朴素贝叶斯算法sklearn实现"""
"""2019/4/15"""

import numpy as np
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn import preprocessing  #预处理

def main():
    X_train=np.array([
                      [1,"S"],
                      [1,"M"],
                      [1,"M"],
                      [1,"S"],
                      [1,"S"],
                      [2,"S"],
                      [2,"M"],
                      [2,"M"],
                      [2,"L"],
                      [2,"L"],
                      [3,"L"],
                      [3,"M"],
                      [3,"M"],
                      [3,"L"],
                      [3,"L"]
                      ])
    y_train=np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(X_train)
    X_train = enc.transform(X_train).toarray()
    print(X_train)
    clf=MultinomialNB(alpha=0.0000001)
    clf.fit(X_train,y_train)
    X_new=np.array([[2,"S"]])
    X_new=enc.transform(X_new).toarray()
    y_predict=clf.predict(X_new)
    print("{}被分类为:{}".format(X_new,y_predict))
    print(clf.predict_proba(X_new))

if __name__=="__main__":
    main()