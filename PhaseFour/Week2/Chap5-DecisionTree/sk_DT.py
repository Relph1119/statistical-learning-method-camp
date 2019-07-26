"""朴决策树sklearn的实现"""
"""2019/4/19"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time

from IPython.display import Image
from sklearn import tree
import pydotplus

def show(clf,features,y_types):
    """决策树的可视化"""
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=features,
                                    class_names=y_types,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    #Image(graph.create_png())  #jupyter里可以显示，pycharm显示不出
    graph.write_png(r'DT_show.png')

def main():
    star=time.time()
    # 原始样本数据
    features=["age","work","house","credit"]
    X_train=pd.DataFrame([
                      ["青年", "否", "否", "一般"],
                      ["青年", "否", "否", "好"],
                      ["青年", "是", "否", "好"],
                      ["青年", "是", "是", "一般"],
                      ["青年", "否", "否", "一般"],
                      ["中年", "否", "否", "一般"],
                      ["中年", "否", "否", "好"],
                      ["中年", "是", "是", "好"],
                      ["中年", "否", "是", "非常好"],
                      ["中年", "否", "是", "非常好"],
                      ["老年", "否", "是", "非常好"],
                      ["老年", "否", "是", "好"],
                      ["老年", "是", "否", "好"],
                      ["老年", "是", "否", "非常好"],
                      ["老年", "否", "否", "一般"]
                      ])
    y_train = pd.DataFrame(["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"])
    # 数据预处理
    le_x=preprocessing.LabelEncoder()
    le_x.fit(np.unique(X_train))
    X_train=X_train.apply(le_x.transform)
    print(X_train)
    le_y=preprocessing.LabelEncoder()
    le_y.fit(np.unique(y_train))
    y_train=y_train.apply(le_y.transform)
    # 调用sklearn.DT建立训练模型
    clf=DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    # 可视化
    show(clf,features,[str(k) for k in np.unique(y_train)])
    # 用训练得到模型进行预测
    X_new=pd.DataFrame([["青年", "否", "是", "一般"]])
    X=X_new.apply(le_x.transform)
    y_predict=clf.predict(X)
    # 结果输出
    X_show=[{features[i]:X_new.values[0][i]} for i in range(len(features))]
    print("{0}被分类为:{1}".format(X_show,le_y.inverse_transform(y_predict)))
    print("time:{:.4f}s".format(time.time()-star))

if __name__=="__main__":
    main()