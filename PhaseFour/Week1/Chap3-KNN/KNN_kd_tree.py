# 构建kd树，搜索待预测点所属区域
from collections import namedtuple
import numpy as np

# 建立节点类
class Node(namedtuple("Node","location left_child right_child")):
    def __repr__(self):
        return str(tuple(self))

# kd tree类
class KdTree():
    def __init(self,k=1,p=2):
        self.k=k
        self.p=p
        self.kdtree=None

    #构建kd tree
    def _fit(self,X,depth=0):
        try:
            k=X.shape[1]
        except IndexError as e:
            return None
        # 这里可以展开，通过方差选择axis
        axis=depth % k
        X=X[X[:,axis].argsort()]
        median=X.shape[0]//2
        try:
            X[median]
        except IndexError:
            return None
        return Node(
            location=X[median],
            left_child=self._fit(X[:median],depth+1),
            right_child=self._fit(X[median+1:],depth+1)
        )

    def _search(self,point,tree=None,depth=0,best=None):
        if tree is None:
            return best
        k=point.shape[1]
        # 更新 branch
        if point[0][depth%k]<tree.location[depth%k]:
            next_branch=tree.left_child
        else:
            next_branch=tree.right_child
        if not next_branch is None:
            best=next_branch.location
        return self._search(point,tree=next_branch,depth=depth+1,best=best)

    def fit(self,X):
        self.kdtree=self._fit(X)
        return self.kdtree

    def predict(self,X):
        res=self._search(X,self.kdtree)
        return res

def main():
    KNN=KdTree()
    X_train=np.array([[5,4],
                      [9,6],
                      [4,7],
                      [2,3],
                      [8,1],
                      [7,2]])
    KNN.fit(X_train)
    X_new=np.array([[5,3]])
    res=KNN.predict(X_new)
    print(res)

if __name__=="__main__":
    main()