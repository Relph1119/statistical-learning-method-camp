import numpy as np
import pandas as pd

class MyHMM:
    def __init__(self,p,A,B,Ob,method="forward"):
        self.p=p  # 初始状态概率
        self.A=A  # 状态转移概率分布
        self.B=B  # 观测概率分布
        self.Ob=Ob # 观测序列
        self.method=method # 采用算法

    def forward(self):
        """前向算法"""
        alpha=self.p*self.B[self.Ob[0]]
        for i in range(1,self.Ob.size):
            alpha=alpha.dot(self.A)*self.B[self.Ob[i]]
        return alpha.sum()

    def back(self):
        """后向算法"""
        beta=np.ones(self.p.size)
        for i in range(1,self.Ob.size):
            beta=self.A.dot(self.B[self.Ob[self.Ob.size-i]]*beta)
        res=self.p.dot(self.B[self.Ob[0]]*beta)
        return res

    def fit(self):
        if "forward" in self.method:
            print("前向算法")
            res=self.forward()
        elif "back" in self.method:
            print("后向算法")
            res=self.back()
        else:
            raise ValueError
        return res

    def best_way(self):
        """最优路径"""
        index=[]
        sigema=self.p*self.B[self.Ob[0]]
        for i in range(1,self.Ob.size):
            _sigema=sigema.values*(self.A.T)
            col_index=np.argmax(_sigema,axis=1)
            row_index=np.arange(self.p.size)
            sigema=_sigema[row_index,col_index]*self.B[self.Ob[i]]
            index.append(dict(enumerate(col_index)))
        best_index=[]
        k=np.argmax(sigema)
        best_index.append(k)
        for i in range(len(index)):
            k=index[len(index)-1-i][k]
            best_index.append(k)
        best_index=np.array(best_index[::-1])+1
        return best_index

def main():
    A=np.array([[0.5,0.2,0.3],
                [0.3,0.5,0.2],
                [0.2,0.3,0.5]
                ])
    B=pd.DataFrame(np.array([[0.5,0.5],
                             [0.4,0.6],
                             [0.7,0.3]]),columns=["红","白"])
    p=np.array([0.2,0.4,0.4])
    Ob=np.array(["红","白","红","白"])
    clf=MyHMM(p,A,B,Ob,method="back")
    res=clf.fit()
    print(round(res,2))
    best_way=clf.best_way()
    print(best_way)

if __name__=="__main__":
    main()