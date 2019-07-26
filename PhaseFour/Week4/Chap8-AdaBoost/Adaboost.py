# 自编程求解例8.1

import numpy as np

class AdaBoost:
    def __init__(self,X,y,tol=0.05,max_iter=10):
        self.X=X   # 训练数据 实例
        self.y=y   # 训练数据 标签
        self.tol=tol # 训练中止条件 right_rate>self.tol
        self.max_iter=max_iter # 最大迭代次数
        self.w=np.full((X.shape[0]),1/X.shape[0]) #初始化样本权重w
        self.alpha=[]  # 弱分类器权重
        self.G=[] # 弱分类器
        self.min_v=min(X)-0.5 #分类阈值下届
        self.max_v=max(X)+0.5 #分类阈值上届

    def _class(self):
        """以带权重的分类误差最小为目标，选择最佳分类阈值"""
        e_min=np.inf   # e_min 分类误差
        v_best=None    # v_best 最佳分类阈值
        sign=None      # sign 小于分类阈值的样本属于的标签类别
        for v in np.arange(self.min_v,self.max_v+0.5,1):
            # 遍历可能v_best可取值，寻找最优解
            e_1=-(self.y[self.X<v]-1)*self.w[self.X<v] # 假设小于阈值 分类为1 获取分类误差*2
            e_2=(self.y[self.X>v]+1)*self.w[self.X>v]  # 假设大于阈值 分类为-1 获取分类误差*2
            e=(e_1.sum()+e_2.sum())/2  # 计算整个分类误差
            if e<0.5:   # 若分类误差小于0.5 说明X<v y->1
                flag=1
            else:       # 若分类误差大于0.5 取其反向 说明X<v y->-1
                e=1-e
                flag=-1
            if e<e_min: # 保留最优解
                e_min=e
                sign=flag
                v_best=v
        return v_best,sign,e_min

    def updata_w(self):
        """更新样本权重w"""
        v,sign=self.G[-1]  # 以下2行 根据上一轮的弱分类器更新样本权重
        alpha=self.alpha[-1]
        G=np.zeros(self.y.size,dtype=int) # 以下三行重建弱分类器
        G[self.X<v]=sign
        G[self.X>v]=-sign
        # G_1=np.full((np.where(self.X<v))[0].shape[0],sign)
        # G_2=np.full((np.where(self.X>v))[0].shape[0],-sign)
        # G=np.hstack([G_1,G_2])
        P=self.w*np.exp(-alpha*self.y*G) #以下2行根据公式8.4 8.5 更新样本权重
        self.w=P/P.sum()

    def base_estimator(self,X,i):
        """计算单个弱分类器预测输出"""
        v,sign = self.G[i]
        _G_1 = np.full((np.where(X<v))[0].shape[0], sign)
        _G_2 = np.full((np.where(X>v))[0].shape[0], -sign)
        _G = np.hstack([_G_1, _G_2])
        return _G

    def fit(self):
        """对训练数据进行学习"""
        G=0
        for i in range(self.max_iter):
            class_v,sign,e=self._class() # 获取当前迭代最佳分类阈值
            alpha=1/2*np.log((1-e)/e)   # 计算本轮弱分类器的系数
            self.alpha.append(alpha)    # 保存弱分类器系数
            self.G.append((class_v,sign)) # 保存弱分类器
            _G=self.base_estimator(self.X,i) # 以下4行计算当前总分类器（之前所有弱分类器加权和）分类效率
            G+=alpha*_G
            y_predict=np.sign(G)
            error_rate=np.sum(np.abs(y_predict-self.y))/2/self.y.shape[0]
            if error_rate<self.tol: # 满足中止条件 则跳出循环
                print("迭代次数:",i+1)
                break
            else:
                self.updata_w()   # 若不满足，更新权重，继续迭代

    def predict(self,X):
        """对新数据进行预测"""
        G=0
        for i in range(len(self.alpha)):
            # 遍历每一个弱分类器，进行加权
            _G=self.base_estimator(X,i)
            alpha = self.alpha[i]
            G+=alpha*_G
        y_predict=np.sign(G)
        return y_predict.astype(int)

    def score(self,X,y):
        """对训练效果进行评价"""
        y_predict=self.predict(X)
        error_rate=np.sum(np.abs(y_predict-y))/2/y.shape[0]
        return 1-error_rate

def main():
    X=np.array([0,1,2,3,4,5,6,7,8,9])
    y=np.array([1,1,1,-1,-1,-1,1,1,1,-1])
    clf=AdaBoost(X,y)
    clf.fit()
    y_predict=clf.predict(X)
    score=clf.score(X,y)
    print("原始输出:",y)
    print("预测输出:",y_predict)
    print("预测正确率：{:.2%}".format(score))

if __name__=="__main__":
    main()