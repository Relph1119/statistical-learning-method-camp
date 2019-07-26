# 利用sympy求解习题7.2

from sympy import *
import numpy as np
import matplotlib.pyplot as plt

"""构造L(a,b,c,d)表达式,e=a+b+c-d"""
def creat(co,X,y,a,b,c,d,e):
    L_0=co*X*y
    L_1=L_0.sum(axis=0)
    L=np.dot(L_1,L_1)/2-co.sum()
    # 将e=a+b+c-d代入，化简整理
    L=expand(L.subs(e,a+b+c-d))
    return L

"""若L无解，则从L的多个边界求解"""
def _find_submin(L,num):
    if num.shape[0]==1:
        return None
    else:
        res=[]
        for i in range(num.shape[0]):
            L_child=L.subs({num[i]: 0})
            num_child=np.delete(num,i,axis=0)
            res.append(_find_min(L_child,num_child))
        return res

"""判断方程是否有唯一不小于0且不全为0的实数解"""
def _judge(res):
    for s in res.values():
        try:
            if float(s)<0:
                return False
        except:
            return False
    return True if sum(res.values())!=0 else False

"""求解所有可能的极值点，若极值不存在或不在可行域内取到，则在边界寻找极值点"""
def _find_min(L,num):
    pro_res=[]
    res=solve(diff(L,num),list(num))
    # 方程有解
    if res:
        # 方程有唯一不小于0且不全为0的实数解
        if _judge(res):
            pro_res.append(res)
            return pro_res
        # 方程有无数组解，到子边界寻找极值点
        else:
            value=_find_submin(L,num)
            pro_res.append(value)
    #方程无解，到子边界寻找极值点
    else:
        value=_find_submin(L,num)
        pro_res.append(value)
    return pro_res

"""将所有结果排列整齐"""
def reset(res):
    if not isinstance(res[0],list):
        if res[0]:
            res_list.append(res[0])
    else:
        for i in res:
            reset(i)

"""求解极小值点"""
def find_min(L,num,a,b,c,d,e):
    # 求解所有可能的极小值点
    results=_find_min(L,num)
    reset(results)
    L_min =float("inf")
    res =None
    # 在所有边界最小值中选取使得L(a,b,c,d)最小的点
    for i in res_list:
        d_i=dict()
        for j in [a,b,c,d]:
            d_i[j]=i.get(j,0)
        result=L.subs(d_i)
        if result<L_min:
            L_min=result
            res=d_i
    # 将e 计算出来并添加到res中
    res[e]=res[a]+res[b]+res[c]-res[d]
    return res

"""计算 w b"""
def calculate_w_b(X,y,res):
    alpha=np.array([[i] for i in res.values()])
    w=(alpha*X*y).sum(axis=0)
    for i in range(alpha.shape[0]):
        if alpha[i]:
            b=y[i]-w.dot(X[i])
            break
    return w,b

"""绘制样本点、分离超平面和间隔边界"""
def draw(X,y,w,b):
    y=np.array([y[i][0] for i in range(y.shape[0])])
    X_po=X[np.where(y==1)]
    X_ne=X[np.where(y==-1)]
    x_1=X_po[:,0]
    y_1=X_po[:,1]
    x_2=X_ne[:,0]
    y_2=X_ne[:,1]
    plt.plot(x_1,y_1,"ro")
    plt.plot(x_2,y_2,"gx")
    x=np.array([0,3])
    y=(-b-w[0]*x)/w[1]
    y_po=(1-b-w[0]*x)/w[1]
    y_ne=(-1-b-w[0]*x)/w[1]
    plt.plot(x,y,"r-")
    plt.plot(x,y_po,"b-")
    plt.plot(x,y_ne,"b-")
    plt.show()

def main():
    # 构建目标函数L(a,b,c,d,e)
    a,b,c,d,e=symbols("a,b,c,d,e")
    X=np.array([[1,2],
                [2,3],
                [3,3],
                [2,1],
                [3,2]])
    y=np.array([[1],[1],[1],[-1],[-1]])
    co=np.array([[a],[b],[c],[d],[e]])
    L=creat(co,X,y,a,b,c,d,e)
    num=np.array([a,b,c,d])
    # 求解极小值点
    global res_list
    res_list=[]
    res=find_min(L,num,a,b,c,d,e)
    # 求w b
    w,b=calculate_w_b(X,y,res)
    print("w",w)
    print("b",b)
    # 绘制样本点、分离超平面和间隔边界
    draw(X,y,w,b)

if __name__=="__main__":
    main()