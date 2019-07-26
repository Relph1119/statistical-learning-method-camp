import numpy as np
import time

def gaussian(y,u,si):
    return 1 / np.sqrt(2 * np.pi * si) * np.exp(-(y - u) ** 2 / (2 * si))

def fun1(y,al,u,si):
    """numpy数组运算"""
    r_jk = al * gaussian(y,u,si)
    return r_jk

def fun2(y,al,u,si):
    """for 循环运算"""
    r_jk=np.zeros((al.size,y.size))
    for j in range(y.size):
        for k in range(al.size):
            r_jk[k][j]=(al[k]*gaussian(y[0][j],u[k],si[k]))[0]
    return r_jk

def main():
    K=2
    Max_iter=10000
    y=np.array([-67,-48,6,8,14,16,23,24,28,29,41,49,56,60,75]).reshape(1,15)
    y_mean=y.mean()//1
    y_std=(y.std()**2)//1
    al=np.full(K,1/K,dtype="float16").reshape(K,1)
    u=np.full(K,y_mean,dtype="float16").reshape(K,1)
    si=np.full(K,y_std,dtype="float16").reshape(K,1)

    star_1=time.time()
    for _ in range(Max_iter):
        res_1=fun1(y,al,u,si)
    end_1=time.time()
    print("numpy 数组用时：{:.2f}s".format(end_1-star_1))

    star_2=time.time()
    for _ in range(Max_iter):
        res_2=fun2(y,al,u,si)
    end_2=time.time()
    print("for 循环用时：{:.2f}s".format(end_2-star_2))

if __name__=="__main__":
    main()