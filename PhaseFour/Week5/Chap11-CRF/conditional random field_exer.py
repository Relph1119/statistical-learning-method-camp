import itertools

A=[[0.0,0.0],[0.5,0.5]]
B=[[0.3,0.7],[0.7,0.3]]
C=[[0.5,0.5],[0.6,0.4]]
D=[[0.0,1.0],[0.0,1.0]]
a=[0,1]
Y=itertools.product(a,repeat=3)
res=[]
for y in Y:
    p=round(A[1][y[0]]*B[y[0]][y[1]]*C[y[1]][y[2]],3)
    print((p,y))
    res.append((p,y))
best_way=max(res,key=lambda x:x[0])
print("最优路径概率：{}\n最优路径：y(1)={},y(2)={},y(3)={}".format(best_way[0],best_way[1][0]+1,best_way[1][1]+1,best_way[1][2]+1))
