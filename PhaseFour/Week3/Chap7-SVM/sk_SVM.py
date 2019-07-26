
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

def draw(X,y,w,b):
    y=np.array([y[i] for i in range(y.shape[0])])
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
    X=np.array([[1,2],
                [2,3],
                [3,3],
                [2,1],
                [3,2]])
    y=np.array([1,1,1,-1,-1])
    clf=SVC(C=0.5,kernel="linear")
    clf.fit(X,y)
    w=clf.coef_[0]
    b=clf.intercept_
    print(clf.support_vectors_)
    print(w,b)
    print(clf.predict([[5,6],[-1,-1]]))
    print(clf.score(X,y))
    draw(X,y,w,b)

if __name__=="__main__":
    main()