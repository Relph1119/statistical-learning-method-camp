from sklearn.linear_model import Perceptron
import numpy as np

X_train = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])

perceptron = Perceptron()
perceptron.fit(X_train, y)
print("w:", perceptron.coef_, "\n", "b:", perceptron.intercept_, "\n", "n_iter:", perceptron.n_iter_)

res = perceptron.score(X_train, y)
print("correct rate:{:.0%}".format(res))

# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# import numpy as np
#
# X_train = np.array([[3, 3], [4, 3], [1, 1]])
# y = np.array([1, 1, -1])
# #perceptron=Perceptron(penalty="l2",alpha=0.01,eta0=1,max_iter=50,tol=1e-3)
# #perceptron=Perceptron()
# perceptron=SGDClassifier(loss="perceptron",eta0=1, learning_rate="constant", penalty=None)
# perceptron.fit(X_train,y)
# print(perceptron.coef_)
# print(perceptron.intercept_)
# print(perceptron.n_iter_)
# X=np.array([[2,2]])
# y=perceptron.predict(X)
