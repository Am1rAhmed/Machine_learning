''' traing a lofistic regression classifier to
predict whether a flower is iris or not '''

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()


X = iris["data"][:,3:]  #--->as a feature
Y = (iris['target']==2).astype(np.int64)

clf = LogisticRegression()
clf.fit(X,Y)
eg = clf.predict(([[2.6]])) 
print(eg)

'''using matplotlib to plot the visualization'''
# X_new = np.linspace(0,3,1000).reshape(-1,1)
# Y_prob = clf.predict_proba(X_new)
# plt.plot(X_new,Y_prob[:,1],"g-",label="virginica")
# plt.show()

# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])
# print(Y)
# print(X)

