import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from logisticClassify2 import *

# Problem 1

iris = np.genfromtxt('data/iris.txt', delimiter=None)
X, Y = iris[:, 0:2], iris[:, -1]  # get first two features & target
X, Y = ml.shuffleData(X, Y)  # reorder randomly (important later)
X, _ = ml.transforms.rescale(X)
XA, YA = X[Y < 2, :], Y[Y < 2]
XB, YB = X[Y > 0, :], Y[Y > 0]

# (a)
plt.scatter(XA[YA == 0, 0], XA[YA == 0, 1], label='Y=0')
plt.scatter(XA[YA == 1, 0], XA[YA == 1, 1], label='Y=1')
plt.legend()
plt.show()
plt.scatter(XB[YB == 1, 0], XB[YB == 1, 1], label='Y=1')
plt.scatter(XB[YB == 2, 0], XB[YB == 2, 1], label='Y=2')
plt.legend()
plt.show()

# (b)
lr_a = logisticClassify2()
lr_a.classes = np.unique(YA)
wts = np.array([0.5, 1, -0.25])
lr_a.theta = wts
lr_a.plotBoundary(XA, YA)
# learner.plotBoundary(XB, YB)

# (c)
count_a = 0
y_pred_a = lr_a.predict(XA)
for i in range(len(y_pred_a)):
    if y_pred_a[i] != YA[i]:
        count_a += 1
print('error for A set', count_a/len(y_pred_a))

count_b = 0
lr_b = logisticClassify2()
lr_b.classes = np.unique(YB)
lr_b.theta = wts
y_pred_b = lr_b.predict(XB)
for i in range(len(y_pred_b)):
    if y_pred_b[i] != YB[i]:
        count_b += 1
print('error for B set', count_b/len(y_pred_b))

# (d)
ml.plotClassify2D(lr_a, X, Y)
plt.show()

# (e)

# (f)

# (g)
lr_a.train(XA, YA)
plt.close()
ml.plotClassify2D(lr_a, X, Y)
plt.show()
lr_b.train(XB, YB)
plt.close()
ml.plotClassify2D(lr_b, X, Y)
plt.show()

# (h)
iArray = [0.5, 1, 2, 3, 4]
for i in iArray:
    lr_a.trainL2(XA, YA, alpha=i)
for i in iArray:
    lr_b.trainL2(XB, YB, alpha=i)


# problem 2
