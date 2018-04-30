import numpy as np
import matplotlib.pyplot as plt
import mltools as ml


np.random.seed(0)
iris = np.genfromtxt("data/iris.txt", delimiter=None)
Y = iris[:, -1]
X = iris[:, 0:-1]

# 1.(a)
print('Number of features: {0}'.format(X.shape[1]))
print('Number of data points: {0}'.format(X.shape[0]))

# 1.(b)
for i in range(0, 4):
    plt.hist(X[:, i], label='feature{}'.format(i+1))
    plt.legend()
    plt.savefig('figure/figure_1_b_{}'.format(i+1))
    plt.close()

# 1.(c)
for i in range(0, 4):
    print('\nMean of feature {}: {}'.format(i, np.mean(X[:, i])))
    print('Std of future {}: {}'.format(i, np.std(X[:, i])))

# 1.(d)
length = X.shape[0]
featArray = [1, 2, 3]
for featNum in featArray:
    for i in range(0, length):
        if Y[i] == 0:
            plt.scatter(X[i, 0], X[i, featNum], c='b')
        if Y[i] == 1:
            plt.scatter(X[i, 0], X[i, featNum], c='g')
        if Y[i] == 2:
            plt.scatter(X[i, 0], X[i, featNum], c='r')
    plt.xlabel('feature1')
    plt.ylabel('feature{}'.format(featNum+1))
    plt.savefig('figure/figure_1_d_{}-{}'.format(1, featNum+1))
    plt.close()


# 2.(a)

Y = iris[:, -1]
X = iris[:, 0:2]
kArray = [1, 5, 10, 50]
X, Y = ml.shuffleData(X, Y)  # shuffle data randomly
Xtr, Xva, Ytr, Yva = ml.splitData(X, Y, 0.75)  # split data into 75/25 train/validation

# plt.scatter(Xtr, Ytr)
# plt.show()

for k in kArray:
    knn = ml.knn.knnClassify()  # create the object and train it
    knn.train(Xtr, Ytr, k)  # where K is an integer, e.g. 1 for nearest neighbor prediction
    ml.plotClassify2D(knn, Xtr, Ytr)  # make 2D classification plot with data (Xtr,Ytr)
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.savefig('figure/figure_2_a_{}'.format(k))
    plt.close()

# 2.(b)

kArray = [1, 2, 5, 10, 50, 100, 200]
errTrain = []
errValid = []

for k in kArray:
    learner = ml.knn.knnClassify()
    learner.train(Xtr, Ytr, k)  # where K is an integer, e.g. 1 for nearest neighbor prediction

    Ytr_hat = learner.predict(Xtr)
    length = Ytr_hat.shape[0]
    count = 0
    for j in range(0, length):
        if Ytr_hat[j] != Ytr[j]:
            count = count + 1
    errTrain.append(count / length)

    Yva_hat = learner.predict(Xva)
    length = Yva_hat.shape[0]
    count = 0
    for j in range(0, length):
        if Yva_hat[j] != Yva[j]:
            count = count + 1
    errValid.append(count / length)

plt.semilogx(kArray, errTrain, c='r', label="training err")
plt.semilogx(kArray, errValid, c='g', label="validation err")
plt.xlabel('k')
plt.ylabel('error')
plt.legend()
plt.savefig('figure/figure_2_b')
