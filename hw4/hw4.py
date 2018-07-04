import numpy as np
import mltools as ml
import matplotlib.pyplot as plt


# (a)
data = np.genfromtxt("data/X_train.txt", delimiter=None)
Xtr = data[0:10000, :]
Xva = data[10000:20000, :]
data = np.genfromtxt("data/Y_train.txt", delimiter=None)
Ytr = data[0:10000]
Yva = data[10000:20000]
Xte = np.genfromtxt("data/X_test.txt", delimiter=None)
learner = ml.dtree.treeClassify(Xtr, Ytr)

# (b)
leaner = ml.dtree.treeClassify(Xtr, Ytr, maxDepth=50)
print('Train: ', leaner.err(Xtr, Ytr))
print('Valid: ', leaner.err(Xva, Yva))


# (c)
train_err = np.zeros(16)
valid_err = np.zeros(16)
for i in range(0, 16):
    learner.train(Xtr, Ytr, maxDepth=i)
    train_err[i] = learner.err(Xtr, Ytr)
    valid_err[i] = learner.err(Xva, Yva)
plt.plot(range(0, 16), train_err, valid_err)
plt.legend(['Train', 'Valid'])
plt.show()

# (d)
train_err = np.zeros(11)
valid_err = np.zeros(11)
for i in range(0, 11):
    learner.train(Xtr, Ytr, maxDepth=50, minLeaf=2**(i+2))
    train_err[i] = learner.err(Xtr, Ytr)
    valid_err[i] = learner.err(Xva, Yva)
plt.plot(range(2, 13), train_err, range(2, 13), valid_err)
plt.legend(['Train', 'Valid'], loc='lower right')
plt.show()

# (e)
train_err = np.zeros(11)
valid_err = np.zeros(11)
for i in range(0, 11):
    learner.train(Xtr, Ytr, minParent=2**(i+3), maxDepth=50)
    train_err[i] = learner.err(Xtr, Ytr)
    valid_err[i] = learner.err(Xva, Yva)
plt.plot(range(3, 14), train_err, range(3, 14), valid_err)
plt.legend(['Train', 'Valid'], loc='lower right')
plt.show()

# # (f)
# learner.train(Xtr, Ytr, minParent=8, maxDepth=14, minLeaf=4)
# fpr, tpr, tnr = learner.roc(Xva, Yva)
# area = learner.auc(Xva, Yva)
# plt.plot(fpr, tpr)
# plt.show()
# print(area)


# best model

data = np.genfromtxt("data/X_train.txt", delimiter=None)
Xtr = data[0:90000, :]
Xva = data[90000:100000, :]
data = np.genfromtxt("data/Y_train.txt", delimiter=None)
Ytr = data[0:90000]
Yva = data[90000:100000]
Xte = np.genfromtxt("data/X_test.txt", delimiter=None)
print(Xte.shape)
learner = ml.dtree.treeClassify(Xtr, Ytr)
#
# dim = 10
# valid_err = [None] * dim
# for i in range(dim):
#     valid_err[i] = [0] * dim
# for i in range(dim):
#     for j in range(dim):
#         valid_err[i][j] = [0] * dim
#
# for i in range(0, 10):
#     for j in range(0, 10):
#         for k in range(0, 10):
#             print(i, j, k)
#             learner.train(Xtr, Ytr, minParent=2**(i+1), maxDepth=2*j, minLeaf=2**k)
#             valid_err[i][j][k] = learner.err(Xva, Yva)
#
# valid_err = np.array(valid_err)
# valid_err = valid_err.flatten()
# print('min: ', min(valid_err))
# for i in range(0, 1000):
#     if valid_err[i] < 0.3:
#         print(i, valid_err[i])

#
# # (g)
# learner.train(Xtr, Ytr, minParent=4, maxDepth=14, minLeaf=4)
# Ypred = learner.predictSoft(Xte)
# print(Ypred.shape)
# # Now output a file with two columns, a row ID and a confidence in class 1:
# np.savetxt('data/Yhat_dtree.txt', np.vstack((np.arange(len(Ypred)), Ypred[:, 1])).T, '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')

# #
# problem 3
#
# (a)
ensemble = [None] * 50

for i in range(0, 50):
    print(i)
    Xtri, Ytri = ml.bootstrapData(Xtr, Ytr)
    ensemble[i] = ml.dtree.treeClassify(Xtri, Ytri, minParent=8, maxDepth=14, minLeaf=4, nFeatures=8)

sizeArray = [1, 5, 10, 25, 50]

Yhat_va = np.zeros(10000)
Yhat_tr = np.zeros(90000)
valid_err = []
train_err = []

for size in sizeArray:
    count = 0
    temp = []
    for j in range(size):
        temp.append(ensemble[j].predict(Xva))
    temp = np.array(temp)
    for i in range(10000):
        if np.mean(temp[:, i]) > 0.5:
            Yhat_va[i] = 1
        else:
            Yhat_va[i] = 0
        if Yhat_va[i] != Yva[i]:
            count += 1
    valid_err.append(count/10000)

    count = 0
    temp = []
    for j in range(size):
        temp.append(ensemble[j].predict(Xtr))
    temp = np.array(temp)
    for i in range(90000):
        if np.mean(temp[:, i]) > 0.5:
            Yhat_tr[i] = 1
        else:
            Yhat_tr[i] = 0
        if Yhat_tr[i] != Ytr[i]:
            count += 1
    train_err.append(count / 90000)

plt.plot(sizeArray, train_err, sizeArray, valid_err)
plt.legend(['Train', 'Valid'])
plt.show()
#
# forest_tr_err = [None] * 4
# forest_va_err = [None] * 4
#
# forest_tr_err[0] = train_err[0]
# forest_va_err[0] = valid_err[0]
# forest_tr_err[1] = np.mean(train_err[0:5])
# forest_va_err[1] = np.mean(valid_err[0:5])
# forest_tr_err[2] = np.mean(train_err[0:10])
# forest_va_err[2] = np.mean(valid_err[0:10])
# forest_tr_err[3] = np.mean(train_err[0:25])
# forest_va_err[3] = np.mean(valid_err[0:25])
#
# num_trees = [1, 5, 10, 25]
# plt.plot(num_trees, forest_tr_err, num_trees, forest_va_err)
# plt.legend(['Train', 'Valid'])
# plt.show()

# # (b)
# size = 50
# ensemble = [None] * size
# for i in range(size):
#     print(i)
#     Xtri, Ytri = ml.bootstrapData(Xtr, Ytr)
#     ensemble[i] = ml.dtree.treeClassify(Xtri, Ytri, maxDepth=15, minLeaf=4, nFeatures=8)
#
# count = 0
# temp = []
# Ypred = np.zeros(Xte.shape[0])
# for i in range(size):
#     temp.append(ensemble[i].predictSoft(Xte))
# print(temp[0].shape)
# temp = np.array(temp)
#
# for i in range(Xte.shape[0]):
#     Ypred[i] = np.mean(temp[:, i, 1])
# print(Ypred.shape)
# # Now output a file with two columns, a row ID and a confidence in class 1:
# np.savetxt('data/Yhat_dtree.txt', np.vstack((np.arange(len(Ypred)), Ypred[:])).T, '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')
