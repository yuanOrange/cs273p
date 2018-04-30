import numpy as np
import matplotlib.pyplot as plt
import mltools as ml


def po_re(x, d, p):
    return ml.transforms.rescale(ml.transforms.fpoly(x, d, False), p)[0]


# 1.(a)
np.random.seed(0)
data = np.genfromtxt("data/curve80.txt", delimiter=None)
X = data[:, 0]
X = X[:, np.newaxis]  # code expects shape (M,N) so make sure it's 2-dimensional
Y = data[:, 1]  # doesn't matter for Y
Xtr, Xva, Ytr, Yva = ml.splitData(X, Y, 0.75)  # split data set 75/25

# 1.(b)
lr = ml.linear.linearRegress(Xtr, Ytr)  # create and train model
xs = np.linspace(0, 10, 200)  # densely sample possible x-values
xs = xs[:, np.newaxis]  # force "xs" to be an Mx1 matrix
ys = lr.predict(xs)  # make predictions at xs
print('Theta for linear regression:', lr.theta)

plt.scatter(Xtr, Ytr, label='training data')
plt.scatter(Xva, Yva, label='validation data')
plt.plot(xs, ys, c='g', label='degree=1')
plt.legend(loc='lower right')
ax = plt.axis()
plt.show()
# plt.savefig('figure/figure_1_b')
plt.close()

print('\nMSE in Training: ', lr.mse(Xtr, Ytr))
print('MSE in Validation: ', lr.mse(Xva, Yva))

# 1.(c)

# degree = 2

XtrP = ml.transforms.fpoly(Xtr, 2, bias=False)
XtrP, params = ml.transforms.rescale(XtrP)
lrP = ml.linear.linearRegress(XtrP, Ytr)

xsP = po_re(xs, 2, params)
ysP = lrP.predict(xsP)  # make predictions at xs
print('\nTheta for quadratic regression:', lrP.theta)

plt.scatter(Xtr, Ytr, label='training data')
plt.scatter(Xva, Yva, label='validation data')
plt.plot(xs, ysP, c='g', label='degree=2')
plt.legend(loc='lower right')
plt.axis(ax)
plt.show()
# plt.savefig('figure/figure_1_c_2')
plt.close()

XvaP = po_re(Xva, 2, params)

print('\nMSE in Training: ', lrP.mse(XtrP, Ytr))
print('MSE in Validation: ', lrP.mse(XvaP, Yva))


# multi degree
degreeArray = [1, 3, 5, 7, 10, 18]
mseTr = []
mseVa = []
for degree in degreeArray:
    XtrP = ml.transforms.fpoly(Xtr, degree, bias=False)
    XtrP, params = ml.transforms.rescale(XtrP)
    lrP = ml.linear.linearRegress(XtrP, Ytr)

    XvaP = po_re(Xva, degree, params)
    xsP = po_re(xs, degree, params)
    ysP = lrP.predict(xsP)

    plt.scatter(Xtr, Ytr, label='training data')
    plt.scatter(Xva, Yva, label='validation data')
    plt.plot(xs, ysP, c='g', label='degree={}'.format(degree))
    plt.legend(loc='lower right')
    plt.axis(ax)
    plt.show()
    # plt.savefig('figure/figure_1_c_{}'.format(degree))
    plt.close()
    err1 = lrP.mse(XtrP, Ytr)
    err2 = lrP.mse(XvaP, Yva)
    print('\nMSE in training when degree={}: {}'.format(degree, err1))
    print('MSE in validation when degree={}: {}'.format(degree, err2))
    mseTr.append(err1)
    mseVa.append(err2)

plt.semilogy(degreeArray, mseTr, 'o-', label='Training Err')
plt.semilogy(degreeArray, mseVa, 'o-', label='Validation Err')
plt.legend(loc='upper left')
plt.show()
# plt.savefig('figure/figure_1_c_error')
plt.close()

# 2
nFolds = 5
J = [None] * nFolds
K = [None] * nFolds
errDegTr = []
errDegVa = []

for degree in degreeArray:

    for iFold in range(nFolds):

        Xti, Xvi, Yti, Yvi = ml.crossValidate(Xtr, Ytr, nFolds, iFold)

        XtiP = ml.transforms.fpoly(Xti, degree, bias=False)
        XtiP, params = ml.transforms.rescale(XtiP)
        lrP = ml.linear.linearRegress(XtiP, Yti)

        XviP = po_re(Xvi, degree, params)

        J[iFold] = lrP.mse(XtiP, Yti)
        K[iFold] = lrP.mse(XviP, Yvi)

    errDegTr.append(np.mean(J))
    errDegVa.append(np.mean(K))

print('\nMSE in training with cross', errDegTr)
print('MSE in validation with cross', errDegVa)

plt.semilogy(degreeArray, errDegTr, 'o-', label='training')
plt.semilogy(degreeArray, errDegVa, 'o-', label='validation')
plt.legend(loc='upper left')
plt.show()
# plt.savefig('figure/figure_2_c_error')
plt.close()
