import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import scipy as sp

# # (a)
# iris = np.genfromtxt("data/iris.txt", delimiter=None)
# X = iris[:, 0:2]
# #
# # plt.scatter(X[:, 0], X[:, 1])
# # plt.show()
#
# # (b)
# k = 5
# z, _, _ = ml.cluster.kmeans(X, k)
# ml.plotClassify2D(None, X, z)
# plt.show()
#
# k = 20
# z, _, _ = ml.cluster.kmeans(X, k)
# ml.plotClassify2D(None, X, z)
# plt.show()
#
# sumd = []
# for i in range(5):
#     _, _, sumdi = ml.cluster.kmeans(X, 20)
#     sumd.append(sumdi)
# print(sumd)

# ## (c)
# k_arr = [5, 20]
# m_arr = ['min', 'max']
# for k in k_arr:
#     for method in m_arr:
#         z, _ = ml.cluster.agglomerative(X, k, method=method)
#         ml.plotClassify2D(None, X, z)
#         plt.title('{} & {}'.format(k, method))
#         plt.show()
# z, _ = ml.cluster.agglomerative(X, 5, method='min')
# ml.plotClassify2D(None, X, z)
# plt.show()
#
# z, _ = ml.cluster.agglomerative(X, 20, method='min')
# ml.plotClassify2D(None, X, z)
# plt.show()
#
# z, _ = ml.cluster.agglomerative(X, 5, method='max')
# ml.plotClassify2D(None, X, z)
# plt.show()
#
# z, _ = ml.cluster.agglomerative(X, 20, method='max')
# ml.plotClassify2D(None, X, z)
# plt.show()

# # (d)
# z, _, _, _ = ml.cluster.gmmEM(X, 5)
# ml.plotClassify2D(None, X, z)
# plt.show()

#
# 2
X = np.genfromtxt('data/faces.txt', delimiter=None)
img = np.reshape(X[1, :], (24, 24))
plt.imshow(img.T, cmap='gray')
plt.show()
#
# (a)
mu = np.mean(X)
print(mu)
#
# # (b) (c)
# U, S, Vh = np.linalg.svd(X0, False)
# W = U.dot(np.diag(S))
# # X0_hat = []
# # mse = []
# # for K in range(1, 11):
# #     X0_hat.append(W[:, :K].dot(Vh[:K, :]))
# #     mse.append(np.mean((X0 - X0_hat) ** 2))
# # plt.plot(range(1, 11), mse)
# # plt.show()
#
# # # (d)
# # for j in range(0, 3):
# #     alpha = 2 * np.mean(np.abs(W[:, j]))
# #     x = mu + alpha * Vh[j, :]
# #     img = np.reshape(x, (24, 24))
# #     # plt.subplot(1, 2, 1)
# #     plt.imshow(img.T, cmap='gray')
# #     plt.show()
# #
# #     x = mu - alpha * Vh[j, :]
# #     img = np.reshape(x, (24, 24))
# #     # plt.subplot(1, 2, 2)
# #     plt.imshow(img.T, cmap='gray')
# #     plt.show()
#
# # (e)
# for i in range(0, 2):
#     for K in [5, 10, 50, 100]:
#         x = mu + W[i, :K].dot(Vh[:K, :])
#         img = np.reshape(x, (24, 24))
#         # plt.subplot(1, 2, 1)
#         plt.imshow(img.T, cmap='gray')
#         plt.show()
#
#         # plt.subplot(1, 2, 2)
#         img = np.reshape(X0[i, :], (24, 24))
#         plt.imshow(img.T, cmap='gray')
#         plt.show()

# # (f)
# idx = range(17, 27)
# coord, params = ml.transforms.rescale(W[:, 0:2])
# plt.figure()
# plt.hold(True)
# for i in idx:
#     loc = (coord[i, 0], coord[i, 0] + 0.5, coord[i, 1], coord[i, 1] + 0.5)
#     img = np.reshape(X[i, :], (24, 24))
#     plt.imshow(img.T, cmap='gray', extent=loc)
#     plt.axis((-2, 2, -2, 2))
# plt.show()
