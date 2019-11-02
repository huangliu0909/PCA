import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

N = 50
Data = ([random.randint(500, 1000) for _ in range(N)], [random.randint(500, 1000) for _ in range(N)], [random.randint(0, 1) for _ in range(N)])
t_data = (np.transpose(Data))
np.random.shuffle(t_data)
# print(t_data.shape[1])
t_data_norm = np.zeros((N, 3))
n = t_data.shape[1]
mu = np.mean(t_data, axis=0)  # axis=0表示列
sigma = np.std(t_data, axis=0)
# print(sigma)
for i in range(n):
    t_data_norm[:, i] = (t_data[:, i] - mu[i]) / sigma[i]

# t_data_norm, mu, sigma = normalize(t_data)
Sigma = np.dot(np.transpose(t_data_norm), t_data_norm) / t_data.shape[0]  # 求Sigma
print(t_data.shape[0])
U, S, V = np.linalg.svd(Sigma)  # 求Sigma的奇异值分解
Z = np.zeros((N, 2))
u = U[0:2, :]  # 取前K个
Z = np.dot(t_data_norm, u.T)  # 投影
print(u)
# x = np.zeros((Z.shape[0], U.shape[0]))
x = np.dot(Z, u)  # 还原数据（近似）

print(x)
fig = plt.figure()
ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')
ax.plot(t_data_norm[:, 0], t_data_norm[:, 1], t_data_norm[:, 2], 'bo')
ax.plot_trisurf(x[:, 0], x[:, 1], x[:, 2], color='#FF7F50')
for i in range(t_data_norm.shape[0]):
    ax.plot(np.array([t_data_norm[i, 0], x[i, 0]]), np.array([t_data_norm[i, 1], x[i, 1]]), np.array([t_data_norm[i, 2], x[i, 2]]), '--k')
plt.show()
