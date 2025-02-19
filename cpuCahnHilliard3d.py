import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from skimage import measure
from tqdm import trange
import os

# Set Initial Value
h = 1/64
m = 2.5
eps = h * m / (2 * np.sqrt(2) * np.arctanh(0.9))
dt = 0.1 * h
alpha = 0.1

x = np.arange(-0.5 * h, 1 + 0.5 * h, h)
y = x
z = x
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
n_xy = len(x)
n_z = len(z)

mu = np.random.randn(n_xy, n_xy, n_z)
phi = mu.copy()
temp = phi.copy()

os.makedirs("3dCahnHilliard_CUBE", exist_ok = False)

# Gauss-Seidel Iteration
for it in trange(1,2000, desc="outer_loop", leave=True):
    phi_pr = phi.copy()
    for ik in trange(50, desc="inner_loop", leave=False):
        # 构建矩阵A的元素
        a = 1 / dt
        b = 6 / (h ** 2)
        c = -3 * (2 * eps ** 2 / (h ** 2) + phi[1:-1, 1:-1, 1:-1] ** 2)
        d = 1

        # 计算行列式的值
        det_A = a * d - b * c

        # 计算A的逆矩阵
        A_inv = np.zeros((2, 2, n_xy - 2, n_xy - 2, n_z - 2))
        A_inv[0, 0] = d / det_A
        A_inv[0, 1] = -b / det_A
        A_inv[1, 0] = -c / det_A
        A_inv[1, 1] = a / det_A

        # 计算 mu_c 和 phi_c
        mu_c = (mu[2:, 1:-1, 1:-1] + mu[:-2, 1:-1, 1:-1] +
                mu[1:-1, 2:, 1:-1] + mu[1:-1, :-2, 1:-1] +
                mu[1:-1, 1:-1, 2:] + mu[1:-1, 1:-1, :-2]) / (h ** 2)
        phi_c = (phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] +
                 phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] +
                 phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]) / (h ** 2)

        # 计算B矩阵
        B = np.zeros((2, n_xy - 2, n_xy - 2, n_z - 2))
        B[0] = phi_pr[1:-1, 1:-1, 1:-1] / dt + mu_c
        B[1] = -phi_pr[1:-1, 1:-1, 1:-1] - 2 * phi[1:-1, 1:-1, 1:-1] ** 3 - eps ** 2 * phi_c

        # 求解线性方程组 C = A_inv * B
        C = np.einsum('ijklm,jklm->iklm', A_inv, B)

        phi[1:-1, 1:-1, 1:-1] = C[0]
        mu[1:-1, 1:-1, 1:-1] = C[1]

        # mu[0, :, :] = mu[1, :, :]
        # mu[:, 0, :] = mu[:, 1, :]
        # mu[:, :, 0] = mu[:, :, 1]
        # mu[-1, :, :] = mu[-2, :, :]
        # mu[:, -1, :] = mu[:, -2, :]
        # mu[:, :, -1] = mu[:, :, -2]

    if it % 10 == 1:
        temp = phi.copy()
        # 使用 marching_cubes 提取等值面
        verts, faces, _, _ = measure.marching_cubes(temp, 0)
        # 调整顶点坐标到实际的网格坐标
        verts[:, 0] = verts[:, 0] * h - 0.5 * h
        verts[:, 1] = verts[:, 1] * h - 0.5 * h
        verts[:, 2] = verts[:, 2] * h - 0.5 * h

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 设置水平集平面为单一颜色（例如白色）
        surf = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, color='green')

        ax.axis('image')
        ax.set_xlim([np.min(x), np.max(x)])
        ax.set_ylim([np.min(y), np.max(y)])
        ax.set_zlim([np.min(z), np.max(z)])
        ax.set_box_aspect([1, 1, 1])

        plt.savefig("3dCahnHilliard_CUBE/time={:.4f}_.png".format(it * dt))
