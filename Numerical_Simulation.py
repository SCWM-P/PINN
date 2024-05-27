import sys
import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_processing.fdm import rk4th, mkc
try:
    matplotlib.use('TkAgg')
    plt.ion()
except Exception as e:
    warnings.warn(e.msg, UserWarning)

use_filedata = True
if not use_filedata:
    # 定义物理参数和初始条件
    L = 2.0  # 总长度
    H = 2122.5  # 横向力
    N = 101  # 分段数
    EI = 3928.0  # 弯曲刚度
    m = 11.3  # 单位长度质量

    # 构造一维数组
    EI = np.ones(N + 1) * EI
    m = np.ones(N - 1) * m
    EA = 100 * np.ones(N - 1)  # 轴向刚度
    y = np.zeros(N + 1)  # 初始位移
    kbc = np.array([0.0, 0.0])  # 边界条件参数

    n = N - 1
    M, K, C = mkc(m, EA, EI, y, H, L, N, 0.005, kbc)

    A = np.zeros((2 * n, 2 * n))
    A[:n, n:2 * n] = np.eye(n)
    A[n:2 * n, :n] = -np.linalg.inv(M) @ K
    A[n:2 * n, n:2 * n] = -np.linalg.inv(M) @ C
    D, V = np.linalg.eig(A)

    n_mode = 10
    # Post-processing
    d = np.diag(D)
    index = np.imag(d) > 0
    omega = d[index]
    V = V[index]
    sorted_indices = np.argsort(np.imag(omega))
    omega = omega[sorted_indices]
    V = V[sorted_indices]
    freq = np.imag(omega[:n_mode]) / (2 * np.pi)

    # Time integration
    t = np.arange(0, 2, 0.00001)  # define time
    # f = np.random.randn(len(t)) * 0.001  # random force
    f = 1e3 * np.sin(2 * np.pi * freq[0] * t)  # periodic force
    w = np.zeros(n)  # force distribution on the cable
    w[n // 2] = 1
    Fext = np.outer(w, f)

    z0 = np.zeros(2 * n)  # initialization
    # z0[100] = 1

    G = np.zeros((2 * n, n))
    G[n:2 * n, :] = np.linalg.inv(M)


    def dz(t, z, f):
        return A @ z + G @ f


    Z, dZ, t_comp = rk4th(dz, t, z0, Fext)
    displ = Z[:n, :]
    acc = dZ[n:2 * n, :]
    T, X = np.meshgrid(np.linspace(0, L, N - 1), t)

    variables = {
        name: val for name, val in globals().items()
        if not name.startswith('_') and not
        callable(val) and not
            isinstance(
                val,
                type(sys)
            )
    }
    np.savez(
        os.path.join(
            'data', 'npy',
            'variables.npz'),
        **variables
    )
else:
    current_path = os.getcwd()
    variables = np.load(os.path.join(
        'data', 'npy',
        'variables.npz'
    ), allow_pickle=True)
    for name, val in variables.items():
        globals()[name] = val


def draw():
    # %%
    alpha = 100
    # Plot results
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t[::alpha], Fext.T[::alpha], '.')
    plt.subplot(2, 1, 2)
    plt.plot(t[::alpha], displ[[10, 20, 30, 50], :].T[::alpha])  # plot displacement at specific points
    # %%
    # Plot all data
    ax = plt.figure(2).add_subplot(111, projection='3d')
    X, T = np.meshgrid(np.linspace(0, L, N - 1), t[::alpha])
    surf = ax.plot_surface(T, X, displ.T[::alpha, :], cmap='viridis')
    ax.set_title('3D Surface plot of Displacement over Time and Length')
    ax.set_xlabel('Time $t$[s]')
    ax.set_ylabel('Position along beam $x$[m]')
    ax.set_zlabel('Displacement $u$[m]')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show(block=False)


draw()
