import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from data_processing.fdm import rk4th, mkc

use_filedata = False
if not use_filedata:
    # Define parameters
    L = 2          # m
    H = 212.25 * 10  # N
    N = 101        # divided into N-1 segments leading to N nodes
    EI = 3928      # Nm2
    m = 11.3       # kg/m

    EI = EI * np.ones(N + 1)
    m = np.ones(N - 1) * m
    EA = 100 * np.ones(N - 1)
    y = np.zeros(N + 1)
    kbc = [0**2, 0**2]

    n = N - 1
    M, K, C = mkc(m, EA, EI, y, H, L, N, 0.005, kbc)

    A = np.zeros((2 * n, 2 * n))
    A[0:n, n:2 * n] = np.eye(n)
    A[n:2 * n, 0:n] = -np.linalg.inv(M) @ K
    A[n:2 * n, n:2 * n] = -np.linalg.inv(M) @ C
    D, V = np.linalg.eig(A)

    n_mode = 10
    # Post processing
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
    ))
    for name, val in variables.items():
        globals()[name] = val


# %%
# Plot results
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t[:100:], Fext.T[:100:], 'o')
plt.subplot(2, 1, 2)
plt.plot(t[:100:], displ[[10, 20, 30, 50], :100:].T)  # plot displacement at specific points
# %%
# Plot all data
plt.figure(2)
x = np.linspace(0, L, N + 1)
T, X = np.meshgrid(t[:100:], x[1:-1])
plt.contourf(X, T, displ[:, :100:], cmap='viridis')
plt.xlabel('$x$ (m)')
plt.ylabel('$t$ (s)')
plt.colorbar(label='$v(x,t)$ (m)')
plt.show()
