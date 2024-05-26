import numpy as np
from tqdm import tqdm


def rk4th(dz, t, z0, Fext):
    dt = t[1] - t[0]
    Nt = len(t)
    Nz = len(z0)

    Z = np.zeros((Nz, Nt))
    dZ = np.zeros((Nz, Nt))
    Z[:, 0] = z0

    for i in tqdm(range(Nt - 1)):
        fext = Fext[:, i]
        k1 = dz(t[i], Z[:, i], fext)
        k2 = dz(t[i] + dt / 2, Z[:, i] + dt / 2 * k1, fext)
        k3 = dz(t[i] + dt / 2, Z[:, i] + dt / 2 * k2, fext)
        k4 = dz(t[i] + dt, Z[:, i] + dt * k3, fext)
        Z[:, i + 1] = Z[:, i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        dZ[:, i] = dz(t[i], Z[:, i], fext)

    t_comp = t
    dZ[:, Nt - 1] = dz(t[Nt - 1], Z[:, Nt - 1], Fext[:, Nt - 1])

    return Z, dZ, t_comp


def fdm(m, EA, EI, y, H, L, N, bc):
    n = N - 1  # number of intermediate mode

    a = L / N
    M = np.diag(m)
    a4 = a**4

    # Boundary conditions
    Q1 = 1 / (2 * a4) * (-5 * EI[2] + 22 * EI[1] - 3 * EI[0]) + 2 * H / a**2
    Q2 = 1 / (2 * a4) * (-3 * EI[2] + 18 * EI[1] - 5 * EI[0]) + 2 * H / a**2
    T1 = 1 / (2 * a4) * (-3 * EI[n+1] + 22 * EI[n] - 5 * EI[n-1]) + 2 * H / a**2
    T2 = 1 / (2 * a4) * (-5 * EI[n+1] + 18 * EI[n] - 3 * EI[n-1]) + 2 * H / a**2

    if bc == 'ff':
        Q, T = Q1, T1
    elif bc == 'fp':
        Q, T = Q1, T2
    elif bc == 'pf':
        Q, T = Q2, T1
    elif bc == 'pp':
        Q, T = Q2, T2
    else:
        raise ValueError('Unknown boundary conditions.')

    K1 = np.zeros((n, n))
    K1[0, 0] = Q
    K1[-1, -1] = T

    for j in range(n):
        D = 1 / a4 * (2 * EI[j+1] - 6 * EI[j+1]) - H / a**2
        U = 1 / a4 * (-6 * EI[j+1] + 2 * EI[j]) - H / a**2
        V = -1 / (2 * a4) * (EI[j+2] - 2 * EI[j+1] - EI[j])
        W = 1 / (2 * a4) * (EI[j+2] + 2 * EI[j+1] - EI[j])

        if j < n - 1:
            K1[j, j+1] = U
        if j < n - 2:
            K1[j, j+2] = W

        if j > 0:
            K1[j, j-1] = D
        if j > 1:
            K1[j, j-1] = D
            K1[j, j-2] = V

        S = 1 / a4 * (-2 * EI[j+2] + 10 * EI[j+1] - 2 * EI[j]) + 2 * H / a**2
        if j > 0 and j < n - 1:
            K1[j, j] = S

    r = np.zeros(n)
    s = np.zeros(n)
    z = ((y[2:] - y[:-2])**2 + 1)**(3/2) / EA

    for j in range(n):
        s[j] = (y[j+2] - 2 * y[j+1] + y[j]) / a**2
        r[j] = (y[j+2] - 2 * y[j+1] + y[j]) / (a**2 * np.sum(z))
    K2 = np.outer(r, s)
    K = K1 + K2

    return M, K


def mkc(m, EA, EI, y, H, L, N, ci, kbc):
    x = np.linspace(0, 1, N+1)
    dx = x[1] - x[0]
    M, K = fdm(m, EA, EI, y, H, L, N, 'pp')
    K2 = np.copy(K)
    K2[0, 0] += kbc[0] / dx**3
    K2[-1, -1] += kbc[1] / dx**3
    C = ci * M
    return M, K2, C

