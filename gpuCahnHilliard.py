import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange
import os
import cupy as cp
from cupyx.scipy.fft import dctn, idctn

def CH2D12(N, T, ep, mu, seed, k, saveInterval):
    """
    CH2D12 computes the solution two two-dimensional Cahn-Hilliard equation with Neumann
    boundary conditions using scheme (b) in "Numerical Methods for the
    Cahn-Hilliard Equation", by Matthew Geleta.

    Inputs:
    -N:     Number of grid-spacings in one-dimension
    -T:     Number of time-steps
    -ep:    epsilon
    -mu:    mu = dt/dx^2
    -seed:  seed for random number generator
    -k:     case for initial condition
    -saveInterval: Interval for saving the concentration distribution

    Outputs:
    -t: total time for matrix solutions
    """
    # Initialise
    h = 1 / N

    # Initialial conditions
    foldername = "CahnHilliard_frames"
    os.makedirs(foldername, exist_ok=True)
    cvec = CH_intial_2D(N, k, seed)
    Cmat = cp.asnumpy(cvec)
    plt.figure()
    plt.pcolormesh(Cmat, shading='gouraud')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(foldername, f"N{N}0{0//saveInterval}T{T//saveInterval}.png"))
    plt.close()
 
    cvec = cp.asarray(cvec)

    _, _, _, wave = eigenPairs(N + 1)
    wave = cp.asarray(wave)
    denominator = cp.ones((N + 1, N + 1)) - mu * wave * (-(ep ** 2 / h ** 2) * wave + 2 * cp.ones((N + 1, N + 1)))
    w11 = - cp.ones((N + 1, N + 1)) / denominator
    w12 = mu * wave / denominator

    
    

    # Solve system
    start_time = time.time()
    for n in trange(1, T, desc="CH-inference"):
        # Load old concentration
        co = -1.0 * cvec
        yo = 3 * co - co ** 3

        # Use DCT instead of matrix multiplication with V
        co = dctn(co, type=2)  # 修改为 DCT-II 类型
        yo = dctn(yo, type=2)  # 修改为 DCT-II 类型

        co = w11 * co + w12 * yo

        # Use inverse DCT
        co = idctn(co, type=2)  # 修改为 DCT-II 类型

        cvec = co

        if n % saveInterval == 0:
            Cmat = cp.asnumpy(cvec)
            plt.figure()
            plt.pcolormesh(Cmat, shading='gouraud')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(os.path.join(foldername, f"N{N}n{n//saveInterval}T{T//saveInterval}.png"))
            plt.close()

    t = time.time() - start_time
    return t


def CH_intial_2D(N, k, seed):
    """
    CH_initial_2D initialised the solution matrix cvecs with user specified
    initial condition.

    Inputs:
    -N:     number of grid spacings in one-dimension
    -k:     specifies which initial condition
    -seed:  seeds the random number generator

    Output:
    -cvec:     initialised concentration matrix
    """
    # Intialise concentration matrix
    if k == 1:  # Random initial condition
        np.random.seed(seed)
        cvec = 2 * np.random.rand((N + 1) ** 2) - 1
    elif k == 2:  # Smooth cosine initial condition
        x = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, x)
        cmat = np.cos(2 * np.pi * X) * np.cos(np.pi * Y)
        cvec = cmat.flatten()
    elif k == 3:  # Double bubble merging
        col = np.linspace(-1 / 2, 1 / 2, N + 1)
        [x, y] = np.meshgrid(col, col)
        left = np.tanh((0.2 - np.sqrt((x - 0.14) ** 2 + y ** 2)) / 0.01)
        right = np.tanh((0.2 - np.sqrt((x + 0.14) ** 2 + y ** 2)) / 0.01)
        cmat = (left + right) / 2 + np.abs(left - right) / 2
        cvec = cmat.flatten()

    cvec = cvec.reshape((N + 1, N + 1))
    return cvec


def eigenPairs(n):
    L = np.diag(-2 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    L[0, 0] = -1
    L[n - 1, n - 1] = -1
    V = np.zeros((n, n))
    for k in range(n):
        for p in range(1, n + 1):
            V[p - 1, k] = np.cos(np.pi * k * (2 * p - 1) / (2 * n))
    V[:, 0] = V[:, 0] / np.sqrt(2)
    V = V / np.sqrt(n / 2)
    eigens = np.zeros(n)
    for i in range(n):
        eigens[i] = 2 * np.cos(i * np.pi / n) - 2
    wave = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            wave[i, j] = eigens[i] + eigens[j]
    return L, V, eigens, wave


if __name__ == "__main__":
    # Set simulation parameters
    N = 512
    T = 200000
    ep = 0.01
    mu = 1
    seed = 114514
    k = 3  # random initial condition
    saveInterval = 1000
    # Run one of the schemes
    t = CH2D12(N, T, ep, mu, seed, k, saveInterval)
    print(f"Total time for simulation: {t} seconds")