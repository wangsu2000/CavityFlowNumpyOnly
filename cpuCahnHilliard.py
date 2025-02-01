import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange
import os

def CH2D12(N, T, ep, mu, seed, k):
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

    Outputs:
    -cvecs: (N+1)^2-by-M matrix of lexicographically ordered concentrations
    -t: total time for matrix solutions
    """
    # Initialise
    h = 1 / N

    # Initialial conditions
    cvecs = CH_intial_2D(N, T, k, seed)

    L, V, eigens, wave = eigenPairs(N + 1)
    denominator = np.ones((N + 1, N + 1)) - mu * wave * (-(ep ** 2 / h ** 2) * wave + 2 * np.ones((N + 1, N + 1)))
    w11 = - np.ones((N + 1, N + 1)) / denominator
    w12 = mu * wave / denominator

    # Solve system
    start_time = time.time()
    for n in trange(1, T, desc = "CH-inference"):
        # Load old concentration
        co = -1.0 * cvecs[:(N + 1) ** 2, n - 1].reshape((N + 1, N + 1))
        yo = 3 * co - co ** 3
        co = V.T @ co @ V
        yo = V.T @ yo @ V
        co = V @ (w11 * co + w12 * yo) @ V.T
        cvecs[:, n] = co.flatten()

    t = time.time() - start_time
    return cvecs


def CH2D_Plot_Evolution(cvecs):
    """
    CH2D_Plot_Evolution plots the evolution in time of the concentration
    distribution in a two-dimensional Cahn-Hilliard simulations.

    Input:
    -cvecs: concentration vectors in the format of the output of CH2D11,
    CH2D12, CH2D13, CH2D14, and CH2D15.

    Output:
    -fighand: figure handle to the concentration plot of the evolution.
    """
    # Obtain parameters from cvecs input
    T = cvecs.shape[1]
    Nsq = cvecs.shape[0]
    N = int(np.sqrt(Nsq) - 1)
    foldername = "CahnHilliard_frames"
    os.makedirs(foldername)
    plt.figure()
    for n in range(T):
        Cmat = cvecs[:, n].reshape((N + 1, N + 1))
        plt.pcolormesh(Cmat, shading='gouraud')
        plt.axis('off')
        # plt.pause(0.01)
        plt.colorbar()
        plt.savefig(os.path.join(foldername,"N{:d}n{:d}T{:d}.png".format(N,n,T)))
        
        plt.clf();


def CH_intial_2D(N, T, k, seed):
    """
    CH_initial_2D initialised the solution matrix cvecs with user specified
    initial condition.

    Inputs:
    -N:     number of grid spacings in one-dimension
    -T:     maximum number of time-steps
    -k:     specifies which initial condition
    -seed:  seeds the random number generator

    Output:
    -cvecs:     initialised concentration matrix in the format used by CH2D11,
            CH2D12, CH2D13, CH2D14, CH2D15.
    """
    # Intialise concentration matrix
    cvecs = np.zeros(((N + 1) ** 2, T))
    # Select initial condition
    if k == 1:  # Random initial condition
        np.random.seed(seed)
        cvecs[:, 0] = 2 * np.random.rand((N + 1) ** 2) - 1
    elif k == 2:  # Smooth cosine initial condition
        x = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, x)
        cmat = np.cos(2 * np.pi * X) * np.cos(np.pi * Y)
        cvecs[:, 0] = cmat.flatten()
    return cvecs


def eigenPairs(n):
    L = np.diag(-2 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    L[0, 0] = -1
    L[n - 1, n - 1] = -1
    V = np.zeros((n, n))
    for k in range(n):
        for p in range(1, n+1):
            V[p-1, k] = np.cos(np.pi * k * (2 * p - 1) / (2 * n))
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
  N = 256
  T = 500
  ep = 0.01
  mu = 1
  seed = 0
  k = 1  # random initial condition
  
  # Run one of the schemes
  cvecs = CH2D12(N, T, ep, mu, seed, k)
  
  # Plot the evolution
  CH2D_Plot_Evolution(cvecs)