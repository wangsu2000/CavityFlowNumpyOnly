import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import trange
import os
def batchDST1(a):
    # Apply DST-I in each column
    B,N1,N2 = a.shape
    xx = cp.concatenate([
        cp.zeros((B, 1 ,N2)),
        a,
        cp.zeros((B, 1, N2)),
        cp.zeros((B, N1, N2))
    ],axis = 1)  # DST1
    tmp = -cp.fft.fft(xx, axis=1) * cp.sqrt(2 / (N1 + 1))
    b = cp.imag(tmp[:,1:N1 + 1, :])
    return b
def batchDST2(a):
    B,N1,N2 = a.shape
    xx = cp.concatenate([a,cp.zeros_like(a)],axis = 1)
    tmp = cp.fft.fft(xx, axis=1)
    xd = tmp[:, 1:N1 + 1, :]
    ww = cp.sqrt(2 / N1) * cp.exp(-1j * cp.pi * cp.arange(1, N1 + 1) / (2 * N1))[:, cp.newaxis]
    b = xd * cp.expand_dims(ww,axis = 0)
    b[:,-1, :] = b[:,-1, :] / cp.sqrt(2)
    b = -cp.imag(b)
    return b
def batchDST3(a):
    # Apply DST-III in each column
    B, N1, N2 = a.shape
    aa = deepcopy(a)
    aa[:, N1 - 1, :] = aa[:, N1 - 1, :] / cp.sqrt(2)
    xx = cp.concatenate([
        cp.zeros((B, 1, N2)),
        aa,
        cp.zeros((B, N1 - 1, N2))
    ],axis = 1)  # DST1
    # weight
    ww = cp.exp(-1j * cp.pi * cp.arange(0, 2 * N1) / (2 * N1))[:, cp.newaxis]
    xxw = xx * cp.expand_dims(ww,axis = 0)
    tmp =  cp.fft.fft(xxw, axis=1) * cp.sqrt(2 / N1)
    b = -cp.imag(tmp[:, :N1, :])
    return b
def BatchU_xyCNAB_dst(u, b, dt, Re, nx, ny, dx, dy):
    kx = cp.arange(1, nx)
    ax = cp.pi * kx / nx
    mwx = 2 * (cp.cos(ax) - 1) / dx**2  # DST-I

    ky = cp.arange(1, ny + 1)
    ay = cp.pi * ky / ny
    mwy = 2 * (cp.cos(ay) - 1) / dy**2  # DST-II

    mw = cp.expand_dims(mwx[:, cp.newaxis] + mwy,axis = 0)  # Modified Wavenumber
    # import pdb;pdb.set_trace();
    # rhshat = batchDST2(batchDST1(b).T.conj()).T.conj()
    
    rhshat = cp.transpose(batchDST2(cp.transpose(batchDST1(b).conj(),(0,2,1))),(0,2,1)).conj()
    # rhshat = np.transpose(batchDST2(np.transpose(batchDST1(b),(0,2,1)).conj()),(0,2,1)).conj()
    uhat = rhshat / (1 - (1 / 2) * dt / Re * mw)

    # xu = batchDST1(batchDST3(uhat.T.conj()).T.conj())
    xu = batchDST1(cp.transpose(batchDST3(cp.transpose(uhat.conj(),(0,2,1))),(0,2,1)).conj())
    return xu

def BatchV_xyCNAB_dst(v, b, dt, Re, nx, ny, dx, dy):
    kx = cp.arange(1, nx + 1)
    ax = cp.pi * kx / nx
    mwx = 2 * (cp.cos(ax) - 1) / dx**2  # DST-I

    ky = cp.arange(1, ny)
    ay = cp.pi * ky / ny
    mwy = 2 * (cp.cos(ay) - 1) / dy**2  # DST-II

    mw = cp.expand_dims(mwx[:, cp.newaxis] + mwy,axis = 0)  # Modified Wavenumber

    # rhshat = batchDST1(batchDST2(b).T.conj()).T.conj()
    rhshat = cp.transpose(batchDST1(cp.transpose(batchDST2(b).conj(),(0,2,1))),(0,2,1)).conj()
    
    vhat = rhshat / (1 - (1 / 2) * dt / Re * mw)

    # xv = batchDST3(batchDST1(vhat.T.conj()).T.conj())
    xv = batchDST3(cp.transpose(batchDST1(cp.transpose(vhat.conj(),(0,2,1))),(0,2,1)).conj())
    return xv

def BatchupdateVelocityField_CNAB_bctop(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, bctop):
  

   

    # Apply boundary conditions:
    u[:, :, 0] = -u[:, :, 1]             # bottom
    v[:, :, 0] = 0.                   # bottom
    u[:, :, -1] = 2 * bctop - u[:, :, -2]  # top
    v[:, :, -1] = 0.                  # top
    u[:, 0, :] = 0.                   # left
    v[:, 0, :] = -v[:, 1, :]             # left
    u[:, -1, :] = 0.                  # right
    v[:, -1, :] = -v[:, -2, :]           # right

    # Get viscous terms for u
    Lux = (u[:, :-2, 1:-1] - 2 * u[:, 1:-1, 1:-1] + u[:, 2:, 1:-1]) / dx**2
    Luy = (u[:, 1:-1, :-2] - 2 * u[:, 1:-1, 1:-1] + u[:, 1:-1, 2:]) / dy**2

    # Get viscous terms for v
    Lvx = (v[:, :-2, 1:-1] - 2 * v[:, 1:-1, 1:-1] + v[:, 2:, 1:-1]) / dx**2
    Lvy = (v[:, 1:-1, :-2] - 2 * v[:, 1:-1, 1:-1] + v[:, 1:-1, 2:]) / dy**2

    # Get nonlinear terms
    # 1. interpolate velocity at cell center/cell corner
    uce = (u[:, :-1, 1:-1] + u[:, 1:, 1:-1]) / 2
    uco = (u[:, :, :-1] + u[:, :, 1:]) / 2
    vco = (v[:, :-1, :] + v[:, 1:, :]) / 2
    vce = (v[:, 1:-1, :-1] + v[:, 1:-1, 1:]) / 2

    # 2. multiply
    uuce = uce * uce
    uvco = uco * vco
    vvce = vce * vce

    # 3-1. get derivative for u
    Nu = (uuce[:, 1:, :] - uuce[:, :-1, :]) / dx
    Nu += (uvco[:, 1:-1, 1:] - uvco[:, 1:-1, :-1]) / dy

    # 3-2. get derivative for v
    Nv = (vvce[:, :, 1:] - vvce[:, :, :-1]) / dy
    Nv += (uvco[:, 1:, 1:-1] - uvco[:, :-1, 1:-1]) / dx

    # Implicit treatment for xy direction
    Lubc = cp.zeros_like(Luy)
    Lubc[:, :, -1] = 2 * bctop / dy**2  # effect of the top BC on Lu
    Lvbc = cp.zeros_like(Lvy)

    b_u = u[:, 1:-1, 1:-1] - dt * ((3 * Nu - Nu_old) / 2 - 1 / (2 * Re) * (Lux + Luy + Lubc))
    xu = BatchU_xyCNAB_dst(u, b_u, dt, Re, nx, ny, dx, dy)
    b_v = v[:, 1:-1, 1:-1] - dt * ((3 * Nv - Nv_old) / 2 - 1 / (2 * Re) * (Lvx + Lvy + Lvbc))
    xv = BatchV_xyCNAB_dst(v, b_v, dt, Re, nx, ny, dx, dy)

    Nu_old = Nu
    Nv_old = Nv
    
    # print(u.shape,xv.shape)
    u[:, 1:-1, 1:-1] = xu
    v[:, 1:-1, 1:-1] = xv

    # RHS of pressure Poisson eq.
    b = ((u[:, 1:, 1:-1] - u[:, :-1, 1:-1]) / dx +
         (v[:, 1:-1, 1:] - v[:, 1:-1, :-1]) / dy)

    # Solve for p

    # dp = np.expand_dims(solvePoissonEquation_2dDCT(b[0,:,:],dx),axis = 0)
    dp = BatchPoissonEquation_2dDCT(b, dx)
    # dp1 = solvePoissonEquation_2dDCT(b[0,:,:],dx)
    # dp2 = solvePoissonEquation_2dDCT(b[1,:,:],dx)
    # dp = np.concatenate([np.expand_dims(dp1,axis = 0),np.expand_dims(dp2,axis = 0)])

    # Correction to get the final velocity
    p = dp
    u[:, 1:-1, 1:-1] -= (p[:, 1:, :] - p[:, :-1, :]) / dx
    v[:, 1:-1, 1:-1] -= (p[:, :, 1:] - p[:, :, :-1]) / dy

    return u, v, p


def BatchPoissonEquation_2dDCT(b,h = 1.0):
        bs, m, n = b.shape
        row = n * cp.cos(cp.pi * cp.arange(0, n) / n)
        row[0] = row[0] + n
        col = m / 2 * cp.ones(m)
        col[0] = col[0] + m / 2
        rc = cp.outer(row, col)
        row = n / 2 * cp.ones(n)
        row[0] = row[0] + n / 2
        col = m * cp.cos(cp.pi * cp.arange(0, m) / m) - 2 * m
        col[0] = col[0] - m
        rc = rc + cp.outer(row, col)
        rc = cp.expand_dims(rc,axis = 0).astype(cp.float64)
        y1 = cp.fft.fft(cp.concatenate([h*h*b, cp.zeros_like(b, dtype=cp.complex128)], axis=1),axis = 1)
        y1 = cp.real(y1[:,:m, :] * cp.expand_dims((cp.cos(cp.arange(0, m).reshape(-1, 1) * cp.pi / (2 * m)) -\
                                                 cp.sin(cp.arange(0, m).reshape(-1, 1) * cp.pi / (2 * m)) * 1j).reshape(-1,1),axis = 0))
        y1 = cp.fft.fft(cp.concatenate([y1.transpose(0,2,1).conj(), cp.zeros_like(y1.transpose(0,2,1), dtype=cp.complex128)], axis=1),axis = 1)
        y1 = cp.real(y1[:,:n, :] * cp.expand_dims((cp.cos(cp.arange(0, n).reshape(-1, 1) * cp.pi / (2 * n)) -\
                                                   cp.sin(cp.arange(0, n).reshape(-1, 1) * cp.pi / (2 * n)) * 1j).reshape(-1,1),axis = 0))
        y1[:, 0, 0] = 0
        y1 = (y1 / rc).transpose(0,2,1).conj()
        y1[: ,0, 0] = 0

        y2 = cp.real( cp.fft.fft(cp.concatenate([ cp.expand_dims(((cp.cos(cp.arange(0, m).reshape(-1, 1) * cp.pi / (2 * m)) ) -\
                                                   (cp.sin(cp.arange(0, m).reshape(-1, 1) * cp.pi / (2 * m))) * 1j),axis = 0)*y1,\
                                                 cp.zeros((bs, m, n))], axis=1),axis = 1))
        y2 = y2[:,:m,:].transpose(0,2,1).conj()
        y2 = cp.real( cp.fft.fft(cp.concatenate([cp.expand_dims((cp.cos(cp.arange(0, n).reshape(-1, 1) * cp.pi / (2 * n)) ),axis = 0)*y2 -\
                                                 cp.expand_dims((cp.sin(cp.arange(0, n).reshape(-1, 1) * cp.pi / (2 * n)) ),axis = 0)*y2*1j,\
                                                 cp.zeros((bs, n, m))], axis=1),axis = 1))
        return   y2[:,:n, :].transpose(0,2,1).conj()

def explicit_velocity(u,v):
# 利用插值还原网格中点处的流速:
    uce = 0.5 * (u[0:-1, 1:-1] + u[1:, 1:-1])
    vce = 0.5 * (v[1:-1, 0:-1] + v[1:-1, 1:])
    return uce,vce

def streamlinePlot(Xce,Yce,u,v,save_path):
    uce,vce = explicit_velocity(u,v)
    # import pdb;pdb.set_trace()
    plt.streamplot(Xce, Yce,uce.T,vce.T, density=1, linewidth=1, color='blue')
    plt.xlim([0, Lx])
    plt.ylim([0, Ly])
#     plt.title('Re = {:.1f},time = {:4f}'.format(Re,ii*dt))
#     save_path = os.path.join(root_path,str(image_name)+".jpg")
    # plt.show()
    if save_path:
        plt.savefig(save_path)
        plt.clf()
    
if __name__ == "__main__":
    # tensor = np.stack([np.random.randn(234,243),np.random.randn(234,243)])
    # dst1 = batchDST1(tensor) - np.stack([mydst1(tensor[0,:,:]),mydst1(tensor[1,:,:])])
    # print(np.linalg.norm(dst1))
    # dst2 = batchDST2(tensor) - np.stack([mydst2(tensor[0,:,:]),mydst2(tensor[1,:,:])])
    # print(np.linalg.norm(dst2))
    # dst3 = batchDST3(tensor.copy()) - np.stack([mydst3(tensor[0,:,:]),mydst3(tensor[1,:,:])])
    # print(np.linalg.norm(dst3))
    inference_root = "cavityFlow_inference"
    os.makedirs(inference_root,exist_ok = True)
    recordRate = 250
    Re = cp.expand_dims(cp.round(cp.logspace(2,4,3)).reshape(-1,1),axis = 2)  # Reynolds number 100,1000,10000
    for re in range(Re.shape[0]):
        os.makedirs(os.path.join(inference_root,str(int(Re[re,0,0]))),exist_ok = True)
    
    nt = 200000  # max time steps
    Lx, Ly = 1, 1  # domain size
    Nx, Ny = 256, 256  # Number of grids
    dt = 0.0001  # time step
    
    # Grid size (Equispaced)
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Coordinate of each grid (cell center)
    xce = (cp.arange(1, Nx+1) - 0.5) * dx
    yce = (cp.arange(1, Ny+1) - 0.5) * dy


    Nu_old = cp.zeros((Re.shape[0], Nx-1, Ny))
    Nv_old = cp.zeros((Re.shape[0], Nx, Ny-1))
    
    # Coordinate of each grid (cell corner)
    xco = cp.arange(Nx+1) * dx
    yco = cp.arange(Ny+1) * dy
    
    # Initialize velocity fields
    u = cp.zeros((Re.shape[0], Nx+1, Ny+2))  # velocity in x direction (u)
    v = cp.zeros((Re.shape[0], Nx+2, Ny+1))  # velocity in y direction (v)
    uce = (u[:, :-1, 1:-1] + u[:, 1:, 1:-1]) / 2  # u at cell center
    vce = (v[:, 1:-1, :-1] + v[:, 1:-1, 1:]) / 2  # v at cell center

    for ii in trange(1, nt+1,desc = "cavityFlow"):
        bctop = 1  # top velocity
    
        # Update the velocity field (uses dct)
        u, v, p = BatchupdateVelocityField_CNAB_bctop(u, v,Nu_old,Nv_old, Nx, Ny, dx, dy, Re, dt, bctop)  # 选用第一种方法:Crank-Nicolson格式
        tb = (u[:, 1:, 1:-1] - u[:, 0:-1, 1:-1]) / dx + (v[:, 1:-1, 1:] - v[:, 1:-1, 0:-1]) / dy
        
        # 
    
        # Update the plot at every recordRate steps
        if ii % recordRate == 0:
            # get velocity at the cell center (for visualization)
            print("time = {:.4f},".format(ii*dt),end = ";")
            print("质量守恒律检测:{:.16f}".format(cp.linalg.norm(tb)))
            xce = (np.arange(1, Nx + 1) - 0.5) * dx
            yce = (np.arange(1, Ny + 1) - 0.5) * dy
            Xce, Yce = np.meshgrid(xce, yce)
            for jj in trange(Re.shape[0],desc = "save_state_dict",leave = False):
                streamlinePlot(Xce,Yce,u[jj,:,:].get(),v[jj,:,:].get(),os.path.join(inference_root,str(int(Re[jj,0,0])),"T = {:.6f}.png".format(ii*dt)))
