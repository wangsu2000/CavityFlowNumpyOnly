
import numpy as np
import cupy as cp
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import trange
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def BatchupdateVelocityField_CNAB_bctop(u, v, nx, ny, dx, dy, Re, dt, bctop, obstacle_mask, niter = 5):
    
   
    
    # Apply boundary conditions:
    u[:,:, 0] = -u[:,:, 1]             # bottom
    v[:,:, 0] = 0.                   # bottom
    u[:,:, -1] = - u[:,:, -2]  # top
    v[:,:, -1] = 0.                  # top
    u[:,0, :] = bctop         # left
    v[:,0, :] = -v[:,1, :]             # left
    u[:,-1, :] = bctop             # right
    v[:,-1, :] = -v[:,-2, :]           # right

    # niter  ? ?         
    uold, vold = u.copy(), v.copy()

    # obstacle_effect

    # obstacle_mask = obstacle_mask_calibration(u, v)
    u = u * obstacle_mask["u"]
    v = v * obstacle_mask["v"]

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
    # Lubc = cp.zeros_like(Luy)
    # Lubc[:, -1] = 2 * bctop / dy**2  # effect of the top BC on Lu
    # Lvbc = cp.zeros_like(Lvy)

    b_u = uold[:,1:-1, 1:-1] - dt * (Nu  - 1 / (2 * Re) * (Lux + Luy))
    xu = BatchU_xyCNAB_dst(uold, b_u, dt, Re, nx, ny, dx, dy)
    b_v = vold[:,1:-1, 1:-1] - dt * (Nv  - 1 / (2 * Re) * (Lvx + Lvy))
    xv = BatchV_xyCNAB_dst(vold, b_v, dt, Re, nx, ny, dx, dy)

    # print(xu.shape,xv.shape,u.shape,v.shape)
    u[:,1:-1, 1:-1] = xu
    v[:,1:-1, 1:-1] = xv

    # obstacle calibration
    # u,v = obstacle_mask_calibration(obstacle_mask, u, v)
    u = u * obstacle_mask["u"]
    v = v * obstacle_mask["v"]

    # RHS of pressure Poisson eq.
    b = ((u[:, 1:, 1:-1] - u[:, :-1, 1:-1]) / dx +
         (v[:, 1:-1, 1:] - v[:, 1:-1, :-1]) / dy)

    # Solve for p

    dp = BatchPoissonEquation_2dDCT(b, dx)
    

    # Correction to get the final velocity
    p = dp
    u[:, 1:-1, 1:-1] -= (p[:, 1:, :] - p[:, :-1, :]) / dx
    v[:, 1:-1, 1:-1] -= (p[:, :, 1:] - p[:, :, :-1]) / dy

    # obstacle calibration
    # u,v = obstacle_mask_calibration(obstacle_mask, u, v)
    u = u * obstacle_mask["u"]
    v = v * obstacle_mask["v"]

    
    for iters in trange(niter, desc = "inner loop", leave = False):
        # Get nonlinear terms
        # 1. interpolate velocity at cell center/cell corner
        # u, v  μ? ? ? :
        utmp, vtmp = (uold + u) / 2.0, (vold + v) / 2.0

        uce = (utmp[:,:-1, 1:-1] + utmp[:,1:, 1:-1]) / 2
        uco = (utmp[:,:, :-1] + utmp[:,:, 1:]) / 2
        vco = (vtmp[:,:-1, :] + vtmp[:,1:, :]) / 2
        vce = (vtmp[:,1:-1, :-1] + vtmp[:,1:-1, 1:]) / 2

        # 2. multiply
        uuce = uce * uce
        uvco = uco * vco
        vvce = vce * vce

        # 3-1. get derivative for u
        Nu = (uuce[:,1:, :] - uuce[:,:-1, :]) / dx
        Nu += (uvco[:,1:-1, 1:] - uvco[:,1:-1, :-1]) / dy

        # 3-2. get derivative for v
        Nv = (vvce[:,:, 1:] - vvce[:,:, :-1]) / dy
        Nv += (uvco[:,1:, 1:-1] - uvco[:,:-1, 1:-1]) / dx

        # Implicit treatment for xy direction
        # Lubc = cp.zeros_like(Luy)
        # Lubc[:, -1] = 2 * bctop / dy**2  # effect of the top BC on Lu
        # Lvbc = cp.zeros_like(Lvy)

        b_u = uold[:,1:-1, 1:-1] - dt * (Nu  - 1 / (2 * Re) * (Lux + Luy))
        xu = BatchU_xyCNAB_dst(uold, b_u, dt, Re, nx, ny, dx, dy)
        b_v = vold[:,1:-1, 1:-1] - dt * (Nv  - 1 / (2 * Re) * (Lvx + Lvy))
        xv = BatchV_xyCNAB_dst(vold, b_v, dt, Re, nx, ny, dx, dy)

        u[:,1:-1, 1:-1] = xu
        v[:,1:-1, 1:-1] = xv

        # obstacle calibration
        # u,v = obstacle_mask_calibration(obstacle_mask, u, v)
        u = u * obstacle_mask["u"]
        v = v * obstacle_mask["v"]

        # RHS of pressure Poisson eq.
        b = ((u[:,1:, 1:-1] - u[:,:-1, 1:-1]) / dx +
            (v[:,1:-1, 1:] - v[:,1:-1, :-1]) / dy)

        # Solve for p

        dp = BatchPoissonEquation_2dDCT(b, dx)
        

        # Correction to get the final velocity
        p = dp
        u[:,1:-1, 1:-1] -= (p[:,1:, :] - p[:,:-1, :]) / dx
        v[:,1:-1, 1:-1] -= (p[:,:, 1:] - p[:,:, :-1]) / dy
        
        tb = (u[:,1:, 1:-1] - u[:,0:-1, 1:-1]) / dx + (v[:,1:-1, 1:] - v[:,1:-1, 0:-1]) / dy
        # print("     ?  ?  :{:.16f}".format(cp.mean(tb**2)))

        # obstacle calibration
        # u,v = obstacle_mask_calibration(obstacle_mask, u, v)
        u = u * obstacle_mask["u"]
        v = v * obstacle_mask["v"]
        

    return u, v, p

def random_mask_calibration(xce, yce, u, v, obstacle_mask):

    Mu = cp.ones(u.shape,dtype = cp.float64)
    Mv = cp.ones(v.shape,dtype = cp.float64)
    M , N = obstacle_mask[0].shape

    # obstacle_mask = obstacle_mask.get()
    # u,v = u.get(),v.get()
    # ti.loop_config(serialize=True) #      Taichi  ?     
    for k in range(len(obstacle_mask)):
        # for i in range(1, M - 1):
        #     for j in range(1, N - 1):
                mask_255 = (obstacle_mask[k][1:-1,1:-1] == 255)
                # import pdb;pdb.set_trace()
                # 利用广播机制和布尔索引更新 Mv
                Mv[k, 3:-1, 2:-1][mask_255] = 0.0
                Mv[k, 1:-3, 2:-1][mask_255] = 0.0

                # 利用广播机制和布尔索引更新 Mu
                Mu[k,2:-1, 3:-1][mask_255] = 0.0
                Mu[k,2:-1, 1:-3][mask_255] = 0.0
    # import pdb; pdb.set_trace()
    res = {"u":Mu, "v":Mv}
    return res

def generate_slice(Xce, Yce, Lx, Ly, Nx, Ny ,radius = 0.15):
    # Generate a random center point for the circle
    # Ensure the center is at least 0.25 from the boundaries
    center_x = 0.50# np.random.uniform(0.25, 1.5)
    center_y = Ly * 0.5# np.random.uniform(0.25, Ly - 0.25)

    # Radius of the circle
    # radius = 0.15

    # Initialize obstacle mask
    obstacle_mask = cp.zeros((Nx, Ny), dtype=cp.uint8)

    # Calculate the distance from each grid point to the center of the circle
    distance = cp.sqrt((Xce - center_x) ** 2 + (Yce - center_y) ** 2).T

    # Set the mask to 255 for points inside the circle
    # import pdb; pdb.set_trace()
    obstacle_mask[distance <= radius] = 255

    return obstacle_mask

def explicit_velocity(u,v):
#    ò ?  ?     е?      :
    uce = 0.5 * (u[0:-1, 1:-1] + u[1:, 1:-1])
    vce = 0.5 * (v[1:-1, 0:-1] + v[1:-1, 1:])
    return uce,vce

def streamlinePlot(Xce,Yce,u,v,save_path):
    uce,vce = explicit_velocity(u,v)
    # import pdb;pdb.set_trace()
    plt.subplots(figsize=(10,2))
    plt.streamplot(Xce, Yce,uce.T,vce.T, density=2.5, linewidth=1.0, color='blue')
    plt.xlim([0, Lx])
    plt.ylim([0, Ly])
#     plt.title('Re = {:.1f},time = {:4f}'.format(Re,ii*dt))
#     save_path = os.path.join(root_path,str(image_name)+".jpg")
    # plt.show()
    xL, yL = [], []
    vortex = (uce ** 2 + vce ** 2) * 0.5
    Nx = uce.shape[0]
    Ny = vce.shape[1]
    N = min(Nx,Ny)
    xx,yy = np.linspace(0,1,Nx),np.linspace(0,1,Ny)
    for i in range(1, Nx-1):
       for j in range(1, Ny-1):
           if vortex[i,j] < min(min(min(vortex[i+1,j+1],vortex[i-1,j+1]),min(vortex[i+1,j-1],vortex[i-1,j-1])),min(min(vortex[i,j-1],vortex[i,j+1]),min(vortex[i-1,j],vortex[i+1,j]))):
              qvalue =  -0.5 * ((uce[i+1,j] - uce[i-1,j]) / 2.0 * N) ** 2 - 0.5 * ((vce[i,j+1] - vce[i,j-1]) / 2.0 * N) ** 2 - ((uce[i,j+1] - uce[i,j-1]) / 2.0 * N) * ((vce[i+1,j] - vce[i-1,j]) / 2.0 * N)
              Sxy = 0.5 * ((uce[i,j+1] - uce[i,j-1]) / 2.0 * N) + ((vce[i+1,j] - vce[i-1,j]) / 2.0 * N) * 0.5
              Oxy = 0.5 * ((uce[i,j+1] - uce[i,j-1]) / 2.0 * N) - ((vce[i+1,j] - vce[i-1,j]) / 2.0 * N) * 0.5
              lambdaS = np.asarray([[(uce[i+1,j] - uce[i-1,j]) / 2.0 * N, Sxy],[Sxy, (vce[i+1,j] - vce[i-1,j]) / 2.0 * N]],dtype = np.float64)
              lambdaO = np.asarray([[0.0, Oxy],[-Oxy,0.0]],dtype = np.float64)
              lambdaM = np.dot(lambdaS,lambdaS) + np.dot(lambdaO,lambdaO)
              sqrt_root = np.sqrt((lambdaM[0,0] - lambdaM[1,1])**2 + 4 * lambdaM[0,1]**2)
              la1 = (sqrt_root + (lambdaM[0,0] + lambdaM[1,1]))/2.0
              la2 = (-sqrt_root + (lambdaM[0,0] + lambdaM[1,1]))/2.0
              # import pdb;pdb.set_trace()
              if qvalue > 0.0:
                  xL.append(xx[i])
                  yL.append(yy[j])
              with open("maskedCylinder.txt", mode = "a") as f:
                   f.write(save_path + "x = {:.4f}, y = {:.4f}, Qvalue = {:.16f}, lambda2 = {:.8f}, {:.8f}\n".format(xx[i],yy[j],qvalue,la1,la2))
    if len(xL) > 0:
       plt.scatter(np.asarray(xL), np.asarray(yL), c = 'r', marker = 'o')
       # Enable grid
       plt.grid(True)

       # Customize grid
       plt.grid(color='gray', linestyle='--', linewidth=0.5)
    if save_path:
        plt.savefig(save_path)
        plt.clf()

if __name__ == "__main__":
    inference_root = "CylinderLoader"
    os.makedirs(inference_root,exist_ok = True)
    cp.random.seed(39)
    Re = cp.ones((1,1,1)) * 10000.0 # cp.random.randint(100, 10000 + 1, size=(32, 1, 1)) 
    for re in range(Re.shape[0]):
        os.makedirs(os.path.join(inference_root,str(int(Re[re,0,0])) +"_"+ str(re)),exist_ok = True)
    recordRate = 1000
    nt = 200000 * 4  # max time steps
    Lx, Ly = 5.0, 1.0  # domain size
    Nx, Ny = 160 * 5, 160 # Number of grids
    dt = 0.0001 / 4.0  # time step
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Coordinate of each grid (cell center)
    xce = (cp.arange(1, Nx+1) - 0.5) * dx
    yce = (cp.arange(1, Ny+1) - 0.5) * dy
    Xce, Yce = cp.meshgrid(xce, yce)


    
    # Coordinate of each grid (cell corner)
    xco = cp.arange(Nx+1) * dx
    yco = cp.arange(Ny+1) * dy

    # Initialize velocity fields
    u = cp.zeros((Re.shape[0], Nx+1, Ny+2))  # velocity in x direction (u)
    v = cp.zeros((Re.shape[0], Nx+2, Ny+1))  # velocity in y direction (v)

    # Cylinder_center
    # obstacle_mask = cp.zeros((Nx, Ny), dtype = cp.uint8)
    obstacle_mask = []
    for i in range(Re.shape[0]):
        obstacle_mask.append(generate_slice(Xce, Yce, Lx, Ly, Nx, Ny))
    for i in range(Re.shape[0]):
        # import pdb; pdb.set_trace()
        cv2.imwrite(os.path.join(inference_root,str(int(Re[i,0,0])) +"_"+ str(i), "mask.png") ,obstacle_mask[i].get())
    obstacle_mask = random_mask_calibration(xce,yce,u,v,obstacle_mask)
    # import pdb; pdb.set_trace()
    

    u = cp.zeros((Re.shape[0], Nx+1, Ny+2))  # velocity in x direction (u)
    v = cp.zeros((Re.shape[0], Nx+2, Ny+1)) 

    for ii in trange(1, nt+1,desc = "cavityFlow"):
        bctop = 1  # top velocity
    
        # Update the velocity field (uses dct)
        u, v, p = BatchupdateVelocityField_CNAB_bctop(u, v, Nx, Ny, dx, dy, Re, dt, bctop, obstacle_mask)  # 选用第一种方法:Crank-Nicolson格式
        tb = (u[:, 1:, 1:-1] - u[:, 0:-1, 1:-1]) / dx + (v[:, 1:-1, 1:] - v[:, 1:-1, 0:-1]) / dy
        
        # 
        if ii % 2500 == 1:
           print("质量守恒律检测:{:.16f}".format(cp.linalg.norm(tb)))
        # Update the plot at every recordRate steps
        if ii % recordRate == 0:
            # get velocity at the cell center (for visualization)
            print("time = {:.4f},".format(ii*dt),end = ";")
            print("质量守恒律检测:{:.16f}".format(cp.linalg.norm(tb)))
            xce = (np.arange(1, Nx + 1) - 0.5) * dx
            yce = (np.arange(1, Ny + 1) - 0.5) * dy
            Xce, Yce = np.meshgrid(xce, yce)
            for re in trange(Re.shape[0],desc = "save_state_dict",leave = False):
                if re == 0:
                    streamlinePlot(Xce,Yce,u[re,:,:].get(),v[re,:,:].get(),os.path.join(inference_root,str(int(Re[re,0,0])) +"_"+ str(re),"T = {:.6f}.png".format(ii*dt)))
                # import pdb;pdb.set_trace()
                # vortexSurf(Xce, Yce,u[jj,:,:].get(),v[jj,:,:].get(),dx ,os.path.join(inference_root,str(int(Re[jj,0,0])),"vortex, T = {:.6f}.png".format(ii*dt))) 
                np.save( os.path.join(inference_root,str(int(Re[re,0,0])) +"_"+ str(re),"u, T = {:.6f}.npy".format(ii*dt)), u[re,:,:].get())
                np.save( os.path.join(inference_root,str(int(Re[re,0,0])) +"_"+ str(re),"v, T = {:.6f}.npy".format(ii*dt)), v[re,:,:].get())