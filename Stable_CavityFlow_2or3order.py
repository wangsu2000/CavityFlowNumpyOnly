import numpy as np
import matplotlib.pyplot as plt
def mydst1(a):
    # Apply DST-I in each column
    N1, N2 = a.shape
    xx = np.vstack([
        np.zeros((1, N2)),
        a,
        np.zeros((1, N2)),
        np.zeros((N1, N2))
    ])  # DST1
    tmp = -np.fft.fft(xx, axis=0) * np.sqrt(2 / (N1 + 1))
    b = np.imag(tmp[1:N1 + 1, :])
    return b

def mydst2(a):
    # Apply DST-II in each column
    N1, N2 = a.shape
    xx = np.vstack([a, np.zeros((N1, N2))])
    tmp = np.fft.fft(xx, axis=0)
    xd = tmp[1:N1 + 1, :]
    ww = np.sqrt(2 / N1) * np.exp(-1j * np.pi * np.arange(1, N1 + 1) / (2 * N1))[:, np.newaxis]
    b = xd * ww
    b[-1, :] = b[-1, :] / np.sqrt(2)
    b = -np.imag(b)
    return b

def mydst3(a):
    # Apply DST-III in each column
    N1, N2 = a.shape
    a[-1, :] = a[-1, :] / np.sqrt(2)
    xx = np.vstack([
        np.zeros((1, N2)),
        a,
        np.zeros((N1 - 1, N2))
    ])  # DST1

    # weight
    ww = np.exp(-1j * np.pi * np.arange(0, 2 * N1) / (2 * N1))[:, np.newaxis]
    xxw = xx * ww
    tmp = np.fft.fft(xxw, axis=0) * np.sqrt(2 / N1)
    b = -np.imag(tmp[:N1, :])
    return b
def getIntermediateU_xyRK3_dst(u, b, dt, Re, nx, ny, dx, dy, id):
    kx = np.arange(1, nx)
    ax = np.pi * kx / nx
    mwx = 2 * (np.cos(ax) - 1) / dx**2  # DST-I

    ky = np.arange(1, ny + 1)
    ay = np.pi * ky / ny
    mwy = 2 * (np.cos(ay) - 1) / dy**2  # DST-II

    mw = mwx[:, np.newaxis] + mwy  # Modified Wavenumber

    rhshat = mydst2(mydst1(b).T).T

    if id == 1:
        uhat = rhshat / (1 - (4 / 15) * dt / Re * mw)
    elif id == 2:
        uhat = rhshat / (1 - (1 / 15) * dt / Re * mw)
    elif id == 3:
        uhat = rhshat / (1 - (1 / 6) * dt / Re * mw)

    xu = mydst1(mydst3(uhat.T).T)
    return xu

def getIntermediateV_xyRK3_dst(v, b, dt, Re, nx, ny, dx, dy, id):
    kx = np.arange(1, nx + 1)
    ax = np.pi * kx / nx
    mwx = 2 * (np.cos(ax) - 1) / dx**2  # DST-I

    ky = np.arange(1, ny)
    ay = np.pi * ky / ny
    mwy = 2 * (np.cos(ay) - 1) / dy**2  # DST-II

    mw = mwx[:, np.newaxis] + mwy  # Modified Wavenumber

    rhshat = mydst1(mydst2(b).T).T

    if id == 1:
        vhat = rhshat / (1 - (4 / 15) * dt / Re * mw)
    elif id == 2:
        vhat = rhshat / (1 - (1 / 15) * dt / Re * mw)
    elif id == 3:
        vhat = rhshat / (1 - (1 / 6) * dt / Re * mw)

    xv = mydst3(mydst1(vhat.T).T)
    return xv
def getIntermediateU_xyCNAB_dst(u, b, dt, Re, nx, ny, dx, dy):
    kx = np.arange(1, nx)
    ax = np.pi * kx / nx
    mwx = 2 * (np.cos(ax) - 1) / dx**2  # DST-I

    ky = np.arange(1, ny + 1)
    ay = np.pi * ky / ny
    mwy = 2 * (np.cos(ay) - 1) / dy**2  # DST-II

    mw = mwx[:, np.newaxis] + mwy  # Modified Wavenumber

    rhshat = mydst2(mydst1(b).T).T
    uhat = rhshat / (1 - (1 / 2) * dt / Re * mw)

    xu = mydst1(mydst3(uhat.T).T)
    return xu

def getIntermediateV_xyCNAB_dst(v, b, dt, Re, nx, ny, dx, dy):
    kx = np.arange(1, nx + 1)
    ax = np.pi * kx / nx
    mwx = 2 * (np.cos(ax) - 1) / dx**2  # DST-I

    ky = np.arange(1, ny)
    ay = np.pi * ky / ny
    mwy = 2 * (np.cos(ay) - 1) / dy**2  # DST-II

    mw = mwx[:, np.newaxis] + mwy  # Modified Wavenumber

    rhshat = mydst1(mydst2(b).T).T

    vhat = rhshat / (1 - (1 / 2) * dt / Re * mw)

    xv = mydst3(mydst1(vhat.T).T)
    return xv

def updateVelocityField_CNAB_bctop(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, bctop):
  

   

    # Apply boundary conditions:
    u[:, 0] = -u[:, 1]             # bottom
    v[:, 0] = 0.                   # bottom
    u[:, -1] = 2 * bctop - u[:, -2]  # top
    v[:, -1] = 0.                  # top
    u[0, :] = 0.                   # left
    v[0, :] = -v[1, :]             # left
    u[-1, :] = 0.                  # right
    v[-1, :] = -v[-2, :]           # right

    # Get viscous terms for u
    Lux = (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / dx**2
    Luy = (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / dy**2

    # Get viscous terms for v
    Lvx = (v[:-2, 1:-1] - 2 * v[1:-1, 1:-1] + v[2:, 1:-1]) / dx**2
    Lvy = (v[1:-1, :-2] - 2 * v[1:-1, 1:-1] + v[1:-1, 2:]) / dy**2

    # Get nonlinear terms
    # 1. interpolate velocity at cell center/cell corner
    uce = (u[:-1, 1:-1] + u[1:, 1:-1]) / 2
    uco = (u[:, :-1] + u[:, 1:]) / 2
    vco = (v[:-1, :] + v[1:, :]) / 2
    vce = (v[1:-1, :-1] + v[1:-1, 1:]) / 2

    # 2. multiply
    uuce = uce * uce
    uvco = uco * vco
    vvce = vce * vce

    # 3-1. get derivative for u
    Nu = (uuce[1:, :] - uuce[:-1, :]) / dx
    Nu += (uvco[1:-1, 1:] - uvco[1:-1, :-1]) / dy

    # 3-2. get derivative for v
    Nv = (vvce[:, 1:] - vvce[:, :-1]) / dy
    Nv += (uvco[1:, 1:-1] - uvco[:-1, 1:-1]) / dx

    # Implicit treatment for xy direction
    Lubc = np.zeros_like(Luy)
    Lubc[:, -1] = 2 * bctop / dy**2  # effect of the top BC on Lu
    Lvbc = np.zeros_like(Lvy)

    b_u = u[1:-1, 1:-1] - dt * ((3 * Nu - Nu_old) / 2 - 1 / (2 * Re) * (Lux + Luy + Lubc))
    xu = getIntermediateU_xyCNAB_dst(u, b_u, dt, Re, nx, ny, dx, dy)
    b_v = v[1:-1, 1:-1] - dt * ((3 * Nv - Nv_old) / 2 - 1 / (2 * Re) * (Lvx + Lvy + Lvbc))
    xv = getIntermediateV_xyCNAB_dst(v, b_v, dt, Re, nx, ny, dx, dy)

    Nu_old = Nu
    Nv_old = Nv

    u[1:-1, 1:-1] = xu
    v[1:-1, 1:-1] = xv

    # RHS of pressure Poisson eq.
    b = ((u[1:, 1:-1] - u[:-1, 1:-1]) / dx +
         (v[1:-1, 1:] - v[1:-1, :-1]) / dy)

    # Solve for p

    dp = solvePoissonEquation_2dDCT(b, dx)
    

    # Correction to get the final velocity
    p = dp
    u[1:-1, 1:-1] -= (p[1:, :] - p[:-1, :]) / dx
    v[1:-1, 1:-1] -= (p[:, 1:] - p[:, :-1]) / dy

    return u, v, p
    


def updateVelocityField_RK3_bctop(u, v,Nu_old ,Nv_old,nx, ny, dx, dy, Re, dt, bctop):
    u, v, p = updateVelocityField_RK3substep(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, bctop,1)
    u, v, p = updateVelocityField_RK3substep(u, v,Nu_old,Nv_old,nx, ny, dx, dy, Re, dt, bctop,2)
    u, v, p = updateVelocityField_RK3substep(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, bctop,3)
    return u, v, p
def solvePoissonEquation_2dDCT(b, h):
    m, n = b.shape

    row = n * np.cos(np.pi * np.arange(0, n) / n)
    row[0] = row[0] + n
    col = m / 2 * np.ones(m)
    col[0] = col[0] + m / 2
    rc = np.outer(row, col)
    row = n / 2 * np.ones(n)
    row[0] = row[0] + n / 2
    col = m * np.cos(np.pi * np.arange(0, m) / m) - 2 * m
    col[0] = col[0] - m
    rc = rc + np.outer(row, col)
    
    y1 = np.fft.fft(np.concatenate([b, np.zeros_like(b, dtype=np.complex128)], axis=0),axis = 0)
#     print(np.concatenate([b, np.zeros_like(b, dtype=np.complex128)], axis=0))
#     print(y1.round(4))
    y1 = np.real(y1[:m, :] * (np.cos(np.arange(0, m).reshape(-1, 1) * np.pi / (2 * m)) - np.sin(np.arange(0, m).reshape(-1, 1) * np.pi / (2 * m)) * 1j).reshape(-1,1))
    y1 = np.fft.fft(np.concatenate([y1.T.conj(), np.zeros_like(y1.T, dtype=np.complex128)], axis=0),axis = 0)
    y1 = np.real(y1[:n, :] * (np.cos(np.arange(0, n).reshape(-1, 1) * np.pi / (2 * n)) - np.sin(np.arange(0, n).reshape(-1, 1) * np.pi / (2 * n)) * 1j).reshape(-1,1))
#     print(rc.round(4))
    y1[0, 0] = 0
    y1 = (y1 / rc).T.conj()
    y1[0, 0] = 0
#     print(y1.shape)
#     print(np.concatenate([ ((np.cos(np.arange(0, m).reshape(-1, 1) * np.pi / (2 * m)) ) - (np.sin(np.arange(0, m).reshape(-1, 1) * np.pi / (2 * m))) * 1j)*y1, np.zeros((m, n))], axis=0).round(2))
    y2 = np.real( np.fft.fft(np.concatenate([ ((np.cos(np.arange(0, m).reshape(-1, 1) * np.pi / (2 * m)) ) - (np.sin(np.arange(0, m).reshape(-1, 1) * np.pi / (2 * m))) * 1j)*y1, np.zeros((m, n))], axis=0),axis = 0))
#     print(y2.shape)
#     print(y2.round(4))
    y2 = y2[:m,:].T.conj()
#     print(y2.round(4))
#     print(y2.shape)
    y2 = np.real( np.fft.fft(np.concatenate([(np.cos(np.arange(0, n).reshape(-1, 1) * np.pi / (2 * n)) )*y2 -(np.sin(np.arange(0, n).reshape(-1, 1) * np.pi / (2 * n)) )*y2*1j, np.zeros((n, m))], axis=0),axis = 0))

    re = (h**2) * y2[:n, :].T.conj()

    return re
def updateVelocityField_RK3substep(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, bctop, id):
    # global Nu_old, Nv_old

    # if not Nu_old  or not Nv_old or Nu_old.shape != (nx-1, ny):
        

    # Apply boundary conditions:
    u[:, 0] = -u[:, 1]
    v[:, 0] = 0.0             # bottom
    u[:, -1] = 2*bctop - u[:, -2]
    v[:, -1] = 0.0            # top
    u[0, :] = 0.0
    v[0, :] = -v[1, :]        # left
    u[-1, :] = 0.0
    v[-1, :] = -v[-2, :]      # right

    # Get viscous terms for u
    Lux = (u[:-2, 1:-1] - 2*u[1:-1, 1:-1] + u[2:, 1:-1]) / dx**2
    Luy = (u[1:-1, :-2] - 2*u[1:-1, 1:-1] + u[1:-1, 2:]) / dy**2

    # Get viscous terms for v
    Lvx = (v[:-2, 1:-1] - 2*v[1:-1, 1:-1] + v[2:, 1:-1]) / dx**2
    Lvy = (v[1:-1, :-2] - 2*v[1:-1, 1:-1] + v[1:-1, 2:]) / dy**2

    # Get nonlinear terms
    uce = (u[:-1, 1:-1] + u[1:, 1:-1]) / 2
    uco = (u[:, :-1] + u[:, 1:]) / 2
    vco = (v[:-1, :] + v[1:, :]) / 2
    vce = (v[1:-1, :-1] + v[1:-1, 1:]) / 2

    uuce = uce * uce
    uvco = uco * vco
    vvce = vce * vce

    Nu = (uuce[1:, :] - uuce[:-1, :]) / dx + (uvco[1:-1, 1:] - uvco[1:-1, :-1]) / dy
    Nv = (vvce[:, 1:] - vvce[:, :-1]) / dy + (uvco[1:, 1:-1] - uvco[:-1, 1:-1]) / dx

    Lubc = np.zeros_like(Luy)
    Lubc[:, -1] = 2*bctop / dy**2

    Lvbc = np.zeros_like(Lvy)

    alpha = [4/15, 1/15, 1/6]
    beta = alpha
    gamma = [8/15, 5/12, 3/4]
    zeta = [0, -17/60, -5/12]

    b = u[1:-1, 1:-1] - dt * (gamma[id-1] * Nu + zeta[id-1] * Nu_old - alpha[id-1] / Re * (Lux + Luy + Lubc))
    xu = getIntermediateU_xyRK3_dst(u, b, dt, Re, nx, ny, dx, dy, id)

    b = v[1:-1, 1:-1] - dt * (gamma[id-1] * Nv + zeta[id-1] * Nv_old - alpha[id-1] / Re * (Lvx + Lvy + Lvbc))
    xv = getIntermediateV_xyRK3_dst(v, b, dt, Re, nx, ny, dx, dy, id)

    Nu_old = Nu
    Nv_old = Nv

    u[1:-1, 1:-1] = xu
    v[1:-1, 1:-1] = xv

    # RHS of pressure Poisson eq.
    b = (u[1:, 1:-1] - u[:-1, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy

    # Solve for p
    dp = solvePoissonEquation_2dDCT(b, dx)

    # correction to get the final velocity
    p = dp
    u[1:-1, 1:-1] = u[1:-1, 1:-1] - (p[1:, :] - p[:-1, :]) / dx
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - (p[:, 1:] - p[:, :-1]) / dy

    return u, v, p
    
def explicit_velocity(u,v):
# 利用插值还原网格中点处的流速:
    uce = 0.5 * (u[0:-1, 1:-1] + u[1:, 1:-1])
    vce = 0.5 * (v[1:-1, 0:-1] + v[1:-1, 1:])
    return uce,vce
def streamlinePlot(Xce,Yce,u,v,save_path):
    uce,vce = explicit_velocity(u,v)
    plt.streamplot(Xce, Yce,uce.T,vce.T, density=1, linewidth=1, color='blue')
    plt.xlim([0, Lx])
    plt.ylim([0, Ly])
#     plt.title('Re = {:.1f},time = {:4f}'.format(Re,ii*dt))
#     save_path = os.path.join(root_path,str(image_name)+".jpg")
    plt.show()
    if save_path:
        plt.savefig(save_path)
        plt.clf()


if __name__ == "__main__":

    recordRate = 250
    Re = 1000  # Reynolds number
    nt = 20000  # max time steps
    Lx, Ly = 1, 1  # domain size
    Nx, Ny = 80, 80  # Number of grids
    dt = 0.001  # time step
    
    # Grid size (Equispaced)
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Coordinate of each grid (cell center)
    xce = (np.arange(1, Nx+1) - 0.5) * dx
    yce = (np.arange(1, Ny+1) - 0.5) * dy


    Nu_old = np.zeros((Nx-1, Ny))
    Nv_old = np.zeros((Nx, Ny-1))
    
    # Coordinate of each grid (cell corner)
    xco = np.arange(Nx+1) * dx
    yco = np.arange(Ny+1) * dy
    
    # Initialize velocity fields
    u = np.zeros((Nx+1, Ny+2))  # velocity in x direction (u)
    v = np.zeros((Nx+2, Ny+1))  # velocity in y direction (v)
    uce = (u[:-1, 1:-1] + u[1:, 1:-1]) / 2  # u at cell center
    vce = (v[1:-1, :-1] + v[1:-1, 1:]) / 2  # v at cell center
    
    for ii in range(1, nt+1):
        bctop = 1  # top velocity
    
        # Update the velocity field (uses dct)
        # u, v, p = updateVelocityField_CNAB_bctop(u, v,Nu_old,Nv_old, Nx, Ny, dx, dy, Re, dt, bctop)  # 选用第一种方法:Crank-Nicolson格式
        u, v, p = updateVelocityField_RK3_bctop(u, v,Nu_old,Nv_old, Nx, Ny, dx, dy, Re, dt, bctop) # 选用第二种方法:RK-3格式
        tb = (u[1:, 1:-1] - u[0:-1, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, 0:-1]) / dy
        
        # 
    
        # Update the plot at every recordRate steps
        if ii % recordRate == 0:
            # get velocity at the cell center (for visualization)
            print("time = {:.4f},".format(ii*dt),end = ";")
            print("mass conserve validate:{:.16f}".format(np.linalg.norm(tb)))
            streamlinePlot(xce,yce,u,v,None)
            # update plot (downsample)
            # Visualization code would go here
