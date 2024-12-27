#coding=gbk
import numpy as np
from datetime import datetime
import os
import torch
from tqdm import tqdm 
import torch.optim as optim
from stable_cavity_flow import solvePoissonEquation_2dDCT,getIntermediateU_xyRK3_dst,getIntermediateV_xyRK3_dst
from stable_cavity_flow import getIntermediateU_xyCNAB_dst,getIntermediateV_xyCNAB_dst


def update_velocity_field_CNAB(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, velbc, uForce, vForce, velbc1):
    # Initialize persistent variables

    # Apply boundary conditions
    u[:, 0] = 2 * velbc['uBottom'].flatten() - u[:, 1]
    v[:, 0] = velbc['vBottom'].flatten()
    u[:, -1] = 2 * velbc['uTop'].flatten() - u[:, -2]
    v[:, -1] = velbc['vTop'].flatten()
    u[0, :] = velbc['uLeft'].flatten()
    v[0, :] = 2 * velbc['vLeft'].flatten()- v[1, :]
    u[-1, :] = velbc['uRight'].flatten()
    v[-1, :] = 2 * velbc['vRight'].flatten() - v[-2, :]

    # Get viscous terms for u
    Lux = (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / dx**2
    Luy = (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / dy**2

    # Get viscous terms for v
    Lvx = (v[:-2, 1:-1] - 2 * v[1:-1, 1:-1] + v[2:, 1:-1]) / dx**2
    Lvy = (v[1:-1, :-2] - 2 * v[1:-1, 1:-1] + v[1:-1, 2:]) / dy**2

    # Get nonlinear terms
    uce = (u[:-1, 1:-1] + u[1:, 1:-1]) / 2
    uco = (u[:, :-1] + u[:, 1:]) / 2
    vco = (v[:-1, :] + v[1:, :]) / 2
    vce = (v[1:-1, :-1] + v[1:-1, 1:]) / 2

    uuce = uce * uce
    uvco = uco * vco
    vvce = vce * vce

    Nu = (uuce[1:, :] - uuce[:-1, :]) / dx
    Nu += (uvco[1:-1, 1:] - uvco[1:-1, :-1]) / dy
    Nu -= uForce

    Nv = (vvce[:, 1:] - vvce[:, :-1]) / dy
    Nv += (uvco[1:, 1:-1] - uvco[:-1, 1:-1]) / dx
    Nv -= vForce

    Lubc = np.zeros(Luy.shape)
    Lubc[0, :] = velbc1['uLeft'].flatten()[1:-1] / dx**2
    Lubc[-1, :] = velbc1['uRight'].flatten()[1:-1] / dx**2
    Lubc[:, 0] = 2 * velbc1['uBottom'].flatten()[1:-1] / dy**2
    Lubc[:, -1] = 2 * velbc1['uTop'].flatten()[1:-1] / dy**2

    Lvbc = np.zeros(Lvy.shape)
    Lvbc[0, :] = 2 * velbc1['vLeft'].flatten()[1:-1] / dx**2
    Lvbc[-1, :] = 2 * velbc1['vRight'].flatten()[1:-1] / dx**2
    Lvbc[:, 0] = velbc1['vBottom'].flatten()[1:-1] / dy**2
    Lvbc[:, -1] = velbc1['vTop'].flatten()[1:-1] / dy**2

    b = u[1:-1, 1:-1] - dt * ((3 * Nu - Nu_old) / 2 - 1 / (2 * Re) * (Lux + Luy + Lubc))
    xu = getIntermediateU_xyCNAB_dst(u, b, dt, Re, nx, ny, dx, dy)
    b = v[1:-1, 1:-1] - dt * ((3 * Nv - Nv_old) / 2 - 1 / (2 * Re) * (Lvx + Lvy + Lvbc))
    xv = getIntermediateV_xyCNAB_dst(v, b, dt, Re, nx, ny, dx, dy)

    u[1:-1, 1:-1] = xu
    v[1:-1, 1:-1] = xv

    Nu_old = Nu
    Nv_old = Nv

    u[:, 0] = 2 * velbc1['uBottom'].flatten() - u[:, 1]
    v[:, 0] =  velbc1['vBottom'].flatten()
    u[:, -1] = 2 * velbc1['uTop'].flatten() - u[:, -2]
    v[:, -1] = velbc1['vTop'].flatten()
    u[0, :] = velbc1['uLeft'].flatten()
    v[0, :] = 2 * velbc1['vLeft'].flatten() - v[1, :]
    u[-1, :] = velbc1['uRight'].flatten()
    v[-1, :] = 2 * velbc1['vRight'].flatten() - v[-2, :]

    b = ((u[1:, 1:-1] - u[:-1, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy)

    
    dp = solvePoissonEquation_2dDCT(b, dx)
    

   
    u[1:-1, 1:-1] -= (dp[1:, :] - dp[:-1, :]) / dx
    v[1:-1, 1:-1] -= (dp[:, 1:] - dp[:, :-1]) / dy

    p = dp

    return u, v, p,Nu_old,Nv_old

def update_velocity_field_RK3(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, velbc, uForce, vForce, velbc1):
    u, v, p,Nu_old,Nv_old = update_velocity_field_RK3substep(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, velbc, 1, uForce, vForce, velbc1)
    u, v, p,Nu_old,Nv_old = update_velocity_field_RK3substep(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, velbc, 2, uForce, vForce, velbc1)
    u, v, p,Nu_old,Nv_old = update_velocity_field_RK3substep(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, velbc, 3, uForce, vForce, velbc1)
    return u, v, p,Nu_old,Nv_old

def update_velocity_field_RK3substep(u, v,Nu_old,Nv_old, nx, ny, dx, dy, Re, dt, velbc, id, uForce, vForce, velbc1):

   

    # Apply boundary conditions:
    # represented as the values on ghost cells
    u[:,0] = 2 * velbc['uBottom'].flatten() - u[:,1]
    v[:,0] = velbc['vBottom'].flatten()
    u[:,-1] = 2 * velbc['uTop'].flatten() - u[:,-2]
    v[:,-1] = velbc['vTop'].flatten()
    u[0,:] = velbc['uLeft'].flatten()
    v[0,:] = 2 * velbc['vLeft'].flatten() - v[1,:]
    u[-1,:] = velbc['uRight'].flatten()
    v[-1,:] = 2 * velbc['vRight'].flatten() - v[-2,:]

    # Get viscous terms for u
    Lux = (u[:-2,1:-1] - 2 * u[1:-1,1:-1] + u[2:,1:-1]) / dx**2
    Luy = (u[1:-1,:-2] - 2 * u[1:-1,1:-1] + u[1:-1,2:]) / dy**2

    # Get viscous terms for v
    Lvx = (v[:-2,1:-1] - 2 * v[1:-1,1:-1] + v[2:,1:-1]) / dx**2
    Lvy = (v[1:-1,:-2] - 2 * v[1:-1,1:-1] + v[1:-1,2:]) / dy**2

    # Interpolate velocity at cell center/cell corner
    uce = (u[:-1,1:-1] + u[1:,1:-1]) / 2
    uco = (u[:, :-1] + u[:,1:]) / 2
    vco = (v[:-1,:] + v[1:,:]) / 2
    vce = (v[1:-1,:-1] + v[1:-1,1:]) / 2

    # Multiply
    uuce = uce * uce
    uvco = uco * vco
    vvce = vce * vce

    # Get derivative for u
    Nu = (uuce[1:] - uuce[:-1]) / dx
    Nu += (uvco[1:-1,1:] - uvco[1:-1,:-1]) / dy
    Nu -= uForce

    # Get derivative for v
    Nv = (vvce[:,1:] - vvce[:,:-1]) / dy
    Nv += (uvco[1:,1:-1] - uvco[:-1,1:-1]) / dx
    Nv -= vForce

    Lubc = np.zeros(Luy.shape)
    Lubc[0,:] = velbc1['uLeft'].flatten()[1:-1] / dx**2
    Lubc[-1,:] = velbc1['uRight'].flatten()[1:-1] / dx**2
    Lubc[:,0] = 2 * velbc1['uBottom'].flatten()[1:-1] / dy**2
    Lubc[:,-1] = 2 * velbc1['uTop'].flatten()[1:-1] / dy**2

    Lvbc = np.zeros(Lvy.shape)
    Lvbc[0,:] = 2 * velbc1['vLeft'].flatten()[1:-1] / dx**2
    Lvbc[-1,:] = 2 * velbc1['vRight'].flatten()[1:-1] / dx**2
    Lvbc[:,0] = velbc1['vBottom'].flatten()[1:-1] / dy**2
    Lvbc[:,-1] = velbc1['vTop'].flatten()[1:-1] / dy**2

    alpha = [4/15, 1/15, 1/6]
    gamma = [8/15, 5/12, 3/4]
    zeta = [0, -17/60, -5/12]

    b = u[1:-1,1:-1] - dt * (gamma[id-1] * Nu + zeta[id-1] * Nu_old - alpha[id-1] / Re * (Lux + Luy + Lubc))
    xu = getIntermediateU_xyRK3_dst(u, b, dt, Re, nx, ny, dx, dy, id)

    b = v[1:-1,1:-1] - dt * (gamma[id-1] * Nv + zeta[id-1] * Nv_old - alpha[id-1] / Re * (Lvx + Lvy + Lvbc))
    xv = getIntermediateV_xyRK3_dst(v, b, dt, Re, nx, ny, dx, dy, id)

    u[1:-1,1:-1] = xu
    v[1:-1,1:-1] = xv

    Nu_old = Nu
    Nv_old = Nv

    u[:,0] = 2 * velbc1['uBottom'].flatten() - u[:,1]
    v[:,0] = velbc1['vBottom'].flatten()
    u[:,-1] = 2 * velbc1['uTop'].flatten() - u[:,-2]
    v[:,-1] = velbc1['vTop'].flatten()
    u[0,:] = velbc1['uLeft'].flatten()
    v[0,:] = 2 * velbc1['vLeft'].flatten() - v[1,:]
    u[-1,:] = velbc1['uRight'].flatten()
    v[-1,:] = 2 * velbc1['vRight'].flatten() - v[-2,:]

    b = ((u[1:,1:-1] - u[:-1,1:-1]) / dx + (v[1:-1,1:] - v[1:-1,:-1]) / dy)


    dp = solvePoissonEquation_2dDCT(b, dx)



    u[1:-1,1:-1] -= (dp[1:,:] - dp[:-1,:]) / dx
    v[1:-1,1:-1] -= (dp[:,1:] - dp[:,:-1]) / dy

    return u, v, dp , Nu_old,Nv_old

def checkNSSolverError(Re, a, N, dt, tEnd, fuFunc, fvFunc, usolFunc, vsolFunc, psolFunc,timeScheme):


    # Simulation Setting
    Lx, Ly = 1, 1
    Nx, Ny = N, N

    dx = Lx / Nx
    dy = Ly / Ny

    Nu_old = np.zeros((Nx-1, Ny))
    Nv_old = np.zeros((Nx, Ny-1))

    xce = (np.arange(1, Nx + 1) - 0.5) * dx
    yce = (np.arange(1, Ny + 1) - 0.5) * dy
    xco = np.arange(0, Nx + 1) * dx
    yco = np.arange(0, Ny + 1) * dy

    Xce, Yce = np.meshgrid(xce, yce)

    # Initialize the memory for flow field
    u = np.full((Nx + 1, Ny + 2), np.nan)
    v = np.full((Nx + 2, Ny + 1), np.nan)
    p = np.full((Nx + 1, Ny + 1), np.nan)

    # For the current boundary condition
    velbc = {
        'uTop': np.full((u.shape[0], 1), np.nan),
        'uBottom': np.full((u.shape[0], 1), np.nan),
        'uLeft': np.full((1, u.shape[1]), np.nan),
        'uRight': np.full((1, u.shape[1]), np.nan),
        'vTop': np.full((v.shape[0], 1), np.nan),
        'vBottom': np.full((v.shape[0], 1), np.nan),
        'vLeft': np.full((1, v.shape[1]), np.nan),
        'vRight': np.full((1, v.shape[1]), np.nan)
    }

    # For the boundary condition at one time step ahead
    velbc1 = velbc.copy()

    # Initialization of the flow field
    t = 0
    uX, uY = np.meshgrid(xco[1:-1], yce)
    usol = usolFunc(uX.T, uY.T, t, Re, a)

    vX, vY = np.meshgrid(xce, yco[1:-1])
    vsol = vsolFunc(vX.T, vY.T, t, Re, a)

    pX, pY = np.meshgrid(xce, yce)
    p = psolFunc(pX.T, pY.T, t, Re, a)

    u[1:-1, 1:-1] = usol
    v[1:-1, 1:-1] = vsol
    

    velbc = getVelocityBoundaryCondition(velbc, usolFunc, vsolFunc, xco, yco, xce, yce, t, Re, a)

    u[:, 0] = 2 * velbc1['uBottom'].flatten() - u[:, 1]
    v[:, 0] = velbc1['vBottom'].flatten()
    u[:, -1] = 2 * velbc1['uTop'].flatten() - u[:, -2]
    v[:, -1] = velbc1['vTop'].flatten()
    u[0, :] = velbc1['uLeft'].flatten()
    v[0, :] = 2 * velbc1['vLeft'].flatten() - v[1, :]
    u[-1, :] = velbc1['uRight'].flatten()
    v[-1, :] = 2 * velbc1['vRight'].flatten() - v[-2, :]

    inflow = np.sum(velbc['vBottom'][1:-1]) * dx + np.sum(velbc['uLeft'][0, 1:-1]) * dy
    outflow = np.sum(velbc['vTop'][1:-1]) * dx + np.sum(velbc['uRight'][0, 1:-1]) * dy
    assert abs(inflow - outflow) < 1.0e-10 # np.finfo(float).eps, "Inflow flux must match the outflow flux."

    tbar = tqdm(total = int(tEnd / dt),desc = "generating groundTruth",leave = False)
    while t < tEnd:
        velbc = getVelocityBoundaryCondition(velbc, usolFunc, vsolFunc, xco, yco, xce, yce, t, Re, a)
        velbc1 = getVelocityBoundaryCondition(velbc1, usolFunc, vsolFunc, xco, yco, xce, yce, t + dt, Re, a)

        uForce = fuFunc(uX.T, uY.T, t, Re, a)
        vForce = fvFunc(vX.T, vY.T, t, Re, a)

        # if timeScheme == 'Euler':
        #     u, v, p = updateVelocityField_Euler(u, v, Nx, Ny, dx, dy, Re, dt, velbc, 'dct', uForce, vForce, velbc1)
        if timeScheme == 'CNAB':
           u, v, p,Nu_old,Nv_old = update_velocity_field_CNAB(u, v, Nu_old, Nv_old, Nx, Ny, dx, dy, Re, dt, velbc, uForce, vForce, velbc1)
        elif timeScheme == 'RK3':
           u, v, p , Nu_old, Nv_old= update_velocity_field_RK3(u, v,Nu_old, Nv_old, Nx, Ny, dx, dy, Re, dt, velbc, uForce, vForce, velbc1)
        # else:
        #     raise ValueError('timeScheme is not recognized')

        t += dt

        uce = (u[:-1, 1:-1] + u[1:, 1:-1]) / 2
        vce = (v[1:-1, :-1] + v[1:-1, 1:]) / 2
        usol = usolFunc(Xce.T, Yce.T, t, Re, a)
        vsol = vsolFunc(Xce.T, Yce.T, t, Re, a)
        psol = psolFunc(pX.T, pY.T, t, Re, a)

        uSim = np.concatenate([uce.ravel(), vce.ravel()])
        uSol = np.concatenate([usol.ravel(), vsol.ravel()])
        L2error = np.linalg.norm(uSim - uSol) / np.linalg.norm(uSol)
        [px,py] = np.gradient(p,dx)
        [lpx,lpy] = np.gradient(psol,dx)
        L2Perror = np.linalg.norm((px - lpx)**2 + (py - lpy)**2)
        # print("L2error相对误差:{:.12f}".format(L2error))
        uforce = fuFunc(Xce.T, Yce.T, t, Re, a)
        vforce = fvFunc(Xce.T, Yce.T, t, Re, a)
        # print(uce.shape,vce.shape,p.shape,uforce.shape,vforce.shape)
        # inp = torch.from_numpy(np.stack([uce,vce,p,uforce,vforce],axis = 0)).cuda().unsqueeze(0)
        # reference = torch.from_numpy(np.stack([usol,vsol],axis = 0)).cuda().unsqueeze(0)
        u_error = uce - usol
        v_error = vce - vsol
        with open('firstOrder_tau{:.8f}.txt'.format(dt), 'a') as file:
              file.write("t = {:.4f},u_max = {:.16f},v_max = {:.16f},u_mse = {:.16f},v_mse = {:.16f}\n".format(
                  t,
                  np.max(np.abs(u_error)),
                  np.max(np.abs(v_error)),
                  np.mean(np.square(u_error)),
                  np.mean(np.square(v_error))
              ))
        # import pdb;pdb.set_trace()
#        print("t = {:.4f},u_max = {:16f},v_max = {:.16f},u_mse = {:.16f},v_mse = {:.16f}".format(t,np.max(np.abs(u_error)),np.max(np.abs(v_error)),np.mean(np.square(u_error)),
#                                                                                           np.mean(np.square(v_error))))
        # inference = nnunet(inp)
        # optimizer.zero_grad()
        # loss = criterion(reference,inference)
        # loss.backward()
        # optimizer.step()
        # print("当前时刻训练误差:{:.12f}".format(loss.detach().cpu().numpy()))

        CFL = np.max([u.ravel() / dx, v.ravel() / dy]) * dt
        tbar.update(1)
    tbar.close()
    final_error = np.concatenate([uce, vce],axis = 1) -  np.concatenate([usol, vsol],axis = 1)
    
    return final_error,uSim,uSol


def getVelocityBoundaryCondition(velbc, usolFunc, vsolFunc, xco, yco, xce, yce, t, Re, a):
    velbc['uTop'] = usolFunc(xco, 1, t, Re, a).T
    velbc['uBottom'] = usolFunc(xco, 0, t, Re, a).T
    velbc['uLeft'][0, 1:-1] = usolFunc(0, yce, t, Re, a)
    velbc['uRight'][0, 1:-1] = usolFunc(1, yce, t, Re, a)
    velbc['vTop'][1:-1, 0] = vsolFunc(xce, 1, t, Re, a).T
    velbc['vBottom'][1:-1, 0] = vsolFunc(xce, 0, t, Re, a).T
    velbc['vLeft'][0, :] = vsolFunc(0, yco, t, Re, a)
    velbc['vRight'][0, :] = vsolFunc(1, yco, t, Re, a)
    # print(velbc['uTop'].dtype)
    return velbc


def cauchyProblem(Re, a, N, dt, tEnd, fuFunc, fvFunc, usolFunc, vsolFunc,timeScheme,saveSchedule):

    
    # Simulation Setting
    Lx, Ly = 1, 1
    Nx, Ny = N, N

    dx = Lx / Nx
    dy = Ly / Ny

    Nu_old = np.zeros((Nx-1, Ny))
    Nv_old = np.zeros((Nx, Ny-1))

    xce = (np.arange(1, Nx + 1) - 0.5) * dx
    yce = (np.arange(1, Ny + 1) - 0.5) * dy
    xco = np.arange(0, Nx + 1) * dx
    yco = np.arange(0, Ny + 1) * dy

    Xce, Yce = np.meshgrid(xce, yce)

    # Initialize the memory for flow field
    u = np.full((Nx + 1, Ny + 2), np.nan)
    v = np.full((Nx + 2, Ny + 1), np.nan)
    p = np.full((Nx + 1, Ny + 1), np.nan)

    # For the current boundary condition
    velbc = {
        'uTop': np.full((u.shape[0], 1), np.nan),
        'uBottom': np.full((u.shape[0], 1), np.nan),
        'uLeft': np.full((1, u.shape[1]), np.nan),
        'uRight': np.full((1, u.shape[1]), np.nan),
        'vTop': np.full((v.shape[0], 1), np.nan),
        'vBottom': np.full((v.shape[0], 1), np.nan),
        'vLeft': np.full((1, v.shape[1]), np.nan),
        'vRight': np.full((1, v.shape[1]), np.nan)
    }

    # For the boundary condition at one time step ahead
    velbc1 = velbc.copy()

    # Initialization of the flow field
    t = 0
    uX, uY = np.meshgrid(xco[1:-1], yce)
    usol = usolFunc(uX.T, uY.T, t, Re, a)

    vX, vY = np.meshgrid(xce, yco[1:-1])
    vsol = vsolFunc(vX.T, vY.T, t, Re, a)

    pX, pY = np.meshgrid(xce, yce)
    # p = psolFunc(pX.T, pY.T, t, Re, a)

    u[1:-1, 1:-1] = usol
    v[1:-1, 1:-1] = vsol
    timepoint = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    timepoint = timepoint.split(" ")
    folderName = os.path.join(r'C:\Users\JLS20\Desktop\anaconda安装截图\Liquid_Caivity_Flow\fft_neumman\cavityDataSet',\
    "homogeneous_" + "_timeScheme="+ timeScheme + "_Re={:.2f}".format(Re)+"_date="+timepoint[0]+"_"+timepoint[1].replace(":","_"))
    os.makedirs(folderName,exist_ok = True)
    np.save(os.path.join(folderName,"u_t = {:.6f}.npy".format(t)),usol)
    np.save(os.path.join(folderName,"v_t = {:.6f}.npy".format(t)),vsol)

    velbc = getVelocityBoundaryCondition(velbc, usolFunc, vsolFunc, xco, yco, xce, yce, t, Re, a)

    u[:, 0] = 2 * velbc1['uBottom'].flatten() - u[:, 1]
    v[:, 0] = velbc1['vBottom'].flatten()
    u[:, -1] = 2 * velbc1['uTop'].flatten() - u[:, -2]
    v[:, -1] = velbc1['vTop'].flatten()
    u[0, :] = velbc1['uLeft'].flatten()
    v[0, :] = 2 * velbc1['vLeft'].flatten() - v[1, :]
    u[-1, :] = velbc1['uRight'].flatten()
    v[-1, :] = 2 * velbc1['vRight'].flatten() - v[-2, :]

    uori = (u[:-1, 1:-1] + u[1:, 1:-1]) / 2
    vori = (v[1:-1, :-1] + v[1:-1, 1:]) / 2

    inflow = np.sum(velbc['vBottom'][1:-1]) * dx + np.sum(velbc['uLeft'][0, 1:-1]) * dy
    outflow = np.sum(velbc['vTop'][1:-1]) * dx + np.sum(velbc['uRight'][0, 1:-1]) * dy
    # assert abs(inflow - outflow) < 1.0e-10 # np.finfo(float).eps, "Inflow flux must match the outflow flux."

    tbar = tqdm(total = int(tEnd / dt),desc = "generating groundTruth",leave = False)
    idx = 0
    while t < tEnd:
        velbc = getVelocityBoundaryCondition(velbc, usolFunc, vsolFunc, xco, yco, xce, yce, t, Re, a)
        velbc1 = getVelocityBoundaryCondition(velbc1, usolFunc, vsolFunc, xco, yco, xce, yce, t + dt, Re, a)

        uForce = fuFunc(uX.T, uY.T, t, Re, a)
        vForce = fvFunc(vX.T, vY.T, t, Re, a)

        # if timeScheme == 'Euler':
        #     u, v, p = updateVelocityField_Euler(u, v, Nx, Ny, dx, dy, Re, dt, velbc, 'dct', uForce, vForce, velbc1)
        if timeScheme == 'CNAB':
           u, v, p,Nu_old,Nv_old = update_velocity_field_CNAB(u, v, Nu_old, Nv_old, Nx, Ny, dx, dy, Re, dt, velbc, uForce, vForce, velbc1)
        elif timeScheme == 'RK3':
           u, v, p , Nu_old, Nv_old= update_velocity_field_RK3(u, v,Nu_old, Nv_old, Nx, Ny, dx, dy, Re, dt, velbc, uForce, vForce, velbc1)
        # else:
        #     raise ValueError('timeScheme is not recognized')


        t += dt
        idx += 1

        uce = (u[:-1, 1:-1] + u[1:, 1:-1]) / 2
        vce = (v[1:-1, :-1] + v[1:-1, 1:]) / 2

        # print(t)
        if idx in saveSchedule:
            np.save(os.path.join(folderName,"u_t = {:.6f}.npy".format(t)),u)
            np.save(os.path.join(folderName,"v_t = {:.6f}.npy".format(t)),v)
            np.save(os.path.join(folderName,"p_t = {:.6f}.npy".format(t)),p)


        CFL = np.max([u.ravel() / dx, v.ravel() / dy]) * dt
        tbar.update(1)
    tbar.close()
    final_velocity = np.concatenate([np.expand_dims(uori,axis = 0),np.expand_dims(vori,axis = 0),\
                                     np.expand_dims(uce,axis = 0),np.expand_dims(vce,axis = 0),\
                                     np.expand_dims(p,axis = 0)],axis = 0)
    
    return final_velocity

def accuracy_validate():
    N = 512
    tEnd = 1.0 #
    dt = 2.0e-3
    psolFunc = lambda x, y, t, Re, a: t**2 * (x - 0.5)
    usolFunc = lambda x, y, t, Re, a: -128.0 * (t**2) * (x**2) * (x - 1.0)**2 * y * (y - 1.0) * (2.0 * y - 1.0)
    vsolFunc = lambda x, y, t, Re, a: 128*(t**2)*(y**2)*(y-1.0)**2 * x * (x - 1.0)*(2.0*x-1.0)
    # 将第一个表达式转换为lambda表达式形式，定义为fu函数
    fuFunc = lambda x, y, t, Re, a: (
            (256 * t ** 2 * x ** 2 * (x - 1) ** 2 * (y - 1)) / 5 +
            (128 * t ** 2 * x ** 2 * (2 * y - 1) * (x - 1) ** 2) / 5 +
            (256 * t ** 2 * x ** 2 * y * (x - 1) ** 2) / 5 +
            (128 * t ** 2 * x ** 2 * y * (2 * y - 1) * (y - 1)) / 5 +
            (128 * t ** 2 * y * (2 * y - 1) * (x - 1) ** 2 * (y - 1)) / 5 +
            (256 * t ** 2 * x * y * (2 * x - 2) * (2 * y - 1) * (y - 1)) / 5 -
            256 * t * x ** 2 * y * (2 * y - 1) * (x - 1) ** 2 * (y - 1) -
            32768 * t ** 4 * x ** 3 * y ** 3 * (2 * x - 1) * (x - 1) ** 3 * (y - 1) ** 3 +
            65536 * t ** 4 * x ** 3 * y ** 2 * (2 * y - 1) ** 2 * (x - 1) ** 4 * (y - 1) ** 2 +
            65536 * t ** 4 * x ** 4 * y ** 2 * (2 * y - 1) ** 2 * (x - 1) ** 3 * (y - 1) ** 2 -
            49152 * t ** 4 * x ** 3 * y ** 2 * (2 * x - 1) * (2 * y - 1) * (x - 1) ** 3 * (y - 1) ** 3 -
            49152 * t ** 4 * x ** 3 * y ** 3 * (2 * x - 1) * (2 * y - 1) * (x - 1) ** 3 * (y - 1) ** 2
    )
    
    # 将第二个表达式转换为lambda表达式形式，定义为fv函数
    fvFunc = lambda x, y, t, Re, a: (
            256 * t * x * y ** 2 * (2 * x - 1) * (x - 1) * (y - 1) ** 2 -
            (128 * t ** 2 * y ** 2 * (2 * x - 1) * (y - 1) ** 2) / 5 -
            (256 * t ** 2 * x * y ** 2 * (y - 1) ** 2) / 5 -
            (128 * t ** 2 * x * y ** 2 * (2 * x - 1) * (x - 1)) / 5 -
            (128 * t ** 2 * x * (2 * x - 1) * (x - 1) * (y - 1) ** 2) / 5 -
            (256 * t ** 2 * x * y * (2 * x - 1) * (2 * y - 2) * (x - 1)) / 5 -
            (256 * t ** 2 * y ** 2 * (x - 1) * (y - 1) ** 2) / 5 -
            32768 * t ** 4 * x ** 3 * y ** 3 * (2 * y - 1) * (x - 1) ** 3 * (y - 1) ** 3 +
            65536 * t ** 4 * x ** 2 * y ** 3 * (2 * x - 1) ** 2 * (x - 1) ** 2 * (y - 1) ** 4 +
            65536 * t ** 4 * x ** 2 * y ** 4 * (2 * x - 1) ** 2 * (x - 1) ** 2 * (y - 1) ** 3 -
            49152 * t ** 4 * x ** 2 * y ** 3 * (2 * x - 1) * (2 * y - 1) * (x - 1) ** 3 * (y - 1) ** 3 -
            49152 * t ** 4 * x ** 3 * y ** 3 * (2 * x - 1) * (2 * y - 1) * (x - 1) ** 2 * (y - 1) ** 3
    )
    a =  2 * np.pi # [2* np.pi,3 * np.pi]
    Re = 10.0
    timeScheme = "CNAB"
    saveSchedule = (np.linspace(0,1,41)/dt).astype(np.int64)
    # print(saveSchedule)
    # tzips = cauchyProblem(Re,a,N,dt,tEnd,fuFunc,fvFunc,usolFunc,vsolFunc,timeScheme,saveSchedule)
    # print("-------------------------------")
    # print(tzips.shape)
    final_error1,uSim1,uSol1 = checkNSSolverError(Re,a,N,dt,tEnd,fuFunc,fvFunc,usolFunc,vsolFunc,psolFunc,timeScheme)
    # final_error2,uSim2,uSol2 = checkNSSolverError(Re,a,N,dt/2,tEnd,fuFunc,fvFunc,usolFunc,vsolFunc,psolFunc,timeScheme)
    # print("relative_rmse = {:.16f}".format(np.sqrt(np.mean(final_error1**2))))
    # print(final_error1.shape,final_error2.shape)
    # print(np.log(np.sqrt(np.mean(final_error1**2) / np.mean(final_error2**2)))/np.log(2.0))
    # print(np.log(np.max(np.abs(final_error1)) / np.max(np.abs(final_error2)))/np.log(2.0))

def generate_data():
    N = 256
    tEnd = 1.0e-1 #
    dt = 1.0e-3
    usolFunc = lambda x,y,t,Re,a:np.exp(-((x-0)**2 + (y-0.5)**2)/0.001)
    vsolFunc = lambda x,y,t,Re,a:np.zeros_like(x)
    fuFunc = lambda x,y,t,Re,a:np.zeros_like(x)
    fvFunc = lambda x,y,t,Re,a:np.zeros_like(x)
    a =  2 * np.pi
    Re = 1000.0
    timeScheme = "CNAB"
    saveSchedule = (np.linspace(0,1,41)/dt).astype(np.int64)
    tzips = cauchyProblem(Re,a,N,dt,tEnd,fuFunc,fvFunc,usolFunc,vsolFunc,timeScheme,saveSchedule)
    Nx,Ny = 256,256
    dx,dy = 1/Nx,1/Ny
    xce = (np.arange(1, Nx + 1) - 0.5) * dx
    yce = (np.arange(1, Ny + 1) - 0.5) * dy
    xco = np.arange(0, Nx + 1) * dx
    yco = np.arange(0, Ny + 1) * dy
    Xce, Yce = np.meshgrid(xce, yce)
    import matplotlib.pyplot as plt
    plt.streamplot(Xce, Yce,tzips[0,:,:].T,tzips[1,:,:].T, density=2, linewidth=1, color='blue')
    plt.title("origin,t = {:.6f}".format(dt))
    plt.show()
    plt.streamplot(Xce, Yce,tzips[2,:,:].T,tzips[3,:,:].T, density=2, linewidth=1, color='blue')
    plt.title("final,t = {:.6f}".format(tEnd))
    plt.show()
if __name__ == "__main__":
    accuracy_validate()