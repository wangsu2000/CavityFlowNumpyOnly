from neumman_3d import KroneckerBMM
import torch
from tqdm import trange
# U:(257,258,258)
# V:(258,257,258)
# W:(258,258,257)
# P:(256,256,256)
# U 的六个边值:U(0,:,:),U(-1,:,:),U(:,0,:),U(:,-1,:),U(:,:,0),U(:,:,-1)
### 程序分为:
#   修正边界条件
#   计算对流项扩散项,得到新的流速
#   计算压力Poisson方程
#   对流速压力校正
# 边界条件的存储格式:
# python 字典:
# boundaries{"Us**"} -> U[0,:,:];boundaries{"U**e"} -> U[:,:,-1]


# 提取边界值,而不是赋值
def boundary_prepare(U,V,W):
    boundaries = {}
    boundaries['Us**'],boundaries['Ue**'],boundaries['Vs**'],boundaries['Ve**'],boundaries['Ws**'],boundaries['We**'] =\
    U[0,:,:],U[-1,:,:],(V[0,:,:]+V[1,:,:])/2.0,(V[-1,:,:]+V[-2,:,:])/2.0,(W[0,:,:]+W[1,:,:])/2.0,(W[-1,:,:]+W[-2,:,:])/2.0
    boundaries['U*s*'],boundaries['U*e*'],boundaries['V*s*'],boundaries['V*e*'],boundaries['W*s*'],boundaries['W*e*'] =\
    (U[:,0,:]+U[:,1,:])/2.0,(U[:,-1,:]+U[:,-2,:])/2.0,V[:,0,:],V[:,-1,:],(W[:,0,:]+W[:,1,:])/2.0,(W[:,-1,:]+W[:,-2,:])/2.0
    boundaries['U**s'],boundaries['U**e'],boundaries['V**s'],boundaries['V**e'],boundaries['W**s'],boundaries['W**e'] =\
    (U[:,:,0]+U[:,:,1])/2.0,(U[:,:,-1]+U[:,:,-2])/2.0,(V[:,:,0]+V[:,:,1])/2.0,(V[:,:,-1]+V[:,:,-2])/2.0,W[:,:,0],W[:,:,-1]
    return boundaries

# 用边界值处理U,V,W:
def boundary_process(U, V, W, boundaries):
    U[0, :, :] = boundaries['Us**']
    U[-1, :, :] = boundaries['Ue**']
    U[:, 0, :] = 2.0 * boundaries['U*s*'] - U[:, 1, :]
    U[:, -1, :] = 2.0 * boundaries['U*e*'] - U[:, -2, :]
    U[:, :, 0] = 2.0 * boundaries['U**s'] - U[:, :, 1]
    U[:, :, -1] = 2.0 * boundaries['U**e'] - U[:, :, -2]

    V[0, :, :] = 2.0 * boundaries['Vs**'] - V[1, :, :]
    V[-1, :, :] = 2.0 * boundaries['Ve**'] - V[-2, :, :] 
    V[:, 0, :] = boundaries['V*s*']
    V[:, -1, :] = boundaries['V*e*']
    V[:, :, 0] = 2.0 * boundaries['V**s'] - V[:, :, 1]
    V[:, :, -1] = 2.0 * boundaries['V**e'] - V[:, :, -2]

    W[0, :, :] = 2.0 * boundaries['Ws**'] - W[1, :, :]
    W[-1, :, :] = 2.0 * boundaries['We**'] - W[-2, :, :]
    W[:, 0, :] = 2.0 * boundaries['W*s*'] - W[:, 1, :]
    W[:, -1, :] = 2.0 * boundaries['W*e*'] - W[:, -2, :]
    W[:, :, 0] = boundaries['W**s']
    W[:, :, -1] = boundaries['W**e']

    return U, V, W

# diffusion项的计算 
def calculate_diffusion_term(X,boundaries,mode,h):
    L = (X[:-2, 1:-1, 1:-1] - 2 * X[1:-1, 1:-1, 1:-1] + X[2:, 1:-1, 1:-1]) / h**2 +\
        (X[1:-1, :-2, 1:-1] - 2 * X[1:-1, 1:-1, 1:-1] + X[1:-1, 2:, 1:-1]) / h**2 +\
        (X[1:-1, 1:-1, :-2] - 2 * X[1:-1, 1:-1, 1:-1] + X[1:-1, 1:-1, 2:]) / h**2 
    if mode == 'U':
        L[:,0,:] += 2 * boundaries["U*s*"][1:-1,1:-1] / h**2 
        L[:,-1,:] += 2 * boundaries["U*e*"][1:-1,1:-1] / h**2
        L[:,:,0] += 2 * boundaries["U**s"][1:-1,1:-1] / h**2 
        L[:,:,-1] += 2 * boundaries["U**e"][1:-1,1:-1] / h**2
    if mode == 'V':
        # print(L.shape,boundaries["Vs**"].shape)
        L[0,:,:] += 2 * boundaries["Vs**"][1:-1,1:-1] / h**2 
        L[-1,:,:] += 2 * boundaries["Ve**"][1:-1,1:-1] / h**2
        L[:,:,0] += 2 * boundaries["V**s"][1:-1,1:-1] / h**2 
        L[:,:,-1] += 2 * boundaries["V**e"][1:-1,1:-1] / h**2
    if mode == 'W':
        L[0,:,:] += 2 * boundaries["Ws**"][1:-1,1:-1] / h**2 
        L[-1,:,:] += 2 * boundaries["We**"][1:-1,1:-1] / h**2
        L[:,0,:] += 2 * boundaries["W*s*"][1:-1,1:-1] / h**2 
        L[:,-1,:] += 2 * boundaries["W*e*"][1:-1,1:-1] / h**2
    return L
    
# convection项的计算
# co:corner;ce:center
# U,V,W物理意义都是面心的流速
# 压强位置在体心
def calculate_convection_term(U,V,W,mode,h):
    
    if mode == 'U':
        uce = (U[:-1, 1:-1, 1:-1] + U[1:, 1:-1, 1:-1]) / 2.0
        uco4v = (U[:,-1,:] + U[:,1:,:]) / 2.0 
        uco4w = (U[:,:,:-1] + U[:,:,1:]) / 2.0 
        vco = (V[:-1, :, :] + V[1:, :, :]) / 2.0
        wco = (W[:-1, :, :] + W[1:, :, :]) / 2.0
        uuce = uce * uce
        uvco = uco4v * vco #体心流速
        uwco = uco4w * wco # 体心流速
        # print(uuce.shape,uvco.shape,uwco.shape)
        Nu = (uuce[1:, :, :] - uuce[:-1, :, :]) / h + (uvco[1:-1 , 1:,1:-1] - uvco[1:-1, :-1,1:-1]) / h +\
        (uwco[1:-1 ,1:-1, 1:] - uwco[1:-1, 1:-1, :-1]) / h
        return Nu
    if mode == 'V':
        vce = (V[1:-1,:-1, 1:-1] + V[ 1:-1, 1:, 1:-1]) / 2.0
        vco4u = (V[:-1,:,:] + V[1:,:,:]) / 2.0 
        vco4w = (V[:,:,:-1] + V[:,:,1:]) / 2.0 
        uco = (U[:,:-1, :] + U[ :,1:, :]) / 2.0
        wco = (W[:,:-1, :] + W[:,1:, :]) / 2.0
        vvce = vce * vce
        vuco = vco4u * uco #体心流速
        vwco = vco4w * wco # 体心流速
        Nv = (vvce[:,1:, :] - vvce[ :,:-1, :]) / h + (vuco[1:,1:-1 ,1:-1] - vuco[ :-1,1:-1,1:-1]) / h +\
        (vwco[1:-1 ,1:-1, 1:] - vwco[1:-1, 1:-1, :-1]) / h
        return Nv
    if mode == 'W':
        wce = (W[1:-1, 1:-1, :-1] + W[1:-1, 1:-1, 1:]) / 2.0
        wco4u = (W[:-1,:,:] + W[1:,:,:]) / 2.0 
        wco4v = (W[:,:-1,:] + W[:,1:,:]) / 2.0 
        uco = (U[:,:, :-1] + U[ :, :,1:]) / 2.0
        vco = (V[:,:, :-1] + V[ :, :,1:]) / 2.0
        wwce = wce * wce
        wuco = wco4u * uco #体心流速
        wvco = wco4v * vco # 体心流速
        Nw = (wwce[:,:,1:] - wwce[ :,:,:-1]) / h + (wvco[1:-1,1: ,1:-1] - wvco[1:-1, :-1,1:-1]) / h +\
        (wuco[1:,1:-1 ,1:-1] - wuco[:-1,1:-1, 1:-1]) / h
        return Nw

# 因为boundaries基本不时变,
# 因此循环使用,U,V,W输入值不包含上一时刻
def inner_circulation(U,V,W,Ucache,Vcache,Wcache,boundaries,Re,dt,h,pressure_poisson_solver,verbose = False):
    
    U,V,W, = boundary_process(U, V, W, boundaries)
    U[1:-1,1:-1,1:-1] = Ucache[1:-1,1:-1,1:-1] + dt/Re * calculate_diffusion_term(U,boundaries,'U',h) -\
    dt * calculate_convection_term(U,V,W,'U',h)
    V[1:-1,1:-1,1:-1] = Vcache[1:-1,1:-1,1:-1] + dt/Re * calculate_diffusion_term(V,boundaries,'V',h) -\
    dt * calculate_convection_term(U,V,W,'V',h)
    W[1:-1,1:-1,1:-1] = Wcache[1:-1,1:-1,1:-1] + dt/Re * calculate_diffusion_term(W,boundaries,'W',h) -\
    dt * calculate_convection_term(U,V,W,'W',h)
    b = (U[1:, 1:-1,1:-1] - U[:-1, 1:-1,1:-1]) / h + (V[1:-1, 1:,1:-1] - V[1:-1, :-1,1:-1]) / h + (W[1:-1,1:-1,1:] - W[1:-1,1:-1,:-1]) / h
    with torch.no_grad():
        dp = pressure_poisson_solver.forward(b,h).view(b.shape)
        # print(dp.shape)
    U[1:-1, 1:-1, 1:-1] = U[1:-1, 1:-1, 1:-1] - (dp[1:, :, :] - dp[:-1, :, :]) / h
    V[1:-1, 1:-1, 1:-1] = V[1:-1, 1:-1, 1:-1] - (dp[:, 1:, :] - dp[:, :-1, :]) / h
    W[1:-1, 1:-1, 1:-1] = W[1:-1, 1:-1, 1:-1] - (dp[:, :, 1:] - dp[:, :, :-1]) / h
    if verbose:
        bnew = (U[1:, 1:-1,1:-1] - U[:-1, 1:-1,1:-1]) / h + (V[1:-1, 1:,1:-1] - V[1:-1, :-1,1:-1]) / h + (W[1:-1,1:-1,1:] - W[1:-1,1:-1,:-1]) / h
        return U,V,W,bnew
    else:
        return U,V,W,dp
    
def main(timeSteps,boundaries,Ucache,Vcache,Wcache,h,dt,Re,pressure_poisson_solver,inner_epoch):
    U = Ucache.clone().cuda().to(torch.float64)
    V = Vcache.clone().cuda().to(torch.float64)
    W = Wcache.clone().cuda().to(torch.float64)
    for oiter in trange(timeSteps,desc = "inference"):
        for niter in range(inner_epoch):
            U,V,W,bnew = inner_circulation(U,V,W,Ucache,Vcache,Wcache,boundaries,Re,dt,h,pressure_poisson_solver,verbose = False)
            rU = torch.norm(U - Ucache).cpu().numpy()
            rV = torch.norm(V - Vcache).cpu().numpy()
            rW = torch.norm(W - Wcache).cpu().numpy()
            # print(f"oiter = {oiter}, niter = {niter}, mse_divergence = {torch.mean(bnew**2).cpu().numpy()}, prev_UVP_residual = {rU},{rV},{rW}")
        print(f"oiter = {oiter}, prev_UVP_residual = {rU},{rV},{rW}")
        Ucache = U.clone()
        Vcache = V.clone()
        Wcache = W.clone()
    return U,V,W
if __name__ == "__main__":
    N = 128
    Ucache = torch.zeros((N+1,N+2,N+2)).cuda().to(torch.float64)
    Vcache = torch.zeros((N+2,N+1,N+2)).cuda().to(torch.float64)
    Wcache = torch.zeros((N+2,N+2,N+1)).cuda().to(torch.float64)
    Ucache[0,0:32,0:32] = 1.0
    Wcache[96:128,96:128,-1] = 1.0
    boundaries = boundary_prepare(Ucache,Vcache,Wcache)
    Re = 100
    dt = 0.0002
    h = 1.0/N
    inner_epoch = 20
    timeSteps = 200
    pressure_poisson_solver = KroneckerBMM(N,N,N)
    U,V,W = main(timeSteps,boundaries,Ucache,Vcache,Wcache,h,dt,Re,pressure_poisson_solver,inner_epoch)



    print(U.shape,V.shape,W.shape)

    U = (U[1:,1:-1,1:-1] + U[:-1,1:-1,1:-1]).cpu().numpy() / 2.0
    V = (V[1:-1,1:,1:-1] + V[1:-1,:-1,1:-1]).cpu().numpy() / 2.0
    W = (W[1:-1,1:-1,1:] + W[1:-1,1:-1,:-1]).cpu().numpy() / 2.0
    import numpy as np
    np.savez('resTemp.npz', U=U, V=V, W=W)
    

     



 
