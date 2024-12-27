from helmholtzSolver import helmholtzStiffnessAssemble,convertAndSolve,reshapeMat,fft_neumann_poisson
from testFunc import TaylorVortex
import numpy as np
from scipy.sparse.linalg import splu
from scipy.signal import convolve2d
from tqdm import trange
import pandas as pd
laplace_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])
def calculateDiffusionTerm(uv,N):
    h = 1.0 / N
    return convolve2d(uv, laplace_kernel, mode='valid') / (h**2)

def calculateConvectionTerm(u,v,N):
    dx,dy = 1.0 / N,1.0 / N
    uce = 0.5 * (u[0:-1, 1:-1] + u[1:, 1:-1])
    uco = 0.5 * (u[:, 0:-1] + u[:, 1:])
    vco = 0.5 * (v[0:-1, :] + v[1:, :])
    vce = 0.5 * (v[1:-1, 0:-1] + v[1:-1, 1:])

    uuce = uce * uce
    uvco = uco * vco
    vvce = vce * vce

    Nu = (uuce[1:, :] - uuce[0:-1, :]) / dx + (uvco[1:-1, 1:] - uvco[1:-1, 0:-1]) / dy
    Nv = (vvce[:, 1:] - vvce[:, 0:-1]) / dy + (uvco[1:, 1:-1] - uvco[0:-1, 1:-1]) / dx
    return Nu,Nv

def calculateExplicitPressureGradient(p,N):
    dx,dy = 1.0 / N,1.0 / N
    px = (p[1:, :] - p[0:-1, :]) / dx
    py = (p[:, 1:] - p[:, 0:-1]) / dy
    return px,py

def bAssemble(semi_tpn,taylor,timePoint,dt,Re,N,u,v,uNu,vNv,savMode = False):
    h = 1.0 / N
    diffusionU = calculateDiffusionTerm(u,N) / (2.0 * Re ) * dt
    diffusionV = calculateDiffusionTerm(v,N) / (2.0 * Re ) * dt
    Nu,Nv = calculateConvectionTerm(uNu,vNv,N)
    pressureX,pressureY = calculateExplicitPressureGradient(-semi_tpn * 2.0,N)
    nonHomogeneousU = dt * taylor.query(timePoint + dt / 2.0,"fu")
    nonHomogeneousV = dt * taylor.query(timePoint + dt / 2.0,"fv")
    if not savMode:
        rhsU = nonHomogeneousU - dt * Nu +  pressureX + diffusionU  + u[1:-1,1:-1]
        rhsV = nonHomogeneousV - dt * Nv +  pressureY + diffusionV  + v[1:-1,1:-1]
    else:
        rhsU = -dt * Nu
        rhsV = -dt * Nv
    return rhsU,rhsV

def pressureCorrection(rU,rV,anaU,anaV,semi_tpn,N,dt):
    h = 1.0 / N
    px,py = calculateExplicitPressureGradient(semi_tpn,N)
    rhsU = rU.copy()
    rhsU[1:-1,1:-1] += px
    rhsV = rV.copy()
    rhsV[1:-1,1:-1] += py
    rhsU[:,0] = anaU[:,0] + anaU[:,1] - rhsU[:,1]
    rhsU[:,-1] = anaU[:,-1] + anaU[:,-2] - rhsU[:,-2]
    rhsV[0,:] = anaV[0,:] + anaV[1,:] - rhsV[1,:]
    rhsV[-1,:] = anaV[-1,:] + anaV[-2,:] - rhsV[-2,:]
    
    b = ((rhsU[1:, 1:-1] - rhsU[0:-1, 1:-1]) / h + (rhsV[1:-1, 1:] - rhsV[1:-1, 0:-1]) / h)
    delta_p = fft_neumann_poisson(b,h)
    px = (delta_p[1:, :] - delta_p[0:-1, :]) / h 
    py = (delta_p[:, 1:] - delta_p[:, 0:-1]) / h 
    rhsU[1:-1, 1:-1] = rhsU[1:-1, 1:-1] - px 
    rhsV[1:-1, 1:-1] = rhsV[1:-1, 1:-1] - py  
    residual = ((rhsU[1:, 1:-1] - rhsU[0:-1, 1:-1]) / h + (rhsV[1:-1, 1:] - rhsV[1:-1, 0:-1]) / h)
    # print("divergence = {:.16f}".format(np.linalg.norm(residual)))
    rhsU[:,0] = anaU[:,0] + anaU[:,1] - rhsU[:,1]
    rhsU[:,-1] = anaU[:,-1] + anaU[:,-2] - rhsU[:,-2]
    rhsV[0,:] = anaV[0,:] + anaV[1,:] - rhsV[1,:]
    rhsV[-1,:] = anaV[-1,:] + anaV[-2,:] - rhsV[-2,:]
    return rhsU,rhsV,delta_p

def updateParameters(errorLoader, u, v, semi_tpn, timePoint, taylor, N, dt, Re, luU=None, luV=None, iterMax=50):
    """
    主函数,利用非线性迭代获取每一步
    的新流速和压强,收敛条件是:（u^n+u^{n+1})/2决定
    的非线性对流项已经收敛
    """
    ### 第一步
    rhsU, rhsV = bAssemble(semi_tpn, taylor, timePoint, dt, Re, N, u, v, u, v)  # 后面两个是用来算非线性对流项的
    anaU, anaV = taylor.query(timePoint + dt, "u"), taylor.query(timePoint + dt, "v")  # 解析解作为参考
    uMana,vMana = taylor.query(timePoint + dt, "um"), taylor.query(timePoint + dt, "vm") # 体素中心值
    Lu, rhs = helmholtzStiffnessAssemble(N, Re, dt, anaU, rhsU, 'u', False)
    if luU is None:
        rU, luU = convertAndSolve(Lu, rhs, luU)
    else:
        rU = convertAndSolve(Lu, rhs, luU)
    rU = reshapeMat(rU, anaU)
    Lv, rhs = helmholtzStiffnessAssemble(N, Re, dt, anaV, rhsV, 'v', False)
    if luV is None:
        rV, luV = convertAndSolve(Lv, rhs, luV)
    else:
        rV = convertAndSolve(Lv, rhs, luV)
    rV = reshapeMat(rV, anaV)
    rU, rV, rP = pressureCorrection(rU, rV, anaU, anaV, semi_tpn, N, dt)

    ### 后续步
    for k in trange(iterMax, desc="niter", leave=False):
        oU, oV, oP = rU.copy(), rV.copy(), rP.copy()
        uNu = (rU + u) * 0.5
        vNv = (rV + v) * 0.5
        rhsU, rhsV = bAssemble(semi_tpn, taylor, timePoint, dt, Re, N, u, v, uNu, vNv)
        rhs = helmholtzStiffnessAssemble(N, Re, dt, anaU, rhsU, 'u', True)
        rU = convertAndSolve(Lu, rhs, luU)
        rU = reshapeMat(rU, anaU)
        rhs = helmholtzStiffnessAssemble(N, Re, dt, anaV, rhsV, 'v', True)
        rV = convertAndSolve(Lv, rhs, luV)
        rV = reshapeMat(rV, anaV)
        rU, rV, rP = pressureCorrection(rU, rV, anaU, anaV, semi_tpn, N, dt)

    print("||\deltaU|| = {:.16f}".format(np.linalg.norm(oU - rU)), end=",")
    print("||\deltaV|| = {:.16f}".format(np.linalg.norm(oV - rV)), end=",")
    print("||\deltaP|| = {:.16f}".format(np.linalg.norm(oP - rP)))

    uce = 0.5 * (rU[0:-1, 1:-1] + rU[1:, 1:-1])
    vce = 0.5 * (rV[1:-1, 0:-1] + rV[1:-1, 1:])
    mseU = np.mean(np.square(uce - uMana))
    mseV = np.mean(np.square(vce - vMana))
    # import pdb;pdb.set_trace()
    maxU = np.max(np.abs(uce - uMana))
    maxV = np.max(np.abs(vce - vMana))
    oriMU = np.max(np.abs(uMana))
    oriMV = np.max(np.abs(vMana))

    # mseU = np.mean(np.square(rU[1:-1, 1:-1] - anaU[1:-1, 1:-1]))
    # mseV = np.mean(np.square(rV[1:-1, 1:-1] - anaV[1:-1, 1:-1]))
    # maxU = np.max(np.abs(rU[1:-1, 1:-1] - anaU[1:-1, 1:-1]))
    # maxV = np.max(np.abs(rV[1:-1, 1:-1] - anaV[1:-1, 1:-1]))

    print("mse of u = {:.16f},mse of v = {:.16f}".format(mseU, mseV))
    print("MAXIMUM of u = {:.16f},MAXIMUM OF v = {:.16f},ORIGIN_MAXIMUM_U = {:.16f},ORIGIN_MAXIMUM_U = {:.16f}".format(maxU, maxV,oriMU,oriMV))
    # import pdb;pdb.set_trace()
    # 将相关变量值写入log.txt文件
    # with open("log{:.8f}.txt".format(dt), mode="a") as f:
    #     f.write("timeAt {:.16f},".format(timePoint + dt))
    #     f.write("mse of u = {:.16f},".format(mseU))
    #     f.write("mse of v = {:.16f},".format(mseV))
    #     f.write("MAXIMUM of u = {:.16f},".format(maxU))
    #     f.write("MAXIMUM of v = {:.16f}\n".format(maxV))
    errorLoader.append({
    "timePoint": timePoint + dt,
    "mseU": mseU,
    "mseV": mseV,
    "maxU": maxU,
    "maxV": maxV,
    "oriMU":oriMU,
    "oriMV":oriMV,
    "pattern":"nonlinear",
    "instance":"ex1",
    "interval":dt
    })
    return errorLoader, rU, rV, rP, luU, luV

def updateSAVParameters(errorLoader, u, v, qn, uNu, vNv, semi_tpn, timePoint, taylor, N, dt, Re, T, luU, luV, tol = 1.0e-14):
    """
    uNu,vNv分别是u,v再t_{n+1/2}处的显格式逼近
    luU,lvV都需要保证非空
    """
    ### 前两次握手
    qnplus = np.exp(-(timePoint + dt)/T)
    Nu,Nv = calculateConvectionTerm(uNu,vNv,N)
    anaU, anaV = taylor.query(timePoint + dt, "u"), taylor.query(timePoint + dt, "v")  # 解析解作为参考
    uMana,vMana = taylor.query(timePoint + dt, "um"), taylor.query(timePoint + dt, "vm") # 体素中心值
    ### 迭代致qnplus收敛
    resume = qnplus
    flag = False
    for inner in range(15):
        exponent_coefficent = (qnplus + qn) / 2.0 / np.exp(-(timePoint + dt * 0.5)/T) # 获取非q^{n+1}项系数
        rhsU, rhsV = bAssemble(semi_tpn, taylor, timePoint, dt, Re, N, u, v,exponent_coefficent * uNu,exponent_coefficent * vNv)  # 后面两个是用来算非线性对流项的
        rhs = helmholtzStiffnessAssemble(N, Re, dt, anaU, rhsU, 'u', True)
        rU = convertAndSolve(rhs, rhs, luU)
        rU = reshapeMat(rU, anaU)
        rhs = helmholtzStiffnessAssemble(N, Re, dt, anaV, rhsV, 'v', True)
        rV = convertAndSolve(rhs, rhs, luV)
        rV = reshapeMat(rV, anaV)
        trilinear_scalar_constant = (np.mean((rU[1:-1,1:-1] + u[1:-1,1:-1]) * (Nu)) + np.mean((rV[1:-1,1:-1] + v[1:-1,1:-1]) * Nv)) / np.exp(-(timePoint + 0.5 * dt)/T) / 2.0
        denominator = (2 * T + dt)
        numerator = (2 * T * dt * trilinear_scalar_constant + 2 * T * qn - qn * dt)
        qnplus = numerator / denominator
        residual = np.abs(resume - qnplus)
        if residual < tol:
            break
        else:
            resume = qnplus
        
    print("iteration_tol = {:.16f}".format(residual))


    ### 第三次握手
    exponent_coefficent = (qnplus + qn) / 2.0 / np.exp(-(timePoint + dt * 0.5)/T) # 获取非q^{n+1}项系数
    print("time = {:.8f},SAVcoef = {:.16f}".format(timePoint+dt*0.5,exponent_coefficent))
    rhsU, rhsV = bAssemble(semi_tpn, taylor, timePoint, dt, Re, N, u, v,exponent_coefficent * uNu,exponent_coefficent * vNv)  # 后面两个是用来算非线性对流项的
    # rhsU, rhsV = bAssemble(semi_tpn, taylor, timePoint, dt, Re, N, u, v,uNu,vNv) # 纯粹的CNABban'yin
    rhs = helmholtzStiffnessAssemble(N, Re, dt, anaU, rhsU, 'u', True)
    rU = convertAndSolve(rhs, rhs, luU)
    rU = reshapeMat(rU, anaU)
    rhs = helmholtzStiffnessAssemble(N, Re, dt, anaV, rhsV, 'v', True)
    rV = convertAndSolve(rhs, rhs, luV)
    rV = reshapeMat(rV, anaV)
    rU, rV, rP = pressureCorrection(rU, rV, anaU, anaV, semi_tpn, N, dt)

    ### 检查误差
    # mseU = np.mean(np.square(rU[1:-1, 1:-1] - anaU[1:-1, 1:-1]))
    # mseV = np.mean(np.square(rV[1:-1, 1:-1] - anaV[1:-1, 1:-1]))
    # maxU = np.max(np.abs(rU[1:-1, 1:-1] - anaU[1:-1, 1:-1]))
    # maxV = np.max(np.abs(rV[1:-1, 1:-1] - anaV[1:-1, 1:-1]))
    uce = 0.5 * (rU[0:-1, 1:-1] + rU[1:, 1:-1])
    vce = 0.5 * (rV[1:-1, 0:-1] + rV[1:-1, 1:])
    mseU = np.mean(np.square(uce - uMana))
    mseV = np.mean(np.square(vce - vMana))
    # import pdb;pdb.set_trace()
    maxU = np.max(np.abs(uce - uMana))
    maxV = np.max(np.abs(vce - vMana))
    oriMU = np.max(np.abs(uMana))
    oriMV = np.max(np.abs(vMana))
    print("mse of u = {:.16f},mse of v = {:.16f}".format(mseU, mseV))
    errorLoader.append({
    "timePoint": timePoint + dt,
    "mseU": mseU,
    "mseV": mseV,
    "maxU": maxU,
    "maxV": maxV,
    "oriMU":oriMU,
    "oriMV":oriMV,
    "pattern":"sav",
    "instance":"ex2",
    "interval":dt
    })
    print("MAXIMUM of u = {:.16f},MAXIMUM OF v = {:.16f}".format(maxU, maxV))
    # with open("log{:.8f}.txt".format(dt), mode="a") as f:
    #     f.write("timeAt {:.16f},".format(timePoint + dt))
    #     f.write("mse of u = {:.16f},".format(mseU))
    #     f.write("mse of v = {:.16f},".format(mseV))
    #     f.write("MAXIMUM of u = {:.16f},".format(maxU))
    #     f.write("MAXIMUM of v = {:.16f}\n".format(maxV))
    return errorLoader,qnplus,rU,rV,rP
    ### 根据线性函数保证还原出三线性形式的具体表达式:
    # qn+1:{0} ->  trilinear_scalar_constant;{1} ->  trilinear_scalar_qnplus
    # import pdb;pdb.set_trace()

def savMain(a,Re,N,dt,T,iterMax = 20):
    taylor = TaylorVortex(a,Re,N)
    pn = taylor.query(0.0,"p")
    un = taylor.query(0.0,"u")
    vn = taylor.query(0.0,"v")
    semi_tpn = pn * dt / 2.0
    luU,luV = None,None
    errorLoader = []
    errorLoader, unplus,vnplus,semi_tpn,luU,luV = updateParameters(errorLoader, un,vn,semi_tpn,0,taylor,N,dt,Re,luU,luV)
    uNu = 1.5 * unplus - 0.5 * un
    vNv = 1.5 * vnplus - 0.5 * vn
    UD = {"-1":un,"0":unplus}
    VD = {"-1":vn,"0":vnplus}
    qn = np.exp(-dt/T)
    for iters in range(iterMax-1):
        errorLoader, qnplus,rU,rV,semi_tpn = updateSAVParameters(errorLoader, UD["0"], VD["0"], qn, uNu, vNv, semi_tpn, (iters + 1) * dt, taylor, N, dt, Re, T, luU, luV)
        ### 更替
        UD = {"-1":UD["0"],"0":rU}
        VD = {"-1":VD["0"],"0":rV}
        uNu = 1.5 * UD["0"] - 0.5 * UD["-1"]
        vNv = 1.5 * VD["0"] - 0.5 * VD["-1"]
        qn = qnplus
    return errorLoader
        
def nonlinearMain(a,Re,N,dt,iterMax = 10):
    taylor = TaylorVortex(a,Re,N)
    pn = taylor.query(0.0,"p")
    un = taylor.query(0.0,"u")
    vn = taylor.query(0.0,"v")
    semi_tpn = pn * dt / 2.0
    luU,luV = None,None
    errorLoader = []
    for i in range(0,iterMax):
        print("iter = {:d}".format(i+1))
        errorLoader, un,vn,semi_tpn,luU,luV = updateParameters(errorLoader, un,vn,semi_tpn,i * dt,taylor,N,dt,Re,luU,luV)
        print("Energy:{:.8f}".format(np.mean(np.square(un[1:-1,1:-1]**2)) + np.mean(np.square(vn[1:-1,1:-1]**2)) + np.mean(np.square(semi_tpn**2))))
    return errorLoader

def save_to_csv(errorLoader,savefile):
    if not errorLoader:
        print("输入的数据列表为空，无需保存。")
        return

    # 获取所有字典的键，作为表头
    columns = list(errorLoader[0].keys())

    # 将字典列表转换为DataFrame
    df = pd.DataFrame(errorLoader, columns=columns)

    # 生成CSV文件名，这里简单示例为 "output.csv"，你可以按需修改
    df.to_csv(savefile, index=False)
    # print(f"数据已成功保存到 {csv_file_name} 文件中。")

if __name__ == "__main__":
    a = 2 * np.pi
    Re = 10.0
    N = 512
    T = 1.0
    dt = 2.0e-3
    e1 = savMain(a,Re,N,dt,T,int(T/dt))
    save_to_csv(e1,"ex1,dt = {:.6f}nonlinear.csv".format(dt))
    # T = 1.0
    # dtList = [1.0e-2,1.25e-2,2.0e-2,2.5e-2,4.0e-2,5.0e-2,8.0e-2,1.0e-1]
    # for dt in dtList:
        # e1 = savMain(a,Re,N,dt,T,int(T/dt))
        # save_to_csv(e1,"ex1,dt = {:.6f}sav.csv".format(dt))
    # e2 = nonlinearMain(a,Re,N,dt,int(T/dt))
    # save_to_csv(e2,"ex3,dt = {:.6f}nonlinear.csv".format(dt))


    
    
    
    
    
    
    