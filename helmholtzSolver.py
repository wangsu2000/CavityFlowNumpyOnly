#coding=utf-8
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.signal import convolve2d
from scipy.sparse.linalg import splu


def helmholtzStiffnessAssemble(N,Re,dt,uv,b,mode,only_rhs):
    """
    
    """
    rhs = {} # 
    h = 1.0 / N
    a = [1.0 + 2.0 * dt / (Re*h*h),-dt / 2.0 / (Re*h*h)]
    if not only_rhs:
        L = {} # 
        
        
        if mode == 'u':
            for i in range(N+1):
                for j in range(N+2):
                    ind = i * (N+2) + j
                    if i == 0 or i == N:#
                        L[(ind,ind)] = 1.0
                        rhs[ind] = uv[i,j]
                    elif j == 0 or j == N+1: #
                        ins = i * (N+2) + j + 1 if j == 0 else i * (N+2) + (j - 1)
                        L[(ind,ind)] = 1.0
                        L[(ind,ins)] = 1.0
                        rhs[ind] = uv[i,j] + uv[i,j+1] if j == 0 else uv[i,j] + uv[i,j-1]
                    else:#非边界层
                        L[(ind,(i-1)*(N+2) + j)] = a[1]
                        L[(ind,(i+1)*(N+2) + j)] = a[1]
                        L[(ind,i*(N+2) + j-1)] = a[1]
                        L[(ind,i*(N+2) + j+1)] = a[1]
                        L[(ind,ind)] = a[0]
                        rhs[ind] = b[i-1,j-1]
        if mode == 'v':
            ## v.T:(N+1,N+2);b.T:(N-1,N)
            return helmholtzStiffnessAssemble(N,Re,dt,uv.T,b.T,"u",False)
        return L,rhs
    else:
        if mode == 'u':
            for i in range(N+1):
                for j in range(N+2):
                    ind = i * (N+2) + j
                    if i == 0 or i == N:#
                   
                        rhs[ind] = uv[i,j]
                    elif j == 0 or j == N+1: #
                        ins = i * (N+2) + j + 1 if j == 0 else i * (N+2) + (j - 1)
                        rhs[ind] = uv[i,j] + uv[i,j+1] if j == 0 else uv[i,j] + uv[i,j-1]
                    else:#
                        rhs[ind] = b[i-1,j-1]
        if mode == 'v':
            ## v.T:(N+1,N+2);b.T:(N-1,N)
            return helmholtzStiffnessAssemble(N,Re,dt,uv.T,b.T,"u",True)
        return rhs                

def convertAndSolve(L, rhs, luDecomp = None):
    """

    """
    # if luDecomp != None:
    #     rows = max([k[0] for k in L.keys()]) + 1
    #     for key,val in rhs.items():
    #         RHS[key] = val
        
    if luDecomp:
        try:
            rows = max([k[0] for k in L.keys()]) + 1
        except:
            rows = max(L.keys()) + 1
        RHS = np.zeros((rows,1),dtype = np.float64)
        for key,val in rhs.items():
            RHS[key] = val
        return luDecomp.solve(RHS)
    rows = max([k[0] for k in L.keys()]) + 1
    cols = max([k[1] for k in L.keys()]) + 1
    sparse_matrix = lil_matrix((rows, cols))
    for key in L.keys():
        sparse_matrix[key] = L[key]
    sparse_matrix = csr_matrix(sparse_matrix)
    RHS = np.zeros((sparse_matrix.shape[0],1),dtype = np.float64)
    for key,val in rhs.items():
        RHS[key] = val
    # sparse_matrix = cupyx.scipy.sparse.csr_matrix(sparse_matrix)
    luDecomp = splu(sparse_matrix)
    solution = luDecomp.solve(RHS)
    return solution,luDecomp

def reshapeMat(solution,uv):
    """

    """
    row,col = uv.shape
    flag = 'u'
    if row > col:
        row,col = col,row
        flag = 'v'
    N = row - 1
    uvs = np.zeros_like(uv)
    if flag == 'u':
        for i in range(N+1):
            for j in range(N+2):
                uvs[i,j] = solution[i * (N+2) + j]
    else:
        for i in range(N+1):
            for j in range(N+2):
                uvs[j,i] = solution[i * (N+2) + j]
    # import pdb;pdb.set_trace()
    return uvs
    
def checkEquationValid(uvs, Re, N, dt, b):
    """
   证
    """
    # 
    h = 1.0 / N
    laplace_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])
    tmp = convolve2d(uvs, laplace_kernel, mode='valid')
    a = [1.0 + 2.0 * dt / (Re*h*h),-dt / 2.0 / (Re*h*h)]
    residual = uvs[1:-1,1:-1] + tmp * a[1] - b
    print("equation's validation residual = {:.16f}".format(np.linalg.norm(residual)))

def fft_neumann_poisson(b, h):
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

def fft_neumann_helmholtz(b,lamb,h):
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
        row = n / 2 * np.ones(n)
        row[0] = n
        col = m / 2 *np.ones(m)
        col[0] = m
        rc = -1*rc + lamb*(h**2)*np.outer(row, col)

        y1 = np.fft.fft(np.concatenate([h*h*b, np.zeros_like(b, dtype=np.complex128)], axis=0),axis = 0)
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
 
        return y2[:n, :].T.conj()


if __name__ == "__main__":
    """
   
    """
    N = 64
    Re = 1000.0
    dt = 1.0e-3
    uv = np.random.randn(N+1,N+2)
    b = np.random.randn(N-1,N)
    L,rhs = helmholtzStiffnessAssemble(N,Re,dt,uv,b,'u',False)
    rU = convertAndSolve(L, rhs)
    rU = reshapeMat(rU,uv)
    checkEquationValid(rU,Re,N,dt,b)
    b = np.random.randn(N-1,N)**2
    rhs = helmholtzStiffnessAssemble(N,Re,dt,uv,b,'u',True)
    rU = convertAndSolve(L, rhs)
    rU = reshapeMat(rU,uv)
    checkEquationValid(rU,Re,N,dt,b)
    uv = np.random.randn(N+2,N+1)
    b = b.T
    L,rhs = helmholtzStiffnessAssemble(N,Re,dt,uv,b,'v',False)
    rV = convertAndSolve(L, rhs)
    rV = reshapeMat(rV,uv)
    checkEquationValid(rV,Re,N,dt,b)
    b = np.random.randn(N,N-1)
    rhs = helmholtzStiffnessAssemble(N,Re,dt,uv,b,'v',True)
    rV = convertAndSolve(L, rhs)
    rV = reshapeMat(rV,uv)
    checkEquationValid(rV,Re,N,dt,b)
