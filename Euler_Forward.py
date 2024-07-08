import numpy as np
import matplotlib.pyplot as plt
import os


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

if __name__ == "__main__":
  Re = 1000  # Reynolds number
  nt = 1600  # max time steps
  Lx, Ly = 1, 1  # domain size
  Nx, Ny = 80, 80  # Number of grids
  dt = 0.01  # time step
  dx = Lx / Nx
  dy = Ly / Ny
  
  # 输入你需要存储图片文件夹的路径
  root_path = "E:\ImageMagick-6.2.7-Q16\segements"
  
  xce = ((np.arange(1, Nx + 1) - 0.5) * dx).reshape((Nx, 1))
  yce = ((np.arange(1, Ny + 1) - 0.5) * dy).reshape((1, Ny))
  
  u = np.zeros((Nx + 1, Ny + 2))
  v = np.zeros((Nx + 2, Ny + 1))
  p = np.zeros((Nx, Ny))
  Xce, Yce = np.meshgrid(xce.flatten(), yce.flatten())
  image_name = 0
  for ii in range(1, nt + 1):
      bctop = 1
  
      # Boundary conditions
      u[:, 0] = -u[:, 1]
      v[:, 0] = 0
      u[:, -1] = 2 * bctop - u[:, -2]
      v[:, -1] = 0
      u[0, :] = 0
      v[0, :] = -v[1, :]
      u[-1, :] = 0
      v[-1, :] = -v[-2, :]
  
      Lux = (u[0:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / dx**2
      Luy = (u[1:-1, 0:-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / dy**2
      Lvx = (v[0:-2, 1:-1] - 2 * v[1:-1, 1:-1] + v[2:, 1:-1]) / dx**2
      Lvy = (v[1:-1, 0:-2] - 2 * v[1:-1, 1:-1] + v[1:-1, 2:]) / dy**2
  
      uce = 0.5 * (u[0:-1, 1:-1] + u[1:, 1:-1])
      uco = 0.5 * (u[:, 0:-1] + u[:, 1:])
      vco = 0.5 * (v[0:-1, :] + v[1:, :])
      vce = 0.5 * (v[1:-1, 0:-1] + v[1:-1, 1:])
  
      uuce = uce * uce
      uvco = uco * vco
      vvce = vce * vce
  
      Nu = (uuce[1:, :] - uuce[0:-1, :]) / dx + (uvco[1:-1, 1:] - uvco[1:-1, 0:-1]) / dy
      Nv = (vvce[:, 1:] - vvce[:, 0:-1]) / dy + (uvco[1:, 1:-1] - uvco[0:-1, 1:-1]) / dx
  
      u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (-Nu + (Lux + Luy) / Re)
      v[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (-Nv + (Lvx + Lvy) / Re)
  
      b = (u[1:, 1:-1] - u[0:-1, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, 0:-1]) / dy
      p = fft_neumann_poisson(b, dx)
  
      u[1:-1, 1:-1] = u[1:-1, 1:-1] - (p[1:, :] - p[0:-1, :]) / dx
      v[1:-1, 1:-1] = v[1:-1, 1:-1] - (p[:, 1:] - p[:, 0:-1]) / dy
  
      uce = 0.5 * (u[0:-1, 1:-1] + u[1:, 1:-1])
      vce = 0.5 * (v[1:-1, 0:-1] + v[1:-1, 1:])
  
      b = (u[1:, 1:-1] - u[0:-1, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, 0:-1]) / dy
      if (ii) % 100 == 1:
          print(f'当前迭代轮数 = {ii}, 散度残差 = {np.linalg.norm(b[1:-1, 1:-1])}')
          plt.streamplot(Xce, Yce,uce.T,vce.T, density=1, linewidth=1, color='blue')
          plt.xlim([0, Lx])
          plt.ylim([0, Ly])
          plt.title('Re = {:.1f},time = {:4f}'.format(Re,ii*dt))
          save_path = os.path.join(root_path,str(image_name)+".jpg")
  #         plt.show()
          plt.savefig(save_path)
          plt.clf()
          image_name = image_name + 1
