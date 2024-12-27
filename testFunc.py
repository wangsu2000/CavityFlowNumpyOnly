import numpy as np


class TaylorVortex:
    def __init__(self, a, Re, N):
        """
        初始化TaylorVortex类的实例，接收泰勒涡旋相关函数中的参数a和Re。

        参数:
        a (float): 泰勒涡旋函数中的参数a
        Re (float): 雷诺数Re
        """
        self.a = a
        self.Re = Re
        self.psol,self.usol,self.vsol,self.fu,self.fv = self.get_functions()
        self.N = N
        

    def get_functions(self):
        """
        根据给定的时刻t，返回对应的泰勒涡旋相关函数的值。

        参数:
        t (float): 时刻

        返回:
        tuple: 包含压力、x方向速度、y方向速度、x方向体积力、y方向体积力的函数值的元组
        """
#        psol = lambda x, y, t: np.exp(-4 * t) * (np.cos(2 * self.a * x) / 4 + np.sin(2 * self.a * y) / 4)
#        usol = lambda x, y, t: -np.exp(-2 * t) * np.cos(self.a * y) * np.sin(self.a * x)
#        vsol = lambda x, y, t: np.exp(-2 * t) * np.cos(self.a * x) * np.sin(self.a * y)
#        fu = lambda x, y, t: 2 * np.exp(-2 * t) * np.cos(self.a * y) * np.sin(self.a * x) - \
#                          (self.a * np.exp(-4 * t) * np.sin(2 * self.a * x)) / 2 + self.a * np.exp(-4 * t) * np.cos(self.a * x) * np.cos(self.a * y) ** 2 * np.sin(self.a * x) + \
#                          self.a * np.exp(-4 * t) * np.cos(self.a * x) * np.sin(self.a * x) * np.sin(self.a * y) ** 2 - \
#                          (2 * self.a ** 2 * np.exp(-2 * t) * np.cos(self.a * y) * np.sin(self.a * x)) / self.Re
#        fv = lambda x, y, t: (self.a * np.exp(-4 * t) * np.cos(2 * self.a * y)) / 2 - \
#                          2 * np.exp(-2 * t) * np.cos(self.a * x) * np.sin(self.a * y) + \
#                          self.a * np.exp(-4 * t) * np.cos(self.a * x) ** 2 * np.cos(self.a * y) * np.sin(self.a * y) + \
#                          self.a * np.exp(-4 * t) * np.cos(self.a * y) * np.sin(self.a * x) ** 2 * np.sin(self.a * y) + \
#                          (2 * self.a ** 2 * np.exp(-2 * t) * np.cos(self.a * x) * np.sin(self.a * y)) / self.Re
#        return psol, usol, vsol, fu, fv
        """
        除TaylorVertex之外的样例:
        ex1
        注意:Re = 10.0
        """
#        psol = lambda x, y, t:  np.sin(t)*(np.sin(np.pi*y) - 2 / np.pi)
#        usol = lambda x, y, t:  np.sin(t)*(np.sin(np.pi*x)**2)*np.sin(2*np.pi*y)
#        vsol = lambda x, y, t:  -np.sin(t)*np.sin(2*np.pi*x)*np.sin(np.pi*y)**2
#        fu = lambda x, y, t: (2 * np.sin(np.pi * y) * (5 * np.cos(np.pi * y) * np.sin(np.pi * x) ** 2 * np.cos(t)
#                                    - np.pi ** 2 * np.cos(np.pi * x) ** 2 * np.cos(np.pi * y) * np.sin(t)
#                                    + 3 * np.pi ** 2 * np.cos(np.pi * y) * np.sin(np.pi * x) ** 2 * np.sin(t)
#                                    + 10 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * x) ** 3 * np.sin(np.pi * y) * np.sin(t) ** 2)) / 5
#        fv = lambda x, y, t: -(2 * np.sin(np.pi * x) * (5 * np.cos(np.pi * x) * np.sin(np.pi * y) ** 2 * np.cos(t)
#                                     - np.pi ** 2 * np.cos(np.pi * x) * np.cos(np.pi * y) ** 2 * np.sin(t)
#                                     + 3 * np.pi ** 2 * np.cos(np.pi * x) * np.sin(np.pi * y) ** 2 * np.sin(t)
#                                     - 10 * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * np.sin(np.pi * y) ** 3 * np.sin(t) ** 2)) / 5
#        return psol, usol, vsol, fu, fv 
        """
        除TaylorVertex之外的样例:
        ex2
        注意:Re = 10.0
        """
        psol = lambda x, y, t: t**2 * (x - 0.5)
        usol = lambda x, y, t: -128.0 * (t**2) * (x**2) * (x - 1.0)**2 * y * (y - 1.0) * (2.0 * y - 1.0)
        vsol = lambda x, y, t: 128*(t**2)*(y**2)*(y-1.0)**2 * x * (x - 1.0)*(2.0*x-1.0)
        # 将第一个表达式转换为lambda表达式形式，定义为fu函数
        fu = lambda x, y, t: (
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
        fv = lambda x, y, t: (
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
        return psol, usol, vsol, fu, fv 
    def query(self, t, mode):
        """
        根据给定的时刻t和分辨率N，生成表示泰勒涡旋相关物理量分布的图像（这里简单返回对应的数据网格，可根据实际需求进一步处理成可视化图像）。

        参数:
        t (float): 时刻
        N (int): 图像分辨率

        返回:
        tuple: 包含压力、x方向速度、y方向速度、x方向体积力、y方向体积力的二维数据网格的元组，每个二维数据网格代表对应物理量的分布情况
        """
        N = self.N
        if mode == "u":
            xx = np.arange(0,N+1).astype(np.float64) / N
            yy = np.arange(0,N+1).astype(np.float64) / N
            yy = yy - 0.5 / N
            yy = list(yy)
            yy[0] = 0.0
            yy.append(1.0)
            yy = np.asarray(yy)
            yy,xx = np.meshgrid(yy,xx)
            usol = self.usol(xx,yy,t)
            usol[:,0] = usol[:,0] * 2.0 - usol[:,1]
            usol[:,-1] = usol[:,-1] * 2.0 - usol[:,-2]
            return usol
        if mode == "v":
            xx = np.arange(0,N+1).astype(np.float64) / N
            yy = np.arange(0,N+1).astype(np.float64) / N
            xx = xx - 0.5 / N
            xx = list(xx)
            xx[0] = 0.0
            xx.append(1.0)
            xx = np.asarray(xx)
            yy,xx = np.meshgrid(yy,xx)
            vsol = self.vsol(xx,yy,t)
            vsol[0,:] = vsol[0,:] * 2.0 - vsol[1,:]
            vsol[-1,:] = vsol[-1,:] * 2.0 - vsol[-2,:]
            return vsol
        if mode == "p":
            xx = np.arange(1,N+1).astype(np.float64) / N
            yy = np.arange(1,N+1).astype(np.float64) / N
            xx = xx - 0.5 / N
            yy = yy - 0.5 / N
            yy,xx = np.meshgrid(yy,xx)
            psol = self.psol(xx,yy,t)
            return psol
        if mode == "fu":
            xx = np.arange(1,N).astype(np.float64) / N
            yy = np.arange(1,N+1).astype(np.float64) / N - 0.5 / N
            yy,xx = np.meshgrid(yy,xx)
            fusol = self.fu(xx,yy,t)
            return fusol
        if mode == "fv":
            xx = np.arange(1,N+1).astype(np.float64) / N - 0.5 / N
            yy = np.arange(1,N).astype(np.float64) / N 
            yy,xx = np.meshgrid(yy,xx)
            fvsol = self.fv(xx,yy,t)
            return fvsol
        # 返回cell中心值
        if mode == "um":
            xx = np.arange(1,N+1).astype(np.float64) / N
            yy = np.arange(1,N+1).astype(np.float64) / N
            xx = xx - 0.5 / N
            yy = yy - 0.5 / N
            yy,xx = np.meshgrid(yy,xx)
            usol = self.usol(xx,yy,t)
            return usol
        if mode == "vm":
            xx = np.arange(1,N+1).astype(np.float64) / N
            yy = np.arange(1,N+1).astype(np.float64) / N
            xx = xx - 0.5 / N
            yy = yy - 0.5 / N
            yy,xx = np.meshgrid(yy,xx)
            vsol = self.vsol(xx,yy,t)
            return vsol
        
        
            

if __name__ == "__main__":
    a = 2 * np.pi
    Re = 100.0
    N = 256
    taylor = TaylorVortex(a,Re,N)
    vsol = taylor.query(0.01,"fv")
    print(vsol.shape)