# neumman_3d.py
import torch
import numpy as np

class KroneckerBMM:
    def __init__(self, mm, nn, ll):
        # nn, mm, ll = 16, 16, 16
        self.nn = nn
        self.mm = mm
        self.ll = ll
        self.Lm = torch.diag(torch.ones(mm)) * (-2) + torch.diag(torch.ones(mm - 1), -1) + torch.diag(torch.ones(mm - 1), 1)
        self.Ln = torch.diag(torch.ones(nn)) * (-2) + torch.diag(torch.ones(nn - 1), -1) + torch.diag(torch.ones(nn - 1), 1)
        self.Ll = torch.diag(torch.ones(ll)) * (-2) + torch.diag(torch.ones(ll - 1), -1) + torch.diag(torch.ones(ll - 1), 1)
        self.Lm[0, 0], self.Lm[-1, -1] = -1, -1
        self.Ln[0, 0], self.Ln[-1, -1] = -1, -1
        self.Ll[0, 0], self.Ll[-1, -1] = -1, -1

        Vm = torch.zeros(mm, mm)
        for i in range(mm):
            for j in range(1, mm + 1):
                Vm[i, j - 1] = np.cos(np.pi * (2 * j - 1) * i / (2 * mm))

        Vn = torch.zeros(nn, nn)
        for i in range(nn):
            for j in range(1, nn + 1):
                Vn[i, j - 1] = np.cos(np.pi * (2 * j - 1) * i / (2 * nn))

        Vl = torch.zeros(ll, ll)
        for i in range(ll):
            for j in range(1, ll + 1):
                Vl[i, j - 1] = np.cos(np.pi * (2 * j - 1) * i / (2 * ll))

        vm = torch.ones(mm) * mm / 2
        vm[0] = mm
        vn = torch.ones(nn) * nn / 2
        vn[0] = nn
        vl = torch.ones(ll) * ll / 2
        vl[0] = ll

        Vm = Vm.to(torch.float64).cuda()
        Vn = Vn.to(torch.float64).cuda()
        Vl = Vl.to(torch.float64).cuda()

        en = (2 * np.cos(np.arange(nn) * np.pi / nn) - 2) * nn / 2
        em = (2 * np.cos(np.arange(mm) * np.pi / mm) - 2) * mm / 2
        el = (2 * np.cos(np.arange(ll) * np.pi / ll) - 2) * ll / 2

        pseudo = torch.from_numpy(np.kron(np.kron(em, vn), vl) + np.kron(np.kron(vm, en), vl) + np.kron(
            np.kron(vm, vn), el)).to(torch.float64)
        pseudo[0] = float('inf')
        self.pseudo = pseudo.cuda().to(torch.float64)
        self.Vm = Vm
        self.Vn = Vn
        self.Vl = Vl

    def forward_(self, X, c,b,a):
        Y = X.view(self.mm, self.nn, self.ll)
        # print(Y.shape,c.shape)
        Y = torch.einsum('ijk,kl->ijl', Y, c.T)
        Y = torch.einsum('ijk,jl->ilk', Y, b.T)
        Y = torch.einsum('ijk,il->ljk', Y, a.T)
        return Y.flatten()

    def forward(self,X, h = 1.0):
        Y = self.forward_(X.flatten(),self.Vl,self.Vn,self.Vm).to(torch.float64).cuda()
        Y = Y / self.pseudo
        Y = self.forward_(Y,self.Vl.T,self.Vn.T,self.Vm.T).to(torch.float64).cuda()
        return Y * (h * h)
        
    def test_parameters(self):
        L = np.kron(np.kron(self.Lm.numpy(), np.eye(nn)), np.eye(ll)) + \
        np.kron(np.kron(np.eye(mm), self.Ln.numpy()), np.eye(ll)) + \
        np.kron(np.kron(np.eye(mm), np.eye(nn)), self.Ll.numpy())
        M = np.kron(np.kron(self.Vm.cpu().numpy(), self.Vn.cpu().numpy()), self.Vl.cpu().numpy())
        return L,M
        
if __name__ == "__main__":
    # Example usage
   
        
    mm,nn,ll = 256,256,256
    KB = KroneckerBMM(mm,nn,ll)
    X = torch.rand(mm, nn, ll).to(torch.float64).cuda()
    with torch.no_grad():
        res = KB.forward(X)
    print(res.view(mm,nn,ll).shape)
    # L,_ = KB.test_parameters()
    # residual = np.dot(L, res.cpu().numpy()) - X.view(-1).cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.plot(residual)
    # plt.show()
