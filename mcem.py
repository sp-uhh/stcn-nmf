import torch
import numpy as np


class MCEM:
    def __init__(self, Y, model, device, niter_MCEM=100, niter_MH=40, 
        burnin=30, var_MH=0.01, NMF_rank=8):

        self.device = device
        self.model = model

        self.Y = Y
        self.F, self.T = self.Y.shape
        self.X =torch.from_numpy((np.abs(Y)**2).astype(np.float32)).to(device)

        # initialize NMF parameters
        np.random.seed(0)
        eps = torch.tensor(np.finfo(float).eps, device=self.device)
        self.K = NMF_rank
        self.W = torch.max(torch.rand(self.F,self.K, device=self.device), eps)
        self.H = torch.max(torch.rand(self.K,self.T, device=self.device), eps)
        self.g = torch.ones((1, self.T)).to(self.device)

        self.Z = model.encode(self.X)
        self.D = self.Z.shape[0]

        self.V_s = model.decode(self.Z) * self.g
        self.V_n = self.W @ self.H 

        self.niter_MH = niter_MH 
        self.niter_MCEM = niter_MCEM
        self.burnin = burnin 
        self.var_MH =  torch.tensor(var_MH)
        
        
    def metropolis_hastings(self, niter_MH, burnin):
        Z_sampled = torch.zeros((self.D, self.T, niter_MH - burnin), 
            device=self.device)

        for i in range(-burnin, niter_MH - burnin):
            Z_new = self.Z + \
                 self.var_MH*torch.randn(self.D, self.T, device=self.device)
 
            V_s_new = self.model.decode(Z_new)*self.g    
    
            acc_prob = (torch.sum(torch.log(self.V_n + self.V_s) 
                - torch.log(self.V_n + V_s_new) + (1/(self.V_n + self.V_s) 
                - 1/(self.V_n + V_s_new)) * self.X, axis=0)
                + .5*torch.sum(self.Z.pow(2) - Z_new.pow(2), axis=0))
            
            idx = torch.log(torch.rand(self.T, device=self.device)) < acc_prob
                        
            self.Z[:,idx] = Z_new[:,idx]
            self.V_s = self.model.decode(self.Z)*self.g

            if i >= 0: Z_sampled[:,:,i] = self.Z
        
        return Z_sampled    
    

    def run(self, tol=1e-4):
        cost_after_M_step = np.zeros(self.niter_MCEM)

        for n in range(self.niter_MCEM):
            Z_sampled = self.metropolis_hastings(self.niter_MH, self.burnin)         
            V_s_sampled = torch.zeros((self.F, self.T, 
                self.niter_MH - self.burnin), device=self.device)

            for i in range(self.niter_MH - self.burnin):
                V_s_sampled[:,:,i] = self.model.decode(Z_sampled[:,:,i])
            V_x = self.V_n[:,:,None] + V_s_sampled*self.g[:,:,None]
                               
            # Udpade W
            self.W = self.W*(((self.X*torch.sum(V_x.pow(-2), axis=-1)) 
                @ self.H.T)/(torch.sum(V_x.pow(-1), axis=-1) 
                @ self.H.T)).pow(0.5)
            self.V_n = self.W @ self.H
            V_x = self.V_n[:,:,None] + V_s_sampled*self.g[:,:,None]

            # Update H
            self.H = self.H*((self.W.T @ (self.X*torch.sum(V_x**-2, axis=-1)))
                    / (self.W.T @ torch.sum(V_x.pow(-1), axis=-1))).pow(0.5)
            self.V_n = self.W @ self.H
            V_x = self.V_n[:,:,None] + V_s_sampled*self.g[:,:,None]

            # Update g
            self.g = self.g*((torch.sum(self.X*torch.sum(V_s_sampled
                *(V_x.pow(-2)),axis=-1), axis=0))/(torch.sum(torch.sum(
                V_s_sampled*(V_x.pow(-1)), axis=-1), axis=0))).pow(0.5)
            V_x = self.V_n[:,:,None] + V_s_sampled*self.g[:,:,None]

            cost_after_M_step[n] = torch.mean(torch.log(V_x) 
                + self.X[:,:,None]/V_x )

            if n>0 and cost_after_M_step[n-1] - cost_after_M_step[n] < tol:
                break


    def separate(self, niter_MH, burnin):
        Z_sampled = self.metropolis_hastings(self.niter_MH, self.burnin)         
        V_s_sampled = torch.zeros((self.F, self.T, 
            self.niter_MH - self.burnin), device=self.device)

        for i in range(self.niter_MH - self.burnin):
            V_s_sampled[:,:,i] = self.model.decode(Z_sampled[:,:,i])

        V_s_sampled = V_s_sampled*self.g[:,:,None]

        self.S_hat = self.Y * torch.mean(V_s_sampled / 
            (V_s_sampled + self.V_n[:,:,None]), axis=-1).cpu().numpy() 