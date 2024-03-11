# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 16:01:41 2021

@author: leixi
"""

#%% Libraries and Dependencies
import sys
sys.path.insert(0, '../Utilities/')
import torch
from collections import OrderedDict
# from pyDOE import lhs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time



np.random.seed(1234)
# CUDA support

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.cuda.empty_cache()
torch.cuda.empty_cache()
#%%  The deep neural network
class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        n_n = 500
        self.input = torch.nn.Sequential(torch.nn.Linear(2, n_n), torch.nn.Tanh())
        self.hidden1 = torch.nn.Sequential(torch.nn.Linear(n_n, n_n), torch.nn.Tanh())
        self.hidden2 = torch.nn.Sequential(torch.nn.Linear(n_n, n_n), torch.nn.Tanh())
        self.hidden3 = torch.nn.Sequential(torch.nn.Linear(n_n, n_n), torch.nn.Tanh())
        self.hidden4 = torch.nn.Sequential(torch.nn.Linear(n_n, n_n), torch.nn.Tanh())
        self.output = torch.nn.Sequential(torch.nn.Linear(n_n, 1))
    
    def forward(self, x):
        out1 = self.input(x)
        # out1 = torch.nn.Tanh(out1)
        out = self.hidden1(out1)
        # out = torch.nn.Tanh(out)
        out2 = self.hidden2(out+out1)
        # out2 = torch.nn.Tanh(out2)
        out = self.hidden3(out2)
        # out = torch.nn.Tanh(out)
        out = self.hidden4(out+out2)
        # out = torch.nn.Tanh(out)
        out = self.output(out)
        return out
    
    
#%%  The physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, x_gvn, y_gvn, x_up, y_up, x_low, y_low, x_lft_1, y_lft_1, x_lft_3, y_lft_3, x_rgt, y_rgt, para):
        # data
        self.q_gvn = torch.tensor(x_gvn[:, 0], requires_grad=True).float().to(device)
        self.p_gvn = torch.tensor(x_gvn[:, 1], requires_grad=True).float().to(device)
        self.h_gvn = torch.tensor(y_gvn, requires_grad=True).float().to(device)
        
        self.q_up = torch.tensor(x_up[:, 0], requires_grad=True).float().to(device)
        self.p_up = torch.tensor(x_up[:, 1], requires_grad=True).float().to(device)
        self.h_up = torch.tensor(y_up, requires_grad=True).float().to(device)
        
        self.q_low = torch.tensor(x_low[:, 0], requires_grad=True).float().to(device)
        self.p_low = torch.tensor(x_low[:, 1], requires_grad=True).float().to(device)
        self.h_low = torch.tensor(y_low, requires_grad=True).float().to(device)
        
        self.q_lft_1 = torch.tensor(x_lft_1[:, 0], requires_grad=True).float().to(device)
        self.p_lft_1 = torch.tensor(x_lft_1[:, 1], requires_grad=True).float().to(device)
        self.h_lft_1 = torch.tensor(y_lft_1, requires_grad=True).float().to(device)
        
        self.q_lft_3 = torch.tensor(x_lft_3[:, 0], requires_grad=True).float().to(device)
        self.p_lft_3 = torch.tensor(x_lft_3[:, 1], requires_grad=True).float().to(device)
        self.h_lft_3 = torch.tensor(y_lft_3, requires_grad=True).float().to(device)
        
        self.q_rgt = torch.tensor(x_rgt[:, 0], requires_grad=True).float().to(device)
        self.p_rgt = torch.tensor(x_rgt[:, 1], requires_grad=True).float().to(device)
        self.h_rgt = torch.tensor(y_rgt, requires_grad=True).float().to(device)
        
        self.gm = torch.tensor(para[0], requires_grad=True).float().to(device)
        self.grav = torch.tensor(para[1], requires_grad=True).float().to(device)
        self.p0 = torch.tensor(para[2], requires_grad=True).float().to(device)
        self.Q = torch.tensor(para[3], requires_grad=True).float().to(device)
        
        self.loss_MSE = torch.nn.MSELoss()
        # deep neural networks
        self.dnn = DNN().to(device)
        # optimizers: using the same settings
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(), lr=0.001, max_iter=50000, max_eval=50000, 
            history_size=50, tolerance_grad=1e-10, 
            tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe" )
        
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        self.iter = 0
        
    def net_u(self, q, p):
        h = self.dnn(torch.cat([torch.unsqueeze(q, 1), torch.unsqueeze(p, 1)], dim=1))
        return h
    
    def net_f(self, q, p):
        """ The pytorch autograd version of calculating residual """
        h = self.net_u(q, p)
        h_q = torch.autograd.grad(
            h, q, grad_outputs=torch.ones_like(h),
            retain_graph=True, create_graph=True)[0]
        h_p = torch.autograd.grad(
            h, p, grad_outputs=torch.ones_like(h),
            retain_graph=True, create_graph=True)[0]
        h_qq = torch.autograd.grad(
            h_q, q, grad_outputs=torch.ones_like(h_q),
            retain_graph=True, create_graph=True)[0]
        h_pp = torch.autograd.grad(
            h_p, p, grad_outputs=torch.ones_like(h_p),
            retain_graph=True, create_graph=True)[0]
        h_qp = torch.autograd.grad(
            h_q, p, grad_outputs=torch.ones_like(h_q),
            retain_graph=True, create_graph=True)[0]
        return h, h_q, h_p, h_qq, h_pp, h_qp
    
    def MSE_gvn(self, h, h_q, h_p, h_qq, h_pp, h_qp):
        # gvn_pred = (1 + h_p**2)*h_pp - 2*h_p*h_q*h_qp + (h_p**2)*h_qq -self.gm*(-self.p_gvn)*(h_p**3)
        gvn_pred = (1 + h_p**2)*h_pp - 2*h_p*h_q*h_qp + (h_p**2)*h_qq -self.gm*(h_p**3)
        loss_gvn = torch.mean(gvn_pred ** 2)
        return loss_gvn
    
    def MSE_up(self, h, h_q, h_p, h_qq, h_pp, h_qp):
        up_pred = 1 + h_q**2 + (2*self.grav*h - self.Q)*(h_p**2)
        loss_up = torch.mean(up_pred ** 2)
        return loss_up
    
    def MSE_low(self, h):
        low_pred = h
        loss_low = torch.mean(low_pred ** 2)
        return loss_low
    
    def MSE_lft(self, h_1, h_3):
        lft_pred = h_1 - h_3
        loss_lft = torch.mean(lft_pred ** 2)
        return loss_lft
    
    def MSE_rgt(self, h_q):
        rgt_pred = h_q
        loss_rgt = torch.mean(rgt_pred ** 2)
        return loss_rgt
    
    def loss_func(self):
        self.optimizer_Adam.zero_grad()
        self.optimizer_LBFGS.zero_grad()

        h_gvn, h_q_gvn, h_p_gvn, h_qq_gvn, h_pp_gvn, h_qp_gvn = self.net_f(self.q_gvn, self.p_gvn)
        h_up, h_q_up, h_p_up, h_qq_up, h_pp_up, h_qp_up = self.net_f(self.q_up, self.p_up)
        h_low, h_q_low, h_p_low, h_qq_low, h_pp_low, h_qp_low = self.net_f(self.q_low, self.p_low)
        h_lft_1, h_q_lft_1, h_p_lft_1, h_qq_lft_1, h_pp_lft_1, h_qp_lft_1 = self.net_f(self.q_lft_1, self.p_lft_1)
        h_lft_3, h_q_lft_3, h_p_lft_3, h_qq_lft_3, h_pp_lft_3, h_qp_lft_3 = self.net_f(self.q_lft_3, self.p_lft_3)
        h_rgt, h_q_rgt, h_p_rgt, h_qq_rgt, h_pp_rgt, h_qp_rgt = self.net_f(self.q_rgt, self.p_rgt)

        loss_h = self.loss_MSE(torch.unsqueeze(self.h_gvn, 1), h_gvn)
        loss_gvn = self.MSE_gvn(h_gvn, h_q_gvn, h_p_gvn, h_qq_gvn, h_pp_gvn, h_qp_gvn)
        loss_up = self.MSE_up(h_up, h_q_up, h_p_up, h_qq_up, h_pp_up, h_qp_up)
        loss_low = self.MSE_low(h_low)
        loss_lft = self.MSE_lft(h_lft_1, h_lft_3)
        loss_rgt = self.MSE_rgt(h_q_rgt)
        
        loss = 10*loss_h + 1*loss_gvn + 10*loss_up + 1*loss_low + 10*loss_lft + 1*loss_rgt

        loss.backward()
        self.iter += 1
        if self.iter % 50 == 0:
            print('----------LOSS----------')
            print('Iter %d' % (self.iter))
            print('Total loss: %.5f' % (loss.item()))
            print('Sub loss')
            print('loss_h: %.5f,  loss_gvn: %.5f' % (loss_h.item(), loss_gvn.item()))
            print('loss_up: %.5f, loss_low: %.5f, loss_lft: %.5f, loss_rgt: %.5f' % (loss_up.item(), loss_low.item(), loss_lft.item(), loss_rgt.item()))
        return loss
    
    def train(self):
        for epoch in range(total_epoch):
            self.dnn.train()
            # Backward and optimize
            self.optimizer_Adam.step(self.loss_func)
        
        self.dnn.train()
        self.optimizer_LBFGS.step(self.loss_func)
        
    def predict(self, X):
        x_q = torch.tensor(X[:, 0], requires_grad=True).float().to(device)
        x_p = torch.tensor(X[:, 1], requires_grad=True).float().to(device)
        self.dnn.eval()
        h = self.net_u(x_q, x_p)
        h = h.detach().cpu().numpy()
        return h
#%%  Configurations
start_time = time.time()
N_train_gvn = 20000
N_train_bry = 300

total_epoch = 2000

mat = scipy.io.loadmat('wave.mat')
gm = mat['gm'][0][0]
grav = mat['grav'][0][0]
p0 = mat['p0'][0][0]
Q = mat['Q'][0][0]

h = np.flipud(np.rot90(mat['h']))
p = mat['p'][0]
q = mat['q'][0]

# plt.plot(h[:,-1])
# %%
X_q, X_p = np.meshgrid(q, p)

X_gvn = np.stack((X_q.flatten(), X_p.flatten()), axis = 1)
Y_gvn = h.flatten()

X_up = np.stack((X_q[-1, :], X_p[-1, :]), axis = 1)
Y_up = h[-1, 1:]
plt.plot(Y_up)
X_low = np.stack((X_q[0, :], X_p[0, :]), axis = 1)
Y_low = h[0, 1:]

X_lft_1 = np.stack((X_q[:, 0], X_p[:, 0]), axis = 1)
Y_lft_1 = h[:, 0]

X_lft_3 = np.stack((X_q[:, 2], X_p[:, 2]), axis = 1)
Y_lft_3 = h[:, 2]

X_rgt = np.stack((X_q[:, -1], X_p[:, -1]), axis = 1)
Y_rgt = h[:, -1]
# %%
idx_gvn = np.random.choice(X_gvn.shape[0], N_train_gvn, replace=False)
idx_up = np.random.choice(X_q.shape[1]-1, X_q.shape[1]-1, replace=False)
idx_low = np.random.choice(X_q.shape[1]-1, N_train_bry, replace=False)
idx_lft = np.random.choice(X_q.shape[0]-1, X_q.shape[0]-1, replace=False)
idx_rgt = np.random.choice(X_q.shape[0]-1, N_train_bry, replace=False)

x_gvn = X_gvn[idx_gvn, :]; y_gvn = Y_gvn[idx_gvn]
x_up = X_up[idx_up, :]; y_up = Y_up[idx_up]
x_low = X_low[idx_low, :]; y_low = Y_low[idx_low]
# x_low = X_low[idx_up, :]; y_low = Y_low[idx_up]
x_lft_1 = X_lft_1[idx_lft, :]; y_lft_1 = Y_lft_1[idx_lft]
x_lft_3 = X_lft_3[idx_lft, :]; y_lft_3 = Y_lft_3[idx_lft]
x_rgt = X_rgt[idx_rgt, :]; y_rgt = Y_rgt[idx_rgt]
# x_rgt = X_rgt[idx_lft, :]; y_rgt = Y_rgt[idx_lft]
para = [gm, grav, p0, Q]
print('--------------------')
print('Data Loading DONE!')
print('--------------------')
#%%  Training
model = PhysicsInformedNN(x_gvn, y_gvn, x_up, y_up, x_low, y_low, x_lft_1, y_lft_1, x_lft_3, y_lft_3, x_rgt, y_rgt, para)
model.train()
print('+++++++@(^**^)@+++++++')
print('Training DONE!')
print('+++++++@(^**^)@+++++++')
#%%  Predicting
h_pred = model.predict(X_gvn)
error_h = np.linalg.norm(Y_gvn-np.squeeze(h_pred),2)/np.linalg.norm(Y_gvn,2)
print('Error h: %e' % (error_h))

fig = plt.figure(figsize=(6*1.,3.6*1.))
ax = fig.gca()
im=ax.contourf(q,p,Y_gvn.reshape(681,442),20,cmap="RdBu_r")
fig.colorbar(im)
plt.title('Exact')

fig = plt.figure(figsize=(6*1.,3.6*1.))
ax = fig.gca()
im=ax.contourf(q,p,h_pred.reshape(681,442),20,cmap="RdBu_r")
fig.colorbar(im)
plt.title('Prediction')

end_time = time.time()
print('Total time cost: %2f mins' % ((end_time-start_time)/60))