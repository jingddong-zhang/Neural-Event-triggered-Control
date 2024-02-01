import numpy as np
from numpy import *
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from functions import *
import os.path as osp

target = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).to(device)

# X = np.linspace(-6, 6, 100)
# Y = np.linspace(-6, 6, 100)
# x1, x2 = np.meshgrid(X,Y)
n = 100
x = torch.linspace(6.25-10, 6.25+10,n)
y = torch.linspace(6.25-10, 6.25+10, n)
X, Y = torch.meshgrid(x, y)
import scipy.io as scio
lqr_data = scio.loadmat('./data/lqr_data.mat')
S = torch.from_numpy(np.array(lqr_data['S1'])).to(device)
Q = torch.from_numpy(np.array(lqr_data['Q'])).to(device)
K = torch.from_numpy(np.array(lqr_data['K1'])).to(device)
model_lqr = LQR_event(S,Q,K,0.5)
inp = torch.stack([X, Y], dim=2).view(-1,2)
V_lqr = model_lqr.lya(inp).view(n,n)
# dV_lqr =  model_lqr.dlya(inp)#.view(n,n)
# f_u = model_lqr.untrigger_fn(1.0,inp)#.view(n,n)
# print(torch.sum(dV_lqr*f_u,dim=1).shape,V_lqr.shape)
# out_lqr = (torch.sum(dV_lqr*f_u,dim=1)+0.1*V_lqr[:,0]).view(n,n)
plt.contour(X,Y,V_lqr-5.0,0,linewidths=2, colors='m',linestyles='--')

case = 'quad'
D_in = 2  # input dimension
H1 = 20  # hidden dimension
D_out = 1  # output dimension

model_nlc = Augment(D_in, H1, D_out, (D_in,), [H1, 1], case).to(device)
model_nlc._control.load_state_dict(torch.load(osp.join('./data/', case + '_control.pkl')))
model_nlc._lya.load_state_dict(torch.load(osp.join('./data/', case + '_lya.pkl')))
V_nlc = model_nlc._lya(inp).view(n,n).detach()

plt.contour(X,Y,V_nlc-0.09,0,linewidths=2, colors='k',linestyles='--')
plt.show()