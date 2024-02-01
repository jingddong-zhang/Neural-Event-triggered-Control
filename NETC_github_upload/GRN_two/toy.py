import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append('D:\Python3.7.6\ETC\torchdiffeq_diy')
from odeint import odeint, odeint_event
from functions import *



class GRN(nn.Module):
    def __init__(self):
        super(GRN, self).__init__()
        self.a1 = 1.
        self.a2 = 1.
        self.b1 = 0.2
        self.b2 = 0.2
        self.k = 1.1
        self.n = 2
        self.s = 0.5
        self.scale = 10.

    def forward(self, t, state):
        dstate = torch.zeros_like(state)
        x,y = state[:,0],state[:,1]
        # dstate[:,0] = self.a1*x**self.n/(self.s**self.n+x**self.n)+self.b1*self.s**self.n/(self.s**self.n+y**self.n)-self.k*x
        # dstate[:,1] = self.a2 * y ** self.n / (self.s ** self.n + y ** self.n) + self.b2 * self.s ** self.n / (self.s ** self.n + x ** self.n) - self.k * y
        dstate[:,0] = self.scale*(self.a1*(x/self.scale)**self.n/(self.s**self.n+(x/self.scale)**self.n)
                                  +self.b1*self.s**self.n/(self.s**self.n+(y/self.scale)**self.n)-self.k*x/self.scale)
        dstate[:,1] = self.scale*(self.a2 * (y/self.scale) ** self.n / (self.s ** self.n + (y/self.scale) ** self.n)
                                  + self.b2 * self.s ** self.n / (self.s ** self.n + (x/self.scale) ** self.n) - self.k * y/self.scale)
        return dstate

class Augment(nn.Module):

    def __init__(self,state0):
        super(Augment, self).__init__()
        self.G = 9.81
        self.L = 0.5
        self.m = 0.15
        self.b = 0.1
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.control = ControlNet(2,10,2)
        self.odeint = odeint
        self.init_pos = nn.Parameter(state0[0:1])
        self.init_vel = nn.Parameter(state0[1:2])
        self.init_pos_err = nn.Parameter(torch.tensor([0.0]))
        self.init_vel_err = nn.Parameter(torch.tensor([0.0]))

    def get_initial_state(self):
        state = (self.init_pos, self.init_vel, self.init_pos_err,self.init_vel_err)
        return self.t0, state

    def forward(self, t, state):
        # u = self.control(state[:,0:2]+state[:,2:4])
        # u1,u2 = u[:,0],u[:,1]
        x,y,e_x,e_y = state
        input = torch.cat((x,y))+torch.cat((e_x,e_y))
        input = input.view(-1,2)
        u = self.control(input)
        u1, u2 = u[:, 0], u[:, 1]
        print(x.shape,input.shape)
        # dx = y+u1
        # dy = self.G*torch.sin(x)/self.L+(-self.b*y)/(self.m*self.L**2)+u2
        # de_x = -y-u1
        # de_y = -self.G*torch.sin(x)/self.L-(-self.b*y)/(self.m*self.L**2)-u2
        dx = y-10*(x+e_x)
        dy = self.G*torch.sin(x)/self.L+(-self.b*y)/(self.m*self.L**2)-10*(y+e_y)#+u2
        de_x = -y+10*(x+e_x)
        de_y = -self.G*torch.sin(x)/self.L-(-self.b*y)/(self.m*self.L**2)+10*(y+e_y)
        return dx,dy,de_x,de_y

    def event_fn(self, t, state):
        # positive before trigger time
        x,y,e_x,e_y = state
        return torch.sqrt(x**2+y**2) - torch.sqrt(e_x**2+e_y**2)


    def get_collision_times(self, ntrigger=1):

        event_times = []
        event_times = torch.zeros(ntrigger)

        t0, state = self.get_initial_state()

        for i in range(ntrigger):
            event_t, solution = odeint_event(
                self,
                state,
                t0,
                event_fn=self.event_fn,
                reverse_time=False,
                atol=1e-8,
                rtol=1e-8,
                odeint_interface=self.odeint,
            )
            # event_times.append(event_t)
            event_times[i] = event_t

            # state = self.state_update(tuple(s[-1] for s in solution))
            # t0 = event_t

        return event_times

'''
[0.62562059 0.62562059]
[0.76344689 0.3274622 ]
[0.0582738  0.85801853]
'''


# true_y0 = torch.Tensor(1,2).uniform_(-5,5)  # 初值
true_y0 = torch.tensor([[0.5,0.5]])
# print(true_y0[0][0:1].shape,torch.tensor([0.]).shape)
# system = Augment(true_y0[0])
# event_t = system.get_collision_times()
# print(event_t)


# true_y0 = torch.tensor([[0.,0.]])
t = torch.linspace(0., 20., 5000)  # 时间点
func = GRN()
with torch.no_grad():
    true_y = odeint(func, true_y0, t, method='dopri5')[:,0,:]

fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(true_y[:,0],true_y[:,1],true_y[:,2])
plt.plot(t,true_y[:,0])
plt.plot(t,true_y[:,1])
plt.show()