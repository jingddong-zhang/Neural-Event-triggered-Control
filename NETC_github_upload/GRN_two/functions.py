import numpy as np
from numpy.core.defchararray import mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm
from MonotonicNN import MonotonicNN
import math
import timeit
# from torchdiffeq import odeint, odeint_adjoint
# from torchdiffeq import odeint_event
from odeint import odeint, odeint_event
from adjoint import odeint_adjoint
torch.set_default_dtype(torch.float64)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
device = 'cpu'
print(device)

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class ICNN(nn.Module):
    def __init__(self, input_shape, layer_sizes, smooth_relu_thresh=0.1,eps=1e-3):
        super(ICNN, self).__init__()
        self._input_shape = input_shape
        self._layer_sizes = layer_sizes
        self._eps = eps
        self._d = smooth_relu_thresh
        # self._activation_fn = activation_fn
        ws = []
        bs = []
        us = []
        prev_layer = input_shape
        w = torch.empty(layer_sizes[0], *input_shape)
        nn.init.xavier_normal_(w)
        ws.append(nn.Parameter(w))
        b = torch.empty([layer_sizes[0], 1])
        nn.init.xavier_normal_(b)
        bs.append(nn.Parameter(b))
        for i in range(len(layer_sizes))[1:]:
            w = torch.empty(layer_sizes[i], *input_shape)
            nn.init.xavier_normal_(w)
            ws.append(nn.Parameter(w))
            b = torch.empty([layer_sizes[i], 1])
            nn.init.xavier_normal_(b)
            bs.append(nn.Parameter(b))
            u = torch.empty([layer_sizes[i], layer_sizes[i - 1]])
            nn.init.xavier_normal_(u)
            us.append(nn.Parameter(u))
        self._ws = nn.ParameterList(ws)
        self._bs = nn.ParameterList(bs)
        self._us = nn.ParameterList(us)
        self.target = torch.tensor([[0.62562059,0.62562059]])*10.

    def smooth_relu(self, x):
        relu = x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        # sq = (2 * self._d * relu.pow(3) - relu.pow(4)) / (2 * self._d ** 3)
        sq = relu.pow(2)/(2*self._d)
        lin = x - self._d / 2
        return torch.where(relu < self._d, sq, lin)

    def dsmooth_relu(self, x):
        relu = x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        # sq = (2 * self._d * relu.pow(3) - relu.pow(4)) / (2 * self._d ** 3)
        sq = relu/self._d
        lin = 1.
        return torch.where(relu < self._d, sq, lin)

    def icnn_fn(self, x):
        # x: [batch, data]
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        else:
            data_dims = list(range(1, len(self._input_shape) + 1))
            x = x.permute(*data_dims, 0)
        z = self.smooth_relu(torch.addmm(self._bs[0], self._ws[0], x))
        # print('--------------------',self._ws[0].shape)
        for i in range(len(self._us)):
            u = F.softplus(self._us[i])
            w = self._ws[i + 1]
            b = self._bs[i + 1]
            z = self.smooth_relu(torch.addmm(b, w, x) + torch.mm(u, z))
        return z

    def inter_dicnn_fn(self, x):
        # x: [batch, data]
        x = x.clone().detach()
        N,dim = x.shape[0],x.shape[1]
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        else:
            data_dims = list(range(1, len(self._input_shape) + 1))
            x = x.permute(*data_dims, 0)
        z = torch.addmm(self._bs[0], self._ws[0], x)
        dz = self.dsmooth_relu(z).unsqueeze(2).repeat(1,1,dim)*self._ws[0].unsqueeze(1).repeat(1,N,1)
        for i in range(len(self._us)):
            u = F.softplus(self._us[i])
            w = self._ws[i + 1]
            b = self._bs[i + 1]
            # print(u.shape, w.shape, b.shape, z.shape)
            z = torch.addmm(b, w, x) + torch.mm(u, self.smooth_relu(z))
            for k in range(dim):
                dz[:,:,k] = torch.mm(u,dz[:,:,k])
            dz = self.dsmooth_relu(z).unsqueeze(2).repeat(1,1,dim)*w.unsqueeze(1).repeat(1,N,1) \
                + self.dsmooth_relu(z).unsqueeze(2).repeat(1,1,dim)*dz
        return dz

    def dicnn_fn(self,x):
        dim = x.shape[1]
        target = self.target.repeat(len(x), 1)
        z = self.icnn_fn(x)
        z0 = self.icnn_fn(target)
        dregular = 2*self._eps*(x-target)
        dz = self.dsmooth_relu(z-z0).unsqueeze(2).repeat(1,1,dim)*self.inter_dicnn_fn(x)+dregular
        return dz[0]

    def forward(self,x):
        target = self.target.repeat(len(x),1)
        z = self.icnn_fn(x)
        z0 = self.icnn_fn(target)
        regular = self._eps * (x-target).pow(2).sum(dim=1).view(-1,1)
        return (self.smooth_relu(z-z0).T+regular)*1.

def lya(ws,bs,us,smooth,x,input_shape):
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    else:
        data_dims = list(range(1, len(input_shape) + 1))
        x = x.permute(*data_dims, 0)
    z = smooth(torch.addmm(bs[0],ws[0], x))
    for i in range(len(us)):
        u = F.softplus(us[i])
        w = ws[i + 1]
        b = bs[i + 1]
        z = smooth(torch.addmm(b, w, x) + torch.mm(u, z))
    return z

def dlya(ws,bs,us,smooth_relu,dsmooth_relu,x,input_shape):
    N, dim = x.shape[0], x.shape[1]
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    else:
        data_dims = list(range(1, len(input_shape) + 1))
        x = x.permute(*data_dims, 0)
    z = torch.addmm(bs[0], ws[0], x)
    dz = dsmooth_relu(z).unsqueeze(2).repeat(1, 1, dim) * ws[0].unsqueeze(1).repeat(1, N, 1)
    for i in range(len(us)):
        u = F.softplus(us[i])
        w = ws[i + 1]
        b = bs[i + 1]
        # print(u.shape, w.shape, b.shape, z.shape)
        z = torch.addmm(b, w, x) + torch.mm(u, smooth_relu(z))
        for k in range(dim):
            dz[:, :, k] = torch.mm(u, dz[:, :, k])
        dz = dsmooth_relu(z).unsqueeze(2).repeat(1, 1, dim) * w.unsqueeze(1).repeat(1, N, 1) \
             + dsmooth_relu(z).unsqueeze(2).repeat(1, 1, dim) * dz
    return dz

class ControlNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(ControlNet, self).__init__()
        self.layer1 = SpectralNorm(torch.nn.Linear(n_input, n_hidden))
        self.layer2 = SpectralNorm(torch.nn.Linear(n_hidden, n_hidden))
        self.layer3 = SpectralNorm(torch.nn.Linear(n_hidden, n_output))
        self.target = torch.tensor([[0.62562059,0.62562059]])*10.

    def function(self, x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out

    def forward(self, x):
        target = self.target.repeat(len(x), 1)
        u = self.function(x)
        u0 = self.function(target)
        return u-u0

class ControlNormalNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(ControlNormalNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
        self.target = torch.tensor([[0.62562059, 0.62562059]]) * 10.
        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)

    def forward(self,x):
        # x_0 = torch.zeros_like(x).to(device)
        # return self.net(x)-self.net(x_0)
        target = self.target.repeat(len(x), 1)
        return self.net(x)-self.net(target)

class ControlNLCNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(ControlNLCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_output),
        )
        self.target = torch.tensor([[0.62562059, 0.62562059]]) * 10.
        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)

    def forward(self,x):
        target = self.target.repeat(len(x), 1)
        return self.net(x)-self.net(target)

class QuadVNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden,n_output,eps):
        super(QuadVNet, self).__init__()
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_output)
        self._eps = eps
        self.target = torch.tensor([[0.62562059,0.62562059]])*10.

    def forward(self, x):
        target = self.target.repeat(len(x), 1)
        x = x-target
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        # return (torch.sum((out*x)**2,dim=1)+self._eps * x.pow(2).sum(dim=1)).view(-1,1)
        return torch.sum((out)**2,dim=1).view(-1,1)+self._eps * x.pow(2).sum(dim=1).view(-1,1)

    def dsigmoid(self,x):
        sigmoid = torch.nn.Tanh()
        return 1.0-sigmoid(x)**2

    def derivative(self,x):
        target = self.target.repeat(len(x), 1)
        x = x-target
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        N,dim = x.shape[0],x.shape[1]
        W1 = self.layer1.weight.unsqueeze(1).repeat(1,N,1)
        W2 = self.layer2.weight
        dh_1 = self.dsigmoid(self.layer1(x)).T.unsqueeze(2).repeat(1,1,dim)*W1
        grad = torch.zeros([W2.shape[0],N,dim]).to(device)
        # print(grad.shape,dh1.shape,W2.shape,W1.shape)
        for i in range(dim):
            grad[:,:,i] = torch.mm(W2,dh_1[:,:,i])
        out = out.T.unsqueeze(2).repeat(1, 1, dim)
        grad = torch.sum(out*grad,dim=0)
        return grad*2+self._eps*x*2


class NLCVNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output,eps):
        super(NLCVNet, self).__init__()
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_output)
        self._eps = eps
        self.target = torch.tensor([[0.62562059, 0.62562059]]) * 10.

    def forward(self, x):
        target = self.target.repeat(len(x), 1)
        x = x - target
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out#+self._eps * x.pow(2).sum(dim=1).view(-1,1)

    def dsigmoid(self,x):
        sigmoid = torch.nn.Tanh()
        return 1.0-sigmoid(x)**2

    def derivative(self,x):
        target = self.target.repeat(len(x), 1)
        x = x-target
        N,dim = x.shape[0],x.shape[1]
        sigmoid = torch.nn.Tanh()
        h_1 = self.layer1(x)
        h_2 = self.layer2(sigmoid(h_1))
        W1 = self.layer1.weight.unsqueeze(1).repeat(1,N,1)
        W2 = self.layer2.weight
        W3 = self.layer3.weight
        dh_1 = self.dsigmoid(h_1).T.unsqueeze(2).repeat(1,1,dim)*W1
        for i in range(dim):
            dh_1[:,:,i] = torch.mm(W2,dh_1[:,:,i])
        dh_2 = self.dsigmoid(h_2).T.unsqueeze(2).repeat(1,1,dim)*dh_1
        for i in range(dim):
            dh_2[:,:,i] = torch.mm(W3,dh_2[:,:,i])
        return dh_2[0]#+self._eps*x*2



class PositiveNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output,regular=False,smooth_relu_thresh=0.1,eps=1e-3):
        super(PositiveNet, self).__init__()
        self._d = smooth_relu_thresh
        self._eps = eps
        self.mode = regular
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def smooth_relu(self, x):
        relu = x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        # sq = (2 * self._d * relu.pow(3) - relu.pow(4)) / (2 * self._d ** 3)
        sq = relu.pow(2)/(2*self._d)
        lin = x - self._d / 2
        return torch.where(relu < self._d, sq, lin)

    def forward(self, state):
        output = self.smooth_relu(self.net(state)-self.net(torch.zeros_like(state)))
        return output+self._eps * state.pow(2).sum(dim=1).view(-1,1) if self.mode else output




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

    def forward(self, t, state):
        dstate = torch.zeros_like(state)
        x,y = state[:,0],state[:,1]
        dstate[:,0] = self.a1*x**self.n/(self.s**self.n+x**self.n)+self.b1*self.s**self.n/(self.s**self.n+y**self.n)-self.k*x
        dstate[:,1] = self.a2 * y ** self.n / (self.s ** self.n + y ** self.n) + self.b2 * self.s ** self.n / (self.s ** self.n + x ** self.n) - self.k * y
        return dstate



class Augment(nn.Module):

    def __init__(self,n_input, n_hidden, n_output,input_shape,layer_sizes=[64, 64],case='icnn', smooth_relu_thresh=0.1, eps=1e-3):
        super(Augment, self).__init__()
        self.a1 = 1.
        self.a2 = 1.
        self.b1 = 0.2
        self.b2 = 0.2
        self.k = 1.1
        self.n = 2
        self.s = 0.5
        self.scale = 10.
        self.strength = 0.5
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self._eps = eps
        self.input_shape = input_shape

        self.case = case
        if case == 'icnn':
            self._control = ControlNet(n_input, n_hidden, n_output)
            self._lya = ICNN(input_shape, layer_sizes, smooth_relu_thresh,eps)
        if case == 'quad':
            self._lya = QuadVNet(n_input, n_hidden, n_output, eps).to(device)
            self._control = ControlNormalNet(n_input, n_hidden, n_output).to(device)
        if self.case == 'nlc':
            self._lya = NLCVNet(n_input, n_hidden, 1,2e-3).to(device)
            self._control = ControlNLCNet(n_input, n_hidden, n_output).to(device)
        self.odeint = odeint_adjoint
        self.init_pos_err = nn.Parameter(torch.tensor([0.0]))
        self.init_vel_err = nn.Parameter(torch.tensor([0.0]))



    def get_initial_state(self,data):
        # data shape: torch.size([2])
        state = (data[0:1], data[1:2], self.init_pos_err,self.init_vel_err)
        return self.t0, state

    def GRN(self,t,state):
        dstate = torch.zeros_like(state)
        x,y = state[:,0],state[:,1]
        dstate[:,0] = self.scale*(self.a1*(x/self.scale)**self.n/(self.s**self.n+(x/self.scale)**self.n)
                                  +self.b1*self.s**self.n/(self.s**self.n+(y/self.scale)**self.n)-self.k*x/self.scale)
        dstate[:,1] = self.scale*(self.a2 * (y/self.scale) ** self.n / (self.s ** self.n + (y/self.scale) ** self.n)
                                  + self.b2 * self.s ** self.n / (self.s ** self.n + (x/self.scale) ** self.n) - self.k * y/self.scale)
        return dstate

    def forward(self, t, state):
        x,y,e_x,e_y = state
        input = torch.cat((x,y))+torch.cat((e_x,e_y))
        input = input.view(-1,2)
        u = self._control(input)[:,0]
        dx = self.scale*(self.a1*(x/self.scale)**self.n/(self.s**self.n+(x/self.scale)**self.n)
                                  +self.b1*self.s**self.n/(self.s**self.n+(y/self.scale)**self.n)-self.k*x/self.scale+u*(x/self.scale)**self.n/(self.s**self.n+(x/self.scale)**self.n))
        dy = self.scale*(self.a2 * (y/self.scale) ** self.n / (self.s ** self.n + (y/self.scale) ** self.n)
                                  + self.b2 * self.s ** self.n / (self.s ** self.n + (x/self.scale) ** self.n) - self.k * y/self.scale)
        # dx = self.a1*x**self.n/(self.s**self.n+x**self.n)+self.b1*self.s**self.n/(self.s**self.n+y**self.n)-self.k*x+u*x**self.n/(self.s**self.n+x**self.n)
        # dy = self.a2 * y ** self.n / (self.s ** self.n + y ** self.n) + self.b2 * self.s ** self.n / (self.s ** self.n + x ** self.n) - self.k * y
        de_x = -dx
        de_y = -dy
        return dx,dy,de_x,de_y

    def untrigger_fn(self, t, state):
        dx = self.GRN(t,state)
        x,y = state[:,0:1],state[:,1:2]
        u = self._control(state) # u represent the parameter adjustment
        dx[:,0:1] += self.scale*u*(x/self.scale)**self.n/(self.s**self.n+(x/self.scale)**self.n) # multiply the hill function term
        return dx

    def event_fn(self, t, state):
        x,y,e_x,e_y = state
        s = torch.cat((x, y)).view(-1, 2)
        e = torch.cat((e_x, e_y)).view(-1, 2)
        # u = self._control(s+e)[0,0]
        # # dx = self.a1 * x ** self.n / (self.s ** self.n + x ** self.n) + self.b1 * self.s ** self.n / (
        # #             self.s ** self.n + y ** self.n) - self.k * x + u * x ** self.n / (self.s ** self.n + x ** self.n)
        # # dy = self.a2 * y ** self.n / (self.s ** self.n + y ** self.n) + self.b2 * self.s ** self.n / (
        # #             self.s ** self.n + x ** self.n) - self.k * y
        # dx = self.scale * (self.a1 * (x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n)
        #                    + self.b1 * self.s ** self.n / (
        #                                self.s ** self.n + (y / self.scale) ** self.n) - self.k * x / self.scale + u * (
        #                                x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n))
        # dy = self.scale * (self.a2 * (y / self.scale) ** self.n / (self.s ** self.n + (y / self.scale) ** self.n)
        #                    + self.b2 * self.s ** self.n / (
        #                                self.s ** self.n + (x / self.scale) ** self.n) - self.k * y / self.scale)
        # vector_field = torch.tensor([[dx,dy]])
        if self.case == 'icnn':
            V = self._lya(s)
            Vx = self._lya.dicnn_fn(s)

            # print(Vx.shape,u.shape,vector_field.shape)
            du = self.scale*((self._control(s+e)-self._control(s))*(x/self.scale)**self.n/(self.s**self.n+(x/self.scale)**self.n))[0,0]
            dU = torch.tensor([[du,0.0]])
            # g = (Vx*vector_field).sum()#+0.2*V.sum()
            g = (Vx*dU).sum()-self.strength*V.sum()

        if self.case == 'quad':
            s = s.requires_grad_(True)
            V = self._lya(s).to(device)
            Vx = self._lya.derivative(s).to(device)
            du = self.scale * ((self._control(s + e) - self._control(s)) * (x / self.scale) ** self.n / (
                        self.s ** self.n + (x / self.scale) ** self.n))[0, 0]
            dU = torch.tensor([[du, 0.0]])
            g = (Vx * dU).sum().to(device) - self.strength * V.sum().to(device)

        if self.case == 'nlc':
            s = s.requires_grad_(True)
            # V = self._lya(s).to(device)
            Vx = self._lya.derivative(s).to(device)
            du = self.scale * ((self._control(s + e) - self._control(s)) * (x / self.scale) ** self.n / (
                    self.s ** self.n + (x / self.scale) ** self.n))[0, 0]
            dU = torch.tensor([[du, 0.0]])
            L_V = (Vx * self.untrigger_fn(0.0,s)).sum()
            g = (Vx * dU).sum().to(device) + self.strength * L_V.to(device)
            # g = (Vx * dU).sum().to(device) - self.strength * V.sum().to(device)

        # print(du.shape,Vx.shape)
        # g = (Vx*(self._control(s+e)-self._control(s))).sum() - 0.5*V.sum()

        # return self._gamma(torch.cat((e_x,e_y)).view(-1,2))-self._lya(torch.cat((x,y)).view(-1,2))*0.5
        # return (e).pow(2).sum(dim=1) - self._lya(s) * 0.5
        return g

    def get_collision_times(self, data,ntrigger=1):

        event_times = torch.zeros(len(data))

        # t0, state = self.get_initial_state()
        # t0,state = torch.tensor([0.0]),data
        for i in range(len(data)):
            t0, state = self.get_initial_state(data[i])
            event_t, solution = odeint_event(
                self,
                state,
                t0,
                event_fn=self.event_fn,
                reverse_time=False,
                atol=1e-2,
                rtol=1e-2,
                odeint_interface=self.odeint,
                method = 'rk4',
                options=dict(step_size=1e-2)
            )
            # event_times.append(event_t)
            event_times[i]=event_t
            # state = self.state_update(tuple(s[-1] for s in solution))
            # t0 = event_t

        return event_times

    def state_update(self, t, state):
        """Updates state based on an event (collision)."""
        x,y,e_x,e_y = state
        # x = (
        #         x + 1e-7
        # )  # need to add a small eps so as not to trigger the event function immediately.
        # y = ( y + 1e-7 )
        e_x = nn.Parameter(torch.tensor([0.0]))
        e_y = nn.Parameter(torch.tensor([0.0]))
        return (x,y,e_x,e_y)

    def get_collision_times_simulate(self, nbounces=1):

        event_times = []

        t0, state = self.get_initial_state()

        for i in range(nbounces):
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
            event_times.append(event_t)

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return event_times

    def simulate_n(self, nbounces=1):
        event_times = self.get_collision_times_simulate(nbounces)

        # get dense path
        t0, state = self.get_initial_state()
        trajectory = [state[0][None]]
        velocity = [state[1][None]]
        times = [t0.reshape(-1)]
        for event_t in event_times:
            tt = torch.linspace(
                float(t0), float(event_t), int((float(event_t) - float(t0)) * 50)
            )[1:-1]
            tt = torch.cat([t0.reshape(-1), tt, event_t.reshape(-1)])
            solution = odeint(self, state, tt, atol=1e-8, rtol=1e-8)

            trajectory.append(solution[0][1:])
            velocity.append(solution[1][1:])
            times.append(tt[1:])

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return (
            torch.cat(times),
            torch.cat(trajectory, dim=0).reshape(-1),
            torch.cat(velocity, dim=0).reshape(-1),
            event_times,
        )

    def simulate_t(self, state0,times):

        t0 = torch.tensor([0.0]).to(times)

        # Add a terminal time to the event function.
        def event_fn(t, state):
            if t > times[-1] + 1e-7:
                return torch.zeros_like(t)
            event_fval = self.event_fn(t, state)
            return event_fval

        # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
        # must appear in the state in the current implementation.
        state = (state0[0:1], state0[1:2], state0[2:3],state0[3:4])
        # print(state)
        event_times = []

        trajectory = [state[0][None]]
        velocity = [state[1][None]]
        trajectory_events = []
        n_events = 0
        max_events = 2000

        while t0 < times[-1] and n_events < max_events:
            last = n_events == max_events - 1

            if not last:
                event_t, solution = odeint_event(
                    self,
                    state,
                    t0,
                    event_fn=event_fn,
                    atol=1e-8,
                    rtol=1e-8,
                    method="dopri5",
                )
            else:
                event_t = times[-1]

            interval_ts = times[times > t0]
            interval_ts = interval_ts[interval_ts <= event_t]
            interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1)])

            solution_ = odeint(
                self, state, interval_ts, atol=1e-8, rtol=1e-8
            )
            traj_ = solution_[0][1:]  # [0] for position; [1:] to remove intial state.
            trajectory.append(traj_)
            velocity.append(solution_[1][1:])

            if event_t < times[-1]:
                state = tuple(s[-1] for s in solution)

                # update velocity instantaneously.
                state = self.state_update(event_t, state)

                # advance the position a little bit to avoid re-triggering the event fn.
                x,y, *rest = state
                x = x + 1e-7 * self.forward(event_t, state)[0]
                y = y + 1e-7 * self.forward(event_t, state)[1]
                state = x,y, *rest

            event_times.append(event_t)
            t0 = event_t

            n_events += 1
            trajectory_events.append([solution_[i][-1] for i in range(2)])
            # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())

        # trajectory = torch.cat(trajectory, dim=0).reshape(-1)
        # return trajectory, event_times

        return (
            torch.cat(trajectory, dim=0).reshape(-1),
            torch.cat(velocity, dim=0).reshape(-1),
            event_times,n_events,torch.tensor(trajectory_events)
        )


class NETC_high(nn.Module):

    def __init__(self,n_input, n_hidden, n_output,input_shape,layer_sizes=[64, 64],smooth_relu_thresh=0.1, eps=1e-3):
        super(NETC_high, self).__init__()
        self.a1 = 1.
        self.a2 = 1.
        self.b1 = 0.2
        self.b2 = 0.2
        self.k = 1.1
        self.n = 2
        self.s = 0.5
        self.scale = 10.
        self.strength = 0.5
        self.t0 = nn.Parameter(torch.tensor([0.0])).to(device)
        self._eps = eps
        self.input_shape = input_shape

        self._control = ControlNet(n_input, n_hidden, n_output).to(device)
        self._lya = ICNN(input_shape, layer_sizes, smooth_relu_thresh,eps).to(device)
        self._alpha = MonotonicNN(1, [10, 10], nb_steps=50, dev=device).to(device)

        self.odeint = odeint_adjoint
        self.init_x_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.init_y_err = nn.Parameter(torch.tensor([0.0])).to(device)

    def get_initial_state(self,data):
        # data shape: torch.size([3])
        state = (data[0:1], data[1:2], self.init_x_err,self.init_y_err)
        return self.t0, state

    def GRN(self, t, state):
        dstate = torch.zeros_like(state)
        x, y = state[:, 0], state[:, 1]
        dstate[:, 0] = self.scale * (
                    self.a1 * (x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n)
                    + self.b1 * self.s ** self.n / (
                                self.s ** self.n + (y / self.scale) ** self.n) - self.k * x / self.scale)
        dstate[:, 1] = self.scale * (
                    self.a2 * (y / self.scale) ** self.n / (self.s ** self.n + (y / self.scale) ** self.n)
                    + self.b2 * self.s ** self.n / (
                                self.s ** self.n + (x / self.scale) ** self.n) - self.k * y / self.scale)
        return dstate

    def forward(self, t, state):
        x, y, e_x, e_y = state
        input = torch.cat((x, y)) + torch.cat((e_x, e_y))
        input = input.view(-1, 2)
        u = self._control(input)[:, 0]
        dx = self.scale * (self.a1 * (x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n)
                           + self.b1 * self.s ** self.n / (
                                       self.s ** self.n + (y / self.scale) ** self.n) - self.k * x / self.scale + u * (
                                       x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n))
        dy = self.scale * (self.a2 * (y / self.scale) ** self.n / (self.s ** self.n + (y / self.scale) ** self.n)
                           + self.b2 * self.s ** self.n / (
                                       self.s ** self.n + (x / self.scale) ** self.n) - self.k * y / self.scale)
        # dx = self.a1*x**self.n/(self.s**self.n+x**self.n)+self.b1*self.s**self.n/(self.s**self.n+y**self.n)-self.k*x+u*x**self.n/(self.s**self.n+x**self.n)
        # dy = self.a2 * y ** self.n / (self.s ** self.n + y ** self.n) + self.b2 * self.s ** self.n / (self.s ** self.n + x ** self.n) - self.k * y
        de_x = -dx
        de_y = -dy
        return dx, dy, de_x, de_y

    def untrigger_fn(self, t, state):
        dx = self.GRN(t, state)
        x, y = state[:, 0:1], state[:, 1:2]
        u = self._control(state)  # u represent the parameter adjustment
        dx[:, 0:1] += self.scale * u * (x / self.scale) ** self.n / (
                    self.s ** self.n + (x / self.scale) ** self.n)  # multiply the hill function term
        return dx

    def event_fn(self, t, state):
        # positive before trigger time
        x,y,e_x,e_y = state
        s = torch.cat((x, y)).view(-1, 2)
        e = torch.cat((e_x, e_y)).view(-1, 2)

        # V = self._lya(s)
        Vx = self._lya.dicnn_fn(s)

        du = self.scale*((self._control(s+e)-self._control(s))*(x/self.scale)**self.n/(self.s**self.n+(x/self.scale)**self.n))[0,0]
        dU = torch.tensor([[du,0.0]])
        g = (Vx*dU).sum()-self.strength*self._alpha(torch.linalg.norm(s, ord=2).view(-1, 1)).sum()

        # Vx = self._lya.dicnn_fn(s).to(device)
        # g = (Vx * (self._control(s + e) - self._control(s))).sum() - self.strength * self._alpha(torch.linalg.norm(s, ord=2).view(-1, 1))
        return g.to(device)

    def get_collision_times(self, data,ntrigger=1):

        event_times = torch.zeros(len(data))
        # solutions = torch.zeros_like(data)
        # t0, state = self.get_initial_state()
        # t0,state = torch.tensor([0.0]),data
        for i in range(len(data)):
            t0, state = self.get_initial_state(data[i])
            event_t, solution = odeint_event(
                self,
                state,
                t0,
                event_fn=self.event_fn,
                reverse_time=False,
                atol=1e-3,
                rtol=1e-3,
                odeint_interface=self.odeint,
                method = 'rk4',
                options=dict(step_size=1e-3)
            )
            # event_times.append(event_t)
            event_times[i]=event_t
            # solutions[i] = solution
            # state = self.state_update(tuple(s[-1] for s in solution))
            # t0 = event_t

        return event_times

    def state_update(self, t, state):
        """Updates state based on an event (collision)."""
        x, y, e_x, e_y = state
        e_x = nn.Parameter(torch.tensor([0.0]))
        e_y = nn.Parameter(torch.tensor([0.0]))
        return (x, y, e_x, e_y)

    def simulate_t(self, state0, times):

        t0 = torch.tensor([0.0]).to(times)

        # Add a terminal time to the event function.
        def event_fn(t, state):
            if t > times[-1] + 1e-7:
                return torch.zeros_like(
                    t)  # event function h=0 relates to the triggering time, use this to mark the last time as tiggering time
            event_fval = self.event_fn(t, state)
            return event_fval.to(device)

        # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
        # must appear in the state in the current implementation.
        state = (state0[0:1].to(device), state0[1:2].to(device), state0[2:3].to(device), state0[3:4].to(device))
        # print(state)
        event_times = []

        trajectory_x = [state[0][None]]
        trajectory_y = [state[1][None]]
        trajectory_events = []
        control_value = []
        n_events = 0
        max_events = 2000

        while t0 < times[-1] and n_events < max_events:
            last = n_events == max_events - 1

            if not last:
                event_t, solution = odeint_event(
                    self,
                    state,
                    t0,
                    event_fn=event_fn,
                    atol=1e-8,
                    rtol=1e-8,
                    method="dopri5",
                    # method='rk4',
                    # options=dict(step_size=1e-3)
                )
            else:
                event_t = times[-1]

            interval_ts = times[times > t0]
            interval_ts = interval_ts[interval_ts <= event_t]
            interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1)])

            solution_ = odeint(
                self, state, interval_ts, atol=1e-8, rtol=1e-8
            )
            traj_ = solution_[0][1:]  # [0] for position; [1:] to remove intial state.
            trajectory_x.append(traj_)
            trajectory_y.append(solution_[1][1:])
            tensor_state = torch.cat((state[0],state[1])).view(-1,2)
            control_value.append(self._control(tensor_state)[0])
            if event_t < times[-1]:
                state = tuple(s[-1] for s in solution)

                # update velocity instantaneously.
                state = self.state_update(event_t, state)

                # advance the position a little bit to avoid re-triggering the event fn.
                x, y, *rest = state
                x = x + 1e-7 * self.forward(event_t, state)[0]
                y = y + 1e-7 * self.forward(event_t, state)[1]
                state = x, y, *rest

            event_times.append(event_t)
            t0 = event_t

            n_events += 1
            trajectory_events.append([solution_[i][-1] for i in range(2)])

            # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())

        # trajectory = torch.cat(trajectory, dim=0).reshape(-1)

        return (
            torch.cat(trajectory_x, dim=0).reshape(-1),
            torch.cat(trajectory_y, dim=0).reshape(-1),
            event_times, n_events, torch.tensor(trajectory_events)
            # ,torch.stack(control_value)
        )


    def get_collision_times_simulate(self, nbounces=1):

        event_times = []

        t0, state = self.get_initial_state()

        for i in range(nbounces):
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
            event_times.append(event_t)

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return event_times


class LQR_event(nn.Module):

    def __init__(self,S,Q,K,strength=0.5):
        super(LQR_event, self).__init__()
        self.a1 = 1.
        self.a2 = 1.
        self.b1 = 0.2
        self.b2 = 0.2
        self.k = 1.1
        self.n = 2
        self.s = 0.5
        self.scale = 10.
        self.t0 = nn.Parameter(torch.tensor([0.0])).to(device)
        self.odeint = odeint_adjoint
        self.init_x_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.init_y_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.strength = strength
        self.S = S.to(device)
        self.Q = Q.to(device)
        self.K = K.to(device)
        self.B = torch.tensor([[self.scale * (0.62562059 )**self.n / (self.s**self.n + (0.62562059) ** self.n)],[0]]).to(device)
        self.target = torch.tensor([[0.62562059, 0.62562059]]) * self.scale

    def lya(self,data):
        # data size: (num,dim)
        data += -self.target
        Sx = torch.mm(self.S,data.T)
        out = torch.sum(data*Sx.T,dim=1)[:,None]
        return out

    def dlya(self,data):
        # data size: (num,dim)
        data += -self.target
        Sx = 2*torch.mm(self.S,data.T)
        return Sx.T

    def lie_derivative(self, data):
        # data size: (num,dim)
        Sx = torch.mm(self.Q, (data-self.target).T)
        out = torch.sum(data * Sx.T, dim=1)[:, None]
        return out

    def lqr(self,data):
        # G = torch.mm(self.B,self.K)
        # data += -self.target
        # return torch.mm(self.K,data.T).T
        # return -10*data[:,0:1]-5*data[:,1:2]
        return -(self.K*(data-self.target)).sum(dim=1)

    def get_initial_state(self,data):
        # data shape: torch.size([3])
        state = (data[0:1], data[1:2],self.init_x_err,self.init_y_err)
        return self.t0, state


    def GRN(self, t, state):
        dstate = torch.zeros_like(state)
        x, y = state[:, 0], state[:, 1]
        dstate[:, 0] = self.scale * (
                    self.a1 * (x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n)
                    + self.b1 * self.s ** self.n / (
                                self.s ** self.n + (y / self.scale) ** self.n) - self.k * x / self.scale)
        dstate[:, 1] = self.scale * (
                    self.a2 * (y / self.scale) ** self.n / (self.s ** self.n + (y / self.scale) ** self.n)
                    + self.b2 * self.s ** self.n / (
                                self.s ** self.n + (x / self.scale) ** self.n) - self.k * y / self.scale)
        return dstate

    def forward(self, t, state):
        x, y, e_x, e_y = state
        input = torch.cat((x, y)) + torch.cat((e_x, e_y))
        input = input.view(-1, 2)
        u = self.lqr(input)
        # u = u[:,0]
        dx = self.scale * (self.a1 * (x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n)
                           + self.b1 * self.s ** self.n / (
                                   self.s ** self.n + (y / self.scale) ** self.n) - self.k * x / self.scale + u * (
                                   x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n))
        dy = self.scale * (self.a2 * (y / self.scale) ** self.n / (self.s ** self.n + (y / self.scale) ** self.n)
                           + self.b2 * self.s ** self.n / (
                                   self.s ** self.n + (x / self.scale) ** self.n) - self.k * y / self.scale)
        de_x = -dx
        de_y = -dy
        return dx, dy, de_x, de_y

    def untrigger_fn(self, t, state):
        # dx = self.GRN(t, state)
        # x, y = state[:, 0:1], state[:, 1:2]
        # u = self.lqr(state)[:,0:1]
        # # u = 0.0
        # # dx[:, 0:1] += self.scale * u * (x / self.scale) ** self.n / (
        # #             self.s ** self.n + (x / self.scale) ** self.n)  # multiply the hill function term
        # dx[:, 0] = dx[:, 0]
        u = self.lqr(state)
        dstate = torch.zeros_like(state)
        x, y = state[:, 0], state[:, 1]
        dstate[:, 0] = self.scale * (
                    self.a1 * (x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n)
                    + self.b1 * self.s ** self.n / (
                                self.s ** self.n + (y / self.scale) ** self.n) - self.k * x / self.scale) + self.scale*u*(
                                   x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n) #(self.K*(state-self.target)).sum(dim=1)
        dstate[:, 1] = self.scale * (
                    self.a2 * (y / self.scale) ** self.n / (self.s ** self.n + (y / self.scale) ** self.n)
                    + self.b2 * self.s ** self.n / (
                                self.s ** self.n + (x / self.scale) ** self.n) - self.k * y / self.scale)
        return dstate


    def event_fn(self, t, state):
        # positive before trigger time
        x, y, e_x, e_y = state
        s = torch.cat((x, y)).view(-1, 2) - self.target
        e = torch.cat((e_x, e_y)).view(-1, 2)
        g = (self.strength - 1.0) * torch.sum(s * torch.mm(self.Q, s.T).T) + 2 * torch.sum(
            s * torch.mm(self.S, torch.mm(torch.mm(self.B,-self.K), e.T)).T)
        return g.to(device)

    def get_collision_times(self, data,ntrigger=1):

        event_times = torch.zeros(len(data))
        # solutions = torch.zeros_like(data)
        # t0, state = self.get_initial_state()
        # t0,state = torch.tensor([0.0]),data
        for i in range(len(data)):
            t0, state = self.get_initial_state(data[i])
            event_t, solution = odeint_event(
                self,
                state,
                t0,
                event_fn=self.event_fn,
                reverse_time=False,
                atol=1e-3,
                rtol=1e-3,
                odeint_interface=self.odeint,
                method = 'rk4',
                options=dict(step_size=1e-3)
            )
            # event_times.append(event_t)
            event_times[i]=event_t
            # solutions[i] = solution
            # state = self.state_update(tuple(s[-1] for s in solution))
            # t0 = event_t

        return event_times

    def state_update(self, t, state):
        """Updates state based on an event (collision)."""
        x, y, e_x, e_y = state
        e_x = nn.Parameter(torch.tensor([0.0]))
        e_y = nn.Parameter(torch.tensor([0.0]))
        return (x, y, e_x, e_y)

    def simulate_t(self, state0, times):

        t0 = torch.tensor([0.0]).to(times)

        # Add a terminal time to the event function.
        def event_fn(t, state):
            if t > times[-1] + 1e-7:
                return torch.zeros_like(
                    t)  # event function h=0 relates to the triggering time, use this to mark the last time as tiggering time
            event_fval = self.event_fn(t, state)
            return event_fval.to(device)

        # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
        # must appear in the state in the current implementation.
        state = (state0[0:1].to(device), state0[1:2].to(device), state0[2:3].to(device), state0[3:4].to(device))
        # print(state)
        event_times = []

        trajectory_x = [state[0][None]]
        trajectory_y = [state[1][None]]
        trajectory_events = []
        control_value = []
        n_events = 0
        max_events = 2000

        while t0 < times[-1] and n_events < max_events:
            last = n_events == max_events - 1

            if not last:
                event_t, solution = odeint_event(
                    self,
                    state,
                    t0,
                    event_fn=event_fn,
                    atol=1e-8,
                    rtol=1e-8,
                    method="dopri5",
                    # method='rk4',
                    # options=dict(step_size=1e-3)
                )
            else:
                event_t = times[-1]

            interval_ts = times[times > t0]
            interval_ts = interval_ts[interval_ts <= event_t]
            interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1)])

            solution_ = odeint(
                self, state, interval_ts, atol=1e-8, rtol=1e-8
            )
            traj_ = solution_[0][1:]  # [0] for position; [1:] to remove intial state.
            trajectory_x.append(traj_)
            trajectory_y.append(solution_[1][1:])
            tensor_state = torch.cat((state[0],state[1])).view(-1,2)
            # control_value.append(self.quad_qp(tensor_state)[0])
            if event_t < times[-1]:
                state = tuple(s[-1] for s in solution)

                # update velocity instantaneously.
                state = self.state_update(event_t, state)

                # advance the position a little bit to avoid re-triggering the event fn.
                x, y, *rest = state
                x = x + 1e-7 * self.forward(event_t, state)[0]
                y = y + 1e-7 * self.forward(event_t, state)[1]
                state = x, y, *rest

            event_times.append(event_t)
            t0 = event_t

            n_events += 1
            trajectory_events.append([solution_[i][-1] for i in range(2)])

            # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())

        # trajectory = torch.cat(trajectory, dim=0).reshape(-1)

        return (
            torch.cat(trajectory_x, dim=0).reshape(-1),
            torch.cat(trajectory_y, dim=0).reshape(-1),
            event_times, n_events, torch.tensor(trajectory_events)
            # ,torch.stack(control_value)
        )


def variance(data,target,n):
    L,dim = data.shape[0],data.shape[1]
    diff = data-target
    var = torch.sum(diff**2,dim=1)[n:].mean()
    return var

# s = torch.from_numpy(np.random.choice(np.arange(500, dtype=np.int64),10,replace=False))
# # s = torch.from_numpy(
# #     np.random.choice(np.arange(50, dtype=np.int64),10, replace=False))
# data = torch.Tensor(500,2).uniform_(-5,5)
# a = nn.Parameter(torch.tensor([0.0]))
# b = a.reshape(-1)
# print(a.shape,b.shape,s,data[s])
