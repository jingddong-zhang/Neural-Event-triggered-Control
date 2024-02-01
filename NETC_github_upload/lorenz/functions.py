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
        z = self.icnn_fn(x)
        z0 = self.icnn_fn(torch.zeros_like(x))
        dregular = 2*self._eps*x
        dz = self.dsmooth_relu(z-z0).unsqueeze(2).repeat(1,1,dim)*self.inter_dicnn_fn(x)+dregular
        return dz[0]

    def forward(self,x):
        z = self.icnn_fn(x)
        z0 = self.icnn_fn(torch.zeros_like(x))
        regular = self._eps * x.pow(2).sum(dim=1).view(-1,1)
        return self.smooth_relu(z-z0).T+regular

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

    def net(self, x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        # mask = torch.zeros_like(x).to(device)
        # mask[:,1] = 1.0
        return out
        # return out*(x[:,0:1]+x[:,1:2])*mask
    def forward(self, x):
        # mask = torch.zeros_like(x).to(device)
        # mask[:,1] = 1.0
        # return (self.net(x)-self.net(torch.zeros_like(x)))*mask
        return self.net(x)*x
    def lip(self):
        W1 = self.layer1.weight
        W2 = self.layer2.weight
        W3 = self.layer3.weight
        return torch.linalg.norm(W1,ord=2)*torch.linalg.norm(W2,ord=2)*torch.linalg.norm(W3,ord=2)

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

        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)

    def forward(self,x):
        # x_0 = torch.zeros_like(x).to(device)
        # return self.net(x)-self.net(x_0)
        return self.net(x)*x

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

        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)

    def forward(self,x):
        return self.net(x)*x




def weight_norm(W,softplus_c):
    # softplus_c = nn.Parameter(softplus_c)
    absrowsum = torch.sum(torch.abs(W), axis=1)
    # scale = torch.min(1.0, softplus_c / absrowsum)
    scale = softplus_c/absrowsum+1.0-torch.abs(softplus_c/absrowsum-1.0)
    return W * scale[:, None]



class ControllipNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(ControllipNet, self).__init__()
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_output)
        self.c = torch.nn.Parameter(2.0*torch.ones([3]).to(device))

    def forward(self, x):
        sigmoid = torch.nn.ReLU()
        softplus = torch.nn.Softplus()

        self.layer1.weight = nn.Parameter(weight_norm(self.layer1.weight, softplus(self.c[0])))
        self.layer2.weight = nn.Parameter(weight_norm(self.layer2.weight, softplus(self.c[1])))
        self.layer3.weight = nn.Parameter(weight_norm(self.layer3.weight, softplus(self.c[2])))

        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out*x

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


class QuadVNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden,n_output,eps):
        super(QuadVNet, self).__init__()
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_output)
        self._eps = eps

    def forward(self, x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        # return (torch.sum((out*x)**2,dim=1)+self._eps * x.pow(2).sum(dim=1)).view(-1,1)
        return torch.sum((out)**2,dim=1).view(-1,1)+self._eps * x.pow(2).sum(dim=1).view(-1,1)

    def dsigmoid(self,x):
        sigmoid = torch.nn.Tanh()
        return 1.0-sigmoid(x)**2

    def derivative(self,x):
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

    def forward(self, x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out#+self._eps * x.pow(2).sum(dim=1).view(-1,1)

    def dsigmoid(self,x):
        sigmoid = torch.nn.Tanh()
        return 1.0-sigmoid(x)**2

    def derivative(self,x):
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
        return dh_2[0]#+self._eps*x

class Lorenz(nn.Module):

    def __init__(self):
        super(Lorenz, self).__init__()
        self.sigma = 10.
        self.rho = 28.
        self.beta = 8/3

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        x,y,z = x[:,0],x[:,1],x[:,2]
        dx[:,0] = self.sigma*(y-x)
        dx[:,1] = self.rho*x-y-x*z
        dx[:,2] = x*y-self.beta*z
        return dx

class Augment(nn.Module):

    def __init__(self,n_input, n_hidden, n_output,input_shape,layer_sizes=[64, 64], case='icnn',strength=0.5,smooth_relu_thresh=0.1, eps=1e-3):
        super(Augment, self).__init__()
        self.sigma = 10.
        self.rho = 28.
        self.beta = 8/3
        self.strength = strength

        self.t0 = nn.Parameter(torch.tensor([0.0])).to(device)
        self._eps = eps
        self.input_shape = input_shape

        self.case = case
        # self._lya = PositiveNet(n_input,n_hidden,1,True,smooth_relu_thresh,eps)
        if self.case == 'icnn':
            self._lya = ICNN(input_shape, layer_sizes, smooth_relu_thresh,eps).to(device)
            self._control = ControlNet(n_input, n_hidden, n_output).to(device)
        if self.case == 'quad':
            self._lya = QuadVNet(n_input,n_hidden,n_output,eps).to(device)
            self._control = ControlNormalNet(n_input, n_hidden, n_output).to(device)
        if self.case == 'nlc':
            self._lya = NLCVNet(n_input, n_hidden, 1,eps).to(device)
            self._control = ControlNLCNet(n_input, n_hidden, n_output).to(device)

        self.odeint = odeint_adjoint
        self.init_x_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.init_y_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.init_z_err = nn.Parameter(torch.tensor([0.0])).to(device)


    def get_initial_state(self,data):
        # data shape: torch.size([3])
        state = (data[0:1], data[1:2],data[2:3], self.init_x_err,self.init_y_err,self.init_z_err)
        return self.t0, state

    def Lorenz(self, x):
        dx = torch.zeros_like(x)
        x,y,z = x[:,0],x[:,1],x[:,2]
        dx[:,0] = self.sigma*(y-x)
        dx[:,1] = self.rho*x-y-x*z
        dx[:,2] = x*y-self.beta*z
        return dx

    def forward(self, t, state):
        # u = self.control(state[:,0:2]+state[:,2:4])
        # u1,u2 = u[:,0],u[:,1]
        x,y,z,e_x,e_y,e_z = state
        input = torch.cat((x,y,z))+torch.cat((e_x,e_y,e_z)).to(device)
        input = input.view(-1,3)
        u = self._control(input)
        u1, u2, u3 = u[:, 0], u[:, 1], u[:,2]

        dx = self.sigma*(y-x)+u1
        dy = self.rho*x-y-x*z+u2
        dz = x*y-self.beta*z+u3
        de_x = -dx
        de_y = -dy
        de_z = -dz
        return dx.to(device),dy.to(device),dz.to(device),de_x.to(device),de_y.to(device),de_z.to(device)

    def untrigger_fn(self, t, state):
        dx = torch.zeros_like(state)
        x,y,z = state[:,0],state[:,1],state[:,2]
        u = self._control(state)
        # print(state.shape)
        dx[:,0] = self.sigma*(y-x)
        dx[:,1] = self.rho*x-y-x*z
        dx[:,2] = x*y-self.beta*z
        return dx+u

    def event_fn(self, t, state):
        # positive before trigger time
        x,y,z,e_x,e_y,e_z = state
        s = torch.cat((x, y,z)).view(-1, 3).to(device)
        e = torch.cat((e_x, e_y,e_z)).view(-1, 3).to(device)

        '''
        without gamma
        '''
        if self.case == 'icnn':
            V = self._lya(s).to(device)
            Vx = self._lya.dicnn_fn(s).to(device)

            g = (Vx*(self._control(s+e)-self._control(s))).sum().to(device) - self.strength*V.sum().to(device)

            # g = (Vx * (self.proj_control(s + e) - self.proj_control(s))).sum().to(device) - self.sigma * V.sum().to(device)

            # return self._gamma(torch.cat((e_x,e_y)).view(-1,2))-self._lya(torch.cat((x,y)).view(-1,2))*0.5
            # return (e).pow(2).sum(dim=1) - self._lya(s) * 0.5
        if self.case == 'quad':
            s = s.requires_grad_(True)
            V = self._lya(s).to(device)
            Vx = self._lya.derivative(s).to(device)
            # Vx = torch.autograd.grad(V.sum(), s, create_graph=True)[0]
            g = (Vx * (self._control(s + e) - self._control(s))).sum().to(device) - self.strength * V.sum().to(device)

        if self.case == 'nlc':
            s = s.requires_grad_(True)
            V = self._lya(s).to(device)
            Vx = self._lya.derivative(s).to(device)
            L_V = (Vx * self.untrigger_fn(0.0,s)).sum()
            g = (Vx * (self._control(s + e) - self._control(s))).sum().to(device) + self.strength * L_V.to(device)
            # g = (Vx * (self._control(s + e) - self._control(s))).sum().to(device) - self.strength * V.sum().to(device)
        return g

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
        x,y,z,e_x,e_y,e_z = state
        # x = (
        #         x + 1e-7
        # )  # need to add a small eps so as not to trigger the event function immediately.
        # y = ( y + 1e-7 )
        e_x = nn.Parameter(torch.tensor([0.0])).to(device)
        e_y = nn.Parameter(torch.tensor([0.0])).to(device)
        e_z = nn.Parameter(torch.tensor([0.0])).to(device)
        return (x,y,z,e_x,e_y,e_z)

    def simulate_t(self, state0,times):

        t0 = torch.tensor([0.0]).to(times)

        # Add a terminal time to the event function.
        def event_fn(t, state):
            if t > times[-1] + 1e-7:
                return torch.zeros_like(t) # event function h=0 relates to the triggering time, use this to mark the last time as tiggering time
            event_fval = self.event_fn(t, state)
            return event_fval.to(device)

        # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
        # must appear in the state in the current implementation.
        state = (state0[0:1].to(device), state0[1:2].to(device), state0[2:3].to(device),state0[3:4].to(device),state0[4:5].to(device),state0[5:6].to(device))
        # print(state)
        event_times = []

        trajectory_x = [state[0][None]]
        trajectory_y = [state[1][None]]
        trajectory_z = [state[2][None]]
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
            trajectory_z.append(solution_[2][1:])
            tensor_state = torch.cat((state[0], state[1], state[2])).view(-1, 3)
            control_value.append(self._control(tensor_state)[0])
            if event_t < times[-1]:
                state = tuple(s[-1] for s in solution)

                # update velocity instantaneously.
                state = self.state_update(event_t, state)

                # advance the position a little bit to avoid re-triggering the event fn.
                x,y,z, *rest = state
                x = x + 1e-7 * self.forward(event_t, state)[0]
                y = y + 1e-7 * self.forward(event_t, state)[1]
                z = z + 1e-7 * self.forward(event_t, state)[2]
                state = x,y,z, *rest

            event_times.append(event_t)
            t0 = event_t

            n_events += 1
            trajectory_events.append([solution_[i][-1] for i in range(3)])
            # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())

        # trajectory = torch.cat(trajectory, dim=0).reshape(-1)
        # return trajectory, event_times

        return (
            torch.cat(trajectory_x, dim=0).reshape(-1),
            torch.cat(trajectory_y, dim=0).reshape(-1),
            torch.cat(trajectory_z, dim=0).reshape(-1),
            event_times,n_events,torch.tensor(trajectory_events)
            # ,torch.stack(control_value)
        )

    # def simulate_t(self, state0,times):
    #
    #     t0 = torch.tensor([0.0]).to(times)
    #
    #     # Add a terminal time to the event function.
    #     def event_fn(t, state):
    #         if t > times[-1] + 1e-7:
    #             return torch.zeros_like(t) # event function h=0 relates to the triggering time, use this to mark the last time as tiggering time
    #         event_fval = self.event_fn(t, state)
    #         return event_fval.to(device)
    #
    #     # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
    #     # must appear in the state in the current implementation.
    #     state = (state0[0:1].to(device), state0[1:2].to(device), state0[2:3].to(device),state0[3:4].to(device),state0[4:5].to(device),state0[5:6].to(device))
    #     # print(state)
    #     event_times = []
    #
    #     trajectory_x = [state[0][None]]
    #     trajectory_y = [state[1][None]]
    #     trajectory_z = [state[2][None]]
    #
    #     n_events = 0
    #     max_events = 2000
    #
    #     while t0 < times[-1] and n_events < max_events:
    #         last = n_events == max_events - 1
    #
    #         if not last:
    #             event_t, solution = odeint_event(
    #                 self,
    #                 state,
    #                 t0,
    #                 event_fn=event_fn,
    #                 atol=1e-8,
    #                 rtol=1e-8,
    #                 method="dopri5",
    #                 # atol=1e-3,
    #                 # rtol=1e-3,
    #                 # method='rk4',
    #                 # options=dict(step_size=1e-3)
    #             )
    #         else:
    #             event_t = times[-1]
    #
    #         interval_ts = times[times > t0]
    #         interval_ts = interval_ts[interval_ts <= event_t]
    #         interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1)])
    #
    #         solution_ = odeint(
    #             self, state, interval_ts, atol=1e-8, rtol=1e-8
    #         )
    #         traj_ = solution_[0][1:]  # [0] for position; [1:] to remove intial state.
    #         trajectory_x.append(traj_)
    #         trajectory_y.append(solution_[1][1:])
    #         trajectory_z.append(solution_[2][1:])
    #
    #         if event_t < times[-1]:
    #             state = tuple(s[-1] for s in solution)
    #
    #             # update velocity instantaneously.
    #             state = self.state_update(event_t, state)
    #
    #             # advance the position a little bit to avoid re-triggering the event fn.
    #             x,y,z, *rest = state
    #             x = x + 1e-7 * self.forward(event_t, state)[0]
    #             y = y + 1e-7 * self.forward(event_t, state)[1]
    #             z = z + 1e-7 * self.forward(event_t, state)[2]
    #             state = x,y,z, *rest
    #
    #         event_times.append(event_t)
    #         t0 = event_t
    #
    #         n_events += 1
    #
    #         # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())
    #
    #     # trajectory = torch.cat(trajectory, dim=0).reshape(-1)
    #     # return trajectory, event_times
    #
    #     return (
    #         torch.cat(trajectory_x, dim=0).reshape(-1),
    #         torch.cat(trajectory_y, dim=0).reshape(-1),
    #         torch.cat(trajectory_z, dim=0).reshape(-1),
    #         event_times,n_events
    #     )

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


class NETC_high(nn.Module):

    def __init__(self,n_input, n_hidden, n_output,input_shape,layer_sizes=[64, 64],strength=0.5,smooth_relu_thresh=0.1, eps=1e-3):
        super(NETC_high, self).__init__()
        self.sigma = 10.
        self.rho = 28.
        self.beta = 8/3
        self.strength = strength
        self.t0 = nn.Parameter(torch.tensor([0.0])).to(device)
        self._eps = eps
        self.input_shape = input_shape

        self._control = ControlNet(n_input, n_hidden, n_output).to(device)
        self._lya = ICNN(input_shape, layer_sizes, smooth_relu_thresh,eps).to(device)
        self._alpha = MonotonicNN(1, [10, 10], nb_steps=50, dev=device).to(device)

        self.odeint = odeint_adjoint
        self.init_x_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.init_y_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.init_z_err = nn.Parameter(torch.tensor([0.0])).to(device)

    def get_initial_state(self,data):
        # data shape: torch.size([3])
        state = (data[0:1], data[1:2],data[2:3], self.init_x_err,self.init_y_err,self.init_z_err)
        return self.t0, state

    def Lorenz(self, x):
        dx = torch.zeros_like(x)
        x,y,z = x[:,0],x[:,1],x[:,2]
        dx[:,0] = self.sigma*(y-x)
        dx[:,1] = self.rho*x-y-x*z
        dx[:,2] = x*y-self.beta*z
        return dx

    def forward(self, t, state):
        # u = self.control(state[:,0:2]+state[:,2:4])
        # u1,u2 = u[:,0],u[:,1]
        x,y,z,e_x,e_y,e_z = state
        input = torch.cat((x,y,z))+torch.cat((e_x,e_y,e_z)).to(device)
        input = input.view(-1,3)
        u = self._control(input)
        u1, u2, u3 = u[:, 0], u[:, 1], u[:,2]

        dx = self.sigma*(y-x)+u1
        dy = self.rho*x-y-x*z+u2
        dz = x*y-self.beta*z+u3
        de_x = -dx
        de_y = -dy
        de_z = -dz
        return dx.to(device),dy.to(device),dz.to(device),de_x.to(device),de_y.to(device),de_z.to(device)

    def untrigger_fn(self, t, state):
        dx = torch.zeros_like(state)
        x,y,z = state[:,0],state[:,1],state[:,2]
        u = self._control(state)
        # print(state.shape)
        dx[:,0] = self.sigma*(y-x)
        dx[:,1] = self.rho*x-y-x*z
        dx[:,2] = x*y-self.beta*z
        return dx+u

    def event_fn(self, t, state):
        # positive before trigger time
        x,y,z,e_x,e_y,e_z = state
        s = torch.cat((x, y,z)).view(-1, 3).to(device)
        e = torch.cat((e_x, e_y,e_z)).view(-1, 3).to(device)


        Vx = self._lya.dicnn_fn(s).to(device)
        g = (Vx * (self._control(s + e) - self._control(s))).sum() - self.strength * self._alpha(torch.linalg.norm(s, ord=2).view(-1, 1))
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
        x,y,z,e_x,e_y,e_z = state
        # x = (
        #         x + 1e-7
        # )  # need to add a small eps so as not to trigger the event function immediately.
        # y = ( y + 1e-7 )
        e_x = nn.Parameter(torch.tensor([0.0])).to(device)
        e_y = nn.Parameter(torch.tensor([0.0])).to(device)
        e_z = nn.Parameter(torch.tensor([0.0])).to(device)
        return (x,y,z,e_x,e_y,e_z)

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
        state = (state0[0:1].to(device), state0[1:2].to(device), state0[2:3].to(device), state0[3:4].to(device),
                 state0[4:5].to(device), state0[5:6].to(device))
        # print(state)
        event_times = []

        trajectory_x = [state[0][None]]
        trajectory_y = [state[1][None]]
        trajectory_z = [state[2][None]]
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
            trajectory_z.append(solution_[2][1:])
            tensor_state = torch.cat((state[0],state[1],state[2])).view(-1,3)
            control_value.append(self._control(tensor_state)[0])
            if event_t < times[-1]:
                state = tuple(s[-1] for s in solution)

                # update velocity instantaneously.
                state = self.state_update(event_t, state)

                # advance the position a little bit to avoid re-triggering the event fn.
                x, y, z, *rest = state
                x = x + 1e-7 * self.forward(event_t, state)[0]
                y = y + 1e-7 * self.forward(event_t, state)[1]
                z = z + 1e-7 * self.forward(event_t, state)[2]
                state = x, y, z, *rest

            event_times.append(event_t)
            t0 = event_t

            n_events += 1
            # trajectory_events.append(solution_[0:3][-1])
            trajectory_events.append([solution_[i][-1] for i in range(3)])

            # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())

        # trajectory = torch.cat(trajectory, dim=0).reshape(-1)
        # return trajectory, event_times

        return (
            torch.cat(trajectory_x, dim=0).reshape(-1),
            torch.cat(trajectory_y, dim=0).reshape(-1),
            torch.cat(trajectory_z, dim=0).reshape(-1),
            event_times, n_events, torch.tensor(trajectory_events)
            # ,torch.stack(control_value)
        )

    # def simulate_t(self, state0,times):
    #
    #     t0 = torch.tensor([0.0]).to(times)
    #
    #     # Add a terminal time to the event function.
    #     def event_fn(t, state):
    #         if t > times[-1] + 1e-7:
    #             return torch.zeros_like(t) # event function h=0 relates to the triggering time, use this to mark the last time as tiggering time
    #         event_fval = self.event_fn(t, state)
    #         return event_fval.to(device)
    #
    #     # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
    #     # must appear in the state in the current implementation.
    #     state = (state0[0:1].to(device), state0[1:2].to(device), state0[2:3].to(device),state0[3:4].to(device),state0[4:5].to(device),state0[5:6].to(device))
    #     # print(state)
    #     event_times = []
    #
    #     trajectory_x = [state[0][None]]
    #     trajectory_y = [state[1][None]]
    #     trajectory_z = [state[2][None]]
    #
    #     n_events = 0
    #     max_events = 200
    #
    #     while t0 < times[-1] and n_events < max_events:
    #         last = n_events == max_events - 1
    #
    #         if not last:
    #             event_t, solution = odeint_event(
    #                 self,
    #                 state,
    #                 t0,
    #                 event_fn=event_fn,
    #                 atol=1e-8,
    #                 rtol=1e-8,
    #                 method="dopri5",
    #                 # method='rk4',
    #                 # options=dict(step_size=1e-3)
    #             )
    #         else:
    #             event_t = times[-1]
    #
    #         interval_ts = times[times > t0]
    #         interval_ts = interval_ts[interval_ts <= event_t]
    #         interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1)])
    #
    #         solution_ = odeint(
    #             self, state, interval_ts, atol=1e-8, rtol=1e-8
    #         )
    #         traj_ = solution_[0][1:]  # [0] for position; [1:] to remove intial state.
    #         trajectory_x.append(traj_)
    #         trajectory_y.append(solution_[1][1:])
    #         trajectory_z.append(solution_[2][1:])
    #
    #         if event_t < times[-1]:
    #             state = tuple(s[-1] for s in solution)
    #
    #             # update velocity instantaneously.
    #             state = self.state_update(event_t, state)
    #
    #             # advance the position a little bit to avoid re-triggering the event fn.
    #             x,y,z, *rest = state
    #             x = x + 1e-7 * self.forward(event_t, state)[0]
    #             y = y + 1e-7 * self.forward(event_t, state)[1]
    #             z = z + 1e-7 * self.forward(event_t, state)[2]
    #             state = x,y,z, *rest
    #
    #         event_times.append(event_t)
    #         t0 = event_t
    #
    #         n_events += 1
    #
    #         # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())
    #
    #     # trajectory = torch.cat(trajectory, dim=0).reshape(-1)
    #     # return trajectory, event_times
    #
    #     return (
    #         torch.cat(trajectory_x, dim=0).reshape(-1),
    #         torch.cat(trajectory_y, dim=0).reshape(-1),
    #         torch.cat(trajectory_z, dim=0).reshape(-1),
    #         event_times,n_events
    #     )

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




def variance(data,target,n):
    L,dim = data.shape[0],data.shape[1]
    diff = data-target
    var = torch.sum(diff**2,dim=1)[n:].mean()
    return var
