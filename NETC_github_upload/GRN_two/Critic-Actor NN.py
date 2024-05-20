import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit
from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event
import torch.nn.functional as F
import torch.nn as nn
from functions import *



class CA_event(nn.Module):

    def __init__(self,S,Q,K,R,err_threshold=0.5,lr=0.001):
        super(CA_event, self).__init__()
        self.a1 = 1.
        self.a2 = 1.
        self.b1 = 0.2
        self.b2 = 0.2
        self.k = 1.1
        self.n = 2
        self.s = 0.5
        self.scale = 10.
        self.dim = 2
        self.t0 = nn.Parameter(torch.tensor([0.0])).to(device)
        self.odeint = odeint_adjoint
        self.init_x_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.init_y_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.S = S.to(device)
        self.Q = Q.to(device)
        self.K = K.to(device)
        self.R = R.to(device)
        self.eth = err_threshold # triggering threshold
        self.lr = lr # learning rate
        self.B = torch.tensor([[self.scale * (0.62562059 )**self.n / (self.s**self.n + (0.62562059) ** self.n)],[0]]).to(device)
        self.target = torch.tensor([[0.62562059, 0.62562059]]) * self.scale



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
        x, y, e_x, e_y = state[0:self.dim*2]
        W_c = torch.cat(state[self.dim * 2:self.dim * 2 + self.dim ** 2])
        W_a = torch.cat(state[self.dim * 2 + self.dim ** 2:self.dim * 2 + self.dim ** 2+self.dim]).reshape(1,self.dim)
        input = torch.cat((x, y)) + torch.cat((e_x, e_y))
        input = input.view(-1, 2)
        # u = self.lqr(input)
        # u = -(self.K*(data-self.target)).sum(dim=1)
        # u = torch.mm(W_a, input.T).T
        u = (W_a*(input-self.target)).sum(dim=1)
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
        output = [dx, dy, de_x, de_y]
        for i in range(self.dim**2 + self.dim):
            output.append(nn.Parameter(torch.tensor([0.0])))
        return tuple(output)
        # return dx, dy, de_x, de_y

    def event_fn(self, t, state):
        # positive before trigger time
        x, y, e_x, e_y = state[0:self.dim*2]
        s = torch.cat((x, y)).view(-1, 2) - self.target
        e = torch.cat((e_x, e_y)).view(-1, 2)
        # g = (self.strength - 1.0) * torch.sum(s * torch.mm(self.Q, s.T).T) + 2 * torch.sum(
        #     s * torch.mm(self.S, torch.mm(torch.mm(self.B,-self.K), e.T)).T)
        g = torch.linalg.norm(e, ord=2) - self.eth
        return g.to(device)

    def state_update(self, t, state):
        """Updates state based on an event (collision)."""
        x, y, e_x, e_y = state[0:self.dim*2]
        W_c = torch.cat(state[self.dim * 2:self.dim * 2 + self.dim ** 2]).reshape(self.dim, self.dim)
        W_a = torch.cat(state[self.dim * 2 + self.dim ** 2:self.dim * 2 + self.dim ** 2+self.dim]).reshape(1, self.dim)
        input = torch.cat((x, y)) + torch.cat((e_x, e_y))
        input = input.view(-1, 2)
        u = (W_a * (input - self.target)).sum(dim=1)
        dx = self.scale * (self.a1 * (x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n)
                           + self.b1 * self.s ** self.n / (
                                   self.s ** self.n + (y / self.scale) ** self.n) - self.k * x / self.scale + u * (
                                   x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n))
        dy = self.scale * (self.a2 * (y / self.scale) ** self.n / (self.s ** self.n + (y / self.scale) ** self.n)
                           + self.b2 * self.s ** self.n / (
                                   self.s ** self.n + (x / self.scale) ** self.n) - self.k * y / self.scale)
        e_x = nn.Parameter(torch.tensor([0.0]))
        e_y = nn.Parameter(torch.tensor([0.0]))
        g = torch.tensor([[(x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n),0.0]])
        e_a = torch.mm(W_a, torch.cat(state[0:self.dim]).reshape(self.dim, 1)) + 0.5 * torch.mm(g,
            torch.mm(W_c, torch.cat(state[0:self.dim]).reshape(self.dim, 1)))
        W_a = W_a - self.lr * e_a * (torch.cat(state[0:self.dim]).reshape(1, self.dim))
        W_a = W_a.flatten()
        e_c = torch.mm(torch.mm(torch.cat(state[0:self.dim]).reshape(1, self.dim), self.Q.to(torch.float64)),
                       torch.cat(state[0:self.dim]).reshape(self.dim, 1)) + \
              u**2*self.R.to(torch.float64)+ torch.sum(
            torch.mm(W_c, torch.cat(state[0:self.dim]).reshape(self.dim, 1)).T[0] * torch.tensor([dx, dy]))
        zeta = (torch.cat(state[0:self.dim]).reshape(1, self.dim)).repeat(self.dim, 1) * torch.tensor([[dx], [dy]]).repeat(1, self.dim)
        W_c = W_c - self.lr * e_c[0] * zeta / (1 + torch.sum(zeta ** 2)) ** 2
        W_c = W_c.flatten()
        output = [x + 1e-7 * dx, y + 1e-7 * dy, e_x, e_y]
        for i in range(self.dim ** 2):
            output.append(W_c[i].reshape(1))
        for i in range(self.dim):
            output.append(W_a[i].reshape(1))
        return tuple(output)
        # return (x, y, e_x, e_y)

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
        # state = (state0[0:1].to(device), state0[1:2].to(device), state0[2:3].to(device), state0[3:4].to(device))
        state = tuple([state0[i:i + 1] for i in range(self.dim * 2+self.dim**2+self.dim)])

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


# seed =  # 4,6
torch.manual_seed(369)
N = 5000  # sample size
D_in = 2  # input dimension
H1 = 20  # hidden dimension
D_out = 1  # output dimension
data = torch.Tensor(N, D_in).uniform_(-1, 1).requires_grad_(True)
target = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).to(device)

import scipy.io as scio
lqr_data = scio.loadmat('./data/lqr_data.mat')
S = torch.from_numpy(np.array(lqr_data['S1'])).to(device)
Q = torch.from_numpy(np.array(lqr_data['Q1'])).to(device)
R = torch.from_numpy(np.array(lqr_data['R'])).to(device)
K = torch.from_numpy(np.array(lqr_data['K1'])).to(device)

print(K.shape)
def table_data():
    model = CA_event(S, Q, K, R, 0.2, 0.01)

    test_times = torch.linspace(0, 20, 1000).to(device)
    init_s = torch.tensor([[0.0582738, 0.85801853]]) * 10.
    init_state = torch.cat((init_s[0], torch.zeros([2]),S.flatten(),-K.flatten()))

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    seed_list = [2, 4, 5, 6, 7]
    for i in range(5):
        with torch.no_grad():
            seed = seed_list[i]  #
            # seed = i
            np.random.seed(seed)
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
            init_noise = torch.cat((data[s][0, 0:2], torch.zeros([2+2**2+2]).to(device))).to(device)
            trajectory_x, trajectory_y, event_times, n_events, traj_events = model.simulate_t(init_state + init_noise,
                                                                                              test_times)
            traj_events += -target
            event_times = torch.tensor(event_times)
            event_num.append(n_events)
            min_traj.append(
                torch.min(torch.sqrt((trajectory_x - target[0, 0]) ** 2 + (trajectory_y - target[0, 0]) ** 2)))
            cat_data = torch.cat((trajectory_x.unsqueeze(1), trajectory_y.unsqueeze(1)), dim=1)
            var_list.append(variance(cat_data, target, n=900))
            min_inter.append((event_times[1:] - event_times[:-1]).min())
            if len(traj_events) >= 11:
                min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            else:
                min_traj_events.append(torch.linalg.norm(traj_events[-1], ord=2))
            print(seed, trajectory_x[-1], min_traj[i], n_events, min_inter[i])

    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(), torch.tensor(var_list).mean())

    '''
    2000.0 tensor(9.6486) tensor(2.3880e-05) tensor(53.0205)
    '''
    # solution = solution.cpu().detach().numpy()
    test_times = test_times.cpu().detach().numpy()
    trajectory_x = trajectory_x.cpu().detach().numpy()

    # torch.save(event_times, './data/netc_event times.pt')
    # np.save('./data/fig1/netc_traj', trajectory_x)

    # plt.subplot(121)
    # plt.plot(test_times, solution[:, 0, 0], label='control')
    # plt.legend()

    # plt.subplot(122)
    plt.plot(test_times, trajectory_x)
    plt.title('n_events:{}'.format(n_events))

    plt.show()
table_data()
'''
results:
threshold = 0.2, R = 0.01, Q = [[10,0],[0,10]], lr = 0.01
2 tensor(6.2063) tensor(0.0182) 257 tensor(0.0005)
4 tensor(6.2707) tensor(0.0022) 774 tensor(0.0005)
5 tensor(6.3529) tensor(0.0012) 472 tensor(0.0005)
6 tensor(6.3502) tensor(0.0006) 1088 tensor(0.0005)
7 tensor(6.0909) tensor(0.0024) 433 tensor(0.0005)
604.8 tensor(0.0049) tensor(0.0005) tensor(4.0908) tensor(0.0078)

'''