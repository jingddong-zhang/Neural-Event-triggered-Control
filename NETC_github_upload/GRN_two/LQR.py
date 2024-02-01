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
Q = torch.from_numpy(np.array(lqr_data['Q'])).to(device)
K = torch.from_numpy(np.array(lqr_data['K1'])).to(device)
model = LQR_event(S,Q,K,0.5)

test_times = torch.linspace(0, 20, 1000).to(device)
init_s = torch.tensor([[0.0582738, 0.85801853]]) * 10.
init_state = torch.cat((init_s[0], torch.zeros([2])))

# func = model.GRN
# original = odeint(func, init_s, test_times)
# solution = odeint(model.untrigger_fn, init_s, test_times)
#
# solution = solution.cpu().detach().numpy()
# original = original.cpu().detach().numpy()
# test_times = test_times.cpu().detach().numpy()
#
# # plt.subplot(121)
# plt.plot(test_times, solution[:, 0, 0], label='control',marker='o')
# plt.plot(test_times, original[:, 0, 0], label='original')
# plt.legend()
# plt.show()

# def table_data():
while True:
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
            init_noise = torch.cat((data[s][0, 0:2], torch.zeros([2]).to(device))).to(device)
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
strength = 0.5, R = 0.01, Q = [[10,0],[0,10]]
2 tensor(6.2564) tensor(0.0006) 1823 tensor(0.0078)
4 tensor(6.2566) tensor(0.0006) 1821 tensor(0.0078)
5 tensor(6.2566) tensor(0.0006) 1824 tensor(0.0013)
6 tensor(6.2562) tensor(0.0006) 1822 tensor(0.0078)
7 tensor(6.2566) tensor(0.0010) 1788 tensor(0.0032)
1815.6 tensor(0.0007) tensor(0.0056) tensor(2.1934) tensor(1.2945e-06)

strength = 0.5, R = 0.1, Q = [[1,0],[0,1]]
2 tensor(6.2564) tensor(0.0006) 1823 tensor(0.0078)
4 tensor(6.2566) tensor(0.0006) 1821 tensor(0.0078)
5 tensor(6.2566) tensor(0.0006) 1824 tensor(0.0013)
6 tensor(6.2562) tensor(0.0006) 1822 tensor(0.0078)
7 tensor(6.2566) tensor(0.0010) 1788 tensor(0.0032)
1815.6 tensor(0.0007) tensor(0.0056) tensor(2.1934) tensor(1.2945e-06)

strength = 0.5, R = 0.1, Q = [[10,0],[0,10]]

'''