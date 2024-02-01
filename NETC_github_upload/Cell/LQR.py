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
        self.B = 1.
        self.A = torch.from_numpy(np.load('./data/A_100.npy'))
        self.h = 2
        self.t0 = nn.Parameter(torch.tensor([0.0])).to(device)
        self.odeint = odeint_adjoint
        self.strength = strength
        self.dim = 100
        self.S = S.to(device)
        self.Q = Q.to(device)
        self.K = K.to(device)
        self.target = torch.from_numpy(np.load('./data/target_100.npy')).view([1,-1])
        self.B_ast = torch.diag((self.target**2/(1+self.target**2))[0])

    def lya(self,data):
        # data size: (num,dim)
        Sx = torch.mm(self.S,data.T)
        out = torch.sum(data*Sx.T,dim=1)[:,None]
        return out

    def dlya(self,data):
        # data size: (num,dim)
        Sx = torch.mm(self.S,data.T)
        return Sx.T

    def lie_derivative(self, data):
        # data size: (num,dim)
        Sx = torch.mm(self.Q, data.T)
        out = torch.sum(data * Sx.T, dim=1)[:, None]
        return out

    def lqr(self,data):
        # return -(self.K*(data-self.target)).sum(dim=1)
        return -torch.mm(self.K,(data-self.target).view(self.dim,-1)).T

    def get_initial_state(self, data):
        # data shape: torch.size([2])
        # state = (data[0:1], data[1:2], self.init_pos_err,self.init_vel_err)
        state = tuple([data[i:i + 1] for i in range(self.dim)]) + tuple(
            [self.init_err[i:i + 1] for i in range(self.dim)])
        return self.t0, state

    def Cell(self, t, state):
        state = state.view(-1,self.dim)
        Y = state.T
        return -self.B * state + torch.mm(self.A, Y ** 2 / (1 + Y ** 2)).T

    def forward(self, t, state):

        x = torch.cat(state[0:self.dim])
        e_x = torch.cat(state[self.dim:self.dim * 2])
        input = (x + e_x).view(-1, self.dim)

        Y = x.view(self.dim, 1)
        dx = -self.B * x + torch.mm(self.A, Y ** 2 / (1 + Y ** 2)).T[0]
        u = self.lqr(input)[0]
        dx += u * x ** 2 / (1 + x ** 2)  # 对耦合矩阵对角元加控制
        output = []
        for i in range(self.dim):
            output.append(dx[i])
        for i in range(self.dim):
            output.append(-dx[i])
        return tuple(output)


    def untrigger_fn(self, t, state):
        dx = self.Cell(t, state)
        x = state[:, 0:self.dim]
        u = self.lqr(state)  # u represent the parameter adjustment
        dx += u * x ** 2 / (1 + x ** 2)  # multiply the hill function term
        return dx


    def event_fn(self, t, state):
        # positive before trigger time
        s = torch.cat(state[0:self.dim]).view(-1, self.dim)- self.target
        e = torch.cat(state[self.dim:self.dim * 2]).view(-1, self.dim)
        # B = torch.diag((s**2/(1+s**2))[0])
        g = (self.strength - 1.0) * torch.sum(s * torch.mm(self.Q, s.T).T) + 2 * torch.sum(
            s * torch.mm(self.S, torch.mm(torch.mm(self.B_ast, -self.K), e.T)).T)

        return g.to(device)

    def state_update(self, t, state):
        """Updates state based on an event (collision)."""
        # x,y,e_x,e_y = state
        # e_x = nn.Parameter(torch.tensor([0.0]))
        # e_y = nn.Parameter(torch.tensor([0.0]))
        state = state[0:self.dim]
        return state + tuple([nn.Parameter(torch.tensor([0.0])) for i in range(self.dim)])

    def simulate_t(self, state0, times):

        t0 = torch.tensor([0.0]).to(times)

        # Add a terminal time to the event function.
        def event_fn(t, state):
            if t > times[-1] + 1e-7:
                return torch.zeros_like(t)
            event_fval = self.event_fn(t, state)
            return event_fval

        # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
        # must appear in the state in the current implementation.
        # state = (state0[0:1], state0[1:2], state0[2:3], state0[3:4])
        state = tuple([state0[i:i + 1] for i in range(self.dim * 2)])
        # print(state)
        event_times = []

        # trajectory = [state[0][None]]
        # velocity = [state[1][None]]
        trajectories = [[] for _ in range(self.dim)]
        for i in range(self.dim):
            trajectories[i] = [state[i][None]]
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

            # traj_ = solution_[0][1:]  # [0] for position; [1:] to remove intial state.
            # trajectory.append(traj_)
            # velocity.append(solution_[1][1:])
            for i in range(self.dim):
                trajectories[i].append(solution_[i][1:])
            if event_t < times[-1]:
                state = tuple(s[-1] for s in solution)

                # update velocity instantaneously.
                state = self.state_update(event_t, state)

                # advance the position a little bit to avoid re-triggering the event fn.
                # x,y, *rest = state
                # x = x + 1e-7 * self.forward(event_t, state)[0]
                # y = y + 1e-7 * self.forward(event_t, state)[1]
                # state = x,y, *rest

                out = self.forward(event_t, state)
                # for i in range(self.dim):
                #     state[i] += 1e-7 * out[i]
                state = tuple([state[i] + 1e-7 * out[i] for i in range(self.dim)]) + state[self.dim:self.dim * 2]

            event_times.append(event_t)
            t0 = event_t

            n_events += 1
            trajectory_events.append([solution_[i][-1] for i in range(self.dim)])
            # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())

        # trajectory = torch.cat(trajectory, dim=0).reshape(-1)
        # return trajectory, event_times
        for i in range(self.dim):
            trajectories[i] = torch.cat(trajectories[i], dim=0).reshape(-1, 1)
        return (
            torch.cat(trajectories[0:self.dim], dim=1),
            event_times, n_events, torch.tensor(trajectory_events)
        )

#
# torch.manual_seed(369)
# N = 1000  # sample size
# D_in = 100  # input dimension
# H1 = 2 * D_in  # hidden dimension
# D_out = 100  # output dimension
# data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
#
#
# import scipy.io as scio
# lqr_data = scio.loadmat('./data/lqr_data.mat')
# S = torch.from_numpy(np.array(lqr_data['S1'])).to(device)
# Q = torch.from_numpy(np.array(lqr_data['Q'])).to(device)
# K = torch.from_numpy(np.array(lqr_data['K1'])).to(device)
# model = LQR_event(S, Q, K, 0.99)
#
# target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])
#
# test_times = torch.linspace(0, 30, 1000).to(device)
# init_state = torch.zeros([2 * D_in])
# init_state[0:D_in] += torch.from_numpy(np.random.uniform(-0.5, 0.5, [D_in]))
#
# init_s = (init_state[0:D_in]).view(1, -1)
# original = odeint(model.Cell, init_s, test_times)
# solution = odeint(model.untrigger_fn, init_s, test_times)
#
# # solution = solution.cpu().detach().numpy()
# original = original.cpu().detach().numpy()
#
# # plt.plot(test_times, solution[:,0,0],label='control')
# plt.plot(test_times, original[:, 0, 0],label='original')
# plt.legend()
# plt.show()


def table_data():
    torch.manual_seed(369)
    N = 1000  # sample size
    D_in = 100  # input dimension
    H1 = 2 * D_in  # hidden dimension
    D_out = 100  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
    target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])


    import scipy.io as scio
    lqr_data = scio.loadmat('./data/lqr_data.mat')
    S = torch.from_numpy(np.array(lqr_data['S1'])).to(device)
    Q = torch.from_numpy(np.array(lqr_data['Q'])).to(device)
    K = torch.from_numpy(np.array(lqr_data['K1'])).to(device)
    model = LQR_event(S,Q,K,0.5)

    test_times = torch.linspace(0, 30, 1000).to(device)
    init_state = torch.zeros([2 * D_in])

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    # seed_list = [0,3,4,5,6]
    for i in range(5):
        with torch.no_grad():
            # seed =  seed_list[i] #
            seed = i
            np.random.seed(seed)
            init_state[0:D_in] += torch.from_numpy(np.random.uniform(-0.1, 0.1, [D_in]))
            trajectory, event_times, n_events, traj_events = model.simulate_t(init_state, test_times)
            traj_events += -target
            trajectory += -target
            event_times = torch.tensor(event_times)
            event_num.append(n_events)
            min_traj.append(
                torch.min(torch.sqrt(torch.sum(trajectory ** 2, dim=1))))
            var_list.append(variance(trajectory, torch.zeros_like(trajectory), n=900))
            min_inter.append((event_times[1:] - event_times[:-1]).min())
            # min_inter.append(0.0)

            if len(traj_events) >= 11:
                min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            else:
                min_traj_events.append(torch.linalg.norm(traj_events[-1], ord=2))
            print(seed, min_traj[i], n_events, min_inter[i], min_traj_events[i])

    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(), torch.tensor(var_list).mean())

    test_times = test_times.cpu().detach().numpy()
    trajectory = trajectory.cpu().detach().numpy()

    plt.plot(test_times, trajectory[:, 0])
    plt.title('n_events:{}'.format(n_events))

    plt.show()
table_data()
'''
results:
strength = 0.99, R = 0.01
0 tensor(1.6161e-09) 216 tensor(0.0136) tensor(0.0030)
1 tensor(1.6205e-09) 210 tensor(0.0139) tensor(0.0019)
2 tensor(1.6216e-09) 256 tensor(0.0157) tensor(0.0030)
3 tensor(1.6225e-09) 237 tensor(0.0144) tensor(0.0115)
4 tensor(1.6064e-09) 180 tensor(0.0121) tensor(0.0014)
219.8 tensor(1.6174e-09) tensor(0.0140) tensor(0.0042) tensor(2.6896e-18)

strength = 0.5, R = 0.01
0 tensor(1.6269e-09) 447 tensor(0.0341) tensor(0.0001)
1 tensor(1.6270e-09) 449 tensor(0.0339) tensor(0.0005)
2 tensor(1.6270e-09) 449 tensor(0.0326) tensor(0.0051)
3 tensor(1.6270e-09) 451 tensor(0.0029) tensor(0.0035)
4 tensor(1.6270e-09) 450 tensor(0.0018) tensor(0.0004)
449.2 tensor(1.6270e-09) tensor(0.0210) tensor(0.0019) tensor(2.8758e-18)


'''