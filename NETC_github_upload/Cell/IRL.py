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

class IRL_event(nn.Module):

    def __init__(self,S,Q,K,R,lr=0.001,eta=1.0,lambday=0.6,lambdax=10.0):
        super(IRL_event, self).__init__()
        self.B = 1.
        self.A = torch.from_numpy(np.load('./data/A_100.npy'))
        self.h = 2
        self.t0 = nn.Parameter(torch.tensor([0.0])).to(device)
        self.odeint = odeint_adjoint
        self.dim = 100
        self.S = S.to(device).to(torch.float64)
        self.Q = Q.to(device).to(torch.float64)
        self.K = K.to(device).to(torch.float64)
        self.R = R.to(device).to(torch.float64)
        self.lr = lr  # learning rate
        self.eta = eta
        self.l = 0.1 * 12
        self.alpha = 0.1
        self.lambday = lambday
        self.lambdax = lambdax
        self.target = torch.from_numpy(np.load('./data/target_100.npy')).view([1,-1])
        self.B_ast = torch.diag((self.target**2/(1+self.target**2))[0])
        self.strength = 0.9
        self.eth = 0.2

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
        W = torch.cat(state[self.dim * 2:self.dim * 2 + self.dim]).reshape(1, self.dim)

        input = (x + e_x).view(-1, self.dim)
        # u = (W_a * (input - self.target))
        # u = self.eta * (-torch.mm(W, (input - self.target).T).T)
        u = -self.eta * torch.mm(W*(input - self.target),self.B_ast)
        # u = -torch.mm(self.K,(input-self.target).view(self.dim,-1)).T
        u = u[0]
        Y = x.view(self.dim, 1)
        dx = -self.B * x + torch.mm(self.A, Y ** 2 / (1 + Y ** 2)).T[0]
        # u = self.lqr(input)[0]
        dx += u * x ** 2 / (1 + x ** 2)  # 对耦合矩阵对角元加控制
        output = []
        for i in range(self.dim):
            output.append(dx[i])
        for i in range(self.dim):
            output.append(-dx[i])
        for i in range(self.dim):
            output.append(nn.Parameter(torch.tensor([0.0])))
        return tuple(output)


    def event_fn(self, t, state):
        # positive before trigger time
        s = torch.cat(state[0:self.dim]).view(-1, self.dim)- self.target
        e = torch.cat(state[self.dim:self.dim * 2]).view(-1, self.dim)
        # B = torch.diag((s**2/(1+s**2))[0])
        # g = (self.strength - 1.0) * torch.sum(s * torch.mm(self.Q, s.T).T) + 2 * torch.sum(
        #     s * torch.mm(self.S, torch.mm(torch.mm(self.B_ast, -self.K), e.T)).T)
        g = torch.linalg.norm(e, ord=2) - self.eth
        return g.to(device)

    def cost_u(self, u, l, num=100):
        '''
        :param u: size (num_controllers,)
        :param l: integral time length
        :param num: num of samples in Riemann integral
        :return: cost_sum of each scaling variable
        '''
        cost = 0.0
        R = torch.diagonal(self.R)
        for i in range(len(u)):
            u_ = u[i]
            r_ = R[i]
            tau = torch.linspace(0, u_, num)
            delta = u_ / num
            cost += 2 * self.eta * l * r_ * delta * torch.sum((tau / self.eta))
        return cost

    def cost_x(self, x, l):
        '''
        :param x: size (length,dim)
        :param l: integral time length
        :param num: num of samples in Riemann integral
        :return: cost_sum of each scaling variable
        '''
        num = len(x)
        discount = torch.exp(-self.alpha * torch.linspace(0, 1, num) * l)
        delta = l / num
        cost = torch.sum(torch.sum(torch.mm(x, self.Q) * x, dim=1) * discount) * delta
        return cost

    def V(self, W, x):
        '''
        :param W: size (dim,dim)
        :param x: size (1,dim)
        :return: V = 0.5x'Wx
        '''
        # return 0.5 * torch.mm(x, torch.mm(W, x.T))[0]
        return 0.5*torch.sum(W*x**2)

    def state_update(self,t,state,solution,hist_t):
        """Updates state based on an event (collision)."""
        if hist_t > self.l:
            hist_t = self.l
        x = torch.cat(state[0:self.dim])
        e_x = torch.cat(state[self.dim:self.dim * 2])
        W = torch.cat(state[self.dim*2:self.dim*2+self.dim]).reshape(1,self.dim)
        input = (x + e_x).view(-1, self.dim)
        # u = -torch.mm(self.K,(input-self.target).view(self.dim,-1)).T
        # u = (W_a * (input - self.target))
        # u = self.eta * (-torch.mm(W, (input-self.target).T).T )
        u = -self.eta * torch.mm(W * (input - self.target), self.B_ast)
        Y = x.view(self.dim, 1)
        dx = -self.B * x + torch.mm(self.A, Y ** 2 / (1 + Y ** 2)).T[0]
        dx += u[0] * x ** 2 / (1 + x ** 2)  # 对耦合矩阵对角元加控制

        e_h = self.cost_u(u[0], hist_t) + self.cost_x(solution - self.target, hist_t) + \
              math.exp(-self.alpha * hist_t) * self.V(W, x.view(1,-1) - self.target) - self.V(W, torch.tensor(
            solution[0:1]) - self.target)
        # zeta = torch.mm(Y - self.target.T, x.view(1,-1) - self.target) * math.exp(
        #     -self.alpha * hist_t) \
        #        - torch.mm(solution[0:1].T - self.target.T, solution[0:1] - self.target)
        zeta = math.exp(-self.alpha * hist_t)*(x.view(1,-1) - self.target)**2-(solution[0:1] - self.target)**2
        W = W - self.lr * e_h * zeta / (1 + torch.sum(zeta ** 2)) ** 2
        W = W.flatten()
        output = []
        for i in range(self.dim):
            output.append(state[i]+1e-7 * dx[i])
        for i in range(self.dim):
            output.append(nn.Parameter(torch.tensor([0.0])))
        for i in range(self.dim):
            output.append(W[i].reshape(1))
        return tuple(output)

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
        # state = tuple([state0[i:i + 1] for i in range(self.dim * 2)])
        state = tuple([state0[i:i + 1] for i in range(self.dim * 2+self.dim)])

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
                inter_traj = torch.cat(([solution_[i].reshape(-1, 1) for i in range(self.dim)]), dim=1)
                state = self.state_update(event_t, state, inter_traj, event_t - t0)
                # advance the position a little bit to avoid re-triggering the event fn.
                # x,y, *rest = state
                # x = x + 1e-7 * self.forward(event_t, state)[0]
                # y = y + 1e-7 * self.forward(event_t, state)[1]
                # state = x,y, *rest



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

import scipy.io as scio
lqr_data = scio.loadmat('./data/lqr_data.mat')
S = torch.from_numpy(np.array(lqr_data['S1'])).to(device)
Q = torch.from_numpy(np.array(lqr_data['Q1'])).to(device)
R = torch.from_numpy(np.array(lqr_data['R'])).to(device)
K = torch.from_numpy(np.array(lqr_data['K1'])).to(device)
print(K.shape,S.shape)

def table_data():
    model = IRL_event(S, Q, K, R, lr = 0.01, eta = 1.0, lambday = 0.6, lambdax = 0.1)
    torch.manual_seed(369)
    N = 1000  # sample size
    D_in = 100  # input dimension
    H1 = 2 * D_in  # hidden dimension
    D_out = 100  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
    target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])




    test_times = torch.linspace(0, 30, 1000).to(device)
    # init_state = torch.cat((torch.zeros([2*D_in]), torch.diagonal(S)))
    init_state = torch.cat((torch.zeros([2 * D_in]), torch.ones(D_in)*5))
    print(init_state.shape)
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
lr = 0.01, eta = 1.0, lambday = 0.6, lambdax = 0.1
0 tensor(0.0002) 384 tensor(0.0034) tensor(38.2861)
1 tensor(0.0001) 388 tensor(0.0035) tensor(38.1486)
2 tensor(0.0001) 370 tensor(0.0035) tensor(38.1054)
3 tensor(6.0386e-05) 360 tensor(0.0033) tensor(38.1118)
4 tensor(0.0002) 350 tensor(0.0033) tensor(37.9627)
370.4 tensor(0.0001) tensor(0.0034) tensor(38.1229) tensor(0.0040)
'''