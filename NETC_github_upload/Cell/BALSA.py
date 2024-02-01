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
from cvxopt import solvers,matrix
from functions import *


class BALSA(nn.Module):

    def __init__(self,p=20.0,strength=0.5):
        super(BALSA, self).__init__()
        self.B = 1.
        self.A = torch.from_numpy(np.load('./data/A_100.npy'))
        self.h = 2
        self.t0 = nn.Parameter(torch.tensor([0.0])).to(device)
        self.odeint = odeint_adjoint
        self.strength = strength
        self.dim = 100
        self.epi = 0.1 #
        self.p = p #
        self.target = torch.from_numpy(np.load('./data/target_100.npy')).view([1,-1])

    def quad_qp(self,state):
        state = state.cpu().detach().numpy()
        target = np.load('./data/target_100.npy')
        target = target.reshape([1,-1])
        state = np.float64(state)
        P = matrix(np.diag([1.0 for i in range(self.dim)]+[2 * self.p]))
        q = matrix(np.zeros([self.dim+1])) # dim+soft constraint variable: dim+1
        G = matrix(np.concatenate(((state-target)*state**2/(1+state**2),np.array([[-1.0]])),axis=1,dtype=np.float64))
        diff = state-target
        Y = state.T
        vector = -self.B * state + np.matmul(self.A.detach().numpy(), Y ** 2 / (1 + Y ** 2)).T
        h = matrix(np.array([-np.sum(diff*vector)-np.sum(diff**2)/2],dtype=np.float64))
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h)  # 调用优化函数solvers.qp求解
        u = np.array(sol['x'])
        return torch.from_numpy(u)

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
        u = self.quad_qp(input)[:self.dim,0]
        dx += u * x ** 2 / (1 + x ** 2)  # 对耦合矩阵对角元加控制
        # dx[0] += u[0]*x[0] ** 2 / (1 + x[0] ** 2)
        output = []
        for i in range(self.dim):
            output.append(dx[i])
        for i in range(self.dim):
            output.append(-dx[i])
        return tuple(output)

    def untrigger_fn(self, t, state):
        dx = self.Cell(t, state)
        x = state[:, 0:self.dim]
        u = self.quad_qp(state)[:self.dim]  # u represent the parameter adjustment
        dx += u * x ** 2 / (1 + x ** 2)  # multiply the hill function term
        return dx

    def event_fn(self, t, state):
        # positive before trigger time
        s = torch.cat(state[0:self.dim]).view(-1, self.dim)
        e = torch.cat(state[self.dim:self.dim * 2]).view(-1, self.dim)
        V = torch.sum(0.5*(s-self.target)**2,dim=1)
        Vx = s-self.target
        du = (((self.quad_qp(s + e) - self.quad_qp(s))[:self.dim]).T * (s) ** self.h / (1 + s ** self.h))
        g = (Vx*du).sum() - self.strength * V.sum()

        return g.to(device)

    def state_update(self, t, state):
        """Updates state based on an event (collision)."""
        # x,y,e_x,e_y = state
        # e_x = nn.Parameter(torch.tensor([0.0]))
        # e_y = nn.Parameter(torch.tensor([0.0]))
        state = state[0:self.dim]
        return state+tuple([nn.Parameter(torch.tensor([0.0])) for i in range(self.dim)])
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
        state = tuple([state0[i:i+1] for i in range(self.dim*2)])
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
                state = tuple([state[i]+1e-7 * out[i] for i in range(self.dim)]) + state[self.dim:self.dim*2]

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

def table_data():
    N = 1000  # sample size
    D_in = 100  # input dimension
    H1 = 2 * D_in  # hidden dimension
    D_out = 100  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
    target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])

    model = BALSA(50.0,0.5)


    test_times = torch.linspace(0, 30, 1000).to(device)
    init_state = torch.zeros([2 * D_in])


    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    seed_list = [0,3,4,5,6]
    for i in range(5):
        with torch.no_grad():
            seed =  seed_list[i] #
            # seed = i
            np.random.seed(seed)
            init_state[0:D_in] += torch.from_numpy(np.random.uniform(-0.1, 0.1, [D_in]))
            trajectory, event_times, n_events, traj_events = model.simulate_t(init_state, test_times)
            traj_events += -target
            trajectory += -target
            event_times = torch.tensor(event_times)
            event_num.append(n_events)
            min_traj.append(
                torch.min(torch.sqrt(torch.sum(trajectory**2,dim=1))))
            var_list.append(variance(trajectory, torch.zeros_like(trajectory), n=900))
            min_inter.append((event_times[1:] - event_times[:-1]).min())
            # min_inter.append(0.0)

            if len(traj_events) >= 11:
                min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            else:
                min_traj_events.append(torch.linalg.norm(traj_events[-1], ord=2))
            print(seed,  min_traj[i], n_events, min_inter[i],min_traj_events[i])

    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(), torch.tensor(var_list).mean())
    test_times = test_times.cpu().detach().numpy()
    trajectory = trajectory.cpu().detach().numpy()


    plt.plot(test_times, trajectory[:,0])
    plt.title('n_events:{}'.format(n_events))

    plt.show()


table_data()
'''
0 tensor(5.0358e-08) 5 tensor(0.1078)
3 tensor(5.0172e-08) 4 tensor(0.3283)
4 tensor(5.0057e-08) 4 tensor(0.4684)
5 tensor(5.0357e-08) 4 tensor(0.4860)
6 tensor(4.9687e-08) 4 tensor(0.7282)
3.8 tensor(5.0440e-08) tensor(0.6242) tensor(6.2555e-08) tensor(1504.7408)

u:coefficient:1.0
0 tensor(5.0190e-08) 25 tensor(0.0017) tensor(36.6207)
1 tensor(5.2767e-08) 20 tensor(0.0006) tensor(36.2098)
2 tensor(4.9738e-08) 16 tensor(0.0026) tensor(35.2452)
3 tensor(5.0238e-08) 17 tensor(0.0057) tensor(35.1037)
4 tensor(5.0485e-08) 15 tensor(0.0060) tensor(35.0403)
5 tensor(5.0348e-08) 15 tensor(0.0071) tensor(35.0939)
6 tensor(5.0012e-08) 12 tensor(0.0082) tensor(20.4279)
7 tensor(5.0371e-08) 11 tensor(0.0242) tensor(8.8379e-08)
8 tensor(5.0443e-08) 9 tensor(0.0431) tensor(6.4740e-08)
9 tensor(5.0964e-08) 6 tensor(0.1110) tensor(6.0609e-08)
10 tensor(5.0101e-08) 7 tensor(0.0949) tensor(8.2206e-08)
11 tensor(5.3190e-08) 6 tensor(0.0824) tensor(7.7751e-08)
12 tensor(4.9285e-08) 6 tensor(0.1561) tensor(5.0084e-08)
13 tensor(4.9980e-08) 5 tensor(0.1983) tensor(5.5037e-08)
14 tensor(4.9228e-08) 6 tensor(0.1130) tensor(5.0719e-08)
15 tensor(4.9757e-08) 7 tensor(0.0937) tensor(7.0126e-08)
16 tensor(5.0028e-08) 6 tensor(0.1608) tensor(6.8512e-08)
17 tensor(4.8823e-08) 7 tensor(0.0780) tensor(8.7088e-08)
18 tensor(5.0203e-08) 7 tensor(0.1079) tensor(9.8179e-08)
19 tensor(4.9793e-08) 7 tensor(0.1037) tensor(5.9251e-08)


0,1,2,3,4:
18.6 tensor(5.0684e-08) tensor(0.0033) tensor(35.6439) tensor(3.4871e-15)

'''