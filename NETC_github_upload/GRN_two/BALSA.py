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
        self.epi = 0.1 #
        self.p = p #
        self.target = torch.tensor([[0.62562059, 0.62562059]]) * self.scale

    def quad_qp(self,state):
        state = state.cpu().detach().numpy()
        state = np.float64(state)
        x, y = state[:, 0], state[:, 1]
        P = matrix(np.diag([1.0, 2 * self.p]))
        q = matrix([0.0, 0.0]) # dim+soft constraint variable: dim+1
        G = matrix(np.array([[self.scale*(x-0.62562059*self.scale)*(
                                       x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n),-1.0]],dtype=np.float64))
        h = matrix(np.array([-(x-0.62562059*self.scale)*self.scale * (
                    self.a1 * (x / self.scale) ** self.n / (self.s ** self.n + (x / self.scale) ** self.n)
                    + self.b1 * self.s ** self.n / (
                                self.s ** self.n + (y / self.scale) ** self.n) - self.k * x / self.scale)-(y-0.62562059*self.scale)*self.scale * (
                    self.a2 * (y / self.scale) ** self.n / (self.s ** self.n + (y / self.scale) ** self.n)
                    + self.b2 * self.s ** self.n / (
                                self.s ** self.n + (x / self.scale) ** self.n) - self.k * y / self.scale)-((x-0.62562059*self.scale)**2+(y-0.62562059*self.scale)**2)/(2)],dtype=np.float64))
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h)  # 调用优化函数solvers.qp求解
        u = np.array(sol['x'])
        # u1,u2,d=osqp(x1,x2)
        return torch.from_numpy(u)

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
        u,d = self.quad_qp(input)
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
        dx = self.GRN(t, state)
        x, y = state[:, 0:1], state[:, 1:2]
        u, d = self.quad_qp(state)
        dx[:, 0:1] += self.scale * u * (x / self.scale) ** self.n / (
                    self.s ** self.n + (x / self.scale) ** self.n)  # multiply the hill function term
        return dx

    def event_fn(self, t, state):
        # positive before trigger time
        x, y, e_x, e_y = state
        s = torch.cat((x, y)).view(-1, 2)
        e = torch.cat((e_x, e_y)).view(-1, 2)
        V = torch.sum(0.5*(s-self.target)**2,dim=1)
        Vx = s-self.target
        du = self.scale * ((self.quad_qp(s + e) - self.quad_qp(s)) * (x / self.scale) ** self.n / (
                    self.s ** self.n + (x / self.scale) ** self.n))[0, 0]
        dU = torch.tensor([[du, 0.0]])
        g = (Vx*dU).sum() - self.strength * V.sum()
        return g.to(device)

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

def table_data():
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 2  # input dimension
    H1 = 20  # hidden dimension
    D_out = 1  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-1, 1).requires_grad_(True)
    target = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).to(device)


    model = BALSA(50.0,0.5)


    test_times = torch.linspace(0, 20, 1000).to(device)
    init_s = torch.tensor([[0.0582738, 0.85801853]]) * 10.
    init_state = torch.cat((init_s[0], torch.zeros([2])))

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    seed_list = [2, 4, 5, 6, 7]
    for i in range(5):
        with torch.no_grad():
            seed =  seed_list[i] #
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

    # solution = solution.cpu().detach().numpy()
    test_times = test_times.cpu().detach().numpy()
    trajectory_x = trajectory_x.cpu().detach().numpy()


    plt.plot(test_times, trajectory_x)
    plt.title('n_events:{}'.format(n_events))

    plt.show()

table_data()
'''
results:
strength = 0.5, p = 30,h:V函数分母为1
0 tensor(6.2570) tensor(0.0028) 1559 tensor(0.0008)
1 tensor(6.2517) tensor(0.0054) 1620 tensor(0.0008)
2 tensor(6.2534) tensor(0.0044) 1039 tensor(0.0012)
3 tensor(6.2571) tensor(0.0025) 1274 tensor(0.0010)
4 tensor(6.2505) tensor(0.0066) 976 tensor(0.0013)
1293.6 tensor(0.0043) tensor(0.0010) tensor(2.2022) tensor(2.8070e-05)

strength = 0.5, p = 30,h:V函数分母为2
0 tensor(6.2531) tensor(0.0049) 10 tensor(0.5987)
1 tensor(6.2530) tensor(0.0046) 10 tensor(0.5976)
2 tensor(6.2525) tensor(0.0048) 10 tensor(0.5998)
3 tensor(6.2543) tensor(0.0033) 10 tensor(0.6210)
4 tensor(6.2525) tensor(0.0048) 10 tensor(0.5701)
10.0 tensor(0.0045) tensor(0.5974) tensor(0.0045) tensor(2.5802e-05)

strength = 0.5, p = 50,h:V函数分母为2
0 tensor(6.2543) tensor(0.0037) 12 tensor(0.4633)
1 tensor(6.2546) tensor(0.0032) 12 tensor(0.4641)
2 tensor(6.2488) tensor(0.0087) 10 tensor(0.4753)
3 tensor(6.2557) tensor(0.0024) 12 tensor(0.4735)
4 tensor(6.2555) tensor(0.0021) 10 tensor(0.3697)
11.2 tensor(0.0040) tensor(0.4492) tensor(0.0106) tensor(2.7112e-05)

[2, 4, 5, 6, 7]: 11.2 tensor(0.0041) tensor(0.4275) tensor(0.0201) tensor(2.9085e-05)

u coeff: 1.0
12.2 tensor(0.0053) tensor(0.2934) tensor(0.0533) tensor(4.8994e-05)

'''