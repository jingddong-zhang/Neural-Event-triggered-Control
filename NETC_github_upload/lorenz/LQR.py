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
        self.sigma = 10.
        self.rho = 28.
        self.beta = 8 / 3
        self.t0 = nn.Parameter(torch.tensor([0.0])).to(device)
        self.odeint = odeint_adjoint
        self.init_x_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.init_y_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.init_z_err = nn.Parameter(torch.tensor([0.0])).to(device)
        self.strength = strength
        self.S = S.to(device)
        self.Q = Q.to(device)
        self.K = K.to(device)

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
        return torch.mm(self.K,data.T).T

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
        u = self.lqr(input)
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
        # u = torch.mm(self.K,state.T).T
        u = self.lqr(state)
        dx[:,0] = self.sigma*(y-x)
        dx[:,1] = self.rho*x-y-x*z
        dx[:,2] = x*y-self.beta*z
        return dx+u


    def event_fn(self, t, state):
        # positive before trigger time
        x,y,z,e_x,e_y,e_z = state
        s = torch.cat((x, y,z)).view(-1, 3).to(device)
        e = torch.cat((e_x, e_y,e_z)).view(-1, 3).to(device)

        g = (self.strength-1.0)*torch.sum(s*torch.mm(self.Q,s.T).T)+2*torch.sum(x*torch.mm(self.S,torch.mm(self.K,e.T)).T)

        # Vx = self.dlya(s).to(device)
        # g = (Vx * (self.lqr(s + e) - self.lqr(s))).sum() - self.strength * self.lie_derivative(s).sum()

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
        )

# seed =  # 4,6
torch.manual_seed(369)

N = 5000             # sample size
D_in = 3            # input dimension
D_out = 3           # output dimension
data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)


import scipy.io as scio
lqr_data = scio.loadmat('./data/lqr_data.mat')
S = torch.from_numpy(np.array(lqr_data['S1'])).to(device)
Q = torch.from_numpy(np.array(lqr_data['Q'])).to(device)
K = torch.from_numpy(np.array(lqr_data['K1'])).to(device)
model = LQR_event(S,Q,K,0.99)


test_times = torch.linspace(0, 2, 1000).to(device)

# init_state = torch.load('./data/fig1/init_state.pt').to(device)
# for i in range(30):
event_num = []
min_traj = []
min_inter = []
min_traj_events = []
seed_list = [0,4,6,20,25]
for i in range(5):
    with torch.no_grad():
        seed =  seed_list[i] # 4,6
        np.random.seed(seed)
        s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
        init_state = torch.cat((data[s][0, 0:3], torch.zeros([3]).to(device))).to(device)
        # solution = odeint(model.untrigger_fn, data[s][:, 0:3], test_times)
        trajectory_x, trajectory_y, trajectory_z, event_times, n_events, traj_events = model.simulate_t(init_state, test_times)
        event_times = torch.tensor(event_times)
        event_num.append(n_events)
        min_traj.append(torch.min(torch.sqrt(trajectory_x**2+trajectory_y**2+trajectory_z**2)))
        min_inter.append((event_times[1:]-event_times[:-1]).min())
        min_traj_events.append(torch.linalg.norm(traj_events[10],ord=2))
        print(seed, trajectory_x[-1], min_traj[i], n_events)

print(np.array(event_num).mean(),torch.tensor(min_traj).mean(),torch.tensor(min_inter).mean(),torch.tensor(min_traj_events).mean())
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

'''
results:
strength = 0.99, R = 0.1
0 tensor(-0.3338) 2000
4 tensor(0.0361) 2000
6 tensor(0.1975) 2000
20 tensor(0.0160) 2000
22 tensor(-0.4249) 2000
25 tensor(0.3366) 2000

strength = 0.5, R = 0.1
5 tensor(-0.0698) 5
12 tensor(0.3783) 4
16 tensor(0.2391) 11

strength = 0.99, R = 0.05
3 tensor(0.2598) 2000
5 tensor(-0.0138) 2000
8 tensor(-0.0322) 2000
20 tensor(0.4322) 2000
24 tensor(-0.0057) 2000

strength = 0.5, R = 0.05
1 tensor(-0.8495) 6
4 tensor(0.8083) 2
16 tensor(-0.5559) 6
23 tensor(0.8076) 3
25 tensor(-0.9150) 2
26 tensor(0.8774) 2
27 tensor(0.2929) 2

'''