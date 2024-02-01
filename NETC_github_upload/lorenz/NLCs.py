import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path as osp
from functions import *

setup_seed(10)

'''
controlled vector field
'''
def control_vector(state,u):
    sigma = 10.
    rho = 28.
    beta = 8 / 3
    x,y,z = state[:,0:1],state[:,1:2],state[:,2:3]
    dx = sigma * (y - x)
    dy = rho * x - y - x * z
    dz = x * y - beta * z
    return torch.cat((dx,dy,dz),dim=1)+u

def methods(case):
    N = 5000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-5, 5).requires_grad_(True).to(device)  # -5,5

    out_iters = 0
    N1 = 600

    while out_iters < 1:
        # break
        start = timeit.default_timer()
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case).to(device)
        x_0 = torch.zeros([1,D_in]).requires_grad_(True).to(device)
        i = 0
        max_iters = N1
        learning_rate = 0.05
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        L = []
        while i < max_iters:
            # break
            V = model._lya(data)
            Vx = torch.autograd.grad(V.sum(), data, create_graph=True)[0]
            u = model._control(data)
            f_u = control_vector(data, u)
            L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
            if case == 'quad':
                loss = (L_V + V).relu().mean() + model._lya(x_0)
            if case == 'nlc':
                loss = (L_V ).relu().mean() + (-V).relu().mean() + model._lya(x_0).relu()

            L.append(loss)
            print(i, 'total loss=',loss.item(),'zero value=',model._lya(x_0).item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if loss < 0.5:
            #     break

            # stop = timeit.default_timer()
            # print('per:',stop-start)
            i += 1
        # print(q)
        torch.save(model._lya.state_dict(),osp.join('./data/', case+'_lya.pkl'))
        torch.save(model._control.state_dict(),osp.join('./data/', case+'_control.pkl'))
        stop = timeit.default_timer()


        print('\n')
        print("Total time: ", stop - start)

        test_times = torch.linspace(0, 2, 1000).to(device)
        s = torch.from_numpy(np.random.choice(np.arange(N,dtype=np.int64),1))
        s = torch.tensor([1])
        func = Lorenz()
        with torch.no_grad():
            original = odeint(func,data[s][:,0:3],test_times)
            solution = odeint(model.untrigger_fn,data[s][:,0:3],test_times)
        solution = solution.cpu().detach().numpy()
        original = original.cpu().detach().numpy()

        plt.subplot(121)
        plt.plot(test_times, solution[:,0,0],label='control')
        plt.plot(test_times, original[:, 0, 0],label='original')
        plt.legend()

        init_state = torch.cat((data[s][0, 0:3], torch.zeros([3]).to(device))).to(device)
        # torch.save(init_state,'./data/fig1/init_state.pt')
        # init_state = torch.load('./data/fig1/init_state.pt').to(device)
        trajectory_x, trajectory_y, trajectory_z, event_times, n_events,traj_events = model.simulate_t(init_state, test_times)
        trajectory_x = trajectory_x.cpu().detach().numpy()
        test_times = test_times.cpu().detach().numpy()
        # torch.save(event_times,'./data/fig1/nlc_event times_2000.pt')
        # np.save('./data/fig1/nlc_traj_2000',trajectory_x)
        # np.save('./data/fig1/nlc_control', control_value.detach().numpy())


        plt.subplot(122)
        plt.plot(test_times,trajectory_x)
        plt.title('n_events:{}'.format(n_events))

        plt.show()

        out_iters+=1



def table_data(case):
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)

    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case).to(device)
    model._control.load_state_dict(torch.load(osp.join('./data/', case+'_control.pkl')))
    test_times = torch.linspace(0, 2, 1000).to(device)

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    seed_list = [0, 1, 4, 6, 7]
    for i in range(5):
        with torch.no_grad():
            seed = seed_list[i]  # 4,6
            # seed = i
            np.random.seed(seed)
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
            init_state = torch.cat((data[s][0, 0:3], torch.zeros([3]).to(device))).to(device)
            # solution = odeint(model.untrigger_fn, data[s][:, 0:3], test_times)
            trajectory_x, trajectory_y, trajectory_z, event_times, n_events, traj_events = model.simulate_t(init_state,
                                                                                                            test_times)
            event_times = torch.tensor(event_times)
            event_num.append(n_events)
            min_traj.append(torch.min(torch.sqrt(trajectory_x ** 2 + trajectory_y ** 2 + trajectory_z ** 2)))
            min_inter.append((event_times[1:] - event_times[:-1]).min())
            min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            print(seed, trajectory_x[-1], min_traj[i], n_events,min_inter[i])

    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean())

methods('quad')
# table_data('quad')
'''
nlc
1602.6 tensor(9.2402) tensor(4.6246e-08) tensor(97.1105)

quad
0 tensor(0.3406) tensor(9.6902) 230 tensor(8.1478e-06)
1 tensor(0.0036) tensor(0.0643) 343 tensor(1.2254e-05)
2 tensor(0.0064) tensor(0.0442) 39 tensor(0.0066)
3 tensor(-0.0417) tensor(0.0531) 611 tensor(7.7246e-06)
4 tensor(-0.5439) tensor(0.0935) 226 tensor(4.3711e-05)
5 tensor(0.0044) tensor(0.0605) 294 tensor(1.6904e-05)
6 tensor(-0.4569) tensor(7.5051) 215 tensor(1.3665e-05)
7 tensor(-0.4863) tensor(5.8042) 196 tensor(3.6694e-05)
8 tensor(-0.0420) tensor(0.0634) 1284 tensor(3.4820e-06)
10 tensor(-0.0152) tensor(0.0409) 259 tensor(1.9946e-05)
242.0 tensor(4.6315) tensor(2.2894e-05) tensor(7.8166)

'''
