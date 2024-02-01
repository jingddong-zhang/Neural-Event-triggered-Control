import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path as osp
from functions import *

setup_seed(10)


def methods(case):
    N = 5000  # sample size
    D_in = 2  # input dimension
    H1 = 20  # hidden dimension
    D_out = 1  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True).to(device)  # -5,5
    out_iters = 0


    while out_iters < 1:
        # break
        start = timeit.default_timer()
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case).to(device)
        x_0 = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).requires_grad_(True).to(device)

        i = 0
        max_iters = 550
        learning_rate = 0.05
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        L = []
        while i < max_iters:
            # break
            V = model._lya(data)
            Vx = torch.autograd.grad(V.sum(), data, create_graph=True)[0]

            f_u = model.untrigger_fn(1.0, data)
            L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
            if case == 'quad':
                loss = (L_V + V).relu().mean() + model._lya(x_0)
            if case == 'nlc':
                loss = (L_V).relu().mean() + (-V).relu().mean() + model._lya(x_0)**2

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

        test_times = torch.linspace(0, 20, 1000)
        s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
        init_s = torch.tensor([[0.0582738, 0.85801853]]) * 10.  # +torch.randn(1,2)*0.1
        func = model.GRN
        with torch.no_grad():
            original = odeint(func, init_s, test_times)
            solution = odeint(model.untrigger_fn, init_s, test_times)
        solution = solution.cpu().detach().numpy()
        original = original.cpu().detach().numpy()

        plt.subplot(121)
        plt.plot(test_times, solution[:,0,0],label='control')
        plt.plot(test_times, original[:, 0, 0],label='original')
        plt.legend()

        init_state = torch.cat((init_s[0], torch.zeros([2])))
        trajectory_x, trajectory_y, event_times, n_events, traj_events = model.simulate_t(init_state, test_times)
        trajectory_x = trajectory_x.cpu().detach().numpy()
        test_times = test_times.cpu().detach().numpy()


        plt.subplot(122)
        plt.plot(test_times,trajectory_x)
        plt.title('n_events:{}'.format(n_events))

        plt.show()

        out_iters+=1



def table_data(case):
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 2  # input dimension
    H1 = 20  # hidden dimension
    D_out = 1  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-1.0, 1.0).requires_grad_(True)
    target = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).to(device)

    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case).to(device)
    model._control.load_state_dict(torch.load(osp.join('./data/', case+'_control.pkl')))
    model._lya.load_state_dict(torch.load(osp.join('./data/', case+'_lya.pkl')))
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
            seed = seed_list[i]
            # seed = i
            np.random.seed(seed)
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
            init_noise = torch.cat((data[s][0, 0:2], torch.zeros([2]).to(device))).to(device)
            trajectory_x, trajectory_y, event_times, n_events, traj_events = model.simulate_t(init_state+init_noise, test_times)
            traj_events += -target
            event_times = torch.tensor(event_times)
            event_num.append(n_events)
            min_traj.append(torch.min(torch.sqrt((trajectory_x - target[0, 0]) ** 2 + (trajectory_y - target[0, 0]) ** 2)))
            cat_data = torch.cat((trajectory_x.unsqueeze(1), trajectory_y.unsqueeze(1)), dim=1)
            var_list.append(variance(cat_data, target, n=900))
            min_inter.append((event_times[1:] - event_times[:-1]).min())
            if len(traj_events) >= 11:
                min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            else:
                min_traj_events.append(torch.linalg.norm(traj_events[-1], ord=2))
            print(seed, trajectory_x[-1], min_traj[i], n_events,min_inter[i])

    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(),torch.tensor(var_list).mean())

# methods('nlc')
table_data('nlc')
'''
nlc
0 tensor(6.1560) tensor(0.1607) 26 tensor(4.3117e-08)
1 tensor(6.2090) tensor(0.0810) 27 tensor(6.2286e-08)
2 tensor(6.2241) tensor(0.0597) 21 tensor(5.2702e-08)
3 tensor(6.1874) tensor(0.1125) 24 tensor(4.3116e-08)
4 tensor(6.2197) tensor(0.0658) 21 tensor(5.2703e-08)
23.8 tensor(0.0960) tensor(5.0785e-08) tensor(0.2028) tensor(0.0177)

2 tensor(6.2241) tensor(0.0597) 21 tensor(5.2702e-08)
4 tensor(6.2197) tensor(0.0658) 21 tensor(5.2703e-08)
5 tensor(6.1957) tensor(0.1004) 23 tensor(3.3532e-08)
6 tensor(6.2212) tensor(0.0637) 21 tensor(5.2703e-08)
7 tensor(6.1712) tensor(0.1370) 27 tensor(5.2705e-08)
22.6 tensor(0.0853) tensor(4.8869e-08) tensor(0.1958) tensor(0.0138)


quad 
0 tensor(3.3113) tensor(0.0060) 2000 tensor(5.1925e-06)
1 tensor(1.9283) tensor(0.0053) 2000 tensor(5.6664e-06)
2 tensor(3.6957) tensor(0.0047) 2000 tensor(5.0654e-06)
3 tensor(3.5414) tensor(0.0019) 2000 tensor(5.1711e-06)
4 tensor(6.2652) tensor(0.0024) 1753 tensor(5.4412e-06)
1950.6 tensor(0.0041) tensor(5.3073e-06) tensor(2.1374) tensor(6.7737)

2 tensor(3.6957) tensor(0.0047) 2000 tensor(5.0654e-06)
4 tensor(6.2652) tensor(0.0024) 1753 tensor(5.4412e-06)
5 tensor(6.2561) tensor(0.0027) 1819 tensor(5.0114e-06)
6 tensor(4.1853) tensor(0.0086) 2000 tensor(5.0243e-06)
7 tensor(0.7686) tensor(0.0270) 2000 tensor(7.3916e-06)
1914.4 tensor(0.0091) tensor(5.5868e-06) tensor(2.2862) tensor(7.8634)
'''
