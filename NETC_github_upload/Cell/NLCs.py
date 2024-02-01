import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path as osp
from functions import *

setup_seed(10)


def methods(case):
    N = 5000  # sample size
    D_in = 100  # input dimension
    H1 = 2*D_in  # hidden dimension
    D_out = 100  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True).to(device)  # -5,5

    out_iters = 0
    N1 = 500
    x_0 = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1]).requires_grad_(True).to(device)

    while out_iters < 1:
        # break
        start = timeit.default_timer()
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case).to(device)
        i = 0
        max_iters = N1
        learning_rate = 0.01
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
                loss = (L_V ).relu().mean() + (-V).relu().mean() + model._lya(x_0)**2

            L.append(loss)
            print(i, 'total loss=',loss.item(),'zero value=',model._lya(x_0).item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
        # print(q)
        torch.save(model._lya.state_dict(),osp.join('./data/', case+'_lya_5000.pkl'))
        torch.save(model._control.state_dict(),osp.join('./data/', case+'_control_5000.pkl'))
        stop = timeit.default_timer()


        print('\n')
        print("Total time: ", stop - start)

        test_times = torch.linspace(0, 30, 1000)
        s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
        init_s = torch.zeros([D_in]).view(-1, D_in) + torch.from_numpy(np.random.uniform(-0.5, 0.5, [1, D_in]))
        func = model.Cell
        with torch.no_grad():
            original = odeint(func, init_s, test_times)
            solution = odeint(model.untrigger_fn, init_s, test_times)
        solution = solution.cpu().detach().numpy()
        original = original.cpu().detach().numpy()

        plt.subplot(121)
        plt.plot(test_times, solution[:,0,0],label='control')
        plt.plot(test_times, original[:, 0, 0],label='original')
        plt.legend()

        init_state = torch.zeros([2 * D_in])
        init_state[0:D_in] += torch.from_numpy(np.random.uniform(-0.5, 0.5, [D_in]))
        trajectory, event_times, n_events, traj_events = model.simulate_t(init_state, test_times)
        trajectory = trajectory.cpu().detach().numpy()
        test_times = test_times.cpu().detach().numpy()

        plt.subplot(122)
        plt.plot(test_times,trajectory[:,0])
        plt.title('n_events:{}'.format(n_events))

        plt.show()

        out_iters+=1



def table_data(case):
    torch.manual_seed(369)
    N = 1000  # sample size
    D_in = 100  # input dimension
    H1 = 2 * D_in  # hidden dimension
    D_out = 100  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
    target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])


    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case).to(device)
    model._control.load_state_dict(torch.load(osp.join('./data/', case+'_control_5000.pkl')))
    model._lya.load_state_dict(torch.load(osp.join('./data/', case+'_lya_5000.pkl')))
    test_times = torch.linspace(0, 30, 1000).to(device)
    init_state = torch.zeros([2 * D_in])

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    # seed_list = [2, 4, 5, 6, 7]
    for i in range(5):
        with torch.no_grad():
            # seed = seed_list[i]
            seed = i
            np.random.seed(seed)
            init_state[0:D_in] += torch.from_numpy(np.random.uniform(-0.5, 0.5, [D_in]))
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
            print(seed, min_traj[i], n_events, min_inter[i])

    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
        torch.tensor(min_traj_events).mean(), torch.tensor(var_list).mean())


# methods('nlc')
table_data('nlc')
'''
nlc
1602.6 tensor(9.2402) tensor(4.6246e-08) tensor(97.1105)

quad
0 tensor(27.8603) 76 tensor(0.0012)
1 tensor(27.1152) 81 tensor(0.0009)
2 tensor(28.2477) 69 tensor(0.0017)
3 tensor(29.2699) 85 tensor(0.0010)
4 tensor(17.4654) 74 tensor(0.0012)
77.0 tensor(25.9917) tensor(0.0012) tensor(54.3880) tensor(1343.4596)

'''
