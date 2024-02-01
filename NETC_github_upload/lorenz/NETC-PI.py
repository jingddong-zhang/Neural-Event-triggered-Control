import matplotlib.pyplot as plt
import torch

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



'''
For learning 
'''
N = 2000            # sample size
D_in = 3            # input dimension
H1 = 64             # hidden dimension
D_out = 3           # output dimension
data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True).to(device) # -5,5


out_iters = 0
ReLU = torch.nn.ReLU()
N1,N2 = 500,100 # 500,30

while out_iters < 1:
    break
    start = timeit.default_timer()
    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],'icnn').to(device)
    i = 0
    t = 0
    max_iters = N1+N2
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    softplus = torch.nn.Softplus()
    while i < max_iters:
        # break
        x = data[:, 0:3].requires_grad_(True)

        V = model._lya(x)
        Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

        u = model._control(x)
        f_u = control_vector(data, u)
        L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
        abs_x = torch.linalg.norm(x, ord=2, dim=1).view(-1, 1)
        loss_stab = (L_V + V).relu().mean()

        loss_event = torch.tensor([0.0]).to(device)
        if i == N1:
            test_times = torch.linspace(0, 2, 1000).to(device)
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
            init_state = torch.cat((data[s][0, 0:3], torch.tensor([0., 0.,0.]).to(device))).to(device)
            stage1_solution,trajectory_y,trajectory_z,event_times, stage1_n_events,stage1_traj_events = model.simulate_t(init_state, test_times)
        if i>N1:
            batch_s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 10))
            batch_data = data[batch_s]
            event_t = model.get_collision_times(batch_data)
            loss_event = (1/event_t).mean()
            # event_t,solutions = model.get_collision_times(batch_data)
            # loss_event = (1/event_t).mean() + torch.sum(solutions**2,dim=1).mean()

        loss = loss_stab + 10*10*loss_event/N
        L.append(loss)
        print(i, 'total loss=',loss.item(),"Lyapunov Risk=",loss_stab.item(),'Zeno_risk=',loss_event.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        # stop = timeit.default_timer()
        # print('per:',stop-start)
        i += 1
    # print(q)
    torch.save(model._lya.state_dict(),'./data/NETC-low_lya.pkl')
    torch.save(model._control.state_dict(),'./data/NETC-low_control.pkl')
    stop = timeit.default_timer()


    print('\n')
    print("Total time: ", stop - start)
    # test_times = torch.linspace(0, 10, 1000)
    # s = torch.from_numpy(np.random.choice(np.arange(500,dtype=np.int64),1))
    func = Lorenz()
    original = odeint(func,data[s][:,0:3],test_times)
    solution = odeint(model.untrigger_fn,data[s][:,0:3],test_times)
    print(solution.shape)

    init_state = torch.cat((data[s][0,0:3],torch.zeros([3]).to(device))).to(device)
    trajectory_x,trajectory_y,trajectory_z,event_times,n_events,traj_events = model.simulate_t(init_state,test_times)
    # print(torch.cat(event_times).shape)

    solution = solution.cpu().detach().numpy()
    original = original.cpu().detach().numpy()
    stage1_solution = stage1_solution.cpu().detach().numpy()
    trajectory_x = trajectory_x.cpu().detach().numpy()
    test_times = test_times.cpu().detach().numpy()
    plt.subplot(131)
    plt.plot(test_times, solution[:,0,0],label='control')
    plt.plot(test_times, original[:, 0, 0],label='original')
    plt.legend()
    plt.subplot(132)
    plt.plot(test_times,stage1_solution)
    plt.title('n_events:{}'.format(stage1_n_events))
    plt.subplot(133)
    plt.plot(test_times,trajectory_x)
    plt.title('n_events:{}'.format(n_events))
    plt.show()

    out_iters+=1


def table_data():
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 3  # input dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)

    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1], 'icnn').to(device)
    model._control.load_state_dict(torch.load('./data/NETC-low_control.pkl'))
    test_times = torch.linspace(0, 2, 1000).to(device)

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    seed_list = [0, 1, 6, 7, 9]
    for i in range(5):
        with torch.no_grad():
            seed = seed_list[i]  # 4,6
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
            cat_data = torch.cat((trajectory_x.unsqueeze(1), trajectory_y.unsqueeze(1), trajectory_z.unsqueeze(1)),
                                 dim=1)
            var_list.append(variance(cat_data, torch.zeros_like(cat_data), n=900))
            print(seed, trajectory_x[-1], min_traj[i], n_events,min_inter[i])

    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(),torch.tensor(var_list).mean())

# table_data()
'''
20.2 tensor(0.0059) tensor(0.0169) tensor(0.1078) tensor(0.0045)

0 tensor(-0.0021) tensor(0.0036) 20 tensor(0.0314)
1 tensor(-0.0044) tensor(0.0081) 20 tensor(0.0049)
2 tensor(4.3123e-06) tensor(8.4230e-06) 276 tensor(7.3538e-05)
3 tensor(2.2139e-05) tensor(9.0918e-05) 204 tensor(0.0001)
4 tensor(-0.0417) tensor(0.0481) 29 tensor(0.0030)
5 tensor(-0.0054) tensor(0.0002) 36 tensor(0.0058)
6 tensor(-0.0016) tensor(0.0028) 17 tensor(0.0352)
7 tensor(-0.0010) tensor(0.0016) 16 tensor(0.0115)
8 tensor(-4.2466e-06) tensor(6.2111e-05) 66 tensor(0.0025)
9 tensor(-0.0939) tensor(0.0135) 28 tensor(0.0016)
'''


def ablation_data():
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 3  # input dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)


    event_num = np.zeros([9,5])
    inter_time = np.zeros([9,5])
    for j in range(9):
        strength = (j+1)/10
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1], 'icnn',strength).to(device)
        model._control.load_state_dict(torch.load('./data/NETC-low_control.pkl'))
        model._lya.load_state_dict(torch.load('./data/NETC-low_lya.pkl'))
        test_times = torch.linspace(0, 2, 1000).to(device)

        # event_num = []
        # min_traj = []
        # min_inter = []
        # min_traj_events = []
        # var_list = []
        seed_list = [0, 1, 6, 7, 9]
        for i in range(5):
            with torch.no_grad():
                seed = seed_list[i]  # 4,6
                np.random.seed(seed)
                s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
                init_state = torch.cat((data[s][0, 0:3], torch.zeros([3]).to(device))).to(device)
                # solution = odeint(model.untrigger_fn, data[s][:, 0:3], test_times)
                trajectory_x, trajectory_y, trajectory_z, event_times, n_events, traj_events = model.simulate_t(init_state,
                                                                                                                test_times)
                event_num[j,i] = n_events
                event_times = torch.tensor(event_times).detach().numpy()
                inter_time[j,i] = (event_times[1:] - event_times[:-1]).min()

                # event_num.append(n_events)
                # min_traj.append(torch.min(torch.sqrt(trajectory_x ** 2 + trajectory_y ** 2 + trajectory_z ** 2)))
                # min_inter.append((event_times[1:] - event_times[:-1]).min())
                # cat_data = torch.cat((trajectory_x.unsqueeze(1), trajectory_y.unsqueeze(1),trajectory_z.unsqueeze(1)), dim=1)
                # var_list.append(variance(cat_data, torch.zeros_like(cat_data), n=900))
                # if len(traj_events) >= 11:
                #     min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
                # else:
                #     min_traj_events.append(torch.linalg.norm(traj_events[-1], ord=2))
                print(j,i)
    np.save('./data/sigma_PI',{'num':event_num,'inter':inter_time})

# ablation_data()

def ablation_models():
    setup_seed(10)
    N = 2000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True).to(device)  # -5,5

    out_iters = 0
    ReLU = torch.nn.ReLU()
    N1, N2 = 500, 100  # 500,30
    lambda_list = [0.005,0.05,0.5]
    while out_iters < 2:
        # break
        lambda_1 = lambda_list[out_iters+1]
        start = timeit.default_timer()
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1], 'icnn').to(device)
        i = 0
        t = 0
        max_iters = N1 + N2
        learning_rate = 0.05
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        L = []
        while i < max_iters:
            # break
            x = data[:, 0:3].requires_grad_(True)

            V = model._lya(x)
            Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

            u = model._control(x)
            f_u = control_vector(data, u)
            L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
            abs_x = torch.linalg.norm(x, ord=2, dim=1).view(-1, 1)
            loss_stab = (L_V + V).relu().mean()

            loss_event = torch.tensor([0.0]).to(device)
            if i == N1:
                test_times = torch.linspace(0, 2, 1000).to(device)
                s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
                init_state = torch.cat((data[s][0, 0:3], torch.tensor([0., 0., 0.]).to(device))).to(device)
                stage1_solution, trajectory_y, trajectory_z, event_times, stage1_n_events, stage1_traj_events = model.simulate_t(
                    init_state, test_times)
            if i > N1:
                batch_s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 10))
                batch_data = data[batch_s]
                event_t = model.get_collision_times(batch_data)
                loss_event = (1 / event_t).mean()

            loss = loss_stab + lambda_1 * loss_event
            L.append(loss)
            print(out_iters,i, 'total loss=', loss.item(), "Lyapunov Risk=", loss_stab.item(), 'Zeno_risk=', loss_event.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
        torch.save(model._lya.state_dict(), './data/NETC-low_lya_{}.pkl'.format(lambda_1))
        torch.save(model._control.state_dict(), './data/NETC-low_control_{}.pkl'.format(lambda_1))
        stop = timeit.default_timer()

        print('\n')
        print("Total time: ", stop - start)

        out_iters += 1

# ablation_models()


def ablation_lambda_data():
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 3  # input dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)


    event_num = np.zeros([3,5])
    inter_time = np.zeros([3,5])
    mse = np.zeros([3,5])
    lambda_list = [0.005, 0.05, 0.5]
    for j in range(3):
        lambda_1 = lambda_list[j]
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1], 'icnn').to(device)
        model._control.load_state_dict(torch.load('./data/NETC-low_control_{}.pkl'.format(lambda_1)))
        model._lya.load_state_dict(torch.load('./data/NETC-low_lya_{}.pkl'.format(lambda_1)))
        test_times = torch.linspace(0, 2, 1000).to(device)

        seed_list = [0, 1, 6, 7, 9]
        for i in range(5):
            with torch.no_grad():
                seed = seed_list[i]  # 4,6
                np.random.seed(seed)
                s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
                init_state = torch.cat((data[s][0, 0:3], torch.zeros([3]).to(device))).to(device)

                trajectory_x, trajectory_y, trajectory_z, event_times, n_events, traj_events = model.simulate_t(init_state,
                                                                                                                test_times)
                event_num[j,i] = n_events
                event_times = torch.tensor(event_times).detach().numpy()
                inter_time[j,i] = (event_times[1:] - event_times[:-1]).min()
                traj = torch.sqrt(trajectory_x ** 2 + trajectory_y ** 2 + trajectory_z ** 2)
                mse[j,i] = np.mean(traj.detach().numpy()[-100:])

                print(j,i)
    np.save('./data/lambda_PI',{'num':event_num,'inter':inter_time,'mse':mse})

# ablation_lambda_data()


def mechanism():
    setup_seed(10)
    N = 2000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True).to(device)  # -5,5
    test_times = torch.linspace(0, 2, 1000).to(device)
    n = 10
    x,y,z = torch.linspace(-2.5,2.5,n),torch.linspace(-2.5,2.5,n),torch.linspace(-2.5,2.5,n)
    X,Y,Z = torch.meshgrid(x,y,z)
    grid = torch.stack([X, Y,Z], dim=3).view(-1,3).requires_grad_(True)
    out_iters = 0
    N1, N2 = 500, 100  # 500,30
    while out_iters < 1:
        # break
        lambda_1 = 0.5
        start = timeit.default_timer()
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1], 'icnn').to(device)
        i = 0
        t = 0
        max_iters = N1 + N2
        learning_rate = 0.05
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        L = []
        hessV_data = torch.zeros([10, n**3, 3, n**3, 3])
        gradu_data = torch.zeros([10, n**3, 3])
        times_list = []
        while i < max_iters:
            # break
            x = data[:, 0:3].requires_grad_(True)

            V = model._lya(x)
            Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

            u = model._control(x)
            f_u = control_vector(data, u)
            L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
            abs_x = torch.linalg.norm(x, ord=2, dim=1).view(-1, 1)
            loss_stab = (L_V + V).relu().mean()

            loss_event = torch.tensor([0.0]).to(device)
            if i == N1:
                test_times = torch.linspace(0, 2, 1000).to(device)
                s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
                init_state = torch.cat((data[s][0, 0:3], torch.tensor([0., 0., 0.]).to(device))).to(device)
                stage1_solution, trajectory_y, trajectory_z, event_times, stage1_n_events, stage1_traj_events = model.simulate_t(
                    init_state, test_times)
            if i >= N1:
                batch_s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 10))
                batch_data = data[batch_s]
                event_t = model.get_collision_times(batch_data)
                loss_event = (1 / event_t).mean()

            loss = loss_stab + lambda_1 * loss_event
            L.append(loss)
            print(out_iters,i, 'total loss=', loss.item(), "Lyapunov Risk=", loss_stab.item(), 'Zeno_risk=', loss_event.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i>=N1 and (i + 1) % 10 == 0:
                def V_dunc(x):
                    return model._lya(x).sum()

                hessV = torch.autograd.functional.hessian(V_dunc, grid)
                hessV_data[int((i + 1) / 10) - 51] = hessV
                u = model._control(grid)
                ux = torch.autograd.grad(u.sum(), grid, create_graph=True)[0]
                gradu_data[int((i + 1) / 10) - 51] = ux
                seed_list = [0, 1, 6, 7, 9]
                sub_times = np.zeros(5)
                for j in range(5):
                    with torch.no_grad():
                        seed = seed_list[j]  # 4,6
                        np.random.seed(seed)
                        s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
                        init_state = torch.cat((data[s][0, 0:3], torch.zeros([3]).to(device))).to(device)

                        trajectory_x, trajectory_y, trajectory_z, event_times, n_events, traj_events = model.simulate_t(
                            init_state,
                            test_times)
                    sub_times[j] = n_events
                times_list.append(sub_times.mean())

            i += 1

        stop = timeit.default_timer()

        print('\n')
        print("Total time: ", stop - start)

        np.save('./data/fig4/PI_times_{}'.format(n),times_list)
        np.save('./data/fig4/PI_ux_{}'.format(n),gradu_data.detach().numpy())
        np.save('./data/fig4/PI_Vxx_{}'.format(n),hessV_data.detach().numpy())
        out_iters += 1

mechanism()

