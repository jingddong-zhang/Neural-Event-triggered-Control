import matplotlib.pyplot as plt
import numpy as np
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
N = 5000             # sample size
D_in = 3            # input dimension
H1 = 64             # hidden dimension
D_out = 3           # output dimension
data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
lip_x = torch.linspace(0,10,1000).view(-1,1).requires_grad_(True).to(device)

out_iters = 0
ReLU = torch.nn.ReLU()
N1 = 500
N2 = 50

while out_iters < 1:
    break
    start = timeit.default_timer()
    model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1]).to(device)
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


        abs_x = torch.linalg.norm(x,ord=2,dim=1).view(-1,1)
        loss_stab = (L_V + model._alpha(abs_x)).relu().mean()
        loss_lip =  (1/model._alpha.integrand(lip_x)).mean()#1/model._alpha.integrand.c**2 # torch.tensor([0.0]).to(device)
        loss_event = torch.tensor([0.0]).to(device)
        # if i > N1:
        #     batch_s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 10))
        #     batch_data = data[batch_s]
        #     event_t = model.get_collision_times(batch_data)
        #     loss_event = (1/event_t).mean()

        loss = loss_stab + 0.1*loss_lip #+ 10*loss_event/N

        L.append(loss)
        print(i, 'total loss=',loss.item(),"Lyapunov Risk=",loss_stab.item(),'Lip loss=', loss_lip.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += 1
    # torch.save(model._lya.state_dict(),'./data/NETC-high_lya.pkl')
    # torch.save(model._control.state_dict(),'./data/NETC-high_control.pkl')
    # torch.save(model._alpha.state_dict(), './data/NETC-high_alpha.pkl')
    stop = timeit.default_timer()
    # model._control.load_state_dict(torch.load('./data/NETC-high_control.pkl'))

    print('\n')
    print("Total time: ", stop - start)
    test_times = torch.linspace(0, 2, 1000).to(device)
    s = torch.from_numpy(np.random.choice(np.arange(N,dtype=np.int64),1))
    func = Lorenz()
    original = odeint(func,data[s][:,0:3],test_times)
    solution = odeint(model.untrigger_fn,data[s][:,0:3],test_times)
    print(solution.shape)

    init_state = torch.cat((data[s][0,0:3],torch.zeros([3]).to(device))).to(device)
    # init_state = torch.load('./data/fig1/init_state.pt').to(device) # init in fig1
    trajectory_x,trajectory_y,trajectory_z,event_times,n_events,traj_events,control_value = model.simulate_t(init_state,test_times)
    # print(torch.cat(event_times).shape)
    # print(control_value.shape)

    solution = solution.cpu().detach().numpy()
    original = original.cpu().detach().numpy()
    trajectory_x = trajectory_x.cpu().detach().numpy()
    test_times = test_times.cpu().detach().numpy()
    '''
    fig1
    '''
    # torch.save(event_times, './data/netc_event times.pt')
    # np.save('./data/fig1/netc_traj',trajectory_x)
    # np.save('./data/fig1/netc_control',control_value.detach().numpy())
    plt.subplot(121)
    plt.plot(test_times, solution[:,0,0],label='control')
    plt.plot(test_times, original[:, 0, 0],label='original')
    plt.legend()

    plt.subplot(122)
    plt.plot(test_times,trajectory_x)
    plt.title('n_events:{}'.format(n_events))
    plt.show()

    out_iters+=1

'''
成功参数： 
N=5000， [-5,5], 500, factor=0.1, H1 = 64, lip_s = linspace(0,10,1000)
N=5000， [-10,10], 500, factor=0.1, H1 = 64, lip_s = linspace(0,10,1000) : 19
'''


def table_data():
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 3  # input dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)

    model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1]).to(device)
    model._control.load_state_dict(torch.load('./data/NETC-high_control.pkl'))
    model._lya.load_state_dict(torch.load('./data/NETC-high_lya.pkl'))
    model._alpha.load_state_dict(torch.load('./data/NETC-high_alpha.pkl'))
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
            cat_data = torch.cat((trajectory_x.unsqueeze(1), trajectory_y.unsqueeze(1),trajectory_z.unsqueeze(1)), dim=1)
            var_list.append(variance(cat_data, torch.zeros_like(cat_data), n=900))
            if len(traj_events) >= 11:
                min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            else:
                min_traj_events.append(torch.linalg.norm(traj_events[-1], ord=2))
            print(seed, trajectory_x[-1], min_traj[i], n_events,min_inter[i])

    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(),torch.tensor(var_list).mean())

# table_data()
'''
12.2 tensor(0.0025) tensor(0.0373) tensor(0.1818)
11.4 tensor(0.0045) tensor(0.0578) tensor(0.1493)
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
        model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1],strength).to(device)
        model._control.load_state_dict(torch.load('./data/NETC-high_control.pkl'))
        model._lya.load_state_dict(torch.load('./data/NETC-high_lya.pkl'))
        model._alpha.load_state_dict(torch.load('./data/NETC-high_alpha.pkl'))
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
    np.save('./data/sigma_MC',{'num':event_num,'inter':inter_time})

# ablation_data()

def ablation_models():
    setup_seed(10)
    N = 5000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
    lip_x = torch.linspace(0, 10, 1000).view(-1, 1).requires_grad_(True).to(device)

    out_iters = 0
    N1 = 500
    N2 = 50
    lambda_list = [0.001,0.01,0.1,1.0]
    while out_iters < 4:
        # break
        lambda_1 = lambda_list[out_iters]
        start = timeit.default_timer()
        model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1]).to(device)
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
            loss_stab = (L_V + model._alpha(abs_x)).relu().mean()
            loss_lip = (1 / model._alpha.integrand(
                lip_x)).mean()  # 1/model._alpha.integrand.c**2 # torch.tensor([0.0]).to(device)

            loss = loss_stab + lambda_1 * loss_lip

            L.append(loss)
            print(i, 'total loss=', loss.item(), "Lyapunov Risk=", loss_stab.item(), 'Lip loss=', loss_lip.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
        torch.save(model._lya.state_dict(),'./data/NETC-high_lya_{}.pkl'.format(lambda_1))
        torch.save(model._control.state_dict(),'./data/NETC-high_control_{}.pkl'.format(lambda_1))
        torch.save(model._alpha.state_dict(), './data/NETC-high_alpha_{}.pkl'.format(lambda_1))
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


    event_num = np.zeros([4,5])
    inter_time = np.zeros([4,5])
    mse = np.zeros([4,5])
    lambda_list = [0.001, 0.01, 0.1, 1.0]
    for j in range(4):
        lambda_1 = lambda_list[j]
        model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1]).to(device)
        model._control.load_state_dict(torch.load('./data/NETC-high_control_{}.pkl'.format(lambda_1)))
        model._lya.load_state_dict(torch.load('./data/NETC-high_lya_{}.pkl'.format(lambda_1)))
        model._alpha.load_state_dict(torch.load('./data/NETC-high_alpha_{}.pkl'.format(lambda_1)))
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
    np.save('./data/lambda_MC',{'num':event_num,'inter':inter_time,'mse':mse})

# ablation_lambda_data()


def mechanism():
    setup_seed(10)
    N = 5000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
    lip_x = torch.linspace(0, 10, 1000).view(-1, 1).requires_grad_(True).to(device)
    test_times = torch.linspace(0, 2, 1000).to(device)
    n = 10
    x,y,z = torch.linspace(-2.5,2.5,n),torch.linspace(-2.5,2.5,n),torch.linspace(-2.5,2.5,n)
    X,Y,Z = torch.meshgrid(x,y,z)
    grid = torch.stack([X, Y,Z], dim=3).view(-1,3).requires_grad_(True)
    out_iters = 0
    N1 = 500
    N2 = 100
    while out_iters < 1:
        # break
        lambda_1 = 0.1
        start = timeit.default_timer()
        model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1]).to(device)
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
            loss_stab = (L_V + model._alpha(abs_x)).relu().mean()
            loss_lip = (1 / model._alpha.integrand(
                lip_x)).mean()  # 1/model._alpha.integrand.c**2 # torch.tensor([0.0]).to(device)

            loss = loss_stab + lambda_1 * loss_lip

            L.append(loss)
            print(i, 'total loss=', loss.item(), "Lyapunov Risk=", loss_stab.item(), 'Lip loss=', loss_lip.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 60 == 0:
                def V_dunc(x):
                    return model._lya(x).sum()
                hessV = torch.autograd.functional.hessian(V_dunc, grid)
                hessV_data[int((i+1)/ 60)-1] = hessV
                # lip_list.append(model._control.lip())
                u = model._control(grid)
                ux = torch.autograd.grad(u.sum(), grid, create_graph=True)[0]
                gradu_data[int((i+1)/ 60)-1] = ux
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

        np.save('./data/fig4/MC_times_{}'.format(n),times_list)
        np.save('./data/fig4/MC_ux_{}'.format(n),gradu_data.detach().numpy())
        np.save('./data/fig4/MC_Vxx_{}'.format(n),hessV_data.detach().numpy())



        out_iters += 1


mechanism()
#
# def convexity():
#     V_model = ICNN((3,), [18, 18, 1], 0.1, 1e-3)  # D_in = 3 , H1 = 3*D_in
#     # V_model2 = ICNN((3,), [18, 18, 1], 0.1, 1e-3)  # D_in = 3 , H1 = 3*D_in
#
#     torch.manual_seed(369)
#     n = 20
#     x,y,z = torch.linspace(-2.5,2.5,n),torch.linspace(-2.5,2.5,n),torch.linspace(-2.5,2.5,n)
#     X,Y,Z = torch.meshgrid(x,y,z)
#     grid = torch.stack([X, Y,Z], dim=3).view(-1,3).requires_grad_(True)
#     # data = torch.Tensor(n,3).uniform_(-2.5,2.5).requires_grad_(True)
#     eigens = torch.zeros([16,len(data)])
#     for k in range(15):
#         V_model.load_state_dict(torch.load('./data/V_linear_{}_100.pkl'.format(k + 1)))
#         for i in range(len(data)):
#             x = data[i:i+1,:]
#             hessV = hessian(V_model(x).sum(),x)
#             # eigens[k,i] =  torch.real(torch.linalg.eigvals(hessV)).max()
#             eigens[k, i] = torch.real(torch.linalg.eigvals(hessV).mean())
#         print(k)
#     V_model.load_state_dict(torch.load('./data/V_AI_2.5.pkl'))
#     for i in range(len(data)):
#         x = data[i:i + 1, :]
#         hessV = hessian(V_model(x).sum(), x)
#         # eigens[-1, i] = torch.real(torch.linalg.eigvals(hessV)).max()
#         eigens[-1, i] = torch.real(torch.linalg.eigvals(hessV).mean())
#     torch.save(eigens,'./data/eigvalstrace_{}_v1.pt'.format(n))