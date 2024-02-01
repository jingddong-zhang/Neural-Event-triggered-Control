import matplotlib.pyplot as plt
import numpy as np
import torch

from functions import *

setup_seed(10)




'''
For learning 
'''
N = 1000             # sample size
D_in = 2            # input dimension
H1 = 20             # hidden dimension
D_out = 1           # output dimension
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

        x = data[:, 0:2].requires_grad_(True)

        V = model._lya(x)
        Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

        f_u = model.untrigger_fn(1.0, x)
        L_V = (Vx * f_u).sum(dim=1).view(-1, 1)


        abs_x = torch.linalg.norm(x,ord=2,dim=1).view(-1,1)
        loss_stab = (L_V + model._alpha(abs_x)).relu().mean()
        loss_lip =  (1/model._alpha.integrand(lip_x)).mean()
        loss = loss_stab + 0.1*loss_lip

        L.append(loss)
        print(i, 'total loss=',loss.item(),"Lyapunov Risk=",loss_stab.item(),'Lip loss=', loss_lip.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += 1
    torch.save(model._lya.state_dict(),'./data/NETC-high_lya.pkl')
    torch.save(model._control.state_dict(),'./data/NETC-high_control.pkl')
    torch.save(model._alpha.state_dict(), './data/NETC-high_alpha.pkl')
    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)

    test_times = torch.linspace(0, 20, 1000)
    s = torch.from_numpy(np.random.choice(np.arange(N,dtype=np.int64),1))
    init_s = torch.tensor([[0.0582738 , 0.85801853]])*10.#+torch.randn(1,2)*0.1
    func = model.GRN
    original = odeint(func,init_s,test_times)
    solution  = odeint(model.untrigger_fn,init_s,test_times)
    print(solution.shape)

    init_state = torch.cat((init_s[0], torch.zeros([2])))
    trajectory_x,trajectory_y,event_times,n_events,traj_events = model.simulate_t(init_state,test_times)

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
N=5000， [-10,10], 550, factor=0.1, H1 = 20, lip_s = linspace(0,10,1000) : 12
'''


def table_data():
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 2  # input dimension
    H1 = 20  # hidden dimension
    D_out = 1  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-1, 1).requires_grad_(True)
    target = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).to(device)

    model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1]).to(device)
    model._control.load_state_dict(torch.load('./data/NETC-high_control.pkl'))
    model._lya.load_state_dict(torch.load('./data/NETC-high_lya.pkl'))
    model._alpha.load_state_dict(torch.load('./data/NETC-high_alpha.pkl'))
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
            # seed = i
            seed = seed_list[i]
            np.random.seed(seed)
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
            init_noise = torch.cat((data[s][0, 0:2], torch.zeros([2]).to(device))).to(device)
            trajectory_x, trajectory_y, event_times, n_events, traj_events = model.simulate_t(init_state+init_noise, test_times)
            traj_events += -target
            event_times = torch.tensor(event_times)
            event_num.append(n_events)
            min_traj.append(torch.min(torch.sqrt((trajectory_x-target[0,0]) ** 2 + (trajectory_y-target[0,0]) ** 2)))
            cat_data = torch.cat((trajectory_x.unsqueeze(1),trajectory_y.unsqueeze(1)),dim=1)
            var_list.append(variance(cat_data,target,n=900))
            min_inter.append((event_times[1:] - event_times[:-1]).min())
            if len(traj_events) >= 11:
                min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            else:
                min_traj_events.append(torch.linalg.norm(traj_events[-1], ord=2))
            print(seed, trajectory_x[-1], min_traj[i], n_events,min_inter[i])

    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(),torch.tensor(var_list).mean())

table_data()
'''
seed: 1,2,3,4,5: 8.2 tensor(0.0314) tensor(7.8305) tensor(0.0707) tensor(0.0052)
seed: 2,4,5,6,7: 4.0 tensor(0.0584) tensor(15.5189) tensor(0.0748) tensor(0.0069)
0 tensor(6.2643) tensor(0.0064) 12 tensor(0.2427)
1 tensor(6.2662) tensor(0.0046) 16 tensor(0.2401)
2 tensor(6.3293) tensor(0.0676) 2 tensor(19.4478)
3 tensor(6.3770) tensor(0.0220) 9 tensor(0.0063)
4 tensor(6.3171) tensor(0.0562) 2 tensor(19.2154)
5 tensor(6.3665) tensor(0.1022) 2 tensor(19.3303)
6 tensor(6.3196) tensor(0.0585) 2 tensor(19.3610)
7 tensor(6.3479) tensor(0.0073) 12 tensor(0.2399)
8 tensor(6.2674) tensor(0.0202) 12 tensor(0.2444)
9 tensor(6.3538) tensor(0.0056) 18 tensor(0.2393)
'''
