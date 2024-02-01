import matplotlib.pyplot as plt
import numpy as np
import torch

from functions import *

setup_seed(10)

'''
For learning 
'''
N = 1000             # sample size
D_in = 100           # input dimension
H1 = 2*D_in             # hidden dimension
D_out = 100           # output dimension
data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
lip_x = torch.linspace(0,5,1000).view(-1,1).requires_grad_(True).to(device)

out_iters = 0
ReLU = torch.nn.ReLU()
N1 = 500

target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])

while out_iters < 1:
    break
    start = timeit.default_timer()
    model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1]).to(device)
    i = 0
    t = 0
    max_iters = N1
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    softplus = torch.nn.Softplus()
    while i < max_iters:
        # break

        x = data[:, 0:D_in].requires_grad_(True)

        V = model._lya(x)
        Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

        f_u = model.untrigger_fn(1.0, x)
        L_V = (Vx * f_u).sum(dim=1).view(-1, 1)

        # target = target.repeat(len(x), 1)
        # x = x-target
        abs_x = torch.linalg.norm(x-target.repeat(len(x), 1),ord=2,dim=1).view(-1,1)
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

    test_times = torch.linspace(0, 30, 1000)
    s = torch.from_numpy(np.random.choice(np.arange(N,dtype=np.int64),1))
    init_s = torch.zeros([D_in]).view(-1,D_in)+torch.from_numpy(np.random.uniform(-0.5,0.5,[1,D_in]))
    func = model.Cell
    original = odeint(func,init_s,test_times)
    solution  = odeint(model.untrigger_fn,init_s,test_times)
    print(original.shape)

    init_state = torch.zeros([2*D_in])
    init_state[0:D_in] += torch.from_numpy(np.random.uniform(-0.5,0.5,[D_in]))
    trajectory,event_times,n_events,traj_events = model.simulate_t(init_state,test_times)



    solution = solution.cpu().detach().numpy()
    original = original.cpu().detach().numpy()
    trajectory = trajectory.cpu().detach().numpy()
    test_times = test_times.cpu().detach().numpy()

    plt.subplot(121)
    plt.plot(test_times, solution[:,0,0],label='control')
    plt.plot(test_times, original[:, 0, 0],label='original')
    plt.legend()

    plt.subplot(122)
    plt.plot(test_times,trajectory[:,0])
    plt.title('n_events:{}'.format(n_events))

    plt.show()

    out_iters+=1



def table_data():
    torch.manual_seed(369)
    D_in = 100  # input dimension
    H1 = 2 * D_in  # hidden dimension
    D_out = 100  # output dimension
    target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])

    model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1]).to(device)
    model._control.load_state_dict(torch.load('./data/NETC-high_control.pkl'))
    model._lya.load_state_dict(torch.load('./data/NETC-high_lya.pkl'))
    model._alpha.load_state_dict(torch.load('./data/NETC-high_alpha.pkl'))
    test_times = torch.linspace(0, 30, 1000).to(device)
    init_state = torch.zeros([2 * D_in])

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    # seed_list = [0,3,4,5,6]
    for i in range(5):
        with torch.no_grad():
            seed = i
            # seed = seed_list[i]
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

table_data()
'''
0 tensor(1.7137) 2 tensor(26.2032)
3 tensor(1.5388) 2 tensor(27.0504)
4 tensor(1.5219) 2 tensor(27.4198)
5 tensor(1.6561) 2 tensor(27.5166)
6 tensor(1.6232) 2 tensor(27.7055)
2.0 tensor(1.6107) tensor(27.1791) tensor(1.6641) tensor(2.7807)
'''
