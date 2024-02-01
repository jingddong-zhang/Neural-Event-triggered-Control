import matplotlib.pyplot as plt
import torch

from functions import *

setup_seed(10)




'''
For learning 
'''
N = 1000             # sample size
D_in = 100           # input dimension
H1 = 64            # hidden dimension
D_out = 100           # output dimension
data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
lip_x = torch.linspace(0,10,1000).view(-1,1).requires_grad_(True).to(device)


out_iters = 0
ReLU = torch.nn.ReLU()
N1,N2 = 500,10 # 500,30

target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])

while out_iters < 1:
    break
    start = timeit.default_timer()
    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],'icnn').to(device)
    i = 0
    t = 0
    max_iters = N1+N2
    learning_rate = 0.01
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
        # loss_stab = (L_V + V).relu().mean()

        abs_x = torch.linalg.norm(x-target.repeat(len(x), 1),ord=2,dim=1).view(-1,1)
        loss_stab = (L_V + model._alpha(abs_x)).relu().mean()
        loss_lip =  (1/model._alpha.integrand(lip_x)).mean()
        loss_stab += 0.1*loss_lip

        loss_event = torch.tensor([0.0]).to(device)
        if i == N1:
            test_times = torch.linspace(0, 20, 1000).to(device)
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
            init_state = torch.zeros([2 * D_in])
            init_state[0:D_in] += +torch.from_numpy(np.random.uniform(-0.5, 0.5, [D_in]))
            stage1_solution,event_times,stage1_n_events,traj_events = model.simulate_t(init_state, test_times)
        if i>N1:
            batch_s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 5))
            batch_data = data[batch_s]
            event_t = model.get_collision_times(batch_data)
            loss_event = (1/event_t).mean()

        loss = loss_stab + 10*10*loss_event/N
        L.append(loss)
        print(i, 'total loss=',loss.item(),"Lyapunov Risk=",loss_stab.item(),'Zeno_risk=',loss_event.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += 1

    torch.save(model._lya.state_dict(),'./data/NETC-low_lya_10.pkl')
    torch.save(model._control.state_dict(),'./data/NETC-low_control_10.pkl')
    stop = timeit.default_timer()


    print('\n')
    print("Total time: ", stop - start)
    # test_times = torch.linspace(0, 10, 1000)
    # s = torch.from_numpy(np.random.choice(np.arange(500,dtype=np.int64),1))

    func = model.Cell
    init_s = (init_state[0:D_in]).view(1,-1)
    original = odeint(func,init_s,test_times)
    solution  = odeint(model.untrigger_fn,init_s,test_times)
    print(solution.shape)

    # init_state = torch.zeros([2*D_in])
    # init_state[0:D_in] += +torch.from_numpy(np.random.uniform(-0.5,0.5,[D_in]))
    trajectory,event_times,n_events,traj_events = model.simulate_t(init_state,test_times)

    solution = solution.cpu().detach().numpy()
    original = original.cpu().detach().numpy()
    stage1_solution = stage1_solution.cpu().detach().numpy()
    trajectory = trajectory.cpu().detach().numpy()
    test_times = test_times.cpu().detach().numpy()
    plt.subplot(131)
    plt.plot(test_times, solution[:,0,0],label='control')
    plt.plot(test_times, original[:, 0, 0],label='original')
    plt.legend()
    plt.subplot(132)
    plt.plot(test_times,stage1_solution[:,0])
    plt.title('n_events:{}'.format(stage1_n_events))
    plt.subplot(133)
    plt.plot(test_times,trajectory[:,0])
    plt.title('n_events:{}'.format(n_events))
    plt.show()

    out_iters+=1


def table_data():
    torch.manual_seed(369)
    D_in = 100  # input dimension
    H1 = 64  # hidden dimension
    D_out = 100  # output dimension
    target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])

    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],'icnn').to(device)
    model._control.load_state_dict(torch.load('./data/NETC-low_control_10.pkl'))
    model._lya.load_state_dict(torch.load('./data/NETC-low_lya_10.pkl'))
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
0 tensor(5.2361e-08) 11 tensor(0.9851)
1 tensor(5.3513e-08) 11 tensor(0.9523)
2 tensor(5.2714e-08) 11 tensor(0.9523)
3 tensor(5.1400e-08) 11 tensor(0.9030)
4 tensor(5.1812e-08) 11 tensor(0.9453)
5 tensor(5.0801e-08) 11 tensor(0.9325)
6 tensor(5.0584e-08) 11 tensor(0.9136)
7 tensor(5.0558e-08) 11 tensor(0.9172)
8 tensor(5.0878e-08) 11 tensor(0.9236)
9 tensor(5.1258e-08) 11 tensor(0.9432)
11.0 tensor(5.2360e-08) tensor(0.9476) tensor(5.3327e-08) tensor(2.9318e-15)

'''
