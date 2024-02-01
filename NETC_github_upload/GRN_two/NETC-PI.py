import matplotlib.pyplot as plt
import torch
from functions import *


setup_seed(10)

N = 1000             # sample size
D_in = 2            # input dimension
H1 = 20             # hidden dimension
D_out = 1           # output dimension
data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)


out_iters = 0

'''
可调参数： lr，rtol，V神经网络架构
'''
while out_iters < 1:
    # break
    start = timeit.default_timer()
    model = Augment(D_in,H1,D_out,(D_in,),[10,10,1])
    # model = Augment(D_in, H1, D_out, (D_in,), [20, 1])
    i = 0
    t = 0
    max_iters = 500+50
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []

    while i < max_iters:
        # break
        x = data[:, 0:2].requires_grad_(True)
        V = model._lya(x)
        Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]


        f_u = model.untrigger_fn(1.,x)
        L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
        loss1 = (L_V+1.0*V ).relu().mean()

        loss2 = torch.tensor([0.0])
        if i == 500:
            test_times = torch.linspace(0, 20, 1000)
            # s = torch.from_numpy(np.random.choice(np.arange(500, dtype=np.int64), 1))
            init_s = torch.tensor([[0.0582738, 0.85801853]])*10.# + torch.randn(1, 2) * 0.5
            init_state = torch.cat((init_s[0], torch.tensor([0., 0.])))
            stage1_x, stage1_y, event_times, stage1_n_events,traj_events = model.simulate_t(init_state, test_times)
        if i>500:
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 10))
            event_t = model.get_collision_times(data[s])
            loss2 = (1/event_t).mean()

        loss = loss1+loss2*10/N
        L.append(loss)
        print(i, 'total loss=',loss.item(),"Lyapunov Risk=",loss1.item(),'Zeno_risk=',loss2.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        i += 1
    stop = timeit.default_timer()


    print('\n')
    print("Total time: ", stop - start)
    torch.save(model._lya.state_dict(), './data/NETC-low_lya.pkl')
    torch.save(model._control.state_dict(),'./data/NETC-low_control.pkl')

    test_times = torch.linspace(0, 20, 1000)
    s = torch.from_numpy(np.random.choice(np.arange(N,dtype=np.int64),1))
    init_s = torch.tensor([[0.0582738 , 0.85801853]])*10.#+torch.randn(1,2)*0.1
    func = model.GRN
    original = odeint(func,init_s,test_times)
    solution  = odeint(model.untrigger_fn,init_s,test_times)
    print(solution.shape)
    test_data = model.untrigger_fn(1.,solution[:,0,:])
    test_control = model._control(solution[:,0,:])
    test_V = model._lya(solution[:,0,:])
    plt.subplot(141)
    plt.plot(test_times.numpy(),test_V.detach().numpy()[:,0])
    # plt.plot(test_times.numpy(), test_data.detach().numpy()[:, 0])
    # plt.yticks([0,0.3])
    # init_state = torch.cat((data[s][0,0:2],torch.tensor([0.,0.])))
    trajectory,velocity,event_times,n_events,traj_events = model.simulate_t(init_state,test_times)
    # print(torch.cat(event_times).shape)
    plt.subplot(142)
    plt.plot(test_times.numpy(), solution.detach().numpy()[:,0,0],label='control',color='r')
    plt.plot(test_times.numpy(), solution.detach().numpy()[:, 0, 1], label='control',color='r')
    plt.plot(test_times.numpy(), original.detach().numpy()[:, 0, 0],label='original',color='g')
    plt.plot(test_times.numpy(), original.detach().numpy()[:, 0, 1], label='original',color='g')
    plt.legend()
    plt.subplot(143)
    plt.plot(test_times.numpy(),stage1_x.detach().numpy(),c='r')
    plt.plot(test_times.numpy(), stage1_y.detach().numpy(),c='r')
    plt.title('n_events:{}'.format(stage1_n_events))
    plt.subplot(144)
    plt.plot(test_times.numpy(),trajectory.detach().numpy())
    plt.plot(test_times.numpy(), velocity.detach().numpy())
    plt.title('n_events:{}'.format(n_events))
    plt.show()

    out_iters+=1



def table_data():
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 2  # input dimension
    H1 = 20  # hidden dimension
    D_out = 1  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-1, 1).requires_grad_(True)
    target = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).to(device)

    model = Augment(D_in, H1, D_out, (D_in,), [10, 10, 1])
    model._control.load_state_dict(torch.load('./data/NETC-low_control.pkl'))
    model._lya.load_state_dict(torch.load('./data/NETC-low_lya.pkl'))
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
            seed = seed_list[i]  # 4,6
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
19.6 tensor(0.0160) tensor(0.1289) tensor(0.0535) tensor(0.0005)
'''
