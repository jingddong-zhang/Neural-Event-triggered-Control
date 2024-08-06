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



def ablation_models():
    setup_seed(10)
    N = 2000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True).to(device)  # -5,5

    N1, N2 = 500, 100  # 500,30
    lambda_list = [0.005,0.05]
    Loss_data = {}
    start = timeit.default_timer()
    max_iters = N1 + N2
    for lambda_1 in lambda_list:
        for lr in [0.01,0.05,0.1]:
            # break
            L = np.zeros([5, max_iters])
            for seed in range(5):
                setup_seed(seed)
                model = Augment(D_in, H1, D_out, (D_in,), [H1, 1], 'icnn').to(device)
                i = 0
                max_iters = N1 + N2
                learning_rate = lr
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
                    L[seed,i] = loss.item()
                    print(lambda_1,lr,seed,i, 'total loss=', loss.item(), "Lyapunov Risk=", loss_stab.item(), 'Zeno_risk=', loss_event.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    i += 1
            Loss_data[f'lambda_1={lambda_1} lr={lr}'] = L
    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    np.save('./rebuttal/NETC_PI_Loss', Loss_data)

ablation_models()


