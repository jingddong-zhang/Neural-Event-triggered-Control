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

def ablation_models():
    setup_seed(10)
    N = 5000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension 64
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
    lip_x = torch.linspace(0, 10, 1000).view(-1, 1).requires_grad_(True).to(device)

    out_iters = 0
    N1 = 500
    N2 = 50
    Loss_data = {}
    lambda_list = [0.1,1.0]
    start = timeit.default_timer()
    for lambda_1 in lambda_list:
        for lr in [0.01,0.05,0.1]:
            # break
            max_iters = N1 + N2
            L = np.zeros([5, max_iters])
            for seed in range(5):
                setup_seed(seed)
                model = NETC_high(D_in, H1, D_out, (D_in,), [H1, 1]).to(device)
                i = 0

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
                    loss_stab = (L_V + model._alpha(abs_x)).relu().mean()
                    loss_lip = (1 / model._alpha.integrand(
                        lip_x)).mean()  # 1/model._alpha.integrand.c**2 # torch.tensor([0.0]).to(device)

                    loss = loss_stab + lambda_1 * loss_lip

                    # L.append(loss.item())
                    L[seed,i] = loss.item()
                    print(i, 'total loss=', loss.item(), "Lyapunov Risk=", loss_stab.item(), 'Lip loss=', loss_lip.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    i += 1

            Loss_data[f'lambda_1={lambda_1} lr={lr}'] = L
    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    np.save('./rebuttal/NETC_MC_Loss',Loss_data)
    # return Loss_data

ablation_models()


