import matplotlib.pyplot as plt
import torch

from functions import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True

setup_seed(10)

'''
controlled vector field
'''
def control_vector(state,u):
    G = 9.81
    L = 0.5
    m = 0.15
    b = 0.1
    x,y = state[:,0:1],state[:,1:2]
    dx = y
    dy = G * torch.sin(x) / L + (-b * y) / (m * L ** 2)
    return torch.cat((dx,dy),dim=1)+u



'''
For learning 
'''
N = 2000             # sample size
D_in = 2            # input dimension
H1 = 20             # hidden dimension
D_out = 2           # output dimension
data = torch.Tensor(N, D_in).uniform_(-5, 5).requires_grad_(True)


eps = 0.001 # L2 regularization coef
kappa = 0.1 # 指数稳定性系数
out_iters = 0
ReLU = torch.nn.ReLU()
# valid = False

# model = ParaFunction(D_in,H1,D_out)
# V, gamma, u = model(data)
# f_u = control_vector(data, u)
# x = data[:,0:2].requires_grad_(True)
# V = model._lya(x)
# Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
# LV = Vx.sum(axis=1)
# ll = LV.relu().mean()
# lll = ll.mean()
# gamma = model._gamma(data[:,2:4])
# print(ll.shape,lll.shape)

while out_iters < 1:
    # break
    start = timeit.default_timer()
    # model = Augment(D_in,H1,D_out,(D_in,),[10,10,1])
    model = Augment(D_in, H1, D_out, (D_in,), [20, 1],'icnn')
    i = 0
    t = 0
    max_iters = 600
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []

    while i < max_iters:
        # break
        # start = timeit.default_timer()
        x,e = data[:, 0:2].requires_grad_(True),data[:, 2:4]
        # e = torch.zeros_like(x)
        V = model._lya(x)
        # print('-------------',V.shape)
        # gamma = model._gamma(e)
        # gamma = e.pow(2).sum(dim=1)
        # u = model._control(x+e)
        # f_u = control_vector(data,u)
        Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        # L_V = (Vx*f_u).sum(dim=1).view(-1,1)
        # loss1 = (L_V+V-gamma).relu().mean()

        e = torch.zeros_like(x)
        u = model._control(x)
        f_u = control_vector(data, u)
        L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
        loss2 = (L_V + V).relu().mean()

        loss3 = torch.tensor([0.0])
        if i == 500:
            test_times = torch.linspace(0, 10, 1000)
            s = torch.from_numpy(np.random.choice(np.arange(500, dtype=np.int64), 1))
            init_state = torch.cat((data[s][0, 0:2], torch.tensor([0., 0.])))
            stage1_solution, velocity, event_times, stage1_n_events = model.simulate_t(init_state, test_times)
        if i>500:
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 30))
            event_t = model.get_collision_times(data[s])
            loss3 = (1/event_t).mean()

        # loss = loss1+loss2+loss3
        loss = loss2+loss3
        L.append(loss)
        print(i, 'total loss=',loss.item(),"Lyapunov Risk=",loss2.item(),'Zeno_risk=',loss3.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if loss < 0.5:
        #     break

        # stop = timeit.default_timer()
        # print('per:',stop-start)
        i += 1
    # print(q)
    # torch.save(model._icnn.state_dict(),'./data/V_0.1.pkl')
    # torch.save(model._control.state_dict(),'./data/control_0.1.pkl')
    # torch.save(model._mono.state_dict(), './data/class_K_0.1.pkl')
    stop = timeit.default_timer()


    print('\n')
    print("Total time: ", stop - start)
    # test_times = torch.linspace(0, 10, 1000)
    # s = torch.from_numpy(np.random.choice(np.arange(500,dtype=np.int64),1))
    func = Invert()
    original = odeint(func,data[s][:,0:2],test_times)
    solution  = odeint(model.untrigger_fn,data[s][:,0:2],test_times)
    print(solution.shape)

    init_state = torch.cat((data[s][0,0:2],torch.tensor([0.,0.])))
    trajectory,velocity,event_times,n_events = model.simulate_t(init_state,test_times)
    # print(torch.cat(event_times).shape)
    plt.subplot(131)
    plt.plot(test_times.numpy(), solution.detach().numpy()[:,0,0],label='control')
    plt.plot(test_times.numpy(), original.detach().numpy()[:, 0, 0],label='original')
    plt.legend()
    plt.subplot(132)
    plt.plot(test_times.numpy(),stage1_solution.detach().numpy())
    plt.title('n_events:{}'.format(stage1_n_events))
    plt.subplot(133)
    plt.plot(test_times.numpy(),trajectory.detach().numpy())
    plt.title('n_events:{}'.format(n_events))
    plt.show()

    out_iters+=1
