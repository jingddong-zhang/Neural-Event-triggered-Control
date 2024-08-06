import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
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



def generate_data(d_sigma=0.0,d_rho=0.0,d_beta=0.0):
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 3  # input dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)

    # d_sigma = 1.0
    # d_rho = 1.0
    # d_beta = 1.0
    model = NETC_high_noise(D_in, H1, D_out, (D_in,), [H1, 1],strength=0.1,d_sigma=d_sigma,d_rho=d_rho,d_beta=d_beta).to(device)
    model._control.load_state_dict(torch.load('./data/NETC-high_control.pkl'))
    model._lya.load_state_dict(torch.load('./data/NETC-high_lya.pkl'))
    model._alpha.load_state_dict(torch.load('./data/NETC-high_alpha.pkl'))
    test_times = torch.linspace(0, 4, 1000).to(device) # here we use long time than the other examples to collect at least 20 triggering times
    start = timeit.default_timer()
    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    seed_list = [0, 1, 6, 7, 9]
    res = {}
    for i in range(1):
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
            res[f'{i}']={'traj':cat_data.numpy(),'num':n_events,'traj_events':traj_events}
    # np.save('./NETC_parameter_perturbation',res)
    # plt.plot(test_times,trajectory_x)
    # plt.show()
    end = timeit.default_timer()
    print(f'average inference time={(end - start) / 5}')
    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(),torch.tensor(var_list).mean())
    return res

def multiple_generate():
    setup_seed(0)
    Res = {}
    for i in tqdm(range(5)):
        d_sigma = torch.from_numpy(np.random.normal(0,1,1))
        d_rho = torch.from_numpy(np.random.normal(0, 1, 1))
        d_beta = torch.from_numpy(np.random.normal(0, 1, 1))
        res = generate_data(d_sigma)
        Res[f'sigma_{i}'] = res
        res = generate_data(d_sigma=0.0,d_rho=d_rho)
        Res[f'rho_{i}'] = res
        res = generate_data(d_sigma=0.0,d_rho=0.0,d_beta=d_beta)
        Res[f'beta_{i}'] = res
    np.save('./rebuttal/parameter_noise',Res)
    print(f'totally done!')

def multiple_generate_v1():
    setup_seed(0)
    Res = {}
    intensity_list = [0.5,1.0,2.0]
    for intensity in intensity_list:
        sub_Res = {}
        for i in tqdm(range(5)):
            d_sigma,d_rho,d_beta = torch.from_numpy(np.random.normal(0,intensity,3))
             # = torch.from_numpy(np.random.normal(0, 1, 1))
             # = torch.from_numpy(np.random.normal(0, 1, 1))
            res = generate_data(d_sigma,d_rho,d_beta)
            sub_Res[f'{i}'] = res[f'{0}']
        Res[f'intensity_{intensity}'] = sub_Res
    np.save('./rebuttal/parameter_noise_v1',Res)
    print(f'totally done!')
multiple_generate_v1()


def generate_data_stochastic(d_x=0.0,d_y=0.0,d_z=0.0):
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 3  # input dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)


    model = NETC_high_stochastic(D_in, H1, D_out, (D_in,), [H1, 1],strength=0.1,d_x=d_x,d_y=d_y,d_z=d_z).to(device)
    model._control.load_state_dict(torch.load('./data/NETC-high_control.pkl'))
    model._lya.load_state_dict(torch.load('./data/NETC-high_lya.pkl'))
    model._alpha.load_state_dict(torch.load('./data/NETC-high_alpha.pkl'))
    test_times = torch.linspace(0, 4, 1000).to(device) # here we use long time than the other examples to collect at least 20 triggering times
    start = timeit.default_timer()
    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    seed_list = [0, 1, 6, 7, 9]
    res = {}
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
            res[f'{i}']={'traj':cat_data.numpy(),'num':n_events,'traj_events':traj_events}
    # np.save('./NETC_parameter_perturbation',res)
    # plt.plot(test_times,trajectory_x)
    # plt.show()
    end = timeit.default_timer()
    print(f'average inference time={(end - start) / 5}')
    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(),torch.tensor(var_list).mean())
    return res

def multiple_generate_stochastic():
    setup_seed(0)
    Res = {}
    intensity_list = [0.1,0.3,0.5]
    for i in tqdm(range(3)):
        d_x,d_y,d_z =intensity_list[i],intensity_list[i],intensity_list[i]
        res = generate_data_stochastic(d_x,d_y,d_z)
        Res[f'noise_{d_x}'] = res
    np.save('./rebuttal/noise_process',Res)
    print(f'totally done!')

# multiple_generate_stochastic()
# generate_data_stochastic(0.5,0.5,0.5)