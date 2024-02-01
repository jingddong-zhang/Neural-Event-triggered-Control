import torch
import torch.nn as nn
from NeuralIntegral import NeuralIntegral
from ParallelNeuralIntegral import ParallelNeuralIntegral
from spectral_normalization import SpectralNorm

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    # def forward(self, x, h):
    #     return self.net(torch.cat((x, h), 1)) + 1.
    def lip(self,x):
        # 比2小
        return x-(x-200.).relu()

    def forward(self, x):
        return IntegrandNN.lip(self,self.net(x) + 1.)
        # return self.net(x) + 1.

class MonotonicNN(nn.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps

    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h is just other conditionning variables.
    '''
    # def forward(self, x, h):
    #     x0 = torch.zeros(x.shape).to(self.device)
    #     out = self.net(h)
    #     offset = out[:, [0]]
    #     scaling = torch.exp(out[:, [1]])
    #     return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset
    def forward(self, x):
        x0 = torch.zeros(x.shape).to(self.device)
        # out = self.net(h)
        # offset = out[:, [0]]
        # scaling = torch.exp(out[:, [1]])
        return ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), self.nb_steps)


def check_run():
    def f(x_1):
        return 1.001*(x_1)

    def create_dataset(n_samples):
        x = torch.randn(n_samples, 3)
        y = f(x[:, 0])
        return x, y
    # train_x, train_y = create_dataset(1000)
    # x = train_x[0:100].requires_grad_()
    # print(x[:, [0]].shape)
    # model_monotonic = MonotonicNN(1, [100, 100, 100], nb_steps=100)
    # y_pred = model_monotonic(x[:, [0]])[:,0]
    # print(y_pred.shape)


    model_monotonic = MonotonicNN(1, [10, 10], nb_steps=50)
    optim_monotonic = torch.optim.Adam(model_monotonic.parameters(), 1e-3, weight_decay=1e-5)

    train_x, train_y = create_dataset(1000)
    # test_x, test_y = create_dataset(args.nb_test)
    b_size = 1000

    for epoch in range(0, 10):
        # Shuffle
        idx = torch.randperm(1000)
        train_x = train_x[idx]
        train_y = train_y[idx]
        avg_loss_mon = 0.
        avg_loss_mlp = 0.
        for i in range(0, 100):
            # Monotonic
            x = train_x[i:i + b_size].requires_grad_()
            y = train_y[i:i + b_size].requires_grad_()
            y_pred = model_monotonic(x[:, [0]])[:, 0]
            loss = ((y_pred - y)**2).sum()
            optim_monotonic.zero_grad()
            loss.backward()
            optim_monotonic.step()
            avg_loss_mon += loss.item()
            print(i,loss)

# check_run()