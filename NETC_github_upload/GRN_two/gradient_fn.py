from functions import *


N = 20        # sample size
D_in = 2            # input dimension
H1 = 20             # hidden dimension
D_out = 2           # output dimension
data = torch.Tensor(N, D_in).uniform_(-5, 5).requires_grad_(True)

model = ICNN((D_in,), [10,10,1])
res1 = model.forward(data)
true_dres1 = torch.autograd.grad(res1.sum(),data,create_graph=True)[0]
dres1 = model.dicnn_fn(data)
# a = torch.randn([3,2])
# c = torch.randn([2,3,4])
# d = torch.mm(a,c)
# b = a.unsqueeze(1).repeat(1,5,1)
print(res1.shape,dres1.shape,true_dres1.shape,torch.std(true_dres1-dres1))