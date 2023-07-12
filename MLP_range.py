import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from numpy.random import randn


inS = 5
outS = 25

interS = 10

nL = 0

bInit = .0

batchS = 200

lr = .03

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        l = [nn.Linear(inS, interS)] + nL * [nn.Linear(interS, interS)] + [nn.Linear(interS, outS)]
        a = [nn.Tanh()] * (nL+2)
        self.layer = nn.Sequential(
          *[l[i//2] if i%2==0 else a[i//2] for i in range(2*(nL+2))]
        )

        for i in range(nL+2):
            torch.nn.init.uniform_(self.layer[2*i].bias, -bInit,bInit)
            torch.nn.init.xavier_normal_(self.layer[2*i].weight)
            # self.layer[2*i].bias.data.fill_(0.0)


    def forward(self,x):
        out = self.layer(x)
        return out





net = Net()
optimizer = optim.Adadelta(net.parameters(), lr=lr)

data = torch.Tensor(randn(batchS, inS))
target = torch.Tensor(randn(batchS, outS))

for i in range(20000):

    optimizer.zero_grad()
    output = net(data)
    loss = F.mse_loss(output, target)
    loss.backward()
    if i % 100 == 0:
        print('Train Epoch: {}, Loss: {:.6f}'.format(i,  loss.item()))
        pass
    optimizer.step()
    
