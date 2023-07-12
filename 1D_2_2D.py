import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from numpy.random import randn

import math
import sys

import matplotlib.pyplot as plt

inS = 1
outS = 2

interS = 64

nL = 5

bInit = .1

batchS = 100

lr = .02

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        l = [nn.Linear(inS, interS)] + nL * [nn.Linear(interS, interS)] + [nn.Linear(interS, outS)]
        a = [nn.Tanh()] * (nL+2)
        self.layer = nn.Sequential(
          *[l[i//2] if i%2==0 else a[i//2] for i in range(2*(nL+2))]
        )

        for i in range(nL+2):
            torch.nn.init.xavier_normal_(self.layer[2*i].weight)
            # torch.nn.init.uniform_(self.layer[2*i].bias, -bInit,bInit)

            self.layer[2*i].bias.data.fill_(0.0)


    def forward(self,x):
        out = self.layer(x)
        return out



def plot(net):
    inputs = []
    x = []
    y = []
    for i in range(-100, 100):
        inputs.append(5 * (i/100)**5)
        r = net(torch.tensor([[inputs[-1]]])).detach().numpy()
        x.append(r[0][0])
        y.append(r[0][1])
    
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.plot(x,y,'o-')
    plt.show()

net = Net()
# sys.exit(0)
optimizer = optim.Adadelta(net.parameters(), lr=lr)

data = torch.Tensor(randn(batchS, inS))
target = torch.clamp(torch.Tensor(randn(batchS, outS))*.3, -1, 1)


def computeSF(net):
    net.eval()
    nx = 100
    ny = 200
    Y = torch.clamp(torch.Tensor(randn(ny, outS))*.3, -1, 1)
    
    s = 0
    for i in range(ny):
        X = torch.Tensor(randn(nx, inS))
        res = net(X)
        y = Y[i]
        dmin = outS * 100
        for j in range(nx):
            d = torch.norm(res[j] - y)**2
            if d < dmin:
                dmin = d

        s += dmin

    return s.detach()/ny


def trainStake(net):
    net.train()
    SFs = []
    es = []
    for i in range(100000):
        Y = torch.clamp(torch.Tensor(randn(1, outS))*.3, -1, 1)
        Xs = torch.Tensor(randn(batchS, inS))
        
        with torch.no_grad():
            bestXid = torch.argmin(torch.norm(net(Xs) - Y.repeat(batchS, 1), dim = 1))

        y = net(Xs[bestXid][None, :]) #reshape to [1, inS]
        optimizer.zero_grad()
        loss = F.mse_loss(y, Y)
        loss.backward()
        if i % 1000 == 0:
            es.append(i)
            SFs.append(computeSF(net).item())
            print("{:.6f}".format(SFs[-1]))
        optimizer.lr = lr * loss.item() #important
        optimizer.step()
    
    plt.plot(es, SFs, 'r-')
    plt.show()
    
def trainRandom(net):
    net.train()
    for i in range(10000):

        optimizer.zero_grad()
        output = net(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        if i % 1000 == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(i,  loss.item()))
            pass
        optimizer.step()

print(computeSF(net))
print(computeSF(net))
print(computeSF(net))
print(computeSF(net))
plot(net)
trainStake(net)
plot(net)
print(computeSF(net))
print(computeSF(net))
print(computeSF(net))
print(computeSF(net))
