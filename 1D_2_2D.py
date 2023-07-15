import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules import Module

from numpy.random import randn

import math
import sys

import matplotlib.pyplot as plt

inS = 4
outS = 16

interS = 32

nL = 6

batchS = 50

lr = .01

nAnchors = 10
anchorD = .5/math.sqrt(outS)

class sine(Module):
    def forward(self, z):
        return torch.sin(z)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        l = [nn.Linear(inS, interS, bias=False)] + nL * [nn.Linear(interS, interS, bias=False)] + [nn.Linear(interS, outS, bias=False)]
        a = [nn.ReLU()] * (nL+1) + [nn.Tanh()]
        # a = [nn.ReLU()] * (nL+1) + [sine()]
        # a = [sine()] * (nL+2)
        # a = [nn.Tanh()] * (nL+2)
        self.layer = nn.Sequential(
          *[l[i//2] if i%2==0 else a[i//2] for i in range(2*(nL+2))]
        )

        for i in range(nL+2):
            torch.nn.init.xavier_normal_(self.layer[2*i].weight)

            # torch.nn.init.uniform_(self.layer[2*i].bias, -bInit,bInit) #I disabled biases entirely
            # self.layer[2*i].bias.data.fill_(0.0)


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
optimizer = optim.Adadelta(net.parameters(), lr=lr,  weight_decay=0) #SGD or Adadelta ?

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
    for i in range(50000):
        Y = torch.clamp(torch.Tensor(randn(1, outS))*.3, -1, 1)
        Xs = torch.Tensor(randn(batchS, inS))
        
        with torch.no_grad():
            bestXid = torch.argmin(torch.norm(net(Xs) - Y.repeat(batchS, 1), dim = 1))
            X = Xs[bestXid][None, :]
            # anchors = torch.Tensor(randn(nAnchors, inS)) * anchorD + X.repeat(nAnchors, 1)
            # anchorsY = net(anchors)

        
        optimizer.zero_grad()
        y = net(X)
        loss = F.mse_loss(y, Y)
        loss.backward()
        lrFactor = loss.item() if loss.item() > 1 else loss.item()**2
        optimizer.lr = lr * lrFactor 
        optimizer.step()
        
        
        # optimizer.zero_grad()
        # newAnchorsY = net(anchors)
        # loss = F.mse_loss(newAnchorsY, anchorsY)
        # loss.backward()
        # # optimizer.lr = optimizer.lr / nAnchors # I should normalize the grad.
        # optimizer.lr = optimizer.lr * .5
        # optimizer.step()

        if i % 1000 == 0:
            es.append(i)
            SFs.append(computeSF(net).item())
            print("{:.6f}".format(SFs[-1]))
    
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
# plot(net)
trainStake(net)
# plot(net)
print(computeSF(net))
print(computeSF(net))
print(computeSF(net))
print(computeSF(net))
