import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules import Module

from numpy.random import randn

import math
import sys

import matplotlib.pyplot as plt

inS = 1
outS = 2

interS = 128 #128

nL = 3 #3

batchS = 25

lr = .001

nAnchors = 10
# anchorD = .5/math.sqrt(outS)
anchorD = .1

nPlotPoints = 1000

class sine(Module):
    def forward(self, z):
        return torch.sin(z)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        l = [nn.Linear(inS, interS, bias=False)] + nL * [nn.Linear(interS, interS, bias=False)] + [nn.Linear(interS, outS, bias=False)]
        # a = [nn.ReLU()] * (nL+1) + [nn.Tanh()]
        # a = [nn.ReLU()] * (nL+1) + [sine()]
        a = [sine()] * (nL+2)
        # a = [nn.Tanh()] * (nL+2)
        self.layer = nn.Sequential(
          *[l[i//2] if i%2==0 else a[i//2] for i in range(2*(nL+2))]
        )

        for i in range(nL+2):
            # torch.nn.init.xavier_normal_(self.layer[2*i].weight)
            torch.nn.init.kaiming_uniform_(self.layer[2*i].weight)
            # torch.nn.init.normal_(self.layer[2*i].weight)

            # torch.nn.init.uniform_(self.layer[2*i].bias, -bInit,bInit) #I disabled biases entirely
            # self.layer[2*i].bias.data.fill_(0.0)


    def forward(self,x):
        out = self.layer(x)
        return out



def plot(net):
    inputs = []
    x = []
    y = []
    for i in range(-nPlotPoints, nPlotPoints):
        inputs.append(5 * (i/nPlotPoints)**5)
        r = net(torch.tensor([[inputs[-1]]])).detach().numpy()
        x.append(r[0][0])
        y.append(r[0][1])
    
    
    plt.show()

net = Net()
optimizer = optim.Adadelta(net.parameters(), lr=lr,  weight_decay=0) #SGD or Adadelta ?

data = torch.Tensor(randn(batchS, inS))
target = torch.clamp(torch.Tensor(randn(batchS, outS))*.3, -1, 1)


def computeSF(net):
    net.eval()
    nx = 20
    ny = 500
    Y = torch.clamp(torch.Tensor(randn(ny, outS))*.3, -1, 1)
    
    s = 0
    for i in range(ny):
        av = 0
        for _ in range(1):
            X = torch.Tensor(randn(nx, inS))
            res = net(X)
            y = Y[i]
            dmin = outS * 100
            for j in range(nx):
                d = torch.norm(res[j] - y)**2
                if d < dmin:
                    dmin = d
            av+=dmin

        s += av/1

    return s.detach()/ny

def visualizeStretch(net):
    inputs = []
    for i in range(-nPlotPoints, nPlotPoints):
        inputs.append(5 * (i/nPlotPoints)**5)

    x0 = [0]*2*nPlotPoints
    y0 = [0]*2*nPlotPoints
    for i in range(-nPlotPoints, nPlotPoints):
        r = net(torch.tensor([[inputs[nPlotPoints+i]]])).detach().numpy()
        x0[i+nPlotPoints] = r[0][0]
        y0[i+nPlotPoints] = r[0][1]

    Y = torch.Tensor([[.3, .3]])


    Xs = torch.Tensor(randn(batchS, inS))
        

    with torch.no_grad():
        bestXid = torch.argmin(torch.norm(net(Xs) - Y.repeat(batchS, 1), dim = 1))
        X = Xs[bestXid][None, :]
        anchors = torch.Tensor(randn(nAnchors, inS)) * anchorD + X.repeat(nAnchors, 1)
        anchorsY = net(anchors)
    
    # print(net.layer[6].weight)
    print("\n\n")
    optimizer.zero_grad()
    y = net(X)
    loss = F.mse_loss(y, Y)
    print(loss.item())
    loss.backward()
    for g in optimizer.param_groups:
        g['lr'] = .1
    optimizer.step()
    # print(net.layer[6].weight)
    x1 = [0]*2*nPlotPoints
    y1 = [0]*2*nPlotPoints
    for i in range(-nPlotPoints, nPlotPoints):
        r = net(torch.tensor([[inputs[nPlotPoints+i]]])).detach().numpy()
        x1[i+nPlotPoints] = r[0][0]
        y1[i+nPlotPoints] = r[0][1]

    optimizer.zero_grad()
    newAnchorsY = net(anchors)
    loss = F.mse_loss(newAnchorsY, anchorsY)
    loss.backward()
    for g in optimizer.param_groups:
        g['lr'] = .1
    optimizer.step()

    x2 = [0]*2*nPlotPoints
    y2 = [0]*2*nPlotPoints
    for i in range(-nPlotPoints, nPlotPoints):
        r = net(torch.tensor([[inputs[nPlotPoints+i]]])).detach().numpy()
        x2[i+nPlotPoints] = r[0][0]
        y2[i+nPlotPoints] = r[0][1]
    
    plt.plot(.3,.3,"y")
    plt.plot(x2, y2, "b-")
    plt.plot(x1, y1, "r-")
    plt.plot(x0, y0, "g-")

    plt.show()


def trainStake(net):
    net.train()
    
    SFs = []
    es = []
    for i in range(15000):
        Y = torch.clamp(torch.Tensor(randn(1, outS))*.3, -1, 1)
        Xs = torch.Tensor(randn(batchS, inS))
        
        with torch.no_grad():
            bestXid = torch.argmin(torch.norm(net(Xs) - Y.repeat(batchS, 1), dim = 1))
            X = Xs[bestXid][None, :]
            # anchors = -X
            # anchors = torch.Tensor(randn(nAnchors, inS))
            anchors = torch.Tensor(randn(nAnchors, inS)) * anchorD + X.repeat(nAnchors, 1)
            anchorsY = net(anchors)

        
        optimizer.zero_grad()
        y = net(X)
        loss = F.mse_loss(y, Y)
        lrFactor = loss.item()
        for g in optimizer.param_groups:
            g['lr'] = lr * lrFactor 
        loss.backward()
        optimizer.step()
        
        
        optimizer.zero_grad()
        for g in optimizer.param_groups:
            g['lr'] = lr * lrFactor *.8
        newAnchorsY = net(anchors)
        loss = F.mse_loss(newAnchorsY, anchorsY)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            es.append(i)
            SFs.append(computeSF(net).item())
            print("{:.6f}".format(SFs[-1]))
    
    plt.plot(es, SFs, 'r-')
    plt.show()
    
def trainRandom(net):
    net.train()
    for i in range(20000):

        optimizer.zero_grad()
        output = net(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        if i % 1000 == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(i,  loss.item()))
            pass
        optimizer.step()


# plt.xlim(-1,1)
# plt.ylim(-1,1)

visualizeStretch(net)

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
