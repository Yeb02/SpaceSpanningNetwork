import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from numpy.random import randn

from scipy.spatial import ConvexHull

import sys

inS = 10
outS = 6

interS = 64

nL = 1

bInit = .0

batchS = 1

lr = .01

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

    
def train(e, optimizer, model):
    for i in range(e):
        data = torch.Tensor(randn(batchS, inS))
        target = torch.Tensor(randn(batchS, outS))

        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            # print('Train Epoch: {}, Loss: {:.6f}'.format(i,  loss.item()))
            pass


def computeVolume(model):
    nPoints = 500 #arbitrary. Exponential ??
    nComps = 5

    v0 = 0
    model.eval()
    for j in range(nComps):
      srcPoints = torch.Tensor(randn(nPoints, inS))
      with torch.no_grad(): 
        cloud = model(srcPoints)
        v0 += ConvexHull(cloud.detach().numpy()).volume
        
    return v0 / nComps


net = Net()
optimizer = optim.Adadelta(net.parameters(), lr=lr)

# three times to estimate variance
print(computeVolume(net))
print(computeVolume(net))
print(computeVolume(net))

train(200, optimizer, net)
print(computeVolume(net))

train(2000, optimizer, net)
print(computeVolume(net))

# a = torch.tensor([inS*[1]], dtype=torch.float32)
# b = torch.tensor([inS*[-1]], dtype=torch.float32)
# c = torch.tensor([inS*[0]], dtype=torch.float32)

# print(net(a).data[0])
# print(net(b).data[0])
# print(net(c).data[0])

# train(200, optimizer, net)

# print(net(a).data[0])
# print(net(b).data[0])
# print(net(c).data[0])

# train(2000, optimizer, net)

# print(net(a).data[0])
# print(net(b).data[0])
# print(net(c).data[0])
