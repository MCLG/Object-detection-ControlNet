import torch
from numpy.random import uniform 
import numpy as np
import matplotlib.pyplot as plt

b = 0.99
l = 1
steps = 1000
R1, R1inverse, R2, L, L_stb = [],[],[],[],[]

for k in range(steps) :

    Lsimple = torch.tensor(uniform(0.001,2,1000), device = 1) * (steps/(steps+k))
    Lc = torch.tensor(uniform(0.0001,.5,1000), device = 1) * (steps/(steps+k))

    r1 = l*torch.mean(Lc/Lsimple)
    r1inv = r1**-1
    r2 = (l**-1)*torch.mean((Lsimple/(Lc)))

    new_l = .5*(r1+r2)
    new_l_stable = .5*(r1+r1inv)

    l = b*l + (1-b)*new_l
    l_stable = b*l + (1-b)*new_l_stable

    R1.append(r1.cpu())
    R1inverse.append(r1inv.cpu())
    R2.append(r2.cpu())
    L.append(l.cpu())
    L_stb.append(l_stable.cpu())

fig,ax = plt.subplots(2,2,figsize=(12,8))
x = [k for k in range(steps)]
ax[0,0].plot(x,R1)
ax[0,0].plot(x,R2)
ax[0,0].set_yscale('log')

ax[0,1].plot(x,R1)
ax[0,1].plot(x,R1inverse)
ax[0,1].set_yscale('log')

ax[1,0].plot(x, L)

ax[1,1].plot(x, L_stb)

def sigminv(x) :
    return 2-torch.sigmoid(x)

fig.savefig('expo_smoothing.png')

multipliers = np.random.uniform(0.7, 1.5, size=len(x))
x = x * multipliers
print(x.shape)
fig2, ax2 = plt.subplots(1,1,figsize = (10,8))
sigm = torch.sigmoid(.01*torch.tensor(x))
sigmainv = sigminv(.01*torch.tensor(x))

one = [1 for k in range(steps)]

ax2.plot([k for k in range(steps)],sigm)
ax2.plot([k for k in range(steps)],sigmainv)
ax2.plot([k for k in range(steps)],one,linestyle = '--')

fig2.savefig('expected_expo_smoothing.png')