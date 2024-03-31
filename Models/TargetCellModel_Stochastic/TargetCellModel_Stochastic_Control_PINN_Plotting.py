import torch
import torch.nn as nn
from torch.autograd import Variable
device=torch.device("cpu")
print(device)
import numpy as np
import time

import matplotlib.pyplot as plt

### Define the constants

T = 10.0

# Constants for the loss function
# Instantaneous costs
e1 = 1.0
e2 = 1.0

# Constants for the dynamics (nondimensionalized)
b = 0.75
a = 1.0
k = 4.0
l = 1.0

# Initial Conditions from ... 
U0 = 1.0
I0 = 0.0
V0 = 1.0

### Define the model.

class Net(nn.Module):
    def __init__(self,size):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(4,size)
        self.hidden_layer2 = nn.Linear(size,size)
        self.hidden_layer3 = nn.Linear(size,size)
        self.hidden_layer4 = nn.Linear(size,size)
        self.hidden_layer5 = nn.Linear(size,size)
        self.output_layer = nn.Linear(size,1)

    def forward(self,U,I,V,t):
        inputs = torch.cat([U,I,V,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out) ## For regression, no activation is used in output layer
        return output

    def predict(self,U,I,V,t):
        U = U.unsqueeze(0)
        I = I.unsqueeze(0)
        V = V.unsqueeze(0)
        t = t.unsqueeze(0)

        U.requires_grad_(True)
        I.requires_grad_(True)
        V.requires_grad_(True)
        t.requires_grad_(True)
        output = self.forward(U,I,V,t)
        gradients = torch.autograd.grad(output,(U,I,V,t))
        return output.item(), gradients

my_model = torch.load("TargetCellModel_Stochastic_Control_Value.pt",map_location=device)

# Euler-Maruyama integrator, which was written a while ago.
def Euler_Maruyama(a, B, X0, T0, T, N):
    """
    Simulate a multivariate stochastic differential equation using the Euler-Maruyama method.

    Parameters:
    - a: Function for the drift coefficient, a(X, t), returning a vector.
    - B: Function for the diffusion coefficient, B(X, t), returning a matrix.
    - X0: Initial condition vector.
    - T0: Initial time.
    - T: Final time.
    - N: Number of steps.

    Returns:
    - t: Array of time points.
    - X: Array of simulated process values, with each row representing a time point.
    """
    dt = (T - T0) / N  # Time step size
    t = np.linspace(T0, T, N+1)
    d = len(X0)  # Dimension of the process
    X = np.zeros((N+1, d))
    X[0, :] = X0

    for i in range(N):
        t_i = T0 + i*dt
        dW = np.random.normal(0, np.sqrt(dt), d)  # Vector of Wiener increments
        X[i+1, :] = X[i, :] + a(X[i, :], t_i) * dt + B(X[i, :], t_i) @ dW

    return t, X

# Uncontrolled system
def f(y,t):
    U,I,V = y
    dU = -b*V*U
    dI = b*V*U - a*I
    dV = k*I - l*V
    return np.array([dU,dI,dV])

def diffusion(y,t):
    U,I,V = y
    return np.array([0.1*U,0.1*I,0.1*V])

# Define the control
def GetControl(network):
    def Control(U,I,V,t):
        if I != 0:
            U = torch.tensor([float(U)])
            I = torch.tensor([float(I)])
            V = torch.tensor([float(V)])
            t = torch.tensor([float(t)])
            pred = network.predict(U,I,V,t)
            delVdelI = pred[1][1].item()
            u = 0.5*I.item()*(1/e2)*(delVdelI)
            return u
        else:
            return 0
    return Control

u = GetControl(my_model)


# Controlled system
def f_controlled(y,t):
    U,I,V = y
    dU = -b*V*U
    dI = b*V*U - a*I - u(U,I,V,t)*I # I know, u is confusing as the control
    dV = k*I - l*V
    return np.array([dU,dI,dV])

# Integrate uncontrolled system
t, results = Euler_Maruyama(f,diffusion,np.array([U0,I0,V0]),0,T,1000)

# Integrate controlled system
t_2, results_2 = Euler_Maruyama(f_controlled,diffusion,np.array([U0,I0,V0]),0,T,1000)

### Make Plots

# Plot Uncontrolled System

plt.figure(figsize=(8,4.0))
ax = plt.subplot(111)

ax.set_title('Uncontrolled Target Cell System')
ax.plot(t, results[:,0],label="U: Uninfected Cells")
ax.plot(t, results[:,1],label="I: Infected Cells")
ax.plot(t, results[:,2],label="V: Virus")
ax.set_ylabel("Population")
ax.set_xlabel("Time")

ax.legend(loc='upper right')
plt.savefig("TargetCellModel_Stochastic_Uncontrolled.png")

# Plot Controlled System
plt.figure()
ax1 = plt.subplot(211)

ax1.set_title('Controlled Target Cell System System')
ax1.plot(t_2, results_2[:,0],label="U: Uninfected Cells")
ax1.plot(t_2, results_2[:,1],label="I: Infected Cells")
ax1.plot(t_2, results_2[:,2],label="V: Virus")
ax1.set_ylabel("Population")

control = []
for i in range(len(t_2)):
    control.append(u(results_2[:,0][i],results_2[:,1][i],results_2[:,2][i],t_2[i]))

ax2 = plt.subplot(212)
ax2.plot(t_2, control,label="u : Control",color="red")
ax2.set_ylabel("Control")
ax2.set_xlabel("Time")

ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

plt.savefig("TargetCellModel_Stochastic_Controlled.png")
