import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import matplotlib.pyplot as plt

# In this case, we only use the CPU.
device=torch.device("cpu")
print(device)

# Our model's class
class Net(nn.Module):
    def __init__(self,size):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(3,size)
        self.hidden_layer2 = nn.Linear(size,size)
        self.hidden_layer3 = nn.Linear(size,size)
        self.hidden_layer4 = nn.Linear(size,size)
        self.hidden_layer5 = nn.Linear(size,size)
        self.output_layer = nn.Linear(size,1)

    def forward(self, x,y,t):
        inputs = torch.cat([x,y,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out) ## For regression, no activation is used in output layer
        return output

    def predict(self,x,y,t):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        t = t.unsqueeze(0)

        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        output = self.forward(x,y,t)
        gradients = torch.autograd.grad(output,(x,y,t))
        return output.item(), gradients

# Euler-Maruyama method, written a while ago.
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
    X,Y = y
    dX = X*(1-Y)
    dY = Y*(X-1)
    return np.array([dX,dY])

def diffusion(x,t):
    X,Y = x
    return np.array([0.1*X,0.1*Y])

t,X = Euler_Maruyama(f,diffusion,np.array([1.0,0.1]),0,5,1000)

plt.figure(figsize=(8,4.0))
ax = plt.subplot(111)

ax.set_title('Uncontrolled Predator-Prey System')
ax.plot(t, X[:,0],label="X: Prey")
ax.plot(t, X[:,1],label="Y: Predator")
ax.set_ylabel("Population")
ax.set_xlabel("Time")

ax.legend(loc='upper right')
plt.savefig("LV_Stochastic_Uncontrolled.png")

# Define the control
def GetControl(network):
    def Control(x,y,t):
        if x != 0:
            x = torch.tensor([float(x)])
            y = torch.tensor([float(y)])
            t = torch.tensor([float(t)])
            pred = network.predict(x,y,t)
            delVdelY = pred[1][1].item()
            u = -0.5*(delVdelY)
            return u
        else:
            return 0
    return Control

# Load our model
my_model = torch.load("LV_Stochastic_Control_Value.pt",map_location=device)

U = GetControl(my_model)

# Uncontrolled system
def f_controlled(y,t):
    X,Y = y
    dX = X*(1-Y)
    dY = Y*(X-1)+U(X,Y,t)
    return np.array([dX,dY])
    
t,X = Euler_Maruyama(f_controlled,diffusion,np.array([1.0,0.1]),0,5,1000)

plt.figure()
ax1 = plt.subplot(211)

ax1.set_title('Controlled Predator-Prey System')
ax1.plot(t, X[:,0],label="X : Prey")
ax1.plot(t, X[:,1],label="Y : Predator")
ax1.set_ylabel("Population")

# Get the control, this code can take a few seconds
control = []
for i in range(len(t)):
    control.append(U(X[:,0][i],X[:,1][i],t[i]))

ax2 = plt.subplot(212)
ax2.plot(t, control,label="u : Control",color="red")
ax2.set_ylabel("Control")
ax2.set_xlabel("Time")

ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

plt.savefig("LV_Stochastic_Controlled.png")
