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

# Initial Conditions from
U0 = 1.0
I0 = 0.0
V0 = 1.0

### Define the model

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

my_model = torch.load("TargetCellModel_Deterministic_Control_Value.pt",map_location=device)

# RK45 integrator, written a while ago.
def RK45(f, x0, t_range, num_steps):
    """
    Solve the initial value problem using the RK45 (Dormand-Prince) method.
    
    Parameters:
        f: function
            The ordinary differential equation, dy/dt = f(t, y).
        x0: float
            Initial condition.
        t_range: tuple
            A tuple (t0, t_end) specifying the start and end points of the interval.
        num_steps: int
            Number of steps for the integration.
            
    Returns:
        t_values: array
            Array of time values.
        x_values: array
            Array of solution values.
    """
    
    t0, t_end = t_range
    t_values = np.linspace(t0, t_end, num_steps)
    h = (t_end - t0) / num_steps
    x_values = [x0]
    
    for i in range(1, num_steps):
        t = t_values[i-1]
        x = x_values[i-1]

        k1 = h * f(t, x)
        k2 = h * f(t + 0.25 * h, x + 0.25 * k1)
        k3 = h * f(t + 3/8 * h, x + 3/32 * k1 + 9/32 * k2)
        k4 = h * f(t + 12/13 * h, x + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
        k5 = h * f(t + h, x + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
        k6 = h * f(t + 0.5 * h, x - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
        
        x_next = x + 16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6

        t_values[i] = t + h
        x_values.append(x_next)
    
    return t_values, np.array(x_values)

# Uncontrolled system
def f(t,y):
    U,I,V = y
    dU = -b*V*U
    dI = b*V*U - a*I
    dV = k*I - l*V
    return np.array([dU,dI,dV])

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

print(u(1,1,1,1))

print(u(2,1,2,1))


# Controlled system
def f_controlled(t,y):
    U,I,V = y
    dU = -b*V*U
    dI = b*V*U - a*I - u(U,I,V,t)*I # I know, u is confusing as the control
    dV = k*I - l*V
    return np.array([dU,dI,dV])


# Integrate uncontrolled system
t, results = RK45(f,[U0,I0,V0],(0.0,T),1000)

t_2, results_2 = RK45(f_controlled,[U0,I0,V0],(0.0,T),1000)

### Make plots

# Plot uncontrolled system
plt.figure(figsize=(8,4.0))
ax = plt.subplot(111)

ax.set_title('Uncontrolled Target Cell System')
ax.plot(t, results[:,0],label="U: Uninfected Cells")
ax.plot(t, results[:,1],label="I: Infected Cells")
ax.plot(t, results[:,2],label="V: Virus")
ax.set_ylabel("Population")
ax.set_xlabel("Time")

ax.legend(loc='upper right')
plt.savefig("TargetCellModel_Deterministic_Uncontrolled.png")

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

plt.savefig("TargetCellModel_Deterministic_Controlled.png")
