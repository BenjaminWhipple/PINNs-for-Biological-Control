import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time

import matplotlib.pyplot as plt


# In this case, we don't need the GPU.
device=torch.device("cpu")
print(device)


# Using our model
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

# Load our model
my_model = torch.load("LV_Deterministic_Control_Value.pt",map_location=device)

# RK45 implementation from some time ago. 
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
    X,Y = y
    dX = X*(1-Y)
    dY = Y*(X-1)
    return np.array([dX,dY])

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

U = GetControl(my_model)

print(U(1,1,1))

print(U(2,1,2))


# Controlled system
def f_controlled(t,y):
    X,Y = y
    dX = X*(1-Y)
    dY = Y*(X-1)+U(X,Y,t)
    return np.array([dX,dY])


# Integrate uncontrolled system
t, results = RK45(f,[1.0,0.1],(0.0,5.0),1000)

# Integrate controlled system
t_2, results_2 = RK45(f_controlled,[1.0,0.1],(0.0,5.0),1000)

### PLOTTING

# Plot Uncontrolled System

plt.figure(figsize=(8,4.0))
ax = plt.subplot(111)

ax.set_title('Uncontrolled Predator-Prey System')
ax.plot(t, results[:,0],label="X: Prey")
ax.plot(t, results[:,1],label="Y: Predator")
ax.set_ylabel("Population")
ax.set_xlabel("Time")

ax.legend(loc='upper right')
plt.savefig("LV_Deterministic_Uncontrolled.png")

# Plot Controlled System

plt.figure()
ax1 = plt.subplot(211)
ax1.set_title('Controlled Predator-Prey System')
ax1.plot(t_2, results_2[:,0],label="X : Prey")
ax1.plot(t_2, results_2[:,1],label="Y : Predator")
ax1.set_ylabel("Population")

# Recover control, this part can take a few seconds.

control = []
for i in range(len(t_2)):
    control.append(U(results_2[:,0][i],results_2[:,1][i],t_2[i]))


ax2 = plt.subplot(212)
ax2.plot(t_2, control,label="u : Control",color="red")
ax2.set_ylabel("Control")
ax2.set_xlabel("Time")

ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

plt.savefig("LV_Deterministic_Controlled.png")

