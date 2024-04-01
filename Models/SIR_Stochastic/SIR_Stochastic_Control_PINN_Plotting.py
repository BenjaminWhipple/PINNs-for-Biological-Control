import torch
import torch.nn as nn
from torch.autograd import Variable
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
import numpy as np
import time

import warnings
import matplotlib.pyplot as plt

### Define many of the constants
#Performance:
iterations = 100000

NETWORK_SIZE=400
MIN_STATE = 0.0
MAX_STATE = 2.0 # Technically, it is 1, but we want to be safe.
T=16.0
BC_SAMPLES=500
PDE_SAMPLES=10000

### Define the parameters
# Cost weights
c1=1.0
c2=1.0

# Dynamics
beta=1.5
gamma=0.3

# Uncertainty in parameters
sigma_beta=0.1
sigma_gamma=0.1

# Uncertainty in state
sigma_S = 0.1
sigma_I = 0.1
sigma_R = 0.1

# Control constraints
u_min = 0.0
u_max = 1.0

# Trajectory Constants
S0 = 0.99
I0 = 0.01
R0 = 0.0


# Define our model
class Net(nn.Module):
    def __init__(self,size):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(4,size)
        self.hidden_layer2 = nn.Linear(size,size)
        self.hidden_layer3 = nn.Linear(size,size)
        self.hidden_layer4 = nn.Linear(size,size)
        self.hidden_layer5 = nn.Linear(size,size)
        self.output_layer = nn.Linear(size,1)

    def forward(self, S,I,R,t):
        inputs = torch.cat([S,I,R,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out) ## For regression, no activation is used in output layer
        return output

    def predict(self,S,I,R,t):
        S = S.unsqueeze(0)
        I = I.unsqueeze(0)
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)

        S.requires_grad_(True)
        I.requires_grad_(True)
        R.requires_grad_(True)
        t.requires_grad_(True)
        output = self.forward(S,I,R,t)
        gradients = torch.autograd.grad(output,(S,I,R,t))
        return output.item(), gradients

my_model = torch.load("Checkpoints/SIR_Stochastic_Control_Value.pt",map_location=device)

# Define the control
def GetControl(network):
    def Control(S,I,R,t):
        if I != 0:
            S = torch.tensor([float(S)])
            I = torch.tensor([float(I)])
            R = torch.tensor([float(R)])
            t = torch.tensor([float(t)])
            pred = network.predict(S,I,R,t)
            
            #print(pred)
            
            V_S = pred[1][0].item()
            V_I = pred[1][1].item()
            u_star = -(1/2)*((V_S - V_I)*beta*S*I)
            #delVdelI = pred[1][1].item()
            u_star_c = torch.clamp(u_star, u_min, u_max)
            #u_star = 
            return u_star_c.item()
        else:
            return 0
    return Control

U = GetControl(my_model)

#print(my_model)
print(U(0.5,0.1,0.4,1.0))

def Euler_Maruyama(a, B, X0, noise_d, T0, T, N):
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
    d = d = len(X0) # Dimension of the process
    X = np.zeros((N+1, d))
    X[0, :] = X0

    for i in range(N):
        t_i = T0 + i*dt
        dW = np.random.normal(0, np.sqrt(dt), noise_d)  # Vector of Wiener increments
        #print(B(X[i, :], t_i))
        X[i+1, :] = X[i, :] + a(X[i, :], t_i) * dt + B(X[i, :], t_i) @ dW

    return t, X

# Uncontrolled system
def f(y,t):
    S,I,R = y
    dS = -beta*S*I
    dI = beta*S*I - gamma*I
    dR = gamma*I
    return np.array([dS,dI,dR])

# Controlled system.
def f_controlled(y,t):
    S,I,R = y
    dS = -(1-U(S,I,R,t))*beta*S*I
    dI = (1-U(S,I,R,t))*beta*S*I - gamma*I
    dR = gamma*I
    return np.array([dS,dI,dR])


# Noise
def diffusion(y,t):
    S,I,R = y
    return np.array([[-sigma_beta*S*I,0,sigma_S*S, 0, 0],[sigma_beta*S*I, -sigma_gamma*I,0,sigma_I*I,0],[0,sigma_gamma*I,0,0,sigma_R*R]])
    
# Integrate uncontrolled system
NUM_TRAJECTORIES = 20

ts = []
Xs = []

iter = 0
while len(ts)<=NUM_TRAJECTORIES:
    iter += 1
    print(iter)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        success = True
        try:
            t, X = Euler_Maruyama(f,diffusion,np.array([S0,I0,R0]),5,0,T,1000)
        except Warning:
            success = False
    
    if success == True:
        if np.max(X) < 2.0:
            # For the SIR model, we have assumed that the population sizes are normalized. We should normalize each time step.
            X_sum = np.sum(X,axis=1)
            X_sum = X_sum[:, None]
            X_norm = X / X_sum
            
            ts.append(t)
            Xs.append(X_norm)
           
    print(success)


# Integrate controlled system
t2s = []
X2s = []

iter = 0
while len(t2s)<=NUM_TRAJECTORIES:
    iter += 1
    print(iter)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        success = True
        try:
            t,X = Euler_Maruyama(f_controlled,diffusion,np.array([S0,I0,R0]),5,0,T,1000)
        except Warning:
            success = False
    
    if success == True:
        if np.max(X) < 5.0:
            X_sum = np.sum(X,axis=1)
            X_sum = X_sum[:, None]
            X_norm = X / X_sum
            t2s.append(t)
            X2s.append(X_norm)
           
    print(success)

# Get controls
controls = []
for j in range(NUM_TRAJECTORIES):
    control = []
    for i in range(len(t2s[j])):
        control.append(U(X2s[j][:,0][i],X2s[j][:,1][i],X2s[j][:,2][i],t2s[j][i]))
    controls.append(control)

### Make Plots

ALPHA = 0.2
# Plot Uncontrolled System

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(4, 6),sharex="col")

axs[0].set_title("Uncontrolled Target Cell System")
for i in range(NUM_TRAJECTORIES):
    axs[0].plot(ts[i], [max(0, x) for x in Xs[i][:,0]],label="U: Uninfected Cells",color="green",alpha=ALPHA)

handles, labels = axs[0].get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
axs[0].legend(handle_list, label_list,loc='upper right')

for i in range(NUM_TRAJECTORIES):
    axs[1].plot(ts[i], [max(0, x) for x in Xs[i][:,1]],label="I: Infected Cells",color="blue",alpha=ALPHA)

handles, labels = axs[1].get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
axs[1].legend(handle_list, label_list,loc='upper right')

for i in range(NUM_TRAJECTORIES):
    axs[2].plot(ts[i], [max(0, x) for x in Xs[i][:,2]],label="P: Virus",color="darkorange",alpha=ALPHA)

axs[2].set_xlabel("t: Time")

handles, labels = axs[2].get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
axs[2].legend(handle_list, label_list,loc='upper right')

plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.savefig("Images/SIR_Stochastic_Uncontrolled_Multiple.png")



# Plot controlled system

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 8),sharex="col")

#plt.suptitle('Overall Title', fontsize=16)
axs[0].set_title("Controlled Target Cell System")
for i in range(NUM_TRAJECTORIES):
    axs[0].plot(t2s[i], [max(0, x) for x in X2s[i][:,0]],label="U: Uninfected Cells",color="green",alpha=ALPHA)

#axs[0].set_ylabel("Population")
handles, labels = axs[0].get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
axs[0].legend(handle_list, label_list,loc='upper right')

for i in range(NUM_TRAJECTORIES):
    axs[1].plot(t2s[i], [max(0, x) for x in X2s[i][:,1]],label="I: Infected Cells",color="blue",alpha=ALPHA)

#axs[1].set_ylabel("Population")
handles, labels = axs[1].get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
axs[1].legend(handle_list, label_list,loc='upper right')
# Create a new legend using the unique labels and handles
#axs[1].legend(unique_labels.values(), unique_labels.keys(),loc='upper right')


for i in range(NUM_TRAJECTORIES):
    axs[2].plot(t2s[i], [max(0, x) for x in X2s[i][:,2]],label="P: Virus",color="darkorange",alpha=ALPHA)

handles, labels = axs[2].get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
axs[2].legend(handle_list, label_list,loc='upper right')


for i in range(NUM_TRAJECTORIES):
    axs[3].plot(t2s[i], [max(0, x) for x in controls[i]],label="u: Control",color="red",alpha=ALPHA)

handles, labels = axs[3].get_legend_handles_labels()
handle_list, label_list = [], []
for handle, label in zip(handles, labels):
    if label not in label_list:
        handle_list.append(handle)
        label_list.append(label)
axs[3].legend(handle_list, label_list,loc='upper right')

axs[3].set_xlabel("t: Time")

plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.savefig("Images/TargetCellModel_Stochastic_Controlled_Multiple.png")
