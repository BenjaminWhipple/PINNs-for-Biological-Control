import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import numpy as np
import time

### Define constants
# Constants of architecture and training.
iterations = 200000

NETWORK_SIZE=400
BC_SAMPLES=500
PDE_SAMPLES=10000

INPUT_DIMENSIONS = 4 # 3 States + 1 Time Dimension

MIN_STATE = 0.0 
MAX_STATE = 5.0
T = 10.0

# Constants for the loss function
# Instantaneous costs
e1 = 5.0
e2 = 1.0

# Constants for the dynamics (nondimensionalized)

b = 0.75 
a = 1.0
k = 4.0
l = 1.0

# Initial Conditions
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
        inputs = torch.cat([U,I,V,t],axis=1)
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
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


net = Net(NETWORK_SIZE)
net = net.to(device)
mse_cost_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())

def f(U,I,V,t,net):
    S = net(U,I,V,t)
    S_U = torch.autograd.grad(S.sum(), U, create_graph=True)[0]
    S_I = torch.autograd.grad(S.sum(), I, create_graph=True)[0]
    S_V = torch.autograd.grad(S.sum(), V, create_graph=True)[0]
    S_t = torch.autograd.grad(S.sum(), t, create_graph=True)[0]
    
    S_UU = torch.autograd.grad(S_U.sum(),U,retain_graph=True)[0]
    S_II = torch.autograd.grad(S_I.sum(),I,retain_graph=True)[0]
    S_VV = torch.autograd.grad(S_V.sum(),V,retain_graph=True)[0]
    
    # PDE has e2 = 1.0 
    
    pde = S_t + e1*I - (S_I**2)*(1/4)*(I**2) + (S_U)*(-b*V*U) + (S_I)*(b*V*U-a*I) + (S_V)*(k*I-l*V) + 0.005*U*S_UU + 0.005*I*S_II + 0.005*V*S_VV
    return pde

## Data from Boundary Conditions

U_bc = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(BC_SAMPLES,1))
I_bc = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(BC_SAMPLES,1))
V_bc = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(BC_SAMPLES,1))
t_bc = np.full((BC_SAMPLES,1),T)

u_bc = np.zeros((BC_SAMPLES,1))


### Training
checkpoint_file = "bestmodel.pt"

previous_validation_loss = 99999999.0
best_loss = 99999999.0
start = time.time()
for epoch in range(iterations):
    optimizer.zero_grad()
    
    # Loss based on boundary conditions
    pt_U_bc = Variable(torch.from_numpy(U_bc).float(), requires_grad=False).to(device)
    pt_I_bc = Variable(torch.from_numpy(I_bc).float(), requires_grad=False).to(device)
    pt_V_bc = Variable(torch.from_numpy(V_bc).float(), requires_grad=False).to(device)
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
    
    net_bc_out = net(pt_U_bc, pt_I_bc, pt_V_bc, pt_t_bc)
    mse_u = mse_cost_function(net_bc_out, pt_u_bc)
    
    # Loss based on PDE
    U_collocation = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(PDE_SAMPLES,1))
    I_collocation = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(PDE_SAMPLES,1))
    V_collocation = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(PDE_SAMPLES,1))
    t_collocation = np.random.uniform(low=0.0, high=T, size=(PDE_SAMPLES,1))
    all_zeros = np.zeros((PDE_SAMPLES,1))
    
    pt_U_collocation = Variable(torch.from_numpy(U_collocation).float(), requires_grad=True).to(device)
    pt_I_collocation = Variable(torch.from_numpy(I_collocation).float(), requires_grad=True).to(device)
    pt_V_collocation = Variable(torch.from_numpy(V_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
    f_out = f(pt_U_collocation, pt_I_collocation, pt_V_collocation, pt_t_collocation, net)
    mse_f = mse_cost_function(f_out, pt_all_zeros)
    
    # Compute the total loss
    loss = mse_u + mse_f
    
    if loss.data < best_loss:
        best_loss = loss.data
        torch.save(net,checkpoint_file)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        with torch.autograd.no_grad():
        	print(f"{epoch}/{iterations}"," Training Loss:",loss.data, " Best: ", best_loss)

end = time.time()
print(f'Time taken: {end-start}')

net = torch.load(checkpoint_file)
net_bc_out = net(pt_U_bc, pt_I_bc, pt_V_bc, pt_t_bc)
mse_u = mse_cost_function(net_bc_out, pt_u_bc)
f_out = f(pt_U_collocation, pt_I_collocation, pt_V_collocation, pt_t_collocation, net)
mse_f = mse_cost_function(f_out, pt_all_zeros)
loss = mse_f+mse_u
print(loss.data)

torch.save(net,"TargetCellModel_Stochastic_Control_Value.pt")
