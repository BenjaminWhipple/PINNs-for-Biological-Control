import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import numpy as np
import time

### Define many of the constants
#Performance:
iterations = 20000

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

net = Net(NETWORK_SIZE)
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters())

## Define the PDE as loss function.
def f(S,I,R,t,net):
    u = net(S,I,R,t)
    u_S = torch.autograd.grad(u.sum(), S, create_graph=True)[0]
    u_I = torch.autograd.grad(u.sum(), I, create_graph=True)[0]
    u_R = torch.autograd.grad(u.sum(), R, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    
    u_SS = torch.autograd.grad(u_S.sum(),S,retain_graph=True)[0]
    u_SI = torch.autograd.grad(u_S.sum(),I,retain_graph=True)[0]
    u_II = torch.autograd.grad(u_I.sum(),I,retain_graph=True)[0]
    u_RI = torch.autograd.grad(u_R.sum(),I,retain_graph=True)[0]
    u_RR = torch.autograd.grad(u_R.sum(),R,retain_graph=True)[0]
    
    u_star = -(1/2)*((u_S - u_I)*beta*S*I)
    # We use 1 for all parameters except for sigma, which we use 0.1 for.
    # pde = u_t + x - (1/4)*u_y**2 + u_x*(x*(1-y)) + u_y*(y*(x-1))+ 0.005*x*u_xx +0.005*y*u_yy
    u_star_c = torch.clamp(u_star, u_min, u_max)
    
    # Define the various components of the HJB pde (so as to avoid an especially long expression).
    Integral_Cost = c1*u_star_c**2 + c2*I 
    
    u_S_term = u_S*(-(1-u_star_c)*beta*S*I)
    u_I_term = u_I*((1-u_star_c)*beta*S*I - gamma*I)
    u_R_term = u_R*(gamma*I)
    
    u_SS_term = u_SS*((sigma_beta*S*I)**2 + (sigma_S*I)**2)
    u_SI_term = 2*u_SI*(-(sigma_beta*S*I)**2)
    u_II_term = u_II*((sigma_beta*S*I)**2 + (sigma_gamma*I)**2 + (sigma_I*I)**2)
    u_RI_term = 2*u_RI*(-(sigma_gamma*I)**2)
    u_RR_term = u_RR*((sigma_gamma*I)**2 + (sigma_R*R)**2)
    
    pde = u_t + Integral_Cost + u_S_term + u_I_term + u_R_term + (1/2)*(u_SS_term + u_SI_term + u_II_term + u_RI_term + u_RR_term)

    # pde = u_t 
    return pde

#BC: V(x,y,T)=0 for all x,y
S_bc = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(BC_SAMPLES,1))
I_bc = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(BC_SAMPLES,1))
R_bc = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(BC_SAMPLES,1))
t_bc = np.full((BC_SAMPLES,1),T)
u_bc = np.zeros((BC_SAMPLES,1))

### Training / Fitting

checkpoint_file = "bestmodel.pt"

previous_validation_loss = 99999999.0
best_loss = 99999999.0

best_losses = []
loss_epochs = []

loss_checkpoints = [1e0,1e-1,1e-2,1e-3,1e-4]
current_loss_checkpoint_index = 0
loss_checkpoint_files = ["bestmodel_1e0.pt","bestmodel_1e-1.pt","bestmodel_1e-2.pt","bestmodel_1e-3.pt","bestmodel_1e-4.pt"]

start = time.time()
for epoch in range(iterations):
    optimizer.zero_grad() # to make the gradients zero
    
    # Loss based on boundary conditions
    pt_S_bc = Variable(torch.from_numpy(S_bc).float(), requires_grad=False).to(device)
    pt_I_bc = Variable(torch.from_numpy(I_bc).float(), requires_grad=False).to(device)
    pt_R_bc = Variable(torch.from_numpy(R_bc).float(), requires_grad=False).to(device)
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
    
    net_bc_out = net(pt_S_bc, pt_I_bc, pt_R_bc, pt_t_bc)
    mse_u = mse_cost_function(net_bc_out, pt_u_bc)
    
    # Loss based on PDE
    S_collocation = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(PDE_SAMPLES,1))
    I_collocation = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(PDE_SAMPLES,1))
    R_collocation = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(PDE_SAMPLES,1))
    t_collocation = np.random.uniform(low=0.0, high=T, size=(PDE_SAMPLES,1))
    all_zeros = np.zeros((PDE_SAMPLES,1))
    
    
    pt_S_collocation = Variable(torch.from_numpy(S_collocation).float(), requires_grad=True).to(device)
    pt_I_collocation = Variable(torch.from_numpy(I_collocation).float(), requires_grad=True).to(device)
    pt_R_collocation = Variable(torch.from_numpy(R_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
    f_out = f(pt_S_collocation, pt_I_collocation, pt_R_collocation, pt_t_collocation, net)
    mse_f = mse_cost_function(f_out, pt_all_zeros)
    
    # Total loss function
    loss = mse_u + mse_f
    
    if loss.data < best_loss:
        best_loss = loss.data.to("cpu")
        torch.save(net,"Checkpoints/"+"bestmodel.pt")
    
    loss.backward() # Computing gradients
    optimizer.step() # ADAM optimization step.
    
    if best_loss < loss_checkpoints[current_loss_checkpoint_index]:
        torch.save(net,"Checkpoints/"+loss_checkpoint_files[current_loss_checkpoint_index])
        current_loss_checkpoint_index += 1
    
    if epoch % 100 == 0:
        best_losses.append(best_loss)
        loss_epochs.append(epoch)
        with torch.autograd.no_grad():
        	print(epoch,"Training Loss:",loss.data)

end = time.time()

np.savetxt("losses.txt",best_losses)
np.savetxt("epochs.txt",loss_epochs)

net = torch.load("Checkpoints/"+"bestmodel.pt")
net_bc_out = net(pt_S_bc, pt_I_bc, pt_R_bc, pt_t_bc)
mse_u = mse_cost_function(net_bc_out, pt_u_bc)
f_out = f(pt_S_collocation, pt_I_collocation, pt_R_collocation, pt_t_collocation, net) # output of f(x,t)
mse_f = mse_cost_function(f_out, pt_all_zeros)
loss = mse_f+mse_u
print(loss.data)

print(f'Time taken: {end-start}')

torch.save(net,"Checkpoints/"+"LV_Stochastic_Control_Value.pt")

