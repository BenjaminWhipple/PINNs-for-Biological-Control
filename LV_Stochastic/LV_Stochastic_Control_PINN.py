import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import numpy as np
import time

### Define many of the constants
#Performance:
iterations = 200000

NETWORK_SIZE=400
MIN_STATE = 0.0
MAX_STATE = 5.0
T=5.0
BC_SAMPLES=500
PDE_SAMPLES=10000

# Define our model
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

net = Net(NETWORK_SIZE)
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters())


## Define the PDE as loss function.
def f(x,y,t,net):
    u = net(x,y,t)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x.sum(),x,retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(),y,retain_graph=True)[0]
    
    # We use 1 for all parameters except for sigma, which we use 0.1 for.
    pde = u_t + x - (1/4)*u_y**2 + u_x*(x*(1-y)) + u_y*(y*(x-1))+ 0.005*x*u_xx +0.005*y*u_yy
    return pde

#BC: V(x,y,T)=0 for all x,y
x_bc = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(BC_SAMPLES,1))
y_bc = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(BC_SAMPLES,1))
t_bc = np.full((BC_SAMPLES,1),T)
u_bc = np.zeros((BC_SAMPLES,1))

### Training / Fitting

checkpoint_file = "bestmodel.pt"

previous_validation_loss = 99999999.0
best_loss = 99999999.0
start = time.time()
for epoch in range(iterations):
    optimizer.zero_grad() # to make the gradients zero
    
    # Loss based on boundary conditions
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    pt_y_bc = Variable(torch.from_numpy(y_bc).float(), requires_grad=False).to(device)
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
    
    net_bc_out = net(pt_x_bc, pt_y_bc, pt_t_bc)
    mse_u = mse_cost_function(net_bc_out, pt_u_bc)
    
    # Loss based on PDE
    x_collocation = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(PDE_SAMPLES,1))
    y_collocation = np.random.uniform(low=MIN_STATE, high=MAX_STATE, size=(PDE_SAMPLES,1))
    t_collocation = np.random.uniform(low=0.0, high=T, size=(PDE_SAMPLES,1))
    all_zeros = np.zeros((PDE_SAMPLES,1))
    
    
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
    f_out = f(pt_x_collocation, pt_y_collocation, pt_t_collocation, net)
    mse_f = mse_cost_function(f_out, pt_all_zeros)
    
    # Total loss function
    loss = mse_u + mse_f
    
    if loss.data < best_loss:
        best_loss = loss.data
        torch.save(net,"bestmodel.pt")
    
    loss.backward() # Computing gradients
    optimizer.step() # ADAM optimization step.

    with torch.autograd.no_grad():
    	print(epoch,"Training Loss:",loss.data)

end = time.time()

net = torch.load("bestmodel.pt")
net_bc_out = net(pt_x_bc, pt_y_bc, pt_t_bc)
mse_u = mse_cost_function(net_bc_out, pt_u_bc)
f_out = f(pt_x_collocation, pt_y_collocation, pt_t_collocation, net) # output of f(x,t)
mse_f = mse_cost_function(f_out, pt_all_zeros)
loss = mse_f+mse_u
print(loss.data)

print(f'Time taken: {end-start}')

torch.save(net,"LV_Stochastic_Control_Value.pt")
