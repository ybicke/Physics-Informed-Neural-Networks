
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd


import os
import datetime

from Common import NeuralNet, MultiVariatePoly

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available")
    
    
# Class for Pinns        
class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        self.a_f = 0.005
        self.h_f = 5
        self.T_hot = 4
        self.T0 = 1
        self.T_cold = 1

        # Extrema of the solution domain (t,x) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[0, 8],  # Time dimension
                                            [0, 1]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        # FF Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                              n_hidden_layers=4,
                                              neurons=80,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)

        # FF Dense NN to approximate the conductivity we wish to infer
        self.approximate_solid = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                           n_hidden_layers=4,
                                           neurons=80,
                                           regularization_param=0.,
                                           regularization_exp=2.,
                                           retrain_seed=42)

        # Generator of soboleng sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # Initial condition to solve the heat equation
    def initial_condition(self, x):
        return torch.full_like(x, self.T0)

    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition(input_tb[:, 1])

        return input_tb, output_tb

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        
       # create spatial boundary points for time intervals from 0 to 8
       # Extract the time and space domain bounds
        t0, tL = self.domain_extrema[0]
        x0, xL = self.domain_extrema[1]
        
        # Create domain ranges for the time of the different phases
        domain_range = [(i, i+ 1) for i in range(int(t0), int(tL))]

        tensor_list = []
        for interval_range in domain_range:
            interval_size = interval_range[1] - interval_range[0]
            interval_points = self.soboleng.draw(self.n_sb) * interval_size + interval_range[0]
            tensor_list.append(interval_points)

        # configure the spatial domains to 0 and 1
        def tensor_expansion(tensor_list, x0, xL):
            expanded_tensor_list = []
            for tensor in tensor_list:
                tensor_copy = tensor.clone()
                tensor[:, 1] = x0
                tensor_copy[:, 1] = xL
                expanded_tensor_list.append(tensor)
                expanded_tensor_list.append(tensor_copy)
            return torch.cat(expanded_tensor_list, dim=0)

        # create the new input tensor
        input_new = tensor_expansion(tensor_list, x0, xL)

        # create the output tensors
        output_sb_0 = torch.zeros(self.n_sb)
        output_sb_L = torch.zeros(self.n_sb)

        # charge phase
        output_sb_charge = torch.full((self.n_sb,), self.T_hot)
        output_charge = torch.cat([output_sb_charge, output_sb_L], 0)

        # Idle phase
        output_idle = torch.cat([output_sb_0, output_sb_L], 0)

        # discharge phase
        output_sb_discharge = torch.full((self.n_sb,), self.T_cold)
        output_discharge = torch.cat([output_sb_0, output_sb_discharge], 0)

        output_new = torch.cat([output_charge, output_idle,
                                output_discharge, output_idle,
                                output_charge, output_idle,
                                output_discharge, output_idle, ], 0)

        return input_new, output_new

    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros(input_int.shape[0])
        return input_int, output_int

    def get_measurement_data(self):
        path_to_txt = 'DataSolution.txt'
        data_array = np.loadtxt(path_to_txt, delimiter=',', skiprows=1)
        data_tensor = torch.from_numpy(data_array)

        # Split the tensor into two tensors
        input_meas = data_tensor[:, :2].to(torch.float32)
        output_meas = data_tensor[:, 2:].to(torch.float32).squeeze()

        return input_meas, output_meas

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()  # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()  # S_int

        # Here I have to be careful with the dataloader It is set up previously in a way to return twice the sice of the n_sb samples
        # Initially we had 64 and extended to two. Now I extended the spatial boundary point again times 8.
        # I have n_sb = 64 and space_dimension = 1, then batch_size = 2 * space_dimensions * n_sb = 128, so I have 8 batches of size 128 for training.  
        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=2 * 8 * self.space_dimensions * self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        u_pred_tb = self.approximate_solution(input_tb).squeeze()
        return u_pred_tb

    def idx_by_phase_uf(self, input_sb):
        # select values for the different phases 
        idx_charge = ((input_sb[:, 0] >= 0) & (input_sb[:, 0] <= 1)) | ((input_sb[:, 0] >= 4) & (input_sb[:, 0] <= 5))
        idx_idle1 = ((input_sb[:, 0] >= 1) & (input_sb[:, 0] <= 2)) | ((input_sb[:, 0] >= 3) & (input_sb[:, 0] <= 4))
        idx_discharge = ((input_sb[:, 0] >= 2) & (input_sb[:, 0] <= 3)) | ((input_sb[:, 0] >= 6) & (input_sb[:, 0] <= 7))
        idx_idle2 = ((input_sb[:, 0] >= 5) & (input_sb[:, 0] <= 6)) | ((input_sb[:, 0] >= 7) & (input_sb[:, 0] <= 8))

        charge = torch.where(idx_charge)[0]
        idle1 = torch.where(idx_idle1)[0]
        idle2 = torch.where(idx_idle2)[0]
        discharge = torch.where(idx_discharge)[0]
        idle = torch.cat([idle1, idle2], dim=0)

        return charge, idle, discharge

    def idx_by_phase_grad(self, input_int):
        # select values for the different phases
        idx_charge = ((input_int[:, 0] >= 0) & (input_int[:, 0] <= 1)) | ((input_int[:, 0] >= 4) & (input_int[:, 0] <= 5))
        idx_idle1 = ((input_int[:, 0] >= 1) & (input_int[:, 0] <= 2)) | ((input_int[:, 0] >= 3) & (input_int[:, 0] <= 4))
        idx_discharge = ((input_int[:, 0] >= 2) & (input_int[:, 0] <= 3)) | ((input_int[:, 0] >= 6) & (input_int[:, 0] <= 7))
        idx_idle2 = ((input_int[:, 0] >= 5) & (input_int[:, 0] <= 6)) | ((input_int[:, 0] >= 7) & (input_int[:, 0] <= 8))

        idle1 = torch.where(idx_idle1)[0]
        idle2 = torch.where(idx_idle2)[0]
        idle = torch.cat([idle1, idle2], dim=0)

        idx_charge_1 = idx_charge & (input_int[:, 1] == 1)
        idx_discharge_0 = idx_discharge & (input_int[:, 1] == 0)
        charge_1 = torch.where(idx_charge_1)[0]
        discharge_0 = torch.where(idx_discharge_0)[0]

        return charge_1, idle, discharge_0
    
    def uf_tensor(self, input_int):
        charge, idle, discharge = self.idx_by_phase_uf(input_int)
        all_idx = torch.cat([charge, idle, discharge], 0)
        num_rows = len(all_idx)

        tensor = torch.zeros(num_rows)
        tensor[charge] = 1
        tensor[idle] = 0
        tensor[discharge] = -1

        return tensor.squeeze()

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        input_sb.requires_grad = True
        u_pred_sb = self.approximate_solution(input_sb).squeeze()
        charge_1, idle, discharge_0 = self.idx_by_phase_grad(input_sb)

        # Compute gradients of u_pred_sb with respect to input_sb
        grad_u_sb = torch.autograd.grad(u_pred_sb.sum(), input_sb, create_graph=True)[0]
        grad_u_sb_x = grad_u_sb[:, 1]

        # Create a new tensor and fill it with the original u_pred_sb values
        u_pred_sb_derived = u_pred_sb.clone()

        # Replace the selected indices with the derived values
        u_pred_sb_derived[charge_1] = grad_u_sb_x[charge_1]
        u_pred_sb_derived[idle] = grad_u_sb_x[idle]
        u_pred_sb_derived[discharge_0] = grad_u_sb_x[discharge_0]

        return u_pred_sb_derived

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True

        uf = self.approximate_solution(input_int).squeeze()
        us = self.approximate_solid(input_int).squeeze()

        grad_u = torch.autograd.grad(uf.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 1]

        U_f = self.uf_tensor(input_int)

        residual_f = grad_u_t + U_f * grad_u_x - self.a_f * grad_u_xx + self.h_f * (uf - us)

        return residual_f

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=True):

        u_pred_sb = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)
        inp_train_meas, u_train_meas = self.get_measurement_data()
        u_pred_meas = self.approximate_solution(inp_train_meas).squeeze()

        assert (u_pred_sb.shape == u_train_sb.shape)
        assert (u_pred_tb.shape == u_train_tb.shape)
        assert (u_pred_meas.shape == u_train_meas.shape)

        r_int = self.compute_pde_residual(inp_train_int)
        r_sb = u_train_sb - u_pred_sb
        r_tb = u_train_tb - u_pred_tb
        r_meas = u_train_meas - u_pred_meas

        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)
        loss_meas = torch.mean(abs(r_meas) ** 2)

        loss_u = loss_sb + loss_tb + loss_meas

        loss = torch.log10(self.lambda_u * loss_u + loss_int)

        log_losses = {"loss_sb": torch.log10(loss_sb).item(),
                      "loss_tb": torch.log10(loss_tb).item(),
                      "loss_int": torch.log10(loss_int).item(),
                      "loss_meas": torch.log10(loss_meas).item()}

        if verbose:
            print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_int).item(), 4),
                  "| Function Loss: ", round(torch.log10(loss_u).item(), 4), "| Individual log losses:", {k: round(v, 4) for k, v in log_losses.items()})

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        self.history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) in enumerate(
                    zip(self.training_set_sb, self.training_set_tb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int,
                                             verbose=verbose)
                    loss.backward()

                    self.history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', self.history[-1])
        self.save_model()  # calling save_model within fit
        return self.history 


    def save_model(self, filename=None):
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = "task2_model_finals_{}.pt".format(timestamp)
        
        # Save model state and final loss
        model_info = {
            'model_state_dict': {
                'approximate_solution': self.approximate_solution.state_dict(),
                'approximate_solid': self.approximate_solid.state_dict()
            },
            'final_loss': self.history[-1]  # You might need to find a way to access the final loss here
        }
        torch.save(model_info, filename)

        # Print out the filename of the saved model
        print("Model saved as:", filename)



    def load_model(self, model_filename):
        model_dict = torch.load(model_filename)

        # Check if 'model_state_dict' key exists
        if "model_state_dict" in model_dict:
            # Check if 'approximate_solution' key exists
            if 'approximate_solution' in model_dict['model_state_dict']:
                self.approximate_solution.load_state_dict(model_dict['model_state_dict']['approximate_solution'])
            # Check if 'approximate_solid' key exists
            if 'approximate_solid' in model_dict['model_state_dict']:
                self.approximate_solid.load_state_dict(model_dict['model_state_dict']['approximate_solid'])
            print("Models loaded")
        else:
            print("No 'model_state_dict' key found in the file")

        # Check if 'final_loss' key exists
        if 'final_loss' in model_dict:
            print("Final loss from the loaded model: ", model_dict['final_loss'])
        else:
            print("No 'final_loss' key found in the file")

            
    
    
    def inference_and_save(self, data_filename, model_filename, output_filename):

        # Load the trained model
        self.load_model(model_filename)

        # Load your data
        data = pd.read_csv(data_filename) 

        # Extract time and space values, assuming they are named 't' and 'x' in your data file
        t_values = torch.tensor(data['t'].values, dtype=torch.float32).reshape(-1,1)
        x_values = torch.tensor(data['x'].values, dtype=torch.float32).reshape(-1,1)

        # Concatenate t and x values as inputs
        inputs = torch.cat((t_values, x_values), dim=1)

        # Get the solid temperature predictions from your model
        solid_temperature_predictions = self.approximate_solid(inputs)

        # Convert predictions to numpy
        ts_values = solid_temperature_predictions.detach().numpy()

        # Create a dataframe for saving
        output_df = pd.DataFrame({
            't': data['t'],
            'x': data['x'],
            'ts': ts_values.reshape(-1)  # reshape to match the shape of 't' and 'x'
        })

        # Save to file
        output_df.to_csv(output_filename, index=False)
        print(f"Inference data saved to {output_filename}")




    ################################################################################################
    def plotting(self,  model_path=None):
        
        # If a model path is given, load the model
        if model_path is not None:
            self.load_model(model_path)
        
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        solution = self.approximate_solution(inputs).reshape(-1, )

        # Create the first plot
        fig1 = plt.figure(figsize=(12, 8), dpi=100)
        plt.scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=solution.detach(), cmap="jet", s=4)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.colorbar()
        plt.grid(True, which="both", ls=":")
        plt.title("Approximate fluid")

        solid = self.approximate_solid(inputs).reshape(-1, )

        # Create the second plot
        fig2 = plt.figure(figsize=(12, 8), dpi=100)
        plt.scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=solid.detach(), cmap="jet",s=4)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.colorbar()
        plt.grid(True, which="both", ls=":")
        plt.title("Approximate solid")

        plt.show()


# initialize boundary points and pinn's class
n_int = 2048
# since I duplicate the points I get them with a factor of 8 (4 times 2 cycles)
n_sb = 256
n_tb = 256 * 8

pinn = Pinns(n_int, n_sb, n_tb)

# Plot the input training points
input_sb_, output_sb_ = pinn.add_spatial_boundary_points()
input_tb_, output_tb_ = pinn.add_temporal_boundary_points()
input_int_, output_int_ = pinn.add_interior_points()
input_meas_, output_meas_ = pinn.get_measurement_data()

# # plt.figure(figsize=(16, 8), dpi=150)
# plt.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 0].detach().numpy(), label="Boundary Points")
# plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label="Interior Points")
# plt.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label="Initial Points")
# plt.scatter(input_meas_[:, 1].detach().numpy(), input_meas_[:, 0].detach().numpy(), label="Sensors Points", marker="*")
# plt.xlabel("x")
# plt.ylabel("t")
# plt.legend()
# plt.show()

n_epochs = 1
optimizer_LBFGS = optim.LBFGS(
    list(pinn.approximate_solution.parameters()) + list(pinn.approximate_solid.parameters()),
    lr=float(0.5),
    max_iter=70000,
    max_eval=70000,
    history_size=150,
    line_search_fn="strong_wolfe",
    tolerance_change=1.0 * np.finfo(float).eps)
optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(),
                            lr=float(0.0005))

hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)

#plt.figure(dpi=150)
#plt.grid(True, which="both", ls=":")
#plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
#plt.xscale("log")
#plt.legend()

pinn.plotting()
#pinn.plotting('task2_4096_finals_2023-06-15_00-55-49.pt')
#pinn.inference_and_save('DataSolution.txt', 'task2_model_2023-06-14_16-20-28.pt', 'Task2!.txt')

