import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from Common import NeuralNet, MultiVariatePoly
import time
import os
import datetime
import pandas as pd

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)

class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        self.a_f = 0.05
        self.h_f = 5
        self.T_hot = 4
        self.U_f = 1
        self.a_s = 0.08
        self.h_s = 6
        self.T0 = 1

        # Extrema of the solution domain (t,x) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[0, 1],  # Time
                                            [0, 1]])  # Space

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=2,
                                              n_hidden_layers=4,
                                              neurons=40,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        '''self.approximate_solution = MultiVariatePoly(self.domain_extrema.shape[0], 3)'''

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # Initial condition to solve the heat equation u0(x)=-sin(pi x)
    def initial_condition_temporal(self, x):
        return torch.full_like(x, self.T0)

    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))

        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition_temporal(input_tb[:, 1])

        return input_tb, output_tb

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))
        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        output_sb_0 = torch.zeros((input_sb.shape[0], 1)).squeeze()
        output_sb_L = torch.zeros((input_sb.shape[0], 1)).squeeze()

        def tf(t, t0, t_hot, coeff):
            return (t_hot - t0) / (1 + torch.exp(coeff * (t - 0.25))) + t0

        # apply function for spatial boundary points for fluid
        output_sbf_0 = tf(input_sb_0[:, 0], self.T0, self.T_hot, -200)
        # output_sbf_0 = input_sb_0[:, 0]

        # concatenate
        input_sb = torch.cat([input_sb_0, input_sb_L], 0)
        output_sbs = torch.cat([output_sb_0, output_sb_L], 0)
        output_sbf = torch.cat([output_sbf_0, output_sb_L], 0)

        return input_sb, output_sbs, output_sbf

    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros(input_int.shape[0])
        return input_int, output_int

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sbs, output_sbf = self.add_spatial_boundary_points()  # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()         # S_int

        # batchsize seems quite large
        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sbs, output_sbf), batch_size=2 * self.space_dimensions * self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb,shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        u_pred_tb = self.approximate_solution(input_tb)
        u_pred_tbs = u_pred_tb[:, 1]
        u_pred_tbf = u_pred_tb[:, 0]
        return u_pred_tbs, u_pred_tbf

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        input_sb.requires_grad = True
        u_pred_sb = self.approximate_solution(input_sb)

        u_sbs = u_pred_sb[:, 1]
        u_sbf = u_pred_sb[:, 0]

        # gradient for the solid Temp. at the boundaries
        grad_u_sbs = torch.autograd.grad(u_sbs.sum(), input_sb, create_graph=True)[0]
        grad_u_sbs_x = grad_u_sbs[:, 1]

        # gradient for the fluid Temp. at the boundaries
        grad_u_sbfL = torch.autograd.grad(u_sbf.sum(), input_sb, create_graph=True)[0]
        grad_u_sbfL_x = grad_u_sbfL[:, 1]

        # Split the tensor and select the derivatives and the non derived BC
        u_sbf0 = u_sbf[:len(u_sbf) // 2]
        grad_u_sbfL_x = grad_u_sbfL_x[len(u_sbf) // 2:]
        t_x0 = input_sb[:len(u_sbf) // 2][:, 0]
        
        # transform the desired points for the NN output and concatenate
        # u_sbf0_T = (self.T_hot - self.T0) / (1 + torch.exp(-200 * (t_x0 - 0.25))) + self.T0 - u_sbf0

        # grad_u_sbf_x = torch.cat((u_sbf0_T, grad_u_sbfL_x), dim=0)
        grad_u_sbf_x = torch.cat((u_sbf0, grad_u_sbfL_x), dim=0)

        return grad_u_sbs_x, grad_u_sbf_x

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u_pred_int = self.approximate_solution(input_int)

        us = u_pred_int[:, 1]
        uf = u_pred_int[:, 0]

        # gradients for solid on the interior
        grad_u_ints = torch.autograd.grad(us.sum(), input_int, create_graph=True)[0]
        grad_us_t = grad_u_ints[:, 0]
        grad_us_x = grad_u_ints[:, 1]
        grad_us_xx = torch.autograd.grad(grad_us_x.sum(), input_int, create_graph=True)[0][:, 1]

        # gradients for fluid on the interior
        grad_u_intf = torch.autograd.grad(uf.sum(), input_int, create_graph=True)[0]
        grad_uf_t = grad_u_intf[:, 0]
        grad_uf_x = grad_u_intf[:, 1]
        grad_uf_xx = torch.autograd.grad(grad_uf_x.sum(), input_int, create_graph=True)[0][:, 1]

        residual_f = grad_uf_t + self.U_f * grad_uf_x - self.a_f * grad_uf_xx + self.h_f * (uf - us)
        residual_s = grad_us_t - self.a_s * grad_us_xx - self.h_s * (uf - us)

        return residual_s, residual_f

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sbs, u_train_sbf, inp_train_tb, u_train_tb, inp_train_int, verbose=True):

        u_pred_sbs, u_pred_sbf = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tbs, u_pred_tbf = self.apply_initial_condition(inp_train_tb)

        # u pred is the approx temp. and u train is the soll temp., here 0 temp., shape 1x128
        # u pred approx temp at space points at time t=0 and u train the sine output wished to achieve 1x64

        assert (u_pred_sbs.shape == u_train_sbs.shape)
        assert (u_pred_tbf.shape == u_train_tb.shape)
        assert (u_pred_sbf.shape == u_train_sbf.shape)
        assert (u_pred_tbs.shape == u_train_tb.shape)

        r_int_s, r_int_f = self.compute_pde_residual(inp_train_int)

        # residual on spatial boundary for solid and fluid
        r_sbs = u_train_sbs - u_pred_sbs  # [128,1]
        r_sbf = u_train_sbf - u_pred_sbf

        # residual on temporal boundary for solid and fluid
        r_tbs = u_train_tb - u_pred_tbs  # [64,1]
        r_tbf = u_train_tb - u_pred_tbf

        # loss spatial boundary for solid and fluid
        loss_sbs = torch.mean(abs(r_sbs) ** 2)
        loss_sbf = torch.mean(abs(r_sbf) ** 2)

        # loss temporal boundary for solid and fluid
        loss_tbs = torch.mean(abs(r_tbs) ** 2)
        loss_tbf = torch.mean(abs(r_tbf) ** 2)

        # loss interior for solid and fluid
        loss_int_s = torch.mean(abs(r_int_s) ** 2)
        loss_int_f = torch.mean(abs(r_int_f) ** 2)
        loss_int = loss_int_s + loss_int_f

        # loss on boundary conditions
        loss_u = loss_sbs + loss_tbs + loss_sbf + loss_tbf

        loss = torch.log10(self.lambda_u * (loss_sbs + loss_tbs + loss_sbf + loss_tbf) + loss_int)
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_u).item(), 4),
                          "| Function Loss: ", round(torch.log10(loss_int).item(), 4))

        # if verbose:
        #     print("Loss_sbs: ", round(loss_sbs.item(), 4),
        #           "| Loss_sbf: ", round(loss_sbf.item(), 4),
        #           "| Loss_tbs: ", round(loss_tbs.item(), 4),
        #           "| Loss_tbf: ", round(loss_tbf.item(), 4),
        #           "| Loss_int_s: ", round(loss_int_s.item(), 4),
        #           "| Loss_int_f: ", round(loss_int_f.item(), 4))

        return loss 

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb, u_train_sbs, u_train_sbf), (inp_train_tb, u_train_tb),
                    (inp_train_int, u_train_int)) in enumerate(
                    zip(self.training_set_sb, self.training_set_tb, self.training_set_int)):

                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sbs, u_train_sbf, inp_train_tb, u_train_tb, inp_train_int, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        # Generate timestamp for the current training run

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Save model state
        self.save_model(timestamp)

        return history, timestamp
    
    def save_model(self, timestamp):
        # Save the model state with timestamp
        model_path = "task1_model_{}.pt".format(timestamp)
        torch.save(self.approximate_solution.state_dict(), model_path)
        print("Model saved with timestamp: {}".format(timestamp))

    def load_model(self, model_filename):
        # Load the model state
        if os.path.exists(model_filename):
            self.approximate_solution.load_state_dict(torch.load(model_filename))
            print("Model loaded from: {}".format(model_filename))
        else:
            print("No model found at: {}".format(model_filename))


    def inference_and_save(self, data_filename, model_filename, output_filename):

        # Load the trained model
        self.load_model(model_filename)

        # Load your data
        data = pd.read_csv(data_filename)

        # Extract time and space values, assuming they are named 't' and 'x' in your data file
        t_values = torch.tensor(data['t'].values, dtype=torch.float32).reshape(-1, 1)
        x_values = torch.tensor(data['x'].values, dtype=torch.float32).reshape(-1, 1)

        # Concatenate t and x values as inputs
        inputs = torch.cat((t_values, x_values), dim=1)

        # Get the temperature predictions from your model
        outputs = self.approximate_solution(inputs)

        predictions_fluid = outputs[:, 0]  # fluid temperature predictions
        predictions_solid = outputs[:, 1]  # solid temperature predictions

        # Convert predictions to numpy
        tf_values = predictions_fluid.detach().numpy()
        ts_values = predictions_solid.detach().numpy()

        # Create a dataframe for saving
        output_df = pd.DataFrame({
            't': data['t'],
            'x': data['x'],
            'tf': tf_values.reshape(-1),  # reshape to match the shape of 't' and 'x'
            'ts': ts_values.reshape(-1)  # reshape to match the shape of 't' and 'x'
        })

        # Save to file
        output_df.to_csv(output_filename, index=False)
        print(f'Inference completed. Predictions saved in {output_filename}')

    ################################################################################################
    def plotting(self, timestamp=None):
        # Only load the model state if a timestamp is provided
        if timestamp is not None:
            model_path = timestamp
            self.approximate_solution.load_state_dict(torch.load(model_path))

        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        # Output I will have two for the network.. now do I have to plot it?
        # Do I need to get the exact solutions?

        # I guess here I can create the output of the dataset given, but what's the output dimension?
        outputs = self.approximate_solution(inputs)

        output_s = outputs[:, 1]
        output_f = outputs[:, 0]

        min_temp = min(torch.min(output_s), torch.min(output_f))
        max_temp = max(torch.max(output_s), torch.max(output_f))


        fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output_s.detach(), cmap="jet", s=1,
                             vmin=min_temp, vmax=max_temp)
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")

        im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output_f.detach(), cmap="jet", s=1,
                             vmin=min_temp, vmax=max_temp)
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")

        axs[0].set_title("solid")
        axs[1].set_title("fluid")
        
        
        # # Indices where x=0
        # idx_x0 = torch.where(inputs[:, 1] == 0)[0]
        #
        # # plot for x=0
        # fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        # ax.plot(inputs[idx_x0, 0].detach(), output_s[idx_x0].detach(), label="solid at x=0")
        # ax.plot(inputs[idx_x0, 0].detach(), output_f[idx_x0].detach(), label="fluid at x=0")
        # ax.set_xlabel("t")
        # ax.set_ylabel("Temperature")
        # ax.legend()
        # ax.grid(True, which="both", ls=":")

                
            

        plt.show()


# Instantiate Network and BP
n_int = 1024
n_sb = 512
n_tb = 512

pinn = Pinns(n_int, n_sb, n_tb)

input_sb_, output_sb_, output_sbf_ = pinn.add_spatial_boundary_points()
input_tb_, output_tb_ = pinn.add_temporal_boundary_points()
input_int_, output_int_ = pinn.add_interior_points()


# plt.figure(figsize=(16, 8), dpi=150)
#plt.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 0].detach().numpy(), label="Boundary Points")
#plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label="Interior Points")
#plt.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label="Initial Points")
#plt.xlabel("x")
#plt.ylabel("t")
#plt.legend()
# plt.show()


n_epochs = 1
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)

optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(),
                            lr=float(0.001))

# Train
# hist, timestamp = pinn.fit(num_epochs=n_epochs,
#                            optimizer=optimizer_LBFGS,
#                            verbose=True)



# train plot
# plt.figure(dpi=150)
# plt.grid(True, which="both", ls=":")
# plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
# plt.xscale("log")
# plt.legend()


hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)

# Plotting
pinn.plotting()
#pinn.plotting("task1_model_2023-06-14_23-41-03.pt")

#pinn.inference_and_save(data_filename='TestingData.txt',
#                         model_filename='task1_model_2023-06-14_23-41-03.pt',
#                         output_filename='Task1.txt')







#
#
#
# # Data Evaluation
# # Define path to your txt file
# path_TestingData = '/content/drive/MyDrive/DLSC/task1/TestingData.txt'
# path_SubExample = '/content/drive/MyDrive/DLSC/task1/SubExample.txt'
#
# # Load the text file into a NumPy array
# TestingData = np.loadtxt(path_TestingData, delimiter=',', skiprows=1)
# SubExample = np.loadtxt(path_SubExample, delimiter=',', skiprows=1)
#
# # Load the text file into a tensor
# TestingData = torch.from_numpy(TestingData)
# SubExample = torch.from_numpy(SubExample)
#
# # Print the tensor
# print(TestingData)
# print(SubExample)
