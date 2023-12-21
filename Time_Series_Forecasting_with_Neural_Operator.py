import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



from tqdm import tqdm

def activation(name):
    # Given a string input return the appropriate activation function
    # If the string doesn't match any known activations, raise a ValueError
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['elu']:
        return nn.ELU()
    elif name in ['mish']:
        return nn.Mish()
    else:
        raise ValueError('Unknown activation function')


################################################################
# Define a 1d Spectral Convolution Layer
################################################################
class SpectralConv1d(nn.Module):
    # This class performs the Fourier transformation, linear transformation, and then inverse Fourier transformation.
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        
        # The weights for the linear transformation
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Function for complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    ######################### TO DO ####################################

    def forward(self, x):
        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]
        # hint: use torch.fft library torch.fft.rfft
        # use DFT to approximate the fourier transform

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Initialize tensor for output Fourier transform
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


####################################################################


# Define a 1D Fourier Neural Operator
class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
                The overall network. It contains 4 layers of the Fourier layer.
                1. Lift the input to the desire channel dimension by self.fc0 .
                2. 4 layers of the integral operators u' = (W + K)(u).
                    W defined by self.w; K defined by self.conv .
                3. Project from the channel space to the output space by self.fc1 and self.fc2 .

                input: the solution of the initial condition and location (a(x), x)
                input shape: (batchsize, x=s, c=2)
                output: the solution of a later timestep
                output shape: (batchsize, x=s, c=1)
                """

        # Define the number of modes and the width of the network
        self.modes1 = modes
        self.width = width
        self.padding = 1  # pad the domain if input is non-periodic

        # Initial linear layer used to lift the input to the desired channel dimension
        self.linear_p = nn.Linear(2, self.width)  # input channel is 2: (u0(x), x)

        # Define the Spectral Convolution layers
        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)

        # Define the convolution layers used within the Fourier layers
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        # Define a linear layer and output layer for final transformations
        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)  # Output 2 features

        # Activation function to use within Fourier layers and after linear transformation
        self.activation = torch.nn.Tanh()

    # Define a function for a Fourier layer
    def fourier_layer(self, x, spectral_layer, conv_layer):
        # Apply spectral convolution and normal convolution, and then the activation function
        return self.activation(spectral_layer(x) + conv_layer(x))

    # Define a function for a linear layer
    def linear_layer(self, x, linear_transformation):
        # Apply linear transformation and then the activation function
        return self.activation(linear_transformation(x))

    # Define forward propagation function
    def forward(self, x):
        # Apply initial linear layer
        x = self.linear_p(x)
        # Permute the tensor for the Conv1d layers
        x = x.permute(0, 2, 1)

        # Apply Fourier layers
        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.fourier_layer(x, self.spect3, self.lin2)

        # Permute the tensor back for the Linear layers
        x = x.permute(0, 2, 1)

        # Apply final linear transformation and output layer
        x = self.linear_layer(x, self.linear_q)
        x = self.output_layer(x)

        return x

class DataPreprocessor:
    def __init__(self, train_file, test_file, window_size, target_size, batch_size):
        self.train_file = train_file
        self.test_file = test_file
        self.window_size = window_size
        self.target_size = target_size
        self.batch_size = batch_size
        self.scaler_t = None
        self.scaler_tf0 = None
        self.scaler_ts0 = None


    def _load_and_preprocess_data(self, file, train=True):
        # Load data
        data = pd.read_csv(file, delimiter=',')

        # Normalize features if it's training data
        if train:
            self.scaler_t = MinMaxScaler().fit(data[['t']])
            self.scaler_tf0 = MinMaxScaler().fit(data[['tf0']])
            self.scaler_ts0 = MinMaxScaler().fit(data[['ts0']])

        data_normalized = data.copy()
        data_normalized['t'] = self.scaler_t.transform(data[['t']])

        # If it's training data, also normalize the target columns
        if train:
            data_normalized['tf0'] = self.scaler_tf0.transform(data[['tf0']])
            data_normalized['ts0'] = self.scaler_ts0.transform(data[['ts0']])

            # Initialize lists to hold the input and target sequences
            inputs1, inputs2, targets1, targets2 = [], [], [], []

            for i in range(self.window_size, len(data_normalized) - self.target_size + 1):
                # The input sequence for the first network consists of the time and solid temperature
                input_seq1 = data_normalized[['ts0', 't']].values[i - self.window_size:i]
                input_seq2 = data_normalized[['tf0', 't']].values[i - self.window_size:i]

                target_seq1 = data_normalized[['ts0', 't']].values[i:i + self.target_size]
                target_seq2 = data_normalized[['tf0', 't']].values[i:i + self.target_size]

                # Append the input and target sequences to the respective lists
                inputs1.append(input_seq1)
                inputs2.append(input_seq2)
                targets1.append(target_seq1)
                targets2.append(target_seq2)

            # Convert the lists into tensors and reshape them to match the model's expected input and output shape
            inputs_tensor1 = torch.tensor(np.array(inputs1), dtype=torch.float32).view(-1, self.window_size, 2)
            inputs_tensor2 = torch.tensor(np.array(inputs2), dtype=torch.float32).view(-1, self.window_size, 2)
            targets_tensor1 = torch.tensor(np.array(targets1), dtype=torch.float32).view(-1, self.target_size, 2)
            targets_tensor2 = torch.tensor(np.array(targets2), dtype=torch.float32).view(-1, self.target_size, 2)

            return inputs_tensor1, targets_tensor1, inputs_tensor2, targets_tensor2, data_normalized

    def load_data(self):
        """
        In the train_dataloader with batch_size=1, it will divide the input tensor of e.g. size [196, 10, 2]
        into 196 batches of size [1, 10, 2]. This means that each batch will contain one sequence of length 10 with 2 features.
        """

        # Load and preprocess training data
        train_inputs1, train_targets1, train_inputs2, train_targets2, train_normalized_data = self._load_and_preprocess_data(
            self.train_file, train=True)

        # Create DataLoader for training set
        train_dataset1 = TensorDataset(train_inputs1, train_targets1)
        train_dataloader1 = DataLoader(train_dataset1, batch_size=self.batch_size, shuffle=False)

        train_dataset2 = TensorDataset(train_inputs2, train_targets2)
        train_dataloader2 = DataLoader(train_dataset2, batch_size=self.batch_size, shuffle=False)

        # Load and preprocess testing data
        # test_inputs1, test_inputs2, _ = self._load_and_preprocess_data(self.test_file, train=False)

        # Create DataLoader for testing set
        # test_dataset1 = TensorDataset(test_inputs1)
        # test_dataloader1 = DataLoader(test_dataset1, batch_size=self.batch_size, shuffle=False)

        # test_dataset2 = TensorDataset(test_inputs2)
        # test_dataloader2 = DataLoader(test_dataset2, batch_size=self.batch_size, shuffle=False)

        return train_dataloader1, train_inputs1, train_dataloader2, train_inputs2, train_normalized_data




# Module for plotting
def plot_data(train_dataloader):
    x, y = next(iter(train_dataloader))
    print(x.shape, y.shape)

    plt.figure(figsize=(10, 6))
    plt.plot(y[0, :, 0].numpy(),
             label='Fluid Temperature')  # Plotting the first target of the first batch (fluid temperature)
    plt.plot(y[0, :, 1].numpy(),
             label='Solid Temperature')  # Plotting the second target of the first batch (solid temperature)
    plt.legend()
    plt.show()


def train_model(fno1, fno2, train_dataloader1, train_dataloader2, optimizer1, optimizer2, scheduler1, scheduler2, device, epochs,
                test_dataloader=None):
    freq_print = 1
    l = torch.nn.MSELoss()

    for epoch in range(epochs):
        fno1.train()  # Set the first model to training mode
        fno2.train()  # Set the second model to training mode
        train_mse1 = 0.0
        train_mse2 = 0.0

        # Zip the two dataloaders together
        zip_dataloader = zip(train_dataloader1, train_dataloader2)

        for step, ((input_batch1, output_batch1), (input_batch2, output_batch2)) in enumerate(zip_dataloader):

            # Move batches to the correct device
            input_batch1, output_batch1 = input_batch1.to(device), output_batch1.to(device)
            input_batch2, output_batch2 = input_batch2.to(device), output_batch2.to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            output_batch1 = output_batch1[..., 0:1]
            output_batch2 = output_batch2[..., 0:1]

            output_pred_batch1 = fno1(input_batch1)#.squeeze(2)
            output_pred_batch2 = fno2(input_batch2)#.squeeze(2)


            loss1 = l(output_pred_batch1, output_batch1)
            loss2 = l(output_pred_batch2, output_batch2)

            # You might want to balance these losses, depending on your specific task
            # loss = loss1 + loss2

            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()

            train_mse1 += loss1.item()
            train_mse2 += loss2.item()

        train_mse1 /= len(train_dataloader1)
        train_mse2 /= len(train_dataloader2)

        # print the loss at each epoch
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss fno1: {train_mse1}, Training Loss fno2: {train_mse2}")

        scheduler1.step()
        scheduler2.step()


def recursive_forecasting(fno1, fno2, last_known_sequence1, last_known_sequence2, forecast_length, device):
    fno1.eval()  # Set the first model to evaluation mode
    fno2.eval()  # Set the second model to evaluation mode

    forecast1 = []
    forecast2 = []
    input_sequence1 = last_known_sequence1.to(device)
    input_sequence2 = last_known_sequence2.to(device)

    with torch.no_grad():
        for i in range(forecast_length):
            # Get the networks' prediction for the next time step
            next_step_prediction1 = fno1(input_sequence1)
            next_step_prediction2 = fno2(input_sequence2)

            # Print the current prediction step
            print(f"Predicting step {i + 1}")

            # Select only the next time step prediction, ensuring correct dimensions
            next_step_prediction1 = next_step_prediction1[:, -1, :].unsqueeze(1)
            next_step_prediction2 = next_step_prediction2[:, -1, :].unsqueeze(1)

            # Append the prediction to the forecasts
            forecast1.append(next_step_prediction1.cpu().numpy())
            forecast2.append(next_step_prediction2.cpu().numpy())

            # Get the time dimension from the last known input sequence
            time_sequence1 = input_sequence1[:, -1, 1:].unsqueeze(-1).to(device)
            time_sequence2 = input_sequence2[:, -1, 1:].unsqueeze(-1).to(device)

            # Concatenate the time dimension to the predictions
            next_step_prediction1_with_time = torch.cat((next_step_prediction1, time_sequence1), dim=-1)
            next_step_prediction2_with_time = torch.cat((next_step_prediction2, time_sequence2), dim=-1)

            # Prepare the inputs for the next prediction
            input_sequence1 = torch.cat((input_sequence1[:, 1:, :], next_step_prediction1_with_time), axis=1)
            input_sequence2 = torch.cat((input_sequence2[:, 1:, :], next_step_prediction2_with_time), axis=1)


        # Convert the lists of arrays into single 2D arrays
        forecast1 = np.concatenate(forecast1, axis=0)
        forecast2 = np.concatenate(forecast2, axis=0)

        forecast1 = np.squeeze(forecast1, axis=1)
        forecast2 = np.squeeze(forecast2, axis=1)

    return forecast1, forecast2


def plot_temperatures(original_data, forecast1, forecast2, data_preprocessor1):
    # Extract the original data
    timestamps = np.array(original_data['t'])
    original_fluid_temperatures = np.array(original_data['tf0'])
    original_solid_temperatures = np.array(original_data['ts0'])

    # Calculate the forecast lengths
    forecast_length1 = len(forecast1)
    forecast_length2 = len(forecast2)

    # Convert the forecasts back to the original scale
    forecast_inverse_transformed1 = data_preprocessor1.scaler_tf0.inverse_transform(forecast1)
    forecast_inverse_transformed2 = data_preprocessor1.scaler_ts0.inverse_transform(forecast2)

    # Create new timestamp arrays for the forecasted data from original
    time_step = timestamps[1] - timestamps[0]  # Time step between each data point

    forecast_timestamps1 = np.arange(timestamps[-1] + time_step, timestamps[-1] + time_step * forecast_length1, time_step)
    forecast_timestamps1 = np.append(forecast_timestamps1, forecast_timestamps1[-1] + time_step)

    forecast_timestamps2 = np.arange(timestamps[-1] + time_step, timestamps[-1] + time_step * forecast_length2, time_step)
    forecast_timestamps2 = np.append(forecast_timestamps2, forecast_timestamps2[-1] + time_step)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2)

    # Plot the original fluid temperature and its forecast
    axs[0].plot(timestamps, original_fluid_temperatures, label='Original fluid temperature')
    axs[0].plot(forecast_timestamps1, forecast_inverse_transformed1, label='Forecast fluid temperature')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Fluid temperature')
    axs[0].legend()

    # Plot the original solid temperature and its forecast
    axs[1].plot(timestamps, original_solid_temperatures, label='Original solid temperature')
    axs[1].plot(forecast_timestamps2, forecast_inverse_transformed2, label='Forecast solid temperature')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Solid temperature')
    axs[1].legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


    # Save the forecasts into a single CSV file
    forecast_df = pd.DataFrame({
        't': forecast_timestamps1,
        'tf0': forecast_inverse_transformed1.flatten(),
        'ts0': forecast_inverse_transformed2.flatten()
    })

    forecast_df.to_csv('forecast_2_Networks.txt', index=False)
    print('forecast models saved!')



def main():
    # Device configuration (use GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameter FNO
    learning_rate = 0.001
    epochs = 300
    step_size = 50
    gamma = 0.5
    modes = 16
    width = 64

    # Hyperparameter for the rolling window. So far sequence size and window size should have the same shape because of
    # the MSE calculation
    window_size = 35 # sequence size
    target_size = 35
    batch_size = 1

    # Instantiate the data preprocessor
    data_preprocessor = DataPreprocessor('TrainingData.txt', 'TestingData.txt', window_size, target_size, batch_size)

    # Load the data
    train_dataloader1, train_inputs1, train_dataloader2, train_inputs2, train_normalized_data  = data_preprocessor.load_data()

    # Initialize two FNO models
    fno1 = FNO1d(modes, width).to(device)
    fno2 = FNO1d(modes, width).to(device)

    # Initialize two optimizers
    optimizer1 = torch.optim.Adam(fno1.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer2 = torch.optim.Adam(fno2.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Initialize two schedulers
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

    # Combine the two dataloaders into a single one
    # train_dataloader = [(x1, x2) for (x1, _), (_, x2) in zip(train_dataloader1, train_dataloader2)]

    # Train the models
    train_model(fno1, fno2, train_dataloader1, train_dataloader2, optimizer1, optimizer2, scheduler1, scheduler2, device, epochs)

    # Get the last known sequence from your training data
    last_known_sequence1 = train_inputs1[-1].unsqueeze(0)
    last_known_sequence2 = train_inputs2[-1].unsqueeze(0)

    # Define the forecast length
    original_data = pd.read_csv('TrainingData.txt')  # replace this with how you get your original data
    timestamps = np.array(original_data['t'])
    time_step = timestamps[1] - timestamps[0]  # Time step between each data point
    forecast_length = int(np.ceil((602168 - 520000) / time_step))

    # Call the recursive_forecasting function
    forecast1, forecast2 = recursive_forecasting(fno1, fno2, last_known_sequence1, last_known_sequence2, forecast_length, device)

    # def recursive_forecasting(fno1, fno2, last_known_sequence1, last_known_sequence2, forecast_length, device):

    # Plot the temperatures
    plot_temperatures(original_data, forecast1, forecast2, data_preprocessor)

    # Plot the normalized temperatures
    # plot_temperatures(train_normalized_data, forecast, data_preprocessor)

if __name__ == "__main__":
    main()

