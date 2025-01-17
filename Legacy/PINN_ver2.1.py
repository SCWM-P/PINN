import torch
import time
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Setting random seeds to ensure reproducible results
np.random.seed(1234)
torch.manual_seed(1234)
torch.autograd.set_detect_anomaly(True)
plt.ion()
plt.rc('font', family='Times New Roman')

# Check CUDA availability (for GPU acceleration)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("========  Using device  ========")
print(f"============  {device}  ============")


#%% Define functions and classes we need
def HotPixel_cleansing(xEvent: np.ndarray, Timestamp: np.ndarray, yEvent: np.ndarray):
    """
    :param xEvent: Event data of x (ndarray)
    :param Timestamp: Event data of time (ndarray)
    :param yEvent: Event data of y (ndarray)
    :return: filtered event data (ndarray)
    This function is used to brush the HotPixel of data via 3-sigma rule.
    """
    df = pd.DataFrame({'Timestamp': Timestamp, 'xEvent': xEvent, 'yEvent': yEvent})
    df['coords'] = list(zip(df['xEvent'], df['yEvent']))
    grouped = df.groupby('coords')
    event = grouped.agg(activity=('coords', 'size'),
                        continuity=('Timestamp', lambda x: np.mean(np.diff(sorted(x))) if len(x) > 1 else np.nan))
    act_mean = event['activity'].mean()
    act_std = event['activity'].std()
    cont_mean = event['continuity'].mean()
    cont_std = event['continuity'].std()
    event_filtered = event[
        (event['activity'] > act_mean - 3 * act_std) & (event['activity'] < act_mean + 1.5 * act_std) &
        (event['continuity'] > cont_mean - 3 * cont_std) & (event['continuity'] < cont_mean + 1.5 * cont_std)]
    filtered_events = df[df['coords'].isin(event_filtered.index)]
    return (filtered_events['xEvent'].to_numpy(),
            filtered_events['Timestamp'].to_numpy(),
            filtered_events['yEvent'].to_numpy())


# Define DNN Classes
class DNN(torch.nn.Module):
    """
    DNN model with residual link\n
    Default activation function is tanh()
    :param layers: list of layer sizes
    :param connections: list of connection scheme
    """
    def __init__(self, layers: list, connections: list):
        super(DNN, self).__init__()
        self.layers = layers
        self.connections = connections
        self.num_layers = len(layers)
        if len(connections) != self.num_layers:
            raise ValueError("Length of connections must match the number of layers")
        # Define Linear Layer
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(layers[i], layers[i + 1]) for i in range(self.num_layers - 1)]
        )
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        residual = [x]
        for i, (linear, conn) in enumerate(zip(self.linears, self.connections)):
            x = linear(x)
            if i < self.num_layers - 2:
                x = self.tanh(x)
            for rsd in residual[-conn:]:
                if x.shape == rsd.shape:
                    x = x + rsd
            residual.append(x)
        return x


# Define Physical Information Neural Network Classes
class PhysicsInformedNN:
    def __init__(
            self,
            layers: list,
            connections: list,
            device: torch.device,
            xEvent: torch.Tensor,
            Timestamp: torch.Tensor,
            yEvent: torch.Tensor,
            epochs: int = 1000,
            validation_ratio=0.2
    ):
        # Configuration
        self.device = device
        self.dnn = DNN(layers, connections).to(device)
        # Learning Parameters
        self.EI = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.T = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.M = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.c = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.gamma = torch.nn.Parameter(torch.tensor([torch.pi / 4], dtype=torch.float32, device=device))
        # Data
        self.xEvent = xEvent
        self.Timestamp = Timestamp
        self.yEvent = yEvent
        # Split train and validation sets
        self.x_train, self.x_val, self.t_train, self.t_val, self.y_train, self.y_val = train_test_split(
            self.xEvent, self.Timestamp, self.yEvent,
            test_size=validation_ratio,
            random_state=42
        )
        # Define the Training Parameters History
        self.history = {
            'train_loss': torch.zeros((epochs,)),
            'train_accuracy': torch.zeros((epochs,)),
            'EI': torch.zeros((epochs,)),
            'T': torch.zeros((epochs,)),
            'M': torch.zeros((epochs,)),
            'c': torch.zeros((epochs,)),
            'gamma': torch.zeros((epochs,))
        }
        # Define Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.dnn.parameters()) + [self.EI, self.T, self.M, self.c, self.gamma],
            lr=0.01,
            betas=(0.9, 0.999),
            eps=1e-15,
            weight_decay=0.0001
        )

    def predict(self, xEvent: torch.Tensor, Timestamp: torch.Tensor):
        y_pred = self.dnn(torch.cat([xEvent, Timestamp], dim=1))
        return y_pred

    def normalize(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Normalize data with mean and std.
        """
        x = (x - self.xEvent.mean()) / self.xEvent.std()
        t = (t - self.Timestamp.mean()) / self.Timestamp.std()
        y = (y - self.yEvent.mean()) / self.yEvent.std()
        return x, t, y

    def denormalize(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Denormalize data with mean and std.
        """
        x = x * self.xEvent.std() + self.xEvent.mean()
        t = t * self.Timestamp.std() + self.Timestamp.mean()
        y = y * self.yEvent.std() + self.yEvent.mean()
        return x, t, y

    def rotate(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, gamma: torch.Tensor, axis='None'):
        """
        Rotate data around the specified axis,For example, if axis='x', the data will be rotated around the x-axis.\n
        The data manipulated by the function are all **torch.Tensor** type.
        :param x: original x-axis data
        :param y: original y-axis data
        :param z: original z-axis data
        :param gamma: the angle of rotation in radians
        :param axis: axis of rotation you specify
        :return: rotated data tuple with (x, y, z)
        """
        if axis == 'x':
            x_r = x
            y_r = y * torch.cos(gamma) - z * torch.sin(gamma)
            z_r = y * torch.sin(gamma) + z * torch.cos(gamma)
        elif axis == 'y':
            x_r = x * torch.cos(gamma) + z * torch.sin(gamma)
            y_r = y
            z_r = -x * torch.sin(gamma) + z * torch.cos(gamma)
            dy_dx = torch.autograd.grad(
                z_r, x_r,
                grad_outputs=torch.ones_like(y),
                retain_graph=True,
                create_graph=True,
            )[0]
        elif axis == 'z':
            x_r = x * torch.cos(gamma) - y * torch.sin(gamma)
            y_r = x * torch.sin(gamma) + y * torch.cos(gamma)
            z_r = z
        else:
            raise Exception(
                '\nNo axis specified! Please specify an axis.\n'
                'For example axis=\'x\' if you want to rotate around the x-axis.'
            )
        return x_r, y_r, z_r

    def rotateLoss(self, xEvent: torch.Tensor, Timestamp: torch.Tensor, yEvent: torch.Tensor):
        """
        Calculate the loss of the rotation of the data.
        """
        xEvent, Timestamp, yEvent = self.rotate(xEvent, Timestamp, yEvent, self.gamma, axis='y')
        loss_rotate = torch.std(yEvent)
        return loss_rotate

    def phi(self, x: float,mu: float, sigma:float ):
        """
        Calculate the value of the normal distribution function.
        """
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))

    def derive(self, xEvent: torch.Tensor, Timestamp: torch.Tensor, yEvent: torch.Tensor):
        """
        Perform partial differential operations.\n
        The data manipulated by the function are all **torch.Tensor** type.
        :return: partial differential tuple (∂4y/∂x4, ∂2y/∂x2, ∂2y/∂t2, ∂y/∂t)
        """
        gamma = self.gamma.clone().detach()
        xEvent, Timestamp, _ = self.normalize(xEvent, Timestamp, yEvent)
        xEvent, Timestamp, _ = self.rotate(xEvent, Timestamp, yEvent, gamma, axis='y')
        y = self.predict(xEvent, Timestamp)
        xEvent, Timestamp, y = self.rotate(xEvent, Timestamp, y, -gamma, axis='y')
        xEvent, Timestamp, y = self.denormalize(xEvent, Timestamp, y)
        dy_dx = torch.autograd.grad(
            y, xEvent,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=True,
        )[0]
        # We need ∂2y/∂x2
        d2y_dx2 = torch.autograd.grad(
            dy_dx, xEvent,
            grad_outputs=torch.ones_like(dy_dx),
            retain_graph=True,
            create_graph=True
        )[0]
        d3y_dx3 = torch.autograd.grad(
            d2y_dx2, xEvent,
            grad_outputs=torch.ones_like(d2y_dx2),
            retain_graph=True,
            create_graph=True
        )[0]
        # We need ∂4y/∂x4
        d4y_dx4 = torch.autograd.grad(
            d3y_dx3, xEvent,
            grad_outputs=torch.ones_like(d3y_dx3),
            retain_graph=True,
            create_graph=True
        )[0]
        # We need ∂y/∂t
        dy_dt = torch.autograd.grad(
            y, Timestamp,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=True
        )[0]
        # We need ∂2y/∂t2
        d2y_dt2 = torch.autograd.grad(
            dy_dt, Timestamp,
            grad_outputs=torch.ones_like(dy_dt),
            retain_graph=True,
            create_graph=True
        )[0]
        return d4y_dx4, d2y_dx2, d2y_dt2, dy_dt

    def physicalLoss(self, xEvent: torch.Tensor, Timestamp: torch.Tensor, yEvent: torch.Tensor):
        """
        Calculates the physical loss.\n
        The physical equation is:\n
        EI * ∂4y/∂x4 - T * ∂2y/∂x2 + M * ∂2y/∂t2 + c * ∂y/∂t = 0
        """
        d4y_dx4, d2y_dx2, d2y_dt2, dy_dt = self.derive(xEvent, Timestamp, yEvent)
        y_PI_pred = \
            self.EI * d4y_dx4 - self.T * d2y_dx2 + self.M * d2y_dt2 + self.c * dy_dt
        loss_Physical = torch.nn.functional.mse_loss(y_PI_pred, torch.zeros_like(y_PI_pred))
        return loss_Physical

    def plot_results(self, epoch: int):
        """
        Plots the results of the model in particular epoch.
        :param epoch: Current number of model training rounds
        :return: None. Just plot!
        """
        self.dnn.eval()
        with torch.no_grad():
            # Plot the results in train set
            x_train, t_train, y_train = self.normalize(self.x_train, self.t_train, self.y_train)
            x_train, t_train, y_train = self.rotate(x_train, t_train, y_train, self.gamma, axis='y')
            y_nn_pred_train = self.predict(x_train, t_train)
            x_train, t_train, y_nn_pred_train = self.rotate(x_train, t_train, y_nn_pred_train, -self.gamma, axis='y')
            x_train, t_train, y_nn_pred_train = self.denormalize(x_train, t_train, y_nn_pred_train)

            # plot the results in validation set
            x_val, t_val, y_val = self.normalize(self.x_val, self.t_val, self.y_val)
            x_val, t_val, y_val = self.rotate(x_val, t_val, y_val, self.gamma, axis='y')
            y_nn_pred_val = self.predict(x_val, t_val)
            x_val, t_val, y_nn_pred_val = self.rotate(x_val, t_val, y_nn_pred_val, -self.gamma, axis='y')
            x_val, t_val, y_nn_pred_val = self.denormalize(x_val, t_val, y_nn_pred_val)

            # Convert torch.Tensor to numpy.ndarray
            x_train = x_train.cpu().detach().numpy()
            t_train = t_train.cpu().detach().numpy()
            y_nn_pred_train = y_nn_pred_train.cpu().detach().numpy()

            x_val = x_val.cpu().detach().numpy()
            t_val = t_val.cpu().detach().numpy()
            y_val = y_val.cpu().detach().numpy()
            y_nn_pred_val = y_nn_pred_val.cpu().detach().numpy()

            # Plot the results
            fig = plt.figure()
            ax_train = fig.add_subplot(121, projection='3d')
            ax_val = fig.add_subplot(122, projection='3d')
            ax_train.scatter(
                self.x_train.cpu().detach().numpy(),
                self.t_train.cpu().detach().numpy(),
                self.y_train.cpu().detach().numpy(),
                c='b', marker='.', alpha=0.2,
                label='Actual Dataset'
            )
            ax_train.scatter(
                x_train, t_train, y_nn_pred_train,
                c='r', marker='.', alpha=0.2,
                label='Prediction of Natural Network'
            )
            ax_train.set_xlabel('$X$', fontsize=18)
            ax_train.set_ylabel('$T (s)$', fontsize=18)
            ax_train.set_zlabel('$Y$', fontsize=18)
            ax_train.legend()
            ax_train.set_title(f'Comparison at Epoch {epoch} in Train Set', fontsize=20)
            ax_val.scatter(
                self.x_val.cpu().detach().numpy(),
                self.t_val.cpu().detach().numpy(),
                self.y_val.cpu().detach().numpy(),
                c='b', marker='.', alpha=0.2,
                label='Actual Dataset'
            )
            ax_val.scatter(
                x_val, t_val, y_nn_pred_val,
                c='r', marker='.', alpha=0.5,
                label='Prediction of Natural Network'
            )
            ax_val.set_xlabel('$X$', fontsize=18)
            ax_val.set_ylabel('$T (s)$', fontsize=18)
            ax_val.set_zlabel('$Y$', fontsize=18)
            ax_val.legend()
            ax_val.set_title(f'Comparison at Epoch {epoch} in Validation Set', fontsize=20)
            plt.show()

    def train(self):
        x_train, t_train, y_train = self.normalize(self.x_train, self.t_train, self.y_train)
        for epoch in range(epochs):
            self.dnn.train()  # Train the  model in training set
            self.optimizer.zero_grad()
            # Loss of Natural Network
            gamma = self.gamma.clone().detach()
            x_train, t_train, y_train = self.rotate(x_train, t_train, y_train, gamma, axis='y')
            y_nn_pred = self.predict(x_train, t_train)
            loss_nn = torch.nn.functional.mse_loss(y_nn_pred, y_train)
            # Loss of Rotation
            loss_rotation = self.rotateLoss(self.x_train, self.t_train, self.y_train)
            # Loss of Physical Informed Equation
            x_train, t_train, y_train = self.rotate(x_train, t_train, y_train, -gamma, axis='y')
            loss_physical = self.physicalLoss(self.x_train, self.t_train, self.y_train)

            Loss = 100 * self.phi(epoch/epochs, 0.5, 1) * loss_nn + \
                        self.phi(epoch/epochs, 0, 0.05) * loss_rotation + \
                        self.phi(epoch/epochs, 1, 0.2) * loss_physical
            Loss.backward(retain_graph=True)
            self.optimizer.step()

            # Record loss
            self.dnn.eval()  # Evaluate the  model in validation set
            truth_train = torch.sum(torch.abs(y_nn_pred - y_train) <= 0.1 * torch.abs(y_train))
            self.history['train_loss'][epoch] = loss_nn.item()
            self.history['train_accuracy'][epoch] = truth_train.item() / y_train.numel() * 100
            self.history['EI'][epoch] = self.EI.item()
            self.history['T'][epoch] = self.T.item()
            self.history['M'][epoch] = self.M.item()
            self.history['c'][epoch] = self.c.item()
            self.history['gamma'][epoch] = self.gamma.item()

            # Print epoch Results
            if epoch % 50 == 0:
                epoch_endTime = time.time()
                epoch_time = (epoch_endTime - epoch_startTime) if locals().get('epoch_startTime') else 0
                x_val, t_val, y_val = self.normalize(self.x_val, self.t_val, self.y_val)
                y_nn_val_pred = self.predict(x_val, t_val)
                truth_val = torch.sum(torch.abs(y_nn_val_pred - y_val) <= 0.1 * torch.abs(y_val))
                loss_nn_val = torch.nn.functional.mse_loss(y_nn_val_pred, y_val)
                loss_physical_val = self.physicalLoss(self.x_val, self.t_val, self.y_val)
                print(f'\n========  Epoch {epoch} / Total {epochs}  =======')
                print(f'======== Cost {epoch_time:.1f}s per 50 epochs ========')
                print('= = == === ====  Train  Set  ==== === == =')
                print(f'Natural Network Loss:{loss_nn.item():.3e}')
                print(f'Physical Equation Loss:{loss_physical.item():.3e}')
                print(f'Accuracy:{self.history["train_accuracy"][epoch]:.2f}%')
                print('= == === ==== Validation Set ==== === == =')
                print(f'Natural Network Loss:{loss_nn_val.item():.3e}')
                print(f'Physical Equation Loss:{loss_physical_val.item():.3e}')
                print(f'Accuracy:{(truth_val.item() / y_val.numel())*100:.2f}%')
                print('------------------------------------------')
                print(f'EI: {self.EI.item():<9.4e}, EI_grad: {self.EI.grad.item():.4e}')
                print(f'T: {self.T.item():<10.4e}, T_grad: {self.T.grad.item():.4e}')
                print(f'M: {self.M.item():<10.4e}, M_grad: {self.M.grad.item():.4e}')
                print(f'c: {self.c.item():<10.4e}, c_grad: {self.c.grad.item():.4e}')
                print(f'gamma: {self.gamma.item():<6.4f}, gamma_grad: {self.gamma.grad.item():.4e}\n')
                epoch_startTime = time.time()

            # Process Visualization
            if epochs <= 1000:
                if epoch % 100 == 0:
                    self.plot_results(epoch)
            else:
                if epoch <= 1000:
                    if epoch % 100 == 0:
                        self.plot_results(epoch)
                elif epoch % (epochs // 20) == 0:
                    if epoch <= 3000:
                        self.plot_results(epoch)


if __name__ == '__main__':
    # Configuration and Load Data
    epochs = 1000
    layers = [2, 80, 80, 80, 80, 80, 80, 1]
    connections = [0, 1, 2, 3, 4, 5, 5, 2]


    # data = scipy.io.loadmat('test16-1.mat')
    # Timestamp = data['brushedData'][:, 0]/1e6
    # xEvent = data['brushedData'][:, 1]
    # yEvent = data['brushedData'][:, 2]

    filename = r'dvSave-2023_03_26_02_21_16.npy'
    data = np.load(f'data/npy/{filename}', allow_pickle=True).item()
    xEvent = data['xEvent']
    Timestamp = data['Timestamp']
    yEvent = data['yEvent']

    # Data Cleansing
    (xEvent, Timestamp, yEvent) = HotPixel_cleansing(xEvent, Timestamp, yEvent)
    # Convert to torch.Tensor
    xEvent = torch.tensor(xEvent, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(1)
    Timestamp = torch.tensor(Timestamp, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(1)
    yEvent = torch.tensor(yEvent, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(1)

    print('====== Data Loading Done! ======')
    print('===== Model Initialization =====')
    pinn = PhysicsInformedNN(layers, connections, device, xEvent, Timestamp, yEvent, epochs)
    print('========= Model Training =======')

    # Training the Model
    start_time = time.time()
    pinn.train()
    end_time = time.time()
    print('==============================================')
    print('============= Model Training Done! ===========')
    print("======== Training time: {:.2f} seconds ======".format(end_time - start_time))
    print('=== Average time per epoch: {:.4f} seconds ==='.format((end_time - start_time) / epochs))
    print('==============================================')

    #%% Draw the final results for visualization

    # Plot parameter change curves
    fig = plt.figure(figsize=(18, 10))
    plt.rcParams.update({'font.size': 16})
    layout = (3, 3)
    subplots = [plt.subplot2grid(layout, (0, 0)), plt.subplot2grid(layout, (0, 1)), plt.subplot2grid(layout, (0, 2)),
                plt.subplot2grid(layout, (1, 1)), plt.subplot2grid(layout, (2, 1))]
    line_styles = ['-', '--', '-.', ':', '-']
    plots = [subplots[0].plot(pinn.history['EI'], c='r', ls=line_styles[0], label='EI'),
             subplots[1].plot(pinn.history['T'], c='g', ls=line_styles[1], label='T'),
             subplots[2].plot(pinn.history['M'], c='b', ls=line_styles[2], label='M'),
             subplots[3].plot(pinn.history['c'], c='c', ls=line_styles[3], label='c'),
             subplots[4].plot(pinn.history['gamma'], c='m', ls=line_styles[4], label='γ')]
    for i, ax in enumerate(subplots):
        ax.set_title(f"Parameter {['EI', 'T', 'M', 'c', 'γ'][i]}", fontsize=16)
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Parameter Value', fontsize=16)
        ax.legend()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle('Parameter Evolution', fontsize=20)
    plt.show()

    # Plotting Loss Function and Accuracy Change
    fig2, ax2_1 = plt.subplots()
    ax2_2 = ax2_1.twinx()
    ax2_1.plot(pinn.history['train_loss'], 'r-', label='Loss', linewidth=2)
    ax2_2.plot(pinn.history['train_accuracy'], 'b.-', label='Accuracy', linewidth=2)
    ax2_1.set_xlabel('Epoch', fontsize=18)
    ax2_1.set_ylabel('Loss', fontsize=18)
    ax2_2.set_ylabel('Accuracy (%)', fontsize=18)
    plt.title('Loss and Accuracy Change', fontsize=20)
    plt.legend()

    pinn.plot_results(epochs)
