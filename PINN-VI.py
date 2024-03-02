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
plt.rc('font', family='Times New Roman')

# Check CUDA availability (for GPU acceleration)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===Using device===")
print(f"=====  {device}  =====")


def HotPixel_brush(xEvent, Timestamp, yEvent):
    """
    :param xEvent: Event data of x
    :param Timestamp: Event data of time
    :param yEvent: Event data of y
    :return: filtered event data
    This function is used to brush the data via 3-sigma rule
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
    return filtered_events['xEvent'].to_numpy(), filtered_events['Timestamp'].to_numpy(), filtered_events['yEvent'].to_numpy()


# 定义DNN类
class DNN(torch.nn.Module):
    def __init__(self, layers, connections):
        super(DNN, self).__init__()
        self.layers = layers
        self.connections = connections
        self.num_layers = len(layers)
        if len(connections) != self.num_layers:
            raise ValueError("Length of connections must match the number of layers")
        # 线性层
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


# 定义物理信息神经网络类
class PhysicsInformedNN:
    def __init__(self, layers, connections, device, xEvent, Timestamp, yEvent, validation_ratio=0.2):

        # Configuration
        self.device = device
        self.dnn = DNN(layers, connections).to(device)
        # Learning Parameters
        self.EI = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.T = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.M = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.c = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.gamma = torch.nn.Parameter(torch.tensor([torch.pi / 4], device=device))
        # Data
        self.xEvent = xEvent
        self.Timestamp = Timestamp
        self.yEvent = yEvent
        # Split train and validation sets
        self.x_train, self.x_val, self.t_train, self.t_val, self.y_train, self.y_val = train_test_split(
            self.xEvent, self.Timestamp, self.yEvent,
            test_size=validation_ratio,
            random_state=42)
        # Define Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.dnn.parameters()) + [self.EI, self.T, self.M, self.c, self.gamma],
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-15,
            weight_decay=0.0001
        )

    def predict(self, xEvent, Timestamp):
        y_pred = self.dnn(torch.cat([xEvent, Timestamp], dim=1))
        return y_pred

    def normalize(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        x = (x - self.xEvent.mean()) / self.xEvent.std()
        t = (t - self.Timestamp.mean()) / self.Timestamp.std()
        y = (y - self.yEvent.mean()) / self.yEvent.std()
        return x, t, y

    def denormalize(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        x = x * self.xEvent.std() + self.xEvent.mean()
        t = t * self.Timestamp.std() + self.Timestamp.mean()
        y = y * self.yEvent.std() + self.yEvent.mean()
        return x, t, y

    def rotate(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, gamma, axis='None'):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        if axis == 'None':
            raise Exception(
                'No axis specified! Please specify an axis.\n For example axis=\'x\' if you want to rotate around the x-axis.'
            )
        elif axis == 'x':
            x = x
            y = y * torch.cos(gamma) - z * torch.sin(gamma)
            z = y * torch.sin(gamma) + z * torch.cos(gamma)
        elif axis == 'y':
            x = x * torch.cos(gamma) + z * torch.sin(gamma)
            y = y
            z = -x * torch.sin(gamma) + z * torch.cos(gamma)
        elif axis == 'z':
            x = x * torch.cos(gamma) - y * torch.sin(gamma)
            y = x * torch.sin(gamma) + y * torch.cos(gamma)
            z = z
        return x, y, z

    def rotateLoss(self, xEvent: torch.Tensor, Timestamp: torch.Tensor, yEvent: torch.Tensor, axis='None'):
        xEvent, Timestamp, yEvent = self.rotate(xEvent, Timestamp, yEvent, self.gamma, axis=axis)
        loss_rotate = torch.std(yEvent)
        return loss_rotate

    def derive(self, xEvent: torch.Tensor, Timestamp: torch.Tensor):
        y = self.predict(xEvent, Timestamp)

        dy_dx = torch.autograd.grad(y, xEvent,
                                    grad_outputs=torch.ones_like(y),
                                    retain_graph=True,
                                    create_graph=True)[0]
        # We need ∂2y/∂x2
        d2y_dx2 = torch.autograd.grad(dy_dx, xEvent,
                                      grad_outputs=torch.ones_like(dy_dx),
                                      retain_graph=True,
                                      create_graph=True)[0]
        d3y_dx3 = torch.autograd.grad(d2y_dx2, xEvent,
                                      grad_outputs=torch.ones_like(d2y_dx2),
                                      retain_graph=True,
                                      create_graph=True)[0]
        # We need ∂4y/∂x4
        d4y_dx4 = torch.autograd.grad(d3y_dx3, xEvent,
                                      grad_outputs=torch.ones_like(d3y_dx3),
                                      retain_graph=True,
                                      create_graph=True)[0]
        # We need ∂y/∂t
        dy_dt = torch.autograd.grad(y, Timestamp,
                                    grad_outputs=torch.ones_like(y),
                                    retain_graph=True,
                                    create_graph=True)[0]
        # We need ∂2y/∂t2
        d2y_dt2 = torch.autograd.grad(dy_dt, Timestamp,
                                      grad_outputs=torch.ones_like(dy_dt),
                                      retain_graph=True,
                                      create_graph=True)[0]
        return d4y_dx4, d2y_dx2, d2y_dt2, dy_dt

    def physicalLoss(self, xEvent: torch.Tensor, Timestamp: torch.Tensor, yEvent: torch.Tensor):
        xEvent, Timestamp, yEvent = self.rotate(xEvent, Timestamp, yEvent, self.gamma, axis='y')
        d4y_dx4, d2y_dx2, d2y_dt2, dy_dt = self.derive(xEvent, Timestamp)
        y_PI_pred = \
        self.EI * d4y_dx4 - self.T * d2y_dx2 + self.M * d2y_dt2 + self.c * dy_dt
        loss_Physical = torch.nn.functional.mse_loss(y_PI_pred, torch.zeros_like(y_PI_pred))
        return loss_Physical

    def plot_results(self, epoch: int):
        self.dnn.eval()
        with torch.no_grad():
            # Plot the results in train set
            x_train, t_train, y_train = self.normalize(self.x_train, self.t_train, self.y_train)
            y_nn_pred_train = self.predict(x_train, t_train).flatten()
            x_train, t_train, y_nn_pred_train = self.denormalize(x_train, t_train, y_nn_pred_train)

            # plot the results in validation set
            x_val, t_val, y_val = self.normalize(self.x_val, self.t_val, self.y_val)
            y_nn_pred_val = self.predict(x_val, t_val).flatten()

            # Convert torch.Tensor to numpy.ndarray
            x_train = x_train.cpu().detach().numpy()
            t_train = t_train.cpu().detach().numpy()
            y_nn_pred_train = y_nn_pred_train.cpu().detach().numpy()

            x_val = x_val.cpu().detach().numpy()
            t_val = t_val.cpu().detach().numpy()
            y_nn_pred_val = y_nn_pred_val.cpu().detach().numpy()

            # Plot the results
            fig = plt.figure()
            ax_train = fig.add_subplot(121, projection='3d')
            ax_val = fig.add_subplot(122, projection='3d')
            ax_train.scatter(x_train, t_train, y_train, c='b', marker='.', label='Actual Dataset', alpha=0.5)
            ax_train.scatter(x_train, t_train, y_nn_pred_train, c='r', marker='.', label='Prediction of Natural Network', alpha=0.5)
            ax_train.set_xlabel('$X$', fontsize=18)
            ax_train.set_ylabel('$T (s)$', fontsize=18)
            ax_train.set_zlabel('$Y$', fontsize=18)
            ax_train.legend()
            ax_train.title(f'Comparison at Epoch {epoch} in Train Set', fontsize=20)
            ax_val.scatter(x_val, t_val, y_val, c='b', marker='.', label='Actual Dataset', alpha=0.5)
            ax_val.scatter(x_val, t_val, y_nn_pred_val, c='r', marker='.', label='Prediction of Natural Network', alpha=0.5)
            ax_val.set_xlabel('$X$', fontsize=18)
            ax_val.set_ylabel('$T (s)$', fontsize=18)
            ax_val.set_zlabel('$Y$', fontsize=18)
            ax_val.legend()
            ax_val.title(f'Comparison at Epoch {epoch} in Validation Set', fontsize=20)
            plt.show()

    def train(self, epochs: int):
        self.history = {
            'train_loss': torch.zeroes((epochs,)),
            'train_accuracy': torch.zeros((epochs,)),
            'EI': torch.zeros((epochs,)),
            'T': torch.zeros((epochs,)),
            'M': torch.zeros((epochs,)),
            'c': torch.zeros((epochs,)),
            'gamma': torch.zeros((epochs,))
        }
        x_train, t_train, y_train = self.normalize(self.x_train, self.t_train, self.y_train)
        for epoch in np.range(epochs):
            self.dnn.train()  # Train the  model in training set
            self.optimizer.zero_grad()
            # Loss of Natural Network
            y_nn_pred = self.predict(x_train, t_train)
            loss_nn = torch.nn.functional.mse_loss(y_nn_pred, y_train)
            # Loss of Rotation
            loss_rotation = self.rotateLoss(self.x_train, self.t_train, self.y_train, axis='y')
            # Loss of Physical Informed Equation
            loss_physical = self.physicalLoss(self.x_train, self.t_train, self.y_train)

            Loss = loss_nn + 100*loss_rotation + 10000*loss_physical
            Loss.backward(retain_graph=True)
            self.optimizer.step()

            # Record loss
            with torch.no_grad():
                self.dnn.eval()  # Evaluate the  model in validation set
                truth_train = torch.sum(y_nn_pred[torch.abs(y_nn_pred - y_train) <= 1e-5])
                self.history['train_loss'][epoch] = loss_nn.item()
                self.history['train_accuracy'][epoch] = truth_train.item() / len(y_train)
                self.history['EI'][epoch] = self.EI.item()
                self.history['T'][epoch] = self.T.item()
                self.history['M'][epoch] = self.M.item()
                self.history['c'][epoch] = self.c.item()
                self.history['gamma'][epoch] = self.gamma.item()

                # Print epoch Results
                if epoch % 50 == 0:
                    x_val, t_val, y_val = self.normalize(self.x_val, self.t_val, self.y_val)
                    y_nn_val_pred = self.predict(x_val, t_val)
                    truth_val = torch.sum(y_nn_val_pred[torch.abs(y_nn_val_pred - y_val) <= 1e-5])
                    loss_nn_val = torch.nn.functional.mse_loss(y_nn_val_pred, y_val)
                    loss_physical_val = self.physicalLoss(self.x_val, self.t_val, self.y_val)
                    print(f'========= Epoch {epoch} =========')
                    print(f'Train Set Natural Loss:{loss_nn.item():.3f}')
                    print(f'Train Set Physical  Equation Loss:{loss_physical.item():.3f}')
                    print(f'Train Set Accuracy:{self.history["train_accuracy"][epoch]:.3f}')
                    print(f'Validation Set Natural Loss:{loss_nn_val.item():.3f}')
                    print(f'Validation Set Physical  Equation Loss:{loss_physical_val.item():.3f}')
                    print(f'Validation Set Accuracy:{truth_val.item() / len(y_val):.3f}')
                    print(f'EI: {self.EI.item():<10.4f}, EI_grad: {self.EI.grad.item():.8f}')
                    print(f'T: {self.T.item():<10.4f}, T_grad: {self.T.grad.item():.8f}')
                    print(f'M: {self.M.item():<10.4f}, M_grad: {self.M.grad.item():.8f}')
                    print(f'c: {self.c.item():<10.4f}, c_grad: {self.c.grad.item():.8f}')
                    print(f'gamma: {self.gamma.item():<10.4f}, gamma_grad: {self.gamma.grad.item():.8f}')

                # Process Visualization
                if epochs <= 10000:
                    if epoch % 1000 == 0:
                        self.plot_results(epoch)
                else:
                    if epoch <= 10000:
                        if epoch % 1000 == 0:
                            self.plot_results(epoch)
                    elif epoch % (epochs // 20) == 0:
                        if epoch <= 30000:
                            self.plot_results(epoch)