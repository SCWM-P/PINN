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

# 检查CUDA可用性（用于GPU加速）
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
        self.m = torch.nn.Parameter(torch.tensor([1.0], device=device))
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
            list(self.dnn.parameters()) + [self.EI, self.T, self.m, self.c],
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
                'No axis specified! Please specify an axis.\n For example axis=\'x\' if you want to rotate around the x-axis.')
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

    def rotateLoss(self, xEvent: torch.Tensor, Timestamp: torch.Tensor, yEvent: torch.Tensor):
        xEvent, Timestamp, yEvent = self.rotate(xEvent, Timestamp, yEvent, self.gamma, axis='y')
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

    def physicalLoss(self, xEvent:torch.Tensor, Timestamp:torch.Tensor, yEvent:torch.Tensor):
        xEvent, Timestamp, yEvent = self.rotate(xEvent, Timestamp, yEvent, self.gamma, axis='y')
        d4y_dx4, d2y_dx2, d2y_dt2, dy_dt = self.derive(xEvent, Timestamp)
        y_PI_pred = \
        self.EI * d4y_dx4 - self.T * d2y_dx2 + self.m * d2y_dt2 + self.c * dy_dt
        loss_Physical = torch.nn.functional.mse_loss(y_PI_pred, torch.zeros_like(y_PI_pred))
        return loss_Physical

    def train(self, epochs):
        self.history = {
            'train_loss': torch.zeroes((epochs,)),
            'val_loss': torch.zeros((epochs,)),
            'EI': torch.zeros((epochs,)),
            'T': torch.zeros((epochs,)),
            'm': torch.zeros((epochs,)),
            'c': torch.zeros((epochs,)),
            'gamma': torch.zeros((epochs,))
        }
        for epoch in np.range(epochs):
            self.dnn.train()
            self.optimizer.zero_grad()