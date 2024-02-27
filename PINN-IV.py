# 导入必要的库
import torch
import time
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果可重复
np.random.seed(1234)
torch.manual_seed(1234)
plt.rc('font',family='Times New Roman')

# 检查CUDA可用性（用于GPU加速）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===Using device===\n", device)

def brush(xEvent, T, yEvent):
    """
    :param xEvent:
    :param T:
    :param yEvent:
    :return: Brushed xEvent, T, yEvent
    This function is used to brush the data via 3-sigma rule
    """
    df = pd.DataFrame({'T': T, 'yEvent': yEvent, 'xEvent': xEvent})
    df['coords'] = list(zip(df.xEvent, df.yEvent))
    unique_coords = df['coords'].unique()
    event = pd.DataFrame(columns=['activity', 'continuity', 'coords'])

    for coords in unique_coords:
        current_events = df[df['coords'] == coords]
        if len(current_events) > 1:
            current_activity = len(current_events)
            current_continuity = np.mean(np.diff(current_events['T'].to_numpy()))
            event.loc[len(event)] = [current_activity, current_continuity, coords]
    act_mean = event.activity.mean()
    cont_mean = event.continuity.mean()
    act_std = event.activity.std()
    cont_std = event.continuity.std()
    event = event[(event.activity > act_mean - 3*act_std) & (event.activity < act_mean + 3*act_std)]
    event = event[(event.continuity > cont_mean - 3*cont_std) & (event.continuity < cont_mean + 3*cont_std)]
    filtered_events = df[df['coords'].isin(event.coords.to_list())]
    return filtered_events['xEvent'].to_numpy(), filtered_events['T'].to_numpy(), filtered_events['yEvent'].to_numpy()


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
        for i, (linear, conn) in enumerate(zip(self.linears,self.connections)):
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
    def __init__(self, layers, connections, device, xEvent, T, yEvent, validation_ratio=0.2):
        """
        layers: 与DNN类中相同，定义网络架构\n
        connections: 与DNN类中相同，定义网络连接\n
        device: 'cuda' 或 'cpu'，用于指定运算设备
        """
        # Configuration
        self.device = device
        self.dnn = DNN(layers, connections).to(device)
        # Data
        self.T_mean, self.T_std = torch.mean(T), torch.std(T)
        self.xEvent_mean, self.xEvent_std = torch.mean(xEvent), torch.std(xEvent)
        self.yEvent_mean, self.yEvent_std = torch.mean(yEvent), torch.std(yEvent)
        # Standardize the data
        T = (T - T.mean()) / T.std()
        xEvent = (xEvent - xEvent.mean()) / xEvent.std()
        yEvent = (yEvent - yEvent.mean()) / yEvent.std()
        self.T = T.clone().detach().requires_grad_(True).to(device).float()
        self.xEvent = xEvent.clone().detach().requires_grad_(True).to(device).float()
        self.yEvent = yEvent.clone().detach().requires_grad_(True).to(device).float()
        y0 = torch.mean(yEvent)
        # Learning Parameters
        self.a = torch.nn.Parameter(torch.tensor([0.1], device=device))
        self.b = torch.nn.Parameter(torch.tensor([0.1], device=device))
        self.omega = torch.nn.Parameter(torch.tensor([0.1], device=device))
        self.zeta = torch.nn.Parameter(torch.tensor([0.05], device=device))
        self.y0 = torch.nn.Parameter(y0)
        self.gamma = torch.nn.Parameter(torch.tensor([torch.pi / 4], device=device))
        # Extend Dimension
        self.T = self.T.unsqueeze(1)
        self.xEvent = self.xEvent.unsqueeze(1)
        self.yEvent = self.yEvent.unsqueeze(1)
        # Split Data
        self.x_train, self.x_val, self.T_train, self.T_val, self.y_train, self.y_val = \
            train_test_split(self.xEvent, self.T, self.yEvent,
                            test_size=validation_ratio,
                            random_state=42
                            )
        # Optimizer
        self.optimizer_Adam = torch.optim.Adam(
            list(self.dnn.parameters()) +
            [self.a,
             self.b,
             self.omega,
             self.zeta,
             self.y0,
             self.gamma
             ],
            lr=0.01, betas=(0.9, 0.999)
        )

    def rotate(self, xEvent, T, yEvent, gamma):
        """
        旋转函数
        """
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)
        # Rotation Matrix
        gamma_mat = torch.cat([
            torch.stack([cos_gamma,              torch.zeros_like(gamma),                   sin_gamma], dim=1),
            torch.stack([torch.zeros_like(gamma), torch.ones_like(gamma),     torch.zeros_like(gamma)], dim=1),
            torch.stack([-sin_gamma,             torch.zeros_like(gamma),                   cos_gamma], dim=1)
        ], dim=0)
        if xEvent.dim() == 1:
            xEvent = xEvent.unsqueeze(1)
        if T.dim() == 1:
            T = T.unsqueeze(1)
        if yEvent.dim() == 1:
            yEvent = yEvent.unsqueeze(1)
        xTy = torch.cat([xEvent, T, yEvent], dim=1)
        xTy = torch.matmul(gamma_mat, xTy.t()).t()
        xEvent = xTy[:,0]
        T = xTy[:,1]
        yEvent = xTy[:,2]
        return xEvent, T, yEvent

    def net_u(self, xEvent, T):
        y_pred = self.dnn(torch.cat([xEvent, T], dim=1))
        return y_pred

    def PIloss(self, xEvent, T, yEvent):
        """
        定义基于物理方程的残差函数
        """
        xEvent, T, yEvent = self.rotate(xEvent, T, yEvent, self.gamma)
        y_loss = self.yEvent_std * yEvent + self.yEvent_mean - torch.sin(self.a * (self.xEvent_std * xEvent + self.xEvent_mean) + self.b) *\
                torch.sin(self.omega * (self.T_std * T + self.T_mean) * torch.sqrt(1 - self.zeta**2)) *\
                torch.exp(-self.zeta * self.omega * (self.T_std * T + self.T_mean)) - (self.y0 * self.yEvent_std + self.yEvent_mean)
        PIloss = torch.nn.functional.mse_loss(y_loss, torch.zeros_like(y_loss))
        return PIloss

    def rotateLoss(self, xEvent, T, yEvent):
        """
        定义旋转算子
        :param xEvent
        :param T
        :param yEvent
        """
        xEvent, T, yEvent = self.rotate(xEvent, T, yEvent, self.gamma)
        rotate_loss = torch.std(yEvent)
        return rotate_loss

    def plot_results(self, x_val, T_val, y_val, epoch):
        self.dnn.eval()
        with torch.no_grad():
            x_val = x_val.cpu().detach().numpy()
            y_val = y_val.cpu().detach().numpy()
            T_val = T_val.cpu().detach().numpy()

            y_pred_val = self.predict(torch.tensor(x_val, device=device), torch.tensor(T_val, device=device)).flatten()

            x_val_R, T_val_R, y_val_R = self.rotate(self.x_val, self.T_val, self.y_val, self.gamma)
            y_pred_PI = torch.sin(self.a * x_val_R + self.b) * \
                        torch.sin(self.omega * T_val_R * torch.sqrt(1 - self.zeta ** 2)) * \
                        torch.exp(-self.zeta * self.omega * T_val_R) + self.y0
            x_val_R, T_val_R, y_pred_PI = self.rotate(x_val_R, T_val_R, y_pred_PI, -self.gamma)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_val, T_val, y_val, color='b', label='Actual')
            ax.scatter(x_val, T_val, y_pred_PI.cpu().detach().numpy(), color='g', label='Predicted (PI)')
            ax.scatter(x_val, T_val, y_pred_val, color='r', label='Predicted (NN)')
            ax.set_xlabel('X', fontsize=18)
            ax.set_ylabel('T (s)', fontsize=18)
            ax.set_zlabel('Y', fontsize=18)
            ax.legend()
            plt.title(f'Comparison at Epoch {epoch}', fontsize=20)
            plt.show()

    def train(self, epochs):
        """
        训练模型
        epochs: 训练迭代次数。
        """
        self.history = {
            'loss_train': [],
            'loss_val': [],
            'a': [],
            'b': [],
            'omega': [],
            'zeta': [],
            'y0': [],
            'gamma':[]
        }
        for epoch in range(epochs):
            self.dnn.train()
            self.optimizer_Adam.zero_grad()
            y_pred = self.net_u(self.x_train, self.T_train)
            loss_rotate =  self.rotateLoss(self.x_train, self.T_train, self.y_train)
            loss_y = torch.nn.functional.mse_loss(y_pred, self.y_train)
            loss_PI = self.PIloss(self.x_train, self.T_train, self.y_train)

            loss = loss_y + 100*loss_PI + 10000*loss_rotate # 这里可以通过调节系数来调整对y和PI的惩罚权重
            loss.backward(retain_graph=True)
            self.optimizer_Adam.step()
            self.history['loss_train'].append(loss.item())
            self.history['a'].append(self.a.item())
            self.history['b'].append(self.b.item())
            self.history['omega'].append(self.omega.item())
            self.history['zeta'].append(self.zeta.item())
            self.history['y0'].append(self.y0.item())
            self.history['gamma'].append(self.gamma.item())

            if epoch % 10 == 0:
                self.dnn.eval()
                with torch.no_grad():
                    y_pred_val = self.net_u(self.x_val, self.T_val)
                    loss_y_val = torch.nn.functional.mse_loss(y_pred_val, self.y_val)
                    loss_PI_val = self.PIloss(self.x_val, self.T_val, self.y_val)
                    loss_val = loss_y_val + loss_PI_val
                print(f'=======Epoch {epoch}=======\n'
                        f'Training Loss: {loss.item():.3f},Validation Loss: {loss_val.item():.3f}\n'
                        f'a:{self.a.item():.3f},a_grad:{self.a.grad.item()}\n'
                        f'b:{self.b.item():.3f},b_grad:{self.b.grad.item()}\n'
                        f'omega:{self.omega.item():.3f},omega_grad:{self.omega.grad.item()}\n'
                        f'zeta:{self.zeta.item():.3f},zeta_grad:{self.zeta.grad.item()}\n'
                        f'y0:{self.y0.item():.3f},y0_grad:{self.y0.grad.item():.5f}\n'
                        f'gamma:{self.gamma.item():.3f},gamma_grad:{self.gamma.grad.item():.5f}\n'
                      )
            if epoch % 1000 == 0:
                if epoch <= 50000:
                    self.plot_results(self.x_val, self.T_val, self.y_val, epoch)

    def predict(self, xEvent, T):
        """
        使用训练好的模型进行预测
        """
        self.dnn.eval()
        y_pred = self.net_u(xEvent, T)
        return y_pred.detach().cpu().numpy()


#%% Configurations
# Load Data
mat = scipy.io.loadmat('test16-1.mat')
layers = [2, 100, 100, 100, 100, 100, 100, 100, 100, 1]
connections = [0, 1, 2, 3, 4, 5, 5, 5, 5, 2]
# Read the Data
T = mat['brushedData'][:, 0]/1e6
xEvent = mat['brushedData'][:, 1]
yEvent = mat['brushedData'][:, 2]
# Brush the data
(xEvent, T, yEvent) = brush(xEvent, T, yEvent)
# Convert to torch tensors
T = torch.tensor(T, dtype=torch.float32, device=device)
xEvent = torch.tensor(xEvent, dtype=torch.float32, device=device)
yEvent = torch.tensor(yEvent, dtype=torch.float32, device=device)


print('=======Data Loading Done!=======')
print('======Model Initialization=====')
pinn = PhysicsInformedNN(layers, connections, device, xEvent, T, yEvent)
print('=======Model Training=======')
# 训练模型
start_time = time.time()
pinn.train(10000)
print('=======Model Training Done!=======')
print("===Training time: {:.2f} seconds==".format(time.time() - start_time))
print('==================================')
#%% Draw
# 绘制参数变化
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()
ax1.plot(pinn.history['a'], label='a')
ax1.plot(pinn.history['b'], label='b')
ax1.plot(pinn.history['omega'], label='ω')
ax1.plot(pinn.history['zeta'], label='ζ')
ax1.plot(pinn.history['gamma'], label='γ')
ax1.set_xlabel('Epoch', fontsize=18)
ax1.set_ylabel('Parameter Value', fontsize=18)
ax2.plot(pinn.history['y0'], label='y0', color='grey')
ax2.set_ylabel('y0 Value', fontsize=18)
fig.legend()
plt.title('Parameter Evolution', fontsize=20)

# 绘制损失变化
plt.figure(figsize=(12, 8))
plt.plot(pinn.history['loss_train'], label='Training Loss')
plt.plot(pinn.history['loss_val'], label='Validation Loss')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend()
plt.title('Training and Validation Loss', fontsize=20)

# 绘制验证集的三维散点图
x_val = pinn.x_val.cpu().detach().numpy()
y_val = pinn.y_val.cpu().detach().numpy()
T_val = pinn.T_val.cpu().detach().numpy()

y_pred_val = pinn.predict(torch.tensor(x_val, device=device), torch.tensor(T_val, device=device)).flatten()

x_val_R, T_val_R, y_val_R = pinn.rotate(pinn.x_val, pinn.T_val, pinn.y_val, pinn.gamma)
y_pred_PI = torch.sin(pinn.a * x_val_R + pinn.b) * \
            torch.sin(pinn.omega * T_val_R * torch.sqrt(1 - pinn.zeta**2)) * \
            torch.exp(-pinn.zeta * pinn.omega * T_val_R) + pinn.y0
x_val_R, T_val_R, y_pred_PI = pinn.rotate(x_val_R, T_val_R, y_pred_PI, -pinn.gamma)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_val, T_val, y_val, color='b', label='Actual')
ax.scatter(x_val, T_val, y_pred_PI.cpu().detach().numpy(), color='g', label='Predicted (PI)')
ax.scatter(x_val, T_val, y_pred_val, color='r', label='Predicted (NN)')
ax.set_xlabel('X', fontsize=18)
ax.set_ylabel('T (s)', fontsize=18)
ax.set_zlabel('Y', fontsize=18)
ax.legend()
plt.title('Validation Data (Actual vs Predicted)')
plt.show()