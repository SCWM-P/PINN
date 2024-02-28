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
print("===Using device===")
print(f"=====  {device}  =====")


def HotPixel_brush(xEvent, T, yEvent):
    """
    :param xEvent: Event data of x
    :param T: Event data of time
    :param yEvent: Event data of y
    :return: filtered event data
    This function is used to brush the data via 3-sigma rule
    """
    df = pd.DataFrame({'T': T, 'xEvent': xEvent, 'yEvent': yEvent})
    df['coords'] = list(zip(df['xEvent'], df['yEvent']))
    grouped = df.groupby('coords')
    event = grouped.agg(activity=('coords', 'size'),
                        continuity=('T', lambda x: np.mean(np.diff(sorted(x))) if len(x) > 1 else np.nan))
    act_mean = event['activity'].mean()
    act_std = event['activity'].std()
    cont_mean = event['continuity'].mean()
    cont_std = event['continuity'].std()
    event_filtered = event[(event['activity'] > act_mean - 3 * act_std) & (event['activity'] < act_mean + 1.5 * act_std) &
                           (event['continuity'] > cont_mean - 3 * cont_std) & (event['continuity'] < cont_mean + 1.5 * cont_std)]
    filtered_events = df[df['coords'].isin(event_filtered.index)]
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
        ======Input & Output======\n
        layers: 与DNN类中相同，定义网络架构\n
        connections: 与DNN类中相同，定义网络连接\n
        device: 'cuda' 或 'cpu'，用于指定运算设备\n
        xEvent: 事件点的x坐标序列\n
        T: 事件点的时间坐标序列\n
        yEvent: 事件点的y坐标序列\n
        validation_ratio: 验证集比例\n
        ======Variable Unification======\n
        1. 学习参数a,b,omega,zeta,gamma,y0\n
        2. 全局坐标: self.xEvent, self.T, self.yEvent\n
        3. 划分训练集与验证集:\n
        self.x_train, self.T_train, self.y_train\n
        self.x_val, self.T_val, self.y_val\n
        =================================
        """
        # Configuration
        self.device = device
        self.dnn = DNN(layers, connections).to(device)
        # Learning Parameters
        self.a = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.b = torch.nn.Parameter(torch.tensor([0.0], device=device))
        self.omega = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.zeta = torch.nn.Parameter(torch.tensor([0.05], device=device))
        self.gamma = torch.nn.Parameter(torch.tensor([torch.pi / 4], device=device))
        self.y0 = torch.nn.Parameter(torch.mean(yEvent))
        self.T = T
        self.xEvent = xEvent
        self.yEvent = yEvent
        # Split Data
        self.x_train, self.x_val, self.T_train, self.T_val, self.y_train, self.y_val = \
            train_test_split(xEvent.unsqueeze(1), T.unsqueeze(1), yEvent.unsqueeze(1),
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
            lr=0.003, betas=(0.9, 0.999)
        )

    def net_u(self, xEvent, T):
        y_pred = self.dnn(torch.cat([xEvent, T], dim=1))
        return y_pred

    def normalize(self, x, T, y):
        """
        标准化算子
        """
        x = (x - torch.mean(xEvent)) / torch.std(xEvent)
        T = (T - torch.mean(T)) / torch.std(T)
        y = (y - torch.mean(yEvent)) / torch.std(yEvent)
        return x, T, y

    def rotate(self, xEvent, T, yEvent, gamma):
        """
        旋转算子，实现坐标的旋转变换：\n
         [x‘]      [cosγ, 0,sinγ]     [x]\n
         [y’]   =  [0   , 1,0   ]  ×  [y]\n
         [z‘]      [-sinγ,0,cosγ]     [z]\n
        """
        if xEvent.dim() == 1:
            xEvent = xEvent.unsqueeze(1)
        if T.dim() == 1:
            T = T.unsqueeze(1)
        if yEvent.dim() == 1:
            yEvent = yEvent.unsqueeze(1)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)
        xEvent = cos_gamma * xEvent + sin_gamma * yEvent
        yEvent = -sin_gamma * xEvent + cos_gamma * yEvent
        # # Rotation Matrix
        # zeros = torch.zeroes_like(gamma)
        # ones = torch.ones_like(gamma)
        # gamma_mat = torch.cat([
        #     torch.stack([cos_gamma,  zeros, sin_gamma], dim=1),
        #     torch.stack([zeros,      ones,      zeros], dim=1),
        #     torch.stack([-sin_gamma, zeros, cos_gamma], dim=1)
        # ], dim=0)
        #
        # xTy = torch.cat([xEvent, T, yEvent], dim=1)
        # xTy = torch.matmul(gamma_mat, xTy.t()).t()
        # xEvent = xTy[:,0]
        # T = xTy[:,1]
        # yEvent = xTy[:,2]
        return xEvent, T, yEvent

    def PIloss(self, xEvent, T, yEvent):
        """
        定义基于物理方程的残差函数\n
        接受标准化的数据\n
        """
        xEvent, T, yEvent = self.rotate(xEvent, T, yEvent, self.gamma)
        y_loss = yEvent - torch.sin(self.a * xEvent + self.b) *\
                torch.sin(self.omega * T * torch.sqrt(1 - self.zeta**2)) *\
                torch.exp(-self.zeta * self.omega * T) - self.y0
        PIloss = torch.nn.functional.mse_loss(y_loss, torch.zeros_like(y_loss))
        return PIloss

    def rotateLoss(self, xEvent, T, yEvent):
        """
        定义旋转误差
        :param xEvent
        :param T
        :param yEvent
        """
        xEvent, T, yEvent = self.rotate(xEvent, T, yEvent, self.gamma)
        rotate_loss = torch.std(yEvent)
        return rotate_loss

    def plot_results(self, epoch):
        self.dnn.eval()
        with torch.no_grad():
            y_val_mean = self.y_val.mean()
            y_val_std = self.y_val.std()
            # Natural Network
            x_val, T_val, _ = self.normalize(self.x_val, self.T_val, self.y_val)
            y_pred_val = self.predict(x_val, T_val).flatten()

            # Physical Equation
            x_val_R, T_val_R, y_val_R = self.rotate(self.x_val, self.T_val, self.y_val, self.gamma)
            y_pred_PI = torch.sin(self.a * x_val_R + self.b) * \
                        torch.sin(self.omega * T_val_R * torch.sqrt(1 - self.zeta ** 2)) * \
                        torch.exp(-self.zeta * self.omega * T_val_R) + self.y0
            x_val_R, T_val_R, y_pred_PI = self.rotate(x_val_R, T_val_R, y_pred_PI, -self.gamma)

            # Convert tensor to numpy
            x_val = self.x_val.cpu().detach().numpy()
            y_val = self.y_val.cpu().detach().numpy()
            T_val = self.T_val.cpu().detach().numpy()
            y_pred_PI = y_pred_PI.cpu().detach().numpy()
            y_pred_val = (y_pred_val* y_val_std + y_val_mean).cpu().detach().numpy()

            # Draw pictures
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_val, T_val, y_val, color='b', label='Actual', alpha=0.5)
            ax.scatter(x_val, T_val, y_pred_PI, color='g', label='Predicted (PI)', alpha=0.5)
            ax.scatter(x_val, T_val, y_pred_val, color='r', label='Predicted (NN)', alpha=0.5)
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
            x_train, T_train, y_train = self.normalize(self.x_train, self.T_train, self.y_train)
            # Loss of Natural Network
            y_pred = self.predict(x_train, T_train)
            loss_y = torch.nn.functional.mse_loss(y_pred, y_train)
            # Loss of Rotation
            loss_rotate = self.rotateLoss(self.x_train, self.T_train, self.y_train)
            # Loss of Physical Equation
            loss_PI = self.PIloss(self.x_train, self.T_train, self.y_train)

            loss = loss_y + 100*loss_PI + 10000*loss_rotate  # 这里可以通过调节系数来调整对y和PI的惩罚权重
            loss.backward(retain_graph=True)
            self.optimizer_Adam.step()
            self.history['loss_train'].append(loss.item())
            self.history['a'].append(self.a.item())
            self.history['b'].append(self.b.item())
            self.history['omega'].append(self.omega.item())
            self.history['zeta'].append(self.zeta.item())
            self.history['y0'].append(self.y0.item())
            self.history['gamma'].append(self.gamma.item())

            if epoch % 50 == 0:
                # 在验证集上评估模型
                self.dnn.eval()
                with torch.no_grad():
                    x_val, T_val, y_val = self.normalize(self.x_val, self.T_val, self.y_val)
                    y_pred_val = self.net_u(x_val, T_val)
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
            if epochs <= 10000:
                if epoch % 1000 == 0:
                    self.plot_results(epoch)
            else:
                if epoch <= 10000:
                    if epoch % 1000 == 0:
                        self.plot_results(epoch)
                elif epoch % (epochs // 10) == 0:
                    if epoch <= 30000:
                        self.plot_results(epoch)

            # 为了避免物理方程被拟合成平面，对a参数对逐渐减弱的扰动🤔
            self.a = torch.nn.Parameter(self.a + ((torch.rand((1,)) - 0.5) * epoch/epochs).to(self.device))


    def predict(self, xEvent, T):
        """
        使用训练好的模型进行预测
        """
        self.dnn.eval()
        y_pred = self.net_u(xEvent, T)
        return y_pred


#%% Configurations
# Configuration
epochs = 20000
layers = [2, 100, 100, 100, 100, 100, 100, 100, 100, 1]
connections = [0, 1, 2, 3, 4, 5, 5, 5, 5, 2]
# Load the Data
mat = scipy.io.loadmat('test16-1.mat')
T = mat['brushedData'][:, 0]/1e6
xEvent = mat['brushedData'][:, 1]
yEvent = mat['brushedData'][:, 2]
# Brush the data
(xEvent, T, yEvent) = HotPixel_brush(xEvent, T, yEvent)
# Convert to torch tensors
T = torch.tensor(T, dtype=torch.float32, device=device)
xEvent = torch.tensor(xEvent, dtype=torch.float32, device=device)
yEvent = torch.tensor(yEvent, dtype=torch.float32, device=device)

print('=======Data Loading Done!=======')
print('===== Model Initialization =====')
pinn = PhysicsInformedNN(layers, connections, device, xEvent, T, yEvent)
print('========= Model Training =======')
# 训练模型
start_time = time.time()
pinn.train(epochs)
end_time = time.time()
print('============Model Training Done!===========')
print("========Training time: {:.2f} seconds======".format(end_time - start_time))
print('====Average time per epoch: {:.3f} seconds==='.format((end_time - start_time)/epochs))
print('===========================================')
#%% Draw the Final Result
# 绘制参数变化曲线
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

# 绘制损失变化曲线
plt.figure(figsize=(12, 8))
plt.plot(pinn.history['loss_train'], label='Training Loss')
plt.plot(pinn.history['loss_val'], label='Validation Loss')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend()
plt.title('Training and Validation Loss', fontsize=20)

# 绘制验证集的三维散点图
pinn.plot_results(epochs)