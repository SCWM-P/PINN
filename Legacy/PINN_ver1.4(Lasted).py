# 导入必要的库
import torch
import time
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果可重复
np.random.seed(1234)
torch.manual_seed(1234)

# 检查CUDA可用性（用于GPU加速）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===Using device===\n", device)


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
            [torch.nn.Linear(layers[i], layers[i + 1]) for i in range(self.num_layers - 1)])
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
    def __init__(self, layers, connections, device, T, xEvent, yEvent, validation_ratio=0.2):
        """
        layers: 与DNN类中相同，定义网络架构\n
        connections: 与DNN类中相同，定义网络连接\n
        device: 'cuda' 或 'cpu'，用于指定运算设备
        """
        # Configuration
        self.device = device
        self.dnn = DNN(layers, connections).to(device)
        # Learning Parameters
        self.a = torch.nn.Parameter(torch.tensor([1.5], device=device))
        self.b = torch.nn.Parameter(torch.tensor([-1.0], device=device))
        self.omega = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.zeta = torch.nn.Parameter(torch.tensor([0.05], device=device))
        self.y0 = torch.nn.Parameter(torch.tensor([1.0], device=device))
        self.gamma = torch.nn.Parameter(torch.tensor([torch.pi / 4], device=device))
        # Data
        self.T = T.clone().detach().requires_grad_(True).to(device).float()
        self.xEvent = xEvent.clone().detach().requires_grad_(True).to(device).float()
        self.yEvent =  yEvent.clone().detach().requires_grad_(True).to(device).float()
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
            lr=0.001, betas=(0.9, 0.999)
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
        y_loss = yEvent - torch.sin(self.a * xEvent + self.b) *\
                torch.sin(self.omega * T * torch.sqrt(1 - self.zeta**2)) *\
                torch.exp(-self.zeta * self.omega * T) - self.y0
        return torch.nn.functional.mse_loss(y_loss, torch.zeros_like(y_loss))

    def rotateLoss(self, xEvent, T, yEvent):
        xEvent, T, yEvent = self.rotate(xEvent, T, yEvent, self.gamma)
        rotate_loss = torch.nn.functional.mse_loss(yEvent, torch.zeros_like(yEvent))
        return rotate_loss

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
            'epoch_time': [],
            'y0': [],
            'gamma':[]
        }
        for epoch in range(epochs):
            start_time = time.time()
            self.dnn.train()
            self.optimizer_Adam.zero_grad()
            y_pred = self.net_u(self.x_train, self.T_train)
            loss_rotate =  self.rotateLoss(self.x_train, self.T_train, self.y_train)
            loss_y = torch.nn.functional.mse_loss(y_pred, self.y_train)
            loss_PI = self.PIloss(self.x_train, self.T_train, self.y_train)

            loss = loss_y + 100*loss_PI + 10000*loss_rotate # 这里可以通过调节系数来调整对y和PI的惩罚权重
            loss.backward(retain_graph=True)
            self.optimizer_Adam.step()
            epoch_time = time.time() - start_time
            self.history['epoch_time'].append(epoch_time)
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
                time10 = np.sum(self.history['epoch_time'][-10:])
                print(f'=======Epoch {epoch}=======\n'
                        f'Training Loss: {loss.item():.3f},Validation Loss: {loss_val.item():.3f},Time: {time10:.2f}s\n'
                        f'a:{self.a.item():.3f},a_grad:{self.a.grad}\n'
                        f'b:{self.b.item():.3f},b_grad:{self.b.grad}\n'
                        f'omega:{self.omega.item():.3f},omega_grad:{self.omega.grad}\n'
                        f'zeta:{self.zeta.item():.3f},zeta_grad:{self.zeta.grad}\n'
                        f'y0:{self.y0.item():.3f},y0_grad:{self.y0.grad}\n'
                        f'gamma:{self.gamma.item():.3f},gamma_grad:{self.gamma.grad}\n'
                      )

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
# 读取数据
T = mat['brushedData'][:,0]
xEvent = mat['brushedData'][:,1]
yEvent = mat['brushedData'][:,2]
# 标准化
T = (T - np.mean(T)) / np.std(T)
xEvent = (xEvent - np.mean(xEvent)) / np.std(xEvent)
yEvent = (yEvent - np.mean(yEvent)) / np.std(yEvent)
# 转化为张量
T = torch.tensor(T, dtype=torch.float32, device = device)
xEvent = torch.tensor(xEvent, dtype=torch.float32, device= device)
yEvent = torch.tensor(yEvent, dtype=torch.float32, device = device)

layers = [2, 100, 102, 100, 100, 100, 100, 100, 100, 1]
connections = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]
print('=======Data Loading Done!=======')
print('======Model Initialization=====')
pinn = PhysicsInformedNN(layers, connections , device, T, xEvent, yEvent)
print('=======Model Training=======')
# 训练模型
pinn.train(10000)


x_val = pinn.x_val.cpu().detach().numpy()
y_val = pinn.y_val.cpu().detach().numpy()
T_val = pinn.T_val.cpu().detach().numpy()

# 绘制参数变化
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.plot(pinn.history['a'], label='a')
plt.plot(pinn.history['b'], label='b')
plt.plot(pinn.history['omega'], label='ω')
plt.plot(pinn.history['zeta'], label='ζ')
plt.plot(pinn.history['y0'], label='y0')
plt.plot(pinn.history['gamma'],label='γ')
plt.xlabel('Epoch')
plt.ylabel('Parameter Value')
plt.legend()
plt.title('Parameter Evolution')

# 绘制损失变化
plt.subplot(222)
plt.plot(pinn.history['loss_train'], label='Training Loss')
plt.plot(pinn.history['loss_val'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# 绘制时间变化
plt.subplot(223)
plt.plot(pinn.history['epoch_time'])
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.title('Time per Epoch')

plt.tight_layout()
plt.show()

# 绘制验证集的三维散点图
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
ax.scatter(x_val, T_val, y_pred_val, color='r', label='Predicted')
ax.set_xlabel('X')
ax.set_ylabel('T (s)')
ax.set_zlabel('Y')
ax.legend()
plt.title('Validation Data (Actual vs Predicted)')
plt.show()
