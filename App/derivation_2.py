"""
用于测试二元函数在使用PINN方法进行拟合与测算时候的效果
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查并使用 CUDA，如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(r"Pytorch支持的CUDA版本", torch.version.cuda, "\t使用设备:", device)

# 定义全局变量
learning_rate = 0.001
epochs = 1000
update_freq = 50  # 动画更新频率
torch.manual_seed(2020)  # 确保实验可重复
isRunning = True


# 定义二元函数和物理约束方程
def custom_func(x, y):
    r"""
    定义函数:
    z = (2x + y) * sin(x + y)

    以及物理约束方程:
    \frac{\partial^2 z}{\partial x^2} - 2 \cdot \frac{\partial^2 z}{\partial y^2} - z = 0
    """
    z = (2 * x + y) * torch.sin(x + y)
    dz_dx = 2 * torch.sin(x + y) + (2 * x + y) * torch.cos(x + y)
    dz_dy = torch.sin(x + y) + (2 * x + y) * torch.cos(x + y)
    d2z_dx2 = 4 * torch.cos(x + y) - (2 * x + y) * torch.sin(x + y)
    d2z_dy2 = 2 * torch.cos(x + y) - (2 * x + y) * torch.sin(x + y)
    return z, dz_dx, dz_dy, d2z_dx2, d2z_dy2


def derivative_func(model, xy_train):
    r"""
    计算 z 对 x 和 y 的各阶导数
    """
    z_pred = model(xy_train)
    grads = torch.autograd.grad(
        outputs=z_pred,
        inputs=xy_train,
        grad_outputs=torch.ones_like(z_pred),
        create_graph=True
    )[0]
    dz_dx_pred = grads[:, 0]
    dz_dy_pred = grads[:, 1]
    d2z_dx2_pred = torch.autograd.grad(
        outputs=dz_dx_pred,
        inputs=xy_train,
        grad_outputs=torch.ones_like(dz_dx_pred),
        create_graph=True
    )[0][:, 0:1]
    d2z_dy2_pred = torch.autograd.grad(
        outputs=dz_dy_pred,
        inputs=xy_train,
        grad_outputs=torch.ones_like(dz_dy_pred),
        create_graph=True
    )[0][:, 1:2]
    return z_pred, dz_dx_pred, dz_dy_pred, d2z_dx2_pred, d2z_dy2_pred


# 定义神经网络模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 80),
            torch.nn.Tanh(),
            torch.nn.Linear(80, 80),
            torch.nn.Tanh(),
            torch.nn.Linear(80, 80),
            torch.nn.Tanh(),
            torch.nn.Linear(80, 80),
            torch.nn.Tanh(),
            torch.nn.Linear(80, 1),
        ).to(device)

    def forward(self, x):
        return self.network(x)


# 定义动画的暂停与继续
def toggle_animation(event):
    global isRunning
    if event.key == 'm':
        if isRunning:
            ani.event_source.stop()
            isRunning = False
        else:
            ani.event_source.start()
            isRunning = True


# 生成训练数据
x_train = torch.linspace(-10, 10, 500, device=device, requires_grad=True)
y_train = torch.linspace(-10, 10, 500, device=device, requires_grad=True)
X, Y = torch.meshgrid(x_train, y_train, indexing='ij')
x_train_flat = X.flatten().unsqueeze(1)
y_train_flat = Y.flatten().unsqueeze(1)
xy_train = torch.cat([x_train_flat, y_train_flat], dim=1).requires_grad_(True)
z_train, dz_dx, dz_dy, d2z_dx2, d2z_dy2 = custom_func(x_train_flat, y_train_flat)

# 初始化模型、优化器和学习率调度器
model = SimpleModel().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    eps=1e-15,
    weight_decay=0.0001
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
loss_history = np.zeros(epochs)

# 创建绘图窗口和子图
fig = plt.figure(figsize=(20, 10))
axs = [fig.add_subplot(2, 3, i+1, projection='3d') if i != 5 else fig.add_subplot(2, 3, i+1) for i in range(6)]

# 绘制真实的 z 函数
Z_true = z_train.cpu().detach().numpy().reshape(X.shape)
surf_true = axs[0].plot_surface(
    X.cpu().detach().numpy(),
    Y.cpu().detach().numpy(),
    Z_true, cmap='viridis', alpha=0.6
)
axs[1].plot_surface(
    X.cpu().detach().numpy(),
    Y.cpu().detach().numpy(),
    dz_dx.cpu().detach().numpy().reshape(X.shape),
    cmap='viridis', alpha=0.6
)
axs[2].plot_surface(
    X.cpu().detach().numpy(),
    Y.cpu().detach().numpy(),
    dz_dy.cpu().detach().numpy().reshape(X.shape),
    cmap='viridis', alpha=0.6
)
axs[3].plot_surface(
    X.cpu().detach().numpy(),
    Y.cpu().detach().numpy(),
    d2z_dx2.cpu().detach().numpy().reshape(X.shape),
    cmap='viridis', alpha=0.6
)
axs[4].plot_surface(
    X.cpu().detach().numpy(),
    Y.cpu().detach().numpy(),
    d2z_dy2.cpu().detach().numpy().reshape(X.shape),
    cmap='viridis', alpha=0.6
)
axs[0].set_title(r'$z= z(x,y) = (2x + y) \cdot \sin(x + y)$', fontsize=16)
axs[1].set_title(r'$\frac{\partial z}{\partial x}=2 \sin (x+y) + (2x+y) \cos (x+y)$', fontsize=16)
axs[2].set_title(r'$\frac{\partial z}{\partial y}=\sin (x+y) + (2x+y) \cos (x+y)$', fontsize=16)
axs[3].set_title(r'$\frac{\partial^2 z}{\partial x^2}=4 \cos (x+y) - (2x+y) \sin (x+y)$', fontsize=16)
axs[4].set_title(r'$\frac{\partial^2 z}{\partial y^2}=2 \cos (x+y) - (2x+y) \sin (x+y)$', fontsize=16)

# 初始化预测的 z 函数图像
Z_pred = np.zeros_like(Z_true)
surf_pred = [
    axs[0].plot_surface(
        X.cpu().detach().numpy(),
        Y.cpu().detach().numpy(),
        Z_pred, cmap='rainbow', alpha=0.8
    ),
    axs[1].plot_surface(
        X.cpu().detach().numpy(),
        Y.cpu().detach().numpy(),
        Z_pred, cmap='rainbow', alpha=0.8
    ),
    axs[2].plot_surface(
        X.cpu().detach().numpy(),
        Y.cpu().detach().numpy(),
        Z_pred, cmap='rainbow', alpha=0.8
    ),
    axs[3].plot_surface(
        X.cpu().detach().numpy(),
        Y.cpu().detach().numpy(),
        Z_pred, cmap='rainbow', alpha=0.8
    ),
    axs[4].plot_surface(
        X.cpu().detach().numpy(),
        Y.cpu().detach().numpy(),
        Z_pred, cmap='rainbow', alpha=0.8
    )
]
axs[0].legend([surf_true, surf_pred[0]], ['真实函数', '预测函数'])
# 绘制损失函数曲线
loss_line, = axs[5].plot([], [], color='purple', label='损失函数')
axs[5].legend()
axs[5].set_title('损失函数', fontsize=16)
axs[5].set_xlim(0, epochs)
axs[5].set_ylim(0, 1)


# 更新动画的函数
def update(frame):
    global surf_pred
    for _ in tqdm(range(update_freq)):
        optimizer.zero_grad()
        # 模型预测
        z_pred, dz_dx_pred, dz_dy_pred, d2z_dx2_pred, d2z_dy2_pred = derivative_func(model, xy_train)
        # 物理约束方程
        physics = d2z_dx2_pred - 2 * d2z_dy2_pred - z_pred
        # 计算损失函数
        mse_loss = torch.nn.functional.mse_loss(z_pred, z_train)
        physics_loss = torch.mean(physics**2)
        loss = mse_loss + physics_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    loss_history[frame] = loss.item()

    # 更新预测的 z 函数图像
    z_pred, dz_dx_pred, dz_dy_pred, d2z_dx2_pred, d2z_dy2_pred = derivative_func(model, xy_train)
    with torch.no_grad():
        # 清除之前的图像
        for surf in surf_pred:
            surf.remove()
        # 绘制新的预测图像
        surf_pred[0] = axs[0].plot_surface(
            X.cpu().detach().numpy(),
            Y.cpu().detach().numpy(),
            z_pred.cpu().detach().numpy().reshape(X.shape),
            cmap='rainbow', alpha=0.8
        )
        surf_pred[1] = axs[1].plot_surface(
            X.cpu().detach().numpy(),
            Y.cpu().detach().numpy(),
            dz_dx_pred.cpu().detach().numpy().reshape(X.shape),
            cmap='rainbow', alpha=0.8
        )
        surf_pred[2] = axs[2].plot_surface(
            X.cpu().detach().numpy(),
            Y.cpu().detach().numpy(),
            dz_dy_pred.cpu().detach().numpy().reshape(X.shape),
            cmap='rainbow', alpha=0.8
        )
        surf_pred[3] = axs[3].plot_surface(
            X.cpu().detach().numpy(),
            Y.cpu().detach().numpy(),
            d2z_dx2_pred.cpu().detach().numpy().reshape(X.shape),
            cmap='rainbow', alpha=0.8
        )
        surf_pred[4] = axs[4].plot_surface(
            X.cpu().detach().numpy(),
            Y.cpu().detach().numpy(),
            d2z_dy2_pred.cpu().detach().numpy().reshape(X.shape),
            cmap='rainbow', alpha=0.8
        )
        axs[0].set_title(r'$z = (2x + y) \cdot \sin(x + y)$', fontsize=16)
        axs[0].legend([surf_true, surf_pred[0]], ['真实函数', '预测函数'])

    # 更新损失函数曲线
    loss_line.set_data(np.array(range(1, frame+2)), loss_history[:frame+1])
    axs[5].set_ylim(0, 1.01 * np.max(loss_history[:frame+1]))
    fig.suptitle(f'Epoch: {frame}/{epochs}, LR: {scheduler.get_last_lr()[0]:.5f}', fontsize=20)
    return surf_pred + [loss_line]


# 创建动画
ani = FuncAnimation(fig, update, frames=epochs, blit=False, interval=20)
fig.canvas.mpl_connect('key_press_event', toggle_animation)
plt.show()
