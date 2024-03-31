import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查并使用CUDA，如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 定义全局变量
learning_rate = 0.01
epochs = 1000
update_freq = 50  # 动画更新频率
torch.manual_seed(2020)  # 确保实验可重复
isRunning = True


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 80),
            torch.nn.Tanh(),
            torch.nn.Linear(80, 80),
            torch.nn.Tanh(),
            torch.nn.Linear(80, 80),
            torch.nn.Tanh(),
            torch.nn.Linear(80, 80),
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


def toggle_animation(event):
    global isRunning
    if event.key == 'm':
        if isRunning:
            ani.pause()
            isRunning = False
        else:
            ani.resume()
            isRunning = True


def custom_func(x):
    '''
    y = sin x \cdot e^{-\frac{x}{3}}
    \frac{dy}{dx} = (cos x - \frac{sin x}{3}) \cdot e^{-\frac{x}{3}}
    \frac{d^2y}{dx^2} = (-\frac{2}{3} \cdot cos x - \frac{8}{9} \cdot sin x) \cdot e^{-\frac{x}{3}}
    \frac{d^3y}{dx^3} = (\frac{26}{27} \cdot sin x - \frac{2}{3} \cdot cos x) \cdot e^{-\frac{x}{3}}
    \frac{d^4y}{dx^4} = (\frac{28}{81} \cdot sin x + \frac{32}{27} \cdot cos x) \cdot e^{-\frac{x}{3}}
    '''
    y = torch.sin(x) * torch.exp(-x / 3)
    dy_dx = (torch.cos(x) - torch.sin(x) / 3) * torch.exp(-x / 3)
    d2y_dx2 = (-2 / 3 * torch.cos(x) - 8 / 9 * torch.sin(x)) * torch.exp(-x / 3)
    d3y_dx3 = (26 / 27 * torch.sin(x) - 2 / 3 * torch.cos(x)) * torch.exp(-x / 3)
    d4y_dx4 = (28 / 81 * torch.sin(x) + 32 / 27 * torch.cos(x)) * torch.exp(-x / 3)
    return y, dy_dx, d2y_dx2, d3y_dx3, d4y_dx4


model = SimpleModel()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    eps=1e-15,
    weight_decay=0.0001
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
x_train = torch.linspace(-20, 2, 240, device=device, requires_grad=True).view(-1, 1)
y_train, dy_dx_train, d2y_dx2_train, d3y_dx3_train, d4y_dx4_train = custom_func(x_train)

fig, axs = plt.subplots(2, 3, figsize=(20, 20))
# 在第一张图上绘制
true_line, = axs[0, 0].plot(
    x_train.cpu().detach().numpy(),
    y_train.cpu().detach().numpy(),
    label=r'真实曲线'
)
scatter_true = axs[0, 0].scatter(
    x_train.cpu().detach().numpy()[::5],
    y_train.cpu().detach().numpy()[::5],
    color='red', label='真实点'
)
pred_line, = axs[0, 0].plot([], [], 'g--', label='预测曲线')
scatter_pred = axs[0, 0].scatter([], [], color='blue', label='预测点')
axs[0, 0].legend()
axs[0, 0].set_title(r'$y = sin x \cdot e^{-\frac{x}{3}}$', fontsize=16)

# 在第二张图上绘制
dy_dx_line, = axs[0, 1].plot(
    x_train.cpu().detach().numpy(),
    dy_dx_train.cpu().detach().numpy(),
    label=r'真实一阶导数'
)
scatter_dy_dx = axs[0, 1].scatter(
    x_train.cpu().detach().numpy()[::5],
    dy_dx_train.cpu().detach().numpy()[::5],
    color='red', label='真实一阶导数点'
)
dy_dx_pred_line, = axs[0, 1].plot([], [], 'g--', label='预测一阶导数')
scatter_dy_dx_pred = axs[0, 1].scatter([], [], color='blue', label='预测一阶导数点')
axs[0, 1].legend()
axs[0, 1].set_title(r'$\frac{dy}{dx} = (cos x - \frac{sin x}{3}) \cdot e^{-\frac{x}{3}}$', fontsize=16)

# 在第三张图上绘制
d2y_dx2_line, = axs[0, 2].plot(
    x_train.cpu().detach().numpy(),
    d2y_dx2_train.cpu().detach().numpy(),
    label=r'真实二阶导数'
)
scatter_d2y_dx2 = axs[0, 2].scatter(
    x_train.cpu().detach().numpy()[::5],
    d2y_dx2_train.cpu().detach().numpy()[::5],
    color='red', label='真实二阶导数点'
)
d2y_dx2_pred_line, = axs[0, 2].plot([], [], 'g--', label='预测二阶导数')
scatter_d2y_dx2_pred = axs[0, 2].scatter([], [], color='blue', label='预测二阶导数点')
axs[0, 2].legend()
axs[0, 2].set_title(r'$\frac{d^2y}{dx^2} = (-\frac{2}{3} \cdot cos x - \frac{8}{9} \cdot sin x) \cdot e^{-\frac{x}{3}}$', fontsize=16)

# 在第四张图上绘制
d3y_dx3_line, = axs[1, 0].plot(
    x_train.cpu().detach().numpy(),
    d3y_dx3_train.cpu().detach().numpy(),
    label=r'真实三阶导数'
)
scatter_d3y_dx3 = axs[1, 0].scatter(
    x_train.cpu().detach().numpy()[::5],
    d3y_dx3_train.cpu().detach().numpy()[::5],
    color='red', label='真实三阶导数点'
)
d3y_dx3_pred_line, = axs[1, 0].plot([], [], 'g--', label='预测三阶导数')
scatter_d3y_dx3_pred = axs[1, 0].scatter([], [], color='blue', label='预测三阶导数点')
axs[1, 0].legend()
axs[1, 0].set_title(r'$\frac{d^2y}{dx^2} = (-\frac{2}{3} \cdot cos x - \frac{8}{9} \cdot sin x) \cdot e^{-\frac{x}{3}}$', fontsize=16)

# 在第五张图上绘制
d4y_dx4_line, = axs[1, 1].plot(
    x_train.cpu().detach().numpy(),
    d4y_dx4_train.cpu().detach().numpy(),
    label=r'真实四阶导数'
)
scatter_d4y_dx4 = axs[1, 1].scatter(
    x_train.cpu().detach().numpy()[::5],
    d4y_dx4_train.cpu().detach().numpy()[::5],
    color='red', label='真实四阶导数点'
)
d4y_dx4_pred_line, = axs[1, 1].plot([], [], 'g--', label='预测四阶导数')
scatter_d4y_dx4_pred = axs[1, 1].scatter([], [], color='blue', label='预测四阶导数点')
axs[1, 1].legend()
axs[1, 1].set_title(r'$\frac{d^4y}{dx^4} = (\frac{28}{81} \cdot sin x + \frac{32}{27} \cdot cos x) \cdot e^{-\frac{x}{3}}$', fontsize=16)


# 更新动画的函数
def update(frame):
    for _ in range(update_freq):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_train)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        # 计算原函数
        pred_curve = model(x_train).cpu().detach().numpy()
        pred_line.set_data(x_train.cpu().detach().numpy(), pred_curve)
        scatter_pred.set_offsets(np.c_[x_train.cpu().detach().numpy()[::5], pred_curve[::5]])

    # 计算一阶导数
    pred_dy_dx_curve = torch.autograd.grad(
        model(x_train), x_train,
        grad_outputs=torch.ones_like(x_train),
        create_graph=True,
        retain_graph=True
    )[0]
    dy_dx_pred_line.set_data(x_train.cpu().detach().numpy(), pred_dy_dx_curve.cpu().detach().numpy())
    scatter_dy_dx.set_offsets(np.c_[x_train.cpu().detach().numpy()[::5], pred_dy_dx_curve.cpu().detach().numpy()[::5]])

    # 计算二阶导数
    pred_d2y_dx2_curve = torch.autograd.grad(
        pred_dy_dx_curve, x_train,
        grad_outputs=torch.ones_like(pred_dy_dx_curve),
        create_graph=True,
        retain_graph=True
    )[0]
    d2y_dx2_pred_line.set_data(x_train.cpu().detach().numpy(), pred_d2y_dx2_curve.cpu().detach().numpy())
    scatter_d2y_dx2.set_offsets(np.c_[x_train.cpu().detach().numpy()[::5], pred_d2y_dx2_curve.cpu().detach().numpy()[::5]])

    # 计算三阶导数
    pred_d3y_dx3_curve = torch.autograd.grad(
        pred_d2y_dx2_curve, x_train,
        grad_outputs=torch.ones_like(pred_d2y_dx2_curve),
        create_graph=True,
        retain_graph=True
    )[0]
    d3y_dx3_pred_line.set_data(x_train.cpu().detach().numpy(), pred_d3y_dx3_curve.cpu().detach().numpy())
    scatter_d3y_dx3.set_offsets(
        np.c_[x_train.cpu().detach().numpy()[::5], pred_d3y_dx3_curve.cpu().detach().numpy()[::5]])

    # 计算四阶导数
    pred_d4y_dx4_curve = torch.autograd.grad(
        pred_d3y_dx3_curve, x_train,
        grad_outputs=torch.ones_like(pred_d3y_dx3_curve),
        create_graph=True,
        retain_graph=True
    )[0]
    d4y_dx4_pred_line.set_data(x_train.cpu().detach().numpy(), pred_d4y_dx4_curve.cpu().detach().numpy())
    scatter_d4y_dx4.set_offsets(
        np.c_[x_train.cpu().detach().numpy()[::5], pred_d4y_dx4_curve.cpu().detach().numpy()[::5]])

    fig.suptitle(f'Epoch: {frame}/{epochs}, LR: {scheduler.get_last_lr()[0]:.5f}', fontsize=20)

    return (pred_line, scatter_pred,
            dy_dx_pred_line, scatter_dy_dx,
            d2y_dx2_pred_line, scatter_d2y_dx2,
            d3y_dx3_pred_line, scatter_d3y_dx3,
            d4y_dx4_pred_line, scatter_d4y_dx4)


ani = FuncAnimation(fig, update, frames=epochs, blit=False, interval=20)
fig.canvas.mpl_connect('key_press_event', toggle_animation)
plt.show()
