import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置字体以支持中文显示
plt.rc('font', family='SimHei', size=14)

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
            torch.nn.ReLU(),
            torch.nn.Linear(80, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 1),
        ).to(device)

    def forward(self, x):
        return self.network(x)


def custom_func(x):
    '''
    y = sin x \cdot e^{-\frac{x}{3}}
    \frac{dy}{dx} = (cos x - \frac{sin x}{3}) \cdot e^{-\frac{x}{3}}
    \frac{d^2y}{dx^2} = (-2/3 \cdot cos x - 8/9 \cdot sin x) \cdot e^{-\frac{x}{3}}
    \frac{d^3y}{dx^3} = (26/27 \cdot sin x - 2/3 \cdot cos x) \cdot e^{-\frac{x}{3}}
    '''
    y = torch.sin(x) * torch.exp(-x / 3)
    dy_dx = (torch.cos(x) - torch.sin(x) / 3) * torch.exp(-x / 3)
    d2y_dx2 = (-2/3 * torch.cos(x) - 8/9 * torch.sin(x)) * torch.exp(-x / 3)
    d3y_dx3 = (26/27 * torch.sin(x) - 2/3 * torch.cos(x)) * torch.exp(-x / 3)
    return y, dy_dx, d2y_dx2, d3y_dx3


model = SimpleModel()
optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-15,
            weight_decay=0.0001
        )
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

# 定义数据
x_train = torch.linspace(-20, 2, 120, device=device, requires_grad=True).view(-1, 1)
y_train, dy_dx_train, d2y_dx2_train, d3y_dx3_train = custom_func(x_train)

# 创建多个子图
fig, axs = plt.subplots(4, 1, figsize=(10, 15))
# 在第一张图上绘制
true_line, = axs[0].plot(
    x_train.cpu().detach().numpy(),
    y_train.cpu().detach().numpy(),
    label=r'真实曲线$y = sin x \cdot e^{-\frac{x}{3}}$'
)
scatter_true = axs[0].scatter(
    x_train.cpu().detach().numpy()[::5],
    y_train.cpu().detach().numpy()[::5],
    color='red', label='真实点'
)
pred_line, = axs[0].plot([], [], 'g--', label='预测曲线')
scatter_pred = axs[0].scatter([], [], color='blue', label='预测点')

# 在第二张图上绘制
true_derivative, = axs[1].plot(
    x_train.cpu().detach().numpy(),
    dy_dx_train.cpu().detach().numpy(),
    label=r'真实一阶导数$\frac{dy}{dx} = (cos x- \frac{sin x}{3}) \cdot e^{-\frac{x}{3}}$'
)

pred_derivative_line, = axs[1].plot([], [], 'b--', label='预测导数')
# 初始化文本标注
annotations = []


# 更新动画的函数
def update(frame):
    for _ in range(update_freq):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_train)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

    pred_curve = model(x_train).detach().cpu()
    axs[0].clear()
    axs[0].plot(x_train.cpu().numpy(), y_train.cpu().numpy(), label='真实曲线')
    axs[0].plot(x_train.cpu().numpy(), pred_curve.numpy(), 'g--', label='预测曲线')
    axs[0].set_title("原始函数")
    axs[0].legend()

    # 计算并更新一阶导数的预测
    y_pred_grad = torch.autograd.grad(y_pred, x_train, grad_outputs=torch.ones_like(y_train), create_graph=True)[0]
    axs[1].clear()
    axs[1].plot(x_train.cpu().numpy(), dy_dx_train.cpu().numpy(), label='真实一阶导数')
    axs[1].plot(x_train.cpu().numpy(), y_pred_grad.detach().cpu().numpy(), 'r--', label='预测一阶导数')
    axs[1].set_title("一阶导数")
    axs[1].legend()

    # 如果需要
    y_pred_2nd_grad = \
    torch.autograd.grad(y_pred_grad, x_train, grad_outputs=torch.ones_like(y_pred_grad), create_graph=True)[0]
    axs[2].clear()
    axs[2].plot(x_train.cpu().numpy(), d2y_dx2_train.cpu().numpy(), label='真实二阶导数')
    axs[2].plot(x_train.cpu().numpy(), y_pred_2nd_grad.detach().cpu().numpy(), 'b--', label='预


    with torch.no_grad():
        # 更新预测曲线
        pred_curve = model(x_train)
        pred_line.set_data(x_train.cpu().detach().numpy(), pred_curve.detach().numpy())

        # 更新预测点
        scatter_pred.set_offsets(np.c_[x_train.cpu().detach().numpy()[::5], pred_curve.numpy()[::5]])

    # 计算预测导数
    pred_dy_dx_curve = torch.autograd.grad(
        model(x_train), x_train,
        grad_outputs=torch.ones_like(y_train),
        create_graph=True,
        retain_graph=True
    )[0].detach().cpu()
    pred_derivative_line.set_data(x_train.cpu().detach().numpy(), pred_dy_dx_curve.numpy())

    for ann in annotations:
        ann.remove()
    annotations.clear()
    for i in range(0, len(x_train), 20):
        annotations.append(
            ax.text(x_train[i].cpu(), y_train[i].cpu(), f'({x_train[i].item():.2f}, {y_train[i].item():.2f})',
                    fontsize=16))

    # 更新标题
    ax.set_title(f'Epoch: {frame}/{epochs}, LR: {scheduler.get_last_lr()[0]:.5f}')

    return true_line, true_derivative, scatter_true, scatter_pred, pred_line, pred_derivative_line, annotations


ani = FuncAnimation(fig, update, frames=epochs, blit=False, interval=20)


def toggle_animation(event):
    global isRunning
    if event.key == 'm':
        if isRunning:
            ani.pause()
            isRunning = False
        else:
            ani.resume()
            isRunning = True


fig.canvas.mpl_connect('key_press_event', toggle_animation)
plt.legend()
plt.show()
