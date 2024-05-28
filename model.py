import os
import torch
import time
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import data_processing as dp


# Plot Configuration
try:
    plt.rc('grid', color='k', alpha=0.2)
    plt.rc('font', family='Times New Roman')
    plt.rc('text', usetex=True)
except Exception as e:
    warnings.warn(e.msg, UserWarning)


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
            layers: list, connections: list, device: torch.device,
            xEvent: torch.Tensor, Timestamp: torch.Tensor, yEvent: torch.Tensor,
            epochs: int,
            *,
            validation_ratio=0.2,
            EI=None, Tension=None, M=None, c=None, history=None
            ):
        from sklearn.model_selection import train_test_split
        # Configuration
        self.device = device
        self.dnn = DNN(layers, connections).to(device)
        # Learning Parameters
        self.def_para = lambda x: torch.nn.Parameter(torch.tensor([x], device=device))
        self.EI = self.def_para(1.0) if EI is None else self.def_para(EI)
        self.Tension = self.def_para(1.0) if Tension is None else self.def_para(Tension)
        self.M = self.def_para(1.0) if M is None else self.def_para(M)
        self.c = self.def_para(1.0) if c is None else self.def_para(c)
        # Data
        self.xEvent = xEvent
        self.Timestamp = Timestamp
        self.yEvent = yEvent
        self.epochs = epochs
        self.lr = 0.1
        # Split train and validation sets
        self.x_train, self.x_val, self.t_train, self.t_val, self.y_train, self.y_val = train_test_split(
            self.xEvent, self.Timestamp, self.yEvent,
            test_size=validation_ratio,
            random_state=42
        )
        # Define the Training Parameters History
        if history is None:
            self.history = {
                'train_loss': [],
                'train_accuracy': [],
                'EI': [],
                'Tension': [],
                'M': [],
                'c': []
            }
        else:
            self.history = history
        # Define Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.dnn.parameters()) + [self.EI, self.Tension, self.M, self.c],
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-15,
            weight_decay=0.0001
        )
        self.exponent_schedule = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.999)

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

    def derive(self, xEvent: torch.Tensor, Timestamp: torch.Tensor, yEvent: torch.Tensor):
        """
        Perform partial differential operations.\n
        The data manipulated by the function are all **torch.Tensor** type.
        :return: partial differential tuple (∂4y/∂x4, ∂2y/∂x2, ∂2y/∂t2, ∂y/∂t)
        """
        xEvent, Timestamp, _ = self.normalize(xEvent, Timestamp, yEvent)
        y = self.predict(xEvent, Timestamp)
        _, _, y = self.denormalize(xEvent, Timestamp, y)
        dy_dx = torch.autograd.grad(
            y, xEvent,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=True,
        )[0] / self.xEvent.std()
        # We need ∂2y/∂x2
        d2y_dx2 = torch.autograd.grad(
            dy_dx, xEvent,
            grad_outputs=torch.ones_like(dy_dx),
            retain_graph=True,
            create_graph=True
        )[0] / self.xEvent.std()
        d3y_dx3 = torch.autograd.grad(
            d2y_dx2, xEvent,
            grad_outputs=torch.ones_like(d2y_dx2),
            retain_graph=True,
            create_graph=True
        )[0] / self.xEvent.std()
        # We need ∂4y/∂x4
        d4y_dx4 = torch.autograd.grad(
            d3y_dx3, xEvent,
            grad_outputs=torch.ones_like(d3y_dx3),
            retain_graph=True,
            create_graph=True
        )[0] / self.xEvent.std()
        # We need ∂y/∂t
        dy_dt = torch.autograd.grad(
            y, Timestamp,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=True
        )[0] / self.Timestamp.std()
        # We need ∂2y/∂t2
        d2y_dt2 = torch.autograd.grad(
            dy_dt, Timestamp,
            grad_outputs=torch.ones_like(dy_dt),
            retain_graph=True,
            create_graph=True
        )[0] / self.Timestamp.std()
        return (
            d4y_dx4.detach(),
            d2y_dx2.detach(),
            d2y_dt2.detach(),
            dy_dt.detach()
        )

    def physicalLoss(self, xEvent: torch.Tensor, Timestamp: torch.Tensor, yEvent: torch.Tensor):
        """
        Calculates the physical loss.\n
        The physical equation is:\n
        EI * ∂4y/∂x4 - T * ∂2y/∂x2 + M * ∂2y/∂t2 + c * ∂y/∂t = 0
        """
        d4y_dx4, d2y_dx2, d2y_dt2, dy_dt = self.derive(xEvent, Timestamp, yEvent)
        y_PI_pred = \
            self.EI * d4y_dx4 - self.Tension * d2y_dx2 + self.M * d2y_dt2 + self.c * dy_dt
        loss_Physical = torch.nn.functional.mse_loss(y_PI_pred, torch.zeros_like(y_PI_pred))
        return loss_Physical

    def plot_results(self, epoch: int, option: str = 'show'):
        """
        Plots the results of the model in a particular epoch.

        This method evaluates the model at the given epoch and plots the predictions against the actual values
        both for the training and validation datasets. It can either show the plot directly or save it as an image
        based on the provided option.

        :param epoch: Current number of model training rounds. Used to label the plot for reference.
        :param option: A string specifying the action to be taken. 'show' to display the plot or 'save' to save it.
                       Defaults to 'show'.
        :return: None.
        """
        # Set the model to evaluation mode
        self.dnn.eval()
        with torch.no_grad():
            # Prepare and plot the results for the training set
            x_train, t_train, y_train = self.normalize(self.x_train, self.t_train, self.y_train)
            y_nn_pred_train = self.predict(x_train, t_train)
            x_train, t_train, y_nn_pred_train = self.denormalize(x_train, t_train, y_nn_pred_train)

            # Prepare and plot the results for the validation set
            x_val, t_val, y_val = self.normalize(self.x_val, self.t_val, self.y_val)
            y_nn_pred_val = self.predict(x_val, t_val)
            x_val, t_val, y_nn_pred_val = self.denormalize(x_val, t_val, y_nn_pred_val)

            # Convert tensors to numpy arrays for plotting
            x_train = x_train.cpu().detach().numpy()
            t_train = t_train.cpu().detach().numpy()
            y_nn_pred_train = y_nn_pred_train.cpu().detach().numpy()

            x_val = x_val.cpu().detach().numpy()
            t_val = t_val.cpu().detach().numpy()
            y_nn_pred_val = y_nn_pred_val.cpu().detach().numpy()

            # Create a figure with two subplots for comparing the model's performance on train and validation sets
            fig = plt.figure(figsize=(18, 10))
            ax_train = fig.add_subplot(121, projection='3d')
            ax_val = fig.add_subplot(122, projection='3d')

            # Plot the actual and predicted values for the training set
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

            # Set axes labels, add legend, and set title for the training set subplot
            ax_train.set_xlabel('X', fontsize=16)
            ax_train.set_ylabel('T (s)', fontsize=16)
            ax_train.set_zlabel('Y', fontsize=16)
            ax_train.legend()
            ax_train.set_title(f'Comparison at Epoch {epoch} in Train Set', fontsize=18)

            # Plot the actual and predicted values for the validation set
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

            # Set axes labels, add legend, and set title for the validation set subplot
            ax_val.set_xlabel('X', fontsize=16)
            ax_val.set_ylabel('T (s)', fontsize=16)
            ax_val.set_zlabel('Y', fontsize=16)
            ax_val.legend()
            ax_val.set_title(f'Comparison at Epoch {epoch} in Validation Set', fontsize=18)

            # Depending on the option, either show the plot or save it as an image
            if option == 'save':
                # Construct the path to save the plot and save it
                current_path = os.getcwd()
                os.makedirs(
                    os.path.join(
                        current_path,
                        'Photo', 'Output'
                    ), exist_ok=True
                )
                save_path = os.path.join(
                    current_path,
                    'Photo', 'Output'
                )
                now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                plt.savefig(
                    os.path.join(
                        save_path,
                        f"{self.lr:.3e}_"
                        f"{self.optimizer.param_groups[0]['lr']:.3e}_"
                        f"{epoch}_{now}.png"
                    ),
                    dpi=300, bbox_inches='tight', transparent=True
                )
                print(f"Plot saved at {save_path}")
            else:
                # Display the plot directly
                plt.show()

    def train(self):
        from tqdm import tqdm
        epochs = self.epochs
        Logs = []
        x_train, t_train, y_train = self.normalize(self.x_train, self.t_train, self.y_train)
        for epoch in tqdm(range(epochs)):
            self.dnn.train()  # Train the  model in training set
            self.optimizer.zero_grad()
            # Loss of Natural Network
            y_nn_pred = self.predict(x_train, t_train)
            loss_nn = torch.nn.functional.mse_loss(y_nn_pred, y_train)
            # Loss of Physical Informed Equation
            loss_physical = self.physicalLoss(self.x_train, self.t_train, self.y_train)

            Loss = loss_nn + loss_physical
            Loss.backward(retain_graph=True)
            self.optimizer.step()
            self.exponent_schedule.step()

            # Record loss
            self.dnn.eval()  # Evaluate the  model in validation set
            truth_train = torch.sum(torch.abs(y_nn_pred - y_train) <= 0.1 * torch.abs(y_train))
            self.history['train_loss'].append(loss_nn.item())
            self.history['train_accuracy'].append(truth_train.item() / y_train.numel() * 100)
            self.history['EI'].append(self.EI.item())
            self.history['Tension'].append(self.Tension.item())
            self.history['M'].append(self.M.item())
            self.history['c'].append(self.c.item())

            # Print epoch Results
            if epoch % 100 == 0:
                epoch_endTime = time.time()
                epoch_time = (epoch_endTime - epoch_startTime) if locals().get('epoch_startTime') else 0
                x_val, t_val, y_val = self.normalize(self.x_val, self.t_val, self.y_val)
                y_nn_val_pred = self.predict(x_val, t_val)
                truth_val = torch.sum(torch.abs(y_nn_val_pred - y_val) <= 0.1 * torch.abs(y_val))
                loss_nn_val = torch.nn.functional.mse_loss(y_nn_val_pred, y_val)
                loss_physical_val = self.physicalLoss(self.x_val, self.t_val, self.y_val)
                log = f"""
                    \n========  Epoch {epoch} / Total {epochs}  =======\n
                    ======== Cost {epoch_time:.1f}s per 50 epochs ========\n
                    ========  Train  Set  ========\n
                    Natural Network Loss:{loss_nn.item():.3e}\n
                    Physical Equation Loss:{loss_physical.item():.3e}\n
                    Accuracy:{self.history["train_accuracy"][epoch]:.2f}%\n
                    ======== Validation Set =======\n
                    Natural Network Loss:{loss_nn_val.item():.3e}\n
                    Physical Equation Loss:{loss_physical_val.item():.3e}\n
                    Accuracy:{(truth_val.item() / y_val.numel()) * 100:.2f}%\n
                    ------------------------------------------\n
                    刚度EI: {self.EI.item():<14.4e}, EI_grad: {self.EI.grad.item():.4e}\n
                    张力Tension: {self.Tension.item():<9.4e}, Tension_grad: {self.Tension.grad.item():.4e}\n
                    质量M: {self.M.item():<15.4e}, M_grad: {self.M.grad.item():.4e}\n
                    阻尼c: {self.c.item():<15.4e}, c_grad: {self.c.grad.item():.4e}\n
                """
                print(log)
                Logs.append(log)
                epoch_startTime = time.time()

            # Process Visualization
            if epochs <= 1000:
                if epoch % 100 == 0:
                    self.plot_results(epoch, option='save')
            else:
                if epoch <= 1000:
                    if epoch % 100 == 0:
                        self.plot_results(epoch, option='save')
                elif epoch % (epochs // 20) == 0:
                    if epoch <= 3000:
                        self.plot_results(epoch, option='save')
        return Logs

    def save(self, file_path: str, option: str = 'state'):
        """
        保存模型的状态或模型本身到指定路径。

        参数:
        - file_path: str, 保存文件的路径。
        - option: str, 保存选项，'state'表示保存模型的状态（默认），'model'表示保存模型本身。

        返回值:
        - 无。
        """
        with torch.no_grad():
            self.dnn.eval()
            loss = self.history['train_loss']
            accuracy = self.history['train_accuracy']
        now = time.strftime(f"{loss:.3e}_{accuracy:.1e}_%Y-%m-%d_%H-%M-%S", time.localtime())
        save_path = os.path.join(file_path, f'{now}.pth')
        save_dic = {
            'optimizer': self.optimizer.state_dict(),
            'EI': self.EI.item(),
            'Tension': self.Tension.item(),
            'M': self.M.item(),
            'c': self.c.item(),
            'history': self.history,
        }

        if option == 'state':
            save_dic['model'] = self.dnn.state_dict()
        elif option == 'model':
            save_dic['model'] = self.dnn
        else:
            raise ValueError("The option must be 'state' or 'model'!")
        torch.save(save_dic, save_path)
        print(f"Model parameters saved to {save_path}!")

    def load(self, save_dic: dict, option: str = 'state'):
        """
        加载模型和优化器的状态以及额外的参数。

        参数:
        - save_dic: 一个字典，包含模型状态、优化器状态和额外参数。
        - option: 一个字符串，指定加载的选项，可以是`state`（加载模型状态）或者`model`（加载整个模型）。

        注意:
        - 如果选项不是`state`或`model`，将抛出ValueError。
        - 不返回任何值，但会更新实例的状态。
        """
        if option == 'state':
            self.dnn.load_state_dict(save_dic['model'])  # 加载模型的状态
        elif option == 'model':
            self.dnn = save_dic['model']  # 直接加载整个模型
        else:
            raise ValueError("The option must be 'state' or 'model'!")
        self.optimizer.load_state_dict(save_dic['optimizer'])
        self.EI = self.def_para(save_dic['EI'])
        self.Tension = self.def_para(save_dic['Tension'])
        self.M = self.def_para(save_dic['M'])
        self.c = self.def_para(save_dic['c'])
        self.history = save_dic['history']
