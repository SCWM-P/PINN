import torch
import time
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import data_processing as dp
from module import PhysicsInformedNN

np.random.seed(1234)
torch.manual_seed(1234)
torch.autograd.set_detect_anomaly(True)
plt.ion()
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex=True)
plt.rc('grid', color='k', alpha=0.2)


if __name__ == '__main__':
    # Configuration
    epochs = 500
    layers = [2, 50, 50, 50, 50, 1]
    connections = [0, 1, 2, 3, 3, 2]

    # Check CUDA availability (for GPU acceleration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("========  Using device  ========")
    print(f"============  {device}  ============")

    # Load Data
    filename = 'test16-1.mat'
    data = scipy.io.loadmat(f'data/mat/{filename}')
    Timestamp = data['brushedData'][:, 0]/1e6
    xEvent = data['brushedData'][:, 1]
    yEvent = data['brushedData'][:, 2]
    polarities = np.zeros_like(xEvent)
    # filename = 'dvSave-2023_03_26_02_21_16.npy'
    # data = np.load(f'data/npy/{filename}', allow_pickle=True).item()
    # xEvent = data['xEvent']
    # Timestamp = data['Timestamp']
    # yEvent = data['yEvent']
    # polarities = data['polarities']

    # Data Cleansing
    fig = plt.figure()
    dp.plot_data(
        fig.add_subplot(131, projection='3d'),
        xEvent, Timestamp, yEvent,
        title='Original Data', color=polarities
    )
    (xEvent, Timestamp, yEvent, polarities) = dp.HotPixel_cleansing(xEvent, Timestamp, yEvent, polarities)
    dp.plot_data(
        fig.add_subplot(132, projection='3d'),
        xEvent, Timestamp, yEvent,
        title='After HotPixel Cleansing', color=polarities
    )
    (xEvent, Timestamp, yEvent) = dp.data_rotate(xEvent, Timestamp, yEvent, option='TLS')
    dp.plot_data(
        fig.add_subplot(133, projection='3d'),
        xEvent, Timestamp, yEvent,
        title='After Data Rotation', color=polarities
    )

    # Convert to torch.Tensor
    xEvent = torch.tensor(
        xEvent,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    ).unsqueeze(1)
    Timestamp = torch.tensor(
        Timestamp,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    ).unsqueeze(1)
    yEvent = torch.tensor(
        yEvent,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    ).unsqueeze(1)
    print('====== Data Loading Done! ======')

#%%
    print('===== Model Initialization =====')
    pinn = PhysicsInformedNN(
        layers, connections, device,
        xEvent, Timestamp, yEvent,
        epochs
    )
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
    fig1 = plt.figure(figsize=(18, 10))
    plt.rcParams.update({'font.size': 16})
    layout = (2, 2)
    subplots = [plt.subplot2grid(layout, (0, 0)), plt.subplot2grid(layout, (0, 1)),
                plt.subplot2grid(layout, (1, 0)), plt.subplot2grid(layout, (1, 1))]
    line_styles = ['-', '--', '-.', ':', '-']
    plots = [subplots[0].plot(pinn.history['EI'], c='r', ls=line_styles[0], label='EI'),
             subplots[1].plot(pinn.history['Tension'], c='g', ls=line_styles[1], label='T'),
             subplots[2].plot(pinn.history['M'], c='b', ls=line_styles[2], label='M'),
             subplots[3].plot(pinn.history['c'], c='c', ls=line_styles[3], label='c')]
    for i, ax in enumerate(subplots):
        ax.set_title(f"Parameter {['EI', 'Tension', 'M', 'c', 'γ'][i]}", fontsize=16)
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Parameter Value', fontsize=16)
        ax.legend()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    fig1.suptitle('Parameter Evolution', fontsize=18)

    # Plotting Loss Function and Accuracy Change
    fig2, ax2_1 = plt.subplots()
    ax2_2 = ax2_1.twinx()
    ax2_1.plot(pinn.history['train_loss'], 'r-', label='Loss', linewidth=2)
    ax2_2.plot(pinn.history['train_accuracy'], 'b.-', label='Accuracy', linewidth=2)
    ax2_1.set_xlabel('Epoch', fontsize=16)
    ax2_1.set_ylabel('Loss', fontsize=16)
    ax2_2.set_ylabel('Accuracy (%)', fontsize=16)
    plt.title('Loss and Accuracy Change', fontsize=18)
    plt.legend()

    # plot the final 3D result
    pinn.plot_results(epochs)
    plt.show()
