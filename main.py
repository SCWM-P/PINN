# -*- coding: utf-8 -*-
# %%
import os
import torch
import time
import json
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import App.data_processing as dp
from model import PhysicsInformedNN

np.random.seed(1234)
torch.manual_seed(1234)
torch.autograd.set_detect_anomaly(True)
plt.rc('grid', color='k', alpha=0.2)
current_path = os.getcwd()
# Configuration
with open('config.json', 'r') as f:
    config = json.load(f)
    epochs = config['epochs']
    layers = config['layers']
    connections = config['connections']
    USE_pth = config['USE_pth']
    optimizer_config = config['optimizer_config']
    lr = optimizer_config['lr']
try:
    if not config['HEADLESS']:
        plt.ion()
        plt.rc('font', family='Times New Roman')
        matplotlib.use('TkAgg')
        # plt.rc('text', usetex=True)
except Exception as e:
    warnings.warn(e.msg, UserWarning)

# Check CUDA availability (for GPU acceleration)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("========  Using device  ========")
print(f"============  {device}  ============")

# Load Data
option = 'aedat4'
# filename = 'variables.npz'
# filename = 'test16-1.mat'
filename = 'dvSave-2023_03_26_02_21_16.npy'
Timestamp, xEvent, yEvent, polarities = dp.load_data(option, current_path, filename)
slice = int(0.75*len(Timestamp))
Timestamp = Timestamp[:slice]
xEvent = xEvent[:slice]
yEvent = yEvent[:slice]
polarities = polarities[slice:]
# Data Cleansing
n_subplot = sum((config['USE_HotPixelCleansing'], config['USE_rotate'])) + 1
fig, axes = dp.set_subplots(n_subplot, projection='3d')
try:
    mp = dp.plot_data(
        axes[0], xEvent, Timestamp, yEvent,
        title='Original Data', color=yEvent
    )
    plt.colorbar(mp, ax=axes[0])
    if config['USE_HotPixelCleansing']:
        (xEvent, Timestamp, yEvent, polarities) = dp.HotPixel_cleansing(xEvent, Timestamp, yEvent, polarities)
        mp = dp.plot_data(
            axes[1], xEvent, Timestamp, yEvent,
            title='After HotPixel Cleansing', color=yEvent
        )
        plt.colorbar(mp, ax=axes[1])
    if config['USE_rotate']:
        (xEvent, Timestamp, yEvent) = dp.data_rotate(xEvent, Timestamp, yEvent, option='TLS')
        mp = dp.plot_data(
            axes[-1], xEvent, Timestamp, yEvent,
            title='After Data Rotation', color=yEvent
        )
        plt.colorbar(mp, ax=axes[-1])
except Exception as e:
    print(str(e))


# Convert to torch.Tensor
xEvent = dp.to_Tensor(xEvent, device=device)
Timestamp = dp.to_Tensor(Timestamp, device=device)
yEvent = dp.to_Tensor(yEvent, device=device)
if config['HEADLESS']:
    fig.savefig(
        os.path.join(
            current_path,
            'Photo', 'data_load.png'
        ), bbox_inches='tight', dpi=300, transparent=True
    )
print(f"Load data figure has been saved at {os.path.join(current_path, 'Photo', 'data_load.png')}!")
print('====== Data Loading Done! ======')

# %%
print('===== Model Initialization =====')
# for lr in [1.0, 8e-1, 5e-1, 1e-1, 5e-2, 1e-2, 1e-3]:
pinn = PhysicsInformedNN(
    layers, connections, device,
    xEvent, Timestamp, yEvent,
    epochs, c=0, M=0.5625
)
pinn.optimizer.param_groups[0]['lr'] = lr
print(pinn.dnn)
print(f'lr: {lr}\t connections: {connections}')
os.makedirs(os.path.join(current_path, 'data', 'pth'), exist_ok=True)
if USE_pth:
    try:
        state_dic = dp.get_state_dic(
            os.path.join(
                current_path,
                'data', 'pth', '1.261e-01_31.9_2024-05-30-03-14-09.pth'
            )
        )
        pinn.load(state_dic)
        print('Model weights loaded!')
    except Exception as e:
        warnings.warn('Failed to load model weights!\nError Info: ' + str(e), UserWarning)
else:
    print('========= Model Training =======')
    # Training the Model
    start_time = time.time()
    Logs = pinn.train()
    pinn.save(os.path.join(current_path, 'data', 'pth'), 'state')
    with open(os.path.join(current_path, "log.txt"), 'a') as f:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(f"\n{current_time}\n")
        f.write(f"{str(pinn.dnn)}\nlr: {lr:.5f}\t connections: {connections}\n")
        f.write("".join(Logs))
    end_time = time.time()
    print('==============================================')
    print('============= Model Training Done! ===========')
    print("======== Training time: {:.2f} seconds ======".format(end_time - start_time))
    print('=== Average time per epoch: {:.4f} seconds ==='.format((end_time - start_time) / epochs))
    print('==============================================')

# dp.draw_results(pinn)
# plt.show()
