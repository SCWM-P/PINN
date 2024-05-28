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
import data_processing as dp
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
try:
    if not config['HEADLESS']:
        plt.ion()
        plt.rc('font', family='Times New Roman')
        matplotlib.use('TkAgg')
        plt.rc('text', usetex=True)
except Exception as e:
    warnings.warn(e.msg, UserWarning)

# Check CUDA availability (for GPU acceleration)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("========  Using device  ========")
print(f"============  {device}  ============")

# Load Data
option = 'npz'
filename = 'variables.npz'
# filename = 'test16-1.mat'
# filename = 'dvSave-2023_03_26_02_21_16.npy'
Timestamp, xEvent, yEvent, polarities = dp.load_data('npz', current_path, filename)
# Data Cleansing
fig = plt.figure(figsize=(18, 10))
dp.plot_data(
    fig.add_subplot(131, projection='3d'),
    xEvent, Timestamp, yEvent,
    title='Original Data', color=yEvent
)
if config['USE_HotPixelCleansing']:
    (xEvent, Timestamp, yEvent, polarities) = dp.HotPixel_cleansing(xEvent, Timestamp, yEvent, polarities)
    dp.plot_data(
        fig.add_subplot(132, projection='3d'),
        xEvent, Timestamp, yEvent,
        title='After HotPixel Cleansing', color=yEvent
    )
if config['USE_rotate']:
    (xEvent, Timestamp, yEvent) = dp.data_rotate(xEvent, Timestamp, yEvent, option='TLS')
    dp.plot_data(
        fig.add_subplot(133, projection='3d'),
        xEvent, Timestamp, yEvent,
        title='After Data Rotation', color=yEvent
    )

# Convert to torch.Tensor
to_Tensor = lambda x: torch.tensor(
    x, dtype=torch.float32,
    device=device, requires_grad=True
).unsqueeze(1)
xEvent = to_Tensor(xEvent)
Timestamp = to_Tensor(Timestamp)
yEvent = to_Tensor(yEvent)
if not config['HEADLESS']:
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
pinn = PhysicsInformedNN(
    layers, connections, device,
    xEvent, Timestamp, yEvent,
    epochs
)
print(pinn.dnn)
os.makedirs(os.path.join(current_path, 'data', 'pth'), exist_ok=True)
if USE_pth:
    try:
        loss_list = [
            i[:-4].split('_')
            for i in os.listdir(
                os.path.join(
                    current_path,
                    'data', 'pth'
                )
            )
        ]
        state_dic = torch.load(
            os.path.join(
                current_path,
                'data', 'pth',
                '_'.join(max(
                    loss_list, key=lambda x: time.mktime(
                        time.strptime(x[2], '%Y-%m-%d-%H-%M-%S')
                    )
                    )) + '.pth'
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
        f.write(f'{str(pinn.dnn)}\n')
        f.write("".join(Logs))
    end_time = time.time()
    print('==============================================')
    print('============= Model Training Done! ===========')
    print("======== Training time: {:.2f} seconds ======".format(end_time - start_time))
    print('=== Average time per epoch: {:.4f} seconds ==='.format((end_time - start_time) / epochs))
    print('==============================================')

dp.draw_results(pinn)
plt.show(block=True)
