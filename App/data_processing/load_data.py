import numpy as np
import scipy.io
import os
import torch
import time
np.random.seed(2024)


def load_data(option: str, current_path, filename: str, alpha: float=1000):
    """
    从指定路径加载数据。

    Args:
        option (str): 数据格式选项，支持 'mat', 'npy' 和 'npz' 三种格式。
        filename (str): 数据文件名。

    Returns:
        tuple: 包含Timestamp, xEvent, yEvent, polarities的元组。

    Raises:
        ValueError: 如果option不是'mat', 'npy' 或 'npz'中的一种，则引发ValueError异常。

    """
    if option == 'mat':
        data = scipy.io.loadmat(
            os.path.join(current_path, 'data', 'mat', filename)
        )
        Timestamp = data['brushedData'][:, 0] / 1e6
        xEvent = data['brushedData'][:, 1]
        yEvent = data['brushedData'][:, 2]
        polarities = np.zeros_like(xEvent)
    elif option == 'npy':
        data = np.load(
            os.path.join(current_path, 'data', 'npy', filename),
            allow_pickle=True
        ).item()
        xEvent = data['xEvent']
        Timestamp = data['Timestamp']
        yEvent = data['yEvent']
        polarities = data['polarities']
    elif option == 'npz':
        data = np.load(
            os.path.join(current_path, 'data', 'npy', filename),
            allow_pickle=True
        )
        data = {key: value for key, value in data.items()}
        Timestamp = data['X'].flatten()[::-1]
        xEvent = data['T'].flatten()
        yEvent = data["displ"].T.flatten()
        polarities = np.ones_like(xEvent)
        index = np.random.choice(
            range(len(Timestamp)),
            int(len(Timestamp)/alpha),
            replace=False
        )
        index.sort()
        Timestamp = Timestamp[index]
        xEvent = xEvent[index]
        yEvent = yEvent[index]
        polarities = polarities[index]
    else:
        raise ValueError('Invalid option')
    return Timestamp, xEvent, yEvent, polarities


def get_state_dic(path: str = None, current_path: str = os.path.dirname(os.path.abspath(__file__))):
    if path:
        state_dic = torch.load(path)
        return state_dic
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
                ))) + '.pth'
        )
    )
    return state_dic


def to_Tensor(x, device):
    return torch.tensor(
        x, dtype=torch.float32,
        device=device, requires_grad=True
    ).unsqueeze(1)
