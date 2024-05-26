import numpy as np
import scipy.io
import os


def load_data(option: str, current_path, filename: str):
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
        Timestamp = data['T'].flatten()
        xEvent = data['X'].flatten()
        yEvent = data['displ'].flatten()
        polarities = np.ones_like(xEvent)
    else:
        raise ValueError('Invalid option')
    return Timestamp, xEvent, yEvent, polarities