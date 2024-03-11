import numpy as np


def rotate(x: np.ndarray, y: np.ndarray, z: np.ndarray, gamma: np.ndarray, axis='None'):
    """
    Rotate data around the specified axis,For example, if axis='x', the data will be rotated around the x-axis.\n
    The data manipulated by the function are all **np.Tensor** type.
    :param x: original x-axis data
    :param y: original y-axis data
    :param z: original z-axis data
    :param gamma: the angle of rotation in radians
    :param axis: axis of rotation you specify
    :return: rotated data tuple with (x, y, z)
    """
    if axis == 'x':
        x_r = x
        y_r = y * np.cos(gamma) - z * np.sin(gamma)
        z_r = y * np.sin(gamma) + z * np.cos(gamma)
    elif axis == 'y':
        x_r = x * np.cos(gamma) + z * np.sin(gamma)
        y_r = y
        z_r = -x * np.sin(gamma) + z * np.cos(gamma)
    elif axis == 'z':
        x_r = x * np.cos(gamma) - y * np.sin(gamma)
        y_r = x * np.sin(gamma) + y * np.cos(gamma)
        z_r = z
    else:
        raise Exception(
            '\nNo axis specified! Please specify an axis.\n'
            'For example axis=\'x\' if you want to rotate around the x-axis.'
        )
    return x_r, y_r, z_r


def data_rotate(xEvent: np.ndarray, Timestamp: np.ndarray, yEvent: np.ndarray):

    [k, b] = np.polyfit(xEvent, yEvent)
    gamma = np.arctan(k)
    xEvent, Timestamp, yEvent = rotate(xEvent, Timestamp, yEvent, gamma, 'y')
    return xEvent, Timestamp, yEvent