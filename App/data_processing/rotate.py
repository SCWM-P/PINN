import numpy as np
from scipy.linalg import svd
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor


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


def data_rotate(xEvent: np.ndarray, Timestamp: np.ndarray, yEvent: np.ndarray, option='polyfit'):
    """
    Rotate the dataset to approximate a one-dimensional linear relationship between xEvent and yEvent.
    This is achieved by selecting an angle for rotation determined by different regression analysis methods.
    :param xEvent: One-dimensional array of independent variables.
    :param Timestamp: Array of timestamps of the same length as xEvent and yEvent, synchronized with them.
    :param yEvent: One-dimensional array of dependent variables.
    :param option: Choose the regression method used to determine the rotation angle.\
                            - 'polyfit' - Standard least squares polynomial fit using numpy's polyfit function.\
                            - 'RANSAC' - Robust regression analysis using sklearn's RANSACRegressor.\
                            - 'Huber' - Robust regression analysis using sklearn's HuberRegressor.\
                            - 'TLS' - Total least squares regression analysis.\
                            Default is 'polyfit'.\n
    :return Tuple containing the rotated xEvent, Timestamp, and yEvent.
        Note:
    - Ensure that xEvent and yEvent are one-dimensional and of the same length.
    - The Timestamp array will not be rotated but will be returned along with the rotated xEvent and yEvent.
    - This function requires the numpy and sklearn libraries.
    """
    if option == 'polyfit':
        [k, b] = np.polyfit(xEvent, yEvent, 1)
        gamma = np.arctan(k)
        xEvent, Timestamp, yEvent = rotate(xEvent, Timestamp, yEvent, gamma, 'y')
    elif option == 'RANSAC':
        ransac = RANSACRegressor()
        ransac.fit(xEvent.reshape(-1, 1), yEvent)
        k = ransac.estimator_.coef_
        gamma = np.arctan(k)
        xEvent, Timestamp, yEvent = rotate(xEvent, Timestamp, yEvent, gamma, 'y')
    elif option == 'Huber':
        huber = HuberRegressor()
        huber.fit(xEvent.reshape(-1, 1), yEvent)
        k = huber.coef_
        gamma = np.arctan(k)
        xEvent, Timestamp, yEvent = rotate(xEvent, Timestamp, yEvent, gamma, 'y')
    elif option == 'TLS':
        X = xEvent - np.mean(xEvent)
        Y = yEvent - np.mean(yEvent)
        M = np.vstack((X, Y)).T
        U, S, Vt = svd(M, full_matrices=False)
        a, b = Vt.T[:, -1]
        k = -a / b
        gamma = np.arctan(k)
        xEvent, Timestamp, yEvent = rotate(xEvent, Timestamp, yEvent, gamma, 'y')
    return xEvent, Timestamp, yEvent