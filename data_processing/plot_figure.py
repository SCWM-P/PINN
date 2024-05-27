import numpy as np
import matplotlib.pyplot as plt
plt.ion()
plt.rc('font', family='Times New Roman')
plt.rc('text', usetex=True)
plt.rc('grid', color='k', alpha=0.2)


def plot_data(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        title: str = '',
        xlabel: str = '$X$',
        ylabel: str = '$Time(s)$',
        zlabel: str = '$Y$',
        color='r',
        label: str = '',
        cmap='viridis'
):
    """
    Plot the data on a given subplot ax in 3D view.
    """
    ax.scatter(
        x, y, z,
        c=color, marker='.', alpha=0.5,
        label=label, cmap=cmap
    )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_zlabel(zlabel, fontsize=14)
    if title:
        ax.set_title(f'{title}', fontsize=16)
    if label:
        ax.legend()
