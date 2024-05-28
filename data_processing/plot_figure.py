import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.rc('font', family='Times New Roman')
    plt.rc('text', usetex=True)
    plt.ion()
    plt.rc('grid', color='k', alpha=0.2)
except Exception as e:
    warnings.warn(e.msg, UserWarning)


def plot_data(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        title: str = '',
        xlabel: str = 'X',
        ylabel: str = 'Time(s)',
        zlabel: str = 'Y',
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


# Draw the final results for visualization
def draw_results(pinn):
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
    fig2, ax2_1 = plt.subplots(figsize=(18, 10))
    ax2_2 = ax2_1.twinx()
    ax2_1.plot(pinn.history['train_loss'], 'r-', label='Loss', linewidth=2)
    ax2_2.plot(pinn.history['train_accuracy'], 'b.-', label='Accuracy', linewidth=2)
    ax2_1.set_xlabel('Epoch', fontsize=32)
    ax2_1.set_ylabel('Loss', fontsize=32)
    ax2_2.set_ylabel(r'Accuracy (\%)', fontsize=32)
    plt.title('Loss and Accuracy Change', fontsize=36)
    ax2_1.legend(loc='upper left')
    ax2_2.legend(loc='upper right')

    # plot the final 3D result
    pinn.plot_results(pinn.epochs, option='save')
    plt.show()


def save_fig(fig, filename: str, save_path: str, format: str= None):
    fig.savefig(
        os.path.join(
            save_path,
            filename + 'png'
        ), dpi=300, format=format
    )
    print(f'{filename} saved to {savepath}')
