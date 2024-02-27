import cv2
import pandas as pd
import numpy as np
from dv import AedatFile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

with AedatFile(r'data/test_2mm_4Hz.aedat4') as f:
    height, width = f['events'].size
    events = np.hstack([packet for packet in f['events'].numpy()])
    T, xEvent, yEvent, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
    T = (T - T[0])/1e6


def HotPixel_brush(xEvent, T, yEvent):
    """
    :param xEvent:
    :param T:
    :param yEvent:
    :return: Brushed xEvent, T, yEvent
    This function is used to brush the data via 3-sigma rule
    """
    df = pd.DataFrame({'T': T, 'yEvent': yEvent, 'xEvent': xEvent})
    df['coords'] = list(zip(df.xEvent, df.yEvent))
    unique_coords = df['coords'].unique()
    event = pd.DataFrame(columns=['activity', 'continuity', 'coords'])

    for coords in unique_coords:
        current_events = df[df['coords'] == coords]
        if len(current_events) > 1:
            current_activity = len(current_events)
            current_continuity = np.mean(np.diff(current_events['T'].to_numpy()))
            event.loc[len(event)] = [current_activity, current_continuity, coords]
    act_mean = event.activity.mean()
    cont_mean = event.continuity.mean()
    act_std = event.activity.std()
    cont_std = event.continuity.std()
    event = event[(event.activity > act_mean - 2*act_std) & (event.activity < act_mean + 2*act_std)]
    event = event[(event.continuity > cont_mean - 3*cont_std) & (event.continuity < cont_mean + 3*cont_std)]
    filtered_events = df[df['coords'].isin(event.coords.to_list())]
    return filtered_events['xEvent'].to_numpy(), filtered_events['T'].to_numpy(), filtered_events['yEvent'].to_numpy()


def update(frame):
    startPoint = frame
    endPoint = frame + 10000
    xEvent_selected = xEvent[startPoint:endPoint]
    T_selected = T[startPoint:endPoint]
    yEvent_selected = yEvent[startPoint:endPoint]
    xEvent_selected, T_selected, yEvent_selected = HotPixel_brush(xEvent_selected, T_selected, yEvent_selected)
    scat._offsets3d = (xEvent_selected, T_selected, yEvent_selected)
    ax.set_xlim(np.min(xEvent_selected)-10, np.max(xEvent_selected)+10)
    ax.set_ylim(np.min(T_selected), np.max(T_selected))
    ax.set_zlim(np.min(yEvent_selected)-10, np.max(yEvent_selected)+10)
    return scat,


def init():
    ax.set_xlim(np.min(xEvent_selected) - 10, np.max(xEvent_selected) + 10)
    ax.set_ylim(np.min(T_selected), np.max(T_selected))
    ax.set_zlim(np.min(yEvent_selected) - 10, np.max(yEvent_selected) + 10)
    return scat,


xEvent_selected = xEvent[0:10000]
T_selected = T[0:10000]
yEvent_selected = yEvent[0:10000]
plt.rc('font', family='Times New Roman')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(xEvent_selected, T_selected, yEvent_selected, marker='.')
ax.set_xlabel('X', fontsize=18)
ax.set_ylabel('T (s)', fontsize=18)
ax.set_zlabel('Y', fontsize=18)
plt.title('$Event from EventCamera of wave 2mm, 4Hz$')
ani = FuncAnimation(fig,
                    update,
                    frames=np.arange(0, len(xEvent), 100),
                    interval=100,
                    init_func=init,
                    blit=False
                    )

plt.show()