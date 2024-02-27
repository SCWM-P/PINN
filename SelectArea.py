import cv2
import pandas as pd
import numpy as np
from dv import AedatFile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def HotPixel_brush(xEvent, T, yEvent):
    """
    :param xEvent: Event data of x
    :param T: Event data of time
    :param yEvent: Event data of y
    :return: filtered event data
    """
    df = pd.DataFrame({'T': T, 'xEvent': xEvent, 'yEvent': yEvent})
    df['coords'] = list(zip(df['xEvent'], df['yEvent']))
    grouped = df.groupby('coords')
    event = grouped.agg(activity=('coords', 'size'),
                        continuity=('T', lambda x: np.mean(np.diff(sorted(x))) if len(x) > 1 else np.nan))
    act_mean = event['activity'].mean()
    act_std = event['activity'].std()
    cont_mean = event['continuity'].mean()
    cont_std = event['continuity'].std()
    event_filtered = event[(event['activity'] > act_mean - 2 * act_std) & (event['activity'] < act_mean + 2 * act_std) &
                           (event['continuity'] > cont_mean - 2 *cont_std) & (event['continuity'] < cont_mean + 2 * cont_std)]
    filtered_events = df[df['coords'].isin(event_filtered.index)]
    return filtered_events['xEvent'].to_numpy(), filtered_events['T'].to_numpy(), filtered_events['yEvent'].to_numpy()


def update(frame):
    startPoint = frame
    endPoint = frame + event_num
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


def pause_and_resume(event):
    global isRunning
    if event.key == 'p':
        if isRunning:
            ani.pause()
            isRunning = False
        else:
            ani.resume()
            isRunning = True

filename = r'test_10mm_2Hz.aedat4'
with AedatFile(f'data/{filename}') as f:
    height, width = f['events'].size
    events = np.hstack([packet for packet in f['events'].numpy()])
    T, xEvent, yEvent, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
    T = (T - T[0]) / 1e6

    xEvent = xEvent[0::20]
    T = T[0::20]
    yEvent = yEvent[0::20]
    polarities = polarities[0::20]


isRunning = True
event_num = 30000
xEvent_selected = xEvent[0:event_num]
T_selected = T[0:event_num]
yEvent_selected = yEvent[0:event_num]
plt.rc('font', family='Times New Roman')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(xEvent_selected, T_selected, yEvent_selected, marker='.')
ax.set_xlabel('X', fontsize=18)
ax.set_ylabel('T (s)', fontsize=18)
ax.set_zlabel('Y', fontsize=18)
plt.title(f'Event from EventCamera of {filename}')
ani = FuncAnimation(fig,
                    update,
                    frames=np.arange(0, len(xEvent), 500),
                    interval=5,
                    init_func=init,
                    blit=False
                    )
fig.canvas.mpl_connect('key_press_event', pause_and_resume)
plt.show()