import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dv import AedatFile


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
    event_filtered = event[(event['activity'] > act_mean - 2 * act_std) & (event['activity'] < act_mean + 1.5 * act_std) &
                           (event['continuity'] > cont_mean - 2 * cont_std) & (event['continuity'] < cont_mean + 1.5 * cont_std)]
    filtered_events = df[df['coords'].isin(event_filtered.index)]
    return filtered_events['xEvent'].to_numpy(), filtered_events['T'].to_numpy(), filtered_events['yEvent'].to_numpy()


filename = r'test_2mm_4Hz.aedat4'
with AedatFile(f'data/{filename}') as f:
    height, width = f['events'].size
    events = np.hstack([packet for packet in f['events'].numpy()])

    # 指定区间
    x_min, x_max = 300, 360
    y_min, y_max = 200, 600
    t_min, t_max = events['timestamp'][0], events['timestamp'][0]+20e6
    events = events[(events['x'] >= x_min) & (events['x'] <= x_max) &
                    (events['y'] >= y_min) & (events['y'] <= y_max) &
                    (events['timestamp'] >= t_min) & (events['timestamp'] <= t_max)]

    T, xEvent, yEvent, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
    T = (T - T[0]) / 1e6

    xEvent = xEvent[0::20]
    T = T[0::20]
    yEvent = yEvent[0::20]
    polarities = polarities[0::20]

xEvent, T, yEvent = HotPixel_brush(xEvent, T, yEvent)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xEvent, T, yEvent, c=polarities, marker='.', alpha=0.5)
plt.show()
