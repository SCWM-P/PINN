import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dv import AedatFile
from .. import data_cleansing

HotPixel_cleansing = data_cleansing.HotPixel_cleansing

filename = r'test_4mm_3Hz.aedat4'
with AedatFile(f'data/aedat4/{filename}') as f:
    height, width = f['events'].size
    events = np.hstack([packet for packet in f['events'].numpy()])

    # 指定区间
    x_min, x_max = 150, 450
    y_min, y_max = 280, 400
    t_min, t_max = events['timestamp'][0]+35e6, events['timestamp'][0]+38e6
    events = events[(events['x'] >= x_min) & (events['x'] <= x_max) &
                    (events['y'] >= y_min) & (events['y'] <= y_max) &
                    (events['timestamp'] >= t_min) & (events['timestamp'] <= t_max)]

    T, xEvent, yEvent, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
    T = (T - T[0]) / 1e6

    xEvent = xEvent[0::20]
    T = T[0::20]
    yEvent = yEvent[0::20]
    polarities = polarities[0::20]

xEvent, T, yEvent, polarities = HotPixel_cleansing(xEvent, T, yEvent, polarities)
data = {'xEvent': xEvent, 'Timestamp': T, 'yEvent': yEvent, 'polarities': polarities}
np.save(f'data/npy/{filename[:-7]}.npy', data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xEvent, T, yEvent, c=polarities, cmap='cividis', marker='.', alpha=0.5)
ax.set_title(f'{filename[:-7]}', fontsize=36)
ax.set_xlabel('$X$', fontsize=24)
ax.set_ylabel('$T$', fontsize=24)
ax.set_zlabel('$Y$', fontsize=24)
plt.show()
