from dv import AedatFile
import cv2
import os
currentpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.dirname(currentpath)
datapath = os.path.join(rootpath, 'data', "aedat4")

# with AedatFile(os.path.join(
#         datapath, 'dvframe-2021_10_07_13_58_32.aedat4'
# )) as f:
#     # list all the names of streams in the file
#     print(f.names)
#
#     # Access dimensions of the event stream
#     height, width = f['events'].size
#
#     # loop through the "events" stream
#     for e in f['events'].numpy():
#         print(e.timestamp)
#         print(e.shape)
#
#     # loop through the "frames" stream
#     for frame in f['events']:
#         print(frame.timestamp)
#         cv2.imshow('out', frame.image)
#         cv2.waitKey(1)

#%%
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
n_point = 1000

with AedatFile(os.path.join(
        datapath, 'dvSave-2023_01_10_19_08_14.aedat4'
)) as f:
    # events will be a named numpy array
    events = np.hstack([packet for packet in f['events'].numpy()])
    # Access information of all events by type
    timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
    # Access individual events information
    event_123_x = events[123]['x']
    # Slice events
    first_100_events = events[:100]
    ax = plt.axes(projection='3d')
    # ax.scatter3D(x[0:n_point],y[0:n_point],polarities[0:n_point])
    
    ax.scatter3D(x[0:n_point],y[0:n_point],timestamps[0:n_point])
    ax.grid(linestyle="--")
    plt.xlabel('$x$')
    plt.ylabel('$y$')


#%%
start_point = 2000000
n_point = 10000
plt.figure()
for j in range(start_point, start_point + n_point):
    if events[j]['polarity'] == 0:
        plt.plot(events[j]['x'],events[j]['y'],'o',color='k')
    else:
        plt.plot(events[j]['x'],events[j]['y'],'o',color='r')
plt.show()