import numpy as np
import matplotlib.pyplot as plt

filedir = r'data\npy\data3.npy'
data = np.load(filedir, allow_pickle=True).item()
xEvent = data['x_event']
Timestamp = data['T']
yEvent = data['y_event']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xEvent, Timestamp, yEvent, c='b', marker='.', alpha=0.5)
plt.show()