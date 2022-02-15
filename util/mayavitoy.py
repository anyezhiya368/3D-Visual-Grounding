import mayavi.mlab as mlab
import numpy as np

def plot3Dboxes(corners):
    for i in range(corners.shape[0]):
        corner = corners[i]
        plot3Dbox(corner)

def plot3Dbox(corner):
    idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
    x = corner[0, idx]
    y = corner[1, idx]
    z = corner[2, idx]
    print(x)
    print(y)
    print(z)
    mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral', representation='wireframe', line_width=2.5)
    mlab.show(stop=False)

corners = np.array([[[0.0, 1, 1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]])

plot3Dboxes(corners)


