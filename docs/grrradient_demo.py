import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from ARRRtomatic_diff.optimization import *

def rosen(x, y):
    term1 = 100 * (y - x ** 2) ** 2
    term2 = (1 - x) ** 2

    total = term1 + term2
    return total

def render_descent(length, w_path, dims=2):
    """
    redering descent path with updates

    length is # of points for the line.
    dims is dimensions in the mapping space
    """
    x0, y0 = w_path[0]
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.array([x0,y0, rosen(x0,y0)])
    for index in range(1, length):
        x, y = w_path[index]
        z = rosen(x, y)
        lineData[:, index] = np.array([x,y,z])

    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines


def update_lines2(num, dataLines, lines, dataLines2, lines2):
    for line, data, line2, data2 in zip(lines, dataLines,lines2, dataLines2):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        line2.set_data(data2[0:2, :num])
        line2.set_3d_properties(data2[2, :num])
    return lines, lines2


# attaching 3D axis
fig = plt.figure(figsize=(10,8))
ax = p3.Axes3D(fig)

num_points = 3000
x = np.linspace(-2.0, 2.0, num_points)
y = np.linspace(-2.0, 2.0, num_points)
X, Y = np.meshgrid(x,y)
# print(X)
z = [rosenbrock((x[i], y[i]))[0].get_value() for i in range(num_points)]



Z = rosen(X, Y)

# print(z)


# ax.plot3D(x, y, z)
ax.contour3D(X,Y,Z, 200, alpha=0.5)

w0 = np.array([-1.9,1.96])
gd_path = do_gradient_descent(w0, rosenbrock, max_iter=num_points, step_size=0.0015, show=True)
# gd_path = do_gradient_descent(w0, rosenbrock, use_momentum=True, max_iter=2000, step_size=0.0005, show=True)

# w0_2 = np.array([-1.5,1.5])
# gd_path2 = do_gradient_descent(w0_2, rosenbrock, use_momentum=True, max_iter=2000, step_size=0.001, show=True)



batch = 1000
frame = 60
interval = 10
data = [render_descent(frame, gd_path, 3) for index in range(batch)]
# data2 = [render_descent(frame, gd_path2, 3) for index in range(batch)]


# Had to do the following because can't pass 3d vector to ax.plot
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], color="red")[0] for dat in data]
# lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], color="blue")[0] for dat in data]


# lines2 = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], color="blue")[0] for dat in data2]



ax.set_xlim3d([-2.0, 2.0])
ax.set_xlabel('X')

ax.set_ylim3d([-2.0, 2.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 2500.0])
ax.set_zlabel('Z')

ax.set_title('GRRRADIENT Descent! (Global Minimum = (0,0))')


# Rendering the animation!
#                                  fig  func to update frame      updated func's args, interval
line_ani = animation.FuncAnimation(fig, update_lines, frame, fargs=(data, lines),
                                   interval=interval, blit=False)

# line_ani2 = animation.FuncAnimation(fig, update_lines, frame, fargs=(data2, lines2),
#                                     interval=interval, blit=False)

Writer = animation.FFMpegWriter()
ax.plot3D([0], [0], [0], marker='x', color='green')

# ax.legend([f'Starting point: ({-1.9}),({1.9}), with Momentum'])
ax.legend([f'Starting point: ({-1.9}),({1.9}), no Momentum'])

print(animation.writers.list())
# plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'

# line_ani.save('GRRRadient_Descent_momentum.mp4', writer=Writer)
line_ani.save('GRRRadient_Descent_no_momentum.mp4', writer=Writer)


# line_ani.save('GRRRadient_Descent_compare2.mp4', writer=Writer)

plt.show()

