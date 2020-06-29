"""
Animation of Elastic collisions with Gravity

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import multiprocessing


class ParticleBox:
    """Orbits class

    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """

    def __init__(self,
                 init_state=[[1, 0, 0, -1],
                             [-0.5, 0.5, 0.5, 0.5],
                             [-0.5, -0.5, -0.5, 0.5]],
                 bounds=[-2, 2, -2, 2],
                 size=0.04,
                 M=0.05):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size * np.ones(self.init_state.shape[0])
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        i_sizes, j_sizes = np.meshgrid(self.size, self.size)
        self.comb_sizes = i_sizes + j_sizes


    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt

        # update positions
        self.state[:, :2] += dt * self.state[:, 2:]

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < self.comb_sizes)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:] = v_cm - v_rel * m1 / (m1 + m2)

            # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < (self.bounds[0] + self.size))
        crossed_x2 = (self.state[:, 0] > (self.bounds[1] - self.size))
        crossed_y1 = (self.state[:, 1] < (self.bounds[2] + self.size))
        crossed_y2 = (self.state[:, 1] > (self.bounds[3] - self.size))

        self.state[crossed_x1, 0] = self.bounds[0] + self.size[crossed_x1]
        self.state[crossed_x2, 0] = self.bounds[1] - self.size[crossed_x2]

        self.state[crossed_y1, 1] = self.bounds[2] + self.size[crossed_y1]
        self.state[crossed_y2, 1] = self.bounds[3] - self.size[crossed_y2]

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1


# ------------------------------------------------------------
# set up initial state

if __name__ == '__main__':
    np.random.seed(0)
    init_state = -0.5 + np.random.random((100, 4))
    init_state[:, :2] *= 3.9

    kb = 3.17E-6  # Hartree atomic units
    R = 3.17E-6  # ideal gas constant in Eh/K/molecule
    T = 1400
    m=0.05
    cmax = 1.2
    c_au = np.linspace(0, cmax, 500)
    #c_conv = 2.188E6  # multiply by au to get m/s
    #c = c_au * c_conv

    box = ParticleBox(init_state, size=0.04, M=m)
    dt = 1. / 30  # 30fps

    # ------------------------------------------------------------
    # set up figure and animation
    fig = plt.figure(figsize=(7, 10))
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # axis for displaying the particles
    ax1 = fig.add_subplot(211, autoscale_on=False,
                          xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth() / np.diff(ax1.get_xbound())[0])

    # axis for displaying the histogram of speeds
    ax2 = fig.add_subplot(212, autoscale_on=False)
    ax2.set_xlim([0,cmax])
    cs = np.linalg.norm(box.state[:, 2:], axis=1)  # particle speeds
    bins = np.arange(0, 1.5 * max(cs), 1.5 * max(cs) / 20)
    pdf = m * c_au / (kb * T) * np.exp(-m * c_au ** 2 / (2 * kb * T))
    # normalize the pdf
    mb, = ax2.plot(c_au, pdf, f'r-', linewidth=3, label=f'T = {T}K')

    # particles holds the locations of the particles
    particles, = ax1.plot([], [], 'bo', ms=ms)

    # rect is the box edge
    rect = plt.Rectangle(box.bounds[::2],
                         box.bounds[1] - box.bounds[0],
                         box.bounds[3] - box.bounds[2],
                         ec='k', lw=2, fc='none')
    ax1.add_patch(rect)

    # def init():
    #     freq, _, hist = plt.hist(cs, bins=bins, density=True, ec='b', fc='b', alpha=0.4)
    #     for patch in hist:
    #         ax2.add_patch(patch)
    #     ax2.set_xlim([0, max(bins)*1.1])
    #     ax2.set_ylim([0, max(freq) * 1.1])
    #     return [particles, rect] + hist

    def animate(i):
        """perform animation step"""
        global box, dt, ax1, fig, particles
        box.step(dt)
        print(i)

        # update pieces of the animation
        particles.set_data(box.state[:, 0], box.state[:, 1])
        #particles.set_markersize(ms)

        if i % 10 == 0:
            # update velocity histogram
            cs = np.linalg.norm(box.state[:, 2:], axis=1)  # particle speeds
            freq, _, hist = ax2.hist(cs, bins=bins, density=True, ec='b', fc='b', alpha=0.4)

            # get temperature
            urms = np.sqrt(np.mean(cs**2))
            T_measured = 2 * np.mean(0.5 * m * cs**2) / (3 * R)
            T_rms = urms**2 * m / (3 * R)
            #print(T_rms)
            text = ax2.text(0.7, 0.9, f'Temp = {T_measured:.1f}K', horizontalalignment='center',
                            verticalalignment='center', transform=ax2.transAxes)

            ymin, ymax = ax2.get_ylim()
            if np.max(freq) >= ymax:
                ax2.set_ylim(ymin, np.max(freq) * 1.1)
                ax2.figure.canvas.draw()

        return [particles, rect, mb, text] + hist


    ani = animation.FuncAnimation(fig, animate, frames=400,
                                  interval=10, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    ani.save('particle_box_speeds.mp4', extra_args=['-vcodec', 'libx264'])
    #ani.save('particle_box.gif', writer='imagemagick')

    plt.show()