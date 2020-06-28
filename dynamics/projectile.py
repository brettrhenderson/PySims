import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# gravitational constant 6.67408 × 10-11 m3 kg-1 s-2
G = 6.67408E-11


class Projectile:
    def __init__(self, h0, v0, rg=6.731e6, g=9.8, dt=1.0, nmax=300000):
        self.dt = dt
        self.h0 = h0
        self.g = g
        self.rg = rg
        self.nmax = nmax
        self.xg = np.array([0, -rg])
        self.x = [np.array([0, h0])]
        self.v = [v0]
        self.a = [self.accel_(self.x[0])]

    def accel_(self, r):
        r_vec = self.xg - r
        r_mag = np.linalg.norm(r_vec)
        return self.g * r_vec / r_mag

    def step_(self):
        self.x.append(self.x[-1] + self.v[-1] * self.dt + 0.5 * self.a[-1] * (self.dt**2))
        new_acc = self.accel_(self.x[-1])
        self.v.append(self.v[-1] + 0.5 * (self.a[-1] + new_acc) * self.dt)
        self.a.append(new_acc)

    def compute_trajectory(self):
        while (np.linalg.norm(self.x[-1] - self.xg) > self.rg) and (len(self.x) < self.nmax):
        #while (self.x[-1][1] > 0) and (len(self.x) < self.nmax):
            self.step_()

    def animate(self, adjust_axes=False, figsize=(10,8)):
        x = np.array(self.x)
        # Plot Stuff
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, autoscale_on=False)
        path, = ax.plot([], [], 'b-', markersize=4, alpha=0.4)
        proj, = ax.plot([], [], 'r-', markersize=4)

        ground = ax.hlines(0, self.x[0] - 10, self.x[-1] + 10)
        textx = ax.text(0.8, 0.9, f'x = {self.x[0][0]:.2f}m', horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
        texty = ax.text(0.8, 0.8, f'y = {self.x[0][1]:.2f}m', horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
        textv = ax.text(0.8, 0.7, f'v = {np.linalg.norm(self.v[0]):.1f}m/s', horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
        ln = [proj, path, ground, textv, textx, texty]

        def init_plot():
            if adjust_axes:
                ax.set_xlim(self.x[0][0] - 2, self.x[0][0] + 10)
                ax.set_ylim(-self.h0/3, self.x[0][1] + 10)
            else:
                ax.set_xlim(self.x[0][0] - 2, self.x[-1][0] + 10)
                ax.set_ylim(-self.h0 / 3, self.x[int(len(self.x)/2)][1] + 10)

            plt.axis('off')
            return ln

        def update_plot(i):

            # get current vector of jav
            javv = 2.3 * (x[i] - x[i-1]) / np.linalg.norm(x[i] - x[i-1])
            jav = np.array([x[i] - javv/2, x[i] + javv/2])

            # proj.set_data(x[i, 0], x[i, 1])
            proj.set_data(jav[:, 0], jav[:, 1])
            path.set_data(x[:i+1, 0], x[:i+1, 1])

            # adjust text
            textx.set_text(f'x = {x[i, 0]:.2f}m')
            texty.set_text(f'y = {np.linalg.norm(self.x[i] - self.xg) - self.rg:.2f}m')
            textv.set_text(f'v = {np.linalg.norm(self.v[i]):.1f}m/s')

            if adjust_axes:
                # adjust axes if needed
                if self.x[i][0] > ax.get_xlim()[1] - 10:
                    ax.set_xlim([ax.get_xlim()[0], x[i, 0] + 10])
                if self.x[i][1] > ax.get_ylim()[1] - 10:
                    ax.set_ylim([ax.get_ylim()[0], x[i, 1] + 10])

            return ln

        return FuncAnimation(fig, update_plot, init_func=init_plot, frames=len(self.x), interval=50, blit=True, repeat=False)

if __name__ == "__main__":
    h0 = 2
    v0 = np.array([26, 15])
    ball = Projectile(h0, v0, rg=6.731e6, g=9.8/6, dt=0.05, nmax=30000)
    ball.compute_trajectory()
    ball.animate(adjust_axes=False, figsize=(15,6))
    plt.show()


plt.show()