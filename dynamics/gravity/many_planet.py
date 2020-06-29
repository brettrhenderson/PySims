import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SolarSystem:

    # gravitational constant 6.67408 × 10-11 m3 kg-1 s-2
    G = 1

    def __init__(self, init_state, masses):
        self.init_state = np.asarray(init_state, dtype=float)
        # add accelerations
        self.init_state = np.concatenate((self.init_state, np.zeros((self.init_state.shape[0], 2))), axis=1)
        self.state = self.init_state.copy()
        self.masses = masses
        self.update_accels()
        self.time_elapsed = 0

    def update_accels(self):
        accels = np.zeros((self.state.shape[0], 2))
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[0]):
                if i < j:
                    r = self.state[j, :2] - self.state[i, :2]
                    d = np.linalg.norm(r)
                    accels[i] += (self.G * self.masses[j] / d**2) * r / d
                    accels[j] += -(self.G * self.masses[i] / d ** 2) * r / d
        self.state[:, 4:] = accels

    def step(self, dt):
        self.time_elapsed += dt
        self.state[:, :2] += self.state[:, 2:4] * dt + self.state[:, 4:] * (dt * dt) * 0.5
        old_acc = self.state[:, 4:].copy()
        self.update_accels()
        self.state[:, 2:4] += 0.5 * (old_acc + self.state[:, 4:]) * dt


# ----------------------------------------------------
# Initial State
initial = np.array([[0, 0, 0, 0],
                    [7, 0, 0, 25],
                    [10, 0, 0, 15],
                    [15, 0, 0, 17.5],
                    [25, 0, 0, 14],
                    [35, 0, 0, 11]])
ms = np.array([5000, 0.2, 0.1, 1, 0.4, 0.1])
sizes = [30, 5, 3, 12, 10, 3]
colors = ['r', 'k', 'b', 'g', 'y', 'm']

SS = SolarSystem(initial, ms)
# ----------------------------------------------------

# ----------------------------------------------------
# Plotting Stuff
dt = 1 / 120    # simulation timestep

# Plot Stuff
# set up figure and animation
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                      xlim=(-40, 40), ylim=(-40, 40))
planets = []
for i, _ in enumerate(initial):
    planets.append(ax1.plot([], [], colors[i] + 'o', markersize=sizes[i])[0])

paths = [ax1.plot([], [], colors[i] + '-', alpha=0.3)[0] for i, _ in enumerate(initial)]
paths_x_data = [[] for path in paths]
paths_y_data = [[] for path in paths]

def init():
    for planet in planets:
        planet.set_data([], [])
    for path in paths:
        path.set_data([], [])
    return planets + paths


def update(i):
    SS.step(dt)
    for idx, planet in enumerate(planets):
        planet.set_data(SS.state[idx, 0], SS.state[idx, 1])

        # update path
        paths_x_data[idx].append(SS.state[idx, 0])
        paths_y_data[idx].append(SS.state[idx, 1])
        paths[idx].set_data(paths_x_data[idx], paths_y_data[idx])
    return planets + paths


anim = FuncAnimation(fig, update, init_func=init, frames=100, interval=10, blit=True)
plt.axis('off')
# anim.save('wobble.gif', writer='imagemagick')
# anim.save('orbits.mp4', extra_args=['-vcodec', 'libx264'])
plt.show()

