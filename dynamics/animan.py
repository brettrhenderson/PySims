import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, autoscale_on=False)

arm_att = [0,1.2]
leg_att = [0, 0.5]
head = [[0.25 * np.cos(t), 0.25 * np.sin(t) + 1.8] for t in np.linspace(0, np.pi * 2, 40)]
leye = [[0.03 * np.cos(t) - 0.09, 0.03 * np.sin(t) + 1.85] for t in np.linspace(0, np.pi * 2, 40)]
reye = [[0.03 * np.cos(t) + 0.09, 0.03 * np.sin(t) + 1.85] for t in np.linspace(0, np.pi * 2, 40)]
mouth = [[t, 1.7 + 0.03 * np.cos(t*100)] for t in np.linspace(-0.12, 0.12, 40)]
torso = [[0,0.5], [0, 1.55]]
larm = [[-0.75, 1.2], [0, 1.2]]
rarm = [[0, 1.2],[0.75, 1.2]]
# larm = [[t, 1.2 + 0.25 * np.sin(t*10) - 0.25 * np.sin(-0.75*10)] for t in np.linspace(-0.75, 0, 40)]
# rarm = [[t, 1.2 + 0.25 * np.sin(t*10) - 0.25 * np.sin(-0*10)] for t in np.linspace(0, 0.75, 40)]
lleg = [[-0.5, 0], [0, 0.5]]
rleg = [[0, 0.5], [0.5, 0]]

def R(theta, objs, o=(0,0), rad=False, return_separate=False):
    if not rad:
        theta *= np.pi / 180
    cost = np.cos(theta)
    sint = np.sin(theta)
    o = np.array(o)
    newobjs = []
    for obj in objs:
        newobj = []
        for p in obj:
            if return_separate:
                newobj.append([p[0] * cost - p[1] * sint, p[0] * sint + p[1] * cost])
            else:
                p[:] = [(p[0] - o[0]) * cost - (p[1] - o[1]) * sint + o[0], (p[0] - o[0]) * sint + (p[1] - o[1]) * cost + o[1]]
        if return_separate:
            newobjs.append(newobj)
    if return_separate:
        return newobjs

phead, = ax.plot([h[0] for h in head], [h[1] for h in head], 'r-')
ptorso, = ax.plot([t[0] for t in torso], [t[1] for t in torso], 'r-')
plarm, = ax.plot([l[0] for l in larm], [l[1] for l in larm], 'r-')
prarm, = ax.plot([r[0] for r in rarm], [r[1] for r in rarm], 'r-')
plleg, = ax.plot([l[0] for l in lleg], [l[1] for l in lleg], 'r-')
prleg, = ax.plot([r[0] for r in rleg], [r[1] for r in rleg], 'r-')
pleye, = ax.plot([r[0] for r in leye], [r[1] for r in leye], 'r-')
preye, = ax.plot([r[0] for r in reye], [r[1] for r in reye], 'r-')
pmouth, = ax.plot([r[0] for r in mouth], [r[1] for r in mouth], 'r-')
text = ax.text(0.5, 0.9, 'WOOT WOOT!', horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=24)
ground = ax.hlines(0, -1, 1)

ln = [phead, ptorso, plarm, prarm, plleg, prleg, ground, pleye, preye, pmouth, text]

colorscheme = cm.get_cmap('hsv')

def init_plot():
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim((-0.5, 2.5))
    plt.axis('off')
    return ln

tot_R = 0

def update_plot(i):
    global tot_R
    if i > 100:
        R(2*np.cos((i-100)/10), [head, torso, leye, reye], o=leg_att)
        tot_R += 2*np.cos((i-100)/10)
        phead.set_data([h[0] for h in head], [h[1] for h in head])
        ptorso.set_data([t[0] for t in torso], [t[1] for t in torso])
        pleye.set_data([r[0] for r in leye], [r[1] for r in leye])
        preye.set_data([r[0] for r in reye], [r[1] for r in reye])
        preye.set_data([r[0] for r in reye], [r[1] for r in reye])
        if i < 200:
            R(2 * np.cos((i-100) / 10), [larm, rarm], o=leg_att)
            plarm.set_data([l[0] for l in larm], [l[1] for l in larm])
            prarm.set_data([r[0] for r in rarm], [r[1] for r in rarm])

    #mouth_new = [[t, 1.7 + 0.03 * np.cos((t)*100 + i/10)] for t in np.linspace(-0.12, 0.12, 40)]
    mouth_new = np.array(mouth)
    mouth_new[:, 1] -= 1.7
    mouth_new[:,1] *= np.cos(i/10)
    mouth_new[:, 1] += 1.7
    if i > 100:
        R(tot_R, [mouth_new], o=leg_att)
        pmouth.set_data([r[0] for r in mouth_new], [r[1] for r in mouth_new])

    if i > 200:
        if i < 300:
            multiplier = 0 + (i-200)/100
        else:
            multiplier = 1
        larm_new = [[t, 1.2 + multiplier * 0.15 * np.cos((t-i/100) * 10) - multiplier * 0.15 * np.cos((-i/100) * 10)] for t in np.linspace(-0.75, 0, 40)]
        rarm_new = [[t, 1.2 + multiplier * 0.15 * np.sin((t-i/100) * 10) - multiplier * 0.15 * np.sin((-i/100) * 10)] for t in np.linspace(0, 0.75, 40)]
        R(tot_R, [larm_new, rarm_new], o=leg_att)
        plarm.set_data([l[0] for l in larm_new], [l[1] for l in larm_new])
        prarm.set_data([r[0] for r in rarm_new], [r[1] for r in rarm_new])

    for item in [phead, ptorso, plleg, plarm, prleg, prarm, pleye, preye, pmouth, text]:
        item.set_color(colorscheme((np.cos(i / 30) + 1) / 2))
    return ln

ani = FuncAnimation(fig, update_plot, init_func=init_plot, frames=int(200*2*np.pi), interval=25, blit=True, repeat=False)
# ani.save('animan.mp4', extra_args=['-vcodec', 'libx264'])
plt.show()