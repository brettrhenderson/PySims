import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

kb = 3.175E-6  # Hartree atomic units

samp_size = 100000
cmax = 12
x = np.linspace(0, cmax, cmax * 100)

fig = plt.figure(figsize=(10, 8))

# plot of distribution at different temperatures
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Distribution of Molecular Speeds at Different Temperatures')
ax1.set_xlabel('Speed / $ms^{-1}$')
ax1.set_ylabel('Frequency')
binstep = 0.1
bins = np.arange(0, (cmax+binstep), binstep)
sig = 1
mu = 0
pdf = 1/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sig)**2)
ax1.plot(x, pdf, 'b', label=f'N(5, 1)')

c_samp = np.linalg.norm(np.random.normal(loc=mu, scale=sig, size=(samp_size, 2)), axis=1)
freq, edges = np.histogram(c_samp, bins)
freq = freq / np.sum(freq) / binstep  # normalize the counts to sum to 1
ax1.bar(edges[:-1], freq, width=np.diff(edges), ec='b', fc='b', alpha=0.3, align="edge")
ax1.legend()

plt.show()