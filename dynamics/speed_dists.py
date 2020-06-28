import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

kb = 3.175E-6  # Hartree atomic units

samp_size = 1000
cmax = 2000
c = np.linspace(0, cmax, cmax * 10)
c_conv = 2.188E6  # multiply by au to get m/s
c_au = c / c_conv

fig = plt.figure(figsize=(15, 8))

# plot of distribution at different temperatures
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('Distribution of Molecular Speeds at Different Temperatures')
ax1.set_xlabel('Speed / $ms^{-1}$')
ax1.set_ylabel('Frequency')
m = 58240  # mass of O2 in Hartree au
binstep = 100
bins = np.arange(0, (cmax+binstep) / c_conv, binstep / c_conv)
colors = 'b r g m'.split()
for i, T in enumerate([100, 200, 500, 1000]):
    pdf = m * c_au / (kb * T) * np.exp(-m * c_au ** 2 / (2 * kb * T))
    # normalize the pdf
    pdf = pdf / np.trapz(pdf, c)
    ax1.plot(c, pdf, f'{colors[i]}-', linewidth=3, label=f'T = {T}K')

    c_samp = np.linalg.norm(np.random.normal(loc=0, scale=np.sqrt(kb * T / m), size=(samp_size, 2)), axis=1)
    freq, edges = np.histogram(c_samp, bins)
    edges = c_conv * edges
    freq = freq / np.sum(freq) / binstep  # normalize the counts to sum to 1
    ax1.bar(edges[:-1], freq, width=np.diff(edges), ec=f"{colors[i]}", fc=f'{colors[i]}', alpha=0.3, align="edge")
ax1.legend()

# plot of distribution for different gases at 300K
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Distribution of Molecular Speeds For Different Gases')
ax2.set_xlabel('Speed / $ms^{-1}$')
ax2.set_ylabel('Frequency')
cmax *= 1.5
c = np.linspace(0, cmax, cmax * 10)
c_au = c / c_conv
bins = np.arange(0, (cmax+binstep) / c_conv, binstep / c_conv)
gases = ['He', '$O_2$', 'Ar', '$Cl_2$']
m_conv = 0.001 / 6.02E23 / 9.1E-31
masses = list(m_conv * np.array([4, 32, 39.9, 70.9]))
T = 300
for i, gas in enumerate(zip(gases, masses)):
    name, m = gas
    pdf = m * c_au / (kb * T) * np.exp(-m * c_au ** 2 / (2 * kb * T))
    # normalize the pdf
    pdf = pdf / np.trapz(pdf, c)
    ax2.plot(c, pdf, f'{colors[-i]}-', linewidth=3, label=f'{name}, {m / m_conv} g/mol')

    c_samp = np.linalg.norm(np.random.normal(loc=0, scale=np.sqrt(kb * T / m), size=(samp_size, 2)), axis=1)
    freq, edges = np.histogram(c_samp, bins)
    edges = c_conv * edges
    freq = freq / np.sum(freq) / binstep # normalize the counts to sum to 1
    ax2.bar(edges[:-1], freq, width=np.diff(edges), ec=f"{colors[-i]}", fc=f'{colors[-i]}', alpha=0.3, align="edge")
ax2.legend()

plt.tight_layout()
plt.show()