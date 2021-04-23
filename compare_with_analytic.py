"""
Compare the numerically obtained energies of the SHO and ISW
with their analytical values.

References for the exact energies of the SHO and ISW:

https://en.wikipedia.org/wiki/Particle_in_a_box#Higher-dimensional_boxes
https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator#
N-dimensional_isotropic_harmonic_oscillator

"""
from discrete_hamiltonian import discrete_hamiltonian
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from constants import *


k = 100


def plot(values, analytic_energies, title):
    fig = plt.figure()
    axes = fig.subplots(2, 1)
    axes[0].scatter([i for i in range(len(analytic_energies))],
                analytic_energies, marker='_', label='Analytic')
    axes[0].scatter([i for i in range(len(values))], values, marker='_', 
                label=f'Numerical ({N}×{N} Grid)')
    axes[0].set_xlim(-1, k)
    axes[0].set_ylim(0, np.amax(analytic_energies)*1.1)
    axes[0].set_ylabel('Energy')
    axes[0].set_title(title)
    axes[0].grid()
    axes[0].legend()
    axes[1].set_xlim(-1, k)
    axes[1].scatter([i for i in range(len(values))],
                    values/np.array(analytic_energies), marker='+')
    axes[1].set_ylabel('numerical divided by analytic')
    axes[1].set_ylim(0.8, 1.2)
    axes[1].set_xlabel('Energy Count')
    axes[1].grid()
    plt.show()
    plt.close()


# Infinite Square Well
V = np.zeros([N, N])
H = discrete_hamiltonian(V)
H = H.reshape([N*N, N*N])
values, vectors = eigsh(H, which='LM', k=k, sigma=0.0)
isw_analytic_energies = []
for n in range(1, 20):
    for m in range(1, 20):
        isw_analytic_energies.append((HBAR*n*np.pi)**2/((L + 2*L/N)**2*2*M_E) + 
                                    (HBAR*m*np.pi)**2/((L + 2*L/N)**2*2*M_E)
                                    )
isw_analytic_energies.sort()
isw_analytic_energies = isw_analytic_energies[0:k]
plot(values, isw_analytic_energies, title='ISW Energy Levels')


# Simple Harmonic Oscillator in 2D
K = 300.0
V = M_E*K*((X/L)**2 + (Y/L)**2)/2.0
OMEGA = np.sqrt(K/L**2)
H = discrete_hamiltonian(V)
H = H.reshape([N*N, N*N])
values, vectors = eigsh(H, which='SM', k=k)
sho_analytic_energies = []
for n in range(20):
    for m in range(20):
        sho_analytic_energies.append(OMEGA*HBAR*(n + m + 1))
sho_analytic_energies.sort()
sho_analytic_energies = sho_analytic_energies[0:k]
plot(values, sho_analytic_energies, title='SHO Energy Levels')