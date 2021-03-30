"""
Display eigenstates by animating it from the ground state
to the specified upper bound.

Starting from line ~20 are the 
mostly commented-out potentials that you can try out.
"""
from discrete_hamiltonian import discrete_hamiltonian
from constants import *
from function_helpers import toggle_blit, complex_to_colour, norm
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Number of eigenstates to show
M = 50

# Various Potentials

# ISW
V = np.zeros([N, N])

# SHO
# V = M_E*200.0*((X/L)**2 + (Y/L)**2)/2.0

# SHO with Gaussian Barrier
V = (M_E*400.0*((X/L)**2 + (Y/L)**2)/2.0 + 
     100*np.exp(-0.5*((X/L)**2 + (Y/L)**2)/0.1**2))

# Cone
# V = 50.0*(np.sqrt((X/L)**2 + (Y/L)**2))

# Circular Well
# from scipy.signal import square
# V = 200.0*(1.0 - square(2.1*np.pi*np.sqrt((X/L)**2 + (Y/L)**2)))

# Circular Well with Partitions
# from scipy.signal import square
# V = 200.0*(1.0 - square(2.0*np.pi*np.sqrt((X/L)**2 + (Y/L)**2)))
# V2 = np.zeros([N, N])
# V2[:, 15*N//32: 17*N//32] = 15.0
# V2[15*N//32: 17*N//32, :] = 15.0
# V += V2

# Four Square Wells
V = np.zeros([N, N])
height = 150.0
V[:, 64*N//128: 65*N//128] = height
V[64*N//128: 65*N//128, :] = height
V[0: 4*N//32, :] = height
V[:, 0: 4*N//32] = height
V[N - 4*N//32: N, :] = height
V[:, N - 4*N//32: N] = height

# Inverted Gaussian Well
# V = 50.0*(1.0 - np.exp(-0.5*((X/L)**2 + (Y/L)**2)/0.25**2))

# Inverted Gaussian Wells
# s = 0.12
# V = 50.0*(1.0 - np.exp(-0.5*((X/L)**2 + (Y/L - 0.25)**2)/s**2)
#           - np.exp(-0.5*((X/L)**2 + (Y/L + 0.25)**2)/s**2)
#           - np.exp(-0.5*((X/L - 0.25)**2 + (Y/L)**2)/s**2)
#           - np.exp(-0.5*((X/L + 0.25)**2 + (Y/L)**2)/s**2))

# Or use an image to define the potential instead?
# Note that the image must be a sqaure whose length is equal to N.
# im = plt.imread('image.png')
# V = (1.0 - (im[0:N, 0:N, 0] + im[0:N, 0:N, 1]
#      + im[0:N, 0:N, 2])/3.0)
# V = 15.0*V/np.amax(V)


t = perf_counter()
H = discrete_hamiltonian(V)
H = H.reshape([N*N, N*N])
values, vectors = eigsh(H, which='SM', k=M)
print(f"Time taken: {perf_counter() - t} s")
V = V.T
data = {'t': 0.0, 'e': np.exp(2.0j*np.pi/32.0), 
        'animation': None, 'count_updated': False
       }
br = 3.0*N/4.0
fig = plt.figure(dpi=120)
axes = fig.subplots(1, 2, squeeze=True)
ax = axes[0]
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Eigenstate'
             # + f', E = $E_0$'
             )
interpolation = 'bilinear' # if N < 128 else 'nearest' 
im = ax.imshow(complex_to_colour(br*np.reshape(vectors.T[0], [N, N])), 
               interpolation=interpolation, 
               extent=(X[0, 0], X[0, -1], Y[0, 0], Y[-1, 0]))
amax_2v = np.amax(V)*2.0
tmp = np.transpose(np.array([V/amax_2v, V/amax_2v, V/amax_2v]), 
                            [2, 1, 0])
im2 = ax.imshow(tmp, interpolation='gaussian',
                extent=(X[0, 0], X[0, -1], Y[0, 0], Y[-1, 0]))
txt = ax.text(2.0, 3.0, 'N = 1', color='white')
s = [0.0, 0.25*(np.max(values) - np.min(values))/values[0]]
for v in values[0:M]:
    axes[1].plot(s, 
                 [v/values[0], v/values[0]], color='gray', alpha=0.5)
axes[1].set_xlim(s[0], s[1])
axes[1].set_xticks(ticks=[])
e_level, = axes[1].plot(s, 
                        [1.0, 1.0], color='white')
axes[1].set_facecolor('black')
axes[1].set_aspect('equal')
# axes[1].set_xlabel('E level (Relative to ground)')
axes[1].set_title('E Level')
axes[1].set_ylabel('$E_N$ (Relative to $E_{1}$)')


def animation_func(*arg):
    dt = 0.1
    data['t'] += dt
    t = data['t'] % 1.0
    if int(data['t']) % 2:
        data['e'] *= np.exp(2.0j*dt*np.pi/16.0)
        e = data['e']
        psi = (1.0 - t)*e*np.reshape(br*vectors.T[int(data['t']*0.5)%M],
                                     [N, N])*(1.0 + 0.0j)
        psi += t*e*np.reshape(br*vectors.T[(int(data['t']*0.5) + 1)%M], 
                              [N, N])
        level = ((1.0 - t)*values[(int(data['t']*0.5))%M]/values[0] +
                 t*values[(int(data['t']*0.5) + 1)%M]/values[0])
        e_level.set_ydata(level)
        psi = norm(psi)
        im.set_data(complex_to_colour(br*e*psi))
    else:
        e = data['e']
        e_level.set_ydata([values])
        level = int(data['t']*0.5)%M
        txt.set_text(f'N = {level + 1}')
        e_level.set_ydata([values[level]/values[0], 
                            values[level]/values[0]])
        im.set_data(complex_to_colour(
                    np.reshape(br*e*vectors.T[level], [N, N])))
        data['count_updated'] = False
    return im, e_level, txt


data['animation'] = animation.FuncAnimation(fig, animation_func,
                                            blit=True, interval=1.0)
plt.show()
plt.close()