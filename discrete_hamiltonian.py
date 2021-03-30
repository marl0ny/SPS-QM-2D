import numpy as np
from constants import *


def discrete_hamiltonian(V: np.ndarray) -> np.ndarray:
    """
    Given the potential V, return the discretized Hamiltonian.
    Note that the discretized Hamiltonian is 4-dimensional.
    You will need to flatten it to 2D if you want to find its eigenvalues
    and eigenvectors with eigh or eigsh. 

    Reference to discretize the Hamiltonian:
    https://wiki.physics.udel.edu/phys824/
    Discretization_of_1D_continuous_Hamiltonian

    Discretizing the Laplacian:
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#
    Implementation_via_operator_discretization
    """
    # TODO: Allocating H uses a lot of unnecessary memory since it's
    # a sparse symmetric matrix. Use Scipy diagonals instead
    # (figure out how to use it to model the 
    #  4 index Hamiltonian).
    H = np.zeros([N, N, N, N], dtype=np.float32)
    for i in range(N):
        for j in range(N):
            H[i, j, i, j] = 3*HBAR**2/(2*M_E*DX**2) + V[i, j]
            if (i+1 < N): 
                H[i+1, j, i, j] = -0.5*HBAR**2/(2*M_E*DX**2)
                if (j+1 < N): H[i+1, j+1, i, j] = -0.25*HBAR**2/(2*M_E*DX**2)
                if (j-1 >= 0): H[i+1, j-1, i, j] = -0.25*HBAR**2/(2*M_E*DX**2)
            if (i-1 >= 0): 
                H[i-1, j, i, j] = -0.5*HBAR**2/(2*M_E*DX**2)
                if (j+1 < N): H[i-1, j+1, i, j] = -0.25*HBAR**2/(2*M_E*DX**2)
                if (j-1 >= 0): H[i-1, j-1, i, j] = -0.25*HBAR**2/(2*M_E*DX**2)
            if (j+1 < N): H[i, j+1, i, j] = -0.5*HBAR**2/(2*M_E*DX**2)
            if (j-1 >= 0): H[i, j-1, i, j] = -0.5*HBAR**2/(2*M_E*DX**2)
    return H


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from scipy.sparse.linalg import eigsh
    from function_helpers import complex_to_colour

    V = 50.0*((X/L)**2 + (Y/L)**2)

    H = discrete_hamiltonian(V).reshape([N*N, N*N])
    ds_eigvals, ds_eigvects =  eigsh(H, which='SM', k=100)
    psi0 = np.exp((-(X/L)**2 - (Y/L + 0.25)**2)/(0.5*0.2**2))
    coeffs = np.dot(psi0.reshape(N*N), ds_eigvects)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_title('Wavefunction')
    im = ax.imshow(complex_to_colour(psi0, dyn_alpha=False), 
                   interpolation='bilinear',
                   extent=[X[0, 1], X[0, -1], Y[0, 0], Y[-1, 0]])
    im2 = ax.imshow(np.transpose(np.array([V, V, V, V])/np.amax(V), 
                    (1, 2, 0)), interpolation='gaussian',
                    extent=[X[0, 1], X[0, -1], Y[0, 0], Y[-1, 0]])
    data = {'t': 0.0}

    def animation_func(*arg):
        data['t'] += 0.01
        psi = np.dot(coeffs*np.exp(-1.0j*ds_eigvals*data['t']/HBAR),
                     ds_eigvects.T)
        im.set_data(complex_to_colour(psi.reshape([N, N]), dyn_alpha=False))
        return im, im2,

    ani = animation.FuncAnimation(fig, animation_func, blit=True, interval=1.0)
    plt.show()

