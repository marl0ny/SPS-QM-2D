import numpy as np
import scipy.sparse as sparse
from constants import *


def discrete_hamiltonian(V: np.ndarray):
    """
    Given the potential V, return the discretized Hamiltonian.

    Reference to discretize the Hamiltonian:
    https://wiki.physics.udel.edu/phys824/
    Discretization_of_1D_continuous_Hamiltonian

    Discretizing the Laplacian:
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#
    Implementation_via_operator_discretization

    Kronecker sum of discrete Laplacians:
    https://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
    """
    diag = HBAR**2/(2*M_E*DX**2)*np.ones([N])
    diags = np.array([-diag, 2.0*diag, -diag])
    kinetic_1d = sparse.spdiags(diags, np.array([1.0, 0.0, -1.0]), N, N)
    T = sparse.kronsum(kinetic_1d, kinetic_1d)
    U = sparse.diags((V.reshape(N*N)), (0))
    return T + U


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from scipy.sparse.linalg import eigsh
    from function_helpers import complex_to_colour

    V = 50.0*((X/L)**2 + (Y/L)**2)

    H = discrete_hamiltonian(V).reshape([N*N, N*N])
    ds_eigvals, ds_eigvects =  eigsh(H, which='SM', k=100)
    psi0 = np.exp((-(X/L)**2 - (Y/L + 0.25)**2)/(0.5*0.18**2))
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
        data['t'] += 0.1
        psi = np.dot(coeffs*np.exp(-1.0j*ds_eigvals*data['t']/HBAR),
                     ds_eigvects.T)
        im.set_data(complex_to_colour(psi.reshape([N, N]), dyn_alpha=False))
        return im, im2,

    ani = animation.FuncAnimation(fig, animation_func, 
                                  blit=True, interval=1.0)
    plt.show()

