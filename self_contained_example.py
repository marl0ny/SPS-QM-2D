import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# Constants
N, L = 101, 8
X, Y = np.meshgrid(np.linspace(-L/2, L/2, N, dtype=float),
                   np.linspace(-L/2, L/2, N, dtype=float))
DX = X[0, 1] - X[0, 0]
HBAR, M_E = 1.0, 1.0 # Use units where these are set to one

# The potential. Try defining your own potential 
# as long as it's defined on an NxN grid
V = M_E*400.0*((X/L)**2 + (Y/L)**2)/2.0

# Construct the discrete Hamiltonian.
diag = HBAR**2/(2*M_E*DX**2)*np.ones([N])
diags = np.array([-diag, 2.0*diag, -diag])
kinetic_1d = sparse.spdiags(diags, np.array([1.0, 0.0, -1.0]), N, N)
T = sparse.kronsum(kinetic_1d, kinetic_1d)
U = sparse.diags((V.reshape(N*N)), (0))
H = T + U

k = 10 # Number of eigenvectors to compute.
eigenvalues, eigenvectors = eigsh(H, which='LM', k=k, sigma=0.0)
# Show the eigenvector. Change the value of m as long as 0 <= m < k.
m = 5
plt.imshow(eigenvectors.T[m].reshape([N, N]), interpolation='bilinear',
           extent=(X[0, 0], X[0, -1], Y[0, 0], Y[-1, 0]))
plt.title(f'Eigenstate, N={m}')
plt.show()
