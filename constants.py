"""
Constant Values.
"""
import numpy


# Note that it took my computer 402 seconds to 
# compute the 100 lowest eigenstates for a 200x200 grid.
N = 129 # Controls the grid size.

L = 8
X, Y = numpy.meshgrid(numpy.linspace(-L/2, L/2, N, dtype=float),
                      numpy.linspace(-L/2, L/2, N, dtype=float))
DX = X[0, 1] - X[0, 0]
HBAR = 1.0
M_E = 1.0