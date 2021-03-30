# Single Particle States in 2D

Various simulations of non-relativistic single particle quantum mechanics in bounded 2D potentials.
The eigenstates animation are inspired by the awesome art by [Hudson Smith](https://www.instagram.com/hudthescientist/).

The prerequisites for using the program are listed in the `requirements.txt` file. Run the script `discrete_hamiltonian.py` to show a numerical simulation of
a wavepacket in a Harmonic Oscillator. The script `eigenstates_animation.py` displays eigenstates in various potentials, and `compare_with_analytic.py` compares
numerically computed energy eigenvalues of the Harmonic Oscillator and Infinite Square Well with their analytical versions. 

## References:

Discretizing the Hamiltonian:
 - [Discretization of 1D continuous Hamiltonian](https://wiki.physics.udel.edu/phys824/Discretization_of_1D_continuous_Hamiltonian)

Discrete Laplacian stencils:
- [Wikipedia - Image Processing via operator discretization](https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation%20via%20operator%20discretization)

Energies for analytically solvable potentials (used to compare it with the numerical solutions):
- [N-dimensional Harmonic Oscillator](https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator#N-dimensional_isotropic_harmonic_oscillator)
- [Infinite Square Well](https://en.wikipedia.org/wiki/Particle_in_a_box#Higher-dimensional_boxes)

Domain colouring using hue angle:
- [Wikipedia - Hue](https://en.wikipedia.org/wiki/Hue)
- [HSV-RGB Comparison](https://en.wikipedia.org/wiki/File:HSV-RGB-comparison.svg)
