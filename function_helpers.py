import matplotlib
import numpy as np


def complex_to_colour(values: np.ndarray, dyn_alpha=True) -> np.ndarray:
    """
    Get a colouring array given a complex array of data.

    How to map an angle on the colour wheel to a colour value:
    https://en.wikipedia.org/wiki/Hue
    https://en.wikipedia.org/wiki/File:HSV-RGB-comparison.svg

    """
    arg = np.angle(values)
    br = np.abs(values)
    coeffs = [0.99502488/2.0, 0.60809906, -0.00411985, -0.135546618]
    # This coefficients are found by Fourier transforming the hue angle to colour
    # relation and only considering the first four frequencies.
    r = sum([c*np.cos(n*arg) for n, c in enumerate(coeffs)])
    b = sum([c*np.cos(n*(2.0*np.pi/3.0 + arg)) for n, c in enumerate(coeffs)])
    g = sum([c*np.cos(n*(arg - 2.0*np.pi/3.0)) for n, c in enumerate(coeffs)])
    return np.transpose(np.array([2.0*br*r, 2.0*br*g, 2.0*br*b, 
                        8.0*br/np.amax(br) if dyn_alpha else r/r]),
                        (1, 2, 0))


def norm(v):
    """
    Get v normalized.
    """
    return v/np.sqrt(np.sum(v*np.conj(v)))

