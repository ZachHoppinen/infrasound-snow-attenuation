import numpy as np

def attenuation_coefficent(z, a_0, a_z):
    """
    z: depth in meters
    a_0: amplitude at 0 meters
    a_z: amplitude at z meters
    """
    return -np.log(a_z/a_0) / z