
from Biot_model import wavenumber
import numpy as np

##### FREQUENCY #####
# frequency 40 Hz
f = 40
# angular frequency
omega = 2*np.pi*f

##### AIR PARAMETERS
# Density
p_air = 1.1
# Compressibility
C_air = 1.3e-5
# Viscosity
nu_air = 8.5e-6

##### ICE PARAMETERS #####
# Density
p_ice = 917
# Bulk modulus
K_ice = 1.19e9
# Compressibility
C_ice = 1/K_ice

##### SNOW PARAMETERS #####
# Density (kg/m3)  - from field average
p_snow = 294
# Youngs Modulus of snow
youngs = 1e-2
# Poissons of snow
poissons = 0.25
# Shear modulus of snow
shear = youngs/(2*(1+poissons))
# radius (assumes sphere)
r = 0.5
# Porosity (0.45- 0.95)
phi = 1 - (p_snow/p_ice)
# Tortuosity with r = 1/2 for sphere
t = 1 - (r*(1- (1/phi)))

# free parameters to vary:
# Permeability
k_snow = 8.5e-9
# Pore characteristic size
d = 0.001
# Stucture Factor
d_bar = 1

##### CREATING MATERIALS #####
# air = [density, kinetic velocity, compressibility]
air = [p_air, C_air,nu_air]
# ice = [density, compressibility]
ice = [p_ice, C_ice]
# snow = [density, youngs modulus, shear modulus, permeability, pore characteristic size, structure factor, tortuosity factor]
snow = [p_snow, youngs, shear, k_snow, d, d_bar, t]

# running model
[delta1, delta2, delta3] = wavenumber(omega = omega, air = air, ice = ice, snow = snow)

##### Attenuation Coefficent from Wave number #####
# delta1 = Fast wave
# delta2 = slow wave
# delta3 = transversal wave
# Looking at: https://physics.byu.edu/faculty/colton/docs/phy442-winter20/lecture-10-complex-wave-number.pdf
print(f'Speed at 40 Hz in our snow is modeled at {omega/(delta2.real)} m/s')
# 2 times imaginary component since power varies with amplitude squared.
print(f'Attenuation at 40 Hz in our snow is modeled at {2*delta2.imag} db/m')


