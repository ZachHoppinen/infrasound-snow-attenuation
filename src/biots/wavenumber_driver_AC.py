
import sys
import numpy as np
import matplotlib.pyplot as plt

path=r'/Users/zachkeskinen/Documents/infrasound/figures/literature/'  # change me

# sys.path.append(path+r'\scripts')
from Biot_model import wavenumber

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
youngs = 25e6      # There was a typo in the paper during publications. E=0.2-10**4 MPa. Sorry for that.
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
k_snow = 30e-4               # Value from table 1 in Capelli et al. 2016. There are different definitions of permeability. 
                            # I mixed them in the paper. Sorry for that 
                            # See function header: permeability (definition of Johnson), per=per'(m^2)/dinamic viscosity(mu)
                                                                                 
# Pore characteristic size. Characteristic linear dimension of the pore cross section
d = 0.0005

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



#  frequency dependent plots

fs=np.linspace(1,100,1000)
omegas=2*np.pi*fs

ks=wavenumber(omega = omegas, air = air, ice = ice, snow = snow)
c1s=omegas/np.real(ks[0])
c2s=omegas/np.real(ks[1])
c3s=omegas/np.real(ks[2])

alpha1=-20*np.imag(ks[0])/np.log10(np.e)   # it should be the conversion to decibel used for amplitudeto get dB/m units
alpha2=-20*np.imag(ks[1])/np.log10(np.e)
alpha3=-20*np.imag(ks[2])/np.log10(np.e)


plt.figure()
plt.plot(fs,c1s,label='fast')
plt.plot(fs,c2s,label='slow')
plt.plot(fs,c3s,label='trasversal')
plt.xlabel('frequency Hz')
plt.ylabel('speed (m/s)')
plt.legend()
plt.savefig(path+'c_f.png')

plt.figure()
plt.plot(fs,alpha1,label='fast')
plt.plot(fs,alpha2,label='slow')
plt.plot(fs,alpha3,label='trasversal')
plt.xlabel('frequency Hz')
plt.ylabel('attenuation coefficient (db/m)')
plt.legend()
# plt.yscale('log')
plt.savefig(path+'alpha_f.png')


# dependence from permeability 

fs=10**np.linspace(-1,4,300)
omegas=2*np.pi*fs

perms=np.array([1,5,10,20,30,50,100])*1e-4

por_sizes=np.array([0.0001,0.0002,0.0005,0.001,0.0015])

plt.figure( figsize=(8,8) )
ax = plt.subplot(211)
ax2 = plt.subplot(212)
for perm in perms:
    snow2 = [p_snow, youngs, shear, perm, d, d_bar, t]
    ks=wavenumber(omega = omegas, air = air, ice = ice, snow = snow2)
    c2s=omegas/np.real(ks[1])
    alpha2=-20*np.imag(ks[1])/np.log10(np.e)

    ax.plot(fs,c2s,label='permeability: {:.2e}'.format(perm))
    ax2.plot(fs,alpha2,label='permeability: {:.2e}'.format(perm))

ax.set_xlabel('frequency Hz')
ax.set_ylabel('speed (m/s)')
ax.legend()
ax.set_xscale('log')
ax2.set_xlabel('frequency Hz')
ax2.set_ylabel('attenuation coefficient (db/m)')
ax2.legend()
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.savefig(path+'alpha_c_permeability.png')


plt.figure( figsize=(8,8) )
ax = plt.subplot(211)
ax2 = plt.subplot(212)
for d2 in por_sizes:
    snow2 = [p_snow, youngs, shear, k_snow, d2, d_bar, t]
    ks=wavenumber(omega = omegas, air = air, ice = ice, snow = snow2)
    c2s=omegas/np.real(ks[1])
    alpha2=-20*np.imag(ks[1])/np.log10(np.e)

    ax.plot(fs,c2s,label='d: {:.1f} mm'.format(d2*1000))
    ax2.plot(fs,alpha2,label='d: {:.1f} mm'.format(d2*1000))

ax.set_xlabel('frequency Hz')
ax.set_ylabel('speed (m/s)')
ax.legend()
ax.set_xscale('log')
ax2.set_xlabel('frequency Hz')
ax2.set_ylabel('attenuation coefficient (db/m)')
ax2.legend()
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.savefig(path+'alpha_c_poresize.png')