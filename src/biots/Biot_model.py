# Biot's model, by J.B. Johnson, 
# 
# iport modules
import numpy as np
from scipy import special



def Johnson2(omega,air,ice,snow):
    """
    Compute the complex wavenumber of the two longitudinal modes of the Biot model for a given freuency and material properties.
    The solution of Deresiewiczis used with the Johnson/Biot definitions of the dissipation therm, use of tortuosity and r factor for rho12.
    
    Johnson2(omega,air,ice,snow)
    
    Parameters:
    -----------    
    omega:          angular frequency
    air:            air properties, [rho_f,nu,C_air] 
    ice:            ice properties, [rho_ice,C_ice]
    snow:           snow properties [rho_snow,yung,shear,per,d,d_bar,r]
    
    
    Variable:       Units:      Property:
    ---------       ------      --------
    rho:            kg/m^2      density
    nu:             m^2/s       kinematic viscosity
    C:              Pa^-1       compressibility
    yung:           Pa          yung modulus of snow
    shear:          Pa          shear modulus of snow
    per:            m^3 s/kg    permeability (definition of Johnson), per=per'(m^2)/dinamic viscosity(mu)
    d:              m           pore characteristic size
    d_bar:          -           structure factor, dimensionless
    r:              -           tortuuosity factor, dimensionless, r=0.5 for spheres
    
    Returns:
    -----------
    delta1:                 wavenumber of mode1
    Lambda1:                variable lambda of mode1
    delta2:                 wavenumber of mode2
    Lambda2:                variable lambda of mode2
    Delta, f, F, T, k,b:    other internal variables  
    
    
    
    Example: 
    
    [delta1,Lambda1,delta2,Lambda2,Delta, f, F, T, k,b]=Johnson2(omega,air,ice,snow)
    
    """
    beta=(ice[0]-snow[0])/(ice[0]-air[0])
    
    [rho_f,nu,C_air]=air
    [rho_ice,C_ice]=ice
    [rho_snow,yung,shear,per,d,d_bar,r]=snow
    
    
    # expressions definition, used to "simplify" equations 
             # shear modulus (Pa)
    N= snow[2] 
    gamma=beta*(air[2]-ice[1])
    kappa=(yung*N)/(9*N-3*snow[1])
    a=(gamma+ice[1]-ice[1]**2*kappa)
    
    
    R= beta**2/a
    Q=beta*(1-beta-ice[1]*kappa)/a          # (Pa)
    D=(gamma*kappa+beta**2+(1-2*beta)*(1-ice[1]*kappa))/a-2*N/3            # (Pa)
    
    Tor=1-r*(1-1/beta)
    rho12=-beta*rho_f*(Tor-1)    
    rho11=(1-beta)*rho_ice-rho12
    rho22=beta*rho_f-rho12
    rho=rho11+rho22+2*rho12
    
    P=D+2*N
    H=P+2*Q+R
    A=P*R/H**2-(Q/H)**2
    B=rho11/rho*R/H+rho22/rho*P/H-2*rho12/rho*Q/H
    C=(rho11*rho22-rho12**2)/rho**2
    
    
    # dissipation therms
    
    b=d_bar*beta**2/per
    k=d*np.sqrt(omega/nu)*d_bar
    T=(special.berp(k)+special.beip(k)*1j)/(special.ber(k)+special.bei(k)*1j)
    F=1/4*((k**2)*T)/(k+2*1j*T)
    f=b*F/(rho*omega)
    
    Delta=np.sqrt(B**2-4*A*C-2*1j*f*(B-2*A)-f**2)
    
    # lambda only difference in delta1/2
    Lambda1=(B-1j*f-Delta)/2/A
    Lambda2=(B-1j*f+Delta)/2/A
    
    delta1=np.sqrt(rho*omega**2/H*Lambda1)
    delta2=np.sqrt(rho*omega**2/H*Lambda2)
    
    return [delta1,Lambda1,delta2,Lambda2,Delta, f, F, T, k,b]

def wavenumber(omega,air,ice,snow):
    """
    Compute the complex wavenumbers of the three modes of the Biot model for a given freuency and material properties.
    The solution of Deresiewiczis used with the Johnson/Biot definitions of the dissipation therm, use of tortuosity and r factor for rho12.
    
    
    
    Parameters:
    -----------    
    omega:          angular frequency
    air:            air properties, [rho_f,nu,C_air] 
    ice:            ice properties, [rho_ice,C_ice]
    snow:           snow properties [rho_snow,yung,shear,per,d,d_bar,r]
    
    
    Variable:       Units:      Property:
    ---------       ------      --------
    rho:            kg/m^2      density
    nu:             m^2/s       kinematic viscosity
    C:              Pa^-1       compressibility
    yung:           Pa          yung modulus of snow
    shear:          Pa          shear modulus of snow
    per:            m^3 s/kg    permeability (definition of Johnson), per=per'(m^2)/dinamic viscosity(mu)
    d:              m           pore characteristic size
    d_bar:          -           structure factor, dimensionless
    r:              -           tortuuosity factor, dimensionless, r=0.5 for spheres
    
    
    Returns:
    -----------
    delta1:                 wavenumber of longitudinal waves mode1
    delta2:                 wavenumber of longitudinal waves  mode2
    dela3:                  wavenumber of transversal waves

    
    
    
    Example: 
    
    [delta1, delta2, delta3]=wavenumber(omega,air,ice,snow)
    
    """
    
    beta=(ice[0]-snow[0])/(ice[0]-air[0])   # porosity
    
    # unpack material properties
    [rho_f,nu,C_air]=air
    [rho_ice,C_ice]=ice
    [rho_snow,yung,shear,per,d,d_bar,r]=snow
    
    
    # expressions definition, used to "simplify" equations 
    N= snow[2] 
    gamma=beta*(air[2]-ice[1])
    kappa=(yung*N)/(9*N-3*snow[1])
    a=(gamma+ice[1]-ice[1]**2*kappa)
    
    
    R= beta**2/a
    Q=beta*(1-beta-ice[1]*kappa)/a          # (Pa)
    D=(gamma*kappa+beta**2+(1-2*beta)*(1-ice[1]*kappa))/a-2*N/3            # (Pa)
    
    Tor=1-r*(1-1/beta)          # tortuosity
    
    rho12=-beta*rho_f*(Tor-1)    
    rho11=(1-beta)*rho_ice-rho12
    rho22=beta*rho_f-rho12
    rho=rho11+rho22+2*rho12
    
    P=D+2*N
    H=P+2*Q+R
    A=P*R/H**2-(Q/H)**2
    B=rho11/rho*R/H+rho22/rho*P/H-2*rho12/rho*Q/H
    C=(rho11*rho22-rho12**2)/rho**2
    
    
    # dissipation therms
    
    b=d_bar*beta**2/per
    k=d*np.sqrt(omega/nu)*d_bar
    T=(special.berp(k)+special.beip(k)*1j)/(special.ber(k)+special.bei(k)*1j)
    F=1/4*((k**2)*T)/(k+2*1j*T)
    f=b*F/(rho*omega)
    
    Delta=np.sqrt(B**2-4*A*C-2*1j*f*(B-2*A)-f**2)
    
    # lambda; only difference in delta1/2/3
    Lambda1=(B-1j*f-Delta)/2/A
    Lambda2=(B-1j*f+Delta)/2/A
    Lambda3=(H/N)*(C-1j*f)/(rho22/rho-1j*f)
    
    
    delta1=np.sqrt(rho*omega**2/H*Lambda1)
    delta2=np.sqrt(rho*omega**2/H*Lambda2)
    delta3=np.sqrt(rho*omega**2/H*Lambda3)
    
    
    return [delta1, delta2, delta3]




    
def c1(air,ice,snow):
    """
    Compute the velocity of mode 1 neglecting the frequency dependence
    
    
    Parameters:
    -----------    
    omega:          angular frequency
    air:            air properties, [rho_f,nu,C_air] 
    ice:            ice properties, [rho_ice,C_ice]
    snow:           snow properties [rho_snow,yung,shear,per,d,d_bar,r]
    
    
    Variable:       Units:      Property:
    ---------       ------      --------
    rho:            kg/m^2      density
    nu:             m^2/s       kinematic viscosity
    C:              Pa^-1       compressibility
    yung:           Pa          yung modulus of snow
    shear:          Pa          shear modulus of snow
    per:            m^3 s/kg    permeability (definition of Johnson), per=per'(m^2)/dinamic viscosity(mu)
    d:              m           pore characteristic size
    d_bar:          -           structure factor, dimensionless
    r:              -           tortuuosity factor, dimensionless, r=0.5 for spheres
    
    
    Returns:
    -----------
    c1:                 wavenumber of longitudinal waves mode1
   

    Example: 
    
    velocity=c1(omega,air,ice,snow)
    
    """
    beta=(ice[0]-snow[0])/(ice[0]-air[0])
    
    [rho_f,nu,C_air]=air
    [rho_ice,C_ice]=ice
    [rho_snow,yung,shear,per,d,d_bar,r]=snow
    
    
    # expressions definition, used to "simplify" equations 
             # shear modulus (Pa)
    N= snow[2] 
    gamma=beta*(air[2]-ice[1])
    kappa=(yung*N)/(9*N-3*snow[1])
    a=(gamma+ice[1]-ice[1]**2*kappa)
    
    
    R= beta**2/a
    Q=beta*(1-beta-ice[1]*kappa)/a          # (Pa)
    D=(gamma*kappa+beta**2+(1-2*beta)*(1-ice[1]*kappa))/a-2*N/3            # (Pa)
    
    Tor=1-r*(1-1/beta)
    rho12=-beta*rho_f*(Tor-1)    
    rho11=(1-beta)*rho_ice-rho12
    rho22=beta*rho_f-rho12
    rho=rho11+rho22+2*rho12
    
    P=D+2*N
    H=P+2*Q+R

    
    return 1/np.sqrt(rho/H)
    
    
def shimizu(grain,rho):
    
    """
    Compute the permeability using shimizu formula
    
    
    Parameters:
    -----------    
    grain:          grain size (m)
    rho:            snow density (kg/m^2)
    
    Returns:
    -----------
    Kappa:         permeability (m^2)
        
    """    
    return 0.077*grain**2*np.exp(-0.0078*rho/1000)


def cond_flow(pore_s,porosity):
        
    """
    Compute the permeability using the cnduct flow model
    
    
    Parameters:
    -----------    
    pore_s:         pore size in (m)
    porosity:       porosity
    
    Returns:
    -----------
    Kappa:         permeability (m^2)
        
    """    
    return porosity*pore_s**2/32


def calonne(rho,ssa,rho_ice):
    
    """
    Compute the permeability using the aproximation of Calonne
    
    
    Parameters:
    -----------    
    rho:            snow density (kg/m^2)
    ssa:            specific surface density (m^-1)
    rho_ice:        ice density (kg/m^2)    
    
    Returns:
    -----------
    Kappa:         permeability (m^2)
        
    """    
    
    r=3/ssa
    return 3*r**2*np.exp(-0.013*rho)