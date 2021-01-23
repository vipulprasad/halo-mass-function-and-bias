import numpy as np
from math import sin, cos, exp, pi, sqrt, log
import matplotlib.pyplot as plt

Mpc = 3.0857e22                     # 1 Mpc in metres.
sm = 2e30                           # Solar mass in Kg.
G = 6.67408e-11*sm/(Mpc**3)         # Gravitational constant.
h0 = 1 #0.6895004                    # Hubble constant in units of 100 Km/hr/Mpc.
H_o = (100*h0)/(Mpc*(10**(-3)))      # Hubble constant.
omega_b = 0.05                      # Baryons density parameter.
omega_m = 0.301685                  # Matter(baryons+dark matter) density parameter.
omega_lambda = 0.698315             # Dark energy density parameter.
deltac = 1.686                      # Critical overdensity for spherical collapse at z = 0.
sigma_8 = 0.853824                  # sigma_8 used for normalization.
ns = 0.9300325                      # spectral indece of initial power spectrum.
rho_crit = (3*(H_o)**2)/(8*np.pi*G) # critical density of the universe.
rho_bar = omega_m*rho_crit          # comoving background density in kg.


'''
Simpson and Trapezoidal rule with increasing step size.
This is because the integrating variable k(Mpc ^-1) spans across 
different order of magnitudes.
'''

def Simpson(f, a, b, m): #if a = 0 this won't work
	x = a				 #also for small intervals Simpson has more relative 
	h  = a*01			 #error, and more error than Trapezoidal below
	s = 0
	
	while x<b:
		s += (h/3)*(f(x,m)+4*f(x+h,m)+f(x+2*h,m))
		x += 2*h
		h*=1.01
		
	return s
	
def Trapezoidal(f, a, b, m): 
	#if a = 0 this won't work
	x = a
	h  = a*1.01
	s = 0
	while x<b:
		s += (h/2)*(f(x,m)+f(x+h,m))
		x += h
		h*=1.01
		
	return s
	

def Transfer(k): # Transfer function

	gamma = h0*h0*omega_m*exp(-omega_b-(sqrt(2*h0)*(omega_b/omega_m)))
	x = (k/gamma)
	A = log(1+2.34*x)/(2.34*x)
	B = (1+(3.89*x)+(16.1*x)**2+(5.46*x)**3+(6.71*x)**4)**(-0.25)
	return A*B

def D(z): 
	# Growth factor normalized to unity at z = 0.
	
	A = omega_m+0.4545*omega_lambda
	B = omega_m*(1+z)**3 + 0.4545*omega_lambda
	return (A/B)**(1/3)

def deltac(z): 
	# Critical overdensity for spherical collapse at different epoch.
	return 1.686/D(z)

def Power_spec(k):
	# Power spectrum
	
	P_i = k**ns # Initial power spectrum
	T = Transfer(k)**2
	return P_i*T 
	
def R(m):
	# comoving radius of a sphere of mass m
	return ((3*m)/(4*pi*rho_bar))**(1/3)

def W_kR(k,R):
	# window function in fourier space, used to smoothen the density field
	# This particular one is the Fourier transform of tophat filter function.
	
	x = k*R
	A = sin(x)
	B = cos(x)
	return (3/(x**3))*(A-x*B)

def sigma_integrand(k,R):
	A = (W_kR(k,R))**2.0
	B = (k**2)*Power_spec(k)
	return A*B/(2*pi**2)

def Sigma(R):
	integral = Simpson(sigma_integrand, 1e-7, 1e7, R)
	return norm8*sqrt(integral)

'''
Normalising the power spectrum for using the known value of sigma,
at R = 8.0/h 
'''

norm8 = 1
R8 = 8.0/h0
sigma8 = Sigma(R8)
norm8 = sigma_8/sigma8
print(R8,Sigma(R8), norm8) 

def Delta(m,r,z):
	# The spherical overdensity function.
	# used to find halos	
	
	rho_bar_z = rho_bar/pow((1+z),3)
	D = (3*m)/(4*PI*pow(r,3)*rho_bar_z)
	return D

def dsigma_integrand(k, R):
	A = (k*k)*(Power_spec(k)*W_kR(k,R))/(2*pi*pi)
	s = k*R
	dw = (3/(R*(s**3)))*((s*s-3.0)*sin(s)+3.0*s*cos(s))
	return A*dw

def sig_dsigma_dm(R):
	mass = (4*pi*(R**3)*rho_bar)/3
	V1 = (norm8**2)*Simpson(dsigma_integrand, 1e-7, 1e7, R)
	V3 = (R/(3*mass))*V1
	return V3

'''
Press-Schechter mass function.
--------------------------------------------------------------
'''
	
def Press_Shechter(R,z): # press-shechter

	mass = (4.0/3.0)*pi*(R**3)*rho_bar
	V1 = -1*rho_bar/mass
	V2 = sqrt(2/pi) 
	
	V3 = sig_dsigma_dm(R)/pow(Sigma(R),2)
	V4 = deltac(z)/Sigma(R)
	
	e1 = pow(deltac(z),2)/(2*pow(Sigma(R),2))
	V5 = exp(-e1)
	
	return V1*V2*V3*V4*V5
	
'''
Sheth-Tormen massfunction
--------------------------------------------------------------
'''
	
def nu_fnu(R, z):
	
	#Sheth-Tormen fitting function
	
	A = 0.3222
	a = 0.707
	p = 0.3
	
	nu = (deltac(z)/Sigma(R))**2 
	
	V1 = A*sqrt((a*nu)/(2*pi))
	V2 = 1 + (1/(a*nu))**p
	V3 = exp(-0.5*a*nu)
	
	return V1*V2*V3

def Sheth_Tormen(R,z):

	mass = (4.0/3.0)*pi*(R**3)*rho_bar
	V1 = -2*rho_bar/(Sigma(R)*mass)
	V2 = nu_fnu(R,z)
	V3 = sig_dsigma_dm(R)/(Sigma(R))
	
	return V1*V2*V3
	

'''
Tinker mass function Tinker et al. 2008
---------------------------------------------------------------
'''	

def f(R,z): # Tinker et al. fitting function

	# parameters
	A = 0.186*pow((1+z),-0.14)
	a = 1.47*pow((1+z),-0.06)
	logalpha = -pow(0.75/(np.log10(200/75)),1.2)
	b = 2.57*pow((1+z),-pow(10,logalpha))
	c = 1.19
	
	V1 = pow((D(z)*Sigma(R)/b),-a) + 1
	V2 = exp(-c/pow(Sigma(R)*D(z),2.0))
	
	return A*V1*V2  

def Tinker(R,z):

	mass = (4.0/3.0)*pi*(R**3)*rho_bar
	V1 = rho_bar/mass
	V2 = f(R,z)
	V3 = -1*sig_dsigma_dm(R)
	
	return V1*V2*V3/pow(Sigma(R),2)
	
	
Radius = [] # comoving scale 
r = 0.01
i = 1
while r<=40:
	r*=i
	Radius.append(r)
	i*=1.001

M = [4*pi*(i**3)*rho_bar/3 for i in Radius] # Halo masses


for j in [0.0, 1.0 ,2.0 ,4.0 ,6.0]:

	Press_massfn = [Press_Shechter(r,j) for r in Radius]
	Pressfn = np.column_stack((M, Press_massfn))
	np.savetxt("Press-Shecter_z{}.txt".format(str(j)),Pressfn)

	Sheth_massfn = [Sheth_Tormen(r,j) for r in Radius]
	Shethfn = np.column_stack((M, Sheth_massfn))
	np.savetxt("Sheth-Tormen_z{}.txt".format(str(j)),Shethfn)	
    
	Tinker_massfn = [Tinker(r,j) for r in Radius]
	Tinkerfn = np.column_stack((M, Tinker_massfn))
	np.savetxt("Tinker_z{}_comp.txt".format(str(j)),Tinkerfn)

