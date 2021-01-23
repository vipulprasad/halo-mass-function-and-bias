import numpy as np
from math import sin, cos, exp, pi, sqrt, log
import matplotlib.pyplot as plt

Mpc = 3.0857e22                     # 1 Mpc in metres.
sm = 2e30                           # Solar mass in Kg.
G = 6.67408e-11*sm/(Mpc**3)         # Gravitational constant.
h = 1 #0.6895004                    # Hubble constant in units of 100 Km/hr/Mpc.
H_o = (100*h)/(Mpc*(10**(-3)))      # Hubble constant.
omega_b = 0.05                      # Baryons density parameter.
omega_m = 0.301685                  # Matter(baryons+dark matter) density parameter.
omega_lambda = 0.698315             # Dark energy density parameter.
deltac = 1.686                      # Critical overdensity for spherical collapse at z = 0.
sigma_8 = 0.853824                  # sigma_8 used for normalization.
ns = 0.9300325                      # spectral indece of initial power spectrum.
rho_crit = (3*(H_o)**2)/(8*np.pi*G) # critical density of the universe.
rho_bar = omega_m*rho_crit          # comoving background density in kg.


def Simpson(f, a, b, m): #if a = 0 this won't work
	x = a				 #also for small intervals Simpson has more relative 
	h  = a*1.01			 #error than Trapezoidal below
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

	gamma = h*h*omega_m*exp(-omega_b-(sqrt(2*h)*(omega_b/omega_m)))
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
	return ((3*m)/(4*PI*rho_bar))**(1/3)

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
R8 = 8.0/h
sigma8 = Sigma(R8)
norm8= sigma_8/sigma8
print(R8,Sigma(R8), norm8) 

'''
Press-Shechter bias
----------------------------------------------------------------
'''

def Press_Shechter(R,z):

	V1 = pow((deltac(z)/Sigma(R)),2)
	V2 = (V1 - 1)/deltac(z)
	return 1 + V2	


'''
Tinker bias.
-------------------------------------------------------------
'''

def Tinker(R, z):
	# Tinker et al. 2010.	
	
	A = 1.0
	B = 0.183
	C = 0.265
	a = 0.132
	b = 1.5
	c = 2.4
	
	nu = deltac(z)/(Sigma(R))
	V1 = pow(nu,a)/(pow(nu,a)+pow(1.686,a))
	V3 = 1-A*V1+B*pow(nu,b)+C*pow(nu,c) 
	return V3
	
Radius = []
r = 0.01
i = 1
while r<=40:
	r*=i
	Radius.append(r)
	i*=1.001

M = [4*pi*(i**3)*rho_bar/3 for i in Radius]


for i in [0.0 ,1.0 ,2.0 ,4.0 ,6.0]:

	Tinker_bias = [Tinker(j,i) for j in Radius]
	Tinker_bias = np.column_stack((M,Tinker_bias))
	np.savetxt("Tinker-bias_z{}.txt".format(i), Tinker_bias)
	
	Press_bias = [Press_Shechter(j,i) for j in Radius]
	Press_bias = np.column_stack((M,Press_bias))
	np.savetxt("Press-bias_z{}.txt".format(i), Press_bias)
	

