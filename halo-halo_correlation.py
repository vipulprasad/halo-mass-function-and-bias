import numpy as np
from math import sin, cos, exp, pi, sqrt, log, log10, floor
import matplotlib.pyplot as plt
from tqdm import tqdm

Mpc = 3.0857e22                     # 1 Mpc in metres.
sm = 2e30                           # Solar mass in Kg.
G = 6.67408e-11*sm/(Mpc**3)         # Gravitational constant.
h = 1#0.68                    	# Hubble constant in units of 100 Km/hr/Mpc.
H_o = (100*h)/(Mpc*(10**(-3)))      # Hubble constant.
omega_b = 0.048                    # Baryons density parameter.
omega_m = 0.31                  	# Matter(baryons+dark matter) density parameter.
omega_lambda = 0.69            	# Dark energy density parameter.
deltac = 1.686                      # Critical overdensity for spherical collapse at z = 0.
sigma_8 = 0.83                  	# sigma_8 used for normalization.
ns = 0.96                      	# spectral indece of initial power spectrum.
rho_crit = (3*(H_o)**2)/(8*np.pi*G) # critical density of the universe.
rho_bar = omega_m*rho_crit          # comoving background density in kg.


'''
Simpson and Trapezoidal rule with increasing step size.
This is because the integrating variable k(Mpc ^-1) spans across
different order of magnitudes.
'''

def Simpson(f, a, b, m): #if a = 0 this won't work
	x = a				 #also for small intervals Simpson has more relative
	dx  = a*1.01			 #error, and more error than Trapezoidal below
	s = 0

	while x<b:
		s += (dx/3)*(f(x,m)+4*f(x+dx,m)+f(x+2*dx,m))
		x += 2*dx
		dx*=1.01

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
	integral = Simpson(sigma_integrand, 1e-3, 1e3, R)
	return norm8*sqrt(integral)

'''
Normalising the power spectrum for using the known value of sigma,
at R = 8.0/h
'''

norm8 = 1
R8 = 8.0/h
sigma8 = Sigma(R8)
norm8 = sigma_8/sigma8
print(R8,Sigma(R8), norm8)

def dsigma_integrand(k, R):
	A = (k*k)*(Power_spec(k)*W_kR(k,R))/(2*pi*pi)
	s = k*R
	dw = (3/(R*(s**3)))*((s*s-3.0)*sin(s)+3.0*s*cos(s))
	return A*dw

def sig_dsigma_dm(R):
	mass = (4*pi*(R**3)*rho_bar)/3
	V1 = (norm8**2)*Simpson(dsigma_integrand, 1e-3, 1e3, R)
	V3 = (R/(3*mass))*V1
	return V3



'''
Tinker mass function Tinker et al. 2008
---------------------------------------------------------------
'''

def Tf(r,z): # Tinker et al. fitting function

	# parameters
	A = 0.186*pow((1+z),-0.14)
	a = 1.47*pow((1+z),-0.06)
	logalpha = -pow(0.75/(np.log10(200/75)),1.2)
	b = 2.57*pow((1+z),-pow(10,logalpha))
	c = 1.19

	V1 = pow((D(z)*Sigma(r)/b),-a) + 1
	V2 = exp(-c/pow(Sigma(r)*D(z),2.0))

	return A*V1*V2

def Tinker_mass(m,z):

	r = R(m)

	mass = (4.0/3.0)*pi*(r**3)*rho_bar
	V1 = rho_bar/mass
	V2 = Tf(r,z)
	V3 = -1*sig_dsigma_dm(r)

	return V1*V2*V3/pow(Sigma(r),2)


'''
Tinker bias.
-------------------------------------------------------------
'''

def Tinker_bias(m, z):
	# Tinker et al. 2010.

	r = R(m)

	A = 1.0
	B = 0.183
	C = 0.265
	a = 0.132
	b = 1.5
	c = 2.4

	nu = deltac(z)/(Sigma(r))
	V1 = pow(nu,a)/(pow(nu,a)+pow(1.686,a))
	V3 = 1-A*V1+B*pow(nu,b)+C*pow(nu,c)
	return V3



#######################################################################################

def integral(f, a, b, Num, z):
	h = (b-a)/Num
	sum = f(a,z) + f(b, z)
	for i in tqdm(range(1, Num)):
		if i%2 == 0:
			sum += 2*f(a+i*h, z)
		else:
			sum += 4*f(a+i*h, z)
	return (h/3)*(sum)

def Simpson1(f, a, b, m): #if a = 0 this won't work
	x = a				 #also for small intervals Simpson has more relative
	dx  = a*1.001			 #error, and more error than Trapezoidal below
	s = 0

	while x<b:
		s += (dx/3)*(f(x,m)+4*f(x+dx,m)+f(x+2*dx,m))
		x += 2*dx
		dx*=1.001

	return s


def bh_integrand(m, z):
	return Tinker_bias(m, z)*Tinker_mass(m, z)

def n(z, m1, m2):
	V2 = integral(Tinker_mass, m1, m2, 100,  z)
	#V2 = Simpson1(Tinker_mass, m1, m2, z)
	return V2

def bh(z, m1, m2):

	#A = integral(bh_integrand, m1, m2, 100, z)
	A = integral(bh_integrand, m1, m2, 100, z)
	return A/n(z, m1, m2)



Ml = 5e13
Mu = 6e13
Z = 0
#print(n(0, Ml, Mu)*(560**3))
#print(D(0))

Bh2 = pow(bh(Z, Ml/h, Mu/h), 2)

###################################################################################

def Simpson2(f, a, b, m, z): #if a = 0 this won't work
	x = a				 #also for small intervals Simpson has more relative
	dx  = a*.001			 #error, and more error than Trapezoidal below
	s = 0

	while x<b:
		s += (dx/3)*(f(x, m, z)+4*f(x+dx, m, z)+f(x+2*dx, m, z))
		x += 2*dx
		dx*=1.001

	return s

def correlation_mm_integrand(k, r ,z):
	B = norm8*norm8*Power_spec(k)*D(z)*D(z)
	A = k*sin(k*r)/r
	return B*A

def correlation_mm(r, z):
	V1 = Simpson2(correlation_mm_integrand, 1e-3, 1e3, r, z)
	return V1/(2*pi*pi)

r_list = np.exp(np.linspace(log(0.1), log(70), 100))#np.linspace(0.1, 50, 100) #

correl = []
for i in tqdm(r_list):
	correl.append(correlation_mm(i, Z))

from cluster_toolkit import xi
xi_mm = xi.xi_mm_at_r(r_list, )

ldr = floor(log10(Ml))
udr = floor(log10(Mu))

correlation_hh = []
for j in correl:
    correlation_hh.append(Bh2*j)

output_file = "analytical_({}-{})_z{}.txt".format(str(Ml/10**ldr) + '^'+str(ldr), str(Mu/10**udr) + '^'+str(udr), Z)

np.savetxt("results/{}".format(output_file), np.column_stack((r_list, correlation_hh)))
