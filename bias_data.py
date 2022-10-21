import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h = 1#0.6895004 hubble constant in units of 100km/hr/Mpc
boxsize = 1100/h # the size of the simulation volume in Mpc.

'''
halos1.txt is random sampled from the original data which contain 8661680 halos.

'''
halos = pd.read_csv('halos1.txt', delimiter = '\t', names = [ 'x', 'y', 'z','mass']) 
halos = halos/h

total_data = 8661680
ratioh = total_data/len(halos)

def deltah(s):

    '''
    Here we devide the box into several smaller boxes and will compute the mass function in each cell.
    The standard deviation of the mass function in each cell is taken as the deviation of halos number density.
    which will be a function of mass.
    '''
    
    split = np.linspace(0, boxsize*((s-1)/s), num = s)
    dl = boxsize/s
    hlist =[]
    for i in split:
        for j in split:
            for k in split:
                lists = halos.mass[(halos.x>=i) & (halos.x<(i+dl)) & (halos.y>=j) & (halos.y<(j+dl)) & (halos.z>=k) & (halos.z<(k+dl))]
                hlist.append(lists)
    
    n=10
    bins=np.linspace(np.log(min(halos.mass)),np.log(max(halos.mass)), num=n)

    binwidth= []
    for i in range(n-1):
        width = bins[i+1]-bins[i]
        binwidth.append(width)
        
    binmid = []
    for i in range(n-1):
        mid = (bins[i+1]+bins[i])*0.5
        binmid.append(mid)
    binmidmass= [np.exp(i) for i in binmid]
    
    massfn = []
    for i in range(s**3):
    	number, bins = np.histogram(np.log(hlist[i]), bins = bins)
    	number = np.divide(number, binwidth)
    	number = np.divide(number, binmidmass)
    	massfn.append(number)
    
    mean_massfn = []	
    for i in range(n-1):
    	mean = []
    	for j in range(s**3):
    		mean.append(massfn[j][i])
    	mean_massfn.append(sum(mean)/(s**3))
    
    std_dev = []
    for i in range(len(massfn[0])):
    	var = []
    	for j in range(s**3):
    		A = (massfn[j][i] - mean_massfn[i])**2
    		var.append(A)
    	std = np.sqrt(sum(var)/(s**3))
    	std_dev.append(std)
    
    delta = np.divide(std_dev,mean_massfn)
    
    return [delta, binmidmass]

   
'''
particles1.txt is a random sampled from the original file which contain roughly 52480000 particles
'''
part = pd.read_csv('particles1.txt', delimiter = '\t', names = ['x', 'y', 'z'])
part = part/h
total_data = 52480000
ratiop = total_data/len(part)

def deltap(s):

    '''
    Here we devide the box into several smaller boxes and will compute the number of particles in each cell.
    The standard deviation of the number of particles in each cell is taken as the deviation of matter density.
    '''
	
    split = np.linspace(0, boxsize*((s-1)/s), num = s)

    particles = []
    for i in split:
        for j in split:
            for k in split:
                par = part[(part.x>=i)&(part.x<(i+(boxsize/s)))&(part.y>=j)&(part.y<(j+(boxsize/s)))&(part.z>=k)&(part.z<(k+(boxsize/s)))]
                particles.append(par)

    part_num = [len(pa) for pa in particles]
    mean_num = sum(part_num)/len(part_num)
    
    std = []
    for i in range(len(part_num)):
    	var = (part_num[i] - mean_num)**2
    	std.append(var)
    msqr = sum(std)/(len(std))
    stdev = np.sqrt(msqr)
    
    delta = stdev/mean_num
    
    return delta
    
delta_halo = [deltah(i) for i in range(3, 9)]
    
delta_particle = [deltap(i) for i in range(3, 9)]

bias = []
for i in range(len(delta_particle)):
	b = np.divide(delta_halo[i][0],delta_particle[i])
	bias.append(b)
	bia = np.column_stack((delta_halo[0][1], b))
	np.savetxt("bias_scale{}.txt".format(str(i+3)), bia)
 
mean_bias = []
for i in range(len(bias[0])):
    b = []
    for f in bias:
        b.append(f[i])
    mean_bias.append(sum(b)/len(b))

np.savetxt("bias_data.txt", np.column_stack((delta_halo[0][1], mean_bias)))

