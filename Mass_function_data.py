import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h = 1 #0.6895004

'''
data = np.loadtxt('halo_saple_new.txt') # total data
halos=np.random.choice(mass, 1000000, replace = False) # randomly picking 
'''

halo_list=np.loadtxt('halos1.txt')/h # randomly picked data.
Boxsize = 1100/h
total_data = 8661680
sample_size = len(halo_list)
print(sample_size)
ratio = total_data/sample_size



def mass_function(s):
    
    '''
    Here we divide the total volume into some subvolumes and calculate the mass function
    in each volume and take the mean.
    The standard deviation of massfunctions in different volume is taken as the errorbar.
    '''

    halos = pd.DataFrame(halo_list, columns=[ 'x', 'y', 'z','mass'])
    dl = Boxsize/s
    split = np.linspace(0, Boxsize*((s-1)/s), num = s)
    hlist =[]
    for i in split:
        for j in split:
            for k in split:
                lists = halos.mass[(halos.x>=i) & (halos.x<(i+dl)) & (halos.y>=j) & (halos.y<(j+dl)) & (halos.z>=k) & (halos.z<(k+dl))]
                hlist.append(lists)
	
    '''
    The mass function is found by plotting a histogram.
    The bins are in logarithmic scale. 
    Or here all the masses are converted to its natural logs, 
    then histogram is computed using linear bins.
    '''

    n=10 # number of bins used for plotting histograms
    bins=np.linspace(np.log(min(halos.mass)),np.log(max(halos.mass)), num=n)
    
    binwidth= []
    for i in range(n-1):
        width = bins[i+1]-bins[i]
        binwidth.append(width)
        
    binmid = []
    for i in range(n-1):
        mid = (bins[i+1]*bins[i])**0.5
        binmid.append(mid)
        
    binmidmass= [np.exp(i) for i in binmid]

    massfunc = []
    for i in hlist:
        (number,bins)=np.histogram(np.log(i), bins=bins)
        number = number*ratio*((s/Boxsize)**3)
        number = np.divide(number, binwidth)
        number = np.divide(number, binmidmass)
        massfunc.append(number)
    
    mean_massfunc = []
    for i in range(n-1):
	    masf = []
	    for fun in massfunc:
		    masf.append(fun[i])
	    mean_massfunc.append(np.sum(masf)/len(masf))
   
    standev = []
    for i in range(n-1):
      var = []
      for f in massfunc:
        var.append((f[i] - mean_massfunc[i])**2)
      std = np.sum(var)/len(var)
      standev.append(np.sqrt(std))
      
    print('std({})'.format(s))
    
    return binmidmass, mean_massfunc, standev
    
'''
mass function for different scales
'''

for i in range(3,9):
	m, f, s = mass_function(i)
	A = np.column_stack((m,f,s))
	np.savetxt("mass-function_{}.txt".format(i), A)

