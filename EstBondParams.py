import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
import mdtraj as md
from scipy.optimize import curve_fit


def Gauss(r, Dist0, FConst): 

   Q = np.sum(np.exp(-FConst * (r - Dist0)**2))
   y = np.exp(-FConst * (r - Dist0)**2) / Q 

   return y 


def Estimate(traj=None, pair=None, offcentered=False, bins=10000):

   select = ' or '.join(['name {}'.format(_a) for _a in pair])

   index = traj.topology.select(select)
   traj_index = traj.atom_slice(index)

   bondpairs = []
   for _b in traj_index.topology.bonds:
      atypes = [_a.name for  _a in _b]
      if atypes == pair or atypes[::-1] == pair:
         bondpairs.append([_b[0].index, _b[1].index])

   print('Detect {} total {} {} bonds'.format(len(bondpairs),pair[0],pair[1]))      
   
   bondlengths = md.compute_distances(traj_index, bondpairs, periodic=False)

   b_AA = np.mean(bondlengths)
   b2_AA = np.mean(bondlengths**2)

   if offcentered:

      b_hist, r_edges = np.histogram(bondlengths, bins=bins)
      r_hist = [(r_edges[i+1]+r_edges[i])/2 for i in range(len(r_edges)-1)]
      b_prob = b_hist / np.sum(b_hist)
      
      #parameters, covariance = curve_fit(Gauss, r_hist, b_prob)
      #Dist0, FConst = parameters[0], parameters[1]

      Dist0 = b_AA
      FConst = 1 / (2*((b2_AA) - b_AA**2))

      fig = plt.figure(dpi = 1000)
      AAfit = Gauss(r_hist, Dist0, FConst)

      plt.plot(r_hist, b_prob, label='AA')
      plt.plot(r_hist, AAfit, label='fit')
      plt.title('{} {} bond'.format(pair[0], pair[1]))
      plt.xlabel('r [nm]')
      plt.ylabel('P(r)')
      plt.legend(frameon=False)
      plt.savefig('bond_{}_{}_GaussianFit.png'.format(pair[0], pair[1]))
  
      
   else:
      Dist0 = 0.0
      FConst =  3 / 2 / b2_AA

   print('{} {} bond pair'.format(pair[0], pair[1]))
   print('AA b: {}'.format(b_AA))
   print('AA b^2: {}'.format(b2_AA))
   print('\n')
   print('Dist0 estimate: {}'.format(Dist0))
   print('FConst estimate: {}'.format(FConst))

   return


   

