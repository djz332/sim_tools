import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import mdtraj as md
   

def TrapezoidQuadrature(r, p, asmear):

   kappa = 1 / (4.*asmear**2)
   dUdlam = 0
   for i in range(1, len(r)):
      dUdlam += 0.5 * (r[i]-r[i-1]) * (np.exp(-kappa*r[i]**2)*p[i] + np.exp(-kappa*r[i-1]**2)*p[i-1])
   
   return dUdlam

def SimpsonQuadrature(r, p, asmear):

   kappa = 1 / (4.*asmear**2)
   y = np.exp(-kappa*np.asarray(r)**2)*np.asarray(p)
   dUdlam = simpson(y, r)

   return dUdlam

def GetEnergyDerivatives(AAtraj=None, CGtraj=None, asmear=None, rcut=None, keywords=None, species=None):

   r_range = [0,rcut]

   index1 = AAtraj.topology.select('{} {}'.format(keywords[0][0],species[0][0]))
   index2 = AAtraj.topology.select('{} {}'.format(keywords[0][1],species[0][1]))

   pairs = AAtraj.topology.select_pairs(selection1=index1, selection2=index2)

   rAA, gAA = md.compute_rdf(AAtraj, pairs=pairs, r_range=r_range)

   index1 = CGtraj.topology.select('{} {}'.format(keywords[1][0],species[1][0]))
   index2 = CGtraj.topology.select('{} {}'.format(keywords[1][1],species[1][1]))

   pairs = CGtraj.topology.select_pairs(selection1=index1, selection2=index2)

   rCG, gCG = md.compute_rdf(CGtraj, pairs=pairs, r_range=r_range)

   pAA = rAA**2 * gAA * 4 * np.pi * (rAA[1]-rAA[0])
   pCG = rCG**2 * gCG * 4 * np.pi * (rCG[1]-rCG[0])

   dUdlam_AA = SimpsonQuadrature(rAA, pAA, asmear)
   dUdlam_CG = SimpsonQuadrature(rCG, pCG, asmear)

   fig = plt.figure(dpi = 1000)
   plt.ylabel('g(r)')
   plt.xlabel('r [nm]')
   plt.plot(rAA,gAA, label='AA')
   plt.plot(rCG,gCG, label='CG')
   plt.legend(frameon=False)
   plt.title('pair {} {}'.format(species[0][0], species[0][1]))
   plt.savefig('{}_{}_g_r.png'.format(species[0][0], species[0][1]))
   
   print('\nConvergence test result for pair potential {} {}:'.format(species[0][0], species[0][1]))
   print('<dUdlam> AA: {}'.format(dUdlam_AA))
   print('<dUdlam> CG: {}'.format(dUdlam_CG))
   print('difference: {} \n'.format(np.abs(dUdlam_AA-dUdlam_CG)))

   return 