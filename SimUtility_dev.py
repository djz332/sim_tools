import os, sys, copy
import subprocess
import time

import argparse as ap
import numpy as np
import mdtraj as md

import sim

class SimExporter():


    def __init__(self):

        self.MapOp='COM' 
        self.ForceFieldFile=None
        self.Bmin=None
        self.Electrostatics=False
        self.NeutralSrel=False 
        self.Coef=1.0
        self.Shift=True
        self.rcut=1.0
        self.Temp=1.0
        self.Pressure=0.0 
        self.Axis=None 
        self.Tension=None 
        self.dt=0.1 
        self.LangevinGamma=1.0 
        self.MDEngine='openmm'
        self.StepsEquil=0 
        self.StepsProd=0 
        self.StepsStride=0
        self.HessianAdjustMode=0
        self.MinReweightFrac=0.1
        self.BondEstimate=False
        self.AngleEstimate=False
        self.SplineEstimate=False
        self.MinBondOrd=None

        # initialize default empty arrays
        self.AtomTypes, self.MolTypes, self.Bonds = [], [], []
        self.PBonds, self.PAngles, self.PNonbonded, self.PElectrostatic, self.PExternal = [], [], [], [], []
        self.Systems, self.Trajectories, self.Optimizers, self.InitialPositions, self.Runs = [], [], [], [], []
        self.MeasuresRee2, self.MeasuresRee, self.MeasuresArea, self.Penalties = [], [], [], []


    def CreateSystem(self):

        print('\n========== Defining System ==========')
        self.atoms_dict = {}
        self.mols_dict = {}
        self.Sys_dict = {}

        for entry in self.AtomTypes:
            self.atoms_dict[entry[0]] = sim.chem.AtomType(entry[0], Mass = entry[1], Charge = entry[2])
        print('Defined atom types:')
        print(self.atoms_dict)

        for entry in self.MolTypes:                  
            atomtypes = [self.atoms_dict[aname] for aname in entry[1]]
            self.mols_dict[entry[0]] = sim.chem.MolType(entry[0], atomtypes)
        
        for entry in self.Bonds:
            for bond in entry[1]:
                self.mols_dict[entry[0]].Bond(bond[0],bond[1])
        print('Defined molecule types:')
        print(self.mols_dict)

        World = sim.chem.World([self.mols_dict[entry[0]] for entry in self.MolTypes], Dim = 3, Units = sim.units.DimensionlessUnits)
        
        for entry in self.Systems:
            Sys = sim.system.System(World, Name = entry[0])
            print('\nCreating system: {}'.format(Sys.Name))
            for molname, nmol in zip(entry[1], entry[2]):
                print('Adding {} {} to system'.format(nmol, molname))
                for n in range(nmol):
                    Sys+= self.mols_dict[molname].New()
            if entry[3] is not None: Sys.BoxL = entry[3]
            self.Sys_dict[entry[0]] = Sys

            # ElecSys: system with Ewald on that is used to run MD, used to speed up when only optimizing non-charged interactions for speed up
            if  self.Electrostatics and self.NeutralSrel: 
                #ElecSys = copy.deepcopy(Sys) 
                #ElecSys.name = 'ElecSys_{}'.format(Sys.name)
                ElecSys = sim.system.System(World, Name = 'ElecSys_{}'.format(Sys.Name))
                print('\nCreating system: {}'.format(ElecSys.Name))
                for molname, nmol in zip(entry[1], entry[2]):
                    print('Adding {} {} to system'.format(nmol, molname))
                    for n in range(nmol):
                        ElecSys += self.mols_dict[molname].New()
                if entry[3] is not None: ElecSys.BoxL = entry[3]
                self.Sys_dict[ElecSys.Name] = ElecSys

        for Sys in self.Sys_dict.values():
            Sys.TempSet = self.Temp
            Sys.PresSet = self.Pressure
            Sys.BarostatOptions['tension'] = self.Tension
            Sys.PresAx = self.Axis
            Sys.Int.Method = Sys.Int.Methods.VVIntegrate
            Sys.Int.Method.Thermostat = Sys.Int.Method.ThermostatLangevin
            Sys.Int.Method.TimeStep = self.dt
            Sys.Int.Method.LangevinGamma = self.LangevinGamma
            
        return 


    def SetupPotentials(self):

        self.Potentials_dict = {}
        print('\n...Setting Bmin to {}'.format(self.Bmin))
        print('\n...Setting cut off radius to {} nm'.format(self.rcut))

        for Sys in self.Sys_dict.values():

            #bonded interactions
            for bond in self.PBonds:
                atomtypes = sim.atomselect.PolyFilter([self.atoms_dict[bond[1][0]], self.atoms_dict[bond[1][1]]], Bonded=True)
                
                if bond[0] == 'harmonic':
                    Label = 'bond_{}_{}'.format(bond[1][0], bond[1][1])
                    Dist0 = bond[2][0]
                    FConst = bond[2][1]
                    P = sim.potential.Bond(Sys, Label=Label, Filter=atomtypes, Dist0=Dist0, FConst=FConst)
                    P.Dist0.Fixed = bond[3][0]
                    P.FConst.Fixed = bond[3][1]
                    Sys.ForceField.append(P)

            #angle interactions
            for angle in self.PAngles:
                atomtypes = sim.atomselect.PolyFilter([self.atoms_dict[angle[1][0]], self.atoms_dict[angle[1][1]], self.atoms_dict[angle[1][2]]], Bonded=True)
                
                if angle[0] == 'harmonic':
                    Label = 'angle_{}_{}_{}'.format(angle[1][0], angle[1][1], angle[1][2])
                    Theta0 = angle[2][0]
                    FConst = angle[2][1]
                    P = sim.potential.Angle(Sys, Label=Label, Theta0=Theta0, FConst=FConst, Filter=atomtypes)
                    P.Theta0.Fixed = angle[3][0]
                    P.FConst.Fixed = angle[3][1]
                    Sys.ForceField.append(P)

                elif angle[0] == 'anglespline':
                    Label = 'anglespline_{}_{}_{}'.format(angle[1][0], angle[1][1], angle[1][2])
                    NKnot =  nonbonded[2]
                    P = sim.potential.AngleSpline(Sys, Label=Label, NKnot=NKnot, Filter=atomtypes)
                    P.Fixed = angle[3]
                    Sys.ForceField.append(P)

            #nonbonded interactions
            for nonbonded in self.PNonbonded:
                atomtypes = sim.atomselect.PolyFilter([self.atoms_dict[nonbonded[1][0]], self.atoms_dict[nonbonded[1][1]]], MinBondOrd=self.MinBondOrd)
                
                if nonbonded[0] == 'gaussian':
                    Label = 'gauss_{}_{}'.format(nonbonded[1][0], nonbonded[1][1])
                    B =  nonbonded[2][0]
                    Kappa = nonbonded[2][1]
                    P = sim.potential.Gaussian(Sys, Label=Label, Filter=atomtypes, Cut=self.rcut, B=B, Kappa=Kappa, Dist0=0.0, Shift=True)
                    P.B.Fixed = nonbonded[3][0]
                    P.Kappa.Fixed = nonbonded[3][1]
                    P.Dist0.Fixed = True
                    P.Param.Min = self.Bmin
                    Sys.ForceField.append(P)

                elif nonbonded[0] == 'pairspline':
                    Label = 'spline_{}_{}'.format(nonbonded[1][0], nonbonded[1][1])
                    NKnot =  nonbonded[2]
                    P = sim.potential.PairSpline(Sys, Label=Label, Cut=self.rcut, NKnot=NKnot, Filter=atomtypes)
                    P.Fixed = nonbonded[3]
                    Sys.ForceField.append(P)

            #electrostatic potentials 
            if self.Electrostatics: #turn on ewald summation and smeared coulumb 
                if (self.NeutralSrel and 'ElecSys_' in Sys.Name) or self.NeutralSrel == False:
                    print('\n...Setting up Ewald summation with Bjerrum length: {} nm for {}'.format(Coef, Sys.Name))
                    P = sim.potential.Ewald(Sys, ExcludeBondOrd=0, Cut=self.rcut, Shift=self.Shift, Coef=Coef, FixedCoef=True, Label='ewald')
                    Sys.ForceField.append(P)

                    for elec in self.PElectrostatic:
                        atomtypes = sim.atomselect.PolyFilter([self.atoms_dict[elec[1][0]], self.atoms_dict[elec[1][1]]])
                        if elec[0] == 'smearedcoulomb':
                            Label = 'smeared_corr_{}_{}'.format(elec[1][0], elec[1][1])
                            Coef=elec[2][0]
                            BornA=elec[2][1]
                            #print(Coef,BornA)
                            P = sim.potential.SmearedCoulombEwCorr(Sys, Label=Label, Filter=atomtypes, Cut=self.rcut, Coef=Coef, BornA=BornA, Shift=self.Shift)
                            P.FixedCoef = elec[3][0]
                            P.FixedBornA = elec[3][1]
                            Sys.ForceField.append(P)
                
            #external potentials
            for external in self.PExternal:
                if external[0] == 'gaussian':
                    atomtypes = sim.atomselect.PolyFilter([self.atoms_dict[external[1][0]], self.atoms_dict[external[1][1]]])
                    Label = 'ext_gauss_{}_{}'.format(external[1][0], external[1][1])
                    B = external[2][0]
                    Kappa = external[2][1]
                    P = sim.potential.Gaussian(Sys, Label=Label, Filter=atomtypes, Cut=self.rcut, B=B, Kappa=Kappa, Dist0=0.0, Shift=True)
                    P.Param.Min = None
                    P.B.Fixed = True
                    P.Kappa.Fixed = True
                    P.Dist0.Fixed = True
                    Sys.ForceField.append(P)

                if external[0] == 'sinusoid':
                    atomtypes = sim.atomselect.PolyFilter([self.atoms_dict[external[1][0]]])
                    Label = 'ext_sin_{}'.format(external[1][0])
                    UConst = external[2][0]
                    NPeriods = external[2][1]
                    PlaneAxis = external[2][2]
                    PlaneLoc = external[2][3]
                    P = sim.potential.ExternalSinusoid(Sys, Label=Label, Filter=atomtypes, Fixed=True, UConst=UConst, NPeriods=NPeriods, PlaneAxis=PlaneAxis, PlaneLoc=PlaneLoc)
                    Sys.ForceField.append(P)
                        
            #setup histograms 
            for P in Sys.ForceField:
                P.Arg.SetupHist(NBin = 10000, ReportNBin=100)
                self.Potentials_dict[P.Label] = P
            
            #read in forcefield field
            if self.ForceFieldFile and os.path.exists(self.ForceFieldFile):
                print('\n...Reading forcefield parameters from {} for {}'.format(self.ForceFieldFile, Sys.Name))
                with open(self.ForceFieldFile, 'r') as of: s = of.read()
                Sys.ForceField.SetParamString(s, CheckMissing = False)

        for Sys_name, Sys in self.Sys_dict.items():
            print('\nForcefield for {}: '.format(Sys_name))
            for P in Sys.ForceField:
                print(P.Label)

        return 


    def AddMeasures(self):

        print('\n========== Adding measures ==========')
        
        if self.MeasuresRee2 == [] and self.MeasuresRee == [] and self.MeasuresArea == []:
            print('\n... No measures specified')
            return

        for entry in self.MeasuresRee2:
            Sys = self.Sys_dict[entry[1]]
            mol_name = entry[2]
            site_IDs = entry[3]
            Sys_mol = [mol for mol in Sys.World if mol.Name == mol_name][0]
            site1, site2 = Sys.World.SiteTypes[Sys_mol[site_IDs[0]].SID], Sys.World.SiteTypes[Sys_mol[site_IDs[1]].SID]
            filter = sim.atomselect.PolyFilter(Filters=[site1, site2],Intra=True)
            dist_ree2 = sim.measure.distang.Distance2(Sys, Filter=filter, Name=entry[0])
            Sys.Measures.append(dist_ree2)
            print('{}: added Ree2 measure between sites {} and {}'.format(Sys.Name, site1, site2))

        for entry in self.MeasuresRee:
            Sys = self.Sys_dict[entry[1]]
            mol_name = entry[2]
            site_IDs = entry[3]
            Sys_mol = [mol for mol in Sys.World if mol.Name == mol_name][0]
            site1, site2 = Sys.World.SiteTypes[Sys_mol[site_IDs[0]].SID], Sys.World.SiteTypes[Sys_mol[site_IDs[1]].SID]
            filter = sim.atomselect.PolyFilter(Filters=[site1, site2],Intra=True)
            dist_ree = sim.measure.distang.Distance(Sys, Filter=filter, Name=entry[0])
            Sys.Measures.append(dist_ree)
            print('{}: added Ree measure between sites {} and {}'.format(Sys.Name, site1, site2))

        for entry in self.MeasuresArea:
            Sys = self.Sys_dict[entry[1]]
            Az = sim.measure.Az(Sys,axis=entry[2])
            Sys.Measures.append(Az)
            print('\n{}: added area measure for axis {} '.format(Sys.Name, entry[2]))

        return 

    def CompileSystems(self):

        print('\n========== Loading and locking system (compiling) ==========')
        for Sys in self.Sys_dict.values():
            Sys.Load()
            
        return 


    def CreateMapping(self):
        
        print('\n========== Creating Atom Maps ==========')

        self.Mapping_dict = {}
        mol_mapping = {}
        for entry in self.MolTypes:
            mol_mapping[entry[0]] = entry[2]

        system_traj_dict = {}
        trajfile_dict = {}
        for Opt_entry in self.Optimizers:
            system_traj_dict[Opt_entry[1]] = Opt_entry[2]
        for traj_entry in self.Trajectories:
            trajfile_dict[traj_entry[0]] = [traj_entry[1],traj_entry[2]]

        for Sys_entry in self.Systems:
            if self.MapOp=='COM':
                traj_name = system_traj_dict[Sys_entry[0]]
                if not os.path.exists(trajfile_dict[traj_name][1]): 
                    Warning('{} topology file does not exits ... defaulting to centroid mapping'.format(trajfile_dict[traj_name][1]))
                    masses=None
                else:
                    traj = md.load(trajfile_dict[traj_name][0], top=trajfile_dict[traj_name][1])
                    masses = np.array([a.element.mass for a in traj.top.atoms])
                    #print(masses)
                    if (np.sum(masses)==np.nan or np.sum(masses)==0.0): 
                        Warning('Atom masses include nan or are all 0.0 ... defaulting to centroid mapping')
                        masses=None 
            elif self.MapOp=='Centroid': masses = None 

            Sys = self.Sys_dict[Sys_entry[0]]
            mapping = sim.atommap.PosMap()
        
            # finally build the mapping
            ia = 0
            shift = 0
            for i in range(len(Sys_entry[1])): # number of CG molecule types
                molmap = mol_mapping[Sys_entry[1][i]] 
                for j in range(Sys_entry[2][i]): # number of CG molecules of type i
                    for k in range(1,len(molmap)):
                        atom = Sys.Atom[ia]
                        aa_indices = molmap[k] + shift
                        if masses is not None: 
                            ia_masses = np.array([masses[ia] for ia in aa_indices])
                        if (np.sum(ia_masses)==np.nan or np.sum(ia_masses)==0.0):
                            Warning('Atom masses are not well defined for CG bead type {} ... defaulting to centroid mapping'.format(atom.Name))
                            ia_masses=None
                        else: 
                            ia_masses = None
                        mapping.Add(Atoms1=aa_indices, Atom2=atom, Mass1=ia_masses)
                        ia += 1
                    shift += molmap[0] # shift index values by number of AA indices in CG molecule type i

            #mapping.Print()
            self.Mapping_dict[Sys_entry[0]] = mapping
            
        return
    

    def DepreciatedCreateMapping(self):
        
        print('\n========== Creating Atom Maps ==========')

        self.Mapping_dict = {}
        mol_mapping = {}
        for entry in self.MolTypes:
            mol_mapping[entry[0]] = entry[2]

        system_traj_dict = {}
        trajfile_dict = {}
        for Opt_entry in self.Optimizers:
            system_traj_dict[Opt_entry[1]] = Opt_entry[2]
        for traj_entry in self.Trajectories:
            trajfile_dict[traj_entry[0]] = [traj_entry[1],traj_entry[2]]

        for Sys_entry in self.Systems:
            if self.MapOp=='COM':
                traj_name = system_traj_dict[Sys_entry[0]]
                if not os.path.exists(trajfile_dict[traj_name][1]): 
                    Warning('{} topology file does not exits ... defaulting to centroid mapping'.format(trajfile_dict[traj_name][1]))
                    masses=None
                else:
                    traj = md.load(trajfile_dict[traj_name][0], top=trajfile_dict[traj_name][1])
                    masses = np.array([a.element.mass for a in traj.top.atoms])
                    #print(masses)
                    if (np.sum(masses)==np.nan or np.sum(masses)==0.0): 
                        Warning('Atom masses include nan or are all 0.0 ... defaulting to centroid mapping')
                        masses=None 
            elif self.MapOp=='Centroid': masses = None 

            Sys = self.Sys_dict[Sys_entry[0]]
            mapping = sim.atommap.PosMap()

            # finally build the mapping
            index=0
            nmol = 0
            j = 0
            k = 0
            molmap = mol_mapping[Sys_entry[1][j]]
            for i, atom in enumerate(Sys.Atom):
                #print(list(range(index,index+molmap[k])),atom)
                aa_indices = list(range(index,index+molmap[k]))
                if masses is not None: 
                    ia_masses = np.array([masses[ia] for ia in aa_indices])
                    if (np.sum(ia_masses)==np.nan or np.sum(ia_masses)==0.0):
                        Warning('Atom masses are not well defined for CG bead type {} ... defaulting to centroid mapping'.format(atom.Name))                    
                        ia_masses = None
                else: 
                    ia_masses = None

                #print(ia_masses)

                mapping.Add(Atoms1=aa_indices, Atom2=atom, Mass1=ia_masses)
                index += molmap[k]
            
                if k == len(molmap) - 1: 
                    nmol += 1
                    k = 0
                else: k += 1
                
                if nmol == Sys_entry[2][j]:
                    if j + 1 < len(Sys_entry[2]):
                        nmol = 0
                        j += 1
                        molmap = mol_mapping[Sys_entry[1][j]]

            #mapping.Print()
            self.Mapping_dict[Sys_entry[0]] = mapping
            
        return


    def LoadTrajectories(self):
        
        self.Traj_dict = {}
        for traj_entry in self.Trajectories:
            if traj_entry[3] == 'dcd':
                traj = sim.traj.dcd.DCD(TrjFile=traj_entry[1], TopFile=traj_entry[2], ConversionFactor=1)
            elif traj_entry[3] == 'pdb':
                traj = sim.traj.pdb.Pdb(PdbFile=traj_entry[1])
            elif traj_entry[3] == 'lammpstrj':
                traj = sim.traj.lammps.Lammps(TrjFile=traj_entry[1])
            elif traj_entry[3] == 'xyz':
                traj = sim.traj.xyz.XYZ(TrjFile=traj_entry[1])
            self.Traj_dict[traj_entry[0]] = traj

        return 


    def EstimateBondParameters(self, Opt):

        for P in Opt.ModSys.ForceField:
            if isinstance(P,sim.potential.Bond):
                if True in [P.Dist0.Fixed, P.FConst.Fixed]:
                    Warning('Cannot estimate bonded parameters for {} due to fixed parameters'.format(P.Label))
                else:
                    try:
                        P.Estimate()
                        print('\n...Estimating bonded parameters for {}:'.format(P.Label))
                        print('     Dist0: {}'.format(P.Dist0[0]))
                        print('     FConst: {}'.format(P.FConst[0]))
                    except:
                        Warning('Could not properly estimate bonded parameters for {}'.format(P.Label))
                    
        return 
    

    def EstimateAngleParameters(self, Opt):

        for P in Opt.ModSys.ForceField:
            if isinstance(P,sim.potential.Angle):
                if True in [P.Theta0.Fixed, P.FConst.Fixed]:
                    Warning('Cannot estimate angle parameters for {} due to fixed parameters'.format(P.Label))
                else:
                    try:
                        P.Estimate()
                        print('\n...Estimating angle parameters for {}:'.format(P.Label))
                        print('     Theta0: {}'.format(P.Theta0[0]))
                        print('     FConst: {}'.format(P.FConst[0]))
                    except:
                        Warning('Could not properly estimate angle parameters for {}'.format(P.Label))
 
        return 
    

    def EstimateSplineParameters(self, Opt):

        for P in Opt.ModSys.ForceField:
            if isinstance(P,sim.potential.PairSpline):
                if True in P.Fixed:
                    Warning('Cannot estimate spline parameters for {} due to fixed parameters'.format(P.Label))
                else:
                    try:
                        P.Estimate()
                        print('\n...Estimating spline parameters for {}:'.format(P.Label))
                    except:
                        Warning('Could not properly estimate spline parameters for {}'.format(P.Label))
        
        return 
    

    def CreateOptimizer(self):

        print('\n========== Making optimizer ==========')

        if self.MDEngine == 'openmm':
            OptClass = sim.srel.optimizetrajomm.OptimizeTrajOpenMMClass
            if self.MinBondOrd:
                sim.export.omm.bondCutoff = self.MinBondOrd - 2
        elif self.MDEngine == 'lammps':
            OptClass = sim.srel.optimizetrajlammps.OptimizeTrajLammpsClass
        elif self.MDEngine == 'sim':
            OptClass = sim.srel.optimizetraj.OptimizeTrajClass

        self.Opt_dict = {}
        RefPos_dict = {}

        for entry in self.InitialPositions:
            RefPos = md.load(entry[1],top=entry[2])
            RefPos_dict[entry[0]] = RefPos.xyz[-1]

        for Opt_entry in self.Optimizers:
            Opt_name = Opt_entry[0]
            Sys_name = Opt_entry[1]
            traj_name = Opt_entry[2]
            Sys = self.Sys_dict[Sys_name]
            mapping = self.Mapping_dict[Sys_name]
            traj = self.Traj_dict[traj_name]

            if Opt_name in RefPos_dict:
                Sys.Pos = RefPos_dict[Opt_name]
                print('\n...Using alternative input starting postitions: \n{}'.format(Sys.Pos))
            else:
                traj_mapped = sim.traj.Mapped(traj, mapping, Sys=Sys)
                Pos = traj_mapped.ParseFrame(0)
                #Pos = traj.ParseFrame(0)
                Sys.Pos = Pos
                #print(Sys.BoxL)
                print('\n...Using mapped trajectory starting postitions: \n{}'.format(Sys.Pos))
        
            #if traj.FrameData.get("BoxL", None) is not None: UseTrajBoxL=True
            #else: UseTrajBoxL=False

            if self.Electrostatics and self.NeutralSrel: ElecSys = self.Sys_dict['ElecSys_{}'.format(Sys_name)]
            else: ElecSys = None
            Optimizer = OptClass(ModSys=Sys, Map=mapping, Traj=traj, 
                FilePrefix=Opt_entry[3], LoadArgData=True, Verbose=True, UseTrajBoxL=True, ElecSys=ElecSys)
            Optimizer.StepsEquil = self.StepsEquil
            Optimizer.StepsProd = self.StepsProd
            Optimizer.StepsStride = self.StepsStride
            Optimizer.HessianAdjustMode = self.HessianAdjustMode
            Optimizer.MinReweightFrac = self.MinReweightFrac
            if Opt_name in RefPos_dict:
                Optimizer.UserRefPos = RefPos_dict[Opt_name] 
            #Optimizer.UseHessian = False

            if self.Runs[0][1] != 3: Optimizer.ParseTarData() 

            if self.BondEstimate:
                self.EstimateBondParameters(Optimizer)
            if self.AngleEstimate:
                self.EstimateAngleParameters(Optimizer)
            if self.SplineEstimate:
                self.EstimateSplineParameters(Optimizer)

            self.Opt_dict[Opt_name] = Optimizer

            print('\nOptimizer Specifications:')
            print('File Prefix: {}'.format(Optimizer.FilePrefix))
            print('Model System: {}'.format(Optimizer.ModSys.Name))
            if Optimizer.ElecSys is not None: print('ElecSys: {}'.format(Optimizer.ElecSys.Name))
            print('NMol: {}'.format(Optimizer.ModSys.NMol))
            print('NAtom: {}'.format(Optimizer.ModSys.NAtom))
            print('NDOF: {}'.format(Optimizer.ModSys.NDOF))
            print('Verbose: {}'.format(Optimizer.Verbose))     

        return 


    def AddPenalties(self):

        print('\n========== Adding penalties ==========')

        if self.Penalties == []:
            print('\n... No pentalties specified')
            return 
        
        for entry in self.Penalties:
            Opt = self.Opt_dict[entry[0]]
            measure = [measure for measure in Opt.ModSys.Measures if measure.Name == entry[1]][0]
            target = entry[2]
            LagMult = entry[3]
            Opt.AddPenalty(measure, target, MeasureScale = 1., Coef=1.e-80)
            Opt.Penalties[-1].LagMult = LagMult
            self.Opt_dict[entry[0]] = Opt
            print('\n{}: added {} penalty with target value: {} and LagMult: {}'.format(Opt.FilePrefix, measure.Name, target, LagMult))

        return 

    
    def SaveHessian(self, Opt, Label):

        H = Opt.DDObj
        #H = [k[k != 0.0] for k in H]
        #H = np.asarray(H)
        #n = int(np.sqrt(len(H)))
        #H.reshape(n,n)
        Use = ~Opt.ConstrMask
        H = H[np.ix_(Use, Use)]
        print('\nHessian for {}:\n{}'.format(Opt.FilePrefix, H))
        np.savetxt('{}_{}_Hessian.dat'.format(Opt.FilePrefix,Label), H)

        return
    
    
    def Run(self):
                    
        for entry in self.Runs:
            Opts = [self.Opt_dict[name] for name in entry[0]]
            mode = entry[1]
            for Opt in Opts: Opt.UpdateMode = mode
            if len(Opts) == 1:
                Opt = Opts[0]
                Sys = Opt.ModSys

                if mode != 3: 
                    print('\n========== RUNNING SREL ==========')
                    if mode == 1:   
                        Opt.Run()
                    elif mode == 0 or mode == 2:
                        StageCoefs = entry[2]
                        Opt.RunStages(StageCoefs = StageCoefs, UseLagMult = True)

                    Opt.OutputPotentials(FilePrefix = 'final')
                    self.SaveHessian(Opt, '1')
                    if Opt.ElecSys is not None: 
                        ParamString = Opt.ElecSys.ForceField.ParamString()
                        paramfile = open(Opt.FilePrefix+'_ElecSys_ff.dat','w')
                        paramfile.write(ParamString)
                        paramfile.close()

                    print('\n========== RUNNING FINAL CGMD: {} =========='.format(Sys.Name))
                    Opt.ReweightOldModTraj = False
                    Opt.UpdateModTraj()
                    Opt.CalcObj()
                    self.SaveHessian(Opt,'2')

                elif mode == 3:
                    Opt.OutputPotentials(FilePrefix = Opt.FilePrefix)
                    if Opt.ElecSys is not None: 
                        ParamString = Opt.ElecSys.ForceField.ParamString()
                        paramfile = open(Opt.FilePrefix+'_ElecSys_ff.dat','w')
                        paramfile.write(ParamString)
                        paramfile.close()
                    print('\n========== RUNNING CGMD ==========')
                    Opt.MakeModTrajFunc(Opt, Sys, Opt.FilePrefix, Opt.StepsEquil, Opt.StepsProd, Opt.StepsStride, Verbose=True)

            elif len(Opts) > 1:   
                weights = entry[3]
                file_prefix = '_'.join([Opt.FilePrefix for Opt in Opts])
                OptMulti = sim.srel.optimizemultitraj.OptimizeMultiTrajClass(Opts ,Weights=weights,FilePrefix=file_prefix)
                
                if mode != 3: 
                    print('\n========== RUNNING SREL ==========')
                    OptMulti.UpdateMode = mode 
                    if mode == 1:   
                        OptMulti.Run()
                    elif mode == 0 or mode == 2:
                        StageCoefs = entry[2]
                        OptMulti.RunStages(StageCoefs = StageCoefs, UseLagMult = True)

                    OptMulti.OutputPotentials(FilePrefix = 'final')
                    self.SaveHessian(Opt, '1')
                    if OptMulti.OptimizeTrajList[0].ElecSys is not None: 
                        ParamString = OptMulti.OptimizeTrajList[0].ElecSys.ForceField.ParamString()
                        paramfile = open(Opt.FilePrefix+'_ElecSys_ff.dat','w')
                        paramfile.write(ParamString)
                        paramfile.close()

                    '''
                    for Opt in OptMulti.OptimizeTrajList:
                        print('\n========== RUNNING FINAL CGMD: {} =========='.format(Sys.Name))
                        Opt.ReweightOldModTraj = False
                        Opt.UpdateModTraj()
                        Opt.CalcObj()
                        SaveHessian(Opt,'2')
                    '''
                    
                elif mode == 3:
                    for Opt in Opts:
                        Sys = Opt.ModSys
                        Opt.OutputPotentials(FilePrefix = Opt.FilePrefix)
                        if Opt.ElecSys is not None: 
                            ParamString = Opt.ElecSys.ForceField.ParamString()
                            paramfile = open(Opt.FilePrefix+'_ElecSys_ff.dat','w')
                            paramfile.write(ParamString)
                            paramfile.close()
                        print('\n========== RUNNING CGMD: {} =========='.format(Sys.Name))
                        Opt.ModTraj = Opt.MakeModTrajFunc(Opt, Sys, Opt.FilePrefix, Opt.StepsEquil, Opt.StepsProd, Opt.StepsStride, Verbose=True)

        return
        

    #========== Execution ==========#


    def ExportSim(self):
        
        self.CreateSystem()
        self.SetupPotentials()
        self.AddMeasures()
        self.CompileSystems()
        self.CreateMapping()
        self.LoadTrajectories()
        self.CreateOptimizer()
        self.AddPenalties()
        self.Run()   

        return 
