import os, sys, copy
import subprocess
import time

import argparse as ap
import numpy as np
import mdtraj as md

import sim

def SaveHessian(Opt,label):

    H = Opt.DDObj
    #H = [k[k != 0.0] for k in H]
    #H = np.asarray(H)
    #n = int(np.sqrt(len(H)))
    #H.reshape(n,n)
    Use = ~Opt.ConstrMask
    H = H[np.ix_(Use, Use)]
    print('\nHessian for {}:\n{}'.format(Opt.FilePrefix, H))
    np.savetxt('{}_{}_Hessian.dat'.format(Opt.FilePrefix,label), H)
    return

def CreateSystem(Systems, AtomTypes, MolTypes, Bonds):

    print('\n========== Defining System ==========')
    atom_dict = {}
    mol_dict = {}
    Sys_dict = {}

    for entry in AtomTypes:
        atom_dict[entry[0]] = sim.chem.AtomType(entry[0], Mass = entry[1], Charge = entry[2])
    print('Defined atom types:')
    print(atom_dict)

    for entry in MolTypes:                  
        atomtypes = [atom_dict[aname] for aname in entry[1]]
        mol_dict[entry[0]] = sim.chem.MolType(entry[0], atomtypes)
    
    for entry in Bonds:
        for bond in entry[1]:
            mol_dict[entry[0]].Bond(bond[0],bond[1])
    print('Defined molecule types:')
    print(mol_dict)

    World = sim.chem.World([mol_dict[entry[0]] for entry in MolTypes], Dim = 3, Units = sim.units.DimensionlessUnits)
    
    for Sys_entry in Systems:
        Sys = sim.system.System(World, Name = Sys_entry[0])
        print('\nCreating system: {}'.format(Sys.Name))
        for molname, nmol in zip(Sys_entry[1], Sys_entry[2]):
            print('Adding {} {} to system'.format(nmol, molname))
            for n in range(nmol):
                Sys+= mol_dict[molname].New()
        if Sys_entry[3] is not None: Sys.BoxL = Sys_entry[3]
        Sys_dict[Sys_entry[0]] = Sys

        # ElecSys: system with Ewald on that is used to run MD, used to speed up when only optimizing non-charged interactions for speed up
        if  Electrostatics and NeutralSrel: 
            #ElecSys = copy.deepcopy(Sys) 
            #ElecSys.name = 'ElecSys_{}'.format(Sys.name)
            ElecSys = sim.system.System(World, Name = 'ElecSys_{}'.format(Sys.Name))
            print('\nCreating system: {}'.format(ElecSys.Name))
            for molname, nmol in zip(Sys_entry[1], Sys_entry[2]):
                print('Adding {} {} to system'.format(nmol, molname))
                for n in range(nmol):
                 ElecSys += mol_dict[molname].New()
            if Sys_entry[3] is not None: ElecSys.BoxL = Sys_entry[3]
            Sys_dict[ElecSys.Name] = ElecSys

    for Sys in Sys_dict.values():
        Sys.TempSet = Temp
        Sys.PresSet = Pressure
        Sys.BarostatOptions['tension'] = Tension
        Sys.PresAx = Axis
        Sys.Int.Method = Sys.Int.Methods.VVIntegrate
        Sys.Int.Method.Thermostat = Sys.Int.Method.ThermostatLangevin
        Sys.Int.Method.TimeStep = dt
        Sys.Int.Method.LangevinGamma = LangevinGamma
        
    return atom_dict, mol_dict, Sys_dict

def SetupPotentials(PBonds, PNonbonded, PElectrostatic, PExternal, atom_dict, mol_dict, Sys_dict):

    Potentials_dict = {}
    print('\n...Setting Bmin to {}'.format(Bmin))
    
    for Sys in Sys_dict.values():

        #bonded interactions
        for bond in PBonds:
            atomtypes = sim.atomselect.PolyFilter([atom_dict[bond[0][0]], atom_dict[bond[0][1]]], Bonded=True)
            label = 'bond_{}_{}'.format(bond[0][0], bond[0][1])
            Dist0 = bond[1][0]
            FConst = bond[1][1]
            P = sim.potential.Bond(Sys, Label=label, Filter=atomtypes, Dist0=Dist0, FConst=FConst)
            P.Dist0.Fixed = bond[2][0]
            P.FConst.Fixed = bond[2][1]
            Sys.ForceField.append(P)

        #nonbonded interactions
        for nonbonded in PNonbonded:
            atomtypes = sim.atomselect.PolyFilter([atom_dict[nonbonded[0][0]], atom_dict[nonbonded[0][1]]])
            label = 'ljg_{}_{}'.format(nonbonded[0][0], nonbonded[0][1])
            B =  nonbonded[1][0]
            Kappa = nonbonded[1][1]
            P = sim.potential.Gaussian(Sys, Label=label, Filter=atomtypes, Cut=rcut, B=B, Kappa=Kappa, Dist0=0.0, Shift=True)
            P.B.Fixed = nonbonded[2][0]
            P.Kappa.Fixed = nonbonded[2][1]
            P.Dist0.Fixed = True
            P.Param.Min = Bmin
            Sys.ForceField.append(P)

        #electrostatic potentials 
        if Electrostatics: #turn on ewald summation and smeared coulumb 
            if (NeutralSrel and 'ElecSys_' in Sys.Name) or NeutralSrel == False:
            
                P = sim.potential.Ewald(Sys, ExcludeBondOrd=0, Cut=rcut, Shift=Shift, Coef=Coef, FixedCoef=True, Label='ewald' )
                Sys.ForceField.append(P)

                for elec in PElectrostatic:
                    atomtypes = sim.atomselect.PolyFilter([atom_dict[elec[0][0]], atom_dict[elec[0][1]]])
                    label = 'smeared_corr_{}_{}'.format(elec[0][0], elec[0][1])
                    Coef=elec[1][0]
                    BornA=elec[1][1]
                    #print(Coef,BornA)
                    P = sim.potential.SmearedCoulombEwCorr(Sys, Label=label, Filter=atomtypes, Cut=rcut, Coef=Coef, BornA=BornA, Shift=Shift)
                    P.FixedCoef = elec[2][0]
                    P.FixedBornA = elec[2][1]
                    Sys.ForceField.append(P)
            
        #external potentials
        for external in PExternal:
            if external[0] == 'gaussian':
                atomtypes = sim.atomselect.PolyFilter([atom_dict[external[1][0]], atom_dict[external[1][1]]])
                label = 'ext_gauss_{}_{}'.format(external[1][0], external[1][1])
                B = external[2][0]
                Kappa = external[2][1]
                P = sim.potential.Gaussian(Sys, Label=label, Filter=atomtypes, Cut=rcut, B=B, Kappa=Kappa, Dist0=0.0, Shift=True)
                P.Param.Min = -100.
                P.B.Fixed = True
                P.Kappa.Fixed = True
                P.Dist0.Fixed = True
                Sys.ForceField.append(P)

            if external[0] == 'sinusoid':
                atomtypes = sim.atomselect.PolyFilter([atom_dict[external[1][0]]])
                label = 'ext_sin_{}'.format(external[1][0])
                UConst = external[2][0]
                NPeriods = external[2][1]
                PlaneAxis = external[2][2]
                PlaneLoc = external[2][3]
                P = sim.potential.ExternalSinusoid(Sys, Label=label, Filter=atomtypes, Fixed=True, UConst=UConst, NPeriods=NPeriods, PlaneAxis=PlaneAxis, PlaneLoc=PlaneLoc)
                Sys.ForceField.append(P)
                    
        #setup histograms 
        for P in Sys.ForceField:
            P.Arg.SetupHist(NBin = 10000, ReportNBin=100)
            Potentials_dict[P.Label] = P
        
        #read in forcefield field
        if ForceFieldFile and os.path.exists(ForceFieldFile):
            print('\n...Reading forcefield parameters from {} for {}'.format(ForceFieldFile, Sys.Name))
            with open(ForceFieldFile, 'r') as of: s = of.read()
            Sys.ForceField.SetParamString(s, CheckMissing = False)

    for Sys_name, Sys in Sys_dict.items():
        print('\nForcefield for {}: '.format(Sys_name))
        for P in Sys.ForceField:
            print(P.Label)

    return Sys_dict

def AddMeasures(Ree2, Ree, Area, Sys_dict):

    print('\n========== Adding measures ==========')
    
    if Ree2 == [] and Ree == [] and Area == []:
        print('\n... No measures specified')
        return Sys_dict

    for entry in Ree2:
        Sys = Sys_dict[entry[1]]
        mol_name = entry[2]
        site_IDs = entry[3]
        Sys_mol = [mol for mol in Sys.World if mol.Name == mol_name][0]
        site1, site2 = Sys.World.SiteTypes[Sys_mol[site_IDs[0]].SID], Sys.World.SiteTypes[Sys_mol[site_IDs[1]].SID]
        filter = sim.atomselect.PolyFilter(Filters=[site1, site2],Intra=True)
        dist_ree2 = sim.measure.distang.Distance2(Sys, Filter=filter, Name=entry[0])
        Sys.Measures.append(dist_ree2)
        print('{}: added Ree2 measure between sites {} and {}'.format(Sys.Name, site1, site2))

    for entry in Ree:
        Sys = Sys_dict[entry[1]]
        mol_name = entry[2]
        site_IDs = entry[3]
        Sys_mol = [mol for mol in Sys.World if mol.Name == mol_name][0]
        site1, site2 = Sys.World.SiteTypes[Sys_mol[site_IDs[0]].SID], Sys.World.SiteTypes[Sys_mol[site_IDs[1]].SID]
        filter = sim.atomselect.PolyFilter(Filters=[site1, site2],Intra=True)
        dist_ree = sim.measure.distang.Distance(Sys, Filter=filter, Name=entry[0])
        Sys.Measures.append(dist_ree)
        print('{}: added Ree measure between sites {} and {}'.format(Sys.Name, site1, site2))

    for entry in Area:
        Sys = Sys_dict[entry[1]]
        Az = sim.measure.Az(Sys,axis=entry[2])
        Sys.Measures.append(Az)
        print('\n{}: added area measure for axis {} '.format(Sys.Name, entry[2]))

    return Sys_dict

def CompileSystems(Sys_dict):

    print('\n========== Loading and locking (compiling) system ==========')
    for Sys in Sys_dict.values():
        Sys.Load()
        
    return Sys_dict

def CreateMapping(MolTypes, Systems, Optimizers, Trajectories, Sys_dict):
    
    print('\n========== Creating Atom Maps ==========')

    mapping_dict = {}
    mol_mapping = {}
    for entry in MolTypes:
        mol_mapping[entry[0]] = entry[2]

    system_traj_dict = {}
    trajfile_dict = {}
    for Opt_entry in Optimizers:
        system_traj_dict[Opt_entry[1]] = Opt_entry[2]
    for traj_entry in Trajectories:
        trajfile_dict[traj_entry[0]] = [traj_entry[1],traj_entry[2]]

    for Sys_entry in Systems:
        if COM==True:
            trajname = system_traj_dict[Sys_entry[0]]
            if not os.path.exists(trajfile_dict[trajname][1]): 
                print('\nWarning: {} topology file does not exits ... defaulting to centroid mapping'.format(trajfile_dict[trajname][1]))
                masses=None
            else:
                traj = md.load(trajfile_dict[trajname][0], top=trajfile_dict[trajname][1])
                masses = np.array([a.element.mass for a in traj.top.atoms])
                #print(masses)
                if (np.sum(masses)==np.nan or np.sum(masses)==0.0): 
                    print('\nWARNING: atom masses include nan or are all 0.0 ... defaulting to centroid mapping')
                    masses=None 
        else: masses = None 

        Sys = Sys_dict[Sys_entry[0]]
        mapping = sim.atommap.PosMap()
    
        # last check for 0.0/nan all-atom masses
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
                    print('\nWARNING: atom masses are not well defined for CG bead type {} ... defaulting to centroid mapping'.format(atom.Name))
                    masses=None
                    break 

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
        mapping_dict[Sys_entry[0]] = mapping
        
    return mapping_dict

def LoadTrajectories(Trajectories):
    
    traj_dict = {}
    for traj_entry in Trajectories:
        if traj_entry[3] == 'dcd':
            traj = sim.traj.dcd.DCD(TrjFile=traj_entry[1], TopFile=traj_entry[2], ConversionFactor=1)
        elif traj_entry[3] == 'pdb':
            traj = sim.traj.pdb.Pdb(PdbFile=traj_entry[1])
        elif traj_entry[3] == 'lammpstrj':
            traj = sim.traj.lammps.Lammps(TrjFile=traj_entry[1])
        elif traj_entry[3] == 'xyz':
            traj = sim.traj.xyz.XYZ(TrjFile=traj_entry[1])
        traj_dict[traj_entry[0]] = traj

    return traj_dict

def CreateOptimizer(Optimizers, InitialPositions, Sys_dict, traj_dict, mapping_dict):

    print('\n========== Making optimizer ==========')

    if MDEngine == 'openmm':
        OptClass = sim.srel.optimizetrajomm.OptimizeTrajOpenMMClass
    elif MDEngine == 'sim':
        OptClass = sim.srel.optimizetraj.OptimizeTrajClass

    Opt_dict = {}
    IniPos_dict = {}

    for entry in InitialPositions:
        IniPos = md.load(entry[1],top=entry[2])
        IniPos_dict[entry[0]] = IniPos.xyz[-1]

    for Opt_entry in Optimizers:
        Opt_name = Opt_entry[0]
        Sys_name = Opt_entry[1]
        traj_name = Opt_entry[2]
        Sys = Sys_dict[Sys_name]
        mapping = mapping_dict[Sys_name]
        traj = traj_dict[traj_name]

        if Opt_name in IniPos_dict:
            Sys.Pos = IniPos_dict[Opt_name]
            print('\nUsing alternative input starting postitions: \n{}'.format(Sys.Pos))
        else:
            traj_mapped = sim.traj.Mapped(traj, mapping, Sys=Sys)
            Pos = traj_mapped.ParseFrame(0)
            #Pos = traj.ParseFrame(0)
            Sys.Pos = Pos
            #print(Sys.BoxL)
            print('\nUsing mapped trajectory starting postitions: \n{}'.format(Sys.Pos))
       
        #if traj.FrameData.get("BoxL", None) is not None: UseTrajBoxL=True
        #else: UseTrajBoxL=False

        if Electrostatics and NeutralSrel: ElecSys = Sys_dict['ElecSys_{}'.format(Sys_name)]
        else: ElecSys = None
        Optimizer = OptClass(ModSys=Sys, Map=mapping, Traj=traj, 
            FilePrefix=Opt_entry[3], LoadArgData=True, Verbose=True, UseTrajBoxL=True, ElecSys=ElecSys)
        Optimizer.StepsEquil = StepsEquil
        Optimizer.StepsProd = StepsProd
        Optimizer.StepsStride = StepsStride
        Optimizer.HessianAdjustMode = HessianAdjustMode
        Optimizer.MinReweightFrac = MinReweightFrac
        #Optimizer.UseHessian = False

        if Runs[0][1] != 3: Optimizer.ParseTarData()   
        Opt_dict[Opt_name] = Optimizer

        print('\nOptimizer Specifications:')
        print('File Prefix: {}'.format(Optimizer.FilePrefix))
        print('Model System: {}'.format(Optimizer.ModSys.Name))
        if Optimizer.ElecSys is not None: print('ElecSys: {}'.format(Optimizer.ElecSys.Name))
        print('NMol: {}'.format(Optimizer.ModSys.NMol))
        print('NAtom: {}'.format(Optimizer.ModSys.NAtom))
        print('NDOF: {}'.format(Optimizer.ModSys.NDOF))
        print('Verbose: {}'.format(Optimizer.Verbose))     

    return Opt_dict

def AddPenalties(Penalties, Opt_dict):

    print('\n========== Adding penalties ==========')

    if Penalties == []:
        print('\n... No pentalties specified')
        return Opt_dict
    
    for entry in Penalties:
        Opt = Opt_dict[entry[0]]
        measure = [measure for measure in Opt.ModSys.Measures if measure.Name == entry[1]][0]
        target = entry[2]
        LagMult = entry[3]
        Opt.AddPenalty(measure, target, MeasureScale = 1., Coef=1.e-80)
        Opt.Penalties[-1].LagMult = LagMult
        Opt_dict[entry[0]] = Opt
        print('\n{}: added {} penalty with target value: {} and LagMult: {}'.format(Opt.FilePrefix, measure.Name, target, LagMult))

    return Opt_dict

def Run(Runs, Opt_dict):
                
    for entry in Runs:
        Opts = [Opt_dict[name] for name in entry[0]]
        mode = entry[1]
        for Opt in Opts: Opt.UpdateMode = mode

        if len(Opts) == 1:
            Opt = Opts[0]
            Sys = Opt.ModSys
            if BondEstimate:
                for P in Opt.ModSys.ForceField:
                    if isinstance(P,sim.potential.Bond):
                        P.Estimate()
                        print('\n...Estimating bonded parameters for {}:'.format(P.Label))
                        print(' Dist0: {}'.format(P.Dist0[0]))
                        print(' FConst: {}'.format(P.FConst[0]))

            if mode != 3: 
                print('\n========== RUNNING SREL ==========')
                if mode == 1:   
                    Opt.Run()
                elif mode == 0 or mode == 2:
                    StageCoefs = entry[2]
                    Opt.RunStages(StageCoefs = StageCoefs, UseLagMult = True)

                Opt.OutputPotentials(FilePrefix = 'final')
                SaveHessian(Opt, '1')
                if Opt.ElecSys is not None: 
                    ParamString = Opt.ElecSys.ForceField.ParamString()
                    paramfile = open(Opt.FilePrefix+'_ElecSys_ff.dat','w')
                    paramfile.write(ParamString)
                    paramfile.close()

                print('\n========== RUNNING FINAL CGMD: {} =========='.format(Sys.Name))
                Opt.ReweightOldModTraj = False
                Opt.UpdateModTraj()
                Opt.CalcObj()
                SaveHessian(Opt,'2')

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
            if BondEstimate:
                for Opt in Opts:   
                    for P in Opt.ModSys.ForceField:
                        if isinstance(P,sim.potential.Bond):
                            P.Estimate()
                            print('\n...Estimating bonded parameters for {}:'.format(P.Label))
                            print('     Dist0: {}'.format(P.Dist0))
                            print('     FConst: {}'.format(P.FConst))

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
                SaveHessian(Opt, '1')
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

def execute():
    
    atom_dict, mol_dict, Sys_dict = CreateSystem(Systems, AtomTypes, MolTypes, Bonds)
    Sys_dict = SetupPotentials(PBonds, PNonbonded, PElectrostatic, PExternal, atom_dict, mol_dict, Sys_dict)
    Sys_dict = AddMeasures(Ree2, Ree, Area, Sys_dict)
    Sys_dict = CompileSystems(Sys_dict)
    mapping_dict = CreateMapping(MolTypes, Systems, Optimizers, Trajectories, Sys_dict)
    traj_dict = LoadTrajectories(Trajectories)
    Opt_dict = CreateOptimizer(Optimizers, InitialPositions, Sys_dict, traj_dict, mapping_dict)
    Opt_dict = AddPenalties(Penalties, Opt_dict)
    Run(Runs, Opt_dict)   
    return 

#========== Script Run Protocal ==========#

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Simulation Parameters')
    parser.add_argument("paramfile", default='srel_run.py', type=str, help="srel_run.py file")
    cmdln_args = parser.parse_args()

    #initialize defaults for simulation parameters
    COM=True 
    ForceFieldFile=None
    asmear=0.5
    rcut=1.0
    Bmin=None
    Electrostatics=False
    NeutralSrel=False 
    Coef=None 
    BornA=None 
    Shift=True
    Temp=1.0
    Pressure=0.0 
    Tension=None,
    Axis=None 
    Tension=None 
    dt=0.1 
    LangevinGamma=1.0 
    MDEngine='openmm'
    StepsEquil=0 
    StepsProd=0 
    StepsStride=0
    HessianAdjustMode=0
    MinReweightFrac=0.1
    BondEstimate=False

    # initialize default empty arrays
    AtomTypes, MolTypes, Bonds = [], [], []
    PBonds, PNonbonded, PElectrostatic, PExternal = [], [], [], []
    Systems, Trajectories, Optimizers, InitialPositions, Runs = [], [], [], [], []
    Ree2, Ree, Area, Penalties = [], [], [], []

    # execute the parameter file code
    with open(cmdln_args.paramfile, 'r') as of:
        filecode = of.read()
    exec(filecode, None, globals())

    # main function call
    execute()