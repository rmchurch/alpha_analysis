"""
Library to prepare a run for the ASCOT code.

This library will contain all the functions to start from a DESC
input, generate all the required inputs for the ASCOT run (in the 
HDF5 file). The generation will be as follows:
- Generation of the magnetic field for ASCOT starting from DESC.
- Generation of the AFSI results for a given equilibrium, using
  either cylindrical or flux coordinates (the latter will be default).
- Generating the particle source in either guiding-center or full orbit.
- Generating diagnostics for the inputs to be stored.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import unyt
from typing import Union, List
import h5py

# This is to retrieve the git commit hash.
from git import Repo, InvalidGitRepositoryError

# The following are to time the results
import time
import datetime

# Importing the ASCOT interface and utilities.
import a5py
import a5py.ascot5io.coreio.tools as a5tools
from a5py.ascot5io.coreio.fileapi import INPUTGROUPS
from a5py.physlib import parseunits
from a5py.ascot5io.dist import DistData
from a5py.ascot5io.options import Opt

# Other utils.
from alpha_analysis import distrz2distrho, convert_flux_to_cylindrical

# Setting up the logger.
import logging
logger = logging.getLogger("StellaratorRun")

# Define custom formatter for colored output
class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "[%(asctime)s - %(funcName)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

# Configure the logger
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)
logger.setLevel(logging.INFO)

def _make_dummy_inputs(a5: a5py.Ascot):
    """
    Create the dummy inputs for ASCOT.

    Parameters
    ----------
    a5 : a5py.Ascot
        ASCOT instance where the dummy inputs will be created.
    """
    if not hasattr(a5.data, "efield"):
        a5.data.create_input("E_TC")
        # a5.data.create_input("E_TC", exyz=np.array([0,0,0]), activate=True, desc="Zero electric field")
        logger.info(" >> Added unused efield")

    if not hasattr(a5.data, "neutral"):
        a5.data.create_input("N0_1D")
        logger.info(" >> Added unused neutral model")

    if not hasattr(a5.data, "boozer"):
        a5.data.create_input("Boozer")
        logger.info(" >> Added unused Boozer")

    if not hasattr(a5.data, "mhd"):
        a5.data.create_input("MHD_STAT")
        logger.info(" >> Added unused MHD_STAT input")

    if not hasattr(a5.data, "asigma_loc"):
        a5.data.create_input("asigma_loc")
        logger.info(" >> Added unused asigma_loc input")

class RunItem:
    """
    Contains all the setup for a single ASCOT run.
    """
    def __init__(self, equ: Union[str, a5py.Ascot], 
                 path: Union[str, os.PathLike]=None,
                 suffix: str='',
                 create: bool=False, **kwargs):
        """
        Initializes the RunItem instance with a given equilibrium.

        The input equilibrium can either be a path or an ASCOT file instance.
        In the case it is a path, it can be either:
        - An ASCOT HDF5 file path.
        - A DESC input file path, in which case an ASCOT
          instance will be created from it. In this case, the optional
          variables nR, nZ, nPhi, ... are all used to generate the ASCOT
          equilibrium.

        Parameters
        ----------
        equ : Union[str, a5py.Ascot]
            Either the path to an ASCOT HDF5 file, or a DESC input file,
            or an already initialized ASCOT instance.
        path : Union[str, os.PathLike]
            Path where the ASCOT run files will be stored. When the create is 
            False and the input is already provided, this path is ignored.
        create : bool
            If this is the case, a new ASCOT will be created, by either
            importing all the inputs from the other ASCOT file or by 
            creating a new ASCOT instance from the DESC input file.
        **kwargs:   
            Additional keyword arguments to initialize the ASCOT instance
            in case equ is a DESC input file. See ASCOT templates for help.        
        """

        # We need to distinguish between the different types of input.
        if isinstance(equ, a5py.Ascot):
            a5src = equ # Already an ASCOT instance.
            fn = os.path.basename(a5src.file_getpath())
            create = False  # The file is just created below.
        elif isinstance(equ, str):
            with h5py.File(equ, 'r') as f:
                if 'bfield' in f: # This is an ASCOT file.
                    is_equ = 'ascot'
                    fn = os.path.basename(equ).split('.')[0] + suffix + '.h5'
                    a5src = a5py.Ascot(equ, create=False)
                else:
                    is_equ = 'desc'
                    create = False  # The file is just created below.
                    fn = os.path.basename(equ).split('.')[0] + suffix + '.h5'

                    # We create the ASCOT input.
                    self.a5fn = os.path.join(path, fn) if path is not None else fn
                    a5src = a5py.Ascot(self.a5fn, create=True)
                    nPhi = kwargs.get('nPhi', 200)
                    nR = kwargs.get('nR', 200)
                    nZ = kwargs.get('nZ', 200)
                    waitingbar = kwargs.get('waitingbar', True)
                    stellsym = kwargs.get('use_stell_sym', True)
                    L_radial = kwargs.get('L_radial', 10)
                    M_poloidal = kwargs.get('M_poloidal', 10)
                    fraction_T = kwargs.get('fraction_T', 0.5)
                    nrho = kwargs.get('nrho', 1024)
                    Zeff = kwargs.get('Zeff', 1.0)
                    include_wall = kwargs.get('include_wall', True)
                    
                    logger.info(f" >> Creating new ASCOT input from DESC file {equ}")
                    logger.info(f"    - nR = {nR}, nZ = {nZ}, nPhi = {nPhi}")
                    logger.info(f"    - Using stellarator symmetry: {stellsym}")
                    logger.info(f"    - Radial resolution= {L_radial}, Poloidal = {M_poloidal}")
                    logger.info(f"    - Tritium fraction = {fraction_T}, nrho = {nrho}")
                    logger.info(f"    - Zeff = {Zeff}")

                    a5src.data.create_input('desc field', fn=equ, nphi=nPhi, nr=nR, nz=nZ,
                                         waitingbar=waitingbar, L_radial=L_radial, 
                                         M_poloidal=M_poloidal,
                                         use_stell_sym=stellsym)
                    a5src.data.create_input('desc profiles', fn=equ, fraction_T=fraction_T, nrho=nrho, Zeff=Zeff)
                    if include_wall:
                        a5src.data.create_input("import_desc_lcfs_as_wall", fn=equ)
        
        # We now create the ASCOT input.
        if create:
            self.a5fn = os.path.join(path, fn) if path is not None else fn
            
            logger.info(f" >> Creating new ASCOT input by copying from {fn}")
            a5 = a5py.Ascot(self.a5fn, create=True)

            # We will now iterate over all the inputs in the source file,
            # but the markers or the options.
            for igroup in INPUTGROUPS:
                if igroup.lower() in ['marker', 'options']:
                    continue
                if igroup in a5src.data:
                    logger.info(f"    - Copying group {igroup}")
                    data = getattr(a5src.data, igroup).active.read()
                    getattr(a5.data, igroup).write_hdf5(data)
            logger.info(f" >> ASCOT input {self.a5fn} created.")
            self.a5 = a5
        else:
            self.a5 = a5src
            self.a5fn = self.a5.file_getpath()

        # Adding dummy inputs if they do not exist.
        _make_dummy_inputs(self.a5)

    def run_afsi(self, nsymm: int=None, mode: str='magnetic',
                 nR: int=101, nz: int=None,
                 nenergy: int=50, npitch: int=1, 
                 descfn: str=None,
                 nmc: int=1000, nthermal_vel: int=10):
        """
        Generate the distribution function using the AFSI solver.

        This allows to use both the magnetic or cylindrical coordinates 
        for the generation of the alpha distribution function. If the magnetic
        coordinates are used, `nR` is interpreted as the number of radial
        points in rho, while `nz` is ignored. 
        If cylindrical coordinates are used, `nR` and `nz` are the
        number of radial and vertical points, respectively. In this case, if
        `nz` is None, it is set equal to `nR + 1`.

        This will call the AFSI solver with the current equilibrium loaded, and 
        generate the distribution function for the alphas, along with several
        useful diagnostics.

        Recommended variables are already set by the default.

        Parameters
        ----------
        mode : str
            Mode for the AFSI grid. Can be either 'magnetic' or 'cylindrical'.
        nR : int
            Number of radial points. In magnetic mode, this is the number of
            points in rho. In cylindrical mode, this is the number of radial
            points in R.
        nz : int
            Number of vertical points in Z. Only used in cylindrical mode.
        nenergy : int
            Number of energy points in the AFSI grid.
        npitch : int
            Number of pitch points in the AFSI grid.
        nmc : int
            Number of Monte Carlo markers to use for the AFSI solver.
        nthermal_vel : int
            Number of the thermal velocities to use to expand the energy
            grid for the AFSI solver.
        """
        # Checking inputs.
        if mode.lower() not in ['magnetic', 'cylindrical']:
            raise ValueError(f" >> Unknown mode {mode} for AFSI grid.")
        if mode.lower() == 'cylindrical' and nz is None:
            nz = nR + 1
        if nR <= 1 or nenergy <= 1 or npitch <= 0:
            raise ValueError(f" >> Invalid grid parameters for AFSI grid.")
        if (nz is not None) and (nz <= 1):
            raise ValueError(f" >> Invalid nz parameter for AFSI grid.")
        if mode.lower() == 'magnetic' and descfn is None:
            raise ValueError(f" >> DESC input file must be provided for magnetic mode.")
        if mode.lower() == 'magnetic' and not os.path.isfile(descfn):
            raise ValueError(f" >> DESC input file {descfn} does not exist.")
        logger.info(f" >> Generating AFSI distribution function in {mode} mode.")

        # We will override the nsymm value by reading from the ascot file.
        self.a5.input_init(bfield=True)
        nsymm = self.a5._sim.B_data.BSTS.Nperiods * 2 # The 2 is the stellarator symmetry.

        # Computing the thermal velocity to set the energy grid.
        self.a5.input_init(plasma=True)
        pls = self.a5.data.plasma.active.read()['etemperature'].max() * unyt.eV
        mHe4 = 4.002602 * unyt.amu
        Ealpha = 3.54 * unyt.MeV
        vth_He4 = np.sqrt(2 * pls / mHe4).to('m/s')
        vmax = nthermal_vel * vth_He4
        Emax = Ealpha + 0.5 * mHe4 * vmax**2
        Emin = Ealpha - 0.5 * mHe4 * vmax**2

        # For the Helium4 species.
        ekin1 = np.linspace(Emin.to('eV').value, Emax.to('eV').value, nenergy) * unyt.eV
        pitch1 = np.linspace(-1.0, 1.0, npitch+1)  # Including endpoints.

        # For the neutrons species, we just set a dummy grid.
        ekin2 = np.array([13.0, 15.0]) * unyt.MeV
        pitch2 = np.array([-1.0, 1.0])

        # We now generate the spatial grid.
        if mode.lower() == 'magnetic':
            rho = np.linspace(1e-3, 0.99, nR)

            # We get the symmetry of the equilibrium.
            phimax = 360.0 * unyt.deg 
            
            self.a5.input_init(bfield=True, plasma=True)
            start_time = time.time()
            distHe, _ = self.a5.afsi.thermal_from_desc('DT_He4n', descfn=descfn, 
                                                        rho=rho, phimax=phimax,
                                                        ekin1=ekin1, pitch1=pitch1,
                                                        ekin2=ekin2, pitch2=pitch2,
                                                        nmc=nmc)
            end_time = time.time()
        else:
            self.a5.input_init(bfield=True)
            rmin = self.a5._sim.B_data.BSTS.B_r.x_min
            rmax = self.a5._sim.B_data.BSTS.B_r.x_max
            zmin = self.a5._sim.B_data.BSTS.B_z.z_min
            zmax = self.a5._sim.B_data.BSTS.B_z.z_max
            R = np.linspace(rmin, rmax, nR)
            Z = np.linspace(zmin, zmax, nz)
            phi = np.array([0.0, 360.0 / nsymm]) * unyt.deg
            spatial_grid = {'R': R, 'Z': Z, 'phi': phi}

            distHe, _ = self.a5.afsi.thermal('DT_He4n',
                                    r=np.linspace(rmin, rmax, nR),
                                    z=np.linspace(zmin, zmax, nz),
                                    phi=phi,
                                    ekin1=ekin1, pitch1=pitch1,
                                    ekin2=ekin2, pitch2=pitch2,
                                    nmc=nmc)
            
            # For consistency, we transform this distribution
            # from (R, z, E, pitch) to (rho, E, pitch)
            # TODO: How can we determine the phi angle to use here without the
            # symmetry info?
            distHe_rho = distrz2distrho(self.a5, distHe,  rhomin=1e-3, rhoout=0.99, 
                                        nrho=100, n_samples=1000, phi=2*np.pi/nsymm
                                        )
        
        # Building diagnostics.
        elapsed_time = end_time - start_time
        logger.info(f" >> AFSI run completed in {elapsed_time:.2f} s.")

        self.afsi_dist = distHe

        # Computing the marginal distributions in energy, pitch and rho. 
        distrho = distHe.integrate(True, theta=np.s_[:], phi=np.s_[:], xi=np.s_[:],
                                      ekin=np.s_[:], charge=np.s_[:], time=np.s_[:])
        distekin = distHe.integrate(True, rho=np.s_[:], theta=np.s_[:], phi=np.s_[:],
                                       xi=np.s_[:], charge=np.s_[:], time=np.s_[:])
        distxi = distHe.integrate(True, rho=np.s_[:], theta=np.s_[:], phi=np.s_[:],
                                     ekin=np.s_[:], charge=np.s_[:], time=np.s_[:])
        self.afsi_distrho = distrho
        self.afsi_distekin = distekin
        self.afsi_distxi = distxi

        # computing the integrals as a diagnostic for the user.
        distekin = distHe.integrate(True, rho=np.s_[:], theta=np.s_[:], phi=np.s_[:],
                             xi=np.s_[:], charge=np.s_[:])
        E = distekin._copy()
        E._multiply(distekin.abscissa('ekin'), 'ekin')
        N = E.integrate(True, ekin=np.s_[:])
        logger.info(f" >> Total number of alphas in the distribution: {N._distribution.to('MW').value} MW")
    
        return
    
    @parseunits(tmax='ms', rhomax='dimensionless', 
                min_energy='eV', strip=False)
    def prepare_markers(self, descfn: str, nmarkers: int, mode: str='gc', 
                        tmax: float=100 * unyt.ms,
                        rhomax: float=0.999 * unyt.dimensionless,
                        enable_collisions: bool=True,
                        afsi_weighting: bool=True, 
                        adaptive: bool=False, min_energy: float=None,
                        thermal_factor: float=None, enable_wall: bool=True,
                        flr_corrections: bool=True, adaptive_opts: dict=None,
                        **dist_config):
        """
        Setup the markers for the run and prepare the options for the
        simulation.

        Parameters
        ----------
        descfn : str
            Path to the DESC input file to use for the marker generation.
        nmarkers : int
            Number of markers to be used in the simulation.
        mode : str
            Marker mode. Can be either 'gc' for guiding-center or 'prt'
            for full-orbit.
        tmax : float
            Maximum simulation time for each marker.
        rhomax : float
            Maximum rho value for the initial position of the markers.
        enable_collisions : bool
            Whether to enable collisions for the markers.
        afsi_weighting : bool
            Whether to use AFSI weighting for the markers. When set to False,
            the code will launch only alpha particles at birth energy covering
            uniformly the spatial and pitch space.
        adaptive : bool
            Whether to use adaptive time stepping for the markers. Only available
            in guiding-center mode.
        min_energy : float
            Minimum energy for the end condition. If None, it is set to be 200 eV.
        thermal_factor : float
            Thermalization condition. Number of times the local thermal temperature
            to consider a particle thermalized. When not set, it is set to 2.
        enable_wall : bool
            Whether to enable wall collisions as an end condition.
        flr_corrections : bool
            Whether to enable FLR corrections for the markers.
        **dist_config:
            Configuration options for the distribution function generation
            in ASCOT file. Refer to documentation for details.
        """
        if mode.lower() not in ['gc', 'prt']:
            raise ValueError(f" >> Unknown marker mode {mode}.")
        if nmarkers <= 0:
            raise ValueError(f" >> Invalid number of markers {nmarkers}.")
        if tmax.value <= 0:
            raise ValueError(f" >> Invalid maximum time {tmax}.")
        if rhomax is not None:
            if rhomax <= 0:
                raise ValueError(f" >> Invalid maximum rho {rhomax}.")
        if not hasattr(self, 'afsi_dist') and afsi_weighting:
            raise ValueError(f" >> AFSI distribution not found. Cannot use AFSI weighting.")
        if descfn is None or not os.path.isfile(descfn):
            raise ValueError(f" >> DESC input file {descfn} does not exist.")
        if adaptive and mode.lower() != 'gc':
            raise ValueError(f" >> Adaptive time stepping only available in GC mode.")
        
        # We need to do our own preparation o the markers, as the 
        # (rho, theta, zeta) -> (R, Z, phi) transformation is better
        # implemented in DESC.
        logger.info(f" >> Preparing {nmarkers} markers in {mode} mode.")
        if afsi_weighting:
            markerdist = self.afsi_dist.integrate(True, charge=np.s_[:], time=np.s_[:])
            particledist = markerdist
        else:
            rho = np.linspace(1e-3, rhomax.value, 2) * unyt.dimensionless
            theta = np.linspace(0.0, 360.0, 2) * unyt.deg
            phi = np.linspace(0.0, 360.0, 2) * unyt.deg
            Ealpha = 3.54e6 * unyt.eV
            energy = np.linspace(Ealpha - 0.1*unyt.MeV, Ealpha + 0.1*unyt.MeV, 2)
            pitch = np.linspace(-1.0, 1.0, 2)
            tmp = np.zeros((2,2,2,2,2), dtype=np.float64)
            abscissae = {
                'rho': rho,
                'theta': theta,
                'phi': phi,
                'ekin': energy,
                'xi': pitch
            }
            particledist = DistData(tmp, **abscissae)
            markerdist = particledist
            logger.info(f" >> Generating uniform distribution for markers.")

        # Number of markers successfully generated
        ngen      = 0
        # Cell indices of generated markers
        icell     = np.zeros((nmarkers,), dtype="i8")
        
        _rng = np.random.default_rng()
        # Generate a number random for each marker, and when that marker is put
        # in the first cell where rand > threshold.
        threshold = np.append(0, np.cumsum(markerdist.histogram().ravel()))
        threshold /= threshold[-1]
        while ngen < nmarkers:
            if ngen == 0: rejected = np.s_[:]
            icell[rejected] = \
                np.digitize( _rng.random(nmarkers-ngen,), bins=threshold ) - 1

            # Each marker is given a weight that corresponds to number of
            # physical particles in that cell, divided by the number of markers
            # in that cell
            _, idx, counts = \
                np.unique(icell, return_inverse=True, return_counts=True)
            weight = particledist.histogram().ravel()[icell] / counts[idx]

            # Reject based on the minweight
            rejected = weight <= 0.0
            ngen = np.sum(~rejected)

        # Shuffle markers just in case the order they were created is biased
        idx = np.arange(nmarkers)
        _rng.shuffle(idx)
        icell  = icell[idx]
        weight = weight[idx].ravel()

        # Init marker species
        mrk = a5py.ascot5io.Marker.generate(mode, n=nmarkers)
        mrk["anum"][:]   = 4
        mrk["znum"][:]   = 2
        mrk["mass"][:]   = 4.002602 * unyt.amu
        mrk["charge"][:] = 2.0 * unyt.e
        mrk["weight"][:] = weight
        mrk["time"][:]   = 0.0 * unyt.s

        # Randomize initial 
        iic1, iic2, iic3, iip1, iip2 = \
            np.unravel_index(icell, markerdist.distribution().shape)
        list_indices = [iic1, iic2, iic3, iip1, iip2]
        def randomize(edges, idx):
            """Picks a random value between [edges[idx+1], edges[idx]]
            """
            return edges[idx] \
                + (edges[idx+1] - edges[idx]) * _rng.random(idx.size,)
        
        # So it may happen that the indices are not properly ordered: ic1 
        # may not correspond to the rho index. We can use the .abscissae 
        # to retrieve the correct edges.
        order = markerdist.abscissae
        idx = order.index('rho')
        ic1 = list_indices[idx]
        idx = order.index('theta')
        ic2 = list_indices[idx]
        idx = order.index('phi')
        ic3 = list_indices[idx]
        idx = order.index('ekin')
        ip1 = list_indices[idx]
        idx = order.index('xi')
        ip2 = list_indices[idx]

        rhos   = randomize(markerdist.abscissa_edges("rho"),   ic1)

        # We consider the magnetic angles to be randomized.
        thetas = _rng.random(nmarkers,) * 2.0 * np.pi
        zeta   = _rng.random(nmarkers,) * 2.0 * np.pi

        # We use DESC now to transform back to cylindrical coordinates.
        r, z, phi = convert_flux_to_cylindrical(descfn, rhos, thetas, zeta)
        mrk["r"][:]   = r
        mrk["z"][:]   = z
        mrk["phi"][:] = phi

        # We now generate the velocities.
        ekin = randomize(markerdist.abscissa_edges("ekin"), ip1)
        xi   = randomize(markerdist.abscissa_edges("xi"),   ip2)
        gyrophase = _rng.random(nmarkers,) * 2.0 * np.pi # Random gyrophase.

        if mode.lower() == 'gc':
            mrk["energy"][:]     = ekin
            mrk["pitch"][:]    = xi
            mrk["zeta"][:] = gyrophase
        else:
            # We need to transform the coordinates to particle coordinates,
            # assuming the distribution function in GC coordinates.
            # This is the 0th order transformation from GC to FO.
            if not 'bfield' in self.a5.input_initialized():
                self.a5.input_init(bfield=True)
            br, bphi, bz = self.a5.input_eval(
                mrk['r'], mrk['phi'], mrk['z'], mrk['time'],
                'br', 'bphi', 'bz')
            bhat = np.array([br, bphi, bz]) \
                / np.sqrt(br**2 + bphi**2 + bz**2).v
            e1 = np.zeros(bhat.shape)
            e1[2,:] = 1
            e2 = np.cross(bhat.T, e1.T).T
            e1 = e2 / np.sqrt(np.sum(e2**2, axis=0))
            e2 = np.cross(bhat.T, e1.T).T

            # We compute the parallel and perpendicular momenta from
            # the energy and pitch.
            pnorm = np.sqrt(2 * mrk['mass'] * ekin)
            ppa = xi * pnorm
            ppe = np.sqrt(1 - xi**2) * pnorm
            perphat = -np.sin(gyrophase)*e1-np.cos(gyrophase)*e2
            pvec = bhat * ppa + perphat * ppe
            mrk['vr']   = pvec[0,:] / mrk['mass']
            mrk['vphi'] = pvec[1,:] / mrk['mass']
            mrk['vz']   = pvec[2,:] / mrk['mass']

        # Let's write the particle data to the ASCOT file.
        self.mrk = mrk
        self.a5.data.create_input(mode, **mrk, activate=True)

        # Let's now generate the options.
        self.opts = Opt.get_default()
        if mode == 'gc':
            self.opts['SIM_MODE'] = 2  # Guiding-center
        else:
            self.opts['SIM_MODE'] = 1  # Full-orbit

        if enable_collisions:
            self.opts['ENABLE_COULOMB_COLLISIONS'] = 1 # Enable collisions.
        else:
            self.opts['ENABLE_COULOMB_COLLISIONS'] = 0 # Disable collisions.

        if adaptive:
            self.opts['ENABLE_ADAPTIVE'] = 1 # Enable/Disable adaptive method (1/0).

            # In this case we update the adaptive options.
            ada_opts_def = {
                'ADAPTIVE_TOL_ORBIT': 1e-8,
                'ADAPTIVE_MAX_DRHO': 0.1,
                'ADAPTIVE_MAX_DPHI': 0.1,
            }

            for key in ada_opts_def:
                # We only update known keys.
                if adaptive_opts is not None and key in adaptive_opts:
                    self.opts[key] = adaptive_opts[key]
                else:
                    self.opts[key] = ada_opts_def[key]
        
        else:
            self.opts['ENABLE_ADAPTIVE'] = 0

        if flr_corrections:
            self.opts['ENABLE_FLR_LOSSES'] = 1 # Enable FLR corrections.
        else:
            self.opts['ENABLE_FLR_LOSSES'] = 0 # Disable FLR corrections.

        self.opts['ENABLE_ORBIT_FOLLOWING'] = 1 # Enable orbit following.
        
        self.opts['ENABLE_MHD'] = 0 # Disable MHD.
        self.opts['ENABLE_DIST_5D'] = 0 # disable here.
        if hasattr(self.opts, '_OPT_ENABLE_RF'):
            self.opts['ENABLE_RF'] = 0 # Disable RF.

        # Time step options.
        if mode.lower() == 'prt' or adaptive:
            self.opts["FIXEDSTEP_USE_USERDEFINED"] = 0 # We let the code decide.
        else:
            self.opts["FIXEDSTEP_USE_USERDEFINED"] = 1 # We use a user-defined time step.
            
            # Let's guess the time step.
            dt = self.guess_time_step(self.a5, Ealpha, mode='gc',
                                      collisions=True)
            self.opts["FIXEDSTEP_USERDEFINED"] = dt.to('s').v

        # Final end conditions.
        self.opts['ENDCOND_SIMTIMELIM'] = 1 # Flag for the simulation to finish at the MAX_MILEAGE.
        if enable_wall:
            self.opts['ENDCOND_WALLHIT'] = 1 # Stop simulation when a particle hits the wall.
        else:
            self.opts['ENDCOND_WALLHIT'] = 0
        if rhomax is not None:
            self.opts['ENDCOND_RHOLIM']  = 1 # Stop the simulation with a given rho limit.
        else:
            self.opts['ENDCOND_RHOLIM'] = 0 # There is no limit on the rho.
        
        if enable_collisions:
            self.opts['ENDCOND_ENERGYLIM'] = 1 # Enable energy limit end condition.
        else:
            self.opts['ENDCOND_ENERGYLIM'] = 0 # Disable the energy limit when there is not 
                                               # collisions to avoid buggy behaviour.

        # Setting up the limits.
        if min_energy is None:
            min_energy = 200.0 * unyt.eV # eV
        if thermal_factor is None:
            thermal_factor = 2.0
        
        self.opts["ENDCOND_MIN_ENERGY"] = min_energy.to('eV').v # Energy in eV.
        self.opts["ENDCOND_MIN_THERMAL"] = thermal_factor # Multiplier for determining the thermal threshold.
        self.opts['ENDCOND_MAX_MILEAGE'] = tmax.to('s').v # Set the maximum simulation time.
        self.opts["ENDCOND_LIM_SIMTIME"] = tmax.to('s').v # Set the maximum simulation time.
        
        if rhomax is None:
            self.opts['ENDCOND_MAX_RHO'] = 100.0
        else:
            self.opts['ENDCOND_MAX_RHO'] = rhomax # Separatrix.

        # Orbit writing options.
        self.opts['ENABLE_ORBITWRITE'] = 0
        self.opts['ORBITWRITE_NPOINT'] = 1 # How many points to write for the orbit.

        # We now have to set up the distribution saving.
        for key in dist_config:
            if key not in self.opts:
                logger.warning(f" >> Unknown distribution config option {key}. Skipping.")
                continue
            if 'DIST' not in key:
                logger.warning(f" >> Distribution config option {key} does not contain 'DIST'. Skipping.")
                continue
            self.opts[key] = dist_config[key]

        # Let's do some safety memory checks.
        total_dist_memory = 0.0
        ndists_on = 0
        if self.opts['ENABLE_DIST_5D']:
            n = self.opts['DIST_NBIN_R'] * self.opts['DIST_NBIN_Z'] * \
            self.opts['DIST_NBIN_PPA'] * self.opts['DIST_NBIN_PPE'] * \
            self.opts['DIST_NBIN_PHI'] * self.opts['DIST_NBIN_CHARGE'] * \
            self.opts['DIST_NBIN_TIME'] * 8
            total_dist_memory += n
            ndists_on += 1
        if self.opts['ENABLE_DIST_6D']:
            n = self.opts['DIST_NBIN_R'] * self.opts['DIST_NBIN_Z'] * self.opts['DIST_NBIN_PHI'] * \
            self.opts['DIST_NBIN_PR'] * self.opts['DIST_NBIN_PPHI'] * \
            self.opts['DIST_NBIN_PZ'] * self.opts['DIST_NBIN_CHARGE'] * \
            self.opts['DIST_NBIN_TIME'] * 8
            total_dist_memory += n
            ndists_on += 1
        if self.opts['ENABLE_DIST_RHO5D']:
            n = self.opts['DIST_NBIN_RHO'] * self.opts['DIST_NBIN_THETA'] * \
            self.opts['DIST_NBIN_PPA'] * self.opts['DIST_NBIN_PPE'] * \
            self.opts['DIST_NBIN_PHI'] * self.opts['DIST_NBIN_CHARGE'] * \
            self.opts['DIST_NBIN_TIME'] * 8
            total_dist_memory += n
            ndists_on += 1
        if self.opts['ENABLE_DIST_RHO6D']:
            n = self.opts['DIST_NBIN_RHO'] * self.opts['DIST_NBIN_THETA'] * \
            self.opts['DIST_NBIN_PHI'] * self.opts['DIST_NBIN_PR'] * \
            self.opts['DIST_NBIN_PPHI'] * self.opts['DIST_NBIN_PZ'] * \
            self.opts['DIST_NBIN_CHARGE'] * self.opts['DIST_NBIN_TIME'] * 8
            total_dist_memory += n
            ndists_on += 1
        if self.opts['ENABLE_DIST_COM']:
            n = self.opts['DIST_NBIN_EKIN'] * self.opts['DIST_NBIN_PTOR'] * \
            self.opts['DIST_NBIN_MU'] * \
            self.opts['DIST_NBIN_CHARGE'] * self.opts['DIST_NBIN_TIME'] * 8
            total_dist_memory += n
            ndists_on += 1

        # Transforming to GB.
        total_dist_memory /= (1024.0**3)
        
        logger.info(f" >> Distribution functions enabled: {ndists_on}, total memory required: {total_dist_memory:.2f} GB.")

        # Writing the options.
        self.a5.data.create_input('opt', **self.opts, activate=True)
        logger.info(f" >> Marker and option preparations completed.")
        return

    def set_wall(self, fn: str, wall_type: str='desc'):
        """
        Adds a wall to the ASCOT input and switch the wall collision
        condition on.

        Parameters
        ----------
        fn : str
            Path to the wall input file.
        type : str
            Type of wall input. Can be either 'desc', 'eqdsk' or 'stl'.
        """
        if wall_type.lower() not in ['desc', 'eqdsk', 'stl']:
            raise ValueError(f" >> Unknown wall type {wall_type}.")
        if not os.path.isfile(fn):
            raise ValueError(f" >> Wall input file {fn} does not exist.")
        logger.info(f" >> Setting wall of type {wall_type} from file {fn}.")
        if wall_type.lower() == 'desc':
            self.a5.data.create_input("import_desc_lcfs_as_wall", fn=fn, activate=True)
        elif wall_type.lower() == 'eqdsk':
            self.a5.data.create_input("wall eqdsk", fn=fn, activate=True)
        elif wall_type.lower() == 'stl':
            raise NotImplementedError(" >> STL wall input not yet implemented.")

        # We read the options.
        qid = self.a5.data.options.active.get_qid()

        # We open the file using directly h5py to modify the wall option.
        with h5py.File(self.a5.file_getpath(), 'a') as f:
            opt_group = f['wall']['qid_'+str(qid)]
            opt_group['ENDCOND_WALLHIT'] = 1.0  # Enable wall hit condition.
        
        logger.info(f" >> Wall set and wall hit condition enabled.")
        return
    
    @parseunits(Eref='eV', mass='amu', charge='e', strip=False)
    @staticmethod
    def guess_time_step(a5, Eref: float, mode: str, collisions: bool=True,
                        steps: int=100, mass: float=4.002602 * unyt.amu,
                        charge: float=2.0 * unyt.e):
        """
        Makes an estimate of the simulation time for the particles.

        This may be particularly useful for the fixed-time step in 
        the guiding-center simulations, as ASCOT makes a horrendous 
        estimation. 

        There are a two level estimate:
        - The transit time, which is only computed to order of magnitude,
        by using the total velocity and the major radius of box.
        - The collision time, which is estimated from the slowing-down
        time formula for fast ions in a plasma.

        Parameters
        ----------
        Eref : float
            Reference energy to use for the time step estimation.
        mode : str
            Marker mode. Can be either 'gc' or 'prt'.
        collisions : bool
            Whether to consider collisions in the time step estimation.
        steps : int
            Number of time steps to divide the total estimated time.
        mass : float
            Mass of the particle species to consider. Defaults to He-4 mass.
        """
        if mode.lower() not in ['gc', 'prt']:
            raise ValueError(f" >> Unknown mode {mode} for time step estimation.")
        if Eref <= 0.0:
            raise ValueError(f" >> Invalid reference energy {Eref} for time step estimation.")
        
        # We distinguis now between GC and FO modes.
        a5.input_init(bfield=True)
        # Getting the magnetic axis position at phi = 0 slice.
        r = 0.5 * (a5._sim.B_data.BSTS.B_r.x_min + a5._sim.B_data.BSTS.B_r.x_max)
        z = 0.5 * (a5._sim.B_data.BSTS.B_z.z_min + a5._sim.B_data.BSTS.B_z.z_max)
        data = a5.input_eval(r*unyt.m, 0.0*unyt.rad, z*unyt.m, 0.0*unyt.s,
                             'axisr', 'axisz', grid=False)
        r_axis = data[0]
        z_axis = data[1]
        vtot = np.sqrt(2 * Eref/ mass).to('m/s')

        if mode.lower() == 'gc':
            transit_time = (2.0 * np.pi * r_axis / vtot).to('ms')
        else:
            # We get the absolute magnetic field at the magnetic axis
            # on the phi = 0 slice.
            b_axis = a5.input_eval(r_axis, 0.0 * unyt.rad, z_axis, 0.0 * unyt.s,
                                        'bnorm')
            omega_c = (np.abs(mass / (charge * b_axis))).to('s')
            transit_time = (2.0 * np.pi / omega_c).to('ms')
        
        # We now estimate the collision time if needed.
        if collisions:
            a5.input_init(plasma=True)

            # Get the collisional time scale.
            nu, pdrag = a5.input_eval_collcoefs(mass, charge,
                                            r_axis, 0*unyt.rad, z_axis, 0*unyt.s,
                                            vtot, 'nu', 'k', grid=False)

            dt_nu = np.min(np.abs(1 / (100*nu)))
            dt_pdrag = np.min(np.abs(vtot / (100 * pdrag)))

            collision_time = min(dt_nu, dt_pdrag).to('ms')
        else:
            collision_time = 1e6 * unyt.s # Very large time.
        
        total_time = min(transit_time, collision_time)
        dt = (total_time / steps).to('ms')

        logger.info(f" >> Estimated total time scale: {dt.to('us')}.")
        return dt

    def copy(self, fnout: str, copy_markers: bool=False):
        """
        Creates a clone of the current ASCOT input file, with optionally
        the same markers and options.

        This is useful to generate different runs with different number of
        markers or changing the options.

        Parameters
        ----------
        fnout : str
            Path to the output ASCOT HDF5 file.
        copy_markers : bool
            Whether to copy the markers and options from the current file.
        """
        # Let's create the new ASCOT file.
        logger.info(f" >> Creating a copy of the current ASCOT file to {fnout}.")
        duplicate_run_with_new_options(self.a5.file_getpath(), fnout,
                                       n=None,
                                       not_to_clone=['marker', 'options'] if not copy_markers else None)

        # We now create the new RunItem
        new_run = RunItem(fnout)
        new_run.afsi_dist = self.afsi_dist
        new_run.afsi_distrho = self.afsi_distrho
        new_run.afsi_distekin = self.afsi_distekin
        new_run.afsi_distxi = self.afsi_distxi

        return new_run

def duplicate_run_with_new_options(pathin: str, pathout: str, 
                                   n: int=None, not_to_clone: str=None,
                                   **opts):
    """
    Duplicate a run previously prepared, but with new options.

    Parameters
    ----------
    pathin : str
        Path to the input ASCOT HDF5 file.
    pathout : str
        Path to the output ASCOT HDF5 file.
    not_to_clone : list, str
        List of input groups not to clone from the input file. If a string is provided,
        it will be converted to a single-element list.
    n: int
        New number of markers. If provided, it will override the number
        of markers in the input file choosing them at random. Otherwise
        all the markers from the input file are copied.
    **opts:
        New options to set in the output file.
    """
    if not os.path.isfile(pathin):
        raise ValueError(f" >> Input ASCOT file {pathin} does not exist.")
    logger.info(f" >> Duplicating ASCOT run from {pathin} to {pathout} with new options.")
    a5src = a5py.Ascot(pathin, create=False)
    a5 = a5py.Ascot(pathout, create=True)

    if not_to_clone is None:
        not_to_clone = []
    elif isinstance(not_to_clone, str):
        not_to_clone = [not_to_clone]
    
    groups2copy = [g for g in INPUTGROUPS if g not in not_to_clone]

    # We will now iterate over all the inputs in the source file.
    for igroup in groups2copy:
        if igroup.lower() == 'options':
            continue
        if igroup in a5src.data:
            logger.info(f"    - Copying group {igroup}")
            data = getattr(a5src.data, igroup).active.read()
            itype = getattr(a5src.data, igroup).active.get_type() 
            if igroup.lower() == 'marker' and (n is not None):
                ntot = data['n']
                if n > ntot:
                    n = ntot # We override n to be ntot.
                    logger.warning(f" >> Requested number of markers {n} exceeds total markers {ntot}. Using {ntot} instead.")
                idx = _rng.choice(ntot, size=n, replace=False)
                mrk = {}
                for key in data:
                    if key == 'n':
                        mrk[key] = n
                        continue
                    mrk[key] = data[key][idx]
                data = mrk # We override the data to write.
                logger.info(f" >> Selected {n} markers out of {ntot} from the input file.")
                breakpoint()
            a5.data.create_input(itype, **data)
    
    if 'options' in groups2copy:
        logger.info(f" >> Writing new options.")
        a5opts = a5src.data.options.active.read()
        for key in opts:
            if key not in a5opts:
                logger.warning(f" >> Unknown option {key}. Skipping.")
                continue
            a5opts[key] = opts[key]
        a5.data.create_input('opt', **a5opts)
    logger.info(f" >> ASCOT run {pathout} created.")

    return


        
        
