"""
This script will generate a quick runnable ASCOT file that generates the distribution
function to verify the workflow in the ResultItem computing the pressure profile.
"""

import alpha_analysis as aa
import numpy as np
import matplotlib.pyplot as plt
import unyt
import os
plt.ion()

# %% Configuration.
equfn = "/scratch/gpfs/rmc2/m5300/results/G1600/G1600_00000/desc_equilibrium.h5"
workpath = "/scratch/gpfs/rmc2/m5300/results/G1600/G1600_00000/initial_pressure/"
waitingbar = True
use_stell_sym = True
L_radial = 4
M_poloidal = 4
nPhi = 100
nR = 100
nZ = 100
Nmarkers=500000

mass = 4.001506179127 * unyt.amu
Ealpha = 3.54 * unyt.MeV
Tmax = 10 * unyt.keV

Emax = Ealpha + 10.0 * 10*unyt.keV  # Max limit for the energy.
pmax = np.sqrt(2.0 * Emax * mass)
nperiods = 2

distopts = dict(ENABLE_DIST_RHO5D=1,
                DIST_MIN_RHO=0.0, DIST_MAX_RHO=1.0, DIST_NBIN_RHO=20,
                DIST_MIN_THETA=0.0, DIST_MAX_THETA=360, DIST_NBIN_THETA=21,
                DIST_MIN_PHI=0.0, DIST_MAX_PHI=360/nperiods, DIST_NBIN_PHI=12,
                DIST_MIN_PPA=-pmax.to('kg*m/s').value, DIST_MAX_PPA=pmax.to('kg*m/s').value,
                DIST_MIN_PPE=0.0, DIST_MAX_PPE=pmax.to('kg*m/s').value,
                DIST_NBIN_PPE=21, DIST_NBIN_PPA=22,
                DIST_MIN_TIME=0.0, DIST_MAX_TIME=5e-3, DIST_NBIN_TIME=9,
                DIST_MIN_CHARGE=0.0, DIST_MAX_CHARGE=2.5, DIST_NBIN_CHARGE=1
                )

# %% Create the ASCOT input file.
run = aa.RunItem(equfn, path=workpath, waitingbar=waitingbar, use_stell_sym=use_stell_sym,
                 L_radial=L_radial, M_poloidal=M_poloidal, nPhi=nPhi, nR=nR, nZ=nZ,
                 create=True)

run.run_afsi(descfn=equfn, nmc=100_000)

# The distribution for the source term is in
# run.afsi_dist

# run.prepare_markers(descfn=equfn, nmarkers=Nmarkers, tmax=20*unyt.ms, enable_collisions=True,
#                     afsi_weighting=True, adaptive=True, **distopts)

# # Now you run with ASCOT

# # %% Analysis part.
# result = aa.ResultItem(filepath=os.path.join(workpath, 'desc_equilibrium.h5'),
#                        descfn = equfn)

# losses = result.load_losses()
# profiles = result.make_profiles()
