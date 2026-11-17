"""Linear and nonlinear parabolised stability equations (LST, LPSE, NPSE)
for hypersonic boundary layers computed with SU2-NEMO
(NEMO_NAVIER_STOKES, two-temperature TTv model, AIR-7), using either SU2's
built-in thermochemistry (SU2TCLib) or Mutation++.  The marching scheme,
stabilisation, mixed-derivative terms and the NPSE harmonic balance follow
Krypton (Lacombe, SoftwareX SOFTX-D-22-00069).
"""
__version__ = "2.0.0"

from . import atmosphere, gas, fluxes, jacobians, grid, operators, bcs, lst
from . import pse, npse, baseflow, gasdynamics, parallel, io, config, su2

__all__ = ["atmosphere", "gas", "fluxes", "jacobians", "grid", "operators",
           "bcs", "lst", "pse", "npse", "baseflow", "gasdynamics", "parallel",
           "io", "config", "su2"]
