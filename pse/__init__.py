"""Parabolised Stability Equations for second Mack-mode instabilities in
11-species (Park 1993) two-temperature air, driven by OpenFOAM v1706 /
hy2Foam solutions.
"""
__version__ = "1.0.0"

from . import atmosphere, species, thermo, transport, chemistry
from . import fluxes, jacobians, grid, operators, bcs, lst, pse, baseflow
from . import gasdynamics, parallel, io, config

__all__ = ["atmosphere", "species", "thermo", "transport", "chemistry",
           "fluxes", "jacobians", "grid", "operators", "bcs", "lst", "pse",
           "baseflow", "gasdynamics", "parallel", "io", "config"]
