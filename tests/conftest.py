import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def gas():
    from pse.gas import make_gas
    return make_gas("su2tclib", transport="WILKE")


@pytest.fixture(scope="session")
def edge():
    from pse.atmosphere import freestream
    from pse.gasdynamics import taylor_maccoll
    fs = freestream(10.0, z_ft=130000.0)
    c = taylor_maccoll(10.0, np.radians(7.0))
    return dict(fs=fs, M_e=c["M_cone"], T_e=fs["T"] * c["T_ratio"],
                p_e=fs["p"] * c["p_ratio"])


@pytest.fixture(scope="session")
def bf_small(gas, edge):
    """Mach 10 / 130 kft, 7 deg cone, Tw = 1000 K; x in [0.6, 0.9] m where
    the 165 kHz second mode is amplified."""
    from pse import baseflow as bfm
    return bfm.similarity_baseflow(
        gas, np.linspace(0.6, 0.9, 7), edge["M_e"], edge["T_e"], edge["p_e"],
        {"N2": 0.767, "O2": 0.233}, Tw=1000.0, cone=True,
        theta_c=np.radians(7.0), ny=101)


@pytest.fixture(scope="session")
def model(gas):
    from pse import fluxes as fx
    return fx.Model(gas)
