"""11-species air data: Park (1990/1993) two-temperature model constants.

Species order is fixed and used throughout the solver.
Electronic level sets for the ionic species are truncated to the levels
tabulated by Park; they are dynamically irrelevant at Mach 10 / 39.6 km
where the ionised fraction is < 1e-8.
"""
import numpy as np
from .constants import R_UNIV

NAMES = ["N2", "O2", "NO", "N", "O", "N2+", "O2+", "NO+", "N+", "O+", "e-"]
NS = len(NAMES)
IDX = {n: i for i, n in enumerate(NAMES)}

# molar mass [kg/mol]
M_E = 5.48579909e-7                    # electron molar mass [kg/mol]
MW = np.array([0.0280134, 0.0319988, 0.0300061, 0.0140067, 0.0159994,
               0.0280134 - M_E, 0.0319988 - M_E, 0.0300061 - M_E,
               0.0140067 - M_E, 0.0159994 - M_E, M_E])
RS = R_UNIV / MW                       # species gas constant [J/(kg K)]

CHARGE = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1], dtype=float)
IS_MOLECULE = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
IS_ELECTRON = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
IS_HEAVY = ~IS_ELECTRON

# heat of formation at 0 K [J/kg]
HF0 = np.array([0.0, 0.0, 2.996123e6, 3.362161e7, 1.542000e7,
                5.375234e7, 3.637808e7, 3.283480e7, 1.340309e8, 9.756308e7, 0.0])

# characteristic vibrational temperature [K] (0 for non-molecules)
THETA_V = np.array([3395.0, 2239.0, 2817.0, 0.0, 0.0,
                    3176.0, 2741.0, 3421.0, 0.0, 0.0, 0.0])

# characteristic rotational temperature is never needed explicitly: all
# molecules here are diatomic and fully excited (T >> theta_r), so
# e_rot = R_s T.  sigma is the symmetry number used in q_rot.
SIGMA_ROT = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
THETA_R = np.array([2.8790, 2.0690, 2.4520, 0.0, 0.0,
                    2.8850, 2.0800, 2.7350, 0.0, 0.0, 0.0])

# electronic levels: (degeneracies, characteristic temperatures [K])
ELEC = {
    "N2": ([1, 3, 6, 6, 3, 1, 2, 2, 5, 1, 6, 6, 10, 6],
           [0.0, 7.2231565e4, 8.5778626e4, 8.6050777e4, 9.5351186e4, 9.8056977e4,
            9.9682444e4, 1.0489765e5, 1.1164896e5, 1.2258365e5, 1.2488569e5,
            1.2824762e5, 1.3380609e5, 1.4042964e5]),
    "O2": ([3, 2, 1, 1, 6, 3, 3],
           [0.0, 1.1391560e4, 1.8984739e4, 4.7559736e4, 4.9912421e4,
            5.0922686e4, 7.1898633e4]),
    "NO": ([4, 8, 2, 4, 4, 4, 4, 2, 4, 2, 4, 4, 2, 2, 2, 4],
           [0.0, 5.4673458e4, 6.3171396e4, 6.5188167e4, 6.9058552e4, 7.0499985e4,
            7.4003188e4, 7.6288694e4, 8.6761885e4, 8.7138349e4, 8.8860771e4,
            8.9909400e4, 9.0754797e4, 9.1214406e4, 9.2410925e4, 9.2529735e4]),
    "N":  ([4, 10, 6], [0.0, 2.7664696e4, 4.1493093e4]),
    "O":  ([5, 3, 1, 5, 1], [0.0, 2.2770000e2, 3.2610000e2, 2.2830000e4, 4.8620000e4]),
    "N2+": ([2, 4, 2, 4, 8, 8, 4, 4, 4, 4],
            [0.0, 1.3189972e4, 3.6633231e4, 3.6688768e4, 5.9315082e4, 5.9364200e4,
             6.6178000e4, 7.5655000e4, 7.8582000e4, 8.9000000e4]),
    "O2+": ([4, 8, 4, 6, 4, 2, 4],
            [0.0, 4.7354408e4, 5.8373987e4, 5.8500878e4, 6.2298663e4,
             6.7334679e4, 7.1219818e4]),
    "NO+": ([1, 3, 6, 6, 3, 1, 2, 2],
            [0.0, 7.5089678e4, 8.5254624e4, 8.9035726e4, 9.7469845e4,
             1.0005530e5, 1.0280093e5, 1.0571863e5]),
    "N+": ([1, 3, 5, 5, 1, 5, 15],
           [0.0, 7.0068352e1, 1.8819180e2, 2.2036569e4, 4.7031835e4,
            6.7312522e4, 1.3271908e5]),
    "O+": ([4, 10, 6], [0.0, 3.8583500e4, 5.8223500e4]),
    "e-": ([1], [0.0]),
}
G_EL = [np.array(ELEC[n][0], dtype=float) for n in NAMES]
T_EL = [np.array(ELEC[n][1], dtype=float) for n in NAMES]

# Blottner viscosity curve fit: mu = 0.1 exp((A lnT + B) lnT + C)  [Pa s]
BLOTTNER = np.array([
    [0.0268142, 0.3177838, -11.3155513],   # N2
    [0.0449290, -0.0826158, -9.2019475],   # O2
    [0.0436378, -0.0335511, -9.5767430],   # NO
    [0.0115572, 0.6031679, -12.4327495],   # N
    [0.0203144, 0.4294404, -11.6031403],   # O
    [0.0268142, 0.3177838, -11.3155513],   # N2+
    [0.0449290, -0.0826158, -9.2019475],   # O2+
    [0.0436378, -0.0335511, -9.5767430],   # NO+
    [0.0115572, 0.6031679, -12.4327495],   # N+
    [0.0203144, 0.4294404, -11.6031403],   # O+
    [0.0000000, 0.0000000, -12.6000000],   # e-
])

# first ionisation energy [J/kg of the produced ion] used for the electron
# energy sink of electron-impact ionisation
ION_ENERGY_EV = {"N": 14.534, "O": 13.618}

# Millikan-White constants use mu_sr (reduced molar mass, g/mol) and theta_v.
# Park limiting cross section for the high-temperature correction [m2]:
PARK_SIGMA_V = 1.0e-21
PARK_SIGMA_TREF = 50000.0


def mass_fraction_vector(Y):
    """Accept a dict {name: Y} or an array; return a normalised (ns,) array."""
    if isinstance(Y, dict):
        v = np.zeros(NS)
        for k, val in Y.items():
            if k not in IDX:
                raise KeyError("unknown species '%s'" % k)
            v[IDX[k]] = float(val)
    else:
        v = np.asarray(Y, dtype=float).copy()
        if v.size != NS:
            raise ValueError("expected %d mass fractions" % NS)
    v = np.maximum(v, 0.0)
    tot = v.sum()
    if tot <= 0.0:
        raise ValueError("zero mass fractions")
    return v / tot
