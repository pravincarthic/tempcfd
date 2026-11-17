"""Physical constants.

Each thermochemical library uses its own constant set so that the stability
operator is built from exactly the numbers the base-flow solver used.
"""

# SU2 (Common/include/option_structure.hpp)
SU2_RU = 8.3144598                 # J/(mol K)
SU2_KB = 1.3806503e-23             # J/K
SU2_NA = 6.0221415e23              # 1/mol  (SU2 stores 6.0221415e26 per kmol)
SU2_E_CGS = 4.8032047e-10          # fundamental charge, statC
SU2_PI = 3.14159265358979323846

# Mutation++ (src/general/Constants.h)
MPP_NA = 6.0221415e23
MPP_KB = 1.3806503e-23
MPP_RU = MPP_NA * MPP_KB
MPP_HP = 6.626068e-34
MPP_C0 = 299792458.0
MPP_ONEATM = 101325.0

# general
R_UNIV = 8.31446261815324          # J/(mol K), atmosphere model only
P_ATM = 101325.0
G0 = 9.80665
R_EARTH = 6356766.0                # US1976 effective earth radius, m
M_AIR = 0.0289644                  # kg/mol
FT_TO_M = 0.3048

CS_H = 1.0e-30                     # complex-step increment
