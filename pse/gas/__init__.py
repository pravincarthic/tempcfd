"""Thermochemical libraries of SU2-NEMO for the AIR-7 mixture.

    library   SU2 option            class
    --------  --------------------  ----------------------------------------
    su2tclib  FLUID_MODEL=SU2_NONEQ  SU2TCLibAir7   (GAS_MODEL= AIR-7)
    mutationpp FLUID_MODEL=MUTATIONPP MutationppAir7 (GAS_MODEL= air_7)

Both use the species order  e-, N2, O2, NO, N, O, NO+.
"""
from .base import GasModel
from .su2tclib import SU2TCLibAir7
from .mutationpp import MutationppAir7


def make_gas(library="su2tclib", gas_model=None, transport="WILKE",
             frozen=False, data_dir=None, transport_fallback="su2tclib"):
    lib = (library or "su2tclib").lower()
    if lib in ("su2tclib", "su2_noneq", "su2"):
        gm = (gas_model or "AIR-7").upper()
        if gm != "AIR-7":
            raise ValueError("only GAS_MODEL= AIR-7 is implemented")
        return SU2TCLibAir7(transport=transport, frozen=frozen)
    if lib in ("mutationpp", "mpp", "mutation++"):
        return MutationppAir7(mixture=gas_model or "air_7", transport=transport,
                              frozen=frozen, data_dir=data_dir,
                              transport_fallback=transport_fallback)
    raise ValueError("unknown thermochemical library '%s'" % library)


__all__ = ["GasModel", "SU2TCLibAir7", "MutationppAir7", "make_gas"]
