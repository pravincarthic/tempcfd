"""The Mutation++ transport bridge, exercised with a stand-in for the
Mutation++ Python module (the real one needs a compiled Mutation++)."""
import sys
import types
import numpy as np

from pse.gas import SU2TCLibAir7


def _fake_module(ref):
    mod = types.ModuleType("mutationpp")

    class MixtureOptions:
        def __init__(self, name):
            self.name = name

        def setStateModel(self, s):
            self.state = s

        def setViscosityAlgorithm(self, a):
            self.visc = a

        def setThermalConductivityAlgorithm(self, a):
            self.cond = a

    class Mixture:
        def __init__(self, opt):
            self.opt = opt

        def nSpecies(self):
            return ref.ns

        def speciesName(self, i):
            return ref.names[i]

        def setState(self, rhos, T, var):
            self.rs = np.asarray(rhos, float)[:, None]
            self.T = np.array([T[0]])
            self.Tv = np.array([T[1]])

        def _tr(self):
            return ref.transport(self.rs, self.T, self.Tv)

        def viscosity(self):
            return float(self._tr()[0][0])

        def frozenThermalConductivityVector(self):
            t = self._tr()
            return np.array([t[1][0], t[2][0]])

        def averageDiffusionCoeffs(self):
            return np.array([float(d[0]) for d in self._tr()[3]])

    mod.MixtureOptions = MixtureOptions
    mod.Mixture = Mixture
    return mod


def test_taylor_transport_reproduces_library_derivatives(monkeypatch):
    ref = SU2TCLibAir7(transport="WILKE")
    monkeypatch.setitem(sys.modules, "mutationpp", _fake_module(ref))
    from pse.gas.mutationpp import MutationppAir7
    g = MutationppAir7(transport="WILKE")
    assert g._mpp is not None and g._fallback is None
    n = 4
    Y = np.maximum(g.mass_fraction_vector({"N2": 0.767, "O2": 0.233}), 1e-10)
    rs = (Y / Y.sum())[:, None] * np.linspace(5e-3, 2e-2, n)[None, :]
    T = np.linspace(400.0, 2000.0, n)
    Tv = 0.9 * T
    g.anchor(rs, T, Tv)
    mu0 = g.transport(rs, T, Tv)[0]
    assert np.allclose(mu0, ref.transport(rs, T, Tv)[0], rtol=1e-12)
    # derivative with respect to T and to rho_N2, by complex step through
    # the expansion versus the analytic library
    h = 1e-30
    for k, pert in (("T", None), ("rho", 1)):
        rsc = rs.astype(complex)
        Tc = T.astype(complex)
        if pert is None:
            Tc = Tc + 1j * h
        else:
            rsc[pert] = rsc[pert] + 1j * h
        a = np.imag(g.transport(rsc, Tc, Tv.astype(complex))[0]) / h
        b = np.imag(ref.transport(rsc, Tc, Tv.astype(complex))[0]) / h
        assert np.allclose(a, b, rtol=1e-5)
