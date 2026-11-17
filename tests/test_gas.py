import numpy as np
import pytest

from pse.gas import make_gas, SU2TCLibAir7
from pse.gas import su2tclib_data as D


def _air(g, rho=1e-2, n=3, floor=1e-12):
    Y = g.mass_fraction_vector({"N2": 0.767, "O2": 0.233})
    Y = np.maximum(Y, floor)
    Y /= Y.sum()
    return Y[:, None] * rho * np.ones((1, n))


def test_species_order_is_su2_air7(gas):
    assert gas.names == ["e-", "N2", "O2", "NO", "N", "O", "NO+"]
    assert gas.nv == 12
    assert gas.i_e == 0


def test_pressure_gas_constant_and_sound_speed(gas):
    T = np.array([250.0, 1000.0, 3000.0])
    rs = _air(gas)
    p = gas.pressure(rs, T, T)
    R = p / (rs.sum(axis=0) * T)
    assert np.allclose(R, 288.2, rtol=2e-3)
    a = gas.sound_speed(rs, T, T)
    assert abs(a[0] / np.sqrt(1.4 * R[0] * 250.0) - 1.0) < 1e-3


def test_energy_inversion_roundtrip(gas):
    T = np.array([300.0, 1500.0, 4000.0])
    Tv = np.array([280.0, 1200.0, 3500.0])
    rs = _air(gas)
    e, ev = gas.energies(rs, T, Tv)
    T2, Tv2 = gas.temperatures(rs, e, ev)
    assert np.allclose(T2, T, rtol=1e-10)
    assert np.allclose(Tv2, Tv, rtol=1e-10)


def test_vibrational_energy_limits(gas):
    s = gas.idx["N2"]
    lo = gas.e_vib_species(np.array([50.0]))[s][0]
    hi = gas.e_vib_species(np.array([50000.0]))[s][0]
    assert lo < 1.0
    assert abs(hi / (gas.Rs[s] * 50000.0) - 1.0) < 0.05


def test_cv_ve_is_derivative_of_e_ve(gas):
    Tv = np.array([800.0, 3000.0, 9000.0])
    cv = gas.cv_ve_species(Tv)
    ec = gas.e_ve_species(Tv + 1e-30j)
    for s in range(gas.ns):
        assert np.allclose(np.imag(ec[s]) / 1e-30, cv[s], rtol=1e-10)


@pytest.mark.parametrize("tr", ["WILKE", "SUTHERLAND"])
def test_transport_prandtl(tr):
    g = SU2TCLibAir7(transport=tr)
    T = np.array([300.0, 1500.0, 3000.0])
    rs = _air(g)
    mu, ktr, kve, D_ = g.transport(rs, T, T)
    Pr = mu * g.cp_frozen(rs, T, T) / (ktr + kve)
    assert np.all((Pr > 0.5) & (Pr < 0.85))
    assert all(np.all(np.asarray(d) > 0) for d in D_)


def test_gupta_yos_partner_quirk_reproduced():
    T = np.array([1500.0])
    a = SU2TCLibAir7(transport="GUPTA-YOS", gy_partner_bug=True)
    b = SU2TCLibAir7(transport="GUPTA-YOS")
    rs = _air(a, n=1)
    ka = sum(a.transport(rs, T, T)[1:3])
    kb = sum(b.transport(rs, T, T)[1:3])
    assert ka[0] < 0.3 * kb[0]                  # SU2's rule underpredicts k
    mu = b.transport(rs, T, T)[0]
    Pr = mu * b.cp_frozen(rs, T, T) / kb
    assert 0.6 < Pr[0] < 0.8


def test_keq_table_row_selection(gas):
    t = D.KEQ_TABLES[0]
    A = gas._keq_coeffs(np.array([5.0e13, 1.5e14, 3.0e16, 2.0e19]))
    assert np.allclose(A[0][0], [t[5, 0], t[0, 0],
                                              t[2, 0] + (t[3, 0] - t[2, 0]) *
                                              (3e16 - 1e16) / 9e16, t[5, 0]])


def test_reactions_conserve_mass_charge_elements(gas):
    nu = gas.NU
    # SU2 gives NO+ the molar mass of NO, so the associative ionisation
    # N + O -> NO+ + e- creates exactly one electron mass per mole
    defect = nu @ gas.MW
    assert np.abs(defect).max() <= gas.MW[gas.i_e] * (1 + 1e-12)
    assert np.count_nonzero(np.abs(defect) > 1e-15) == 1
    assert np.abs(nu @ gas.charge).max() == 0.0
    assert gas.nr == 22
    rs = _air(gas, n=2)
    rs[3] = 0.05 * rs.sum(axis=0)
    rs[4] = 0.05 * rs.sum(axis=0)
    rs[5] = 0.05 * rs.sum(axis=0)
    w = gas.production_rates(rs, np.array([6000.0, 9000.0]),
                             np.array([5000.0, 8000.0]))
    tot = sum(w)
    ion = np.abs(w[gas.idx["NO+"]]) / gas.MW[gas.idx["NO+"]] * gas.MW[gas.i_e]
    assert np.all(np.abs(tot) < 1e-9 * np.abs(w[gas.idx["N"]]).max() + 2 * ion)
    q = sum(w[s] * gas.charge[s] / gas.MW[s] for s in range(gas.ns))
    assert np.all(np.abs(q) < 1e-9 * np.abs(w[gas.idx["NO+"]]).max() /
                  gas.MW[gas.idx["NO+"]] + 1e-30)


def test_relaxation_drives_tv_to_t(gas):
    rs = _air(gas, n=2)
    T = np.array([3000.0, 3000.0])
    Q = gas.source_ve(rs, T, np.array([2000.0, 4000.0]))
    assert Q[0] > 0.0 and Q[1] < 0.0


def test_mutationpp_backend_thermo_and_kinetics():
    with pytest.warns(RuntimeWarning):
        g = make_gas("mutationpp")
    assert g.names == ["e-", "N2", "O2", "NO", "N", "O", "NO+"]
    # formation enthalpy recovered at the 298.15 K standard state
    T = np.array([298.15])
    h = g.h_species(T, T)
    s = g.idx["NO"]
    assert abs(h[s][0] * g.MW[s] - 91089.0) < 1e-6
    kinds = {r["formula"]: r["kind"] for r in g.rxns}
    assert kinds["N2+M=2N+M"] == "DISSOCIATION_M"
    assert kinds["N2+e-=2N+e-"] == "DISSOCIATION_E"
    assert kinds["N+O=NO++e-"] == "ASSOCIATIVE_IONIZATION"
    rs = _air(g, n=1)
    rs[4] = rs[5] = 0.02 * rs.sum()
    w = g.production_rates(rs, np.array([7000.0]), np.array([6000.0]))
    assert abs(float(sum(w)[0])) < 1e-9 * abs(float(w[g.idx["N"]][0]))


def test_mutationpp_and_su2_agree_on_forward_chemistry(gas):
    with pytest.warns(RuntimeWarning):
        m = make_gas("mutationpp")
    rs = _air(gas, n=1)
    T = np.array([8000.0])
    w1 = gas.production_rates(rs, T, T)
    w2 = m.production_rates(rs, T, T)
    k = gas.idx["O"]
    assert abs(w1[k][0] / w2[k][0] - 1.0) < 0.05
