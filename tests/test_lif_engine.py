"""
Tests that the engine implements the *published* LIF model, not an ad hoc one.

The key test is `test_matches_literal_reference_implementation`, which compares
the optimised engine against a deliberately naive, slow, literal transcription
of the Brian2 model from Shiu et al. (2024) running on a real FlyWire
subnetwork. If the optimisations ever change the biology, that test fails.
"""
from __future__ import annotations

import numpy as np
import pytest

from brain.neuron_models.lif import LIFParams, DEFAULT
from brain.neurons.registry import load_connectome
from simulation.engine.lif_engine import LIFEngine


@pytest.fixture(scope="module")
def c():
    return load_connectome()


@pytest.fixture(scope="module")
def sub(c):
    """A real ~4000-neuron FlyWire subnetwork centred on the escape circuit."""
    seeds = c.by_cell_types(["LPLC2", "LC4", "DNp01"])["idx"].to_numpy()
    keep = set(int(i) for i in seeds)
    for i in seeds:                      # one hop downstream, real connectivity
        row = c.w.getrow(int(i))
        keep.update(int(j) for j in row.indices)
    idx = np.array(sorted(keep))[:4000]
    return c.subgraph(idx)


# --------------------------------------------------------------- published params
def test_parameters_match_published_values():
    p = DEFAULT
    assert (p.v_0, p.v_rst, p.v_th) == (-52.0, -52.0, -45.0)
    assert (p.t_mbr, p.tau, p.t_rfc, p.t_dly) == (20.0, 5.0, 2.2, 1.8)
    assert p.w_syn == 0.275
    assert p.r_poi == 150.0 and p.f_poi == 250.0
    assert p.dt == 0.1
    assert p.delay_steps == 18
    assert p.refractory_steps == 22
    assert p.poisson_weight == pytest.approx(68.75)


# ------------------------------------------------------------- exact integration
def test_exact_integration_matches_numerical_solution():
    """Our closed-form step equals Heun integration of the published ODEs."""
    p = DEFAULT
    a, b = 1.0 / p.t_mbr, 1.0 / p.tau
    ev, eg = np.exp(-a * p.dt), np.exp(-b * p.dt)
    kg = a * (eg - ev) / (a - b)

    for v0, g0 in [(-52.0, 5.0), (-48.0, 0.0), (-52.0, -3.0), (-46.0, 12.0)]:
        v, g, h = v0, g0, p.dt / 200000
        for _ in range(200000):
            dv, dg = (p.v_0 - v + g) / p.t_mbr, -g / p.tau
            v2, g2 = v + h * dv, g + h * dg
            v += h * 0.5 * (dv + (p.v_0 - v2 + g2) / p.t_mbr)
            g += h * 0.5 * (dg + (-g2 / p.tau))
        assert p.v_0 + (v0 - p.v_0) * ev + kg * g0 == pytest.approx(v, abs=1e-9)
        assert g0 * eg == pytest.approx(g, abs=1e-9)


def test_membrane_decays_to_rest_with_no_input(c):
    e = LIFEngine(c.subgraph(np.arange(200)), seed=0)
    e.v[:] = -48.0
    e.run(200.0)
    assert np.allclose(e.v, DEFAULT.v_0, atol=1e-3)


def test_single_synapse_psp_amplitude(c):
    """A known real connection produces the voltage step the model prescribes."""
    p = DEFAULT
    # float64 here: the PSP of a single event is ~0.014 mV on a ~-52 mV
    # baseline, which is at the edge of float32 resolution.
    e = LIFEngine(c.subgraph(np.arange(50)), seed=0, dtype=np.float64)
    e.g[:] = 0.0
    n_syn = 10
    e.g[0] = n_syn * p.w_syn           # one presynaptic event of 10 synapses
    v_before = float(e.v[0])
    e.step()
    # dv over one step = kg * g  (exact linear solution)
    a, b = 1.0 / p.t_mbr, 1.0 / p.tau
    kg = a * (np.exp(-b * p.dt) - np.exp(-a * p.dt)) / (a - b)
    assert float(e.v[0]) - v_before == pytest.approx(kg * n_syn * p.w_syn, rel=1e-4)


# ------------------------------------------------------------ literal reference
def _literal_reference(conn, exc_idx, rate_hz, duration_ms, seed, params=DEFAULT):
    """
    Naive, slow, literal transcription of the Brian2 model in
    https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py

        dv/dt = (v_0 - v + g)/t_mbr   (unless refractory)
        dg/dt = -g/tau                (unless refractory)
        v > v_th -> v = v_rst; g = 0; refractory t_rfc
        on_pre (delayed t_dly): g += w
    Uses float64 dense arrays and no optimisation of any kind.
    """
    p = params
    n = conn.n
    W = np.asarray(conn.w.todense(), dtype=np.float64) * p.w_syn
    n_steps = int(round(duration_ms / p.dt))
    D = int(round(p.t_dly / p.dt))
    R = int(round(p.t_rfc / p.dt))

    v = np.full(n, p.v_0, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    rfc = np.zeros(n, dtype=np.int64)
    ring = np.zeros((D + 1, n), dtype=np.float64)
    rfc_len = np.full(n, R, dtype=np.int64)
    rfc_len[exc_idx] = 0

    a, b = 1.0 / p.t_mbr, 1.0 / p.tau
    ev, eg = np.exp(-a * p.dt), np.exp(-b * p.dt)
    kg = a * (eg - ev) / (a - b)

    rng = np.random.default_rng(seed)
    prob = rate_hz * p.dt * 1e-3
    out = []
    for s in range(n_steps):
        slot = s % (D + 1)
        g += ring[slot]
        ring[slot] = 0.0

        for i in range(n):                       # deliberately per-neuron
            if rfc[i] == 0:
                v[i] = p.v_0 + (v[i] - p.v_0) * ev + kg * g[i]
                g[i] = g[i] * eg
            else:
                rfc[i] -= 1

        fired = rng.random(len(exc_idx)) < prob
        for k, i in enumerate(exc_idx):
            if fired[k]:
                v[i] += p.w_syn * p.f_poi

        spk = [i for i in range(n) if v[i] > p.v_th and rfc[i] == 0]
        tgt = (slot + D) % (D + 1)
        for i in spk:
            v[i] = p.v_rst
            g[i] = 0.0
            rfc[i] = rfc_len[i]
            ring[tgt] += W[i]
        out.append(np.array(spk, dtype=np.int64))
    return out


def test_matches_literal_reference_implementation(c):
    """Optimised engine == literal transcription of the published Brian2 model."""
    small = c.subgraph(c.by_cell_types(["LPLC2", "DNp01"])["idx"].to_numpy()[:120])
    exc = np.arange(min(20, small.n))

    ref = _literal_reference(small, exc, 300.0, 25.0, seed=7)

    eng = LIFEngine(small, seed=7, dtype=np.float64)
    eng.set_poisson(exc, 300.0)
    got = eng.run(25.0)

    assert len(ref) == len(got)
    for s, (r, gspk) in enumerate(zip(ref, got)):
        assert np.array_equal(r, gspk), "spike mismatch at step %d: %s vs %s" % (s, r, gspk)
    assert sum(len(x) for x in ref) > 0, "reference produced no spikes; test is vacuous"


# ----------------------------------------------------------------- behaviour
def test_poisson_drive_produces_expected_firing_rate(c):
    """Poisson-driven neurons fire at approximately the requested rate."""
    e = LIFEngine(c.subgraph(np.arange(300)), seed=3)
    idx = np.arange(50)
    e.set_poisson(idx, 150.0)
    e.run(2000.0)
    rates = e.spike_counts[idx] / 2.0        # 2 s
    assert 130 < rates.mean() < 170, rates.mean()


def test_refractory_period_limits_rate(c):
    """A non-Poisson neuron cannot exceed 1 / t_rfc."""
    e = LIFEngine(c.subgraph(np.arange(300)), seed=4)
    e.rfc_len[:] = DEFAULT.refractory_steps
    for _ in range(10000):
        e.g[7] = 500.0            # drive far above threshold every step
        e.step()
    max_rate = 1000.0 / DEFAULT.t_rfc
    assert e.spike_counts[7] / 1.0 <= max_rate + 1


def test_inhibitory_input_hyperpolarises(c):
    e = LIFEngine(c.subgraph(np.arange(50)), seed=0)
    e.g[0] = -5.0
    e.run(2.0)
    assert e.v[0] < DEFAULT.v_0


def test_determinism(c):
    sg = c.subgraph(c.by_cell_types(["LPLC2", "DNp01"])["idx"].to_numpy()[:150])
    runs = []
    for _ in range(2):
        e = LIFEngine(sg, seed=11)
        e.set_poisson(np.arange(20), 200.0)
        e.run(50.0)
        runs.append(e.spike_counts.copy())
    assert np.array_equal(runs[0], runs[1])


def test_silencing_removes_downstream_drive(c):
    """Shiu et al. `silence()`: a silenced neuron makes no output."""
    idx = c.by_cell_types(["LPLC2", "LC4", "DNp01"])["idx"].to_numpy()
    sg = c.subgraph(idx)
    lplc2 = sg.by_cell_type("LPLC2")["idx"].to_numpy()

    a = LIFEngine(sg, seed=5)
    a.set_poisson(lplc2, 150.0)
    a.run(100.0)

    b = LIFEngine(sg, seed=5)
    b.set_poisson(lplc2, 150.0)
    b.silence(lplc2)
    b.run(100.0)

    gf = sg.by_cell_type("DNp01")["idx"].to_numpy()
    assert a.spike_counts[gf].sum() >= b.spike_counts[gf].sum()
    non_lplc2 = np.setdiff1d(np.arange(sg.n), lplc2)
    assert b.spike_counts[non_lplc2].sum() == 0


def test_float32_and_float64_give_identical_spikes(c):
    """
    The engine runs in float32 for speed. This proves that choice does not
    change the biology: spike trains are identical to float64 on a real
    FlyWire subnetwork driven through the escape circuit.
    """
    sg = c.subgraph(c.by_cell_types(["LPLC2", "LC4", "DNp01"])["idx"].to_numpy())
    exc = sg.by_cell_type("LPLC2")["idx"].to_numpy()

    out = []
    for dt in (np.float32, np.float64):
        e = LIFEngine(sg, seed=21, dtype=dt)
        e.set_poisson(exc, 150.0)
        e.run(150.0)
        out.append(e.spike_counts.copy())
    assert out[0].sum() > 0
    assert np.array_equal(out[0], out[1])
