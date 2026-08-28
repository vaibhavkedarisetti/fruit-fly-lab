"""
Tests for the retinotopic map, the sensory encoders, and the body model.

The important claims checked here:
  - the retinotopic map is derived from real column assignments and matches
    real anatomy (not invented);
  - unsupported stimuli are refused rather than faked;
  - the body cannot move without descending-neuron activity, so no behaviour
    can be produced by a stimulus short-circuiting the brain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from brain.neurons.registry import load_connectome
from brain.sensory.encoders import LoomingEncoder, PopulationEncoder
from brain.sensory.modalities import ALL_MODALITIES, BY_KEY, census, resolve_neurons
from brain.sensory.retinotopy import load_retinotopy
from fly.body.fly_body import FlyBody
from simulation.stimuli.looming import LoomingStimulus, angular_distance_deg
from simulation.stimuli.pulse import PulseStimulus


@pytest.fixture(scope="module")
def c():
    return load_connectome()


@pytest.fixture(scope="module")
def retino(c):
    return load_retinotopy(c)


# ============================== retinotopy ================================
def test_column_counts_match_the_real_optic_lobe(retino):
    """~750-800 ommatidia per eye in Drosophila."""
    n = retino.n_columns
    assert 700 < n["left"] < 850
    assert 700 < n["right"] < 850


def test_hex_axes_are_oriented_by_real_anatomy(c):
    """
    Re-derives the claim in retinotopy.py: over the 1,581 Mi1 neurons (one per
    column), u = p + q/2 tracks the dorsoventral axis and the orthogonal hex
    axis tracks the anteroposterior axis. Measured, not assumed.
    """
    ca = pd.read_csv(config.SRC["column_assignment"], dtype={"root_id": np.int64})
    pos = c.neurons.set_index("root_id")[["pos_y_nm", "pos_z_nm"]]
    mi1 = ca[ca["type"] == "Mi1"].join(pos, on="root_id").dropna()
    assert len(mi1) > 1500

    for hemi in ("left", "right"):
        s = mi1[mi1["hemisphere"] == hemi]
        u = (s["p"] + s["q"] / 2.0).to_numpy()
        v = ((np.sqrt(3) / 2) * s["q"]).to_numpy()
        # u explains the dorsoventral axis almost perfectly
        assert abs(np.corrcoef(u, s["pos_y_nm"])[0, 1]) > 0.95

        # v explains the anteroposterior axis once u is regressed out
        def resid(t, x):
            b = np.c_[x, np.ones(len(x))]
            k, *_ = np.linalg.lstsq(b, t, rcond=None)
            return t - b @ k
        r = np.corrcoef(resid(v, u), resid(s["pos_z_nm"].to_numpy(), u))[0, 1]
        assert abs(r) > 0.9, "%s: partial corr %.2f" % (hemi, r)


@pytest.mark.parametrize("cell_type", ["LC4", "LPLC2"])
def test_every_looming_detector_gets_a_receptive_field(retino, cell_type):
    rf = retino.receptive_fields(cell_type)
    assert rf["azimuth_deg"].notna().all(), "some cells got no RF"
    # RF sizes in the published range for lobula columnar neurons (~10-30 deg)
    assert 5 < rf["rf_radius_deg"].mean() < 35


@pytest.mark.parametrize("cell_type", ["LC4", "LPLC2"])
def test_receptive_fields_are_correctly_lateralised(retino, cell_type):
    """Left-brain visual neurons look at the left visual field, and vice versa."""
    rf = retino.receptive_fields(cell_type).dropna(subset=["azimuth_deg"])
    left = rf[rf["side"] == "left"]["azimuth_deg"]
    right = rf[rf["side"] == "right"]["azimuth_deg"]
    assert left.mean() < -20 and right.mean() > 20
    # a modest frontal binocular overlap, as in the real fly
    assert left.max() > 0 > right.min()


# ============================== looming geometry ==========================
def test_angular_size_follows_exact_geometry():
    s = LoomingStimulus(half_size_mm=5.0, speed_mm_s=250.0,
                        start_distance_mm=50.0, t_start_ms=0.0)
    for t in (0.0, 50.0, 100.0, 150.0):
        d = 50.0 - 250.0 * t / 1000.0
        assert s.half_angle_deg(t) == pytest.approx(np.degrees(np.arctan(5.0 / d)), rel=1e-9)


def test_expansion_rate_matches_the_analytic_derivative():
    s = LoomingStimulus(half_size_mm=5.0, speed_mm_s=250.0,
                        start_distance_mm=50.0, t_start_ms=0.0)
    for t in (10.0, 60.0, 120.0):
        num = ((s.half_angle_deg(t + 0.01) - s.half_angle_deg(t - 0.01)) / 0.02) * 1000.0
        assert s.expansion_rate_deg_s(t) == pytest.approx(num, rel=1e-3)


def test_l_over_v_is_the_standard_parameter():
    assert LoomingStimulus(half_size_mm=5.0, speed_mm_s=250.0).l_over_v_ms == 20.0
    assert LoomingStimulus(speed_mm_s=0.0).l_over_v_ms == float("inf")


def test_angular_distance_is_a_great_circle():
    assert angular_distance_deg(0, 0, 0, 0) == pytest.approx(0)
    assert angular_distance_deg(0, 0, 90, 0) == pytest.approx(90)
    assert angular_distance_deg(0, -45, 0, 45) == pytest.approx(90)


# ============================== encoders ==================================
def test_receding_object_does_not_drive_looming_detectors(c, retino):
    """LPLC2 is outward-motion selective, LC4 is expansion-driven."""
    enc = LoomingEncoder(c, retino)
    recede = LoomingStimulus(azimuth_deg=45.0, speed_mm_s=-250.0,
                             start_distance_mm=50.0, t_start_ms=0.0)
    assert enc.rates_hz(60.0, recede).max() == 0.0


def test_static_object_does_not_drive_looming_detectors(c, retino):
    enc = LoomingEncoder(c, retino)
    static = LoomingStimulus(azimuth_deg=45.0, speed_mm_s=0.0,
                             start_distance_mm=50.0, t_start_ms=0.0)
    assert enc.rates_hz(60.0, static).max() == 0.0


def test_looming_drive_grows_as_the_object_approaches(c, retino):
    enc = LoomingEncoder(c, retino)
    loom = LoomingStimulus(azimuth_deg=45.0, speed_mm_s=250.0,
                           start_distance_mm=50.0, t_start_ms=0.0)
    seq = [enc.rates_hz(t, loom).sum() for t in (20.0, 80.0, 140.0, 180.0)]
    assert seq == sorted(seq), seq
    assert seq[-1] > 5 * seq[0]


def test_looming_drive_is_spatially_selective(c, retino):
    """An object on the right drives right-eye cells more than left-eye cells."""
    enc = LoomingEncoder(c, retino)
    loom = LoomingStimulus(azimuth_deg=80.0, speed_mm_s=250.0,
                           start_distance_mm=50.0, t_start_ms=0.0)
    r = enc.rates_hz(120.0, loom)
    right = r[(enc.rf["side"] == "right").to_numpy()].sum()
    left = r[(enc.rf["side"] == "left").to_numpy()].sum()
    assert right > 3 * left, "right %.1f vs left %.1f" % (right, left)


# ============================== modalities ================================
def test_every_supported_modality_resolves_to_real_neurons(c):
    for m in ALL_MODALITIES:
        idx = resolve_neurons(m, c)
        if m.supported:
            assert len(idx) > 0, "%s resolves to nothing" % m.key
            assert len(np.unique(idx)) == len(idx)
        else:
            assert len(idx) == 0


def test_every_supported_modality_carries_a_citation():
    for m in ALL_MODALITIES:
        if m.supported:
            assert m.citation and m.doi, "%s lacks a citation" % m.key
        else:
            assert m.unsupported_reason, "%s lacks a reason" % m.key


def test_unsupported_modalities_state_why(c):
    """VNC-innervating touch and histaminergic vision must be refused."""
    for key in ("touch_thorax", "touch_abdomen", "touch_leg"):
        m = BY_KEY[key]
        assert not m.supported
        assert "ventral nerve cord" in m.unsupported_reason
    light = BY_KEY["light_photoreceptor"]
    assert not light.supported
    assert "HISTAMINERGIC" in light.unsupported_reason


def test_population_encoder_scales_with_intensity(c):
    enc = PopulationEncoder(c, BY_KEY["taste_sugar"])
    stim = PulseStimulus(modality_key="taste_sugar", intensity=0.5,
                         t_start_ms=0.0, rise_ms=0.0)
    r = enc.rates_hz(10.0, stim)
    assert r.max() == pytest.approx(BY_KEY["taste_sugar"].max_rate_hz * 0.5)
    off = PulseStimulus(modality_key="taste_sugar", t_start_ms=100.0)
    assert enc.rates_hz(10.0, off).max() == 0.0


def test_census_is_consistent(c):
    rows = census(c)
    assert len(rows) == len(ALL_MODALITIES)
    assert sum(1 for r in rows if r["supported"]) >= 14
    assert sum(1 for r in rows if not r["supported"]) >= 4


# ============================== body model ================================
def test_body_does_nothing_without_descending_activity():
    b = FlyBody()
    for k in range(400):
        b.update(1.0, {}, float(k))
    s = b.state
    assert s.behaviour == "resting"
    assert not s.airborne
    assert (s.x_mm, s.y_mm, s.z_mm) == (0.0, 0.0, 0.0)
    assert b.events == []


def test_giant_fibre_command_produces_short_mode_takeoff():
    """Published: GF escape has ~5 ms latency and no preparatory wing raising."""
    b = FlyBody()
    for k in range(60):
        b.update(1.0, {"escape_takeoff": 0.9}, float(k))
    kinds = [e["event"] for e in b.events]
    assert any("GF" in e for e in kinds)
    cmd = next(e for e in b.events if "GF" in e["event"])
    jump = next(e for e in b.events if "takeoff" in e["event"])
    assert jump["t_ms"] - cmd["t_ms"] == pytest.approx(5.0, abs=1.5)
    assert "short" in jump["event"]


def test_long_mode_escape_takes_about_200ms_of_preparation():
    b = FlyBody()
    for k in range(400):
        b.update(1.0, {"escape_long_mode": 0.9}, float(k))
    cmd = next(e for e in b.events if "long-mode" in e["event"])
    jump = next(e for e in b.events if "takeoff" in e["event"])
    assert 180 < jump["t_ms"] - cmd["t_ms"] < 220
    assert "long" in jump["event"]


def test_one_threat_produces_one_takeoff():
    b = FlyBody()
    for k in range(900):
        b.update(1.0, {"escape_takeoff": 0.9}, float(k))
    assert sum(1 for e in b.events if "takeoff" in e["event"]) == 1


def test_proboscis_extends_only_with_motor_neuron_drive():
    b = FlyBody()
    for k in range(200):
        b.update(1.0, {}, float(k), proboscis_drive=0.0)
    assert b.state.proboscis_extension == pytest.approx(0.0)
    for k in range(200, 400):
        b.update(1.0, {}, float(k), proboscis_drive=0.9)
    assert b.state.proboscis_extension > 0.9


# ============================== composite food ============================
def test_place_food_composite_resolves_to_three_real_populations(c):
    """
    The [PLACE FOOD] control is a composite of three separately cited real
    populations. It is an ENVIRONMENT description -- what a fly meets when food
    is put in front of it -- not a behavioural rule. Each component must resolve
    to real neurons and carry its own citation.
    """
    from visualization.server import Runner
    total = 0
    for key in Runner.FOOD_COMPONENTS:
        m = BY_KEY[key]
        assert m.supported, "%s is not supported" % key
        idx = resolve_neurons(m, c)
        assert len(idx) > 0
        assert m.citation and m.doi
        total += len(idx)
    assert total > 300, "food composite drives only %d neurons" % total


def test_food_components_are_chemosensory(c):
    from visualization.server import Runner
    allowed = {"olfactory", "gustatory"}
    for key in Runner.FOOD_COMPONENTS:
        idx = resolve_neurons(BY_KEY[key], c)
        classes = set(c.neurons.iloc[idx]["class"].astype(str))
        assert classes <= allowed, "%s spans %s" % (key, classes)
