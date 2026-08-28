"""
Tests that prove the loaded data really is the FlyWire FAFB v783 connectome.

These are deliberately paranoid. They are the guard against silently
substituting a different, simplified, or synthetic dataset. Every constant
checked here is a published property of FlyWire v783, not something this
project invented.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import config
from brain.neurons.registry import load_connectome


# Published FlyWire FAFB v783 reference figures.
PUBLISHED_NEURON_COUNT = 139255
PUBLISHED_SYNAPSE_COUNT = 50_666_648   # sum of syn_count in connections_princeton
PUBLISHED_EDGE_ROWS = 5_342_446        # (pre, post, neuropil) rows

# SHA-256 of the exact source files this project was built against.
EXPECTED_SHA256 = {
    "neurons.csv.gz":
        "6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719",
    "classification.csv.gz":
        "e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663",
    "consolidated_cell_types.csv.gz":
        "8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d",
    "connections_princeton.csv.gz":
        "445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b",
    "coordinates.csv.gz":
        "14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05",
    "visual_neuron_types.csv.gz":
        "4bcc6a2f98b86e6c3fb7eaddb49736f3d81ab65bda35da8f740641201a1e379f",
}


@pytest.fixture(scope="module")
def c():
    return load_connectome()


# --------------------------------------------------------------- source files
def test_source_files_present():
    config.require_source_files()


@pytest.mark.parametrize("fname,sha", sorted(EXPECTED_SHA256.items()))
def test_source_file_checksum(fname, sha):
    """The bytes on disk are the exact FlyWire v783 files we documented."""
    path = config.FLYWIRE_DIR / fname
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == sha, (
        "%s does not match the documented FlyWire v783 file. The dataset has "
        "been substituted or altered." % fname
    )


# ------------------------------------------------------------------- identity
def test_neuron_count_matches_published_v783(c):
    assert c.n == PUBLISHED_NEURON_COUNT


def test_all_root_ids_have_fafb_segmentation_prefix(c):
    """FlyWire FAFB v783 root IDs all begin 720575940 (CAVE segmentation)."""
    s = c.root_ids.astype(str)
    assert np.all(np.char.startswith(s, config.ROOT_ID_PREFIX))


def test_root_ids_are_unique_and_sorted(c):
    assert len(np.unique(c.root_ids)) == c.n
    assert np.all(np.diff(c.root_ids) > 0)


def test_synapse_total_matches_published(c):
    assert int(np.abs(c.w.data).sum()) == PUBLISHED_SYNAPSE_COUNT


def test_manifest_records_real_dataset(c):
    m = c.manifest
    assert m["dataset"] == "FlyWire FAFB"
    assert m["version"] == "783"
    assert m["n_neurons"] == PUBLISHED_NEURON_COUNT
    assert m["n_synapses"] == PUBLISHED_SYNAPSE_COUNT
    assert m["n_edge_rows_in_source"] == PUBLISHED_EDGE_ROWS
    assert "Shiu" in m["model_reference"]


# --------------------------------------------------- specific real neuron IDs
# These are individually verifiable at https://codex.flywire.ai/app/cell_details?root_id=...
KNOWN_NEURONS = {
    720575940622838154: ("DNp01", "descending", "left"),    # Giant Fiber, left
    720575940632499757: ("DNp01", "descending", "right"),   # Giant Fiber, right
}


@pytest.mark.parametrize("root_id,expected", sorted(KNOWN_NEURONS.items()))
def test_known_real_neuron_is_present_with_correct_annotation(c, root_id, expected):
    cell_type, super_class, side = expected
    i = c.idx(root_id)
    row = c.neurons.iloc[i]
    assert int(row["root_id"]) == root_id
    assert str(row["primary_type"]) == cell_type
    assert str(row["super_class"]) == super_class
    assert str(row["side"]) == side


def test_a_fabricated_root_id_is_rejected(c):
    """Guard against a loader that would invent neurons on demand."""
    with pytest.raises(KeyError):
        c.idx(720575940000000001)


# ------------------------------------------------------- annotation integrity
def test_super_class_counts_match_v783(c):
    """Published v783 super_class census."""
    expected = {
        "optic": 77873, "central": 32381, "sensory": 16938,
        "visual_projection": 7684, "ascending": 1750, "descending": 1305,
        "sensory_ascending": 612, "visual_centrifugal": 522,
        "motor": 110, "endocrine": 80,
    }
    got = c.neurons["super_class"].value_counts().to_dict()
    for k, v in expected.items():
        assert got.get(k) == v, "super_class %s: expected %d, got %s" % (k, v, got.get(k))


def test_neurotransmitter_census_matches_v783(c):
    """Published v783 neuron-level NT predictions (before our fallback)."""
    import pandas as pd
    nt = pd.read_csv(config.SRC["neurons"], usecols=["nt_type"])["nt_type"]
    counts = nt.value_counts(dropna=False).to_dict()
    assert counts.get("ACH") == 82298
    assert counts.get("GLUT") == 19605
    assert counts.get("GABA") == 16017
    assert counts.get("SER") == 1021
    assert counts.get("DA") == 584
    assert counts.get("OCT") == 72


def test_cell_type_population_sizes(c):
    """Real population sizes of the cell types used by our experiments."""
    expected = {"LPLC2": 210, "LC4": 104, "LC6": 125, "LPLC1": 140,
                "DNp01": 2, "DNp02": 2, "DNp04": 2, "DNp11": 2}
    for t, n in expected.items():
        assert len(c.by_cell_type(t)) == n, "%s population size" % t


def test_signs_are_dale_consistent(c):
    """Every neuron's outgoing weights share one sign (Shiu et al. assumption)."""
    w = c.w
    for i in c.neurons.sample(500, random_state=0)["idx"]:
        row = w.data[w.indptr[i]:w.indptr[i + 1]]
        if row.size:
            assert (row > 0).all() or (row < 0).all()


def test_checksum_file_is_present_and_consistent():
    text = config.CHECKSUM_FILE.read_text()
    for fname, sha in EXPECTED_SHA256.items():
        assert sha in text and fname in text


def test_neuron_positions_span_a_real_fly_brain(c):
    """
    Codex coordinates are in NANOMETRES, not 4x4x40 nm voxels.

    Read as nm the neuron cloud spans roughly 810 x 390 x 280 um, which matches
    a real adult Drosophila brain (including both optic lobes). Read as voxels
    it would span ~3600 x 1770 x 11000 um, i.e. a brain the size of a mouse.
    This test fails if the unit convention is ever changed back.
    """
    xyz = c.neurons[["pos_x_nm", "pos_y_nm", "pos_z_nm"]].to_numpy() / 1000.0
    span = np.nanmax(xyz, axis=0) - np.nanmin(xyz, axis=0)
    assert 700 < span[0] < 950, "medio-lateral span %.0f um" % span[0]
    assert 300 < span[1] < 550, "dorso-ventral span %.0f um" % span[1]
    assert 200 < span[2] < 400, "antero-posterior span %.0f um" % span[2]


def test_every_neuron_has_a_position(c):
    xyz = c.neurons[["pos_x_nm", "pos_y_nm", "pos_z_nm"]].to_numpy()
    assert not np.isnan(xyz).any()
