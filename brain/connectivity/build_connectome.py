"""
Build the simulation-ready connectome from the real FlyWire FAFB v783 release.

PROVENANCE
----------
A. REAL DATA (unmodified from FlyWire v783):
   - neuron root IDs, cell classes, cell types, side, coordinates
   - synaptic connectivity (pre_root_id, post_root_id, syn_count) per neuropil
   - neurotransmitter predictions (per neuron and per connection)

B. PUBLISHED MODEL ASSUMPTION (Shiu et al. 2024, Nature 634:210-219):
   - a neuron's sign is set by its predicted neurotransmitter:
     ACH/DA/OCT/SER -> excitatory (+1);  GABA/GLUT -> inhibitory (-1)
     (glutamate is treated as inhibitory in Drosophila via GluCl-alpha)
   - synaptic weight = sign * syn_count * w_syn

C. OUR APPROXIMATION (documented, counted, reported in the build manifest):
   - neurons whose neuron-level NT prediction is blank get the syn_count-weighted
     majority NT of their own outgoing connections. Count is reported.
   - connections are summed across neuropils into a single (pre, post) weight.
   - a neuron's "primary neuropil" is the neuropil holding most of its synapses.

Run:  python -m brain.connectivity.build_connectome
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config

# --- B: published sign convention (Shiu et al. 2024) ------------------------
EXCITATORY = ("ACH", "DA", "OCT", "SER")
INHIBITORY = ("GABA", "GLUT")
NT_SIGN = dict([(nt, +1) for nt in EXCITATORY] + [(nt, -1) for nt in INHIBITORY])

# Codex `coordinates.csv` stores positions in NANOMETRES already, not in
# 4x4x40 nm voxels. Verified against the raw ranges: read as nm the neuron
# cloud spans 901 x 443 x 279 um, which matches the real Drosophila brain;
# read as voxels it would span an impossible 3604 x 1772 x 11165 um.
# tests/test_data_provenance.py::test_neuron_positions_span_a_real_fly_brain
# pins this down.
COORD_TO_NM = np.array([1.0, 1.0, 1.0], dtype=np.float64)


def _log(msg):
    print("[build] " + msg, flush=True)


def load_neuron_table():
    """Load and join the real per-neuron annotation tables."""
    config.require_source_files()

    _log("reading neurons.csv.gz ...")
    neurons = pd.read_csv(
        config.SRC["neurons"],
        usecols=["root_id", "nt_type", "nt_type_score"],
        dtype={"root_id": np.int64, "nt_type": "object", "nt_type_score": np.float64},
    )

    if len(neurons) != config.EXPECTED_NEURONS:
        raise ValueError(
            "Expected %d neurons for FlyWire v783, got %d. This does not look like "
            "the v783 release; refusing to build."
            % (config.EXPECTED_NEURONS, len(neurons))
        )
    _log("  %s neurons (matches published v783 count)" % format(len(neurons), ","))

    _log("reading classification.csv.gz ...")
    cls = pd.read_csv(
        config.SRC["classification"],
        usecols=["root_id", "flow", "super_class", "class", "sub_class", "side", "nerve"],
        dtype={"root_id": np.int64},
    )

    _log("reading consolidated_cell_types.csv.gz ...")
    ctypes = pd.read_csv(
        config.SRC["consolidated_cell_types"],
        usecols=["root_id", "primary_type"],
        dtype={"root_id": np.int64, "primary_type": "object"},
    )

    _log("reading coordinates.csv.gz ...")
    coords = pd.read_csv(
        config.SRC["coordinates"],
        usecols=["root_id", "position"],
        dtype={"root_id": np.int64, "position": "object"},
    )
    # Codex lists >= 1 anchor point per neuron; keep the first (Codex convention).
    coords = coords.drop_duplicates("root_id", keep="first").copy()
    xyz = np.array(
        [np.fromstring(str(p).strip("[]"), sep=" ") for p in coords["position"]],
        dtype=np.float64,
    ) * COORD_TO_NM
    coords["pos_x_nm"] = xyz[:, 0]
    coords["pos_y_nm"] = xyz[:, 1]
    coords["pos_z_nm"] = xyz[:, 2]
    coords = coords.drop(columns=["position"])

    df = (
        neurons.merge(cls, on="root_id", how="left")
        .merge(ctypes, on="root_id", how="left")
        .merge(coords, on="root_id", how="left")
    )
    # Deterministic ordering: ascending root_id defines the simulation index.
    df = df.sort_values("root_id", kind="mergesort").reset_index(drop=True)
    df.insert(0, "idx", np.arange(len(df), dtype=np.int32))
    return df


def load_connections(root_ids):
    """Load real synaptic connectivity and map root IDs to simulation indices."""
    _log("reading connections_princeton.csv.gz (5.3M edge rows) ...")
    con = pd.read_csv(
        config.SRC["connections"],
        usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"],
        dtype={
            "pre_root_id": np.int64,
            "post_root_id": np.int64,
            "neuropil": "object",
            "syn_count": np.int32,
            "nt_type": "object",
        },
    )
    _log("  %s (pre, post, neuropil) rows; %s synapses total"
         % (format(len(con), ","), format(int(con["syn_count"].sum()), ",")))

    pre_raw = con["pre_root_id"].to_numpy()
    post_raw = con["post_root_id"].to_numpy()
    n = len(root_ids)

    # root_ids is sorted ascending -> searchsorted yields the simulation index.
    pre_i = np.searchsorted(root_ids, pre_raw)
    post_i = np.searchsorted(root_ids, post_raw)
    pre_c = np.clip(pre_i, 0, n - 1)
    post_c = np.clip(post_i, 0, n - 1)
    ok = (root_ids[pre_c] == pre_raw) & (root_ids[post_c] == post_raw)

    n_drop = int((~ok).sum())
    if n_drop:
        _log("  WARNING: %s edges reference root IDs absent from neurons.csv (dropped)"
             % format(n_drop, ","))

    return (pre_i[ok].astype(np.int32),
            post_i[ok].astype(np.int32),
            con["syn_count"].to_numpy()[ok].astype(np.int32),
            con.loc[ok])


def resolve_nt_signs(df, con):
    """
    Assign each neuron an excitatory/inhibitory sign (Shiu et al. convention).

    Blank neuron-level NT predictions fall back to the syn_count-weighted majority
    NT over that neuron's own outgoing connections (our approximation, counted).
    """
    nt = df["nt_type"].fillna("").astype(str).str.upper().to_numpy().astype(object)
    blank = np.array([n not in NT_SIGN for n in nt])
    n_blank = int(blank.sum())
    _log("  %s neurons have no neuron-level NT prediction -> edge-majority fallback"
         % format(n_blank, ","))

    if n_blank:
        votes = (
            con.groupby(["pre_root_id", "nt_type"], observed=True)["syn_count"]
            .sum()
            .reset_index()
            .sort_values("syn_count", ascending=False)
            .drop_duplicates("pre_root_id", keep="first")
            .set_index("pre_root_id")["nt_type"]
        )
        fill = (df.loc[blank, "root_id"].map(votes)
                .fillna("").astype(str).str.upper().to_numpy())
        nt[blank] = fill

    resolved = int((blank & np.array([n in NT_SIGN for n in nt])).sum())
    unresolved = int(sum(1 for n in nt if n not in NT_SIGN))
    _log("  %s resolved from edges; %s remain unknown (sign 0 -> makes no output)"
         % (format(resolved, ","), format(unresolved, ",")))

    sign = np.array([NT_SIGN.get(n, 0) for n in nt], dtype=np.int8)
    return nt, sign, n_blank


def primary_neuropil(df, con):
    """Neuropil holding the most of each neuron's synapses (pre + post roles)."""
    _log("computing primary neuropil per neuron ...")
    a = con.groupby(["pre_root_id", "neuropil"], observed=True)["syn_count"].sum()
    a.index.names = ["root_id", "neuropil"]
    b = con.groupby(["post_root_id", "neuropil"], observed=True)["syn_count"].sum()
    b.index.names = ["root_id", "neuropil"]
    tot = a.add(b, fill_value=0).reset_index()
    best = (tot.sort_values("syn_count", ascending=False)
              .drop_duplicates("root_id", keep="first")
              .set_index("root_id")["neuropil"])
    return df["root_id"].map(best).fillna("UNKNOWN").astype(str).to_numpy()


def main():
    t0 = datetime.now(timezone.utc)
    df = load_neuron_table()
    root_ids = df["root_id"].to_numpy()

    pre_i, post_i, syn, con = load_connections(root_ids)
    nt, sign, n_blank = resolve_nt_signs(df, con)
    df["nt_resolved"] = nt
    df["sign"] = sign
    df["primary_neuropil"] = primary_neuropil(df, con)

    _log("building sparse CSR connectivity (summing across neuropils) ...")
    n = len(df)
    counts = sp.coo_matrix(
        (syn.astype(np.int32), (pre_i, post_i)), shape=(n, n)
    ).tocsr()                       # duplicate (pre, post) rows are summed
    counts.sum_duplicates()
    counts.eliminate_zeros()

    n_pairs = int(counts.nnz)
    n_syn = int(counts.data.sum())
    _log("  %s unique (pre, post) neuron pairs; %s synapses"
         % (format(n_pairs, ","), format(n_syn, ",")))

    # B: signed weight = sign(presynaptic NT) * syn_count
    signed = sp.csr_matrix(
        (counts.data * sign[np.repeat(np.arange(n), np.diff(counts.indptr))].astype(np.int32),
         counts.indices.copy(), counts.indptr.copy()),
        shape=(n, n),
    )
    signed.sort_indices()

    _log("saving %s ..." % config.CONNECTOME_NPZ.name)
    np.savez_compressed(
        config.CONNECTOME_NPZ,
        root_ids=root_ids.astype(np.int64),
        indptr=signed.indptr.astype(np.int64),
        indices=signed.indices.astype(np.int32),
        data=signed.data.astype(np.int32),      # signed synapse counts
        shape=np.array([n, n], dtype=np.int64),
        dataset=np.array(["%s v%s" % (config.DATASET_NAME, config.DATASET_VERSION)]),
    )

    _log("saving %s ..." % config.NEURON_INDEX.name)
    df.to_csv(config.NEURON_INDEX, index=False, compression="gzip")

    manifest = {
        "dataset": config.DATASET_NAME,
        "version": config.DATASET_VERSION,
        "source_url": config.DATASET_URL,
        "source_dir": str(config.FLYWIRE_DIR),
        "built_utc": t0.isoformat(),
        "n_neurons": int(n),
        "n_neuron_pairs": n_pairs,
        "n_synapses": n_syn,
        "n_edge_rows_in_source": int(len(con)),
        "excitatory_neurons": int((sign > 0).sum()),
        "inhibitory_neurons": int((sign < 0).sum()),
        "unknown_sign_neurons": int((sign == 0).sum()),
        "nt_blank_in_source": int(n_blank),
        "nt_sign_convention": {
            "excitatory": list(EXCITATORY),
            "inhibitory": list(INHIBITORY),
        },
        "connectivity_file": config.SRC["connections"].name,
        "model_reference": (
            "Shiu et al. 2024, Nature 634:210-219, doi:10.1038/s41586-024-07763-9"
        ),
    }
    config.BUILD_MANIFEST.write_text(json.dumps(manifest, indent=2))
    keys = ("n_neurons", "n_neuron_pairs", "n_synapses",
            "excitatory_neurons", "inhibitory_neurons", "unknown_sign_neurons")
    _log("manifest: " + json.dumps({k: manifest[k] for k in keys}, indent=2))
    _log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
