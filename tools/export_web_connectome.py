"""
Export the real FlyWire v783 connectome into browser-loadable binary assets.

This exists because the interactive laboratory has to run somewhere without a
persistent server process. Nothing is simplified, sampled, or approximated: the
exported files carry all 139,255 neurons and all 3,732,460 connections with
their real signed synapse counts. The browser engine reads exactly the same
numbers the Python engine does, and `tools/verify_web_engine.py` proves the two
produce identical spike trains.

Layout
------
web/data/connectome.bin
    int32  indptr[n+1]                    CSR row pointers
    int32  indices_delta[nnz]             column indices, delta-coded within row
    int16  weights[nnz]                   signed synapse counts (max |w| = 2633)

web/data/neurons.bin
    int64   root_id[n]                    real FlyWire root IDs
    float32 pos[n*3]                      anatomical position, micrometres
    uint16  type_code[n]                  index into meta.cell_types
    uint8   class_code[n]                 index into meta.super_classes
    uint8   side_code[n]                  index into meta.sides
    int8    sign[n]                       +1 excitatory, -1 inhibitory, 0 unknown

web/data/meta.json
    counts, build manifest, string tables, LC4/LPLC2 receptive fields,
    stimulus modality definitions with citations, descending-neuron commands.

Run:  python -m tools.export_web_connectome
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import config
from brain.motor.descending import DN_COMMANDS, CHANNEL_HALF_MAX_HZ
from brain.neuron_models.lif import DEFAULT as LIF
from brain.neurons.registry import load_connectome
from brain.sensory.encoders import LoomingTuning
from brain.sensory.modalities import ALL_MODALITIES, resolve_neurons
from brain.sensory.retinotopy import load_retinotopy

WEB = config.PROJECT_ROOT / "web"
DATA = WEB / "data"


def _log(m):
    print("[export] " + m, flush=True)


def delta_encode(indices: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    """Delta-code CSR column indices within each row (they are sorted)."""
    d = indices.astype(np.int64).copy()
    prev = np.empty_like(d)
    prev[0] = 0
    prev[1:] = d[:-1]
    delta = d - prev
    # the first entry of each row keeps its absolute value
    row_starts = indptr[:-1]
    row_starts = row_starts[row_starts < len(d)]
    delta[row_starts] = d[row_starts]
    return delta.astype(np.int32)


def export_connectome(c) -> dict:
    w = c.w.tocsr()
    w.sort_indices()
    indptr = w.indptr.astype(np.int32)
    delta = delta_encode(w.indices, w.indptr)
    weights = w.data.astype(np.int16)

    if int(np.abs(w.data).max()) > 32767:
        raise ValueError("synapse counts exceed int16; widen the export format")

    # sanity: decoding the deltas must reproduce the original indices exactly
    check = delta.astype(np.int64).copy()
    for a, b in zip(w.indptr[:-1], w.indptr[1:]):
        if b > a:
            check[a:b] = np.cumsum(check[a:b])
    if not np.array_equal(check, w.indices.astype(np.int64)):
        raise ValueError("delta encoding round-trip failed")

    blob = indptr.tobytes() + delta.tobytes() + weights.tobytes()
    path = DATA / "connectome.bin"
    path.write_bytes(blob)
    _log("connectome.bin  %8.2f MB  (gzip %.2f MB)"
         % (len(blob) / 1e6, len(gzip.compress(blob, 6)) / 1e6))
    return {"bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "nnz": int(w.nnz), "n": int(w.shape[0])}


def export_neurons(c) -> tuple:
    n = c.neurons
    # Unannotated fields are empty strings, never NaN, so the string tables sort.
    ptype = n["primary_type"].fillna("").astype(str)
    pclass = n["super_class"].fillna("").astype(str)
    pside = n["side"].fillna("").astype(str)
    cell_types = sorted(set(ptype))
    classes = sorted(set(pclass))
    sides = sorted(set(pside))
    if len(cell_types) > 65535:
        raise ValueError("more cell types than uint16 can index")

    tmap = {t: i for i, t in enumerate(cell_types)}
    cmap = {t: i for i, t in enumerate(classes)}
    smap = {t: i for i, t in enumerate(sides)}

    root = n["root_id"].to_numpy().astype(np.int64)
    pos = (n[["pos_x_nm", "pos_y_nm", "pos_z_nm"]].to_numpy(dtype=np.float64) / 1000.0)
    pos = (pos - np.nanmean(pos, axis=0)).astype(np.float32)     # centre, micrometres
    tcode = ptype.map(tmap).to_numpy().astype(np.uint16)
    ccode = pclass.map(cmap).to_numpy().astype(np.uint8)
    scode = pside.map(smap).to_numpy().astype(np.uint8)
    sign = n["sign"].to_numpy().astype(np.int8)

    blob = (root.tobytes() + pos.tobytes() + tcode.tobytes()
            + ccode.tobytes() + scode.tobytes() + sign.tobytes())
    path = DATA / "neurons.bin"
    path.write_bytes(blob)
    _log("neurons.bin     %8.2f MB  (gzip %.2f MB)"
         % (len(blob) / 1e6, len(gzip.compress(blob, 6)) / 1e6))
    return ({"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()},
            cell_types, classes, sides)


def export_meta(c, conn_info, neu_info, cell_types, classes, sides) -> None:
    retino = load_retinotopy(c)

    # receptive fields for the looming detectors, derived from real columns
    rfs = {}
    for t in ("LC4", "LPLC2"):
        df = retino.receptive_fields(t).dropna(subset=["azimuth_deg"])
        rfs[t] = {
            "idx": [int(x) for x in df["idx"]],
            "az": [round(float(x), 3) for x in df["azimuth_deg"]],
            "el": [round(float(x), 3) for x in df["elevation_deg"]],
            "r": [round(float(x), 3) for x in df["rf_radius_deg"]],
            "side": [str(x) for x in df["side"]],
        }

    modalities = []
    for m in ALL_MODALITIES:
        idx = resolve_neurons(m, c)
        modalities.append({
            "key": m.key, "label": m.label, "group": m.group,
            "supported": bool(m.supported and len(idx) > 0),
            "n_neurons": int(len(idx)),
            "idx": [int(x) for x in idx] if m.key != "looming" else [],
            "cell_types": list(m.cell_types), "label_group": m.label_group,
            "description": m.description, "citation": m.citation, "doi": m.doi,
            "notes": m.notes,
            "max_rate_hz": m.max_rate_hz,
            "unsupported_reason": m.unsupported_reason,
        })

    # neuron index lists the UI needs
    def idx_of_type(t):
        return [int(x) for x in c.by_cell_type(t)["idx"]]

    tracked = {}
    for cmd in DN_COMMANDS:
        cells = c.by_cell_type(cmd.cell_type)
        for side in ("left", "right"):
            s = cells[cells["side"].astype(str) == side]
            tracked["%s_%s" % (cmd.cell_type, side)] = [int(x) for x in s["idx"]]

    from brain.neurons.labels import functional_group
    prob = [c.idx(r) for r in functional_group("proboscis_motor")
            if int(r) in c._id2idx]

    watch = c.neurons[c.neurons["super_class"].astype(str).isin(
        ["descending", "visual_projection", "sensory"])]["idx"].to_numpy()

    neuropils = c.neurons["primary_neuropil"].fillna("").astype(str)
    npl = sorted(set(neuropils))
    npmap = {k: i for i, k in enumerate(npl)}

    meta = {
        "dataset": c.dataset,
        "manifest": c.manifest,
        "files": {"connectome.bin": conn_info, "neurons.bin": neu_info},
        "n": int(c.n),
        "nnz": int(c.w.nnz),
        "cell_types": cell_types,
        "super_classes": classes,
        "sides": sides,
        "neuropils": npl,
        "neuropil_code": [int(npmap[x]) for x in neuropils],
        "lif": LIF.as_dict(),
        "looming_tuning": vars(LoomingTuning()),
        "receptive_fields": rfs,
        "modalities": modalities,
        "dn_commands": [
            {"cell_type": d.cell_type, "channel": d.channel,
             "behaviour": d.behaviour, "laterality": d.laterality,
             "citation": d.citation, "doi": d.doi} for d in DN_COMMANDS
        ],
        "dn_tracked": tracked,
        "channel_half_max_hz": CHANNEL_HALF_MAX_HZ,
        "proboscis_motor_idx": [int(x) for x in prob],
        "watch_idx": [int(x) for x in watch],
        "lesionable": {t: idx_of_type(t)
                       for t in ("LC4", "LPLC2", "DNp01", "JO-A", "JO-B")},
        "dnp01_root_ids": [int(x) for x in c.by_cell_type("DNp01")["root_id"]],
    }
    path = DATA / "meta.json"
    path.write_text(json.dumps(meta, separators=(",", ":")))
    _log("meta.json       %8.2f MB  (gzip %.2f MB)"
         % (path.stat().st_size / 1e6,
            len(gzip.compress(path.read_bytes(), 6)) / 1e6))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    c = load_connectome()
    _log("source: %s  (%d neurons, %d connections, %d synapses)"
         % (c.dataset, c.n, c.w.nnz, int(np.abs(c.w.data).sum())))

    if c.n != config.EXPECTED_NEURONS:
        raise ValueError("refusing to export: neuron count is not %d"
                         % config.EXPECTED_NEURONS)

    conn = export_connectome(c)
    neu, cell_types, classes, sides = export_neurons(c)
    export_meta(c, conn, neu, cell_types, classes, sides)

    total = sum(p.stat().st_size for p in DATA.glob("*"))
    _log("total %.2f MB raw across %d files" % (total / 1e6, len(list(DATA.glob('*')))))
    _log("done -> %s" % DATA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
