"""
Central configuration: dataset locations and canonical dataset identity.

Nothing in this project may load neural data from anywhere other than the
paths declared here. See DATA_SOURCES.md for provenance.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Source dataset (read-only, never modified by this project) -------------
# FlyWire FAFB v783 public release, downloaded from https://codex.flywire.ai/api/download
DEFAULT_FLYWIRE_DIR = Path(r"D:\Fruitfly\FlyWire Brain Dataset (FAFB v783)")
FLYWIRE_DIR = Path(os.environ.get("FLYWIRE_V783_DIR", DEFAULT_FLYWIRE_DIR))

# Canonical identity of the dataset this project is built against.
DATASET_NAME = "FlyWire FAFB"
DATASET_VERSION = "783"
DATASET_URL = "https://codex.flywire.ai/api/download"

# Reference counts published for FlyWire FAFB v783. The build refuses to run
# if the loaded data does not match EXPECTED_NEURONS.
EXPECTED_NEURONS = 139255

# FlyWire/CAVE segmentation IDs for FAFB v783 all share this numeric prefix.
ROOT_ID_PREFIX = "720575940"

# --- Source files -----------------------------------------------------------
SRC = {
    "neurons":                FLYWIRE_DIR / "neurons.csv.gz",
    "classification":         FLYWIRE_DIR / "classification.csv.gz",
    "consolidated_cell_types":FLYWIRE_DIR / "consolidated_cell_types.csv.gz",
    "visual_neuron_types":    FLYWIRE_DIR / "visual_neuron_types.csv.gz",
    "coordinates":            FLYWIRE_DIR / "coordinates.csv.gz",
    "connections":            FLYWIRE_DIR / "connections_princeton.csv.gz",
    "column_assignment":      FLYWIRE_DIR / "column_assignment.csv.gz",
    "cell_stats":             FLYWIRE_DIR / "cell_stats.csv.gz",
    "names":                  FLYWIRE_DIR / "names.csv.gz",
    "labels":                 FLYWIRE_DIR / "labels.csv.gz",
}

# --- Derived (built) artefacts ---------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data"
DERIVED_DIR   = DATA_DIR / "derived"
METADATA_DIR  = DATA_DIR / "metadata"
OUTPUT_DIR    = PROJECT_ROOT / "simulation" / "outputs"

CONNECTOME_NPZ = DERIVED_DIR / "connectome_v783.npz"
NEURON_INDEX   = DERIVED_DIR / "neuron_index_v783.csv.gz"
BUILD_MANIFEST = METADATA_DIR / "build_manifest.json"
CHECKSUM_FILE  = METADATA_DIR / "flywire_v783_checksums.txt"

for _d in (DERIVED_DIR, METADATA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require_source_files() -> None:
    """Fail loudly if the real FlyWire files are absent. Never silently substitute."""
    missing = [f"{k}: {p}" for k, p in SRC.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Real FlyWire FAFB v783 source files are missing. This project will not "
            "run on substituted or synthetic data.\nMissing:\n  " + "\n  ".join(missing)
            + f"\n\nExpected under: {FLYWIRE_DIR}\nDownload from: {DATASET_URL}"
        )
