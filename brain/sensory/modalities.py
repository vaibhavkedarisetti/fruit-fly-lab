"""
Registry of stimuli the laboratory can deliver, and the REAL FlyWire neurons
each one drives.

Every entry names the neurons by their real FlyWire identity -- either a
consolidated cell type, or a curated FlyWire community-label group -- and cites
the experimental work that justifies the assignment.

A stimulus the connectome cannot support is present here with
`supported = False` and an explanation. The user interface displays those as
"Not currently modeled" rather than faking a response. That is deliberate:
FlyWire FAFB v783 is a BRAIN dataset, so sensory neurons of the thorax, abdomen
and legs -- whose axons terminate in the ventral nerve cord -- are simply not in
the data.

PROVENANCE
  A. REAL DATA  : the neuron sets (cell types / labels) and their counts
  B. PUBLISHED  : each `citation` -- what the neurons respond to
  C. APPROX     : intensity -> firing rate is linear up to `max_rate_hz`,
                  following the reference model's Poisson activation scheme
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Modality:
    key: str
    label: str                       # UI label
    group: str                       # UI grouping
    cell_types: tuple = ()           # consolidated primary_type values
    label_group: str = None          # curated FlyWire-label functional group
    description: str = ""
    citation: str = ""
    doi: str = ""
    max_rate_hz: float = 150.0       # matches Shiu et al. r_poi
    supported: bool = True
    unsupported_reason: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# VISION
# ---------------------------------------------------------------------------
VISUAL = (
    Modality(
        key="looming", label="Looming threat (throw rock)", group="Vision",
        cell_types=("LC4", "LPLC2"),
        description=(
            "An object on a collision course. LC4 encodes angular velocity and "
            "LPLC2 angular size; both converge on the Giant Fibre DNp01."),
        citation=("von Reyn et al. 2017, Nat Neurosci 20:1176; "
                  "Klapoetke et al. 2017, Nature 551:237"),
        doi="10.1038/nn.4600",
        notes=("Driven through receptive fields derived from real FlyWire "
               "column assignments. See brain/sensory/encoders.py."),
    ),
    Modality(
        key="light_photoreceptor", label="Light (photoreceptors R1-6)",
        group="Vision", cell_types=("R1-6",),
        description="Direct activation of the 8,456 real R1-6 photoreceptors.",
        citation="Rister et al. 2007, Neuron 56:155 (R1-6 -> L1/L2 pathway)",
        doi="10.1016/j.neuron.2007.09.014",
        supported=False,
        unsupported_reason=(
            "Photoreceptors are HISTAMINERGIC and their synapse onto lamina "
            "neurons is INHIBITORY (sign-inverting). FlyWire predicts only six "
            "neurotransmitters (ACh, GABA, Glu, DA, 5-HT, OA) and histamine is "
            "not among them, so R1-6 are mis-assigned as excitatory. Driving "
            "them would invert the sign of the entire early visual pathway. "
            "Modelling light this way would be wrong, so it is disabled."),
    ),
)

# ---------------------------------------------------------------------------
# CHEMOSENSATION
# ---------------------------------------------------------------------------
CHEMICAL = (
    Modality(
        key="taste_sugar", label="Sugar on proboscis", group="Taste",
        label_group="sugar_grn",
        description=(
            "Sugar-sensing gustatory receptor neurons. Drives the feeding "
            "circuit and proboscis extension."),
        citation=("Engert et al. 2022, eLife 11:e78110; "
                  "Shiu et al. 2024, Nature 634:210"),
        doi="10.7554/eLife.78110",
        notes=("These neurons carry FlyWire labels contributed by Philip Shiu "
               "(Kristin Scott Lab), the reference model's own author. Only "
               "left-side sugar GRNs are labelled in v783."),
    ),
    Modality(
        key="taste_bitter", label="Bitter on proboscis", group="Taste",
        label_group="bitter_grn",
        description=(
            "Bitter gustatory receptor neurons. Suppresses feeding and drives "
            "proboscis retraction."),
        citation="Engert et al. 2022, eLife 11:e78110",
        doi="10.7554/eLife.78110",
    ),
    Modality(
        key="odor_vinegar", label="Food odour (apple cider vinegar)",
        group="Smell",
        cell_types=("ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_VM2", "ORN_VA2"),
        description=(
            "Attractive food odour. Activates the glomeruli that mediate "
            "vinegar attraction."),
        citation=("Semmelhack & Wang 2009, Nature 459:218; "
                  "Hallem & Carlson 2006, Cell 125:143"),
        doi="10.1038/nature07983",
    ),
    Modality(
        key="odor_geosmin", label="Danger odour (geosmin)", group="Smell",
        cell_types=("ORN_DA2",),
        description=(
            "Geosmin signals harmful microbes and drives innate avoidance "
            "through a single dedicated glomerulus, DA2."),
        citation="Stensmyr et al. 2012, Cell 151:1345",
        doi="10.1016/j.cell.2012.09.046",
    ),
    Modality(
        key="odor_cva", label="Pheromone (cVA)", group="Smell",
        cell_types=("ORN_DA1",),
        description="cis-vaccenyl acetate, the male-derived social pheromone.",
        citation="Kurtovic et al. 2007, Nature 446:542",
        doi="10.1038/nature05672",
    ),
    Modality(
        key="odor_co2", label="CO2 (avoidance)", group="Smell",
        cell_types=("ORN_V",),
        description="Carbon dioxide, an innately aversive stress odour.",
        citation="Suh et al. 2004, Nature 431:854",
        doi="10.1038/nature02980",
    ),
)

# ---------------------------------------------------------------------------
# MECHANOSENSATION
# ---------------------------------------------------------------------------
MECHANICAL = (
    Modality(
        key="wind", label="Wind / air puff", group="Air & sound",
        cell_types=("JO-C", "JO-D", "JO-E", "JO-EV", "JO-EVM", "JO-EDM",
                    "JO-EVL", "JO-EVP", "JO-EDC", "JO-EDP", "JO-CA", "JO-CM",
                    "JO-CL"),
        description=(
            "Sustained antennal deflection by airflow, detected by the "
            "static-deflection subgroups of Johnston's organ."),
        citation=("Yorozu et al. 2009, Nature 458:201; "
                  "Kamikouchi et al. 2009, Nature 458:165"),
        doi="10.1038/nature07843",
    ),
    Modality(
        key="sound", label="Sound / courtship song", group="Air & sound",
        cell_types=("JO-A", "JO-B"),
        description=(
            "Antennal vibration. JO-A and JO-B are the sound-sensitive "
            "subgroups; they also project to the Giant Fibre."),
        citation="Kamikouchi et al. 2009, Nature 458:165",
        doi="10.1038/nature07843",
    ),
    Modality(
        key="touch_head", label="Touch: HEAD", group="Touch",
        cell_types=("BM_InOm", "BM_Ant", "BM_Or", "BM_FrOr", "BM_dPoOr",
                    "BM_Fr", "BM_Oc", "BM_InOc", "BM_Vt_PoOc", "BM_dOcci",
                    "BM_vOcci_vPoOr", "BM_Vib", "BM_MaPa", "BM_Hau"),
        description=(
            "Mechanosensory bristles of the head capsule: interommatidial, "
            "antennal, orbital, ocellar, frontal, vibrissae, maxillary palp "
            "and haustellum. Drives grooming."),
        citation=("Hampel et al. 2015, eLife 4:e08758; "
                  "Seeds et al. 2014, eLife 3:e02951"),
        doi="10.7554/eLife.08758",
    ),
    Modality(
        key="touch_leg_taste", label="Leg contact chemosensation (tarsus)",
        group="Touch", cell_types=("claw_tpGRN", "dorsal_tpGRN"),
        description=(
            "Tarsal gustatory receptor neurons on the foreleg: what the fly "
            "tastes when it steps on food."),
        citation="Ledue et al. 2015, Curr Biol 25:1466",
        doi="10.1016/j.cub.2015.03.020",
    ),
    Modality(
        key="touch_thorax", label="Touch: THORAX", group="Touch",
        supported=False,
        unsupported_reason=(
            "Thoracic bristle mechanosensory neurons project to the ventral "
            "nerve cord. FlyWire FAFB v783 contains the brain only, so these "
            "neurons are absent from the dataset."),
    ),
    Modality(
        key="touch_abdomen", label="Touch: ABDOMEN", group="Touch",
        supported=False,
        unsupported_reason=(
            "Abdominal mechanosensory neurons project to the ventral nerve "
            "cord and are absent from FlyWire FAFB v783 (brain only)."),
    ),
    Modality(
        key="touch_leg", label="Touch: LEG (mechanical)", group="Touch",
        supported=False,
        unsupported_reason=(
            "Leg bristle and campaniform mechanosensory neurons terminate in "
            "the ventral nerve cord and are absent from FlyWire FAFB v783. "
            "Leg CHEMOsensation is available separately: the tarsal gustatory "
            "neurons do ascend to the brain."),
    ),
)

# ---------------------------------------------------------------------------
# TEMPERATURE AND HUMIDITY
# ---------------------------------------------------------------------------
THERMAL = (
    Modality(
        key="cold", label="Cold", group="Temperature",
        cell_types=("TRN_VP2",),
        description=(
            "Cold-activated thermoreceptor neurons of the arista, projecting "
            "to glomerulus VP2."),
        citation=("Gallio et al. 2011, Cell 144:614; "
                  "Frank et al. 2015, Cell Rep 11:1345"),
        doi="10.1016/j.cell.2011.01.028",
        notes="Only 7 TRN_VP2 neurons are present in v783.",
    ),
    Modality(
        key="heat", label="Heat", group="Temperature",
        cell_types=("TRN_VP3a", "TRN_VP3b"),
        description=(
            "Hot-activated thermoreceptor neurons projecting to glomerulus "
            "VP3."),
        citation="Frank et al. 2015, Cell Rep 11:1345",
        doi="10.1016/j.celrep.2015.04.064",
        notes="Only 9 TRN_VP3 neurons are present in v783.",
    ),
    Modality(
        key="humidity", label="Humidity", group="Temperature",
        cell_types=("HRN_VP4", "HRN_VP1d", "HRN_VP5", "HRN_VP1l"),
        description="Hygrosensory neurons of the sacculus.",
        citation="Enjin et al. 2016, Curr Biol 26:1352",
        doi="10.1016/j.cub.2016.03.049",
    ),
)

ALL_MODALITIES = VISUAL + CHEMICAL + MECHANICAL + THERMAL
BY_KEY = {m.key: m for m in ALL_MODALITIES}


def resolve_neurons(modality: Modality, connectome) -> np.ndarray:
    """Simulation indices of the real neurons a modality drives."""
    if not modality.supported:
        return np.empty(0, dtype=np.int64)
    if modality.label_group:
        from brain.neurons.labels import functional_group
        rids = functional_group(modality.label_group)
        return np.array([connectome.idx(r) for r in rids
                         if int(r) in connectome._id2idx], dtype=np.int64)
    df = connectome.by_cell_types(modality.cell_types)
    return df["idx"].to_numpy(dtype=np.int64)


def census(connectome) -> list:
    """How many real neurons back each modality. Used by the UI."""
    out = []
    for m in ALL_MODALITIES:
        idx = resolve_neurons(m, connectome)
        out.append({
            "key": m.key, "label": m.label, "group": m.group,
            "supported": m.supported and len(idx) > 0,
            "n_neurons": int(len(idx)),
            "cell_types": list(m.cell_types),
            "label_group": m.label_group,
            "description": m.description,
            "citation": m.citation, "doi": m.doi, "notes": m.notes,
            "unsupported_reason": (
                m.unsupported_reason if not m.supported
                else ("No neurons of these types are present in this dataset."
                      if len(idx) == 0 else "")),
        })
    return out
