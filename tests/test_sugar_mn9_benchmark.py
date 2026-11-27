from brain.neurons.registry import load_connectome
from simulation.benchmark_sugar_mn9 import (
    MN9_ROOT_ID,
    REFERENCE_SUGAR_GRN_ROOT_IDS_V630,
    benchmark_population,
)


def test_reference_sugar_population_has_twenty_v783_members():
    connectome = load_connectome()
    present, missing = benchmark_population(connectome)

    assert len(REFERENCE_SUGAR_GRN_ROOT_IDS_V630) == 21
    assert len(present) == 20
    assert missing == (720575940620900446,)
    assert connectome.idx(MN9_ROOT_ID) == 138947
