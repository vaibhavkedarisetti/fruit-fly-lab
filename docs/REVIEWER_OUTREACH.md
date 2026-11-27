# Reviewer outreach draft

Use this as a starting point when contacting a computational-neuroscience
reviewer, FlyWire contributor, or *Drosophila* circuit specialist. Replace the
bracketed fields and send only after selecting recipients yourself.

## Short email

**Subject:** Request for technical/scientific feedback on an open FlyWire v783 LIF simulator

Hello [Name],

I am preparing an open-source, interactive simulator built from the FlyWire
FAFB v783 connectome and the published Shiu et al. whole-brain LIF model. The
project builds its graph from checksummed primary data, provides stimulus-to-
readout demonstrations, and now includes an independent Brian2 validation for
an exact sugar→MN9 benchmark, L4/L5 stability cases, and refractory-event
semantics.

I would value your feedback on [sensory encoder assumptions / descending-neuron
readout mapping / the validation protocol / a specific circuit]. I am not
asking you to endorse its biological claims; I am specifically seeking critique
of documented assumptions and reproducibility.

Repository: [insert the authorised repository URL after licensing is resolved]

Key reading: `docs/TECHNICAL_VALIDATION_REPORT.md` and
`BIOLOGICAL_ASSUMPTIONS.md`.

Thank you for considering it,

Vaibhav Kedarisetti

## Suggested review questions

1. Is the selected FlyWire population appropriate for the stated stimulus?
2. Are the sensory drive and descending-neuron mapping labels sufficiently
   separated from measured biology?
3. Are important biological limitations missing from the documentation?
4. Is the Brian2 validation protocol fair and reproducible?
5. Which experiment would most improve scientific usefulness without silently
   altering topology, signs, or published parameters?
