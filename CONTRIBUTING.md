# Contributing to Fruit Fly Laboratory

Thank you for helping make the project more reproducible and useful.

## Before opening a pull request

1. Keep source data, derived data, virtual environments, local outputs, and scratch experiments out of commits.
2. Preserve FlyWire root IDs and dataset versioning in every new experiment.
3. Run the relevant tests:

   ```powershell
   python -m pytest tests/test_lif_engine.py -q
   python -m compileall -q brain simulation fly visualization experiments
   ```

4. Document the stimulus population, seed, duration, parameters, and expected output for any new scientific benchmark.

## Scientific-change policy

Do not change FlyWire topology or dataset version, synaptic signs, published core LIF parameters, or a biological mechanism without evidence, a written rationale, and maintainer approval.

Keep model facts, published assumptions, project approximations, and software engineering clearly separated as described in [BIOLOGICAL_ASSUMPTIONS.md](BIOLOGICAL_ASSUMPTIONS.md).

## Pull-request expectations

- Keep changes focused and explain the user-facing/scientific effect.
- Add or update tests; never weaken an existing test just to make a change pass.
- Update `PROJECT_STATUS_TODAY.md` when a validation result or benchmark interpretation changes.
- Do not commit raw FlyWire downloads. Follow the data licence and citation terms in [DATA_SOURCES.md](DATA_SOURCES.md).

## Good first contributions

- Documentation improvements and reproducible tutorials
- Faster, deterministic tests for documented engine semantics
- Better setup/missing-data error messages
- Data-provenance checks and benchmark reporting
- Accessibility and visualization improvements that do not misrepresent simulated activity as measured biology
