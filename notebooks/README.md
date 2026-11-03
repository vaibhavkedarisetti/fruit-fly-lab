# Guided notebooks

These notebooks are guided, reproducible entry points to the supported
experiments. They require the local FlyWire FAFB v783 data and a built derived
connectome; see the root [README](../README.md).

| Notebook | Demonstrates | Typical runtime |
| --- | --- | --- |
| `01_looming_escape.ipynb` | Looming stimulus → LC4/LPLC2 → DNp01 readout | a few seconds |
| `02_sugar_to_mn9.ipynb` | Exact published sugar-GRN population → MN9 benchmark | about a minute |
| `03_lesion_controls.ipynb` | Looming, receding, static, and LC4/LPLC2 lesion controls | a few minutes |

Run them from the repository root with Jupyter so project imports resolve:

```powershell
python -m pip install jupyter
python -m jupyter lab
```

Outputs are written only to the ignored `simulation/outputs/` directory.
