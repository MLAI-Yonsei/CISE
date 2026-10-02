# External model and calibration assets

Raw data, trained weights and calibration binaries are not distributed. QETB download instructions are in [QETB.md](QETB.md). Paths are relative to the configuration file; model interpreters can use separate environments.

## Proxy models

`surrogates.mgt_bundle` provides the MGT adapter layout:

```text
repos/MGT/models/...
repos/MGT/config/finetune.yml
repos/MGT/ckpt/finetuned/gap pbe/gap pbe_checkpoint_best.pt
upstream_preprocessing/cif2dataset_finetune_megnet.py
mgt_normalizer.json
mgt_validation.json
```

Its Python interpreter requires torch, torch_geometric, jarvis-tools, pymatgen, PyYAML and the model's dependencies. The adapter uses the supplied inverse normalizer and requires `valid_for_comparison=true` in the validation metadata. This declaration must reflect actual model validation.

`surrogates.alignn_root` contains `alignn/` Python code and the external archives `alignn/jv_formation_energy_peratom_alignn.zip` and `alignn/jv_epsx_alignn.zip`. Its interpreter requires torch, DGL, jarvis-tools and the upstream ALIGNN dependencies. `cise doctor` checks local dependencies and paths without sending requests.

## CISE calibration

`cci.calibration_dir` contains `<property>_scale.joblib`, `<property>_residuals.json`, and the selected task JSON files. Properties are `band_gap`, `formation_energy` and `dielectric_constant`. QR-offset metadata must reference matching local quantile model files and can declare their SHA256 hashes. Optional fixed-basis and training metadata are preserved. Install any external estimator classes required by serialized models. Only load trusted joblib artifacts.

To import assets from a compatible run with an `interval/` folder:

```bash
cise import-calibration --source /path/to/source-run --destination /path/to/calibration
```

For externally prepared manifest rows, `cise calibrate --manifest /path/to/prepared.jsonl --destination /path/to/calibration` creates ordinary scale/absolute-score assets. It expects nonempty string IDs/groups, a property and split (`fit`, `dre_source`, `calibration`, `diagnostic`), and 27 structure features or a CIF path. Fit/calibration/diagnostic rows also supply finite prediction and truth values. Each property requires fit and DRE rows plus at least 19 calibration rows. The standard `quantile_offset=false` template can use this output; QR models are prepared externally.

Prepare datasets, splits and trained models separately before running `init` or `run`.

## Optional DFT

`qe.enabled=true` requires `pw.x`, `ph.x`, GBRV PBEsol pseudopotentials, protocol-matched elemental references and source-replay evidence. The included worker implements fixed-structure QETB PBEsol reconstruction. Models, labels and tensor-axis/protocol definitions must match the intended comparison. Leave QE disabled to run search alone.
