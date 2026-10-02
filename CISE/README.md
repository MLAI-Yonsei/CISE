# CISE: Conformal Interval-Driven Self-Evolution

CISE is a framework for materials discovery with large language models and imperfect property predictors. It combines evolutionary structure search with candidate-specific conformal intervals to guide feedback and select materials that satisfy the target constraints.

This repository includes two search methods:

- **LLEMA:** evolutionary search using point predictions from fixed property models.
- **CISE:** evolutionary search using Gibbs conformal intervals and online density-ratio estimation. Configuration files use the method identifier `cci`.

## Installation

Requires Python 3.11 or later on Linux/POSIX.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Quick start

Run an offline demonstration with synthetic model responses and property predictions:

```bash
cise init --config configs/cci.demo.json
cise run --config configs/cci.demo.json
```

To run the LLEMA demonstration:

```bash
cise init --config configs/llema.demo.json
cise run --config configs/llema.demo.json
```

The demonstrations require no API key, GPU, pretrained models, or DFT installation. Results are written to `runs/`.

## Running a search

Prepare the external property models and calibration assets described in [Asset setup](docs/ASSETS.md). Dataset download instructions are available in [QETB data](docs/QETB.md).

Update the asset paths in `configs/cci.json` or `configs/llema.json`. Relative paths are resolved from the configuration file's directory. Set `OPENROUTER_API_KEY` in your environment and configure the model through `llm.model`.

```bash
cise doctor --config configs/cci.json
cise init --config configs/cci.json
cise run --config configs/cci.json
```

For LLEMA, use `configs/llema.json` in the same commands.

The configuration controls the task list, number of iterations, output directory, model workers, and optional DFT evaluation. The three tasks are:

| Identifier | Task |
| --- | --- |
| `wbg` | Wide-bandgap semiconductors |
| `sse` | Solid-state electrolytes |
| `pv` | Photovoltaic absorbers |

Inspect progress and results with:

```bash
cise status --config configs/cci.json
cise report --config configs/cci.json
```

Rerun `cise run` with the same configuration to resume a checkpoint. Use a new output directory after changing the code, configuration, or model assets.

## Optional DFT evaluation

Quantum ESPRESSO evaluation is enabled through `qe.enabled`. It requires the configured executables, pseudopotentials, and reference assets. See [Asset setup](docs/ASSETS.md) for requirements. Unmeasured candidates remain unresolved in reports.

Use one controller per output directory. API requests are cached for resume, but the package does not enforce a monetary budget.

See [Method](docs/METHOD.md) for interval, selection and duplicate-handling semantics.

## Development

```bash
pytest
ruff check src tests scripts
python -m build
python -m twine check dist/*
python scripts/check_release.py
```

## Repository structure

```text
src/cise/cci/         CISE search and conformal inference
src/cise/llema/       LLEMA search
src/cise/shared/      Structure edits, prompts, archives, and features
src/cise/workers/     Property-model adapters
src/cise/validation/  Quantum ESPRESSO evaluation
configs/             Search configurations and offline demonstrations
docs/                Setup and method documentation
tests/               Automated tests
```

## License

Released under the [MIT License](LICENSE). External datasets, model weights, and dependencies are subject to their respective licenses. See [Third-party notices](THIRD_PARTY_NOTICES.md).
