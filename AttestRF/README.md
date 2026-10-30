# AttestRF

AttestRF performs reference-assisted RF fingerprint identification across
receivers. Authenticated devices estimate a feature correction and its
variation across devices. A separate reference group checks that range, and
the classifier reports an identity only when its class margin stays positive
over the complete range.

This repository contains the inference and reference-fitting core, encoders,
dataset adapters, numerical checks, and executable synthetic examples. Public
RF datasets, trained weights, and experiment outputs are obtained separately;
running the synthetic examples does not reproduce the paper's dataset results.

## Installation

Use Python 3.10 or newer and install PyTorch for your intended CPU/CUDA setup.
From this directory, run:

```bash
python -m pip install -e '.[test]'
python scripts/check_install.py
python -m pytest -q
```

The core depends on NumPy and PyTorch. Pandas and PyArrow support the dataset
manifests; PyYAML supports loading saved source-run configurations.
If the listed dependencies and setuptools are already installed, an offline
editable installation is also supported:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

## Run the complete decision rule

```bash
python examples/reference_identification.py
```

The example fits a scalar action using center references, obtains an interval
from separate range references, and evaluates independent checking devices.
Their accuracy is measured against the complete source identity table. It
then reports candidate identities only when both the reference check and the
continuous interval margin pass. A second case demonstrates that a failed
reference check withholds every candidate decision.

For the additive and rank-one numerical calculations alone, run:

```bash
python examples/minimal_certificate.py
```

The finite grid in that example is a numerical comparison. The continuous
margin routine evaluates the whole scalar interval analytically, using
endpoints and applicable stationary points. Its margin is the unnormalized
linear score gap. The sign agrees with the cosine classifier; the magnitude
must not be compared to a normalized cosine-margin magnitude.

## Use with a trained model

1. Train the encoder, source prototypes, and action using source recordings.
   `IdentificationModel` provides the two-channel I/Q model, while
   `MagnitudeIdentificationModel` provides the RF-SCNL magnitude model.
2. At the target receiver, encode authenticated references and divide their
   physical identities into disjoint center, range, and checking groups.
   Keep those roles fixed across measurement strata.
3. Call `fit_reference_stratum` for each receiver/day or distance stratum.
   Supply the complete, unnormalized source prototype table for reference
   fitting and checking accuracy. Checking devices do not alter the interval.
4. Apply `evaluate_reference_gate` using thresholds fixed with source data.
   It supports overall/per-stratum containment and optional residual and
   minimum-accuracy criteria.
5. Call `identify_with_reference_check` with the candidate-device prototypes.
   A returned identity of `-1` means that the decision is withheld.

The reference labels must come from a separate authentication mechanism. The
continuous guarantee concerns decision stability under the learned feature
action and the supplied interval; it does not establish that every physical
receiver change belongs to that family. The independent reference check
assesses agreement across the measured devices.

Reference-rule examples are in [configs/reference_rules.yaml](configs/reference_rules.yaml).
The dataset formats are described in [docs/data.md](docs/data.md). The API
accepts no test-device labels during reference fitting or identification.

## Included modules

| Module | Purpose |
|---|---|
| `attestrf.core` | Exact scalar-interval margin and finite action designs |
| `attestrf.references` | Disjoint reference fitting, independent checks, final decision |
| `attestrf.standalone_landmarks` | Frozen-model action fitting and calibration envelopes |
| `attestrf.standalone_inference` | I/Q identification model and invertible feature actions |
| `attestrf.rf_scnl_magnitude` | Magnitude preprocessing and identification model |
| `attestrf.stability_radius` | Nearest scalar-action identity boundary |
| `attestrf.wisig`, `attestrf.migration`, `attestrf.rf_scnl` | Dataset adapters |
| `attestrf.baselines`, `attestrf.attacks`, `attestrf.signal` | Baseline components and signal transformations |

## Data and weights

Obtain WiSig, RF Fingerprinting Migration, and RF-SCNL from their dataset
providers under their respective access terms. Dataset adapters read staged
NumPy arrays and Parquet manifests; they do not download, redistribute, or
automatically split the raw datasets. The repository includes no raw signals
or trained checkpoints. Load only checkpoints you trust: PyTorch checkpoint
loading may deserialize Python objects.

## Rights

Copyright Yikai Gao and Hui Liu. All rights reserved; see [LICENSE](LICENSE).
