# Dataset adapters

The adapters consume NumPy signals plus a Parquet manifest. Every row has a
unique `sample_id` and identifies its stored array location. Use dataset
provider metadata to retain physical transmitter, receiver, day, and capture
groups during splitting. Repeated or sequence-matched observations must not
cross training and evaluation roles.

Preprocessing and role assignment are explicit caller responsibilities. The
adapters validate storage/index consistency but do not establish statistical
independence or authenticate a transmitter identity.

## WiSig

```text
dataset/
  manifest.parquet
  shards/
    shard_00000.npy
    shard_00001.npy
```

Each shard has shape `[N, 2, L]`, containing real I/Q channels. Required
manifest columns are `sample_id`, `tx_id_raw`, `rx_id_raw`, `day_id`, `variant`,
`shard_id`, `offset`, `channels`, and `length`. Class indices follow the sorted
string transmitter IDs over the complete manifest, including when a subset
of row indices is requested.

```python
from attestrf.wisig import GlobalIQNormalize, WiSigCanonicalDataset

data = WiSigCanonicalDataset(
    "data/wisig", transform=GlobalIQNormalize(), return_metadata=True,
)
signal, identity_index, metadata = data[0]
```

`load_wisig_source_run` reads a saved source run containing
`resolved_config.yaml`, `split.json`, `training_label_maps.json`, and
`checkpoints/best.pt`. The checkpoint contains the model state dictionary
under `model`. Pass `canonical_root` explicitly to select your local dataset
directory. Training label maps must match the prototype ordering.

## RF Fingerprinting Migration

The directory layout matches WiSig, with shards of shape `[N, 2, 1024]`.
Required manifest columns are `sample_id`, `transmitter`, `capture_id`,
`shard_id`, `offset`, `channels`, `length`, `role`, `payload_stage`, and
`global_class_index`. Transmitter names `T01` through `T30` map to class
indices 0 through 29. Preserve receiver, distance, and matched-sequence
metadata in additional columns for group-based splits.

```python
from attestrf.migration import MigrationCanonicalDataset
from attestrf.signal import IQTransform

data = MigrationCanonicalDataset("data/migration", transform=IQTransform())
signal, identity_index, metadata = data[0]
```

## RF-SCNL

```text
dataset/
  manifest.parquet
  signals.npy
```

`signals.npy` has shape `[N, 2, 128]`. Manifest rows occur in array order,
with dense offsets from 0 to N-1. Required columns are `sample_id`, `day`,
`device`, `role`, `offset`, `class_index`, `content_sha256`, and
`segment_sha256`. The digest fields record provenance; the loader does not
recompute them. Use the eight-device order in `RF_SCNL_IDENTITIES` to assign
class indices.

```python
from attestrf.rf_scnl import RFSCNLCanonicalDataset
from attestrf.rf_scnl_magnitude import magnitude_ltf

data = RFSCNLCanonicalDataset("data/rf_scnl", transform=magnitude_ltf)
envelope, identity_index, metadata = data[0]
```

The magnitude transform standardizes both I/Q channels with one mean and
scale before forming the envelope. Its output shape is `[1, 128]`, for
`MagnitudeIdentificationModel`.
