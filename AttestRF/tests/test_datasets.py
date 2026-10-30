"""Storage and identity-order checks using small synthetic dataset files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from attestrf.migration import MigrationCanonicalDataset
from attestrf.rf_scnl import RFSCNLCanonicalDataset
from attestrf.wisig import WiSigCanonicalDataset


def test_wisig_subset_preserves_complete_manifest_identity_order(tmp_path) -> None:
    (tmp_path / "shards").mkdir()
    signals = np.arange(32, dtype=np.float32).reshape(2, 2, 8)
    np.save(tmp_path / "shards" / "shard_00000.npy", signals)
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "tx_id_raw": ["tx_z", "tx_a"],
            "rx_id_raw": ["rx_1"] * 2,
            "day_id": ["day_1"] * 2,
            "variant": ["raw"] * 2,
            "shard_id": [0, 0],
            "offset": [0, 1],
            "channels": [2, 2],
            "length": [8, 8],
        }
    ).to_parquet(tmp_path / "manifest.parquet", index=False)
    data = WiSigCanonicalDataset(tmp_path, indices=[0], return_metadata=True)
    signal, identity, metadata = data[0]
    assert identity == 1 and metadata["sample_id"] == "a"
    torch.testing.assert_close(signal, torch.from_numpy(signals[0]))


def test_migration_uses_fixed_transmitter_labels(tmp_path) -> None:
    (tmp_path / "shards").mkdir()
    np.save(
        tmp_path / "shards" / "shard_00000.npy", np.ones((1, 2, 1024), dtype=np.float32)
    )
    manifest = pd.DataFrame(
        {
            "sample_id": ["a"],
            "transmitter": ["T30"],
            "capture_id": ["capture"],
            "shard_id": [0],
            "offset": [0],
            "channels": [2],
            "length": [1024],
            "role": ["source_train"],
            "payload_stage": ["source"],
            "global_class_index": [29],
        }
    )
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)
    assert MigrationCanonicalDataset(tmp_path)[0][1] == 29
    manifest.loc[0, "global_class_index"] = 0
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)
    with pytest.raises(ValueError, match="mapping"):
        MigrationCanonicalDataset(tmp_path)


def test_rf_scnl_manifest_must_match_dense_array_order(tmp_path) -> None:
    np.save(tmp_path / "signals.npy", np.ones((2, 2, 128), dtype=np.float32))
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "day": ["day1", "day2"],
            "device": ["Device1", "Device10"],
            "role": ["source_train", "target_check"],
            "offset": [0, 1],
            "class_index": [0, 7],
            "content_sha256": ["synthetic"] * 2,
            "segment_sha256": ["synthetic"] * 2,
        }
    )
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)
    assert RFSCNLCanonicalDataset(tmp_path)[1][1] == 7
    manifest["offset"] = [1, 0]
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)
    with pytest.raises(ValueError, match="dense and ordered"):
        RFSCNLCanonicalDataset(tmp_path)
