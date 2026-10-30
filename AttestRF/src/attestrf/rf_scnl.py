"""Dataset helpers for staged RF-SCNL temporal measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


RF_SCNL_IDENTITIES = (
    "Device1",
    "Device2",
    "Device4",
    "Device5",
    "Device6",
    "Device8",
    "Device9",
    "Device10",
)
RF_SCNL_CLASS_MAP = {
    identity: index for index, identity in enumerate(RF_SCNL_IDENTITIES)
}


class RFSCNLCanonicalDataset(Dataset):
    """Read a staged ``signals.npy`` array with its immutable row manifest."""

    def __init__(
        self,
        root: str | Path,
        indices: Sequence[int] | None = None,
        transform: Callable | None = None,
        return_metadata: bool = True,
    ):
        self.root = Path(root).resolve()
        self.manifest = pd.read_parquet(self.root / "manifest.parquet")
        required = {
            "sample_id",
            "day",
            "device",
            "role",
            "offset",
            "class_index",
            "content_sha256",
            "segment_sha256",
        }
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"RF-SCNL manifest lacks columns: {sorted(missing)}")
        if self.manifest.sample_id.duplicated().any():
            raise ValueError("RF-SCNL sample IDs are not unique")
        expected = self.manifest.device.map(RF_SCNL_CLASS_MAP)
        if expected.isna().any() or not np.array_equal(
            expected.to_numpy(np.int64), self.manifest.class_index.to_numpy(np.int64)
        ):
            raise ValueError("RF-SCNL class indices violate the frozen identity order")
        self.signals = np.load(
            self.root / "signals.npy", mmap_mode="r", allow_pickle=False
        )
        if self.signals.ndim != 3 or self.signals.shape[1:] != (2, 128):
            raise ValueError(f"invalid RF-SCNL signal shape: {self.signals.shape}")
        if len(self.signals) != len(self.manifest):
            raise ValueError("RF-SCNL signal and manifest row counts differ")
        offsets = self.manifest.offset.to_numpy(np.int64)
        if not np.array_equal(offsets, np.arange(len(offsets), dtype=np.int64)):
            raise ValueError("RF-SCNL canonical offsets must be dense and ordered")
        self.indices = (
            np.arange(len(self.manifest), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if len(self.indices) and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= len(self.manifest)
        ):
            raise IndexError("RF-SCNL dataset index lies outside the manifest")
        self.transform = transform
        self.return_metadata = bool(return_metadata)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        manifest_index = int(self.indices[int(item)])
        row = self.manifest.iloc[manifest_index]
        signal = np.array(self.signals[int(row.offset)], dtype=np.float32, copy=True)
        if self.transform is not None:
            signal = self.transform(signal)
        tensor = torch.as_tensor(signal, dtype=torch.float32)
        target = int(row.class_index)
        if not self.return_metadata:
            return tensor, target
        metadata = row.to_dict()
        metadata["manifest_index"] = manifest_index
        return tensor, target, metadata
