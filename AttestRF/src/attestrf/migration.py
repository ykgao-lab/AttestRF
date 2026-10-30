"""Dataset helpers for staged RF fingerprinting data."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


GLOBAL_TRANSMITTERS = tuple(f"T{index:02d}" for index in range(1, 31))
GLOBAL_CLASS_MAP = {
    transmitter: index for index, transmitter in enumerate(GLOBAL_TRANSMITTERS)
}
# Dataset-qualified aliases are provided for readable experiment code.
MIGRATION_TRANSMITTERS = GLOBAL_TRANSMITTERS
MIGRATION_CLASS_TO_INDEX = GLOBAL_CLASS_MAP


def global_class_indices(transmitters: pd.Series) -> pd.Series:
    """Map the frozen T01..T30 identity vocabulary to stable class indices."""

    values = transmitters.astype(str)
    invalid = sorted(set(values) - set(MIGRATION_CLASS_TO_INDEX))
    if invalid:
        raise ValueError(f"unknown RF Migration transmitters: {invalid}")
    return values.map(MIGRATION_CLASS_TO_INDEX).astype(np.int64)


class MigrationCanonicalDataset(Dataset):
    """Read fixed-shape NumPy shards with manifest-validated identity indices."""

    def __init__(
        self,
        root: str | Path,
        indices: Sequence[int] | None = None,
        transform: Callable | None = None,
        return_metadata: bool = True,
        max_open_shards: int = 8,
    ):
        self.root = Path(root).resolve()
        self.manifest = pd.read_parquet(self.root / "manifest.parquet")
        required = {
            "sample_id",
            "transmitter",
            "capture_id",
            "shard_id",
            "offset",
            "channels",
            "length",
            "role",
            "payload_stage",
            "global_class_index",
        }
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"migration manifest lacks columns: {sorted(missing)}")
        if self.manifest.sample_id.duplicated().any():
            raise ValueError("migration manifest sample IDs are not unique")
        if not pd.api.types.is_integer_dtype(self.manifest.global_class_index.dtype):
            raise ValueError("migration global_class_index must have an integer dtype")
        expected_indices = global_class_indices(self.manifest.transmitter)
        observed_indices = self.manifest.global_class_index.astype(np.int64)
        mismatch = observed_indices.ne(expected_indices)
        if mismatch.any():
            examples = (
                self.manifest.loc[mismatch, ["transmitter", "global_class_index"]]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "migration global_class_index violates the frozen T01..T30 mapping: "
                f"{examples}"
            )
        if (self.manifest[["shard_id", "offset"]] < 0).any().any():
            raise ValueError("migration manifest has negative shard offsets")
        self.indices = (
            np.arange(len(self.manifest), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if len(self.indices) and (
            self.indices.min() < 0 or self.indices.max() >= len(self.manifest)
        ):
            raise IndexError("migration dataset index is outside the manifest")
        self.transform = transform
        self.return_metadata = bool(return_metadata)
        self.max_open_shards = max(1, int(max_open_shards))
        self._shards: OrderedDict[int, np.ndarray] = OrderedDict()
        self.tx_map = dict(MIGRATION_CLASS_TO_INDEX)

    def __len__(self) -> int:
        return len(self.indices)

    def _load_shard(self, shard_id: int) -> np.ndarray:
        shard_id = int(shard_id)
        if shard_id in self._shards:
            array = self._shards.pop(shard_id)
            self._shards[shard_id] = array
            return array
        path = self.root / "shards" / f"shard_{shard_id:05d}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[1:] != (2, 1024):
            raise ValueError(f"invalid migration shard shape at {path}: {array.shape}")
        self._shards[shard_id] = array
        while len(self._shards) > self.max_open_shards:
            self._shards.popitem(last=False)
        return array

    def __getitem__(self, item: int):
        manifest_index = int(self.indices[int(item)])
        row = self.manifest.iloc[manifest_index]
        signal = np.array(
            self._load_shard(int(row.shard_id))[int(row.offset)],
            dtype=np.float32,
            copy=True,
        )
        if self.transform is not None:
            signal = self.transform(signal)
        tensor = torch.as_tensor(signal, dtype=torch.float32)
        target = int(row.global_class_index)
        if not self.return_metadata:
            return tensor, target
        metadata = row.to_dict()
        metadata["manifest_index"] = manifest_index
        return tensor, target, metadata
