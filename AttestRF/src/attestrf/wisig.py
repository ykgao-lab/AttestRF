"""Manifest-based WiSig signal loading and source-model checkpoint loading."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from .standalone_inference import IdentificationModel


class GlobalIQNormalize:
    """Normalize one I/Q example using one mean and scale over both channels."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = float(epsilon)

    def __call__(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != 2:
            raise ValueError(f"expected I/Q input [2,L], found {tuple(array.shape)}")
        scale = max(float(array.std()), self.epsilon)
        return ((array - float(array.mean())) / scale).astype(np.float32, copy=False)


class WiSigCanonicalDataset(Dataset):
    """Read fixed-shape WiSig shards and their sample manifest."""

    REQUIRED_COLUMNS = {
        "sample_id",
        "tx_id_raw",
        "rx_id_raw",
        "day_id",
        "variant",
        "shard_id",
        "offset",
        "channels",
        "length",
    }

    def __init__(
        self,
        root: str | Path,
        indices: Sequence[int] | None = None,
        transform: Callable | None = None,
        return_metadata: bool = False,
        max_open_shards: int = 8,
    ):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.parquet"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"canonical manifest not found: {manifest_path}")
        self.manifest = pd.read_parquet(manifest_path)
        missing = self.REQUIRED_COLUMNS - set(self.manifest.columns)
        if missing:
            raise ValueError(f"WiSig manifest lacks columns: {sorted(missing)}")
        if self.manifest.sample_id.duplicated().any():
            raise ValueError("WiSig manifest sample IDs are not unique")
        if (self.manifest[["shard_id", "offset"]] < 0).any().any():
            raise ValueError("WiSig manifest has negative shard offsets")
        self.indices = (
            np.arange(len(self.manifest), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if len(self.indices) and (
            self.indices.min() < 0 or self.indices.max() >= len(self.manifest)
        ):
            raise IndexError("WiSig dataset index is outside the manifest")
        self.transform = transform
        self.return_metadata = bool(return_metadata)
        self.max_open_shards = max(1, int(max_open_shards))
        self._shards: OrderedDict[int, np.ndarray] = OrderedDict()
        identities = sorted(self.manifest.tx_id_raw.astype(str).unique())
        self.tx_map = {identity: index for index, identity in enumerate(identities)}

    def __len__(self) -> int:
        return len(self.indices)

    def _load_shard(self, shard_id: int) -> np.ndarray:
        shard_id = int(shard_id)
        if shard_id in self._shards:
            array = self._shards.pop(shard_id)
            self._shards[shard_id] = array
            return array
        path = self.root / "shards" / f"shard_{shard_id:05d}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"canonical shard not found: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[1] != 2:
            raise ValueError(f"invalid WiSig shard shape at {path}: {array.shape}")
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
        target = self.tx_map[str(row.tx_id_raw)]
        if not self.return_metadata:
            return tensor, target
        metadata = row.to_dict()
        metadata["manifest_index"] = manifest_index
        return tensor, target, metadata


def _read_mapping(path: Path, kind: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"saved run lacks {path.name}: {path}")
    value = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        if kind == "yaml"
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return value


def load_wisig_source_run(
    run_dir: str | Path,
    device: str = "auto",
    canonical_root: str | Path | None = None,
):
    """Load a saved encoder/action model without depending on its training code."""

    run = Path(run_dir).resolve()
    config = _read_mapping(run / "resolved_config.yaml", "yaml")
    split = _read_mapping(run / "split.json", "json")
    maps = _read_mapping(run / "training_label_maps.json", "json")
    checkpoint_path = run / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"saved run lacks best checkpoint: {checkpoint_path}")

    requested = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if device == "auto" and not torch.cuda.is_available():
        requested = "cpu"
    device_object = torch.device(requested)
    model_config = dict(config.get("model", {}))
    transport = str(model_config.get("transport", "lowrank"))
    model = IdentificationModel(
        len(maps["tx_map"]),
        len(maps["rx_map"]),
        embedding_dim=int(model_config.get("embedding_dim", 64)),
        receiver_code_dim=int(model_config.get("receiver_code_dim", 1)),
        transport=transport,
        temperature=float(model_config.get("temperature", 0.1)),
        receiver_code_init_std=float(model_config.get("receiver_code_init_std", 0.0)),
    ).to(device_object)
    checkpoint = torch.load(
        checkpoint_path, map_location=device_object, weights_only=False
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise ValueError(f"invalid checkpoint structure: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"], strict=True)

    canonical = Path(
        canonical_root
        if canonical_root is not None
        else config["dataset"]["canonical_root"]
    ).resolve()
    transform_spec = config.get("dataset", {}).get("eval_transform") or {}
    unsupported = set(transform_spec) - {"global_normalize", "epsilon"}
    if unsupported:
        raise ValueError(
            f"unsupported WiSig evaluation transforms: {sorted(unsupported)}"
        )
    transform = (
        GlobalIQNormalize(float(transform_spec.get("epsilon", 1e-8)))
        if bool(transform_spec.get("global_normalize", False))
        else None
    )
    dataset = WiSigCanonicalDataset(
        canonical, transform=transform, return_metadata=False
    )
    return (
        run,
        config,
        split,
        maps,
        model.eval(),
        device_object,
        canonical,
        dataset.manifest,
        dataset,
    )


def stack_signals(dataset: Dataset, indices: Sequence[int]) -> torch.Tensor:
    """Materialize an ordered batch of canonical I/Q examples."""

    return torch.stack([dataset[int(index)][0] for index in indices])
