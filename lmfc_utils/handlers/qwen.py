from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import einops
import torch

from .base_handler import BaseHandler, FeatureData, PackerBase
from .utils import inspect_structure, load_tensor


@dataclass
class QwenFeatures(FeatureData):
    """Parsed and organized Qwen features.

    Attributes:
        caches: List of (k_cache, v_cache) tuples, sorted by layer index.
        output: Output tensor from the last layer (or None if not present).
        last_layer_idx: Index of the last layer that contains output feature.
        layer_indices: Sorted list of layer indices.
    """

    caches: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    output: Optional[torch.Tensor] = None
    last_layer_idx: int = -1
    layer_indices: List[int] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        """Number of layers with caches."""
        return len(self.caches)

    @property
    def cache_shape(self) -> Tuple[int, ...]:
        """Shape of a single cache tensor (assuming all same shape)."""
        if self.caches:
            return tuple(self.caches[0][0].shape)
        return ()

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Shape of the output feature tensor."""
        if self.output is not None:
            return tuple(self.output.shape)
        return ()

    def summary(self) -> str:
        """Return a summary string of the features."""
        lines = [
            "-" * 60,
            "Qwen Features Summary",
            "-" * 60,
            f"Number of layers: {self.num_layers}",
            f"Layer indices: {self.layer_indices}",
            f"Last layer index: {self.last_layer_idx}",
            f"Cache shape: {self.cache_shape}",
            f"Output shape: {self.output_shape}",
            f"Data type: {self.dtype}",
            "-" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert features back to the original dict structure.

        Returns:
            Dict with structure: {
                'model.layers.0': (k_cache, v_cache),
                ...
                'model.layers.N': (feature, (k_cache, v_cache))  # for last layer
            }
        """
        result = {}
        for i, layer_idx in enumerate(self.layer_indices):
            key = f"layer.{layer_idx}"
            k_cache, v_cache = self.caches[i]

            result[f"{key}.k_cache"] = k_cache
            result[f"{key}.v_cache"] = v_cache

        assert layer_idx == self.last_layer_idx and self.output is not None
        result[f"{key}.output"] = self.output
        return result


class IndividualPacker(PackerBase):
    """将所有的k-cache/v-cache/hidden_state都单独打包

    IndividualPacker:
        layer.0.k_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.0.v_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.1.k_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.1.v_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.2.k_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.2.v_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.3.k_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.3.v_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.4.k_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.4.v_cache: torch.Size([1, 4, N, 128]) -> torch.Size([1, 1, N, 512])
        layer.4.output: torch.Size([1, N, 3584]) -> torch.Size([1, 1, N, 3584])
    """

    def _pack_impl(self, features: QwenFeatures):
        packed = {}
        for layer_idx, (k_cache, v_cache) in zip(features.layer_indices, features.caches):
            packed[f"layer.{layer_idx}.k_cache"] = (
                einops.rearrange(k_cache, "1 nh nt hd -> 1 1 nt (nh hd)"),
                k_cache.shape,
            )
            packed[f"layer.{layer_idx}.v_cache"] = (
                einops.rearrange(v_cache, "1 nh nt hd -> 1 1 nt (nh hd)"),
                v_cache.shape,
            )

        assert layer_idx == features.last_layer_idx
        packed[f"layer.{layer_idx}.output"] = (
            einops.rearrange(features.output, "1 nt d -> 1 1 nt d"),
            features.output.shape,
        )

        self.log("  IndividualPacker:")
        for name, (tensor, ori_shape) in packed.items():
            self.log(f"    {name}: {ori_shape} -> {tensor.shape}")
        return packed

    def _unpack_impl(self, tensors: Dict[str, tuple]) -> QwenFeatures:
        caches = []
        for layer_idx in self._last_reference.layer_indices:
            k_cache, meta = tensors[f"layer.{layer_idx}.k_cache"]
            k_cache = einops.rearrange(k_cache, "1 1 nt (nh hd) -> 1 nh nt hd", nh=meta["ori_shape"][1])

            v_cache, meta = tensors[f"layer.{layer_idx}.v_cache"]
            v_cache = einops.rearrange(v_cache, "1 1 nt (nh hd) -> 1 nh nt hd", nh=meta["ori_shape"][1])

            caches.append((k_cache, v_cache))

        assert layer_idx == self._last_reference.last_layer_idx
        output, _ = tensors[f"layer.{layer_idx}.output"]
        output = einops.rearrange(output, "1 1 nt d -> 1 nt d")

        self.log("  IndividualUnPacker:")
        for layer_idx, (k_cache, v_cache) in zip(self._last_reference.layer_indices, caches):
            self.log(f"    layer.{layer_idx}.k_cache: {k_cache.shape}")
            self.log(f"    layer.{layer_idx}.v_cache: {v_cache.shape}")
        self.log(f"    layer.{layer_idx}.output: {output.shape}")

        return QwenFeatures(
            model_type=self._last_reference.model_type,
            source_path=self._last_reference.source_path,
            dtype=self._last_reference.dtype,
            last_layer_idx=self._last_reference.last_layer_idx,
            layer_indices=self._last_reference.layer_indices,
            caches=caches,
            output=output,
        )


class QwenHandler(BaseHandler):
    """Handler for Qwen model features."""

    model_type = "qwen"
    SUPPORTED_STRATEGIES = {
        "individual": IndividualPacker,
    }

    def parse(self, path: str) -> QwenFeatures:
        """Parse Qwen features from a .zst file."""
        # Load raw data
        data = load_tensor(path)

        if self.log_fn is not None:
            self.log("\nOriginal data structure:")
            inspect_structure(data, prefix="root", print_fn=self.log_fn)

        output = data["output"]
        k_caches = data["key"]
        v_caches = data["value"]
        layer_indices = [idx for idx in range(len(k_caches))]

        return QwenFeatures(
            model_type=self.model_type,
            source_path=path,
            dtype=output.dtype,
            caches=[(k, v) for k, v in zip(k_caches, v_caches)],
            output=output,
            last_layer_idx=layer_indices[-1],
            layer_indices=layer_indices,
            metadata={"original_keys": list(data.keys())},
        )

    def restore_format(self, features: QwenFeatures) -> Dict[str, torch.Tensor]:
        """Restore features to a dictionary as the original format."""
        restored = {
            "output": features.output,
            "key": [cache[0] for cache in features.caches],
            "value": [cache[1] for cache in features.caches],
        }
        inspect_structure(restored, prefix="Restored Feature Format", print_fn=self.log_fn)
        return restored

    def _is_last_layer(self, layer_data: Any) -> bool:
        """Check if layer_data is the last layer."""
        if not isinstance(layer_data, (tuple, list)) or len(layer_data) != 2:
            return False

        # Check if second item is also a tuple (kv_cache)
        if isinstance(layer_data[1], (tuple, list)) and len(layer_data[1]) == 2:
            if isinstance(layer_data[1][0], torch.Tensor) and isinstance(layer_data[1][1], torch.Tensor):
                if isinstance(layer_data[0], torch.Tensor):
                    return len(layer_data[0].shape) == 3
        return False
