import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import einops
import torch

from .base_handler import BaseHandler, FeatureData, PackerBase
from .utils import inspect_structure, load_tensor


@dataclass
class KimiAudioFeatures(FeatureData):
    """Parsed and organized KimiAudio features.

    Attributes:
        caches: List of (k_cache, v_cache) tuples, sorted by layer index.
        feature: Output feature tensor from the last layer (or None if not present).
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
        """Shape of the output output tensor."""
        if self.output is not None:
            return tuple(self.output.shape)
        return ()

    def summary(self) -> str:
        """Return a summary string of the features."""
        lines = [
            "-" * 60,
            "KimiAudio Features Summary",
            "-" * 60,
            f"Number of layers: {self.num_layers}",
            f"Layer indices: {self.layer_indices}",
            f"Last layer index: {self.last_layer_idx}",
            f"Cache shape: {self.cache_shape}",
            f"Output shape: {self.output_shape}",
            f"Data type: {self.dtype}",
            f"Has output output: {self.output is not None}",
            "-" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert features back to the original dict structure."""
        result = {}
        for i, layer_idx in enumerate(self.layer_indices):
            k_cache, v_cache = self.caches[i]
            result[f"layer.{layer_idx}.k_cache"] = k_cache
            result[f"layer.{layer_idx}.v_cache"] = v_cache
            if layer_idx == self.last_layer_idx and self.output is not None:
                result[f"layer.{layer_idx}.output"] = self.output
        return result


class IndividualPacker(PackerBase):
    """将所有的k-cache/v-cache/hidden_state都单独打包"""

    def _pack_impl(self, features: KimiAudioFeatures):
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

    def _unpack_impl(self, tensors: Dict[str, tuple]) -> KimiAudioFeatures:
        caches = []
        for layer_idx in self._last_reference.layer_indices:
            k_cache, meta = tensors[f"layer.{layer_idx}.k_cache"]
            k_cache = einops.rearrange(k_cache, "1 1 nt (nh hd) -> 1 nh nt hd", nh=meta["ori_shape"][1])
            v_cache, meta = tensors[f"layer.{layer_idx}.v_cache"]
            v_cache = einops.rearrange(v_cache, "1 1 nt (nh hd) -> 1 nh nt hd", nh=meta["ori_shape"][1])
            caches.append((k_cache, v_cache))

        assert layer_idx == self._last_reference.last_layer_idx and self._last_reference.output is not None
        output, meta = tensors[f"layer.{layer_idx}.output"]
        output = einops.rearrange(output, "1 1 nt d -> 1 nt d")

        self.log("  IndividualUnPacker:")
        for layer_idx, (k_cache, v_cache) in zip(self._last_reference.layer_indices, caches):
            self.log(f"    layer.{layer_idx}.k_cache: {k_cache.shape}")
            self.log(f"    layer.{layer_idx}.v_cache: {v_cache.shape}")
        self.log(f"    layer.{layer_idx}.output: {output.shape}")

        return KimiAudioFeatures(
            model_type=self._last_reference.model_type,
            source_path=self._last_reference.source_path,
            dtype=self._last_reference.dtype,
            last_layer_idx=self._last_reference.last_layer_idx,
            layer_indices=self._last_reference.layer_indices,
            caches=caches,
            output=output,
        )


class KimiAudioHandler(BaseHandler):
    """Handler for KimiAudio model features."""

    model_type = "kimiaudio"
    SUPPORTED_STRATEGIES = {
        "individual": IndividualPacker,
    }

    def parse(self, path: str) -> KimiAudioFeatures:
        """Parse KimiAudio features from a .zst file.

        ```
        root: [Dict] with 5 keys
        key['model.layers.0']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-165.0000, 171.0000]
            item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-1.1172, 1.9531]
        key['model.layers.4']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 78, 3584]), dtype=torch.bfloat16, device=cpu, range=[-10688.0000, 3904.0000]
            item[1]: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-11.0625, 8.7500]
            item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-3.2031, 4.1562]
        key['model.layers.2']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-33.2500, 30.2500]
            item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-2.2500, 2.1875]
        key['model.layers.3']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-43.7500, 71.5000]
            item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-1.6875, 2.0781]
        key['model.layers.1']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-39.7500, 63.0000]
            item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-1.1406, 1.2266]
        ```
        """
        # Load raw data
        data = load_tensor(path)
        if "features" in data:
            data = data["features"]

        if self.log_fn is not None:
            self.log("\nOriginal data structure:")
            inspect_structure(data, prefix="root", print_fn=self.log_fn)

        # Extract layer indices and sort
        layer_pattern = re.compile(r"model\.layers\.(\d+)")
        layer_info = []
        for key in data.keys():
            match = layer_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                layer_info.append((layer_idx, key))

        # Sort by layer index
        layer_info.sort(key=lambda x: x[0])
        layer_indices = [idx for idx, _ in layer_info]

        # Identify last layer (has output feature + kv_cache)
        caches = []
        output = None
        last_layer_idx = -1
        dtype = torch.bfloat16
        for layer_idx, key in layer_info:
            layer_data = data[key]

            # Check if this is the last layer with output feature
            if self._is_last_layer(layer_data):
                # Last layer structure: (output_feature, (k_cache, v_cache))
                output = layer_data[0]
                kv_cache = layer_data[1]
                caches.append((kv_cache[0], kv_cache[1]))
                last_layer_idx = layer_idx
                dtype = output.dtype
            else:
                # Regular layer: (k_cache, v_cache)
                caches.append((layer_data[0], layer_data[1]))
                dtype = layer_data[0].dtype

        return KimiAudioFeatures(
            model_type=self.model_type,
            source_path=path,
            dtype=dtype,
            caches=caches,
            output=output,
            last_layer_idx=last_layer_idx,
            layer_indices=layer_indices,
            metadata={"original_keys": list(data.keys())},
        )

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

    def restore_format(self, features: KimiAudioFeatures) -> Dict[str, torch.Tensor]:
        """Restore features to a dictionary as the original format.

        ```
        root: [Dict] with 5 keys
        key['model.layers.0']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-165.0000, 171.0000]
            item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-1.1172, 1.9531]
        key['model.layers.4']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 78, 3584]), dtype=torch.bfloat16, device=cpu, range=[-10688.0000, 3904.0000]
            item[1]: [Tuple] with 2 items
                item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-11.0625, 8.7500]
                item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-3.2031, 4.1562]
        key['model.layers.2']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-33.2500, 30.2500]
            item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-2.2500, 2.1875]
        key['model.layers.3']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-43.7500, 71.5000]
            item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-1.6875, 2.0781]
        key['model.layers.1']: [Tuple] with 2 items
            item[0]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-39.7500, 63.0000]
            item[1]: [Tensor] shape=torch.Size([1, 4, 78, 128]), dtype=torch.bfloat16, device=cpu, range=[-1.1406, 1.2266]
        ```

        NOTE:
            这里的实现隐含“layer_idx == list index”假设。当前直接用层号当列表索引。
            若层号不连续或非 0 开始，恢复结构会错位或越界。不过在我们的设定中，缓存序号是从0开始的。
        """
        restored = {}
        for layer_idx in features.layer_indices:
            ori_layer_name = f"model.layers.{layer_idx}"

            if layer_idx == features.last_layer_idx:  # Last layer with output feature
                restored[ori_layer_name] = (features.output, features.caches[layer_idx])
            else:  # shallower layer
                restored[ori_layer_name] = features.caches[layer_idx]

        inspect_structure(restored, prefix="Restored Feature Format", print_fn=self.log_fn)
        return restored
