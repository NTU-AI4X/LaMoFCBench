from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import einops
import torch

from .base_handler import BaseHandler, FeatureData, PackerBase
from .utils import inspect_structure, load_tensor


@dataclass
class FalconMambaFeatures(FeatureData):
    """Parsed and organized FalconMamba features.

    Attributes:
        ssm_states: List of SSM state tensors, one per layer.
        conv_states: List of Conv state tensors, one per layer.
        output: Output hidden states (tuple or tensor).
    """

    ssm_states: List[torch.Tensor] = field(default_factory=list)
    conv_states: List[torch.Tensor] = field(default_factory=list)
    output: Optional[torch.Tensor] = None
    last_layer_idx: int = -1
    layer_indices: List[int] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        """Number of layers with states."""
        return len(self.ssm_states)

    @property
    def ssm_shape(self) -> Tuple[int, ...]:
        """Shape of a single SSM state tensor."""
        if self.ssm_states:
            return tuple(self.ssm_states[0].shape)
        return ()

    @property
    def conv_shape(self) -> Tuple[int, ...]:
        """Shape of a single Conv state tensor."""
        if self.conv_states:
            return tuple(self.conv_states[0].shape)
        return ()

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Shape of the output tensor."""
        if self.output is not None:
            return tuple(self.output.shape)
        return ()

    def summary(self) -> str:
        """Return a summary string of the features."""
        lines = [
            "-" * 60,
            "FalconMamba Features Summary",
            "-" * 60,
            f"Number of layers: {self.num_layers}",
            f"SSM state shape: {self.ssm_shape}",
            f"Conv state shape: {self.conv_shape}",
            f"Output shape: {self.output_shape}",
            f"Data type: {self.dtype}",
            "-" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert features back to the original dict structure."""
        result = {}
        for i, layer_idx in enumerate(self.layer_indices):
            result[f"layer.{layer_idx}.ssm_state"] = self.ssm_states[i]
            result[f"layer.{layer_idx}.conv_state"] = self.conv_states[i]

        assert layer_idx == self.last_layer_idx and self.output is not None
        result[f"layer.{layer_idx}.output"] = self.output
        return result


class IndividualPacker(PackerBase):
    """Pack each state tensor individually.
    IndividualPacker:
        layer.0.ssm_state: torch.Size([1, 8192, 16]) -> torch.Size([1, 1, 8192, 16])
        layer.0.conv_state: torch.Size([1, 8192, 4]) -> torch.Size([1, 1, 8192, 4])
        layer.1.ssm_state: torch.Size([1, 8192, 16]) -> torch.Size([1, 1, 8192, 16])
        layer.1.conv_state: torch.Size([1, 8192, 4]) -> torch.Size([1, 1, 8192, 4])
        layer.2.ssm_state: torch.Size([1, 8192, 16]) -> torch.Size([1, 1, 8192, 16])
        layer.2.conv_state: torch.Size([1, 8192, 4]) -> torch.Size([1, 1, 8192, 4])
        layer.3.ssm_state: torch.Size([1, 8192, 16]) -> torch.Size([1, 1, 8192, 16])
        layer.3.conv_state: torch.Size([1, 8192, 4]) -> torch.Size([1, 1, 8192, 4])
        layer.4.ssm_state: torch.Size([1, 8192, 16]) -> torch.Size([1, 1, 8192, 16])
        layer.4.conv_state: torch.Size([1, 8192, 4]) -> torch.Size([1, 1, 8192, 4])
        layer.4.output: torch.Size([1, N, 4096]) -> torch.Size([1, 1, N, 4096])
    """

    def _pack_impl(self, features: FalconMambaFeatures):
        packed = {}
        for layer_idx, (ssm, conv) in enumerate(zip(features.ssm_states, features.conv_states)):
            packed[f"layer.{layer_idx}.ssm_state"] = (einops.rearrange(ssm, "1 d l -> 1 1 d l"), ssm.shape)
            packed[f"layer.{layer_idx}.conv_state"] = (einops.rearrange(conv, "1 d l -> 1 1 d l"), conv.shape)

        # 最后一层的hidden_state 一定存在，所以无需判断
        packed[f"layer.{layer_idx}.output"] = (
            einops.rearrange(features.output, "1 nt d-> 1 1 nt d"),
            features.output.shape,
        )

        self.log("  IndividualPacker:")
        for name, (tensor, ori_shape) in packed.items():
            self.log(f"    {name}: {ori_shape} -> {tensor.shape}")
        return packed

    def _unpack_impl(self, tensors: Dict[str, tuple]) -> FalconMambaFeatures:
        ssm_states = []
        conv_states = []
        output = None
        for layer_idx in range(self._last_reference.num_layers):
            ssm_cache, meta = tensors[f"layer.{layer_idx}.ssm_state"]
            ssm_cache = einops.rearrange(ssm_cache, "1 1 d l -> 1 d l")
            ssm_states.append(ssm_cache)

            conv_cache, meta = tensors[f"layer.{layer_idx}.conv_state"]
            conv_cache = einops.rearrange(conv_cache, "1 1 d l -> 1 d l")
            conv_states.append(conv_cache)

        assert layer_idx == self._last_reference.last_layer_idx and self._last_reference.output is not None
        output, _ = tensors[f"layer.{layer_idx}.output"]
        output = einops.rearrange(output, "1 1 nt d -> 1 nt d")

        self.log("  IndividualUnPacker:")
        for layer_idx, ssm_cache, conv_cache in zip(self._last_reference.layer_indices, ssm_states, conv_states):
            self.log(f"    layer.{layer_idx}.ssm_state: {ssm_cache.shape}")
            self.log(f"    layer.{layer_idx}.conv_state: {conv_cache.shape}")
        self.log(f"    layer.{layer_idx}.output: {output.shape}")

        return FalconMambaFeatures(
            model_type=self._last_reference.model_type,
            source_path=self._last_reference.source_path,
            dtype=self._last_reference.dtype,
            last_layer_idx=self._last_reference.last_layer_idx,
            layer_indices=self._last_reference.layer_indices,
            ssm_states=ssm_states,
            conv_states=conv_states,
            output=output,
        )


class FalconMambaHandler(BaseHandler):
    """Handler for FalconMamba model features.

    root: [Dict] with 3 keys
        key['output']: [Tensor] shape=torch.Size([1, 250, 4096]), dtype=torch.float32, device=cpu, range=[-2.4409, 1.2303]
        key['ssm_state']: [List] with 5 items
            item[0]: [Tensor] shape=torch.Size([1, 8192, 16]), dtype=torch.float16, device=cpu, range=[-0.0765, 0.1431]
            item[1]: [Tensor] shape=torch.Size([1, 8192, 16]), dtype=torch.float16, device=cpu, range=[-0.6221, 0.8804]
            item[2]: [Tensor] shape=torch.Size([1, 8192, 16]), dtype=torch.float16, device=cpu, range=[-0.0705, 0.0706]
            item[3]: [Tensor] shape=torch.Size([1, 8192, 16]), dtype=torch.float16, device=cpu, range=[-0.1410, 0.0731]
            item[4]: [Tensor] shape=torch.Size([1, 8192, 16]), dtype=torch.float16, device=cpu, range=[-0.1869, 0.2224]
        key['conv_state']: [List] with 5 items
            item[0]: [Tensor] shape=torch.Size([1, 8192, 4]), dtype=torch.float16, device=cpu, range=[-8.2656, 7.5078]
            item[1]: [Tensor] shape=torch.Size([1, 8192, 4]), dtype=torch.float16, device=cpu, range=[-9.7344, 12.1016]
            item[2]: [Tensor] shape=torch.Size([1, 8192, 4]), dtype=torch.float16, device=cpu, range=[-6.2969, 5.3906]
            item[3]: [Tensor] shape=torch.Size([1, 8192, 4]), dtype=torch.float16, device=cpu, range=[-4.5195, 4.8750]
            item[4]: [Tensor] shape=torch.Size([1, 8192, 4]), dtype=torch.float16, device=cpu, range=[-13.7031, 9.9062]
    """

    model_type = "falconmamba"
    SUPPORTED_STRATEGIES = {
        "individual": IndividualPacker,
    }

    def parse(self, path: str) -> FalconMambaFeatures:
        """Parse FalconMamba features from a .zst file."""
        # Load raw data
        data = load_tensor(path)

        if self.log_fn is not None:
            self.log("\nOriginal data structure:")
            inspect_structure(data, prefix="root", print_fn=self.log_fn)

        ssm_states = data["ssm_state"]
        conv_states = data["conv_state"]
        output = data["output"]
        layer_indices = [idx for idx in range(len(ssm_states))]

        if isinstance(ssm_states, tuple):
            ssm_states = list(ssm_states)
        if isinstance(conv_states, tuple):
            conv_states = list(conv_states)

        return FalconMambaFeatures(
            model_type=self.model_type,
            source_path=path,
            dtype=output.dtype,
            ssm_states=ssm_states,
            conv_states=conv_states,
            output=output,
            last_layer_idx=layer_indices[-1],
            layer_indices=layer_indices,
            metadata={"original_keys": list(data.keys())},
        )

    def restore_format(self, features: FalconMambaFeatures) -> Dict[str, torch.Tensor]:
        """Restore features to a dictionary as the original format."""
        restored = {
            "output": features.output,
            "ssm_state": features.ssm_states,
            "conv_state": features.conv_states,
        }
        inspect_structure(restored, prefix="Restored Feature Format", print_fn=self.log_fn)
        return restored
