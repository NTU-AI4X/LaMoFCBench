import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import einops
import torch

from .base_handler import BaseHandler, FeatureData, PackerBase
from .utils import inspect_structure, load_tensor


@dataclass
class DINOv3TotalFeatures(FeatureData):
    """Parsed and organized DINOv3 features.

    Attributes:
        layer_features: Dict mapping layer names to their output tensors/tuples.
        layer_names: List of layer names in order.
    """

    layer_features: Dict[str, Any] = field(default_factory=dict)
    layer_names: List[str] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        """Number of layers with features."""
        return len(self.layer_features)

    def summary(self) -> str:
        """Return a summary string of the features."""
        lines = [
            "-" * 60,
            "DINOv3 Features Summary",
            "-" * 60,
            f"Number of layers: {self.num_layers}",
            f"Layer names: {self.layer_names}",
            f"Data type: {self.dtype}",
        ]

        for name, feat in self.layer_features.items():
            if isinstance(feat, torch.Tensor):
                lines.append(f"  {name}: {feat.shape}, {feat.dtype}")
            elif isinstance(feat, (tuple, list)):
                lines.append(f"  {name}: Tuple with {len(feat)} elements")
                for i, f in enumerate(feat):
                    if isinstance(f, torch.Tensor):
                        lines.append(f"    [{i}]: {f.shape}, {f.dtype}")

        lines.append("-" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to a dictionary for saving or processing."""
        return self.layer_features


class IndividualPacker(PackerBase):
    """Pack each layer's tensors individually."""

    def _pack_impl(self, features: DINOv3TotalFeatures):
        packed = {}
        for name, feat in features.layer_features.items():
            assert isinstance(feat, torch.Tensor)
            packed[name] = (einops.rearrange(feat, "nt d -> 1 1 nt d"), feat.shape, name)

        self.log("  IndividualPacker:")
        for name, (tensor, ori_shape, feat_key) in packed.items():
            self.log(f"    {name}: {ori_shape} -> {tensor.shape}\n\tFrom {feat_key}")
        return packed

    def _unpack_impl(self, tensors: Dict[str, tuple]) -> DINOv3TotalFeatures:
        layer_features = {}
        for name, (tensor, meta) in tensors.items():
            # shape = meta["ori_shape"]
            # 因为外部使用的是*feat_key接受的参数，meta["feat_key"]是list
            layer_features[name] = einops.rearrange(tensor, "1 1 nt d -> 1 nt d")

        self.log("  IndividualUnPacker:")
        for layer_name, tensor in layer_features.items():
            self.log(f"    {layer_name}: {tensor.shape}")

        return DINOv3TotalFeatures(
            model_type=self._last_reference.model_type,
            source_path=self._last_reference.source_path,
            dtype=self._last_reference.dtype,
            layer_features=layer_features,
            layer_names=self._last_reference.layer_names,
        )


class DINOv3TotalHandler(BaseHandler):
    """Handler for DINOv3 Vision Transformer features.

    # classification
    root: [Dict] with 2 keys
        key['backbone.blocks.39']: [List] with 1 items
            item[0]: [Tensor] shape=torch.Size([1, N, 4096]), dtype=torch.float32, device=cpu, range=[-12637.4629, 7239.9453]
        key['backbone.blocks.9']: [List] with 1 items
            item[0]: [Tensor] shape=torch.Size([1, N, 4096]), dtype=torch.float32, device=cpu, range=[-7080.7744, 1001.5547]

    # segmentation

    # depth estimation
    root: [Dict] with 4 keys
        key['encoder.backbone.blocks.39']: [Tensor] shape=torch.Size([2, N, 4096]), dtype=torch.float32, device=cpu, range=[-12987.6357, 7768.3691]
        key['encoder.backbone.blocks.19']: [Tensor] shape=torch.Size([2, N, 4096]), dtype=torch.float32, device=cpu, range=[-7909.9863, 1052.8085]
        key['encoder.backbone.blocks.9']: [Tensor] shape=torch.Size([2, N, 4096]), dtype=torch.float32, device=cpu, range=[-7205.2227, 1020.2337]
        key['encoder.backbone.blocks.29']: [Tensor] shape=torch.Size([2, N, 4096]), dtype=torch.float32, device=cpu, range=[-8824.8057, 1061.2744]
    """

    model_type = "dinov3_total"
    NAME_MAPPING = {
        # cls
        "backbone.blocks.9": "layer.9",
        "backbone.blocks.39": "layer.39",
        # dep
        "encoder.backbone.blocks.9": "layer.9",
        "encoder.backbone.blocks.19": "layer.19",
        "encoder.backbone.blocks.29": "layer.29",
        "encoder.backbone.blocks.39": "layer.39",
        # seg
        "segmentation_model.0.backbone.blocks.9": "layer.9",
        "segmentation_model.0.backbone.blocks.19": "layer.19",
        "segmentation_model.0.backbone.blocks.29": "layer.29",
        "segmentation_model.0.backbone.blocks.39": "layer.39",
    }
    SUPPORTED_STRATEGIES = {
        "individual": IndividualPacker,
    }

    def parse(self, path: str) -> DINOv3TotalFeatures:
        """Parse DINOv3 features from a .zst file."""
        # Load raw data
        data = load_tensor(path)

        if self.log_fn is not None:
            self.log("\nOriginal data structure:")
            inspect_structure(data, prefix="root", print_fn=self.log_fn)

        # Extract layer features, sort by layer number in "blocks.XX"
        def extract_block_num(key: str) -> int:
            match = re.search(r"blocks\.(\d+)", key)
            return int(match.group(1)) if match else 0

        layer_names = []
        layer_features = {}
        for name in sorted(data.keys(), key=extract_block_num):
            feat = data[name]
            if isinstance(feat, (tuple, list)):
                assert len(feat) == 1, (name, len(feat))
                feat = feat[0]  # 移除classification特征中无意中添加的单元素列表形式
            assert isinstance(feat, torch.Tensor)

            layer_name = self.NAME_MAPPING[name]
            assert isinstance(feat, torch.Tensor)
            for i, f in enumerate(feat):
                layer_names.append(layer_name)
                layer_features[f"{layer_name}.{i}"] = f

        features = DINOv3TotalFeatures(
            model_type=self.model_type,
            source_path=path,
            dtype=feat.dtype,  # Determine dtype from last tensor found
            layer_features=layer_features,
            layer_names=layer_names,
            metadata={"original_keys": list(data.keys())},
        )
        if self.log_fn is not None:
            self.log(f"\nParsed features:\n{features.summary()}")
        return features

    def restore_format(self, features: DINOv3TotalFeatures) -> Dict[str, torch.Tensor]:
        """Restore features to a dictionary as the original format."""
        restored = {}
        for layer_name in sorted(set(features.layer_names), key=lambda x: int(x.split(".")[-1])):
            batch_combined = []
            for item_idx in range(3):  # Check up to 3 samples
                if f"{layer_name}.{item_idx}" not in features.layer_features:
                    continue
                batch_combined.append(features.layer_features[f"{layer_name}.{item_idx}"])
            # Combine all samples into a batch
            if batch_combined:
                restored[layer_name] = torch.cat(batch_combined, dim=0)

        inspect_structure(restored, prefix="Restored Feature Format", print_fn=self.log_fn)
        return restored
