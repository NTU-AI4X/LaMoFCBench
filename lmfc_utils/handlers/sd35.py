from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import einops
import torch

from .base_handler import BaseHandler, FeatureData, PackerBase
from .utils import inspect_structure, load_tensor


@dataclass
class SD35Features(FeatureData):
    """Parsed and organized SD3.5 features.

    Attributes:
        text_features: Dict mapping encoder name to their feature dicts.
        latents: VAE decoder latent tensor (if present).
        encoder_names: List of text encoder names.
    """

    text_features: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    latents: Optional[torch.Tensor] = None
    encoder_features: Dict[str, torch.Tensor] = field(default_factory=dict)  # VAE encoder features
    encoder_names: List[str] = field(default_factory=list)

    @property
    def num_encoders(self) -> int:
        """Number of text encoders."""
        return len(self.encoder_names)

    @property
    def has_latents(self) -> bool:
        """Whether VAE latents are present."""
        return self.latents is not None

    @property
    def has_encoder_features(self) -> bool:
        """Whether VAE encoder features are present."""
        return len(self.encoder_features) > 0

    def summary(self) -> str:
        """Return a summary string of the features."""
        lines = [
            "-" * 60,
            "SD3.5 Features Summary",
            "-" * 60,
            f"Number of text encoders: {self.num_encoders}",
            f"Encoder names: {self.encoder_names}",
            f"Has VAE latents: {self.has_latents}",
            f"Has VAE encoder features: {self.has_encoder_features}",
            f"Data type: {self.dtype}",
        ]

        for name, feats in self.text_features.items():
            lines.append(f"  {name}:")
            for key, tensor in feats.items():
                if isinstance(tensor, torch.Tensor):
                    lines.append(f"    {key}: {tensor.shape}, {tensor.dtype}")

        if self.latents is not None:
            lines.append(f"  vae.decoder latents: {self.latents.shape}")

        if self.encoder_features:
            lines.append("  vae.encoder features:")
            for name, tensor in self.encoder_features.items():
                lines.append(f"    {name}: {tensor.shape}, {tensor.dtype}")

        lines.append("-" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert features back to the original dict structure.

        Returns:
            Dict with structure matching original format.
        """
        unpacked = {}
        # Text features
        for enc_name, enc_feats in self.text_features.items():
            for key, feat in enc_feats.items():
                unpacked[f"{enc_name}.{key}"] = feat

        # VAE encoder features
        if self.has_encoder_features:
            unpacked.update(self.encoder_features)

        # VAE decoder
        if self.has_latents:
            unpacked["vae.decoder"] = self.latents
        return unpacked


class IndividualPacker(PackerBase):
    """Pack each tensor individually."""

    def _pack_impl(self, features: SD35Features):
        packed = {}

        # Text encoder features
        for enc_name, enc_feats in features.text_features.items():
            for feat_name, tensor in enc_feats.items():
                assert isinstance(tensor, torch.Tensor)
                ori_shape = tensor.shape
                if "pool" in feat_name:
                    tensor = einops.rearrange(tensor, "c -> 1 1 1 c")
                else:
                    tensor = einops.rearrange(tensor, "nt c -> 1 1 nt c")
                packed[f"{enc_name}.{feat_name}"] = (tensor, ori_shape)

        # VAE encoder features (two tensors)
        if features.has_encoder_features:
            for feat_name, tensor in features.encoder_features.items():
                ori_shape = tensor.shape
                tensor = einops.rearrange(tensor, "(ch cw) h w -> 1 1 (ch h) (cw w)", ch=4, cw=8)
                packed[feat_name] = (tensor, ori_shape)

        # VAE decoder latents
        if features.has_latents:
            ori_shape = features.latents.shape
            latents = einops.rearrange(features.latents, "(ch cw) h w -> 1 1 (ch h) (cw w)", ch=4, cw=4)
            packed["vae.decoder"] = (latents, ori_shape)

        self.log("  IndividualPacker:")
        for name, (tensor, ori_shape) in packed.items():
            self.log(f"    {name}: {ori_shape}->{tensor.shape}")
        return packed

    def _unpack_impl(self, tensors) -> SD35Features:
        text_features: Dict[str, Dict[str, torch.Tensor]] = {}
        encoder_features: Dict[str, torch.Tensor] = {}
        latents: Optional[torch.Tensor] = None
        encoder_names: List[str] = []

        for key, (tensor, meta) in tensors.items():
            if key == "vae.decoder":
                latents = einops.rearrange(tensor, "1 1 (ch h) (cw w) -> (ch cw) h w", ch=4, cw=4)

            elif key.startswith("vae.encoder_"):  # vae.encoder_f0/_f1
                encoder_features[key] = einops.rearrange(tensor, "1 1 (ch h) (cw w) -> (ch cw) h w", ch=4, cw=8)

            elif key.startswith("text_encoder"):  # Text encoder features
                enc_name, feat_name = key.rsplit(".", 1)
                if enc_name not in text_features:
                    text_features[enc_name] = {}
                    encoder_names.append(enc_name)
                # Restore original shape based on ori_shape dimensions
                # Pooled embeds: was (c,) -> (1, 1, 1, c)
                # Regular embeds: was (nt, c) -> (1, 1, nt, c)
                text_features[enc_name][feat_name] = tensor.squeeze()
            else:
                raise KeyError(f"Unexpected key format during unpacking: {key}")

        self.log("  IndividualUnPacker:")
        for enc_name, feats in text_features.items():
            for k, v in feats.items():
                self.log(f"    {enc_name}.{k}: {v.shape}")
        if latents is not None:
            self.log(f"    vae.decoder: {latents.shape}")
        for k, v in encoder_features.items():
            self.log(f"    {k}: {v.shape}")

        return SD35Features(
            model_type=self._last_reference.model_type,
            source_path=self._last_reference.source_path,
            dtype=self._last_reference.dtype,
            text_features=text_features,
            latents=latents,
            encoder_features=encoder_features,
            encoder_names=encoder_names,
        )


class SD35Handler(BaseHandler):
    """Handler for SD3.5 diffusion model features.

    This class handles the SD3.5 feature format:
    - text_encoder, text_encoder_2, text_encoder_3 outputs
    - vae.decoder latents
    - vae.encoder features

    Supports flexible tensor grouping via predefined grouping specs.
    """

    model_type = "sd35"
    SUPPORTED_STRATEGIES = {
        "individual": IndividualPacker,
    }

    def parse(self, path: str) -> SD35Features:
        """Parse SD3.5 features from a .zst file.

        Args:
            path: Path to the .zst file containing SD3.5 features.

        Returns:
            SD35Features object with organized features.
        """
        # Load raw data
        data = load_tensor(path)

        if self.log_fn is not None:
            self.log("\nOriginal data structure:")
            inspect_structure(data, prefix="root", print_fn=self.log_fn)

        # Extract features
        text_features = {}
        encoder_features = {}  # VAE encoder features
        latents = None
        encoder_names = []
        dtype = torch.float16

        for key, value in data.items():
            if key.startswith("vae.decoder"):  # VAE latent tensor
                if isinstance(value, torch.Tensor):
                    latents = value
                    dtype = value.dtype
            elif key.startswith("vae.encoder"):  # VAE encoder features (dict with multiple feature tensors)
                assert isinstance(value, dict), "VAE encoder features should be a dict of tensors."
                for feat_key, feat_val in value.items():
                    if isinstance(feat_val, torch.Tensor):
                        encoder_features[feat_key] = feat_val
                        dtype = feat_val.dtype
                # elif isinstance(value, torch.Tensor):
                #     encoder_features[key] = value
                #     dtype = value.dtype
                # else:
                #     raise TypeError(f"Only torch.Tensor or dict of torch.Tensor are supported, but got {type(value)}.")
            elif key.startswith("text_encoder"):  # Text encoder output (dict with embeddings)
                encoder_name = key.split("-item")[0] if "-item" in key else key
                if encoder_name not in encoder_names:
                    encoder_names.append(encoder_name)

                if isinstance(value, dict):
                    text_features[key] = {}
                    for feat_key, feat_val in value.items():
                        if isinstance(feat_val, torch.Tensor):
                            text_features[key][feat_key] = feat_val
                            dtype = feat_val.dtype
                elif isinstance(value, torch.Tensor):
                    text_features[key] = {"embedding": value}
                    dtype = value.dtype
                else:
                    raise TypeError(f"Only torch.Tensor or dict of torch.Tensor are supported, but got {type(value)}.")

        return SD35Features(
            model_type=self.model_type,
            source_path=path,
            dtype=dtype,
            text_features=text_features,
            latents=latents,
            encoder_features=encoder_features,
            encoder_names=encoder_names,
            metadata={"original_keys": list(data.keys())},
        )

    def restore_format(self, features: SD35Features) -> Dict[str, torch.Tensor]:
        """Restore features to a dictionary as the original format.

        text_encoder-item0:
            clip_pooled_prompt_embeds: torch.Size([768]), torch.float16
            clip_prompt_embeds: torch.Size([77, 768]), torch.float16
        text_encoder_2-item1:
            clip_pooled_prompt_embeds: torch.Size([1280]), torch.float16
            clip_prompt_embeds: torch.Size([77, 1280]), torch.float16
        text_encoder_3-item2:
            t5_prompt_embeds: torch.Size([77, 4096]), torch.float16
        text_encoder-item3:
            clip_pooled_prompt_embeds: torch.Size([768]), torch.float16
            clip_prompt_embeds: torch.Size([77, 768]), torch.float16
        text_encoder_2-item4:
            clip_pooled_prompt_embeds: torch.Size([1280]), torch.float16
            clip_prompt_embeds: torch.Size([77, 1280]), torch.float16
        text_encoder_3-item5:
            t5_prompt_embeds: torch.Size([77, 4096]), torch.float16
        vae.decoder latents: torch.Size([16, 128, 128])
        vae.encoder features:
            vae.encoder_f0: torch.Size([32, 128, 128]), torch.float16
            vae.encoder_f1: torch.Size([32, 128, 128]), torch.float16
        """
        restored = {
            **features.text_features,
            "vae.decoder": features.latents,
            "vae.encoder": features.encoder_features,
        }
        inspect_structure(restored, prefix="Restored Feature Format", print_fn=self.log_fn)
        return restored
