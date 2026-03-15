"""
DT-UFC Evaluation Script - Universal Feature Coding with Distribution Transformation.

This script evaluates DT-UFC's approach using learned codecs (hyperprior, elic)
with nonlinear quantization for universal feature coding.

Based on: "DT-UFC: Universal Large Model Feature Coding via Peaky-to-Balanced Distribution Transformation"
Paper: https://arxiv.org/abs/2506.16495
"""

import argparse
import glob
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import compressai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd
from compressai.ops import compute_padding
from tabulate import SEPARATING_LINE, tabulate

sys.path.append("../lmfc_utils")
from custom_codecs import AVAILABLE_CODECS
from handlers import AVAILABLE_HANDLERS, get_handler
from handlers.utils import DTYPE_TORCHTYPE_TO_BITS, load_zst_tensor, recursive_check_equal

# =============================================================================
seed = 20260123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 固定 cuDNN 行为（你已经在用）
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# 强烈建议：关闭 TF32（对 learned compression 很关键）
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# 可选：更“狠”的确定性（可能会变慢，或遇到不支持的算子直接报错）
# torch.use_deterministic_algorithms(True)
# 可选：让 cuBLAS 更确定性（有时 GEMM 相关会需要）
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# =============================================================================

MINIMAL_DIV = 64  # Minimal divisor for spatial padding (2^6 = 64)
BASE_BITS = 32  # Base bits for EBPFP calculation (normalized to 32-bit floating point)


# =============================================================================
# Utility Functions for Compressed Feature Saving
# =============================================================================


def save_compressed(data, path, algorithm="zstd", level=None):
    """
    保存 PyTorch 数据并压缩。

    参数:
        data: 要保存的对象 (Tensor, dict, model state_dict 等)
        path (str): 保存路径 (建议后缀: .zst)
        algorithm (str): 'zstd' (推荐) 或 None (不压缩)
        level (int): 压缩等级。zstd: 默认3 (1-22)
    """
    if algorithm == "zstd":
        level = level if level is not None else 3
        cctx = zstd.ZstdCompressor(level=level)
        with zstd.open(path, "wb", cctx=cctx) as f:
            torch.save(data, f)
    elif algorithm is None or algorithm == "none":
        torch.save(data, path)
    else:
        raise ValueError(f"不支持的压缩算法: {algorithm}")


# =============================================================================
# Nonlinear Transform Functions (adapted from DT-UFC)
# =============================================================================


def load_quantization_points(file_path: Union[str, List[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Load quantization points from a JSON file or list of files.

    Args:
        file_path: Path to JSON file or list of paths.

    Returns:
        Quantization points as torch.Tensor or list of tensors (sorted).
    """

    def load_file(path: str) -> torch.Tensor:
        with open(path, "r") as f:
            data = json.load(f)
        # Return sorted tensor for use with torch.bucketize
        data = torch.tensor(data, dtype=torch.float32).sort().values
        logger.info(f"Loaded quantization points from {path}: {data.shape}")
        return data

    if isinstance(file_path, list):
        return [load_file(p) for p in file_path]
    elif isinstance(file_path, str):
        return load_file(file_path)
    else:
        raise ValueError("file_path must be a string or a list of strings.")


def load_per_key_quantization_points(config_path: str, model_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Load per-key quantization points from a JSON config file.

    Config format (nested by model name):
        {
            "model_name_1": {
                "key1": "path/to/mapping1.json",
                "key2": "path/to/mapping2.json"
            },
            "model_name_2": {
                ...
            }
        }

    Args:
        config_path: Path to JSON config file.
        model_name: Model name to lookup in config (e.g., 'cls_in1kval').

    Returns:
        Dict mapping feature keys to quantization points tensors.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get the mapping dict for this model
    if model_name:
        if model_name not in config:
            available = list(config.keys())
            raise ValueError(f"Model '{model_name}' not found in config. Available: {available}")
        key_mapping = config[model_name]
    else:
        # Assume flat config (legacy format)
        key_mapping = config

    # Resolve relative paths relative to config file directory
    mapping_root = Path(config_path).parent
    mappings = {}
    for key, mapping_path in key_mapping.items():
        mapping_path = mapping_root.joinpath(mapping_path).as_posix()
        assert os.path.isfile(mapping_path), f"Mapping path ({mapping_path}) for key '{key}' must be a file."
        mappings[key] = load_quantization_points(mapping_path)
        logger.info(f"Loaded per-key quantization points for key '{key}' from {mapping_path}")
    return mappings


def nonlinear_quantization(
    data: torch.Tensor, quantization_points: Union[torch.Tensor, List[torch.Tensor]], bit_depth: int
) -> torch.Tensor:
    """Apply K-Means based nonlinear quantization (PyTorch version).

    Maps floating-point data to quantized indices based on pre-computed K-Means centroids.

    Args:
        data: Original floating-point tensor with shape (N, C, H, W).
        quantization_points: K-Means centroids (single tensor or list per channel), must be sorted.
        bit_depth: Number of bits for quantization.

    Returns:
        Quantized tensor normalized to [0, 1) range.
    """
    device = data.device

    if isinstance(quantization_points, torch.Tensor):
        # Single quantization points for all channels
        qp = quantization_points.to(device=device)
        num_levels = qp.size(0)
        # np.digitize(x, bins) - 1 uses left-closed, right-open intervals: bins[i-1] <= x < bins[i]
        # torch.bucketize(x, bins, right=True) - 1 matches this behavior exactly (verified)
        quantized_data = torch.bucketize(data, qp, right=True) - 1
        quantized_data = quantized_data.clamp(0, num_levels - 1)
    elif isinstance(quantization_points, list):
        # Per-channel quantization points
        if len(quantization_points) != data.shape[1]:
            raise ValueError("Length of quantization_points list must match number of channels (C).")
        quantized_data = torch.zeros_like(data, dtype=torch.long, device=device)
        for i, qp in enumerate(quantization_points):
            qp = qp.to(device=device)
            num_levels = qp.size(0)
            channel_data = data[:, i, :, :]
            quantized_channel = torch.bucketize(channel_data, qp, right=True) - 1
            quantized_channel = quantized_channel.clamp(0, num_levels - 1)
            quantized_data[:, i, :, :] = quantized_channel
    else:
        raise ValueError("quantization_points must be a torch.Tensor or a list of torch.Tensors.")

    # Normalize to [0, 1) range
    quantized_data = quantized_data.float() / (2**bit_depth)
    return quantized_data


def nonlinear_dequantization(
    quantized_data: torch.Tensor, quantization_points: Union[torch.Tensor, List[torch.Tensor]], bit_depth: int
) -> torch.Tensor:
    """Dequantize quantized data back to floating-point values (PyTorch version).

    Uses lookup table to map quantized indices back to K-Means centroids.

    Args:
        quantized_data: Quantized tensor normalized to [0, 1) range.
        quantization_points: K-Means centroids (single tensor or list per channel), must be sorted.
        bit_depth: Number of bits for quantization.

    Returns:
        Dequantized floating-point tensor.
    """
    device = quantized_data.device
    indices = torch.clamp(torch.round(quantized_data * (2**bit_depth)).long(), 0, 2**bit_depth - 1)

    if isinstance(quantization_points, torch.Tensor):  # Single quantization points for all channels
        # Clamp indices to valid range of quantization points (handles floating point precision issues)
        qp = quantization_points.to(device=device)
        dequantized_data = qp[indices]

    elif isinstance(quantization_points, list):
        if len(quantization_points) != quantized_data.shape[1]:
            raise ValueError("Length of quantization_points list must match number of channels (C).")

        dequantized_data = torch.zeros_like(quantized_data, dtype=torch.float32, device=device)
        for i, qp in enumerate(quantization_points):
            qp = qp.to(device=device)
            channel_indices = indices[:, i, :, :]
            dequantized_data[:, i, :, :] = qp[channel_indices]

    else:
        raise ValueError("quantization_points must be a torch.Tensor or a list of torch.Tensors.")
    return dequantized_data


# =============================================================================
# DT-UFC Codec Configuration
# =============================================================================


@dataclass
class DTUFCCodecConfig:
    """Configuration for DT-UFC Codec."""

    # Architecture
    arch: str = "hyperprior-featurecoding"
    checkpoint_path: str = None
    handler: str = None

    # Transform configuration
    transform_type: str = "nonlinear"  # "uniform" or "nonlinear"
    transform_mapping: Optional[str] = None  # Per-key config JSON file
    bit_depth: int = 10

    # Truncation (for uniform transform)
    trun_low: float = -5.0
    trun_high: float = 5.0
    trun_flag: bool = False

    # Device
    device: str = "cuda"

    def __post_init__(self):
        if self.arch not in AVAILABLE_CODECS:
            raise ValueError(f"Unknown architecture: {self.arch}. Available: {AVAILABLE_CODECS}")

    def __str__(self) -> str:
        """Return a formatted string representation of the config."""
        return self.summary()

    def summary(self, prefix: str = "") -> str:
        """Generate a summary string of the configuration.

        Args:
            prefix: Optional prefix for each line.

        Returns:
            Formatted configuration summary.
        """
        lines = [
            f"{prefix}DTUFCCodecConfig:",
            f"{prefix}  arch:             {self.arch}",
            f"{prefix}  handler:          {self.handler}",
            f"{prefix}  checkpoint:       {self.checkpoint_path}",
            f"{prefix}  transform_type:   {self.transform_type}",
            f"{prefix}  transform_mapping:{self.transform_mapping}",
            f"{prefix}  bit_depth:        {self.bit_depth}",
            f"{prefix}  device:           {self.device}",
        ]
        if self.trun_flag:
            lines.append(f"{prefix}  truncation:       [{self.trun_low}, {self.trun_high}]")
        return "\n".join(lines)


# =============================================================================
# DT-UFC Codec
# =============================================================================


class DTUFCCodec:
    """DT-UFC Codec - Wraps CompressAI models with distribution transformation.

    This codec applies DT-UFC's nonlinear quantization (K-Means based) before
    compressing features with learned image codecs (Hyperprior, ELIC).

    Pipeline:
        Encode: Feature → Truncation → Nonlinear Quantization → Pack → CompressAI Compress
        Decode: CompressAI Decompress → Unpack → Nonlinear Dequantization → Restored Feature
    """

    def __init__(self, config: DTUFCCodecConfig, print_fn=print):
        """Initialize DT-UFC Codec.

        Args:
            config: DTUFCCodecConfig configuration.
            print_fn: Print function for logging.
        """
        self.config = config
        self.print_fn = print_fn

        # Set entropy coder
        compressai.set_entropy_coder("ans")

        # Load model
        self.model = self._load_model()

        # Load transform mapping(s) if using nonlinear
        self.quantization_points = None  # Single shared mapping (legacy)
        self.per_key_quantization_points: Dict[str, torch.Tensor] = {}  # Per-key mappings

        if config.transform_type == "nonlinear":
            assert config.transform_mapping
            # Load per-key mappings from config file
            self.per_key_quantization_points = load_per_key_quantization_points(
                config.transform_mapping, config.handler
            )
            self.print_fn(f"Loaded per-key mappings: model={config.handler}")
            self.print_fn(f"  Keys: {list(self.per_key_quantization_points.keys())}")

    def _get_quantization_points(self, key: Optional[str] = None) -> torch.Tensor:
        """Get quantization points for a given key.

        Args:
            key: Feature key (e.g., 'backbone_blocks_39_0'). If None, uses shared mapping.

        Returns:
            Quantization points tensor.

        Raises:
            ValueError: If no mapping found for key and no shared mapping available.
        """
        # Try per-key mapping first
        # if key and key in self.per_key_quantization_points:
        for _k, _v in self.per_key_quantization_points.items():
            # NOTE: 多 key 重叠命名时可能命中错误 mapping
            # 在 mapping.json 配置两个可同时命中的键（如 layer_1 与 layer_10）。
            # 不过在我们的设置中，不会存在这样的情况，就先忽视这个问题
            if _k in key:  # 兼容dinov3不同层和增强样本特征的名字
                if _v.shape[0] != 2**self.config.bit_depth:
                    raise ValueError(f"Quantization points shape mismatch for {key} and {_k}, {_v.shape}.")
                else:
                    self.print_fn(f"    Using per-key quantization points ({_k}: {_v.shape}) for {key}")
                return _v

        # Fall back to shared mapping
        if self.quantization_points is not None:
            return self.quantization_points

        # No mapping found
        available_keys = list(self.per_key_quantization_points.keys())
        raise ValueError(
            f"No quantization mapping found for key '{key}'. "
            f"Available keys: {available_keys[:5]}{'...' if len(available_keys) > 5 else ''}"
        )

    def _load_model(self) -> nn.Module:
        """Load CompressAI model from checkpoint with dynamic architecture adjustment.

        DT-UFC trains models with 1-channel input/output (modified from standard 3-channel).
        This method creates a 1-channel model variant and loads the pretrained weights.
        """
        arch = self.config.arch
        checkpoint_path = self.config.checkpoint_path
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"DT-UFC requires pre-trained weights. Please provide a valid checkpoint path."
            )

        # Load checkpoint
        self.print_fn(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract state dict
        state_dict = checkpoint
        for key in ["network", "state_dict", "model_state_dict"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if "epoch" in checkpoint:
            self.print_fn(f"Checkpoint epoch: {checkpoint['epoch']}")

        # Create 1-channel model variant from the checkpoint
        model = AVAILABLE_CODECS[arch].from_state_dict(state_dict)

        # Move to device and set eval mode
        model = model.to(device=self.config.device).eval()

        # Update CDFs for entropy coding
        model.update(force=True)

        self.print_fn(f"Loaded {arch} (1-channel) on {self.config.device}")
        return model

    @property
    def device(self):
        """Get device of the model."""
        return next(self.model.parameters()).device

    def _truncate(self, data: torch.Tensor) -> torch.Tensor:
        """Apply truncation if enabled (PyTorch version)."""
        if self.config.trun_flag:
            return torch.clamp(data, self.config.trun_low, self.config.trun_high)
        return data

    def _quantize(self, data: torch.Tensor, key: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply quantization based on transform type (PyTorch version).

        Args:
            data: Input tensor.
            key: Feature key for per-key quantization mapping lookup.

        Returns:
            Tuple of (quantized_data, quantization_points).
            quantization_points is only returned for CDF mode (real-time computed).
        """
        if self.config.transform_type == "nonlinear":  # K-Means quantization
            qp = self._get_quantization_points(key)
            return nonlinear_quantization(data, qp, self.config.bit_depth), None
        else:
            raise ValueError(f"Unknown transform type: {self.config.transform_type}")

    def _dequantize(
        self,
        data: torch.Tensor,
        key: Optional[str] = None,
        cdf_qp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply dequantization based on transform type (PyTorch version).

        Args:
            data: Quantized tensor.
            key: Feature key for per-key quantization mapping lookup.
            cdf_qp: CDF quantization points (required for CDF mode, computed during encoding).
        """
        if self.config.transform_type == "nonlinear":  # K-Means dequantization
            qp = self._get_quantization_points(key)
            return nonlinear_dequantization(data, qp, self.config.bit_depth)
        else:
            raise ValueError(f"Unknown transform type: {self.config.transform_type}")

    @torch.no_grad()
    def encode(self, tensor: torch.Tensor, key: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        """Encode tensor to compressed representation.

        Args:
            tensor: Input tensor with shape (1, 1, H, W) - already packed by handler.
            key: Feature key for per-key quantization mapping lookup.

        Returns:
            Tuple of (total_bytes, metadata).
            total_bytes is the size of compressed bitstream for BPFP calculation.
            metadata contains all data needed for decoding.
        """
        # Record original info
        original_shape = tensor.shape
        original_dtype = tensor.dtype

        # Convert to float and move to device for processing
        data = tensor.float().to(device=self.device)

        # Truncate
        data = self._truncate(data)

        # Quantize (result normalized to [0, 1))
        # For CDF mode, cdf_qp contains the real-time computed quantile points
        x, cdf_qp = self._quantize(data, key)

        # Spatial padding to multiple of MINIMAL_DIV
        N, C, H, W = x.shape
        pad, unpad = compute_padding(H, W, min_div=MINIMAL_DIV)
        x_padded = F.pad(x, pad, mode="constant", value=0)

        # Compress
        out_enc = self.model.compress(x_padded)

        # Calculate total bytes for BPFP
        total_bytes = sum(len(s) for sl in out_enc["strings"] for s in sl)

        # Build metadata (includes everything needed for decoding)
        metadata = {
            "original_shape": original_shape,
            "original_dtype": original_dtype,
            "spatial_shape": (N, C, H, W),
            "spatial_unpad": unpad,
            "latent_shape": out_enc["shape"],
            "strings": out_enc["strings"],  # Directly store compressed strings
            "bpfp": (total_bytes * 8) / tensor.numel(),
            "key": key,  # Store key for decoding
            # Pre-codec dequantized result for MSE breakdown (stays on device)
            "quantized_tensor": x.clone(),
            # CDF quantization points (only for CDF mode)
            "cdf_qp": cdf_qp,
        }

        return total_bytes, metadata

    @torch.no_grad()
    def decode(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decode tensor from metadata.

        Args:
            metadata: Metadata from encode(), contains all compressed data.

        Returns:
            Reconstructed tensor with shape (1, 1, H, W).
        """
        # Extract info from metadata
        N, C, H, W = metadata["spatial_shape"]
        latent_shape = metadata["latent_shape"]
        strings = metadata["strings"]
        key = metadata["key"]  # Retrieve key for dequantization
        cdf_qp = metadata.get("cdf_qp")  # CDF quantization points (only for CDF mode)

        # Decompress
        out_dec = self.model.decompress(strings, latent_shape)
        x_hat = out_dec["x_hat"]

        # Remove padding
        x_hat = F.pad(x_hat, metadata["spatial_unpad"])

        # Clamp to valid quantized range [0, 1) before dequantization
        # Neural network codec may produce values slightly outside [0, 1) range
        max_quantized_value = (2**self.config.bit_depth - 1) / (2**self.config.bit_depth)
        x_hat = x_hat.clamp(0, max_quantized_value)

        # Dequantize (stays on device)
        result = self._dequantize(x_hat, key, cdf_qp=cdf_qp)

        # Convert to original dtype
        result = result.to(dtype=metadata["original_dtype"])
        return result


# =============================================================================
# Evaluation Pipeline
# =============================================================================


def setup_logging(args, log_level: int = logging.INFO) -> logging.Logger:
    """Setup dual logging to terminal and file."""
    os.makedirs(args.output, exist_ok=True)

    logger = logging.getLogger("dtufc_eval")
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    exp_name = f"dtufc_{args.arch}_{args.handler}_{args.strategy}"
    log_path = os.path.join(args.output, f"{exp_name}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Log file:   {log_path}")
    return logger


def roundtrip_once(args, codec: DTUFCCodec):
    """Single sample roundtrip evaluation."""
    # Load features using handler
    logger.info(f"\n[1/4] FRONTEND: Loading features from {args.input}...")
    handler = get_handler(args.handler, log_fn=logger.info)

    t_load_start = time.perf_counter()
    features = handler.parse(args.input)
    t_load = time.perf_counter() - t_load_start
    logger.info(f"⌛️ [1/4] FRONTEND: Load time: {t_load:.3f}s")
    handler.log(f"\n{features.summary()}")

    # Pack features using handler
    logger.info(f"\n[2/4] FRONTEND: Pack + Encode (strategy: {args.strategy})...")

    t_frontend_start = time.perf_counter()
    packed = handler.pack(features, args.strategy)

    # Encode all packed tensors
    feature_stats = {}
    meta_data = {}
    for name, (ori_tensor, ori_shape, *feat_key) in packed.items():
        ori_float_tensor = ori_tensor.float()
        numel = ori_tensor.numel()
        # Encode (pass name as key for per-key quantization)
        enc_bytes, encode_meta = codec.encode(ori_float_tensor, key=name)

        bpfp = (enc_bytes * 8) / numel
        logger.info(f"      {name}: {enc_bytes:,}B, BPFP={bpfp:.4f}")

        meta_data[name] = {
            "ori_shape": ori_shape,
            "ori_dtype": ori_tensor.dtype,
            "ori_device": ori_tensor.device,
            "numel": numel,
            "enc_bytes": enc_bytes,
            "feat_key": feat_key[0] if len(feat_key) == 1 else feat_key,
            **encode_meta,
        }
        feature_stats[name] = {
            "ori_shape": ori_shape,
            "ori_dtype": ori_tensor.dtype,
            "ori_bytes": ori_tensor.element_size(),
            "ori_device": ori_tensor.device,
            "ori_numel": numel,
            "enc_bytes": enc_bytes,  # total_bytes
            "enc_latent_shape": encode_meta["latent_shape"],
            # "enc_strings": encode_meta["strings"], # 对于特征而言，占用的空间有点大了
            "enc_bpfp": encode_meta["bpfp"],
        }

    t_frontend = time.perf_counter() - t_frontend_start
    logger.info(f"⌛️ [2/4] FRONTEND: Frontend time: {t_frontend:.3f}s (Pack+Encode)")

    # Decode
    logger.info("\n[3/4] BACKEND: Decode + Unpack...")
    t_backend_start = time.perf_counter()

    decoded_tensors = {}
    for name, meta in meta_data.items():
        decoded = codec.decode(meta)
        decoded_tensors[name] = (
            decoded.to(dtype=meta["ori_dtype"], device=meta["ori_device"]),
            meta,
        )

    # Unpack to original structure
    restored_data = handler.unpack(decoded_tensors)

    t_backend = time.perf_counter() - t_backend_start
    logger.info(f"⌛️ [3/4] BACKEND: Backend time: {t_backend:.3f}s")

    # Compute MSE Breakdown
    # =========================================================================
    # MSE Breakdown:
    # 1. Quantization MSE: original -> truncate+quantize+dequantize (before codec)
    # 2. Codec MSE: pre_codec_dequantized -> decoded (codec compression/decompression)
    # 3. Total MSE: original -> decoded (end-to-end)
    # =========================================================================
    logger.info("\n[4/4] METRICS: Computing MSE Breakdown...")

    # Collect pre-codec dequantized tensors for MSE breakdown
    pre_codec_dict = {}
    for name, meta in meta_data.items():
        _quantized_tensor = meta.pop("quantized_tensor")
        pre_codec_dict[name] = codec._dequantize(_quantized_tensor, key=meta["key"], cdf_qp=meta["cdf_qp"])
    pre_codec_restored = handler.unpack({name: (tensor, meta_data[name]) for name, tensor in pre_codec_dict.items()})

    original_dict = features.to_dict()
    pre_codec_restored_dict = pre_codec_restored.to_dict()
    final_restored_dict = restored_data.to_dict()

    # Compute per-key MSE breakdown
    logger.info("  Per-key MSE Breakdown (Quant=Truncation+Quantization, Codec=Compression):")
    logger.info(f"  {'Key':<45} {'Quant-MSE':>12} {'Total-MSE':>12}")
    logger.info("  " + "-" * 85)

    total_quant_mse_sum = 0.0
    total_final_mse_sum = 0.0
    total_count = 0
    for key in original_dict.keys():
        assert key in feature_stats, f"Key {key} not found in feature_stats: {feature_stats.keys()}"

        # 类型转换只针对MSE的计算，不影响后续保存
        final = final_restored_dict[key].float()
        pre_codec = pre_codec_restored_dict[key].float().to(device=final.device)
        orig = original_dict[key].float().to(device=final.device)

        count = orig.numel()
        assert count > 0, key
        total_count += count

        # Quantization MSE: original vs pre-codec (truncation + quantization error)
        quant_mse_sum = ((orig - pre_codec) ** 2).sum().item()
        total_quant_mse_sum += quant_mse_sum
        # Total MSE: original vs final (end-to-end error)
        final_mse_sum = ((orig - final) ** 2).sum().item()
        total_final_mse_sum += final_mse_sum

        quant_mse = quant_mse_sum / count
        final_mse = final_mse_sum / count
        feature_stats[key]["quant_mse"] = quant_mse
        feature_stats[key]["final_mse"] = final_mse
        logger.info(f"  {key:<45} {quant_mse:>12.8f} {final_mse:>12.8f}")

    # Compute averages
    assert total_count > 0
    avg_quant_mse = total_quant_mse_sum / total_count
    avg_mse = total_final_mse_sum / total_count

    logger.info("  " + "-" * 85)
    logger.info(f"  {'TOTAL':<45} {avg_quant_mse:>12.8f} {avg_mse:>12.8f}")
    logger.info(f"  (elements={total_count:,})")

    # Summary
    sample_bits = sample_equ_bits = total_bytes = total_elements = 0
    for name, meta in meta_data.items():
        feature_bytes = meta["enc_bytes"]
        feature_bits = feature_bytes * 8
        feature_equ_bits = feature_bits * 32 / DTYPE_TORCHTYPE_TO_BITS[meta["ori_dtype"]]

        total_bytes += feature_bytes
        sample_bits += feature_bits
        sample_equ_bits += feature_equ_bits
        total_elements += meta["numel"]

    assert total_elements == total_count, (total_elements, total_count)
    avg_bpfp = sample_bits / total_elements
    avg_ebpfp = sample_equ_bits / total_elements

    t_total = t_load + t_frontend + t_backend
    logger.info(
        tabulate(
            [
                ["SAMPLE-WISE STATISTICS", ""],
                SEPARATING_LINE,
                ["Handler", args.handler],
                ["Strategy", args.strategy],
                ["Architecture", args.arch],
                SEPARATING_LINE,
                ["Total Elements", total_elements],
                ["Total Bytes", total_bytes],
                ["BPFP", f"{avg_bpfp:.4f} bits/point"],
                ["EBPFP", f"{avg_ebpfp:.4f} equivalent bits/point"],
                ["MSE", f"{avg_mse:.6f}"],
                SEPARATING_LINE,
                [
                    f"Time:  {t_total:.3f}s",
                    f"Load: {t_load:.3f}s, Pack+Encode: {t_frontend:.3f}s, Decode+Unpack: {t_backend:.3f}s",
                ],
            ]
        )
    )

    # =========================================================================
    # Save Restored Features to Output Directory (zst format)
    # 保存重构后的特征到 output 文件夹，格式与 main_for_qwen3.py 相同
    # =========================================================================
    output_path = Path(args.output) / Path(args.input).name

    reconstructed_data = handler.restore_format(restored_data)
    if args.update:
        assert output_path.exists(), f"Output file does not exist for update: {output_path}"
        original_features = load_zst_tensor(output_path.as_posix())
        assert recursive_check_equal(original_features["features"], reconstructed_data)
        logger.info(f"✅ Verified restored features match original for update: {output_path}")

    # 将重构的特征转换为可导出的形式并保存
    save_compressed(
        data={
            "features": reconstructed_data,
            "metadata": {
                "input_file": args.input,
                "arch": args.arch,
                "handler": args.handler,
                "strategy": args.strategy,
                "transform_type": args.transform_type,
                "bit_depth": args.bit_depth,
                "bpfp": avg_bpfp,
                "ebpfp": avg_ebpfp,
                "mse": avg_mse,
                "frontend_seconds": t_frontend,
                "backend_seconds": t_backend,
                **feature_stats,
            },
        },
        path=str(output_path),
    )
    logger.info(f"💾 Converting with {avg_mse:.04f} MSE:\n\tfrom {args.input}\n\tto {output_path}")
    return {"bpfp": avg_bpfp, "ebpfp": avg_ebpfp, "mse": avg_mse, "t_total": t_total}


def featurecoding_roundtrip(args):
    """Batch processing wrapper for roundtrip evaluation."""
    global logger
    logger = setup_logging(args)

    # Initialize codec
    config = DTUFCCodecConfig(
        arch=args.arch,
        checkpoint_path=args.checkpoint,
        handler=args.handler,
        transform_type=args.transform_type,
        transform_mapping=args.transform_mapping,
        bit_depth=args.bit_depth,
        device="cpu" if args.cpu else "cuda:0",
    )
    logger.info(config)

    codec = DTUFCCodec(config, print_fn=logger.info)
    logger.info(
        tabulate(
            [
                ["Handler", args.handler],
                ["Strategy", args.strategy],
                ["Architecture", args.arch],
                ["Checkpoint", args.checkpoint],
                ["Transform type", args.transform_type],
                ["Transform config", args.transform_mapping],
                ["Input", args.input],
                ["Output", args.output],
            ],
        )
    )

    # Process directory or single file
    if os.path.isdir(args.input):
        # Collect both .zst and .npy files
        feature_files = sorted(
            set(glob.glob(os.path.join(args.input, "*.zst"))) | set(glob.glob(os.path.join(args.input, "*.npy")))
        )
        if not feature_files:
            logger.info(f"No .zst or .npy files found in: {args.input}")
            return
        if args.num_samples and args.num_samples > 0:
            logger.info(f"Files found: {len(feature_files)} (limited to first {args.num_samples})")
            feature_files = feature_files[: args.num_samples]
        else:
            logger.info(f"Files found: {len(feature_files)}")
        logger.info("-" * 70)

        all_stats = []
        for idx, feature_file in enumerate(feature_files, 1):
            logger.info(f"\n 💪 Processing: {Path(feature_file).as_posix()} ({idx}/{len(feature_files)})")
            file_args = argparse.Namespace(**vars(args))
            file_args.input = feature_file
            all_stats.append(roundtrip_once(file_args, codec))

        # Aggregate statistics
        if all_stats:
            total_files = len(all_stats)
            avg_bpfp = sum(s["bpfp"] for s in all_stats) / total_files
            avg_ebpfp = sum(s["ebpfp"] for s in all_stats) / total_files
            avg_mse = sum(s["mse"] for s in all_stats) / total_files
            avg_time = sum(s["t_total"] for s in all_stats) / total_files
            logger.info(
                tabulate(
                    [
                        ["TOTAL PROCESSING SUMMARY", ""],
                        SEPARATING_LINE,
                        ["Total files", total_files],
                        ["Avg BPFP", f"{avg_bpfp:.4f} bits/point"],
                        ["Avg EBPFP", f"{avg_ebpfp:.4f} equivalent bits/point"],
                        ["Avg MSE", f"{avg_mse:.6f}"],
                        ["Avg Time", f"{avg_time:.3f}s"],
                    ],
                )
            )
    else:
        roundtrip_once(args, codec)


def list_models(args):
    """List available DT-UFC architectures."""
    print("\n" + "-" * 60)
    print("Available DT-UFC Architectures")
    print("-" * 60)
    for arch in AVAILABLE_CODECS:
        print(f"  {arch}")
    print("-" * 60)
    print("Note: DT-UFC requires custom pre-trained weights (--checkpoint)")
    print("-" * 60 + "\n")


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="DT-UFC Evaluation Script", formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list-models command
    list_parser = subparsers.add_parser("list-models", help="List available architectures")
    list_parser.set_defaults(func=list_models)

    # roundtrip command
    rt_parser = subparsers.add_parser("roundtrip", help="Roundtrip evaluation")
    # Input/Output
    rt_parser.add_argument("input", type=str, help="Input .zst/.npy file or directory")
    rt_parser.add_argument("--output", type=str, required=True, help="Output directory")
    rt_parser.add_argument("--num_samples", type=int, default=None, help="Limit to first N samples")
    # Handler settings
    rt_parser.add_argument("--handler", type=str, required=True, choices=list(AVAILABLE_HANDLERS.keys()), help="Handler type")
    rt_parser.add_argument("--strategy", type=str, default="individual", help="Packing strategy", choices=["individual"])
    # Model settings
    rt_parser.add_argument("--arch", type=str, required=True, choices=AVAILABLE_CODECS, help="Model architecture")
    rt_parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained weights")
    rt_parser.add_argument("--transform_type", type=str, default="nonlinear", choices=["nonlinear"], help="Transform type (nonlinear)")
    rt_parser.add_argument("--transform_mapping", type=str, default="../lmfc_utils/transform_mapping/10samples-8bits/mapping.json", help="Path to per-key mapping config JSON")
    rt_parser.add_argument("--bit_depth", type=int, default=8, choices=[8], help="Bit depth for quantization")
    rt_parser.add_argument("--cpu", action="store_true", help="Use CPU for inference")
    rt_parser.add_argument("--update", action="store_true", help="Update feature file in-place (not recommended)")
    rt_parser.set_defaults(func=featurecoding_roundtrip)
    # fmt: on

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
