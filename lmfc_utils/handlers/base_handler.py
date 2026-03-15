from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class FeatureData:
    """Container for parsed model features.

    Attributes:
        model_type: Type of model (e.g., "kimiaudio", "qwen", "dinov3")
        source_path: Path to the original feature file
        dtype: Original tensor dtype
        tensors: Dictionary of tensor name -> tensor
        metadata: Additional metadata from the feature file
    """

    model_type: str
    source_path: str
    dtype: torch.dtype
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a summary string of the feature data."""
        lines = [
            f"FeatureData Summary ({self.model_type})",
            "-" * 50,
            f"Source: {self.source_path}",
            f"Dtype: {self.dtype}",
            f"Tensors: {len(self.tensors)}",
        ]

        total_elements = 0
        for name, tensor in self.tensors.items():
            shape_str = "x".join(map(str, tensor.shape))
            lines.append(f"  - {name}: [{shape_str}] {tensor.dtype}")
            total_elements += tensor.numel()

        total_bytes = total_elements * (2 if self.dtype in [torch.float16, torch.bfloat16] else 4)
        lines.append(f"\nTotal elements: {total_elements:,}")
        lines.append(f"Total size: {total_bytes / 1024 / 1024:.2f} MB")

        return "\n".join(lines)


class BaseHandler(ABC):
    """Abstract base class for model feature handlers.

    Each handler is responsible for:
    1. Parsing model-specific feature files
    2. Packing tensors for compression
    3. Unpacking tensors after decompression
    """

    model_type: str = "base"
    SUPPORTED_STRATEGIES: Dict[str, type] = None

    def __init__(self, log_fn: Callable[[str], None] = None):
        """Initialize the handler.

        Args:
            log_fn: Function to use for logging.
        """
        self.log_fn = log_fn
        self._packer: Optional[PackerBase] = None

    @abstractmethod
    def parse(self, path: str) -> FeatureData:
        """Parse a feature file and return FeatureData.

        Args:
            path: Path to the feature file.

        Returns:
            FeatureData containing parsed tensors and metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def restore_format(self, features: FeatureData) -> Dict[str, torch.Tensor]:
        """Restore features to a dictionary as the original format.

        Args:
            features: FeatureData containing parsed tensors and metadata.
        """
        raise NotImplementedError

    def pack(self, features: FeatureData, strategy: str) -> Dict[str, tuple]:
        """Pack features into tensors ready for compression.

        Args:
            features: Parsed feature data.
            strategy: Packing strategy name.

        Returns:
            Dictionary of packed tensor name -> (tensor, metadata).
        """
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy} for {self.model_type}")

        self._packer = self.SUPPORTED_STRATEGIES[strategy](self.log_fn)
        return self._packer.pack(features)

    def unpack(self, decoded_tensors: Dict[str, tuple]) -> FeatureData:
        """Unpack compressed tensors back to original structure.

        Args:
            decoded_tensors: Dictionary of decoded tensors (tensor, metadata).

        Returns:
            FeatureData with original structure restored.
        """
        if self._packer is None:
            raise RuntimeError("Call pack() first to initialize packer state")
        return self._packer.unpack(decoded_tensors)

    def log(self, message: str):
        """Print a message if verbose mode is enabled."""
        if self.log_fn is not None:
            self.log_fn(message)


class PackerBase(ABC):
    """Base class for packing strategies."""

    def __init__(self, log_fn=None):
        self.log = log_fn or (lambda x: None)
        self._last_reference: Optional[FeatureData] = None
        self._packed_keys: List[str] = []

    def pack(self, features: FeatureData) -> Dict[str, tuple]:
        """Pack features and store state."""
        self._last_reference = features
        packed = self._pack_impl(features)
        self._packed_keys = list(packed.keys())
        return packed

    def unpack(self, tensors: Dict[str, tuple]) -> FeatureData:
        """Unpack using stored state."""
        if self._last_reference is None:
            raise RuntimeError("Call pack() first")
        return self._unpack_impl(tensors)

    @abstractmethod
    def _pack_impl(self, features: FeatureData) -> Dict[str, tuple]:
        raise NotImplementedError

    @abstractmethod
    def _unpack_impl(self, tensors: Dict[str, tuple]) -> FeatureData:
        raise NotImplementedError
