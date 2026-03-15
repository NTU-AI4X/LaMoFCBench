# coding/my_demo/handlers/__init__.py
"""Multi-model feature handlers for tensor compression."""

from .base_handler import BaseHandler, FeatureData
from .dinov3_total import DINOv3TotalFeatures, DINOv3TotalHandler
from .falconmamba import FalconMambaFeatures, FalconMambaHandler
from .kimiaudio import KimiAudioFeatures, KimiAudioHandler
from .qwen import QwenFeatures, QwenHandler
from .sd35 import SD35Features, SD35Handler

AVAILABLE_HANDLERS = {
    "kimiaudio": KimiAudioHandler,
    "qwen": QwenHandler,
    "falconmamba": FalconMambaHandler,
    "dinov3-total": DINOv3TotalHandler,
    "sd35": SD35Handler,
}


def get_handler(model_type: str, **kwargs) -> BaseHandler:
    """Factory function to get the appropriate handler for a model type.

    Args:
        model_type: One of "kimiaudio", "qwen", "falconmamba", "dinov3", "sd35"
        **kwargs: Additional arguments passed to the handler constructor.

    Returns:
        Handler instance for the specified model type.
    """
    if model_type not in AVAILABLE_HANDLERS:
        available = ", ".join(AVAILABLE_HANDLERS.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    return AVAILABLE_HANDLERS[model_type](**kwargs)
