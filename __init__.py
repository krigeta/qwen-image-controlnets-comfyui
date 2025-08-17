# __init__.py for ComfyUI-Qwen-Blockwise-ControlNet
# Save as: custom_nodes/ComfyUI-Qwen-Blockwise-ControlNet/__init__.py

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Package metadata
__version__ = "1.0.0"
__author__ = "Qwen ControlNet Integration"
__description__ = "Qwen Image Blockwise ControlNet nodes for ComfyUI using DiffSynth Studio"
