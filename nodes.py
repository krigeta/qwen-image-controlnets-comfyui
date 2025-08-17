# ComfyUI nodes for Qwen Image Blockwise ControlNet
# Based on DiffSynth Studio's actual implementation
# Save as: custom_nodes/ComfyUI-Qwen-Blockwise-ControlNet/nodes.py

import torch
import torch.nn as nn
import comfy.model_management
import comfy.utils
import folder_paths
import os
import safetensors.torch
import numpy as np
from PIL import Image


class RMSNorm(nn.Module):
    """RMS Normalization from DiffSynth"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class BlockWiseControlBlock(torch.nn.Module):
    """Individual control block for blockwise ControlNet"""
    def __init__(self, dim: int = 3072):
        super().__init__()
        self.x_rms = RMSNorm(dim, eps=1e-6)
        self.y_rms = RMSNorm(dim, eps=1e-6)
        self.input_proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(self, x, y):
        x, y = self.x_rms(x), self.y_rms(y)
        x = self.input_proj(x + y)
        x = self.act(x)
        x = self.output_proj(x)
        return x
    
    def init_weights(self):
        # Zero initialize output_proj (following ControlNet tradition)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)


class QwenImageBlockWiseControlNet(torch.nn.Module):
    """Qwen Image Blockwise ControlNet - exact copy from DiffSynth"""
    def __init__(
        self,
        num_layers: int = 60,
        in_dim: int = 64, 
        dim: int = 3072,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dim = dim
        
        self.img_in = nn.Linear(in_dim, dim)
        self.controlnet_blocks = nn.ModuleList([
            BlockWiseControlBlock(dim) for _ in range(num_layers)
        ])
    
    def init_weight(self):
        """Initialize weights following DiffSynth's approach"""
        nn.init.zeros_(self.img_in.weight)
        nn.init.zeros_(self.img_in.bias)
        for block in self.controlnet_blocks:
            block.init_weights()
    
    def process_controlnet_conditioning(self, controlnet_conditioning):
        """Process control conditioning through input projection"""
        return self.img_in(controlnet_conditioning)
    
    def blockwise_forward(self, img, controlnet_conditioning, block_id):
        """Apply control for a specific transformer block"""
        return self.controlnet_blocks[block_id](img, controlnet_conditioning)


class QwenImageBlockwiseControlNetLoader:
    """Load Qwen Image Blockwise ControlNet model"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "controlnet_name": (folder_paths.get_filename_list("controlnet"), ),
            }
        }
    
    RETURN_TYPES = ("QWEN_BLOCKWISE_CONTROLNET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Qwen Image/ControlNet"

    def load_controlnet(self, controlnet_name):
        controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
        
        # Load the safetensors file
        try:
            state_dict = safetensors.torch.load_file(controlnet_path)
        except:
            state_dict = torch.load(controlnet_path, map_location="cpu")
        
        device = comfy.model_management.get_torch_device()
        
        # Create controlnet instance with proper dimensions
        # These may need adjustment based on your specific model
        controlnet = QwenImageBlockWiseControlNet(
            num_layers=60,  # Standard Qwen-Image has 60 layers
            in_dim=64,      # Input dimension for control features  
            dim=3072,       # Hidden dimension
        )
        
        # Load state dict
        missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys in Qwen blockwise controlnet: {missing}")
        if unexpected:
            print(f"Unexpected keys in Qwen blockwise controlnet: {unexpected}")
        
        controlnet.to(device)
        controlnet.eval()
        
        return (controlnet,)


class QwenImageBlockwiseControlNetApply:
    """Apply Qwen Image Blockwise ControlNet to DiffSynth pipeline"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("QwenImageDiffSynthiPipe",),  # From existing custom node
                "controlnet": ("QWEN_BLOCKWISE_CONTROLNET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("QwenImageDiffSynthiPipe",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "Qwen Image/ControlNet"

    def apply_controlnet(self, pipe, controlnet, image, strength):
        # Convert ComfyUI image format to what DiffSynth expects
        if len(image.shape) == 4:  # (B, H, W, C)
            # Convert to PIL for processing
            image_pil = self.comfy_to_pil(image)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Create a modified pipeline that includes the blockwise controlnet
        modified_pipe = self.create_controlnet_pipeline(pipe, controlnet, image_pil, strength)
        
        return (modified_pipe,)
    
    def comfy_to_pil(self, image):
        """Convert ComfyUI tensor to PIL Image"""
        i = 255. * image.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img
    
    def create_controlnet_pipeline(self, original_pipe, controlnet, control_image, strength):
        """Create a wrapper pipeline that includes blockwise controlnet"""
        
        class ControlNetPipeWrapper:
            def __init__(self, pipe, controlnet, control_image, strength):
                self.pipe = pipe
                self.controlnet = controlnet
                self.control_image = control_image
                self.strength = strength
                # Copy all attributes from original pipe
                for attr in dir(pipe):
                    if not attr.startswith('_') and not callable(getattr(pipe, attr)):
                        setattr(self, attr, getattr(pipe, attr))
            
            def __call__(self, *args, **kwargs):
                # Add blockwise controlnet inputs to kwargs
                kwargs['blockwise_controlnet_inputs'] = [{
                    'image': self.control_image,
                    'strength': self.strength
                }]
                
                return self.pipe(*args, **kwargs)
            
            def __getattr__(self, name):
                # Delegate any missing attributes to original pipe
                return getattr(self.pipe, name)
        
        return ControlNetPipeWrapper(original_pipe, controlnet, control_image, strength)


class QwenImageCannyPreprocessor:
    """Canny edge detection preprocessor for Qwen Image ControlNet"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edge"
    CATEGORY = "Qwen Image/ControlNet"

    def detect_edge(self, image, low_threshold, high_threshold):
        try:
            import cv2
        except ImportError:
            raise Exception("opencv-python is required for canny edge detection. Install with: pip install opencv-python")
        
        # Convert to numpy
        np_image = (image.cpu().numpy() * 255).astype(np.uint8)
        
        output_images = []
        for img in np_image:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply Canny edge detection
            canny = cv2.Canny(gray, low_threshold, high_threshold)
            
            # Convert back to RGB
            canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
            output_images.append(canny_rgb)
        
        # Convert back to tensor
        output = torch.from_numpy(np.array(output_images)).float() / 255.0
        
        return (output,)


class QwenImageDepthPreprocessor:
    """Depth estimation preprocessor for Qwen Image ControlNet"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"
    CATEGORY = "Qwen Image/ControlNet"

    def estimate_depth(self, image):
        try:
            from transformers import pipeline
        except ImportError:
            raise Exception("transformers is required for depth estimation. Install with: pip install transformers")
        
        # Use MiDaS for depth estimation
        depth_estimator = pipeline('depth-estimation', model='Intel/dpt-large')
        
        np_image = (image.cpu().numpy() * 255).astype(np.uint8)
        
        output_images = []
        for img in np_image:
            # Convert to PIL
            pil_image = Image.fromarray(img)
            
            # Estimate depth
            depth = depth_estimator(pil_image)
            depth_image = depth['depth']
            
            # Convert to RGB and normalize
            depth_array = np.array(depth_image)
            depth_normalized = (depth_array / depth_array.max() * 255).astype(np.uint8)
            depth_rgb = np.stack([depth_normalized] * 3, axis=-1)
            
            output_images.append(depth_rgb)
        
        # Convert back to tensor
        output = torch.from_numpy(np.array(output_images)).float() / 255.0
        
        return (output,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenImageBlockwiseControlNetLoader": QwenImageBlockwiseControlNetLoader,
    "QwenImageBlockwiseControlNetApply": QwenImageBlockwiseControlNetApply,
    "QwenImageCannyPreprocessor": QwenImageCannyPreprocessor,
    "QwenImageDepthPreprocessor": QwenImageDepthPreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageBlockwiseControlNetLoader": "Qwen Image Blockwise ControlNet Loader",
    "QwenImageBlockwiseControlNetApply": "Qwen Image Blockwise ControlNet Apply",
    "QwenImageCannyPreprocessor": "Qwen Image Canny Preprocessor",
    "QwenImageDepthPreprocessor": "Qwen Image Depth Preprocessor",
}
