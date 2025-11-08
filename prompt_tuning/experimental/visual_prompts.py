# Copyright 2024 Google.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visual Prompt Tuning for vision and vision-language models.

This module implements prompt tuning techniques specifically designed for
visual inputs, including pixel-space prompts, patch-level prompts, and
prompts for vision transformers.

References:
  - Jia et al. (2022). "Visual Prompt Tuning."
    https://arxiv.org/abs/2203.12119
  - Bahng et al. (2022). "Visual Prompting via Image Inpainting."
    https://arxiv.org/abs/2209.00647
  - Zhou et al. (2022). "Learning to Prompt for Vision-Language Models."
    https://arxiv.org/abs/2109.01134
"""

from typing import Callable, Optional, Sequence, Tuple
import flax.linen as nn
import jax.numpy as jnp
from flaxformer.types import Array, DType


class PixelSpacePrompt(nn.Module):
  """Prompts in pixel space for vision models.
  
  This module adds learnable perturbations to input images, either as
  borders/padding or as patches within the image.
  
  Attributes:
    image_size: Size of input images (height, width).
    prompt_size: Size of the prompt region (pixels or patches).
    num_channels: Number of image channels (typically 3 for RGB).
    prompt_location: Where to place prompts ('border', 'corners', 'random').
    prompt_init: Initializer for prompt pixels.
    dtype: Data type for parameters.
  """
  image_size: Tuple[int, int]
  prompt_size: int
  num_channels: int = 3
  prompt_location: str = 'border'
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform(scale=0.1)
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, images: Array) -> Array:
    """Add pixel-space prompts to images.
    
    Args:
      images: Input images with shape [B, H, W, C].
      
    Returns:
      Images with prompts added, shape depends on prompt_location.
    """
    batch_size = images.shape[0]
    height, width = self.image_size
    
    if self.prompt_location == 'border':
      # Add prompt as a border around the image
      # Top and bottom borders
      top_border = self.param(
          'top_border',
          self.prompt_init,
          (self.prompt_size, width, self.num_channels),
          self.dtype)
      bottom_border = self.param(
          'bottom_border',
          self.prompt_init,
          (self.prompt_size, width, self.num_channels),
          self.dtype)
      
      # Left and right borders (excluding corners to avoid double-counting)
      left_border = self.param(
          'left_border',
          self.prompt_init,
          (height, self.prompt_size, self.num_channels),
          self.dtype)
      right_border = self.param(
          'right_border',
          self.prompt_init,
          (height, self.prompt_size, self.num_channels),
          self.dtype)
      
      # Expand borders to batch size
      top_border_batch = jnp.tile(
          jnp.expand_dims(top_border, 0), [batch_size, 1, 1, 1])
      bottom_border_batch = jnp.tile(
          jnp.expand_dims(bottom_border, 0), [batch_size, 1, 1, 1])
      left_border_batch = jnp.tile(
          jnp.expand_dims(left_border, 0), [batch_size, 1, 1, 1])
      right_border_batch = jnp.tile(
          jnp.expand_dims(right_border, 0), [batch_size, 1, 1, 1])
      
      # Concatenate: top border + (left border + image + right border) + bottom border
      middle = jnp.concatenate(
          [left_border_batch, images, right_border_batch], axis=2)
      result = jnp.concatenate(
          [top_border_batch, middle, bottom_border_batch], axis=1)
      
      return result
    
    elif self.prompt_location == 'corners':
      # Add small prompt patches in the corners
      corner_size = self.prompt_size
      
      # Four corner prompts
      top_left = self.param(
          'top_left_corner',
          self.prompt_init,
          (corner_size, corner_size, self.num_channels),
          self.dtype)
      top_right = self.param(
          'top_right_corner',
          self.prompt_init,
          (corner_size, corner_size, self.num_channels),
          self.dtype)
      bottom_left = self.param(
          'bottom_left_corner',
          self.prompt_init,
          (corner_size, corner_size, self.num_channels),
          self.dtype)
      bottom_right = self.param(
          'bottom_right_corner',
          self.prompt_init,
          (corner_size, corner_size, self.num_channels),
          self.dtype)
      
      # Copy images to avoid modifying input
      result = jnp.copy(images)
      
      # Place corners (using dynamic slice update)
      # Top-left
      result = result.at[:, :corner_size, :corner_size, :].set(
          jnp.tile(jnp.expand_dims(top_left, 0), [batch_size, 1, 1, 1]))
      # Top-right
      result = result.at[:, :corner_size, -corner_size:, :].set(
          jnp.tile(jnp.expand_dims(top_right, 0), [batch_size, 1, 1, 1]))
      # Bottom-left
      result = result.at[:, -corner_size:, :corner_size, :].set(
          jnp.tile(jnp.expand_dims(bottom_left, 0), [batch_size, 1, 1, 1]))
      # Bottom-right
      result = result.at[:, -corner_size:, -corner_size:, :].set(
          jnp.tile(jnp.expand_dims(bottom_right, 0), [batch_size, 1, 1, 1]))
      
      return result
    
    else:
      raise ValueError(
          f"Unknown prompt_location: {self.prompt_location}. "
          f"Must be one of: 'border', 'corners'")


class PatchPrompt(nn.Module):
  """Prompts at the patch level for Vision Transformers.
  
  This module adds learnable prompt patches to the sequence of image patches
  in a Vision Transformer, similar to text prompt tuning.
  
  Attributes:
    num_prompts: Number of prompt patches to add.
    patch_dim: Dimension of each patch embedding.
    prompt_init: Initializer for prompt patches.
    dtype: Data type for parameters.
    prompt_location: Where to place prompts ('prefix', 'suffix', 'mixed').
  """
  num_prompts: int
  patch_dim: int
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32
  prompt_location: str = 'prefix'

  @nn.compact
  def __call__(self, patch_embeddings: Array) -> Array:
    """Add prompt patches to patch embeddings.
    
    Args:
      patch_embeddings: Patch embeddings with shape [B, N, D] where N is
        the number of patches and D is the patch dimension.
      
    Returns:
      Patch embeddings with prompts, shape [B, N + num_prompts, D].
    """
    batch_size = patch_embeddings.shape[0]
    
    # Create prompt patches
    prompts = self.param(
        'prompt_patches',
        self.prompt_init,
        (self.num_prompts, self.patch_dim),
        self.dtype)
    
    # Expand to batch size
    prompts_batch = jnp.tile(
        jnp.expand_dims(prompts, 0), [batch_size, 1, 1])
    
    if self.prompt_location == 'prefix':
      # Add prompts at the beginning
      return jnp.concatenate([prompts_batch, patch_embeddings], axis=1)
    elif self.prompt_location == 'suffix':
      # Add prompts at the end
      return jnp.concatenate([patch_embeddings, prompts_batch], axis=1)
    elif self.prompt_location == 'mixed':
      # Interleave prompts with patches
      # Split prompts into two groups
      mid_point = self.num_prompts // 2
      prefix_prompts = prompts_batch[:, :mid_point, :]
      suffix_prompts = prompts_batch[:, mid_point:, :]
      return jnp.concatenate(
          [prefix_prompts, patch_embeddings, suffix_prompts], axis=1)
    else:
      raise ValueError(
          f"Unknown prompt_location: {self.prompt_location}. "
          f"Must be one of: 'prefix', 'suffix', 'mixed'")


class DeepPatchPrompt(nn.Module):
  """Deep prompts at the patch level for multiple ViT layers.
  
  Similar to P-Tuning v2 but for vision transformers, adding prompts
  at multiple layers of the visual encoder.
  
  Attributes:
    num_layers: Number of ViT layers.
    num_prompts: Number of prompt patches per layer.
    patch_dim: Dimension of patch embeddings.
    shared_prompts: Whether to share prompts across layers.
    prompt_init: Initializer for prompts.
    dtype: Data type for parameters.
  """
  num_layers: int
  num_prompts: int
  patch_dim: int
  shared_prompts: bool = False
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self) -> Array:
    """Generate deep patch prompts for all layers.
    
    Returns:
      Deep prompts with shape [num_layers, num_prompts, patch_dim].
    """
    if self.shared_prompts:
      # Single set of prompts shared across layers
      prompts = self.param(
          'shared_prompts',
          self.prompt_init,
          (self.num_prompts, self.patch_dim),
          self.dtype)
      return jnp.tile(
          jnp.expand_dims(prompts, 0), [self.num_layers, 1, 1])
    else:
      # Separate prompts for each layer
      return self.param(
          'layer_prompts',
          self.prompt_init,
          (self.num_layers, self.num_prompts, self.patch_dim),
          self.dtype)


class VisionLanguagePrompt(nn.Module):
  """Prompts for vision-language models like CLIP.
  
  This module creates coordinated prompts for both the visual and textual
  encoders of a vision-language model.
  
  Attributes:
    num_text_prompts: Number of text prompt tokens.
    num_visual_prompts: Number of visual prompt patches.
    text_dim: Dimension of text embeddings.
    visual_dim: Dimension of visual embeddings.
    shared_projection: Whether to use a shared projection between modalities.
    prompt_init: Initializer for prompts.
    dtype: Data type for parameters.
  """
  num_text_prompts: int
  num_visual_prompts: int
  text_dim: int
  visual_dim: int
  shared_projection: bool = True
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self) -> Tuple[Array, Array]:
    """Generate coordinated text and visual prompts.
    
    Returns:
      Tuple of (text_prompts, visual_prompts) with shapes:
        - text_prompts: [num_text_prompts, text_dim]
        - visual_prompts: [num_visual_prompts, visual_dim]
    """
    if self.shared_projection:
      # Use a shared latent space and project to each modality
      shared_dim = min(self.text_dim, self.visual_dim)
      
      # Shared prompt representations
      shared_text_prompts = self.param(
          'shared_text_prompts',
          self.prompt_init,
          (self.num_text_prompts, shared_dim),
          self.dtype)
      shared_visual_prompts = self.param(
          'shared_visual_prompts',
          self.prompt_init,
          (self.num_visual_prompts, shared_dim),
          self.dtype)
      
      # Project to modality-specific dimensions
      text_prompts = nn.Dense(
          features=self.text_dim,
          dtype=self.dtype,
          name='text_projection')(shared_text_prompts)
      visual_prompts = nn.Dense(
          features=self.visual_dim,
          dtype=self.dtype,
          name='visual_projection')(shared_visual_prompts)
      
      return text_prompts, visual_prompts
    else:
      # Independent prompts for each modality
      text_prompts = self.param(
          'text_prompts',
          self.prompt_init,
          (self.num_text_prompts, self.text_dim),
          self.dtype)
      visual_prompts = self.param(
          'visual_prompts',
          self.prompt_init,
          (self.num_visual_prompts, self.visual_dim),
          self.dtype)
      
      return text_prompts, visual_prompts


class AdaptiveVisualPrompt(nn.Module):
  """Visual prompts that adapt based on image content.
  
  This module generates visual prompts conditioned on the input image,
  allowing the prompt to adapt to different types of visual content.
  
  Attributes:
    num_prompts: Number of prompt patches.
    patch_dim: Dimension of patch embeddings.
    hidden_dim: Hidden dimension of the generator network.
    prompt_init: Initializer for base prompts.
    dtype: Data type for parameters.
  """
  num_prompts: int
  patch_dim: int
  hidden_dim: int = 256
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, patch_embeddings: Array) -> Array:
    """Generate adaptive visual prompts based on image patches.
    
    Args:
      patch_embeddings: Patch embeddings with shape [B, N, D].
      
    Returns:
      Adaptive prompts with shape [B, num_prompts, D].
    """
    batch_size = patch_embeddings.shape[0]
    
    # Base prompts
    base_prompts = self.param(
        'base_prompts',
        self.prompt_init,
        (self.num_prompts, self.patch_dim),
        self.dtype)
    
    # Compute image representation (global average pooling)
    image_repr = jnp.mean(patch_embeddings, axis=1)  # [B, D]
    
    # Generate adaptation parameters
    hidden = nn.Dense(
        features=self.hidden_dim,
        dtype=self.dtype,
        name='adapter_hidden')(image_repr)
    hidden = nn.gelu(hidden)
    
    # Generate prompt adjustments
    adjustments = nn.Dense(
        features=self.num_prompts * self.patch_dim,
        dtype=self.dtype,
        name='adapter_output')(hidden)
    adjustments = jnp.reshape(
        adjustments, (batch_size, self.num_prompts, self.patch_dim))
    
    # Combine base prompts with adjustments
    adapted_prompts = (
        jnp.expand_dims(base_prompts, 0) + adjustments)
    
    return adapted_prompts


def create_visual_prompt_mask(
    image_size: Tuple[int, int],
    prompt_size: int,
    prompt_location: str) -> Array:
  """Create a binary mask indicating prompt regions in an image.
  
  Args:
    image_size: Size of the image (height, width).
    prompt_size: Size of the prompt region.
    prompt_location: Location of prompts ('border', 'corners').
    
  Returns:
    Binary mask with shape [H, W] where 1 indicates prompt regions.
  """
  height, width = image_size
  mask = jnp.zeros((height, width), dtype=jnp.float32)
  
  if prompt_location == 'border':
    # Mark border regions
    mask = mask.at[:prompt_size, :].set(1.0)  # Top
    mask = mask.at[-prompt_size:, :].set(1.0)  # Bottom
    mask = mask.at[:, :prompt_size].set(1.0)  # Left
    mask = mask.at[:, -prompt_size:].set(1.0)  # Right
  elif prompt_location == 'corners':
    # Mark corner regions
    mask = mask.at[:prompt_size, :prompt_size].set(1.0)  # Top-left
    mask = mask.at[:prompt_size, -prompt_size:].set(1.0)  # Top-right
    mask = mask.at[-prompt_size:, :prompt_size].set(1.0)  # Bottom-left
    mask = mask.at[-prompt_size:, -prompt_size:].set(1.0)  # Bottom-right
  
  return mask
