# Copyright 2022 Google LLC.
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

"""ResNet."""


import functools
from typing import Any, Callable, Optional, Sequence, Tuple
from flax import linen as nn
import jax.numpy as jnp
import timm

ModuleDef = Any

# New: LR
class LR(nn.Module):
  """Logistic Regression model."""
  num_classes: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train=True):
    x = x.reshape((x.shape[0], -1))  # Flatten input
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    return x


################################################################################
# ResNet V2
################################################################################
class BasicBlockV2(nn.Module):
  """Basic Block for a ResNet V2."""

  channels: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    preact = self.act(self.norm()(x))
    y = self.conv(self.channels, (3, 3), self.strides)(preact)
    y = self.act(self.norm()(y))
    y = self.conv(self.channels, (3, 3))(y)

    if y.shape != x.shape:
      shortcut = self.conv(self.channels, (1, 1), self.strides)(preact)
    else:
      shortcut = x
    return shortcut + y


class BottleneckBlockV2(nn.Module):
  """Bottleneck Block for a ResNet V2."""

  channels: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    preact = self.act(self.norm()(x))
    y = self.conv(self.channels, (1, 1))(preact)
    y = self.act(self.norm()(y))
    y = self.conv(self.channels, (3, 3), self.strides)(y)
    y = self.act(self.norm()(y))
    y = self.conv(self.channels * 4, (1, 1))(y)

    if y.shape != x.shape:
      shortcut = self.conv(self.channels * 4, (1, 1), self.strides)(preact)
    else:
      shortcut = x

    return shortcut + y


class ResNetV2(nn.Module):
  """ResNet v2.

      K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual
      networks. In ECCV, pages 630–645, 2016.
  """

  stage_sizes: Sequence[int]
  block_class: ModuleDef
  num_classes: Optional[int] = None
  base_channels: int = 64
  act: Callable = nn.relu
  dtype: Any = jnp.float32
  small_image: bool = False
  # if not None, batch statistics are sync-ed across replica according to
  # this axis_name used in pmap
  bn_cross_replica_axis_name: Optional[str] = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(
        nn.BatchNorm, use_running_average=not train,
        momentum=0.9, epsilon=1e-5, dtype=self.dtype,
        axis_name=self.bn_cross_replica_axis_name)

    if self.small_image:  # suitable for Cifar
      x = conv(self.base_channels, (3, 3), padding='SAME')(x)
    else:
      x = conv(self.base_channels, (7, 7), (2, 2), padding=[(3, 3), (3, 3)])(x)
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

    for i, n_blocks in enumerate(self.stage_sizes):
      for j in range(n_blocks):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_class(self.base_channels * 2 ** i, strides=strides,
                             conv=conv, norm=norm, act=self.act)(x)

    x = self.act(norm(name='bn_final')(x))
    x = jnp.mean(x, axis=(1, 2))
    if self.num_classes is not None:
      x = nn.Dense(self.num_classes, dtype=self.dtype, name='classifier')(x)
    return x


CifarResNet18V2 = functools.partial(
    ResNetV2, stage_sizes=[2, 2, 2, 2], block_class=BasicBlockV2,
    small_image=True)

CifarResNet50V2 = functools.partial(
    ResNetV2, stage_sizes=[3, 4, 6, 3], block_class=BottleneckBlockV2,
    small_image=True)

ResNet50V2 = functools.partial(
    ResNetV2, stage_sizes=[3, 4, 6, 3], block_class=BottleneckBlockV2)


################################################################################
# ResNet V1
################################################################################
class BasicBlockV1(nn.Module):
  """Basic block for a ResNet V1."""

  channels: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.channels, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.channels, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.channels, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckBlockV1(nn.Module):
  """Bottleneck block for ResNet V1."""

  channels: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.channels, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.channels, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.channels * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.channels * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNetV1(nn.Module):
  """ResNetV1.

      K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image
      recognition. In CVPR, pages 770–778, 2016.
  """

  stage_sizes: Sequence[int]
  block_class: ModuleDef
  num_classes: Optional[int] = None
  base_channels: int = 64
  act: Callable = nn.relu
  dtype: Any = jnp.float32
  small_image: bool = False
  # if not None, batch statistics are sync-ed across replica according to
  # this axis_name used in pmap
  bn_cross_replica_axis_name: Optional[str] = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(
        nn.BatchNorm, use_running_average=not train, momentum=0.9,
        epsilon=1e-5, dtype=self.dtype,
        axis_name=self.bn_cross_replica_axis_name)

    if self.small_image:  # suitable for Cifar
      x = conv(self.base_channels, (3, 3), padding='SAME', name='conv_init')(x)
      x = norm(name='bn_init')(x)
      x = self.act(x)
    else:
      x = conv(self.base_channels, (7, 7), (2, 2), padding=[(3, 3), (3, 3)],
               name='conv_init')(x)
      x = norm(name='bn_init')(x)
      x = nn.relu(x)
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_class(self.base_channels * 2 ** i, strides=strides,
                             conv=conv, norm=norm, act=self.act)(x)

    x = jnp.mean(x, axis=(1, 2))
    if self.num_classes is not None:
      x = nn.Dense(self.num_classes, dtype=self.dtype, name='classifier')(x)
    return x


CifarResNet18V1 = functools.partial(
    ResNetV1, stage_sizes=[2, 2, 2, 2], block_class=BasicBlockV1,
    small_image=True, num_classes=10)

CifarResNet50V1 = functools.partial(
    ResNetV1, stage_sizes=[3, 4, 6, 3], block_class=BottleneckBlockV1,
    small_image=True, num_classes=10)

ResNet50V1 = functools.partial(
    ResNetV1, stage_sizes=[3, 4, 6, 3], block_class=BottleneckBlockV1)

LogisticRegression = functools.partial(
  LR, num_classes=10
)

import functools
from typing import Any, Optional
import numpy as np

from flax import linen as nn
import jax.numpy as jnp
import timm


class TimmConfigViT(nn.Module):
    """
    JAX/Flax ViT implementation using exact timm model configurations.
    This replicates timm architectures without requiring PyTorch conversion.
    """
    
    timm_model_name: str = 'vit_small_patch16_224'
    num_classes: int = 10
    img_size: int = 32
    dtype: Any = jnp.float32
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    
    def setup(self):
        # Get exact timm model configuration
        timm_model = timm.create_model(self.timm_model_name, pretrained=False)
        
        # Extract configuration from timm model
        if hasattr(timm_model, 'embed_dim'):
            self.embed_dim = timm_model.embed_dim
        else:
            self.embed_dim = timm_model.blocks[0].attn.head_dim * timm_model.blocks[0].attn.num_heads
            
        self.depth = len(timm_model.blocks)
        self.num_heads = timm_model.blocks[0].attn.num_heads
        self.mlp_ratio = timm_model.blocks[0].mlp.fc1.out_features // self.embed_dim
        
        # egt patch embedding config
        self.patch_size = timm_model.patch_embed.patch_size[0]
        
    @nn.compact 
    def __call__(self, x, train: bool = True):
        batch_size = x.shape[0]
        
        # Patch Embedding (matching timm.layers.PatchEmbed)
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=True,  # timm uses bias in patch embedding
            dtype=self.dtype,
            name='patch_embed_proj'
        )(x)
        
        # Flatten spatial dimensions
        x = x.reshape(batch_size, -1, self.embed_dim)
        
        # Class token (matching timm)
        cls_token = self.param(
            'cls_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
            self.dtype
        )
        cls_tokens = jnp.tile(cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        
        # Position embedding (matching timm)
        num_patches = (self.img_size // self.patch_size) ** 2
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, num_patches + 1, self.embed_dim),
            self.dtype
        )
        x = x + pos_embed
        
        # Dropout after position embedding
        x = nn.Dropout(rate=self.drop_rate, deterministic=not train)(x)
        
        # Transformer blocks (matching timm.models.vision_transformer.Block)
        for i in range(self.depth):
            # Pre-norm (LayerNorm before attention)
            x_norm = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name=f'norm1_{i}')(x)
            
            # Multi-head self-attention (matching timm.models.vision_transformer.Attention)
            x_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                use_bias=True,  # timm uses bias in attention
                dropout_rate=self.attn_drop_rate,
                deterministic=not train,
                dtype=self.dtype,
                name=f'attn_{i}'
            )(x_norm, x_norm)
  
            x = x + x_attn
            x_norm = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name=f'norm2_{i}')(x)
            hidden_features = int(self.embed_dim * self.mlp_ratio)
            x_mlp = nn.Dense(features=hidden_features, use_bias=True, dtype=self.dtype, name=f'mlp_fc1_{i}')(x_norm)
            x_mlp = nn.gelu(x_mlp)
            x_mlp = nn.Dropout(rate=self.drop_rate, deterministic=not train)(x_mlp)
            x_mlp = nn.Dense(features=self.embed_dim, use_bias=True, dtype=self.dtype, name=f'mlp_fc2_{i}')(x_mlp)
            x_mlp = nn.Dropout(rate=self.drop_rate, deterministic=not train)(x_mlp)

            x = x + x_mlp
 
        x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name='norm')(x)
        x = x[:, 0]
        
        # Head (matching timm classification head)
        if self.num_classes > 0:
            x = nn.Dense(features=self.num_classes, use_bias=True, dtype=self.dtype, name='head')(x)
        
        return x


def load_timm_config(model_name: str):
    """Load configuration from a timm model without creating the full model."""
    model = timm.create_model(model_name, pretrained=False)
    
    config = {
        'embed_dim': getattr(model, 'embed_dim', model.blocks[0].attn.head_dim * model.blocks[0].attn.num_heads),
        'depth': len(model.blocks),
        'num_heads': model.blocks[0].attn.num_heads,
        'mlp_ratio': model.blocks[0].mlp.fc1.out_features // getattr(model, 'embed_dim', model.blocks[0].attn.head_dim * model.blocks[0].attn.num_heads),
        'patch_size': model.patch_embed.patch_size[0],
        'drop_rate': getattr(model, 'drop_rate', 0.0),
        'attn_drop_rate': getattr(model.blocks[0].attn, 'attn_drop', nn.Dropout(0.0)).rate if hasattr(getattr(model.blocks[0].attn, 'attn_drop', None), 'rate') else 0.0,
    }
    
    del model  # Clean up
    return config

def create_vit_small_patch16_224_cifar(num_classes: int = 10, **kwargs):
    """Create ViT-Small/16 (timm: vit_small_patch16_224) adapted for CIFAR."""
    return TimmConfigViT(
        timm_model_name='vit_small_patch16_224',
        num_classes=num_classes,
        img_size=32,
        **kwargs
    )


def create_vit_base_patch16_224_cifar(num_classes: int = 10, **kwargs):
    """Create ViT-Base/16 (timm: vit_base_patch16_224) adapted for CIFAR."""
    return TimmConfigViT(
        timm_model_name='vit_base_patch16_224',
        num_classes=num_classes,
        img_size=32,
        **kwargs
    )

def init_vit_weights_like_timm(key, shape, dtype=jnp.float32, layer_type='linear'):
    """Initialize weights to match timm's initialization."""
    if layer_type == 'linear':
        # timm uses trunc_normal for linear layers
        return jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype) * 0.02
    elif layer_type == 'embed':
        # timm uses normal(0, 0.02) for embeddings
        return jax.random.normal(key, shape, dtype) * 0.02
    elif layer_type == 'cls_token':
        # Class token initialized to zeros in timm
        return jnp.zeros(shape, dtype)
    else:
        return jax.random.normal(key, shape, dtype) * 0.02

    
    
CifarViTSmall16 = functools.partial(
    TimmConfigViT,
    timm_model_name='vit_small_patch16_224',
    img_size=224
)

CifarViTBase16 = functools.partial(
    TimmConfigViT, 
    timm_model_name='vit_base_patch16_224',
    img_size=224
)