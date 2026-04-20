# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file located at the root directory of this repository.

# NOTE:
# The original SAM import path is intentionally preserved as a commented fallback.
# from .sam import Sam

# Core SAM model
from .sam_model import Sam

# Vision Transformer-based image encoder used in SAM
from .image_encoder import ImageEncoderViT

# Wavelet-based encoder variant used in Dual-Wave SAM
from .wave_encoder import WaveEncoder

# Mask prediction decoder module
from .mask_decoder import MaskDecoder

# Prompt encoding module (handles points, boxes, masks, etc.)
from .prompt_encoder import PromptEncoder

# Two-way transformer used in mask decoding stage
from .transformer import TwoWayTransformer
