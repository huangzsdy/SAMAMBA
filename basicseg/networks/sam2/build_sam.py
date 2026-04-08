# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# 简化版 SAM2 模型加载

import logging


def build_sam2(ckpt_path=None, device='cuda', mode='eval'):
    """直接返回 dummy 对象，让 SAMamba 内部直接使用简化配置"""

    logging.info("Using simplified build_sam2 (SAMamba will create encoder internally)")

    class DummySAM2:
        """空的 SAM2 模型，SAMamba 会重新创建 encoder"""
        def __init__(self):
            self.image_encoder = None
            self.sam_mask_decoder = None
            self.sam_prompt_encoder = None
            self.memory_encoder = None
            self.memory_attention = None
            self.mask_downsample = None
            self.obj_ptr_tpos_proj = None
            self.obj_ptr_proj = None

        def to(self, d):
            return self

        def eval(self):
            return self

        def train(self):
            return self

    return DummySAM2()