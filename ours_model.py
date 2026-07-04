"""
Model: CLIP Baseline + GTA module.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

# Self-contained: bundled CLIP package (SEDCLIP/clip) + bundled weights (SEDCLIP/weights).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from clip import clip

from temporal_module import GTA


class Model(nn.Module):
    def __init__(self, cfg, clslist, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_classes = cfg.num_classes

        clip_name = cfg.clip_weights if os.path.isfile(cfg.clip_weights) else cfg.clip_backbone
        self.clipmodel, _ = clip.load(
            clip_name, device='cpu', jit=False,
            ignore_last_atten=cfg.ignore_last_atten)
        for p in self.clipmodel.parameters():
            p.requires_grad = False

        text_features = self._encode_fixed_prompt(clslist)      # [C, 512]
        self.register_buffer("text_features", text_features)    # moves with .to(device)

        # ---- Graph-attention temporal adaptor ----
        self.temporal = cfg.temporal
        if self.temporal:
            self.temporal_module = GTA(
                cfg, cfg.feat_dim, cfg.head_num, cfg.lambda_gta)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.no_grad()
    def _encode_fixed_prompt(self, clslist):
        text = [self.cfg.prompt_template + c for c in clslist]
        tokens = clip.tokenize(text)                       # [C, 77] on cpu
        embedding = self.clipmodel.encode_token(tokens)    # [C, 77, 512]
        text_features = self.clipmodel.encode_text(embedding, tokens)  # [C, 512]
        return text_features.float()

    def forward(self, frames):
        """frames: [B, T, 3, H, W] -> logits [B, T, num_classes]."""
        B, T = frames.shape[0], frames.shape[1]
        x = einops.rearrange(frames, 'b t c h w -> (b t) c h w')
        tokens = self.clipmodel.encode_image(x)     # [(B*T), 197, 512]
        cls = tokens[:, 0, :]                        # CLS token -> [(B*T), 512]

        feats = einops.rearrange(cls, '(b t) c -> b t c', b=B, t=T)  # [B, T, 512]
        if self.temporal:
            feats = self.temporal_module(feats)             # [B, T, 512]

        feats = F.normalize(feats, dim=-1, p=2)
        tfeat = F.normalize(self.text_features.type_as(feats), dim=-1, p=2)

        logits = self.logit_scale.exp() * (feats @ tfeat.t())        # [B, T, C]
        
        return logits
