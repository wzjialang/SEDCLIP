import os
import argparse

import numpy as np
import torch
import einops
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from SEDCLIP.config import Config
from SEDCLIP.dataset import RARPFrameDataset
from SEDCLIP.ours_model import Model


def main():
    parser = argparse.ArgumentParser("Minimal CLIP Baseline + GTA Module Implementation.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_epoch", type=int, default=50)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = Config(device=args.device)

    cfg.image_roots = [os.path.join(here, "example_data", "frames")]
    cfg.label_roots = [os.path.join(here, "example_data", "labels")]
    example_list = os.path.join(here, "example_data", "train_list.csv")

    clslist = [c.strip().lower() for c in open(cfg.clslist) if c.strip()]

    transform = AutoImageProcessor.from_pretrained(
        'openai/clip-vit-base-patch16',
        size={"height": cfg.input_size, "width": cfg.input_size},
        do_center_crop=False)

    dataset = RARPFrameDataset(
        cfg.image_roots, cfg.label_roots, example_list,
        seq_len=cfg.frame_lenth, stride=cfg.train_stride,
        num_classes=cfg.num_classes, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg.train_bs, shuffle=True,
                        num_workers=cfg.workers, pin_memory=True, drop_last=True)
    print(f"[example] {len(dataset)} windows -> {len(loader)} batches/epoch (bs={cfg.train_bs})")

    model = Model(cfg, clslist, device=cfg.device).to(cfg.device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    criterion = torch.nn.BCEWithLogitsLoss().to(cfg.device)

    print(f"[example] training {args.max_epoch} epochs")
    model.train()
    for epoch in range(args.max_epoch):
        losses = []
        for i, (frames, _label, multi_label, *_ ) in enumerate(loader):
            frames = frames.float().to(cfg.device)                    # [B,T,3,224,224]
            multi_label = einops.rearrange(
                multi_label.to(cfg.device), 'b t c -> (b t) c').float()

            logits = model(frames)                                    # [B,T,C]
            logits = einops.rearrange(logits, 'b t c -> (b t) c')
            loss = criterion(logits, multi_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch {epoch:02d}  mean_loss={np.mean(losses):.4f}")


if __name__ == "__main__":
    main()
