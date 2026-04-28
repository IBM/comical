"""
COMICAL Pretrained Encoder Loading Script

Demonstrates how to load the pretrained COMICAL dual encoders
(SNP encoder + IDP encoder) and attach a downstream classification head.

The pretrained encoders were trained via CLIP-style contrastive learning
on UK Biobank SNP-IDP pairs. They can be frozen and used as feature
extractors for downstream disease classification tasks.

Usage:
    # Load encoders only (for embedding extraction)
    python load_pretrained_encoders.py

    # Run downstream classification via the full pipeline
    python wrapper.py \
        -fo my_experiment \
        -pck results/my_experiment/checkpoints \
        -ckpt_name pretrained_weights/comical_pretrained_encoders.pt \
        -pt data/neuroDx_wPRS.csv \
        -p_sx data/all_subjects_idx.csv \
        -out_flag clf -target PD -dwn 1 -tune 0 \
        -dm 64 -df 128 -nh 4 -nl 2 \
        -lr 0.001 -wd 0.01 -bz 1024 -e 10 \
        -dp 0.0 -u 16 -top_n_perc 5 -svemb 0 -pltemb 0
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import comical_new_emb, comical_new_emb_clasf


def load_pretrained_encoders(checkpoint_path, device="cpu"):
    """Load the pretrained COMICAL dual encoders from a checkpoint file.

    Args:
        checkpoint_path: Path to comical_pretrained_encoders.pt
        device: Device to load the model onto

    Returns:
        model: comical_new_emb model with loaded weights
        config: Architecture config dict
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt["architecture"]
    config["units"] = 16  # unused by pairs model but required by constructor

    model = comical_new_emb(config)
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, config


def load_downstream_classifier(checkpoint_path, num_classes=2, device="cpu"):
    """Load pretrained encoders into a downstream classification model.

    The encoder weights are loaded and frozen. Only the MLP head
    (h1, proj_down) is randomly initialized and should be trained.

    Args:
        checkpoint_path: Path to comical_pretrained_encoders.pt
        num_classes: Number of output classes (2 for binary classification)
        device: Device to load the model onto

    Returns:
        model: comical_new_emb_clasf with frozen encoders + trainable head
        config: Architecture config dict
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt["architecture"]
    config["units"] = 16
    config["num_classes"] = num_classes

    model = comical_new_emb_clasf(config)
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained encoder weights from {checkpoint_path}")
    print(f"  Newly initialized (trainable head): {missing}")
    print(f"  Unexpected keys (should be empty):  {unexpected}")

    # Freeze encoder parameters — only train the MLP head
    trainable_prefixes = ("h1.", "proj_down.")
    for name, param in model.named_parameters():
        if not name.startswith(trainable_prefixes):
            param.requires_grad = False

    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_total:,}  Trainable: {n_trainable:,}  Frozen: {n_total - n_trainable:,}")

    return model, config


if __name__ == "__main__":
    ckpt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "comical_pretrained_encoders.pt",
    )

    print("=" * 60)
    print("1. Loading pretrained encoders (pairs model)")
    print("=" * 60)
    model, config = load_pretrained_encoders(ckpt_path)
    print(f"Architecture: {config}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("=" * 60)
    print("2. Loading downstream classifier (frozen encoders + MLP head)")
    print("=" * 60)
    clf_model, clf_config = load_downstream_classifier(ckpt_path, num_classes=2)
    print()

    print("=" * 60)
    print("3. Checkpoint metadata")
    print("=" * 60)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for section in ["architecture", "training"]:
        print(f"\n  {section}:")
        for k, v in ckpt[section].items():
            print(f"    {k}: {v}")
