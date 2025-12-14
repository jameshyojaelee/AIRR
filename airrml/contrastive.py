"""
Contrastive sequence pretraining utilities for AIRR-ML.

Trains a small transformer encoder on sequences with simple augmentations
using InfoNCE (SimCLR-style) and saves encoder weights for reuse in DeepMIL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from airrml.models.deep_mil import AA_TO_ID, UNK_AA_ID, encode_aa_sequence


def aa_dropout(seq: str, drop_prob: float, rng: np.random.Generator) -> str:
    return "".join([ch for ch in seq if rng.random() > drop_prob])


def aa_shuffle(seq: str, max_swap: int, rng: np.random.Generator) -> str:
    seq_list = list(seq)
    swaps = rng.integers(0, max_swap + 1)
    n = len(seq_list)
    for _ in range(swaps):
        if n < 2:
            break
        i, j = rng.integers(0, n, size=2)
        seq_list[i], seq_list[j] = seq_list[j], seq_list[i]
    return "".join(seq_list)


def aa_mask(seq: str, mask_prob: float, mask_token: str = "X", rng: Optional[np.random.Generator] = None) -> str:
    rng = rng or np.random.default_rng()
    return "".join([mask_token if rng.random() < mask_prob else ch for ch in seq])


def apply_augmentations(seq: str, cfg: Dict, rng: np.random.Generator) -> str:
    out = seq
    if cfg.get("drop_prob", 0) > 0:
        out = aa_dropout(out, cfg["drop_prob"], rng)
    if cfg.get("mask_prob", 0) > 0:
        out = aa_mask(out, cfg["mask_prob"], rng=rng)
    if cfg.get("max_swap", 0) > 0:
        out = aa_shuffle(out, cfg["max_swap"], rng)
    return out if out else seq


@dataclass
class SeqExample:
    sequence: str


class SequenceContrastiveDataset(Dataset):
    def __init__(self, sequences: List[str], max_len: int, aug_cfg: Dict, seed: int = 42) -> None:
        self.sequences = sequences
        self.max_len = max_len
        self.aug_cfg = aug_cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        seq = self.sequences[idx]
        view1 = apply_augmentations(seq, self.aug_cfg, self.rng)
        view2 = apply_augmentations(seq, self.aug_cfg, self.rng)
        return encode_aa_sequence(view1, max_len=self.max_len), encode_aa_sequence(view2, max_len=self.max_len)


def collate_contrastive(batch: List[Tuple[List[int], List[int]]], device: torch.device):
    def pad_to_max(tensors: List[List[int]]) -> torch.Tensor:
        max_len = max(len(t) for t in tensors) if tensors else 1
        padded = torch.zeros((len(tensors), max_len), dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        return padded

    v1 = [b[0] for b in batch]
    v2 = [b[1] for b in batch]
    t1 = pad_to_max(v1).to(device)
    t2 = pad_to_max(v2).to(device)
    return {"view1": t1, "view2": t2}


class ContrastiveEncoder(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_layers: int, ff_dim: int, dropout: float, max_len: int):
        super().__init__()
        vocab_size = len(AA_TO_ID) + 2
        self.embed = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, model_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(model_dim, model_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L]
        x = self.embed(tokens)
        # trim/extend positional to length
        pos = self.pos_embed[:, : x.size(1), :]
        x = x + pos
        key_padding_mask = tokens.eq(0)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        mask = (~key_padding_mask).unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        pooled = self.dropout(pooled)
        return self.proj(pooled)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z1 = nn.functional.normalize(z1, dim=-1)
    z2 = nn.functional.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss


def train_contrastive(
    sequences: List[str],
    model_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    ff_dim: Optional[int] = None,
    dropout: float = 0.1,
    max_len: int = 72,
    batch_size: int = 256,
    num_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    aug_cfg: Optional[Dict] = None,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ff_dim = ff_dim or model_dim * 4
    aug_cfg = aug_cfg or {"drop_prob": 0.05, "mask_prob": 0.05, "max_swap": 1}

    dataset = SequenceContrastiveDataset(sequences, max_len=max_len, aug_cfg=aug_cfg, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_contrastive(b, device_t))

    model = ContrastiveEncoder(model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout, max_len=max_len).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, num_epochs))

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for batch in loader:
            z1 = model(batch["view1"])
            z2 = model(batch["view2"])
            loss = info_nce_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        scheduler.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    return model.state_dict()


def save_state_dict(state: Dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
