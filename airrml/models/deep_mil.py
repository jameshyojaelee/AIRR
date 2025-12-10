"""
PyTorch Multiple-Instance Learning model for AIRR-ML-25.

This implementation consumes variable-length sets of sequences per repertoire,
produces repertoire-level probabilities, and exposes attention weights as
sequence-level importance scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from airrml.models import register_model
from airrml.models.base import BaseRepertoireModel

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:  # pragma: no cover - deep models are optional
    # Fail fast with a clear error instead of letting class definitions crash later
    raise ImportError(f"PyTorch is required for DeepMILModel; torch import failed: {e}") from e


# ---------------------- Tokenization utilities ---------------------- #
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {aa: idx + 1 for idx, aa in enumerate(AA_ALPHABET)}  # 0 reserved for PAD
UNK_AA_ID = len(AA_TO_ID) + 1


def encode_aa_sequence(seq: str, max_len: Optional[int] = None) -> List[int]:
    tokens = [AA_TO_ID.get(ch, UNK_AA_ID) for ch in seq]
    if max_len is not None:
        tokens = tokens[:max_len]
    return tokens


@dataclass
class RepertoireExample:
    rep_id: str
    sequences: List[str]
    v_calls: List[str]
    j_calls: List[str]
    label: Optional[float]


class RepertoireDataset(Dataset):
    """
    Dataset of repertoires where each item holds all sequences belonging to one repertoire.
    """

    def __init__(self, sequences_df: pd.DataFrame, labels: Optional[pd.Series] = None, max_len: int = 40) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for DeepMILModel.")
        required_cols = {"ID", "junction_aa", "v_call", "j_call"}
        missing = required_cols - set(sequences_df.columns)
        if missing:
            raise ValueError(f"sequences_df missing required columns: {missing}")

        labels_series = None
        if labels is not None:
            labels_series = labels
            if isinstance(labels_series, pd.DataFrame):
                if "ID" in labels_series.columns:
                    labels_series = labels_series.set_index("ID")[labels_series.columns[-1]]
                else:
                    labels_series = labels_series.squeeze()
            if not isinstance(labels_series, pd.Series):
                raise ValueError("labels must be a pandas Series or DataFrame with an ID column")

        self.max_len = max_len
        self.examples: List[RepertoireExample] = []

        for rep_id, group in sequences_df.groupby("ID"):
            seqs = group["junction_aa"].fillna("").tolist()
            v_calls = group.get("v_call", pd.Series([""] * len(group))).fillna("").tolist()
            j_calls = group.get("j_call", pd.Series([""] * len(group))).fillna("").tolist()
            label_val = None
            if labels_series is not None and rep_id in labels_series.index:
                label_val = float(labels_series.loc[rep_id])
            self.examples.append(RepertoireExample(rep_id=str(rep_id), sequences=seqs, v_calls=v_calls, j_calls=j_calls, label=label_val))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> RepertoireExample:
        return self.examples[idx]


def build_gene_vocab(values: List[str]) -> Dict[str, int]:
    vocab = {"<pad>": 0}
    for v in values:
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab


def collate_repertoires(batch: List[RepertoireExample], gene_vocabs: Dict[str, Dict[str, int]], max_len: int, device: torch.device):
    # Flatten sequences across batch and keep mapping to repertoire index
    all_tokens = []
    all_v = []
    all_j = []
    rep_index = []
    labels = []
    rep_ids: List[str] = []

    for rep_idx, ex in enumerate(batch):
        rep_ids.append(ex.rep_id)
        if ex.label is not None:
            labels.append(ex.label)
        for seq, v_call, j_call in zip(ex.sequences, ex.v_calls, ex.j_calls):
            token_ids = encode_aa_sequence(seq, max_len=max_len)
            all_tokens.append(torch.tensor(token_ids, dtype=torch.long))
            all_v.append(gene_vocabs["v"].get(v_call, 0))
            all_j.append(gene_vocabs["j"].get(j_call, 0))
            rep_index.append(rep_idx)

    # Handle empty repertoires
    if not all_tokens:
        tokens_padded = torch.zeros((1, 1), dtype=torch.long)
        v_ids = torch.zeros((1,), dtype=torch.long)
        j_ids = torch.zeros((1,), dtype=torch.long)
        rep_index = torch.zeros((1,), dtype=torch.long)
    else:
        max_seq_len = max(t.size(0) for t in all_tokens)
        tokens_padded = torch.zeros((len(all_tokens), max_seq_len), dtype=torch.long)
        for i, t in enumerate(all_tokens):
            tokens_padded[i, : t.size(0)] = t
        v_ids = torch.tensor(all_v, dtype=torch.long)
        j_ids = torch.tensor(all_j, dtype=torch.long)
        rep_index = torch.tensor(rep_index, dtype=torch.long)

    labels_tensor = torch.tensor(labels, dtype=torch.float32) if len(labels) == len(batch) else None

    return {
        "tokens": tokens_padded.to(device),
        "v_ids": v_ids.to(device),
        "j_ids": j_ids.to(device),
        "rep_index": rep_index.to(device),
        "labels": labels_tensor.to(device) if labels_tensor is not None else None,
        "rep_ids": rep_ids,
        "num_reps": len(batch),
    }


# ---------------------- Model components ---------------------- #
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding added to token embeddings.
    """

    def __init__(self, dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class SequenceEncoder(nn.Module):
    """
    Transformer-based encoder over amino acid tokens with optional V/J embeddings.
    """

    def __init__(
        self,
        aa_vocab_size: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
        gene_vocab_sizes: Dict[str, int],
        gene_dims: Dict[str, int],
    ):
        super().__init__()
        self.model_dim = model_dim
        self.aa_embed = nn.Embedding(aa_vocab_size, model_dim, padding_idx=0)
        self.v_embed = nn.Embedding(gene_vocab_sizes.get("v", 1), gene_dims.get("v", 0), padding_idx=0) if gene_dims.get("v", 0) > 0 else None
        self.j_embed = nn.Embedding(gene_vocab_sizes.get("j", 1), gene_dims.get("j", 0), padding_idx=0) if gene_dims.get("j", 0) > 0 else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(model_dim, max_len=max_len, dropout=dropout)
        self.pool_dropout = nn.Dropout(dropout)

        extra = (gene_dims.get("v", 0) + gene_dims.get("j", 0))
        self.proj = nn.Linear(model_dim + extra, model_dim)

    def forward(self, tokens: torch.Tensor, v_ids: torch.Tensor, j_ids: torch.Tensor) -> torch.Tensor:
        # tokens: [N_seq, L]
        key_padding_mask = tokens.eq(0)  # [N_seq, L], True on padding
        x = self.aa_embed(tokens)  # [N_seq, L, model_dim]
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Masked mean pooling over sequence length
        mask = (~key_padding_mask).unsqueeze(-1)  # [N_seq, L, 1]
        summed = (x * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        seq_emb = summed / lengths
        seq_emb = self.pool_dropout(seq_emb)

        extras = []
        if self.v_embed is not None:
            extras.append(self.v_embed(v_ids))
        if self.j_embed is not None:
            extras.append(self.j_embed(j_ids))
        if extras:
            seq_emb = torch.cat([seq_emb] + extras, dim=-1)
        return self.proj(seq_emb)


class AttentionMIL(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq_embs: torch.Tensor, rep_index: torch.Tensor, num_reps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # seq_embs: [N_seq, emb_dim]; rep_index: [N_seq]
        att_logits = self.attention(seq_embs).squeeze(-1)  # [N_seq]
        rep_embs = []
        att_weights_list = []
        for r in range(num_reps):
            mask = rep_index == r
            if mask.sum() == 0:
                rep_embs.append(torch.zeros(seq_embs.size(1), device=seq_embs.device))
                att_weights_list.append(torch.empty(0, device=seq_embs.device))
                continue
            logits_r = att_logits[mask]
            weights = torch.softmax(logits_r, dim=0)
            reps = torch.sum(seq_embs[mask] * weights.unsqueeze(-1), dim=0)
            rep_embs.append(reps)
            att_weights_list.append(weights)
        rep_embs = torch.stack(rep_embs, dim=0)
        att_weights = torch.cat(att_weights_list, dim=0) if att_weights_list else torch.empty(0, device=seq_embs.device)
        return rep_embs, att_weights


class DeepMILNet(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
        gene_vocab_sizes: Dict[str, int],
        gene_dims: Dict[str, int],
        classifier_dim: int,
    ):
        super().__init__()
        aa_vocab_size = len(AA_TO_ID) + 2  # pad + unk
        self.encoder = SequenceEncoder(
            aa_vocab_size=aa_vocab_size,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_len=max_len,
            gene_vocab_sizes=gene_vocab_sizes,
            gene_dims=gene_dims,
        )
        self.pool = AttentionMIL(model_dim, model_dim)
        self.classifier = nn.Sequential(nn.Linear(model_dim, classifier_dim), nn.ReLU(), nn.Linear(classifier_dim, 1))

    def forward(self, tokens: torch.Tensor, v_ids: torch.Tensor, j_ids: torch.Tensor, rep_index: torch.Tensor, num_reps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_embs = self.encoder(tokens, v_ids, j_ids)
        rep_embs, att_weights = self.pool(seq_embs, rep_index, num_reps)
        logits = self.classifier(rep_embs).squeeze(-1)
        return logits, att_weights


# ---------------------- Public model ---------------------- #
@register_model("deep_mil")
class DeepMILModel(BaseRepertoireModel):
    """
    Attention-based MIL model operating on variable-length sets of sequences per repertoire.
    """

    def __init__(
        self,
        model_dim: Optional[int] = None,
        aa_dim: Optional[int] = None,  # backward compatibility
        hidden_dim: Optional[int] = None,  # backward compatibility
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        gene_dims: Optional[Dict[str, int]] = None,
        classifier_dim: int = 128,
        max_len: int = 60,
        batch_size: int = 8,
        num_epochs: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        max_sequences_per_rep: Optional[int] = None,
        random_state: Optional[int] = None,
        pretrained_encoder_path: Optional[str] = None,
        freeze_encoder: bool = False,
        aug_dropout: float = 0.0,
        aug_mask: float = 0.0,
        aug_shuffle_max: int = 0,
        class_balance_sequences: bool = False,
        use_amp: bool = True,
        grad_clip: Optional[float] = 1.0,
        early_stop_patience: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if torch is None:
            raise ImportError("PyTorch is required for DeepMILModel.")

        resolved_dim = model_dim or aa_dim or hidden_dim or 256
        self.model_dim = int(resolved_dim)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim or model_dim * 4
        self.dropout = dropout
        self.gene_dims = gene_dims or {"v": 16, "j": 8}
        self.classifier_dim = classifier_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_sequences_per_rep = max_sequences_per_rep
        self.pretrained_encoder_path = pretrained_encoder_path
        self.freeze_encoder = freeze_encoder
        self.aug_dropout = aug_dropout
        self.aug_mask = aug_mask
        self.aug_shuffle_max = aug_shuffle_max
        self.class_balance_sequences = class_balance_sequences
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.early_stop_patience = early_stop_patience
        self.rng = np.random.default_rng(random_state)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model_: Optional[DeepMILNet] = None
        self.gene_vocabs_: Dict[str, Dict[str, int]] = {}

    def _build_gene_vocabs(self, sequences_df: pd.DataFrame) -> None:
        v_vocab = build_gene_vocab(sequences_df.get("v_call", pd.Series(dtype=str)).fillna("").tolist())
        j_vocab = build_gene_vocab(sequences_df.get("j_call", pd.Series(dtype=str)).fillna("").tolist())
        self.gene_vocabs_ = {"v": v_vocab, "j": j_vocab}

    def _make_loader(self, sequences_df: pd.DataFrame, labels: Optional[pd.Series], shuffle: bool) -> DataLoader:
        dataset = RepertoireDataset(sequences_df, labels, max_len=self.max_len)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_repertoires(batch, self.gene_vocabs_, self.max_len, self.device),
        )

    def _augment_sequences(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        if self.aug_dropout <= 0 and self.aug_mask <= 0 and self.aug_shuffle_max <= 0:
            return sequences_df
        df = sequences_df.copy()
        aug_seqs = []
        for seq in df["junction_aa"].astype(str):
            s = list(seq)
            # dropout
            if self.aug_dropout > 0:
                s = [ch for ch in s if self.rng.random() > self.aug_dropout]
            # mask
            if self.aug_mask > 0:
                s = [("X" if self.rng.random() < self.aug_mask else ch) for ch in s]
            # shuffle
            if self.aug_shuffle_max > 0 and len(s) > 1:
                swaps = self.rng.integers(0, self.aug_shuffle_max + 1)
                for _ in range(swaps):
                    i, j = self.rng.integers(0, len(s), size=2)
                    s[i], s[j] = s[j], s[i]
            aug_seqs.append("".join(s) if s else seq)
        df["junction_aa"] = aug_seqs
        return df

    def _balance_sequences(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        if not self.class_balance_sequences or "label_positive" not in sequences_df.columns:
            return sequences_df
        grouped = sequences_df.groupby("label_positive")
        max_count = grouped.size().max()
        balanced = []
        for _, group in grouped:
            if len(group) < max_count:
                reps = int(np.ceil(max_count / len(group)))
                balanced.append(pd.concat([group] * reps, ignore_index=True).iloc[:max_count])
            else:
                balanced.append(group)
        return pd.concat(balanced, ignore_index=True)
    def _subsample_sequences(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally subsample sequences per repertoire to cap memory.
        """
        if self.max_sequences_per_rep is None:
            return sequences_df
        sampled_parts: List[pd.DataFrame] = []
        for rep_id, group in sequences_df.groupby("ID"):
            if len(group) > self.max_sequences_per_rep:
                idx = self.rng.choice(len(group), self.max_sequences_per_rep, replace=False)
                sampled_parts.append(group.iloc[idx])
            else:
                sampled_parts.append(group)
        return pd.concat(sampled_parts, ignore_index=True)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "DeepMILModel":
        """
        Train on sequence-level DataFrame (X_train) with labels provided as Series/DataFrame indexed by repertoire ID.
        """
        X_train = self._augment_sequences(X_train)
        X_train = self._balance_sequences(X_train)
        X_train = self._subsample_sequences(X_train)
        if X_val is not None:
            X_val = self._augment_sequences(X_val)
            X_val = self._subsample_sequences(X_val)

        self._build_gene_vocabs(X_train)

        # Initialize model
        gene_vocab_sizes = {k: len(v) for k, v in self.gene_vocabs_.items()}
        self.model_ = DeepMILNet(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            max_len=self.max_len,
            gene_vocab_sizes=gene_vocab_sizes,
            gene_dims=self.gene_dims,
            classifier_dim=self.classifier_dim,
        ).to(self.device)
        if self.pretrained_encoder_path:
            try:
                state = torch.load(self.pretrained_encoder_path, map_location=self.device)
                missing, unexpected = self.model_.encoder.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"Loaded encoder with missing keys {missing} and unexpected keys {unexpected}")
            except Exception as e:
                print(f"Warning: failed to load pretrained encoder: {e}")
        if self.freeze_encoder:
            for p in self.model_.encoder.parameters():
                p.requires_grad = False

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None and y_val is not None else None

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.num_epochs))

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.device.type == "cuda")
        best_val_loss = float("inf")
        patience = self.early_stop_patience
        patience_ctr = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(self.num_epochs):
            self.model_.train()
            for batch in train_loader:
                with torch.cuda.amp.autocast(enabled=self.use_amp and self.device.type == "cuda"):
                    logits, _ = self.model_(batch["tokens"], batch["v_ids"], batch["j_ids"], batch["rep_index"], batch["num_reps"])
                    if batch["labels"] is None:
                        continue
                    loss = criterion(logits, batch["labels"].to(self.device))
                optimizer.zero_grad()
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                    optimizer.step()
            scheduler.step()
            if torch.cuda.is_available() and self.device.type == "cuda":
                alloc = torch.cuda.memory_allocated(self.device) / 1e6
                torch.cuda.reset_peak_memory_stats(self.device)
                print(f"[Epoch {epoch+1}] GPU memory allocated: {alloc:.1f} MB")

            if val_loader is not None:
                self.model_.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        logits, _ = self.model_(batch["tokens"], batch["v_ids"], batch["j_ids"], batch["rep_index"], batch["num_reps"])
                        if batch["labels"] is None:
                            continue
                        loss = criterion(logits, batch["labels"].to(self.device))
                        val_losses.append(loss.item())
                avg_val = np.mean(val_losses) if val_losses else float("inf")
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_state = self.model_.state_dict()
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Store feature metadata
        self.feature_info = {
            "gene_vocabs": self.gene_vocabs_,
            "max_len": self.max_len,
            "max_sequences_per_rep": self.max_sequences_per_rep,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        }

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted.")
        if not self.gene_vocabs_:
            self._build_gene_vocabs(X)
        self.model_.eval()

        X = self._subsample_sequences(X)
        loader = self._make_loader(X, labels=None, shuffle=False)
        all_probs: List[float] = []
        all_rep_ids: List[str] = []
        with torch.no_grad():
            for batch in loader:
                logits, _ = self.model_(batch["tokens"], batch["v_ids"], batch["j_ids"], batch["rep_index"], batch["num_reps"])
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs)
                all_rep_ids.extend(batch["rep_ids"])
        # Ensure alignment to repertoire order
        prob_series = pd.Series(all_probs, index=all_rep_ids)
        return prob_series.reindex(X["ID"].unique()).to_numpy()

    def get_sequence_importance(
        self,
        sequences_df: pd.DataFrame,
        sequence_col: str = "junction_aa",
    ) -> pd.DataFrame:
        """
        Score sequences using attention weights combined with gradient-based saliency.
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted.")
        if not self.gene_vocabs_:
            self._build_gene_vocabs(sequences_df)
        self.model_.eval()

        sequences_df = self._subsample_sequences(sequences_df)
        # Build loader with labels absent
        loader = self._make_loader(sequences_df, labels=None, shuffle=False)
        records: List[Dict[str, Any]] = []
        for batch in loader:
            batch_tokens = batch["tokens"].clone().detach().to(self.device)
            batch_tokens.requires_grad = True
            logits, att_weights = self.model_(batch_tokens, batch["v_ids"], batch["j_ids"], batch["rep_index"], batch["num_reps"])
            probs = torch.sigmoid(logits)
            prob_sum = probs.sum()
            prob_sum.backward()
            grads = batch_tokens.grad  # [N_seq, L]

            att_cpu = att_weights.detach().cpu().numpy()
            grad_scores = grads.detach().abs().sum(dim=1).cpu().numpy()

            seq_idx = 0
            for rep_idx, rep_id in enumerate(batch["rep_ids"]):
                mask = batch["rep_index"] == rep_idx
                num_seqs = int(mask.sum().item())
                rep_seqs = sequences_df[sequences_df["ID"] == rep_id]
                if num_seqs != len(rep_seqs):
                    num_seqs = min(num_seqs, len(rep_seqs))
                weights_rep = att_cpu[seq_idx : seq_idx + num_seqs]
                grad_rep = grad_scores[seq_idx : seq_idx + num_seqs]
                rep_slice = rep_seqs.iloc[:num_seqs].copy()
                rep_slice["importance_score"] = 0.5 * weights_rep + 0.5 * grad_rep
                records.append(rep_slice)
                seq_idx += num_seqs
        if records:
            out_df = pd.concat(records, ignore_index=True)
        else:
            out_df = pd.DataFrame(columns=[sequence_col, "v_call", "j_call", "importance_score"])
        return out_df[[sequence_col, "v_call", "j_call", "importance_score"]]
