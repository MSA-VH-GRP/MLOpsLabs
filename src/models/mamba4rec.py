"""
Mamba4Rec - Hybrid Mamba for Sequential Movie Recommendation.

Architecture:
- Item Embedding + Genre Pooling + User Profile (broadcasted) + Time Embedding
- Mamba SSM Backbone for sequence modeling
- Prediction head for next-item prediction

Adapted from the original implementation.
If mamba_ssm is not installed, falls back to a simplified GRU-based implementation.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import mamba_ssm, fall back to simplified implementation if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class SimplifiedMamba(nn.Module):
    """
    Simplified Mamba-like block using 1D convolutions and gating.
    Used as fallback when mamba_ssm is not available (e.g., CPU-only or Windows).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # SSM-like recurrence approximation
        self.ssm = nn.GRU(d_inner, d_inner, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = self.act(x)

        x, _ = self.ssm(x)

        x = x * self.act(z)
        x = self.out_proj(x)

        return x


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection and layer norm."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = SimplifiedMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class GenrePooling(nn.Module):
    """
    Genre Pooling Layer.

    Input:  (Batch, Seq, Num_Genres, Dim)
    Output: (Batch, Seq, Dim)
    """

    def __init__(self, pooling_type: str = "mean"):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, genre_emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            genre_emb: (batch, seq_len, num_genres, dim)
            mask:      (batch, seq_len, num_genres) - 1 for valid genres, 0 for padding

        Returns:
            Pooled genre embeddings (batch, seq_len, dim)
        """
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (batch, seq, num_genres, 1)
            genre_emb = genre_emb * mask

            if self.pooling_type == "mean":
                num_valid = mask.sum(dim=2).clamp(min=1)
                return genre_emb.sum(dim=2) / num_valid
            else:
                return genre_emb.sum(dim=2)
        else:
            if self.pooling_type == "mean":
                return genre_emb.mean(dim=2)
            else:
                return genre_emb.sum(dim=2)


class Mamba4Rec(nn.Module):
    """
    Hybrid Mamba for Sequential Movie Recommendation.

    Combines:
    - Item embeddings
    - Multi-genre embeddings with mean pooling
    - User profile (age, gender, occupation) broadcasted across the sequence
    - Time-of-day embeddings
    - Positional embeddings

    Fusion:
        hidden = Item + GenrePool + UserProfile_broadcast + Time + Position
    """

    def __init__(
        self,
        num_items: int,
        num_genres: int,
        num_ages: int,
        num_genders: int,
        num_occupations: int,
        num_time_slots: int,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_items = num_items
        self.max_seq_len = max_seq_len

        # Embeddings
        self.item_embedding = nn.Embedding(num_items, d_model, padding_idx=0)
        self.genre_embedding = nn.Embedding(num_genres, d_model, padding_idx=0)
        self.age_embedding = nn.Embedding(num_ages, d_model)
        self.gender_embedding = nn.Embedding(num_genders, d_model)
        self.occupation_embedding = nn.Embedding(num_occupations, d_model)
        self.time_embedding = nn.Embedding(num_time_slots, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Genre Pooling
        self.genre_pooling = GenrePooling(pooling_type="mean")

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout)

        # Mamba Backbone
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_items)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_user_profile_embedding(
        self,
        age_idx: torch.Tensor,
        gender_idx: torch.Tensor,
        occupation: torch.Tensor,
    ) -> torch.Tensor:
        """Return summed user profile embedding (batch, 1, d_model)."""
        return (
            self.age_embedding(age_idx)
            + self.gender_embedding(gender_idx)
            + self.occupation_embedding(occupation)
        )

    def forward(
        self,
        item_seq: torch.Tensor,
        genre_seq: torch.Tensor,
        time_seq: torch.Tensor,
        age_idx: torch.Tensor,
        gender_idx: torch.Tensor,
        occupation: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            item_seq:    (batch, seq_len)
            genre_seq:   (batch, seq_len, num_genres)
            time_seq:    (batch, seq_len)
            age_idx:     (batch, 1)
            gender_idx:  (batch, 1)
            occupation:  (batch, 1)
            return_hidden: if True, return full hidden states instead of logits

        Returns:
            logits (batch, num_items)  or  hidden (batch, seq_len, d_model)
        """
        batch_size, seq_len = item_seq.shape

        item_emb = self.item_embedding(item_seq)                  # (batch, seq, d_model)
        genre_emb = self.genre_embedding(genre_seq)               # (batch, seq, num_genres, d_model)
        genre_mask = (genre_seq != 0)
        genre_pooled = self.genre_pooling(genre_emb, genre_mask)  # (batch, seq, d_model)
        time_emb = self.time_embedding(time_seq)                  # (batch, seq, d_model)

        user_emb = self.get_user_profile_embedding(age_idx, gender_idx, occupation)  # (batch, 1, d_model)
        user_emb = user_emb.expand(-1, seq_len, -1)

        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)              # (1, seq, d_model)

        hidden = item_emb + genre_pooled + user_emb + time_emb + pos_emb
        hidden = self.emb_dropout(hidden)

        padding_mask = (item_seq != 0).float().unsqueeze(-1)      # (batch, seq, 1)
        hidden = hidden * padding_mask

        for layer in self.mamba_layers:
            hidden = layer(hidden)
            hidden = hidden * padding_mask

        hidden = self.final_norm(hidden)

        if return_hidden:
            return hidden

        # For LEFT-padded sequences [0,…,0, item1,…,itemN] the GRU processes
        # left-to-right and by position seq_len-1 has seen every real item.
        # The old seq_lengths = N-1 calculated the count of non-padding tokens
        # and used it as an index from the START, which landed in the padding
        # region for short histories (e.g. N=3 → index 2 → still all-zeros).
        # Fix: always take the last position — correct for any left-padded input.
        last_hidden = hidden[:, -1, :]                             # (batch, d_model)

        return self.output_proj(last_hidden)                       # (batch, num_items)

    def predict_scores(
        self,
        item_seq: torch.Tensor,
        genre_seq: torch.Tensor,
        time_seq: torch.Tensor,
        age_idx: torch.Tensor,
        gender_idx: torch.Tensor,
        occupation: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict scores for candidate items.

        Args:
            candidate_items: (batch, num_candidates)  — if None, score all items

        Returns:
            scores (batch, num_candidates) or (batch, num_items)
        """
        hidden = self.forward(
            item_seq, genre_seq, time_seq,
            age_idx, gender_idx, occupation,
            return_hidden=True,
        )

        # Same left-padding fix as forward(): use the last position.
        last_hidden = hidden[:, -1, :]                             # (batch, d_model)

        if candidate_items is not None:
            candidate_emb = self.item_embedding(candidate_items)   # (batch, num_cand, d_model)
            return torch.bmm(candidate_emb, last_hidden.unsqueeze(-1)).squeeze(-1)
        else:
            return self.output_proj(last_hidden)


def create_mamba4rec(metadata: Dict, **kwargs) -> Mamba4Rec:
    """Factory — build Mamba4Rec from the metadata dict produced by materialization."""
    return Mamba4Rec(
        num_items=metadata["num_items"],
        num_genres=metadata["num_genres"],
        num_ages=metadata["num_ages"],
        num_genders=metadata["num_genders"],
        num_occupations=metadata["num_occupations"],
        num_time_slots=metadata["num_time_slots"],
        **kwargs,
    )
