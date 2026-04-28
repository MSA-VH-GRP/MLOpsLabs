"""
Mamba4Rec - Hybrid Mamba for Sequential Movie Recommendation.

Architecture:
- Item Embedding + Genre Pooling + User Profile + Time Embedding
- Mamba SSM Backbone for sequence modeling
- Prediction head for next-item prediction

User profile fusion modes (user_fusion_mode):
  "broadcast"   — classic: user_emb broadcast across input sequence (default)
  "film"        — FiLM conditioning: scale+shift last_hidden AFTER backbone
                    gamma * last_hidden + beta  where gamma, beta = f(user_emb)
  "head"        — simple additive head: last_hidden + MLP(user_emb)
  "gated_head"  — gated head: last_hidden + sigmoid(W_gate·user_emb) * MLP(user_emb)
  "normed_head" — head + LayerNorm: LayerNorm(last_hidden + MLP(user_emb))
  "hybrid"      — light broadcast (learnable alpha) + full gated_head at output

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

    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:           (batch, seq_len, d_model)
            seq_lengths: (batch,) int64 CPU tensor — actual non-padding lengths.
                         When provided, uses pack_padded_sequence so the GRU
                         only processes real tokens; padding positions never
                         contaminate the GRU hidden state.
        """
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = self.act(x)

        # Fix A: use pack_padded_sequence to skip padding in the GRU
        if seq_lengths is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.ssm(packed)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_len
            )
        else:
            x, _ = self.ssm(x)

        x = x * self.act(z)
        x = self.out_proj(x)

        return x


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection and layer norm."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        force_gru: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self._use_real_mamba = MAMBA_AVAILABLE and not force_gru

        if self._use_real_mamba:
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

    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._use_real_mamba:
            return x + self.mamba(self.norm(x))
        return x + self.mamba(self.norm(x), seq_lengths)


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

    Architecture enhancements (v4):

    TUPE (Transformer with Untied Positional Encoding — adapted for GRU/Mamba):
        Original TUPE separates content and position in attention scores:
            score = content_score + position_score
        For GRU/Mamba (no explicit attention), we adapt this by:
        - Removing pos_emb from the INPUT fusion (content path stays pure)
        - Injecting positional information as an OUTPUT bias via a separate
          learned projection: hidden_out = hidden_content + pos_proj(pos_emb)
        Result: the recurrent backbone learns purely content-driven transitions;
        position awareness is added as a post-hoc, untied correction.

    SUM Token (causal variant of Prefix Token):
        A learnable token is APPENDED at the end of each sequence:
            [0, …, 0, item1, item2, itemN, SUM]   (length = max_seq_len + 1)
        Because the GRU processes left-to-right, the SUM token at position -1
        has seen every real item in the history by the time it is read.
        It acts as a dedicated "next-item query" whose hidden state captures the
        full sequential context — analogous to [CLS] in BERT but placed at the
        END to respect causality.
        User representation = hidden[:, -1, :] = the SUM token's hidden state.

    Combined fusion:
        content = Item + GenrePool + UserProfile_broadcast + Time   (no pos)
        hidden  = Mamba(content)
        hidden  = hidden + pos_proj(pos_emb)                        (TUPE)
        repr    = hidden[:, -1, :]                                  (SUM token)
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
        # ── Ablation flags ────────────────────────────────────────────────────
        use_sum_token: bool = True,         # append learnable SUM token at end of seq
        use_tupe: bool = True,              # inject positional bias AFTER backbone (TUPE)
        # ── Time interval (TiSASRec-style) ────────────────────────────────────
        use_time_interval: bool = False,    # add consecutive time-gap embedding to input
        num_time_interval_bins: int = 256,  # vocab size for Δt embeddings (0 = pad)
        # ── User profile fusion mode ──────────────────────────────────────────
        # "broadcast"   — user_emb added to input sequence (original behaviour)
        # "film"        — FiLM scale+shift on last_hidden after backbone
        # "head"        — simple additive MLP head
        # "gated_head"  — sigmoid-gated MLP head (recommended)
        # "normed_head" — MLP head + LayerNorm
        # "hybrid"      — light broadcast + gated_head at output
        user_fusion_mode: str = "broadcast",
        # ── Backbone override ─────────────────────────────────────────────────
        # force_gru=True forces the GRU fallback (SimplifiedMamba) even when
        # mamba_ssm is installed.  Enables fair GRU vs Mamba comparisons.
        force_gru: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.use_sum_token = use_sum_token
        self.use_tupe = use_tupe
        self.use_time_interval = use_time_interval
        self.user_fusion_mode = user_fusion_mode
        self.force_gru = force_gru

        _VALID_FUSION_MODES = ("broadcast", "film", "head", "gated_head", "normed_head", "hybrid")
        assert user_fusion_mode in _VALID_FUSION_MODES, (
            f"user_fusion_mode must be one of {_VALID_FUSION_MODES}, got {user_fusion_mode!r}"
        )

        # ── Content embeddings (no positional — TUPE untied) ──────────────────
        self.item_embedding       = nn.Embedding(num_items,       d_model, padding_idx=0)
        self.genre_embedding      = nn.Embedding(num_genres,      d_model, padding_idx=0)
        self.age_embedding        = nn.Embedding(num_ages,        d_model)
        self.gender_embedding     = nn.Embedding(num_genders,     d_model)
        self.occupation_embedding = nn.Embedding(num_occupations, d_model)
        self.time_embedding       = nn.Embedding(num_time_slots,  d_model)

        # ── Time interval embedding (TiSASRec) ────────────────────────────────
        # Index 0 = padding / no-previous-item; indices 1…num_bins-1 = Δt buckets.
        if use_time_interval:
            self.time_interval_embedding = nn.Embedding(
                num_time_interval_bins, d_model, padding_idx=0
            )
        else:
            self.time_interval_embedding = None

        # ── User conditioning projections (FiLM / Head modes) ────────────────
        if user_fusion_mode == "film":
            # FiLM: learn per-sample scale (gamma) and shift (beta) from user_emb
            self.user_gamma_proj = nn.Linear(d_model, d_model)
            self.user_beta_proj  = nn.Linear(d_model, d_model)
        elif user_fusion_mode in ("head", "gated_head", "normed_head", "hybrid"):
            # 2-layer MLP projection (richer than a single Linear)
            self.user_head_proj = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            if user_fusion_mode == "gated_head":
                # Sigmoid gate: controls how much user info enters last_hidden
                self.user_gate = nn.Linear(d_model, d_model)
            elif user_fusion_mode == "normed_head":
                self.user_head_norm = nn.LayerNorm(d_model)
            elif user_fusion_mode == "hybrid":
                # Learnable scale for the light broadcast component
                self.hybrid_alpha = nn.Parameter(torch.tensor(0.1))
                # Gate for the head component
                self.user_gate = nn.Linear(d_model, d_model)

        # ── TUPE: position embedding applied AFTER backbone (untied) ─────────
        # Size = max_seq_len + 1 to cover all item positions (0…max_seq_len-1)
        # plus the SUM token position (max_seq_len).
        self.position_embedding = nn.Embedding(max_seq_len + 1, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        # ── SUM Token: learnable query token appended at end of sequence ──────
        # Initialized near zero; model learns to use it as a summary position.
        self.sum_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # ── Genre Pooling ─────────────────────────────────────────────────────
        self.genre_pooling = GenrePooling(pooling_type="mean")

        # ── Dropout ───────────────────────────────────────────────────────────
        self.dropout    = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout)

        # ── Mamba Backbone ────────────────────────────────────────────────────
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, force_gru=force_gru)
            for _ in range(n_layers)
        ])

        self.final_norm  = nn.LayerNorm(d_model)
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
        # SUM token: small normal init (same scale as embeddings)
        nn.init.normal_(self.sum_token, mean=0, std=0.02)

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

    def _apply_user_conditioning(
        self,
        last_hidden: torch.Tensor,   # (B, d_model)
        user_emb: torch.Tensor,      # (B, 1, d_model)
    ) -> torch.Tensor:
        """
        Apply user conditioning to the last hidden state based on fusion mode.

        "broadcast"   — no-op (user was already fused in input)
        "film"        — FiLM: gamma * last_hidden + beta
        "head"        — additive MLP: last_hidden + MLP(user_emb)
        "gated_head"  — gated MLP:   last_hidden + sigmoid(W·user_emb) * MLP(user_emb)
        "normed_head" — normed MLP:  LayerNorm(last_hidden + MLP(user_emb))
        "hybrid"      — gated head:  last_hidden + sigmoid(W·user_emb) * MLP(user_emb)
                        (light broadcast is handled separately in forward())
        """
        if self.user_fusion_mode == "broadcast":
            return last_hidden

        user_vec = user_emb.squeeze(1)  # (B, d_model)

        if self.user_fusion_mode == "film":
            gamma = self.user_gamma_proj(user_vec)
            beta  = self.user_beta_proj(user_vec)
            return gamma * last_hidden + beta

        proj = self.user_head_proj(user_vec)       # (B, d_model)

        if self.user_fusion_mode == "head":
            return last_hidden + proj

        if self.user_fusion_mode in ("gated_head", "hybrid"):
            gate = torch.sigmoid(self.user_gate(user_vec))   # (B, d_model)
            return last_hidden + gate * proj

        # "normed_head"
        return self.user_head_norm(last_hidden + proj)

    def forward(
        self,
        item_seq: torch.Tensor,
        genre_seq: torch.Tensor,
        time_seq: torch.Tensor,
        age_idx: torch.Tensor,
        gender_idx: torch.Tensor,
        occupation: torch.Tensor,
        return_hidden: bool = False,
        delta_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            item_seq:      (batch, seq_len)                — left-padded item IDs
            genre_seq:     (batch, seq_len, num_genres)
            time_seq:      (batch, seq_len)
            age_idx:       (batch, 1)
            gender_idx:    (batch, 1)
            occupation:    (batch, 1)
            return_hidden: if True, return full hidden states instead of logits
            delta_seq:     (batch, seq_len) optional — TiSASRec consecutive Δt buckets

        Returns:
            logits (batch, num_items)
            or hidden (batch, seq_len + 1, d_model)  — +1 for the SUM token
        """
        batch_size, seq_len = item_seq.shape

        # ── Content embeddings (TUPE: no pos_emb here) ───────────────────────
        item_emb     = self.item_embedding(item_seq)                   # (B, L, d)
        genre_emb    = self.genre_embedding(genre_seq)                 # (B, L, G, d)
        genre_mask   = (genre_seq != 0)
        genre_pooled = self.genre_pooling(genre_emb, genre_mask)       # (B, L, d)
        time_emb     = self.time_embedding(time_seq)                   # (B, L, d)

        # user_emb_raw: (B, 1, d) — kept for FiLM / Head conditioning later
        user_emb_raw = self.get_user_profile_embedding(age_idx, gender_idx, occupation)

        # ── Time interval embedding (TiSASRec) ───────────────────────────────────
        # Consecutive Δt bucket per position; index 0 = no prev / padding.
        if self.use_time_interval and delta_seq is not None:
            delta_emb = self.time_interval_embedding(delta_seq)        # (B, L, d)
        else:
            delta_emb = 0

        # ── Input content fusion ──────────────────────────────────────────────
        # "broadcast": user_emb is added here at full scale (original behaviour)
        # "hybrid":    user_emb is added at learnable small scale (hybrid_alpha)
        # others:      backbone sees only sequential signals; user applied later
        if self.user_fusion_mode in ("broadcast", "hybrid"):
            user_emb_seq = user_emb_raw.expand(-1, seq_len, -1)        # (B, L, d)
            alpha = 1.0 if self.user_fusion_mode == "broadcast" \
                        else self.hybrid_alpha.clamp(0.0, 1.0)
            hidden = item_emb + genre_pooled + alpha * user_emb_seq + time_emb + delta_emb
        else:
            hidden = item_emb + genre_pooled + time_emb + delta_emb    # (B, L, d)
        hidden = self.emb_dropout(hidden)

        # Zero-out padding positions before the backbone
        padding_mask = (item_seq != 0).float().unsqueeze(-1)           # (B, L, 1)
        hidden = hidden * padding_mask

        # ── SUM Token (ablation flag: use_sum_token) ─────────────────────────
        if self.use_sum_token:
            sum_tok  = self.sum_token.expand(batch_size, -1, -1)       # (B, 1, d)
            hidden   = torch.cat([hidden, sum_tok], dim=1)             # (B, L+1, d)
            sum_mask = torch.ones(batch_size, 1, 1, device=item_seq.device)
            full_mask = torch.cat([padding_mask, sum_mask], dim=1)     # (B, L+1, 1)
            seq_out_len = seq_len + 1
        else:
            # No SUM token: sequence length stays L, use last real item position
            full_mask   = padding_mask                                 # (B, L, 1)
            seq_out_len = seq_len

        # ── Mamba / GRU backbone ─────────────────────────────────────────────
        for layer in self.mamba_layers:
            hidden = layer(hidden)
            hidden = hidden * full_mask

        hidden = self.final_norm(hidden)                               # (B, seq_out_len, d)

        # ── TUPE: inject position bias AFTER backbone (ablation flag: use_tupe) ──
        if self.use_tupe:
            positions = torch.arange(seq_out_len, device=item_seq.device).unsqueeze(0)
            pos_emb   = self.position_embedding(positions)             # (1, seq_out_len, d)
            pos_bias  = self.pos_proj(pos_emb)                        # (1, seq_out_len, d)
            hidden    = hidden + pos_bias * full_mask                  # mask out padding

        if return_hidden:
            return hidden

        # User representation: SUM token (last pos) or last real item position.
        # Sequences are LEFT-padded: real items occupy the RIGHTMOST positions,
        # so hidden[:, -1, :] always holds the most-recent item's representation.
        # (The old sum-based last_idx was computing the count of non-zero items
        # which gave the wrong position index for left-padded tensors.)
        last_hidden = hidden[:, -1, :]

        # Apply FiLM / Head user conditioning (no-op for "broadcast")
        last_hidden = self._apply_user_conditioning(last_hidden, user_emb_raw)

        return self.output_proj(last_hidden)                           # (B, num_items)

    def predict_scores(
        self,
        item_seq: torch.Tensor,
        genre_seq: torch.Tensor,
        time_seq: torch.Tensor,
        age_idx: torch.Tensor,
        gender_idx: torch.Tensor,
        occupation: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
        delta_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict scores for candidate items.

        Args:
            candidate_items: (batch, num_candidates)  — if None, score all items
            delta_seq:       (batch, seq_len)  — optional TiSASRec Δt buckets

        Returns:
            scores (batch, num_candidates) or (batch, num_items)
        """
        hidden = self.forward(
            item_seq, genre_seq, time_seq,
            age_idx, gender_idx, occupation,
            return_hidden=True,
            delta_seq=delta_seq,
        )
        # Last position = SUM token (full-sequence user representation)
        last_hidden = hidden[:, -1, :]                                 # (batch, d_model)

        # Apply FiLM / Head user conditioning (no-op for "broadcast" mode).
        # Re-compute user_emb_raw here — cheap embedding lookup, avoids API change.
        if self.user_fusion_mode != "broadcast":
            user_emb_raw = self.get_user_profile_embedding(age_idx, gender_idx, occupation)
            last_hidden  = self._apply_user_conditioning(last_hidden, user_emb_raw)

        if candidate_items is not None:
            candidate_emb = self.item_embedding(candidate_items)       # (batch, num_cand, d_model)
            return torch.bmm(candidate_emb, last_hidden.unsqueeze(-1)).squeeze(-1)
        else:
            return self.output_proj(last_hidden)


def create_mamba4rec(metadata: Dict, **kwargs) -> Mamba4Rec:
    """Factory — build Mamba4Rec from the metadata dict produced by materialization.

    All ablation flags (use_sum_token, use_tupe, use_time_interval,
    num_time_interval_bins) can be passed via **kwargs.
    """
    return Mamba4Rec(
        num_items=metadata["num_items"],
        num_genres=metadata["num_genres"],
        num_ages=metadata["num_ages"],
        num_genders=metadata["num_genders"],
        num_occupations=metadata["num_occupations"],
        num_time_slots=metadata["num_time_slots"],
        **kwargs,
    )
