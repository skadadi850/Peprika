from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    CATEGORICAL_AXES,
    CONTINUOUS_AXES,
    DROPOUT_EXEMPT_AXES,
    LEGACY_CATEGORICAL_AXES,
    LEGACY_CONTINUOUS_AXES,
    MAX_CYSTEINES,
    MAX_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    STANDARD_AA,
    WHEEL_BIN_COUNTS,
    WHEEL_BIN_UNITS,
    WHEEL_FORWARD,
    WHEEL_OFFSETS,
    WHEEL_PITCHES,
    WHEEL_REVERSE,
    WHEEL_SIZE,
    categorical_sizes,
    concat_conditions,
    drop_condition_axes,
    normalize_condition,
)

PAD_TOKEN = len(STANDARD_AA)
MASK_TOKEN = len(STANDARD_AA) + 1
VOCAB_SIZE = len(STANDARD_AA) + 2

AA_TO_TOKEN = {aa: index for index, aa in enumerate(STANDARD_AA)}
TOKEN_TO_AA = {index: aa for aa, index in AA_TO_TOKEN.items()}
CYS_TOKEN = AA_TO_TOKEN["C"]

# wheel: helical-wheel order chosen from the amphipathic face of the sequence (the wheel class is an input);
# forward: left to right; random: a new random permutation of the positions for every sequence and every call
ORDERS = ("wheel", "forward", "random")


def encode_sequence(sequence: str, canvas: int = MAX_PEPTIDE_LENGTH) -> list[int]:
    tokens = [AA_TO_TOKEN[aa] for aa in str(sequence).upper() if aa in AA_TO_TOKEN]
    tokens = tokens[:canvas]
    return tokens + [PAD_TOKEN] * (canvas - len(tokens))


def decode_tokens(tokens) -> str:
    residues = []
    for raw_token in tokens:
        token = int(raw_token)
        if token == PAD_TOKEN:
            break
        if token == MASK_TOKEN:
            continue
        residues.append(TOKEN_TO_AA[token])
    return "".join(residues)


class AMPConditionEncoder(nn.Module):
    def __init__(self, num_species: int, embed_dim: int = 128, output_dim: int = 1024, continuous_axes=None, categorical_axes=None):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.output_dim = int(output_dim)
        self.num_species = int(num_species)
        self.categorical_axes = tuple(CATEGORICAL_AXES if categorical_axes is None else categorical_axes)
        self.continuous_axes = tuple(CONTINUOUS_AXES if continuous_axes is None else continuous_axes)
        self.axis_sizes = categorical_sizes(num_species, self.categorical_axes)
        self.categorical = nn.ModuleDict({axis: nn.Embedding(size, self.embed_dim) for axis, size in self.axis_sizes.items()})
        hidden_dim = max(1, self.embed_dim // 2)
        self.continuous = nn.ModuleDict(
            {
                axis: nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.embed_dim))
                for axis in self.continuous_axes
            }
        )
        self.null = nn.ParameterDict({axis: nn.Parameter(torch.empty(1, self.embed_dim)) for axis in self.continuous_axes})
        concat_dim = self.embed_dim * (len(self.categorical_axes) + len(self.continuous_axes))
        self.output_proj = nn.Sequential(nn.Linear(concat_dim, self.output_dim), nn.GELU(), nn.Linear(self.output_dim, self.output_dim))
        for embedding in self.categorical.values():
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        for parameter in self.null.values():
            nn.init.normal_(parameter, std=0.02)

    def null_condition(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        condition: Dict[str, torch.Tensor] = {}
        for axis in self.categorical_axes:
            condition[f"{axis}_idx"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        for axis in self.continuous_axes:
            condition[axis] = torch.zeros(batch_size, 1, device=device)
            condition[f"{axis}_mask"] = torch.zeros(batch_size, 1, device=device)
        return condition

    def forward(self, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for axis in self.categorical_axes:
            index = condition[f"{axis}_idx"].long().reshape(-1)
            parts.append(self.categorical[axis](index))
        for axis in self.continuous_axes:
            value = condition[axis].float().reshape(-1, 1)
            mask = condition.get(f"{axis}_mask")
            if mask is None:
                mask = torch.ones_like(value)
            else:
                mask = mask.float().reshape(-1, 1)
            projected = self.continuous[axis](value * mask)
            parts.append(projected * mask + self.null[axis] * (1.0 - mask))
        return self.output_proj(torch.cat(parts, dim=-1))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("time embedding dimension must be even and >= 4")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / (half - 1)
        frequencies = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        angles = t.float()[:, None] * 1000.0 * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, cond_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).unsqueeze(1).chunk(6, dim=-1)
        hidden = self.norm1(x) * (1.0 + scale1) + shift1
        attended, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        x = x + gate1 * attended
        hidden = self.norm2(x) * (1.0 + scale2) + shift2
        return x + gate2 * self.ff(hidden)


class MaskedDiffusionModel(nn.Module):
    def __init__(
        self,
        num_species: int,
        canvas: int = MAX_PEPTIDE_LENGTH,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        cond_embed_dim: int = 128,
        min_length: int = MIN_PEPTIDE_LENGTH,
        max_length: int = MAX_PEPTIDE_LENGTH,
        max_cysteines: int = MAX_CYSTEINES,
        wheel_prior=None,
        wheel_threshold: Optional[float] = None,
        order: str = "wheel",
        continuous_axes=None,
        categorical_axes=None,
        device: str = "cuda",
    ):
        super().__init__()
        if wheel_prior is None or wheel_threshold is None:
            raise ValueError("wheel_prior and wheel_threshold are required")
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        continuous_axes = tuple(CONTINUOUS_AXES if continuous_axes is None else continuous_axes)
        categorical_axes = tuple(CATEGORICAL_AXES if categorical_axes is None else categorical_axes)
        prior = torch.tensor(wheel_prior, dtype=torch.float32)
        if prior.shape != (3, WHEEL_SIZE):
            raise ValueError(f"wheel_prior must have shape (3, {WHEEL_SIZE})")
        self.config = {
            "continuous_axes": continuous_axes,
            "categorical_axes": categorical_axes,
            "num_species": int(num_species),
            "canvas": int(canvas),
            "d_model": int(d_model),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "ff_dim": int(ff_dim),
            "dropout": float(dropout),
            "cond_embed_dim": int(cond_embed_dim),
            "min_length": int(min_length),
            "max_length": int(max_length),
            "max_cysteines": int(max_cysteines),
            "wheel_prior": tuple(tuple(float(value) for value in row) for row in prior.tolist()),
            "wheel_threshold": float(wheel_threshold),
            "order": order,
        }
        self.canvas = int(canvas)
        self.d_model = int(d_model)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.max_cysteines = int(max_cysteines)
        self.wheel_threshold = float(wheel_threshold)
        self.order = order
        self.vocab_size = VOCAB_SIZE
        self.token_embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, self.canvas, self.d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(256), nn.Linear(256, self.d_model), nn.SiLU(), nn.Linear(self.d_model, self.d_model)
        )
        self.condition_encoder = AMPConditionEncoder(
            num_species=num_species,
            embed_dim=cond_embed_dim,
            output_dim=self.d_model,
            continuous_axes=continuous_axes,
            categorical_axes=categorical_axes,
        )
        self.blocks = nn.ModuleList(
            [AdaLNTransformerBlock(self.d_model, num_heads, ff_dim, self.d_model, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.d_model, 2 * self.d_model))
        self.output_proj = nn.Linear(self.d_model, VOCAB_SIZE)
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        kind = [0] * WHEEL_SIZE
        bins = [1] * WHEEL_SIZE
        unit = [0] * WHEEL_SIZE
        center = [0] * WHEEL_SIZE
        for pitch_index in range(len(WHEEL_PITCHES)):
            for bin_index in range(WHEEL_BIN_COUNTS[pitch_index]):
                wheel = WHEEL_OFFSETS[pitch_index] + bin_index
                kind[wheel] = 1
                bins[wheel] = WHEEL_BIN_COUNTS[pitch_index]
                unit[wheel] = WHEEL_BIN_UNITS[pitch_index]
                center[wheel] = bin_index
        kind[WHEEL_FORWARD] = 2
        kind[WHEEL_REVERSE] = 3
        positions = torch.arange(self.canvas, dtype=torch.float32)
        self.register_buffer("_positions", positions, persistent=False)
        self.register_buffer("_wheel_kind", torch.tensor(kind, dtype=torch.long), persistent=False)
        self.register_buffer("_wheel_bins", torch.tensor(bins, dtype=torch.long), persistent=False)
        self.register_buffer("_wheel_unit", torch.tensor(unit, dtype=torch.long), persistent=False)
        self.register_buffer("_wheel_center", torch.tensor(center, dtype=torch.long), persistent=False)
        self.register_buffer("_wheel_prior", prior, persistent=False)
        self.to(device)

    def logits(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Mapping[str, torch.Tensor]] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        if condition is None:
            condition = self.condition_encoder.null_condition(batch_size, tokens.device)
        device_type = tokens.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            condition_vector = self.time_embedding(t) + self.condition_encoder(condition)
            x = self.token_embedding(tokens) + self.position_embedding
        use_autocast = autocast_dtype is not None and device_type == "cuda"
        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
            for block in self.blocks:
                x = block(x, condition_vector)
        with torch.autocast(device_type=device_type, enabled=False):
            x = x.float()
            shift, scale = self.final_modulation(condition_vector).unsqueeze(1).chunk(2, dim=-1)
            x = self.final_norm(x) * (1.0 + scale) + shift
            output = self.output_proj(x)
        output[..., MASK_TOKEN] = torch.finfo(output.dtype).min
        output[..., PAD_TOKEN] = torch.finfo(output.dtype).min
        return output

    def forward_rank(self, lengths: torch.Tensor) -> torch.Tensor:
        positions = self._positions.long()[None, :]
        key = torch.where(positions < lengths[:, None], positions, 10**6 + positions)
        return key.argsort(dim=1).argsort(dim=1)

    def order_rank(self, wheel_idx: torch.Tensor, lengths: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # rank[b, p] is the step at which position p of sequence b is revealed; positions beyond the length come last
        if self.order == "forward":
            return self.forward_rank(lengths)
        canvas = self.canvas
        positions = self._positions.long()[None, :]
        inside = positions < lengths[:, None]
        if self.order == "random":
            draw = torch.rand(lengths.shape[0], canvas, device=lengths.device, generator=generator)
            key = draw.argsort(dim=1).argsort(dim=1)
        else:
            kind = self._wheel_kind[wheel_idx][:, None]
            bins = self._wheel_bins[wheel_idx][:, None]
            unit = self._wheel_unit[wheel_idx][:, None]
            center = self._wheel_center[wheel_idx][:, None]
            bin_index = (positions * unit) % bins
            diff = (bin_index - center) % bins
            distance = torch.minimum(diff, bins - diff)
            side = (diff > bins - diff).long()
            wheel_key = distance * (2 * canvas) + side * canvas + positions
            key = torch.where(kind == 1, wheel_key, torch.where(kind == 2, positions, (canvas - 1) - positions))
        key = torch.where(inside, key, 10**6 + positions)
        return key.argsort(dim=1).argsort(dim=1)

    def wheel_prior_row(self, condition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        value = condition["mu_h"].float().reshape(-1)
        observed = condition["mu_h_mask"].float().reshape(-1) > 0.5
        high = (value >= self.wheel_threshold).long()
        return torch.where(observed, high, torch.full_like(high, 2))

    def draw_wheel(self, condition: Mapping[str, torch.Tensor], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        probabilities = self._wheel_prior[self.wheel_prior_row(condition)]
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)

    def _ordered_log_probability(
        self,
        tokens: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
        rank: torch.Tensor,
        lengths: torch.Tensor,
        autocast_dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        # log probability of every sequence under one reveal order: the sum over steps of the log probability of the
        # residue at the frontier given the residues revealed before it
        batch_size = tokens.shape[0]
        device = tokens.device
        max_length = int(lengths.max().item())
        positions = self._positions.long()[None, :]
        batch_index = torch.arange(batch_size, device=device)
        safe_tokens = tokens.masked_fill(tokens == PAD_TOKEN, 0)
        order = rank.argsort(dim=1)
        total = torch.zeros(batch_size, device=device)
        for step in range(max_length):
            active = lengths > step
            masked = (positions < lengths[:, None]) & (rank >= step)
            corrupted = torch.where(masked, torch.full_like(tokens, MASK_TOKEN), tokens)
            t = (lengths - step).clamp_min(1).float() / lengths.float()
            logits = self.logits(corrupted, t, condition, autocast_dtype=autocast_dtype)
            frontier = order[:, step]
            log_probability = torch.log_softmax(logits[batch_index, frontier].float(), dim=-1)
            picked = log_probability.gather(1, safe_tokens[batch_index, frontier][:, None]).squeeze(1)
            total += torch.where(active, picked, torch.zeros_like(picked))
        return total

    @torch.no_grad()
    def exact_log_likelihood(
        self,
        tokens: torch.Tensor,
        condition: Optional[Mapping[str, torch.Tensor]] = None,
        autocast_dtype: Optional[torch.dtype] = None,
        random_orders: int = 16,
        order_seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Log probability of each sequence given its length and the condition.
        #   wheel   exact: log of the prior-weighted sum over the wheel classes, each class being one reveal order.
        #           The prior depends on the requested hydrophobic moment only, never on the sequence.
        #   forward exact: the left-to-right factorization.
        #   random  log of the mean over --random_orders random orders, a stochastic lower bound of the log
        #           probability under the uniform mixture of all orders.
        # The second output holds the log probability under each order, one column per wheel class (column 0 unused)
        # or per random order.
        self.eval()
        device = self.position_embedding.device
        tokens = tokens.to(device)
        batch_size = tokens.shape[0]
        condition = normalize_condition({} if condition is None else condition, batch_size, device)
        lengths = (tokens != PAD_TOKEN).sum(dim=1)
        if self.order == "wheel":
            totals = torch.zeros(batch_size, WHEEL_SIZE, device=device)
            for wheel in range(1, WHEEL_SIZE):
                wheel_idx = torch.full((batch_size,), wheel, dtype=torch.long, device=device)
                local = dict(condition)
                local["wheel_idx"] = wheel_idx
                rank = self.order_rank(wheel_idx, lengths)
                totals[:, wheel] = self._ordered_log_probability(tokens, local, rank, lengths, autocast_dtype)
            prior = self._wheel_prior[self.wheel_prior_row(condition)]
            joint = torch.log(prior) + totals
            return torch.logsumexp(joint, dim=1), totals
        local = dict(condition)
        local["wheel_idx"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        if self.order == "forward":
            rank = self.forward_rank(lengths)
            total = self._ordered_log_probability(tokens, local, rank, lengths, autocast_dtype)
            return total, total[:, None]
        generator = torch.Generator(device=device)
        generator.manual_seed(int(order_seed))
        columns = []
        for _ in range(int(random_orders)):
            rank = self.order_rank(local["wheel_idx"], lengths, generator)
            columns.append(self._ordered_log_probability(tokens, local, rank, lengths, autocast_dtype))
        totals = torch.stack(columns, dim=1)
        return torch.logsumexp(totals, dim=1) - math.log(totals.shape[1]), totals

    def forward(
        self,
        tokens: torch.Tensor,
        condition: Mapping[str, torch.Tensor],
        p_axis: float = 0.15,
        p_all: float = 0.10,
        aux_weight: float = 1.0,
        p_tasks: float = 0.0,
        p_properties: float = 0.0,
        p_measurements: float = 0.0,
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        device = tokens.device
        lengths = (tokens != PAD_TOKEN).sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("every sequence needs at least one residue")
        condition = normalize_condition(condition, batch_size, device)
        wheel_idx = condition["wheel_idx"].long()
        if self.order == "wheel":
            if (wheel_idx == 0).any():
                raise ValueError("wheel_idx is unset; build the dataset with wheel_thresholds")
            flip = (wheel_idx == WHEEL_FORWARD) & (torch.rand(batch_size, device=device) < 0.5)
            wheel_idx = torch.where(flip, torch.full_like(wheel_idx, WHEEL_REVERSE), wheel_idx)
        else:
            wheel_idx = torch.zeros_like(wheel_idx)
        condition["wheel_idx"] = wheel_idx
        rank = self.order_rank(wheel_idx, lengths)
        positions = self._positions.long()[None, :]
        real = positions < lengths[:, None]
        strata = (torch.randperm(batch_size, device=device).float() + torch.rand(batch_size, device=device)) / batch_size
        step = torch.minimum((strata * lengths.float()).long(), lengths - 1)
        masked = real & (rank >= step[:, None])
        frontier = rank == step[:, None]
        t = (lengths - step).float() / lengths.float()
        corrupted = torch.where(masked, torch.full_like(tokens, MASK_TOKEN), tokens)
        if p_axis or p_all or p_tasks or p_properties or p_measurements:
            condition = drop_condition_axes(
                condition, p_axis=p_axis, p_all=p_all, p_tasks=p_tasks,
                p_properties=p_properties, p_measurements=p_measurements,
            )
        logits = self.logits(corrupted, t, condition)
        safe_tokens = tokens.masked_fill(tokens == PAD_TOKEN, 0)
        token_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), safe_tokens.reshape(-1), reduction="none"
        ).reshape(batch_size, self.canvas)
        frontier_loss = (token_loss * frontier).sum(dim=1)
        frontier_term = (frontier_loss * lengths.float()).sum() / lengths.sum()
        if not aux_weight:
            return frontier_term
        aux_mask = masked & ~frontier
        aux_term = (token_loss * aux_mask).sum() / aux_mask.sum().clamp_min(1)
        return frontier_term + float(aux_weight) * aux_term

    def guided_logits(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Mapping[str, torch.Tensor]],
        guidance_weight: float,
        exempt_axes: Sequence[str] = DROPOUT_EXEMPT_AXES,
        autocast_dtype: Optional[torch.dtype] = None,
        null_axes: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if condition is None:
            return self.logits(tokens, t, None, autocast_dtype=autocast_dtype)
        if guidance_weight == 1.0:
            return self.logits(tokens, t, condition, autocast_dtype=autocast_dtype)
        batch_size = tokens.shape[0]
        null = self.condition_encoder.null_condition(batch_size, tokens.device)
        aligned = {key: condition[key] for key in null}
        if null_axes is None:
            for axis in exempt_axes:
                key = f"{axis}_idx"
                if key in condition and key in null:
                    null[key] = condition[key].long().reshape(-1).clone()
        else:
            null = {key: value.clone() for key, value in aligned.items()}
            for axis in null_axes:
                key = f"{axis}_idx"
                if key in null:
                    null[key] = torch.zeros_like(null[key])
        merged = concat_conditions(aligned, null)
        both = self.logits(torch.cat((tokens, tokens), dim=0), torch.cat((t, t), dim=0), merged, autocast_dtype=autocast_dtype)
        conditional = both[:batch_size]
        unconditional = both[batch_size:]
        return unconditional + float(guidance_weight) * (conditional - unconditional)

    def cysteine_cap(self, condition: Optional[Mapping[str, torch.Tensor]], batch_size: int, device) -> torch.Tensor:
        # cys_class 1 allows no cysteine, 2 allows up to MAX_CYSTEINES, 3 and unset allow up to
        # the model limit. A checkpoint whose limit is below MAX_CYSTEINES keeps its own limit.
        cap = torch.full((batch_size,), self.max_cysteines, dtype=torch.long, device=device)
        if condition is None or "cys_class_idx" not in condition:
            return cap
        cys_class = condition["cys_class_idx"].long().reshape(-1)
        cap = torch.where(cys_class == 1, torch.zeros_like(cap), cap)
        cap = torch.where(cys_class == 2, cap.clamp_max(MAX_CYSTEINES), cap)
        return cap

    def constrain_logits(self, logits: torch.Tensor, tokens: torch.Tensor, cap: Optional[torch.Tensor] = None) -> torch.Tensor:
        neg_inf = torch.finfo(logits.dtype).min
        if cap is None:
            cap = torch.full((tokens.shape[0],), self.max_cysteines, dtype=torch.long, device=tokens.device)
        committed_cys = (tokens == CYS_TOKEN).sum(dim=1)
        exhausted = committed_cys >= cap
        if exhausted.any():
            column = logits[..., CYS_TOKEN]
            logits[..., CYS_TOKEN] = torch.where(exhausted[:, None], torch.full_like(column, neg_inf), column)
        return logits

    def _sample_revealed_tokens(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        reveal: torch.Tensor,
        temperature: float,
        generator: Optional[torch.Generator],
        cap: Optional[torch.Tensor] = None,
    ) -> None:
        coordinates = reveal.nonzero(as_tuple=False)
        if not coordinates.numel():
            return
        batch_index = coordinates[:, 0]
        position_index = coordinates[:, 1]
        selected_logits = logits[batch_index, position_index]
        selected_logits = selected_logits / max(float(temperature), 1e-6)
        probabilities = F.softmax(selected_logits, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        committed_cys = (tokens == CYS_TOKEN).sum(dim=1)
        if (committed_cys > self.max_cysteines).any():
            raise ValueError("initial tokens exceed the cysteine limit")
        if cap is None:
            cap = torch.full_like(committed_cys, self.max_cysteines)
        remaining = (cap - committed_cys).clamp_min(0)
        sampled_cys = sampled == CYS_TOKEN
        if sampled_cys.any():
            sampled_cys_count = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
            sampled_cys_count.scatter_add_(0, batch_index, sampled_cys.long())
            overflow_batches = (sampled_cys_count > remaining).nonzero(as_tuple=True)[0]
            replace_indices = []
            for batch in overflow_batches.tolist():
                local = ((batch_index == batch) & sampled_cys).nonzero(as_tuple=True)[0]
                keep = int(remaining[batch].item())
                if keep < local.numel():
                    order = torch.randperm(local.numel(), device=tokens.device, generator=generator)
                    replace_indices.append(local[order[keep:]])
            if replace_indices:
                replace_indices = torch.cat(replace_indices)
                alternate_logits = selected_logits[replace_indices].clone()
                alternate_logits[:, CYS_TOKEN] = torch.finfo(alternate_logits.dtype).min
                alternate = torch.multinomial(F.softmax(alternate_logits, dim=-1), 1, generator=generator).squeeze(1)
                sampled[replace_indices] = alternate
        tokens[batch_index, position_index] = sampled

    def target_lengths(self, condition: Optional[Mapping[str, torch.Tensor]], batch_size: int, device) -> torch.Tensor:
        if condition is None or "length_norm" not in condition:
            value = torch.full((batch_size,), 0.5, device=device)
        else:
            value = condition["length_norm"].float().reshape(-1)
            mask = condition.get("length_norm_mask")
            if mask is not None:
                mask = mask.float().reshape(-1)
                value = value * mask + 0.5 * (1.0 - mask)
        span = float(self.max_length - self.min_length)
        lengths = (value * span + self.min_length).round().long()
        return lengths.clamp(self.min_length, self.max_length)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: Optional[Mapping[str, torch.Tensor]] = None,
        guidance_weight: float = 3.0,
        num_steps: int = 128,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        free_length: bool = False,
        reveal: str = "wheel",
        pad_bias: float = 0.0,
        guidance_exempt_axes: Sequence[str] = DROPOUT_EXEMPT_AXES,
        guidance_null_axes: Optional[Sequence[str]] = None,
        init_tokens: Optional[torch.Tensor] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> list[str]:
        if free_length:
            raise ValueError("free_length is unsupported: PAD positions are outside the generation order")
        if pad_bias:
            raise ValueError("pad_bias is unsupported: PAD positions are outside the generation order")
        if reveal != "wheel":
            raise ValueError("reveal must be 'wheel'")
        self.eval()
        device = self.position_embedding.device
        if device.type != "cuda":
            autocast_dtype = None
        if init_tokens is None:
            tokens = torch.full((batch_size, self.canvas), MASK_TOKEN, dtype=torch.long, device=device)
        else:
            tokens = init_tokens.to(device=device, dtype=torch.long).clone()
            if tokens.ndim != 2 or tokens.shape[1] != self.canvas:
                raise ValueError(f"init_tokens must have shape (batch, {self.canvas})")
            batch_size = tokens.shape[0]
        condition = normalize_condition({} if condition is None else condition, batch_size, device)
        if ((tokens == CYS_TOKEN).sum(dim=1) > self.max_cysteines).any():
            raise ValueError("initial sequence exceeds cysteine limit")
        positions = self._positions.long()[None, :]
        if init_tokens is not None:
            occupied = tokens != PAD_TOKEN
            target_length = torch.where(occupied, positions + 1, torch.zeros_like(positions)).amax(dim=1)
            target_length = target_length.clamp(self.min_length, self.max_length)
        else:
            target_length = self.target_lengths(condition, batch_size, device)
        outside = positions >= target_length[:, None]
        tokens = torch.where(~outside & (tokens == PAD_TOKEN), torch.full_like(tokens, MASK_TOKEN), tokens)
        tokens = torch.where(outside, torch.full_like(tokens, PAD_TOKEN), tokens)
        wheel_idx = condition["wheel_idx"].long()
        if self.order == "wheel":
            missing = wheel_idx == 0
            if missing.any():
                wheel_idx = torch.where(missing, self.draw_wheel(condition, generator), wheel_idx)
        else:
            wheel_idx = torch.zeros_like(wheel_idx)
        condition["wheel_idx"] = wheel_idx
        rank = self.order_rank(wheel_idx, target_length, generator)
        cap = self.cysteine_cap(condition, batch_size, device)
        initial_masked = (tokens == MASK_TOKEN).sum(dim=1)
        if not initial_masked.any():
            return [decode_tokens(row.tolist()) for row in tokens]
        fractions = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for step in range(num_steps):
            still_masked = tokens == MASK_TOKEN
            remaining = still_masked.sum(dim=1)
            if not remaining.any():
                break
            target_remaining = torch.round(initial_masked.float() * fractions[step + 1]).long()
            count = (remaining - target_remaining).clamp_min(0)
            if not count.any():
                continue
            masked_rank = torch.where(still_masked, rank, torch.full_like(rank, self.canvas))
            local = masked_rank.argsort(dim=1).argsort(dim=1)
            to_reveal = still_masked & (local < count[:, None])
            t_now = (remaining.float() / target_length.float()).clamp_min(1e-6)
            logits = self.guided_logits(
                tokens, t_now, condition, guidance_weight,
                exempt_axes=guidance_exempt_axes, autocast_dtype=autocast_dtype, null_axes=guidance_null_axes,
            )
            logits = self.constrain_logits(logits, tokens, cap)
            self._sample_revealed_tokens(tokens, logits, to_reveal, temperature, generator, cap)
        return [decode_tokens(row.tolist()) for row in tokens]

    def save(self, path: str) -> None:
        import os

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(target.suffix + ".tmp")
        torch.save({"model_state_dict": self.state_dict(), "config": self.config}, staging)
        os.replace(staging, target)

    @classmethod
    def load(cls, path: str, device: str = "cuda"):
        from ..predictor import fetch_if_missing

        fetch_if_missing(path)
        checkpoint = torch.load(path, map_location=device)
        config = dict(checkpoint["config"])
        config["device"] = device
        config.setdefault("continuous_axes", LEGACY_CONTINUOUS_AXES)
        config.setdefault("categorical_axes", LEGACY_CATEGORICAL_AXES)
        config.setdefault("order", "wheel")
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model