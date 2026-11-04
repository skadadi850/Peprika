from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .models.data import CATEGORICAL_AXES, CONTINUOUS_AXES, DROPOUT_EXEMPT_AXES, condition_values


def fetch_if_missing(path, base_url=None) -> None:
    return None


DEFAULT_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"


DEFAULT_ESM_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"


class ESM2Encoder:
    def __init__(
        self,
        model_path: str = DEFAULT_ESM_MODEL,
        device: str = "cuda",
        max_length: int = 64,
        revision: Optional[str] = DEFAULT_ESM_REVISION,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_path = model_path
        self.device = device
        self.max_length = int(max_length)
        self.revision = revision

        hub_args = {} if Path(model_path).exists() else {"revision": revision}

        def load(local_files_only: bool):
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=local_files_only, **hub_args
            )
            model = AutoModel.from_pretrained(
                model_path,
                add_pooling_layer=False,
                local_files_only=local_files_only,
                **hub_args,
            )
            return tokenizer, model

        try:
            self.tokenizer, model = load(True)
        except OSError:
            self.tokenizer, model = load(False)

        self.model = model.to(device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, sequences: Iterable[str], batch_size: int = 128) -> np.ndarray:
        sequences = list(sequences)

        if not sequences:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        outputs: List[np.ndarray] = []

        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            spaced = [" ".join(str(sequence).upper()) for sequence in chunk]

            encoded = self.tokenizer(
                spaced,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            hidden = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

            outputs.append(pooled.float().cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    @torch.no_grad()
    def encode_one(self, sequence: str) -> np.ndarray:
        return self.encode([sequence])[0]


DEFAULT_ESMC_MODEL = "esmc_300m"


class ESMCEncoder:
    def __init__(self, model_name: str = DEFAULT_ESMC_MODEL, device: str = "cuda"):
        import sys
        import types

        # esm imports zstd when it loads; it is used only for structure compression, which sequence
        # embedding never calls.
        if "zstd" not in sys.modules:
            try:
                import zstd  # noqa: F401
            except ModuleNotFoundError:
                sys.modules["zstd"] = types.ModuleType("zstd")

        from esm.models.esmc import ESMC

        self.device = device
        self.model = ESMC.from_pretrained(model_name, device=torch.device(device)).eval().float()
        self.tokenizer = self.model.tokenizer
        self.embedding_dim: Optional[int] = None

    @torch.no_grad()
    def encode(self, sequences: Iterable[str], batch_size: int = 128) -> np.ndarray:
        sequences = [str(sequence).upper() for sequence in sequences]

        if not sequences:
            return np.empty((0, self.embedding_dim or 0), dtype=np.float32)

        order = np.argsort([len(sequence) for sequence in sequences], kind="stable")
        pooled: List[Optional[np.ndarray]] = [None] * len(sequences)

        for start in range(0, len(sequences), batch_size):
            index = order[start : start + batch_size]
            encoded = self.tokenizer([sequences[i] for i in index], return_tensors="pt", padding=True)

            tokens = encoded["input_ids"].to(self.device)
            attention = encoded["attention_mask"].to(self.device)

            hidden = self.model(sequence_tokens=tokens).embeddings.float()

            residues = attention.clone()
            residues[:, 0] = 0
            last = attention.sum(dim=1) - 1
            residues[torch.arange(len(index), device=self.device), last] = 0

            weights = residues.unsqueeze(-1).to(hidden.dtype)
            vectors = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

            for position, row in zip(index, vectors.cpu().numpy()):
                pooled[int(position)] = row

        result = np.stack(pooled).astype(np.float32)
        self.embedding_dim = int(result.shape[1])

        return result


def resolve_device(device: Optional[str] = None) -> str:
    if device not in (None, "auto"):
        return str(device)

    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_library(
    model,
    condition: Mapping[str, torch.Tensor],
    guidance_weight: float,
    steps: int,
    temperature: float,
    batch_size: int,
    seed: Optional[int],
    free_length: bool = False,
    reveal: str = "wheel",
    pad_bias: float = 0.0,
    autocast_dtype=None,
    guidance_null_axes=None,
) -> list[str]:
    device = next(model.parameters()).device
    total = condition["species_idx"].shape[0]

    generator = None

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    sequences: list[str] = []

    for start in tqdm(range(0, total, batch_size), desc="sampling"):
        stop = min(start + batch_size, total)

        chunk = {
            key: value[start:stop].to(device, non_blocking=True)
            for key, value in condition.items()
        }

        sequences.extend(
            model.sample(
                batch_size=stop - start,
                condition=chunk,
                guidance_weight=guidance_weight,
                num_steps=steps,
                temperature=temperature,
                generator=generator,
                free_length=free_length,
                reveal=reveal,
                pad_bias=pad_bias,
                guidance_exempt_axes=DROPOUT_EXEMPT_AXES,
                guidance_null_axes=guidance_null_axes,
                autocast_dtype=autocast_dtype,
            )
        )

    return sequences


def condition_matrix(
    requests: Sequence[Mapping],
    counts: Sequence[int],
    species_vocab: Mapping[str, int],
    device: str,
) -> Dict[str, torch.Tensor]:
    if len(requests) != len(counts):
        raise ValueError("requests and counts must have equal length")

    scalar_conditions = [condition_values(request, species_vocab) for request in requests]

    counts_array = np.asarray(counts, dtype=np.int64)

    if (counts_array < 0).any():
        raise ValueError("condition counts cannot be negative")

    condition: Dict[str, torch.Tensor] = {}

    for axis in CATEGORICAL_AXES:
        key = f"{axis}_idx"
        values = np.asarray([int(item[key]) for item in scalar_conditions], dtype=np.int64)
        repeated = np.repeat(values, counts_array)

        condition[key] = torch.as_tensor(repeated, dtype=torch.long, device=device)

    for axis in CONTINUOUS_AXES:
        values = np.asarray(
            [np.nan if item[axis] is None else float(item[axis]) for item in scalar_conditions],
            dtype=np.float32,
        )
        repeated = np.repeat(values, counts_array)
        observed = np.isfinite(repeated)

        condition[axis] = torch.as_tensor(
            np.nan_to_num(repeated, nan=0.0), dtype=torch.float32, device=device
        ).view(-1, 1)

        condition[f"{axis}_mask"] = torch.as_tensor(
            observed.astype(np.float32), dtype=torch.float32, device=device
        ).view(-1, 1)

    return condition
