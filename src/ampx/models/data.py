from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

MIN_PEPTIDE_LENGTH = 8
MAX_PEPTIDE_LENGTH = 50
LENGTH_SPAN = float(MAX_PEPTIDE_LENGTH - MIN_PEPTIDE_LENGTH)

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
VALID_AA = frozenset(STANDARD_AA)
MAX_CYSTEINES = 2

TASKS = (
    "antibacterial", "antifungal", "anticancer", "antiviral", "antibiofilm",
    "antiparasitic", "antitubercular", "antiinflammatory", "hemolytic", "cytotoxic",
)

# hemolytic and cytotoxic are anti-targets: state 2 (active) means the peptide is harmful.
ANTI_TARGET_TASKS = ("hemolytic", "cytotoxic")
TASK_STATE_VALUES = {"unknown": 0, "inactive": 1, "active": 2}

MU_H_CEILING = 2.6

KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}  # fmt: skip

EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}  # fmt: skip

RESIDUE_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "E": 129.1155, "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}  # fmt: skip

WATER_MASS = 18.0153

LOG_CONC_MIN = -1.0
LOG_CONC_MAX = 3.0
LOG_CONC_SPAN = LOG_CONC_MAX - LOG_CONC_MIN
SELECTIVITY_LOG_LIMIT = 3.0

GRAM_INDEX = {"unknown": 0, "negative": 1, "positive": 2, "fungal": 3, "mycobacterial": 4}

GENUS_GRAM = {
    "escherichia": "negative", "pseudomonas": "negative", "klebsiella": "negative",
    "acinetobacter": "negative", "salmonella": "negative", "shigella": "negative",
    "enterobacter": "negative", "serratia": "negative", "proteus": "negative",
    "burkholderia": "negative", "stenotrophomonas": "negative", "haemophilus": "negative",
    "neisseria": "negative", "helicobacter": "negative", "campylobacter": "negative",
    "vibrio": "negative", "yersinia": "negative", "citrobacter": "negative",
    "moraxella": "negative", "legionella": "negative", "bordetella": "negative",
    "brucella": "negative", "francisella": "negative",
    "staphylococcus": "positive", "streptococcus": "positive", "enterococcus": "positive",
    "bacillus": "positive", "listeria": "positive", "clostridium": "positive",
    "clostridioides": "positive", "corynebacterium": "positive", "micrococcus": "positive",
    "lactobacillus": "positive", "lactococcus": "positive", "cutibacterium": "positive",
    "propionibacterium": "positive", "nocardia": "positive",
    "mycobacterium": "mycobacterial", "mycobacteroides": "mycobacterial",
    "mycolicibacterium": "mycobacterial",
    "candida": "fungal", "cryptococcus": "fungal", "aspergillus": "fungal",
    "saccharomyces": "fungal", "fusarium": "fungal", "trichophyton": "fungal",
    "malassezia": "fungal",
}  # fmt: skip

_KD_LOOKUP = np.zeros(256, dtype=np.float32)
_EISENBERG_LOOKUP = np.zeros(256, dtype=np.float32)
for _aa, _value in KYTE_DOOLITTLE.items():
    _KD_LOOKUP[ord(_aa)] = _value
for _aa, _value in EISENBERG.items():
    _EISENBERG_LOOKUP[ord(_aa)] = _value

_HM_TRIG = {}
for _width in range(1, 12):
    _angles = np.arange(_width, dtype=np.float32) * np.deg2rad(100.0)
    _HM_TRIG[_width] = (np.cos(_angles).astype(np.float32), np.sin(_angles).astype(np.float32))


def clean_sequence(sequence) -> str:
    return "".join(c for c in str(sequence).upper() if c in VALID_AA)


def is_valid_sequence(sequence: str) -> bool:
    return MIN_PEPTIDE_LENGTH <= len(sequence) <= MAX_PEPTIDE_LENGTH and all(c in VALID_AA for c in sequence)


def peptide_mass(sequence: str) -> float:
    if not sequence:
        return float("nan")
    return sum(RESIDUE_MASS.get(a, 0.0) for a in sequence) + WATER_MASS


def net_charge(sequence: str, ph: float = 7.4) -> float:
    del ph
    sequence = str(sequence).upper()
    return sequence.count("K") + sequence.count("R") + 0.1 * sequence.count("H") - sequence.count("D") - sequence.count("E")


def hydrophobic_moment(sequence: str, window: int = 11, delta_deg: float = 100.0) -> float:
    sequence = str(sequence).upper()
    if not sequence:
        return 0.0
    codes = np.frombuffer(sequence.encode("ascii", "ignore"), dtype=np.uint8)
    values = _EISENBERG_LOOKUP[codes]
    width = min(int(window), values.size)
    if width <= 0:
        return 0.0
    if delta_deg == 100.0 and width in _HM_TRIG:
        cos_terms, sin_terms = _HM_TRIG[width]
    else:
        angles = np.arange(width, dtype=np.float32) * np.deg2rad(delta_deg)
        cos_terms = np.cos(angles)
        sin_terms = np.sin(angles)
    windows = sliding_window_view(values, width)
    real = windows @ cos_terms
    imag = windows @ sin_terms
    return float(np.hypot(real, imag).max() / width)


def gravy(sequence: str) -> float:
    sequence = str(sequence).upper()
    if not sequence:
        return 0.0
    codes = np.frombuffer(sequence.encode("ascii", "ignore"), dtype=np.uint8)
    return float(_KD_LOOKUP[codes].sum() / len(sequence))


def cys_class_index(sequence: str) -> int:
    count = str(sequence).upper().count("C")
    if count == 0:
        return 1
    if count <= MAX_CYSTEINES:
        return 2
    return 3


def gram_for_species(species) -> str:
    if species is None:
        return "unknown"
    if isinstance(species, float) and np.isnan(species):
        return "unknown"
    text = str(species).strip().lower()
    if not text:
        return "unknown"
    return GENUS_GRAM.get(text.split()[0], "unknown")


def normalize_log_concentration(value_um) -> float:
    if value_um is None:
        return float("nan")
    value_um = float(value_um)
    if not np.isfinite(value_um) or value_um <= 0:
        return float("nan")
    value = (np.log10(value_um) - LOG_CONC_MIN) / LOG_CONC_SPAN
    return float(np.clip(value, 0.0, 1.0))


def denormalize_log_concentration(value_norm: float) -> float:
    return float(10.0 ** (float(value_norm) * LOG_CONC_SPAN + LOG_CONC_MIN))


def normalize_selectivity_ratio(log_ratio) -> float:
    if log_ratio is None:
        return float("nan")
    log_ratio = float(log_ratio)
    if not np.isfinite(log_ratio):
        return float("nan")
    value = log_ratio / (2.0 * SELECTIVITY_LOG_LIMIT) + 0.5
    return float(np.clip(value, 0.0, 1.0))


def denormalize_selectivity_ratio(value_norm: float) -> float:
    return float((float(value_norm) - 0.5) * 2.0 * SELECTIVITY_LOG_LIMIT)


def normalize_length(length) -> float:
    return float(np.clip((float(length) - MIN_PEPTIDE_LENGTH) / LENGTH_SPAN, 0.0, 1.0))


def denormalize_length(length_norm: float) -> int:
    return int(round(float(length_norm) * LENGTH_SPAN + MIN_PEPTIDE_LENGTH))


def normalize_charge_per_residue(charge: float, length: int) -> float:
    if length <= 0:
        return 0.5
    return float(np.clip((float(charge) / int(length) + 1.0) / 2.0, 0.0, 1.0))


def normalize_gravy(value: float) -> float:
    return float(np.clip((float(value) + 4.5) / 9.0, 0.0, 1.0))


def _sequence_descriptor_row(sequence: str) -> tuple:
    sequence = str(sequence).upper()
    length = len(sequence)
    if length:
        codes = np.frombuffer(sequence.encode("ascii", "ignore"), dtype=np.uint8)
        gravy_raw = float(_KD_LOOKUP[codes].sum() / length)
    else:
        gravy_raw = 0.0
    charge = sequence.count("K") + sequence.count("R") + 0.1 * sequence.count("H") - sequence.count("D") - sequence.count("E")
    cys_count = sequence.count("C")
    cys_class = 1 if cys_count == 0 else 2 if cys_count <= MAX_CYSTEINES else 3
    return (
        length,
        normalize_length(length),
        charge,
        normalize_charge_per_residue(charge, length),
        min(hydrophobic_moment(sequence), MU_H_CEILING),
        gravy_raw,
        normalize_gravy(gravy_raw),
        cys_count,
        cys_class,
    )


_SEQUENCE_FEATURE_COLUMNS = (
    "length", "length_norm", "net_charge", "charge_per_res", "mu_h",
    "gravy_raw", "gravy", "cys_count", "cys_class_idx",
)


def compute_sequence_features(sequences) -> pd.DataFrame:
    rows = [_sequence_descriptor_row(sequence) for sequence in sequences]
    if not rows:
        return pd.DataFrame(columns=_SEQUENCE_FEATURE_COLUMNS)
    frame = pd.DataFrame(np.asarray(rows, dtype=np.float64), columns=_SEQUENCE_FEATURE_COLUMNS)
    frame["length"] = frame["length"].astype(np.int32)
    frame["cys_count"] = frame["cys_count"].astype(np.int16)
    frame["cys_class_idx"] = frame["cys_class_idx"].astype(np.int8)
    return frame


def attach_conditioning_features(metadata: pd.DataFrame, sequence_column: str = "sequence") -> pd.DataFrame:
    frame = metadata.copy().reset_index(drop=True)
    frame[sequence_column] = frame[sequence_column].map(clean_sequence)
    dense = compute_sequence_features(frame[sequence_column])
    for column in dense.columns:
        frame[column] = dense[column].to_numpy()
    if "species" not in frame:
        frame["species"] = np.nan
    if "gram" not in frame or frame["gram"].isna().all():
        frame["gram"] = frame["species"].map(gram_for_species)
    frame["gram_idx"] = (
        frame["gram"].fillna("unknown").astype(str).str.strip().str.lower().map(GRAM_INDEX).fillna(0).astype(np.int8)
    )
    for column in ("mic_um", "hc50_um", "gram_selectivity_raw", "ic50_um", "cc50_um"):
        if column not in frame:
            frame[column] = np.nan
    frame["log_mic"] = frame["mic_um"].map(normalize_log_concentration)
    frame["log_hc50"] = frame["hc50_um"].map(normalize_log_concentration)
    frame["log_ic50"] = frame["ic50_um"].map(normalize_log_concentration)
    frame["log_cc50"] = frame["cc50_um"].map(normalize_log_concentration)
    frame["gram_selectivity"] = frame["gram_selectivity_raw"].map(normalize_selectivity_ratio)
    if "mic_censored" not in frame:
        frame["mic_censored"] = np.nan
    frame["mic_censored_idx"] = (
        frame["mic_censored"].map({False: 1, True: 2, 0: 1, 1: 2, "exact": 1, "right": 2}).fillna(0).astype(np.int8)
    )
    frame.loc[frame["log_mic"].isna(), "mic_censored_idx"] = 0
    if "is_amp_idx" not in frame:
        frame["is_amp_idx"] = 2
    else:
        frame["is_amp_idx"] = frame["is_amp_idx"].fillna(2).astype(np.int8)
    if "hc50_source" not in frame:
        frame["hc50_source"] = np.nan
    frame["hc50_source_idx"] = frame["hc50_source"].map({"measured": 1, "predicted": 2}).fillna(0).astype(np.int8)
    frame.loc[frame["log_hc50"].isna(), "hc50_source_idx"] = 0
    return frame


def build_species_vocabulary(metadata: pd.DataFrame, min_count: int = 20) -> dict[str, int]:
    counts = metadata["species"].dropna().astype(str).str.strip().str.lower().value_counts()
    kept = sorted(counts[counts >= min_count].index)
    return {name: index + 1 for index, name in enumerate(kept)}


def species_index(metadata: pd.DataFrame, species_vocab: Mapping[str, int]) -> pd.Series:
    if "species" not in metadata:
        return pd.Series(0, index=metadata.index, dtype=np.int64)
    normalized = metadata["species"].astype(str).str.strip().str.lower()
    return normalized.map(species_vocab).fillna(0).astype(np.int64)


WHEEL_PITCHES = (100, 180)
WHEEL_BIN_COUNTS = tuple(360 // math.gcd(pitch, 360) for pitch in WHEEL_PITCHES)
WHEEL_BIN_UNITS = tuple(pitch // math.gcd(pitch, 360) for pitch in WHEEL_PITCHES)
WHEEL_OFFSETS = tuple(1 + sum(WHEEL_BIN_COUNTS[:index]) for index in range(len(WHEEL_PITCHES)))
WHEEL_FORWARD = 1 + sum(WHEEL_BIN_COUNTS)
WHEEL_REVERSE = WHEEL_FORWARD + 1
WHEEL_SIZE = WHEEL_REVERSE + 1


def wheel_index(pitch: int, bin_index: int) -> int:
    position = WHEEL_PITCHES.index(int(pitch))
    if not 0 <= bin_index < WHEEL_BIN_COUNTS[position]:
        raise ValueError(f"bin_index must be in [0, {WHEEL_BIN_COUNTS[position]})")
    return WHEEL_OFFSETS[position] + int(bin_index)


def hydrophobic_moment_face(sequence: str, delta_deg: float, window: int = 11) -> tuple[float, float]:
    sequence = str(sequence).upper()
    if not sequence:
        return 0.0, 0.0
    codes = np.frombuffer(sequence.encode("ascii", "ignore"), dtype=np.uint8)
    values = _EISENBERG_LOOKUP[codes].astype(np.float64)
    width = min(int(window), values.size)
    if width <= 0:
        return 0.0, 0.0
    angles = np.arange(width, dtype=np.float64) * np.deg2rad(delta_deg)
    windows = sliding_window_view(values, width)
    real = windows @ np.cos(angles)
    imag = windows @ np.sin(angles)
    magnitude = np.hypot(real, imag)
    start = int(magnitude.argmax())
    theta = (np.degrees(np.arctan2(imag[start], real[start])) + start * delta_deg) % 360.0
    return float(magnitude[start] / width), float(theta)


def wheel_class(sequence: str, thresholds: Sequence[float]) -> int:
    best_ratio = 0.0
    best_class = WHEEL_FORWARD
    for index, pitch in enumerate(WHEEL_PITCHES):
        magnitude, theta = hydrophobic_moment_face(sequence, float(pitch))
        ratio = magnitude / float(thresholds[index])
        if ratio >= 1.0 and ratio > best_ratio:
            bins = WHEEL_BIN_COUNTS[index]
            center = int(round(theta / (360.0 / bins))) % bins
            best_ratio = ratio
            best_class = WHEEL_OFFSETS[index] + center
    return best_class


def assign_wheel_classes(sequences, thresholds: Sequence[float]) -> np.ndarray:
    return np.asarray([wheel_class(sequence, thresholds) for sequence in sequences], dtype=np.int64)


def fit_wheel_thresholds(sequences, quantile: float = 0.9) -> tuple[float, ...]:
    sequences = [clean_sequence(sequence) for sequence in sequences]
    if not sequences:
        raise ValueError("no sequences to fit wheel thresholds")
    thresholds = []
    for pitch in WHEEL_PITCHES:
        values = np.fromiter(
            (hydrophobic_moment_face(sequence, float(pitch))[0] for sequence in sequences),
            dtype=np.float64,
            count=len(sequences),
        )
        thresholds.append(float(np.quantile(values, quantile)))
    return tuple(thresholds)


def fit_wheel_prior(classes, mu_h, threshold: float) -> np.ndarray:
    classes = np.asarray(classes, dtype=np.int64)
    mu_h = np.asarray(mu_h, dtype=np.float64)
    if classes.size and (classes.min() < 0 or classes.max() >= WHEEL_SIZE):
        raise ValueError("wheel class out of range")
    observed = np.isfinite(mu_h)
    groups = (observed & (mu_h < threshold), observed & (mu_h >= threshold), np.ones(classes.shape[0], dtype=bool))
    table = np.zeros((3, WHEEL_SIZE), dtype=np.float64)
    for row, members in enumerate(groups):
        counts = np.bincount(classes[members], minlength=WHEEL_SIZE).astype(np.float64)
        if counts[0]:
            raise ValueError("wheel class 0 present in the training rows")
        directional = counts[WHEEL_FORWARD] + counts[WHEEL_REVERSE]
        counts[WHEEL_FORWARD] = 0.5 * directional
        counts[WHEEL_REVERSE] = 0.5 * directional
        if counts.sum() == 0:
            raise ValueError(f"no training rows for wheel prior row {row}")
        table[row] = counts / counts.sum()
    return table


LEGACY_CATEGORICAL_AXES = ("is_amp", "species", "gram", "cys_class", "mic_censored", "hc50_source", "wheel")
TASK_AXES = tuple(f"act_{task}" for task in TASKS)
CATEGORICAL_AXES = LEGACY_CATEGORICAL_AXES + TASK_AXES
DROPOUT_EXEMPT_AXES = ("is_amp",)
PROPERTY_AXES = ("charge_per_res", "mu_h", "gravy", "cys_class")
MEASUREMENT_AXES = (
    "log_mic", "log_hc50", "log_ic50", "log_cc50", "gram_selectivity",
    "species", "gram", "mic_censored", "hc50_source",
)
LEGACY_CONTINUOUS_AXES = ("log_mic", "log_hc50", "length_norm", "charge_per_res", "mu_h", "gravy")
CONTINUOUS_AXES = (
    "log_mic", "log_hc50", "length_norm", "charge_per_res", "mu_h", "gravy",
    "gram_selectivity", "log_ic50", "log_cc50",
)

# Continuous columns that older metadata tables do not carry; they are filled as unobserved.
OPTIONAL_CONTINUOUS_AXES = ("log_ic50", "log_cc50")

CATEGORICAL_SIZES = {
    "is_amp": 3, "gram": 5, "cys_class": 4, "mic_censored": 3, "hc50_source": 3,
    "wheel": WHEEL_SIZE, **{axis: 3 for axis in TASK_AXES},
}

DENSE_AXES = ("length_norm", "charge_per_res", "mu_h", "gravy", "cys_class")

CONDITION_KEYS = (
    tuple(f"{axis}_idx" for axis in CATEGORICAL_AXES)
    + CONTINUOUS_AXES
    + tuple(f"{axis}_mask" for axis in CONTINUOUS_AXES)
)

CATEGORICAL_COLUMNS = {axis: f"{axis}_idx" for axis in CATEGORICAL_AXES}
CONTINUOUS_COLUMNS = {axis: axis for axis in CONTINUOUS_AXES}

IS_AMP_VALUES = {"unknown": 0, "general": 1, "amp": 2}
HC50_SOURCE_VALUES = {"unknown": 0, "measured": 1, "predicted": 2}


def categorical_sizes(num_species: int, axes: Optional[Sequence[str]] = None) -> Dict[str, int]:
    sizes = {"species": int(num_species)}
    sizes.update(CATEGORICAL_SIZES)
    selected = CATEGORICAL_AXES if axes is None else tuple(axes)
    return {axis: sizes[axis] for axis in selected}


def _batch_vector(value, batch_size: int, device, dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(f"condition has {tensor.numel()} rows, expected {batch_size}")
    return tensor


def normalize_condition(condition: Mapping[str, torch.Tensor], batch_size: int, device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for axis in CATEGORICAL_AXES:
        key = f"{axis}_idx"
        value = condition.get(key)
        if value is None:
            out[key] = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            out[key] = _batch_vector(value, batch_size, device, torch.long)
    for axis in CONTINUOUS_AXES:
        value = condition.get(axis)
        mask = condition.get(f"{axis}_mask")
        if value is None:
            out[axis] = torch.zeros(batch_size, 1, device=device)
            out[f"{axis}_mask"] = torch.zeros(batch_size, 1, device=device)
            continue
        value_tensor = _batch_vector(value, batch_size, device, torch.float32).view(batch_size, 1)
        if mask is None:
            mask_tensor = torch.ones_like(value_tensor)
        else:
            mask_tensor = _batch_vector(mask, batch_size, device, torch.float32).view(batch_size, 1)
        out[axis] = value_tensor.contiguous()
        out[f"{axis}_mask"] = mask_tensor.contiguous()
    return out


def drop_condition_axes(
    condition: Mapping[str, torch.Tensor],
    p_axis: float = 0.15,
    p_all: float = 0.10,
    p_tasks: float = 0.0,
    p_properties: float = 0.0,
    p_measurements: float = 0.0,
) -> Dict[str, torch.Tensor]:
    for name, probability in (
        ("p_axis", p_axis), ("p_all", p_all), ("p_tasks", p_tasks),
        ("p_properties", p_properties), ("p_measurements", p_measurements),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    categorical = [axis for axis in CATEGORICAL_AXES if f"{axis}_idx" in condition]
    continuous = [axis for axis in CONTINUOUS_AXES if axis in condition]
    reference = condition[f"{categorical[0]}_idx"]
    batch_size = reference.shape[0]
    device = reference.device
    keep_all = torch.rand(batch_size, 1, device=device) >= p_all
    keep_tasks = torch.rand(batch_size, 1, device=device) >= p_tasks
    keep_properties = torch.rand(batch_size, 1, device=device) >= p_properties
    keep_measurements = torch.rand(batch_size, 1, device=device) >= p_measurements
    droppable_categorical = [axis for axis in categorical if axis not in DROPOUT_EXEMPT_AXES]
    categorical_keep = torch.rand(batch_size, len(droppable_categorical), device=device) >= p_axis
    categorical_keep &= keep_all
    continuous_keep = torch.rand(batch_size, len(continuous), device=device) >= p_axis
    continuous_keep &= keep_all
    out: Dict[str, torch.Tensor] = {}
    drop_index = 0
    for axis in categorical:
        key = f"{axis}_idx"
        values = condition[key].long().reshape(-1)
        if axis in DROPOUT_EXEMPT_AXES:
            out[key] = values
            continue
        keep = categorical_keep[:, drop_index]
        if axis in TASK_AXES:
            keep = keep & keep_tasks[:, 0]
        if axis in PROPERTY_AXES:
            keep = keep & keep_properties[:, 0]
        if axis in MEASUREMENT_AXES:
            keep = keep & keep_measurements[:, 0]
        out[key] = values * keep.long()
        drop_index += 1
    for index, axis in enumerate(continuous):
        value = condition[axis].float().reshape(batch_size, 1)
        mask = condition.get(f"{axis}_mask")
        if mask is None:
            mask = torch.ones_like(value)
        else:
            mask = mask.float().reshape(batch_size, 1)
        keep = continuous_keep[:, index : index + 1]
        if axis in PROPERTY_AXES:
            keep = keep & keep_properties
        if axis in MEASUREMENT_AXES:
            keep = keep & keep_measurements
        out[axis] = value
        out[f"{axis}_mask"] = mask * keep
    return out


def concat_conditions(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: torch.cat((first[key], second[key]), dim=0) for key in first}


def select_condition(condition: Mapping[str, torch.Tensor], start: int, stop: int) -> Dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in condition.items()}


def condition_from_batch(batch: Mapping[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {key: batch[key].to(device) for key in CONDITION_KEYS if key in batch}


def prepare_metadata(metadata: pd.DataFrame, species_vocab: Mapping[str, int], continuous_axes=None) -> pd.DataFrame:
    frame = metadata.copy()
    frame["species_idx"] = species_index(frame, species_vocab)
    if "wheel_idx" not in frame:
        frame["wheel_idx"] = 0
    axes = tuple(CONTINUOUS_AXES if continuous_axes is None else continuous_axes)
    for axis, column in CATEGORICAL_COLUMNS.items():
        if column not in frame:
            if axis not in TASK_AXES:
                raise KeyError(f"missing categorical column {column!r} for {axis!r}")
            frame[column] = 0
        frame[column] = frame[column].fillna(0).astype(np.int64)
    for axis in axes:
        if axis not in frame:
            if axis not in OPTIONAL_CONTINUOUS_AXES:
                raise KeyError(f"missing continuous column {axis!r}")
            frame[axis] = np.nan
        frame[axis] = pd.to_numeric(frame[axis], errors="coerce")
    return frame.reset_index(drop=True)


def condition_tensors(metadata: pd.DataFrame, continuous_axes=None) -> Dict[str, torch.Tensor]:
    axes = tuple(CONTINUOUS_AXES if continuous_axes is None else continuous_axes)
    out: Dict[str, torch.Tensor] = {}
    for axis, column in CATEGORICAL_COLUMNS.items():
        out[f"{axis}_idx"] = torch.as_tensor(metadata[column].to_numpy(dtype=np.int64), dtype=torch.long)
    for axis in axes:
        values = metadata[axis].to_numpy(dtype=np.float32)
        observed = np.isfinite(values)
        out[axis] = torch.from_numpy(np.nan_to_num(values, nan=0.0).reshape(-1, 1))
        out[f"{axis}_mask"] = torch.from_numpy(observed.astype(np.float32).reshape(-1, 1))
    return out


class AMPDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame, species_vocab: Mapping[str, int], continuous_axes=None):
        if len(embeddings) != len(metadata):
            raise ValueError(f"embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch")
        self.continuous_axes = tuple(CONTINUOUS_AXES if continuous_axes is None else continuous_axes)
        self.embeddings = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
        self.metadata = prepare_metadata(metadata, species_vocab, self.continuous_axes)
        self.condition = condition_tensors(self.metadata, self.continuous_axes)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {"embedding": self.embeddings[index]}
        for axis in CATEGORICAL_AXES:
            key = f"{axis}_idx"
            item[key] = self.condition[key][index]
        for axis in self.continuous_axes:
            item[axis] = self.condition[axis][index]
            item[f"{axis}_mask"] = self.condition[f"{axis}_mask"][index]
        return item


class DecoderDataset(AMPDataset):
    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame, species_vocab: Mapping[str, int], decoder, sequence_column: str = "sequence"):
        super().__init__(embeddings, metadata, species_vocab)
        token_length = decoder.max_peptide_length + 2
        self.tokens = torch.tensor(
            [decoder.encode_tokens(str(sequence), max_length=token_length) for sequence in self.metadata[sequence_column]],
            dtype=torch.long,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        item["tokens"] = self.tokens[index]
        return item


class TokenDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        species_vocab: Mapping[str, int],
        sequence_column: str = "sequence",
        canvas: Optional[int] = None,
        continuous_axes=None,
        wheel_thresholds: Optional[Sequence[float]] = None,
    ):
        from .masked_diffusion import encode_sequence

        self.canvas = int(canvas or MAX_PEPTIDE_LENGTH)
        self.continuous_axes = tuple(CONTINUOUS_AXES if continuous_axes is None else continuous_axes)
        self.metadata = prepare_metadata(metadata, species_vocab, self.continuous_axes)
        if wheel_thresholds is not None:
            self.metadata["wheel_idx"] = assign_wheel_classes(
                self.metadata[sequence_column].map(clean_sequence), wheel_thresholds
            )
        self.condition = condition_tensors(self.metadata, self.continuous_axes)
        self.tokens = torch.tensor(
            [encode_sequence(str(sequence), canvas=self.canvas) for sequence in self.metadata[sequence_column]],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {"tokens": self.tokens[index]}
        for axis in CATEGORICAL_AXES:
            key = f"{axis}_idx"
            item[key] = self.condition[key][index]
        for axis in self.continuous_axes:
            item[axis] = self.condition[axis][index]
            item[f"{axis}_mask"] = self.condition[f"{axis}_mask"][index]
        return item


def cluster_split(
    metadata: pd.DataFrame,
    splits: Optional[pd.DataFrame] = None,
    sequence_column: str = "sequence",
    cluster_column: str = "cluster",
    fold_column: str = "fold",
    holdout_fold: int = 0,
    random_seed: int = 42,
    split_column: str = "split",
) -> tuple[np.ndarray, np.ndarray]:
    del cluster_column
    frame = metadata.reset_index(drop=True)
    if splits is not None and sequence_column in splits and split_column in splits:
        # A split table with train, val and test labels takes precedence over folds. Test rows
        # belong to neither returned index; rows absent from the table (pretraining background
        # peptides) are training rows.
        keyed = frame[[sequence_column]].merge(
            splits[[sequence_column, split_column]].drop_duplicates(subset=[sequence_column]),
            on=sequence_column,
            how="left",
        )
        assignment = keyed[split_column].fillna("train").to_numpy()
        if (assignment == "val").any():
            return np.flatnonzero(assignment == "train"), np.flatnonzero(assignment == "val")
    if splits is not None and sequence_column in splits and fold_column in splits:
        keyed = frame[[sequence_column]].merge(
            splits[[sequence_column, fold_column]].drop_duplicates(subset=[sequence_column]),
            on=sequence_column,
            how="left",
        )
        folds = keyed[fold_column]
        if folds.notna().any():
            is_val = (folds == holdout_fold).fillna(False).to_numpy()
            return np.flatnonzero(~is_val), np.flatnonzero(is_val)
    order = np.random.default_rng(random_seed).permutation(len(frame))
    cut = int(0.9 * len(order))
    return order[:cut], order[cut:]


def stratified_sampler_weights(metadata: pd.DataFrame) -> np.ndarray:
    has_mic = metadata["log_mic"].notna().to_numpy()
    species = metadata["species_idx"].to_numpy(dtype=np.int64)
    labels = np.where(has_mic, species + 1, 0)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float64)


def task_sampler_weights(metadata: pd.DataFrame, power: float = 0.5) -> np.ndarray:
    # Each row is assigned to the rarest (task, state) cell it belongs to, active or inactive,
    # or to a background bucket when it has no task label. A bucket receives sampling mass
    # proportional to size ** power; power 1 keeps the natural mix, power 0 draws all buckets
    # equally.
    states = np.stack(
        [
            metadata[f"{axis}_idx"].to_numpy(dtype=np.int64)
            if f"{axis}_idx" in metadata
            else np.zeros(len(metadata), dtype=np.int64)
            for axis in TASK_AXES
        ],
        axis=1,
    )
    cells = np.concatenate((states == 2, states == 1), axis=1)
    per_cell = cells.sum(axis=0).astype(np.float64)
    rarity = np.where(cells, per_cell[None, :], np.inf)
    bucket = np.where(cells.any(axis=1), rarity.argmin(axis=1), cells.shape[1])
    bucket_size = np.bincount(bucket, minlength=cells.shape[1] + 1).astype(np.float64)
    return bucket_size[bucket] ** (float(power) - 1.0)


def task_state(value) -> int:
    if isinstance(value, bool):
        return 2 if value else 1
    if isinstance(value, (int, float)):
        state = int(value)
        if state not in (0, 1, 2):
            raise ValueError("task state must be 0, 1 or 2")
        return state
    key = str(value).strip().lower()
    if key not in TASK_STATE_VALUES:
        raise ValueError(f"task state must be one of {sorted(TASK_STATE_VALUES)}, got {value!r}")
    return TASK_STATE_VALUES[key]


def is_amp_sampler_weights(metadata: pd.DataFrame, negative_ratio: float = 3.0, positive_ratio: float = 1.0) -> np.ndarray:
    is_amp = metadata["is_amp_idx"].to_numpy(dtype=np.int64)
    weights = np.zeros(len(is_amp), dtype=np.float64)
    positive = is_amp == 2
    negative = is_amp == 1
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive:
        weights[positive] = positive_ratio / n_positive
    if n_negative:
        weights[negative] = negative_ratio / n_negative
    return weights


def condition_values(request: Mapping, species_vocab: Mapping[str, int]) -> dict:
    values = {f"{axis}_idx": 0 for axis in CATEGORICAL_AXES}
    values.update({axis: None for axis in CONTINUOUS_AXES})
    raw_is_amp = request.get("is_amp")
    if isinstance(raw_is_amp, (int, float)):
        values["is_amp_idx"] = int(raw_is_amp)
    elif raw_is_amp is not None:
        values["is_amp_idx"] = IS_AMP_VALUES.get(str(raw_is_amp).strip().lower(), 0)
    species = request.get("species")
    if species is not None:
        species_key = str(species).strip().lower()
        values["species_idx"] = species_vocab.get(species_key, 0)
        gram = request.get("gram") or gram_for_species(species_key)
    else:
        gram = request.get("gram", "unknown")
    values["gram_idx"] = GRAM_INDEX.get(str(gram).strip().lower(), 0)
    cys_class = request.get("cys_class")
    if cys_class is not None:
        values["cys_class_idx"] = int(cys_class)
    wheel = request.get("wheel")
    if wheel is not None:
        wheel = int(wheel)
        if not 0 <= wheel < WHEEL_SIZE:
            raise ValueError(f"wheel must be in [0, {WHEEL_SIZE})")
        values["wheel_idx"] = wheel
    mic_um = request.get("mic_um")
    if mic_um is not None:
        values["log_mic"] = normalize_log_concentration(mic_um)
        values["mic_censored_idx"] = 2 if bool(request.get("mic_censored", False)) else 1
    hc50_um = request.get("hc50_um")
    if hc50_um is not None:
        values["log_hc50"] = normalize_log_concentration(hc50_um)
        source = request.get("hc50_source")
        if isinstance(source, (int, float)):
            values["hc50_source_idx"] = int(source)
        elif source is not None:
            values["hc50_source_idx"] = HC50_SOURCE_VALUES.get(str(source).strip().lower(), 0)
    selectivity = request.get("gram_selectivity")
    if selectivity is not None:
        values["gram_selectivity"] = normalize_selectivity_ratio(selectivity)
    ic50_um = request.get("ic50_um")
    if ic50_um is not None:
        values["log_ic50"] = normalize_log_concentration(ic50_um)
    cc50_um = request.get("cc50_um")
    if cc50_um is not None:
        values["log_cc50"] = normalize_log_concentration(cc50_um)
    tasks = request.get("tasks")
    if tasks:
        for task, state in dict(tasks).items():
            if task not in TASKS:
                raise ValueError(f"unknown task {task!r}; tasks are {TASKS}")
            values[f"act_{task}_idx"] = task_state(state)
    length = request.get("length")
    values["length_norm"] = normalize_length(length) if length is not None else 0.5
    charge_per_res = request.get("charge_per_res")
    if charge_per_res is None and request.get("net_charge") is not None and length is not None:
        charge_per_res = normalize_charge_per_residue(float(request["net_charge"]), int(length))
    if charge_per_res is not None:
        values["charge_per_res"] = float(charge_per_res)
    mu_h = request.get("mu_h")
    if mu_h is not None:
        values["mu_h"] = float(mu_h)
    raw_gravy = request.get("gravy")
    if raw_gravy is not None:
        values["gravy"] = normalize_gravy(raw_gravy)
    return values


def build_condition(request: Mapping, batch_size: int, species_vocab: Mapping[str, int], device: str = "cuda") -> Dict[str, torch.Tensor]:
    values = condition_values(request, species_vocab)
    condition: Dict[str, torch.Tensor] = {}
    for axis in CATEGORICAL_AXES:
        key = f"{axis}_idx"
        condition[key] = torch.full((batch_size,), int(values[key]), dtype=torch.long, device=device)
    for axis in CONTINUOUS_AXES:
        value = values[axis]
        observed = value is not None and np.isfinite(float(value))
        condition[axis] = torch.full((batch_size, 1), float(value) if observed else 0.0, dtype=torch.float32, device=device)
        condition[f"{axis}_mask"] = torch.full((batch_size, 1), 1.0 if observed else 0.0, dtype=torch.float32, device=device)
    return condition


def length_grid(request: Mapping, counts: Mapping[int, int], species_vocab: Mapping[str, int], device: str = "cuda") -> Dict[str, torch.Tensor]:
    blocks = []
    for length, count in counts.items():
        if count <= 0:
            continue
        local = dict(request)
        local["length"] = int(np.clip(length, MIN_PEPTIDE_LENGTH, MAX_PEPTIDE_LENGTH))
        blocks.append(build_condition(local, count, species_vocab, device=device))
    if not blocks:
        raise ValueError("length_grid received no positive counts")
    return {key: torch.cat([block[key] for block in blocks], dim=0) for key in blocks[0]}


def empirical_length_counts(lengths, total: int, min_length: int = MIN_PEPTIDE_LENGTH, max_length: int = MAX_PEPTIDE_LENGTH) -> Dict[int, int]:
    observed = np.asarray(list(lengths), dtype=np.int64)
    observed = observed[(observed >= min_length) & (observed <= max_length)]
    if not observed.size:
        raise ValueError("no lengths inside the requested window")
    values, counts = np.unique(observed, return_counts=True)
    exact = total * counts / counts.sum()
    allocation = np.floor(exact).astype(np.int64)
    remainder = total - int(allocation.sum())
    if remainder:
        order = np.argsort(-(exact - allocation), kind="stable")
        allocation[order[:remainder]] += 1
    return {int(value): int(count) for value, count in zip(values, allocation) if count}


def candidate_frame(sequences) -> pd.DataFrame:
    unique = list(dict.fromkeys(str(sequence).upper().strip() for sequence in sequences if str(sequence).strip()))
    frame = pd.DataFrame({"sequence": unique})
    if frame.empty:
        return frame
    features = compute_sequence_features(frame["sequence"])
    return pd.concat([frame.reset_index(drop=True), features], axis=1)