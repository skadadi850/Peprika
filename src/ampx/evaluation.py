"""
Shared evaluation code for the unified corpus: metrics, feature matrices, evidence-matched subsets,
similarity search, the corpus frame, the ESM-2 embedding cache, boosted-tree fitting, task oracles and
the bootstrap over homology components. Used by scripts/benchmark.py and scripts/evaluate_generator.py.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy.stats import rankdata

from .models.data import STANDARD_AA, TASKS, compute_sequence_features
from .predictor import ESM2Encoder

BENCHMARK_TASKS = (
    "antibacterial",
    "antifungal",
    "anticancer",
    "antiviral",
    "antiinflammatory",
    "hemolytic",
)

# Evidence-matched subsets, fixed before any benchmark result existed. Evidence code 2 or 4 means the
# label comes from a measurement under the task rule; 1 is a positive-only annotation and 3 is an
# explicit curated label. Tasks without a rule use all labels.
SUBSETS = {
    "antibacterial": ("evidence", (2, 4)),
    "antifungal": ("evidence", (2, 4)),
    "anticancer": ("source", "manual:anticp2_main"),
    "antiviral": ("source", "manual:avppred"),
}

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "seed": 0,
}
MAX_ROUNDS = 600
EARLY_STOPPING_ROUNDS = 30
FIXED_ROUNDS = 100

PROBE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "learning_rate": 0.1,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 0,
}
PROBE_MAX_ROUNDS = 400

MIN_VALID_DRAWS = 50


def auroc(target: np.ndarray, scores: np.ndarray) -> float:
    positive = target == 1
    n_positive = int(positive.sum())
    n_negative = len(target) - n_positive

    if n_positive == 0 or n_negative == 0:
        return float("nan")

    ranks = rankdata(scores)

    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def average_precision(target: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    hits = target[order] == 1

    if not hits.any():
        return float("nan")

    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)

    return float(precision[hits].mean())


def balanced_accuracy(target: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    positive = target == 1

    true_positive_rate = (predicted & positive).sum() / max(int(positive.sum()), 1)
    true_negative_rate = (~predicted & ~positive).sum() / max(int((~positive).sum()), 1)

    return float(0.5 * (true_positive_rate + true_negative_rate))


def best_threshold(target: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    sorted_target = target[order]
    sorted_scores = scores[order]

    positives = np.cumsum(sorted_target == 1)
    negatives = np.cumsum(sorted_target == 0)

    true_positive_rate = positives / positives[-1]
    true_negative_rate = (negatives[-1] - negatives) / negatives[-1]
    value = 0.5 * (true_positive_rate + true_negative_rate)

    distinct = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    value = np.where(distinct, value, -1.0)

    return float(sorted_scores[int(np.argmax(value))])


def percentile_interval(values) -> tuple[float, float]:
    if len(values) < MIN_VALID_DRAWS:
        return float("nan"), float("nan")

    low, high = np.percentile(values, [2.5, 97.5])

    return float(low), float(high)


def component_groups(components: np.ndarray) -> list[np.ndarray]:
    unique, inverse = np.unique(components, return_inverse=True)

    return [np.flatnonzero(inverse == index) for index in range(len(unique))]


def resample_components(groups: list[np.ndarray], generator: np.random.Generator) -> np.ndarray:
    chosen = generator.integers(0, len(groups), len(groups))

    return np.concatenate([groups[value] for value in chosen])


def subset_mask(frame: pd.DataFrame, task: str) -> np.ndarray:
    rule = SUBSETS.get(task)

    if rule is None:
        return np.ones(len(frame), dtype=bool)

    kind, value = rule

    if kind == "evidence":
        return frame[f"evidence_{task}"].isin(value).to_numpy()

    return frame["sources"].str.split(";").map(lambda names: value in names).to_numpy()


def task_selection(frame: pd.DataFrame, task: str, matched: bool):
    state = frame[f"act_{task}_idx"].to_numpy()
    selected = state > 0

    if matched:
        selected = selected & subset_mask(frame, task)

    keep = np.flatnonzero(selected)
    target = (state[keep] == 2).astype(np.int64)

    return keep, target, frame["split"].to_numpy()[keep]


def split_indices(split: np.ndarray, target: np.ndarray):
    parts = tuple(np.flatnonzero(split == name) for name in ("train", "val", "test"))

    if any(len(set(target[part])) < 2 for part in parts):
        return None

    return parts


def composition_matrix(sequences: pd.Series) -> np.ndarray:
    index = {residue: column for column, residue in enumerate(STANDARD_AA)}
    matrix = np.zeros((len(sequences), len(STANDARD_AA) + 1), dtype=np.float32)

    for row, sequence in enumerate(sequences):
        for residue in sequence:
            matrix[row, index[residue]] += 1.0

        matrix[row, :-1] /= len(sequence)
        matrix[row, -1] = len(sequence)

    return matrix


def dipeptide_matrix(sequences: pd.Series) -> np.ndarray:
    index = {a + b: column for column, (a, b) in enumerate(product(STANDARD_AA, repeat=2))}
    matrix = np.zeros((len(sequences), len(index)), dtype=np.float32)

    for row, sequence in enumerate(sequences):
        for position in range(len(sequence) - 1):
            matrix[row, index[sequence[position : position + 2]]] += 1.0

        matrix[row] /= max(len(sequence) - 1, 1)

    return matrix


def property_matrix(sequences: pd.Series) -> np.ndarray:
    features = compute_sequence_features(sequences.tolist())
    length = features["length"].to_numpy(dtype=np.float32)
    charge = features["net_charge"].to_numpy(dtype=np.float32)

    return np.stack(
        [
            length,
            charge,
            charge / length,
            features["mu_h"].to_numpy(dtype=np.float32),
            features["gravy_raw"].to_numpy(dtype=np.float32),
            features["cys_count"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )


def source_matrix(sources: pd.Series) -> np.ndarray:
    names = sorted({name for value in sources for name in value.split(";")})
    index = {name: column for column, name in enumerate(names)}
    matrix = np.zeros((len(sources), len(names)), dtype=np.float32)

    for row, value in enumerate(sources):
        for name in value.split(";"):
            matrix[row, index[name]] = 1.0

    return matrix


def other_task_matrix(frame: pd.DataFrame, task: str) -> np.ndarray:
    blocks = []

    for other in TASKS:
        if other == task:
            continue

        state = frame[f"act_{other}_idx"].to_numpy()
        blocks.append(np.stack([state == 1, state == 2], axis=1).astype(np.float32))

    return np.concatenate(blocks, axis=1)


def combined_features(sequences: pd.Series, embeddings: np.ndarray) -> np.ndarray:
    return np.hstack([embeddings, composition_matrix(sequences), dipeptide_matrix(sequences)])


def feature_sets(frame: pd.DataFrame, embeddings: np.ndarray) -> dict[str, np.ndarray]:
    composition = composition_matrix(frame["sequence"])
    dipeptide = dipeptide_matrix(frame["sequence"])
    length = frame["sequence"].str.len().to_numpy(dtype=np.float32)[:, None]

    return {
        "length": length,
        "composition": composition,
        "dipeptide": dipeptide,
        "esm2": embeddings,
        "combined": np.hstack([embeddings, composition, dipeptide]),
    }


def sequence_features(sequences: list[str], encoder: ESM2Encoder) -> tuple[np.ndarray, np.ndarray]:
    embeddings = encoder.encode(sequences, batch_size=256)

    return embeddings, combined_features(pd.Series(sequences), embeddings)


def max_similarity(queries: list[str], references: list[str], block: int = 500) -> np.ndarray:
    maxima = np.zeros(len(queries), dtype=np.float32)

    for start in range(0, len(queries), block):
        scores = process.cdist(
            queries[start : start + block],
            references,
            scorer=Levenshtein.normalized_similarity,
            dtype=np.float32,
            workers=-1,
        )
        maxima[start : start + block] = scores.max(axis=1)

    return maxima


def load_frame(corpus: Path, split_columns=("sequence", "split")) -> pd.DataFrame:
    return (
        pd.read_csv(corpus / "labels.csv")
        .merge(pd.read_csv(corpus / "sequences.csv"), on="sequence")
        .merge(pd.read_csv(corpus / "splits.csv", usecols=list(split_columns)), on="sequence")
        .reset_index(drop=True)
    )


def corpus_embeddings(
    sequences: list[str], cache: Path, device: str, encoder: ESM2Encoder | None = None
) -> np.ndarray:
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)

        if stored["sequences"].tolist() == sequences:
            return stored["embeddings"]

    encoder = encoder or ESM2Encoder(device=device)
    embeddings = encoder.encode(sequences, batch_size=256)
    np.savez(cache, sequences=np.array(sequences), embeddings=embeddings)

    return embeddings


def esmc_lookup(corpus_sequences, corpus_embeddings: np.ndarray, cache_path: Path):
    stored = np.load(cache_path, allow_pickle=False)
    names = list(corpus_sequences) + stored["sequences"].tolist()
    matrix = np.concatenate([corpus_embeddings, stored["embeddings"]])

    index: dict[str, int] = {}

    for position, sequence in enumerate(names):
        index.setdefault(sequence, position)

    def lookup(sequences) -> np.ndarray:
        missing = [sequence for sequence in sequences if sequence not in index]

        if missing:
            raise SystemExit(
                f"{len(missing)} sequences are missing from the ESM C cache, for example "
                f"{missing[0]}; run the cache step first"
            )

        return matrix[[index[sequence] for sequence in sequences]]

    return lookup


class BoostedModel:
    def __init__(self, features, target, train, val, params=PARAMS, max_rounds=MAX_ROUNDS):
        if len(set(target[val].tolist())) < 2:
            self.booster = xgb.train(
                params,
                xgb.DMatrix(features[train], label=target[train]),
                num_boost_round=FIXED_ROUNDS,
                verbose_eval=False,
            )
            self.rounds = FIXED_ROUNDS
            return

        self.booster = xgb.train(
            params,
            xgb.DMatrix(features[train], label=target[train]),
            num_boost_round=max_rounds,
            evals=[(xgb.DMatrix(features[val], label=target[val]), "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        self.rounds = int(self.booster.best_iteration) + 1

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(features), iteration_range=(0, self.rounds))


def fit_oracles(frame: pd.DataFrame, features: np.ndarray, tasks, matched: bool = True):
    oracles = {}
    rows = []

    for task in tasks:
        keep, target, split = task_selection(frame, task, matched)
        parts = split_indices(split, target)

        if parts is None:
            print(f"{task}: oracle skipped, a split lacks one of the two classes", flush=True)
            continue

        train, val, test = parts
        selected = features[keep]

        oracle = BoostedModel(selected, target, train, val)
        scores = oracle.score(selected[test])

        oracles[task] = oracle
        rows.append(
            {
                "task": task,
                "n_train": len(train),
                "test_auroc": auroc(target[test], scores),
                "test_p_active_for_active": float(scores[target[test] == 1].mean()),
                "test_p_active_for_inactive": float(scores[target[test] == 0].mean()),
            }
        )
        print(rows[-1], flush=True)

    return oracles, pd.DataFrame(rows)
