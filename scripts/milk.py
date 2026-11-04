"""
Zero-shot antibacterial candidates among human milk peptides, scored by two models and reported in one figure.

Source: Table S2 (sheet 2) of the supplement of the preterm infant milk and gastric peptidomics paper
(pr8b00604_si_002.xlsx). The only functional annotation in the sheet is "Homologous (%)", the percent match of a
peptide to a known bioactive peptide of the Milk Bioactive Peptide Database; a blank means no such match. A
homology-labeled human peptide is called antimicrobial when it lies in one of the two regions the paper assigns to
antimicrobial homologs (beta-casein f(179-211) and lactoferrin f(269-288) in mature numbering, 194-226 and 288-307
in the precursor numbering of Table S2).

Two scorers rank the same peptides for antibacterial and for hemolytic activity:

    oracle     boosted trees on ESM-2 embeddings with composition and dipeptide fractions, fitted on the matched
               labels of the training split (the oracle of the generator audit).
    ensemble   rank average of the Multi-task network and the boosted ESM-C model with composition and dipeptide
               fractions (the Ensemble of the benchmark). Both are fitted on the matched labels of the training
               split and stopped on the validation split; the scores of --seeds networks are averaged.

Stages. Each reads what the previous one wrote.

    filter     Fixed before any scoring: species Human, modifications blank, 8 to 50 residues of the 20 standard
               amino acids, deduplicated, maximum similarity below 0.5 to every corpus peptide. Writes
               results/milk/pool.csv (before the similarity filter), primary.csv and funnel.csv.
    score      Fits both scorers and scores the primary set. Writes scores.csv.
    select     Top sets of the two scorers, their overlap with intervals that resample components, the shortlist
               for wet-lab testing with its properties, and the agreement with the known antimicrobial homologs.
               Writes shortlist.csv, top_sets.csv and overlap.csv.
    figure     Draws paper/figures/fig-milk.png.
    all        Every stage in order.

The shortlist holds peptides in the top --top share of antibacterial score under both scorers and outside the top
--hemolytic-top share of hemolytic score under both, one peptide per homology component, ranked by the mean
antibacterial percentile. The unit for every interval is the homology component at --identity. Every result is a
model prediction for peptides without activity measurements.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/milk.py all
    PYTHONUNBUFFERED=1 uv run python scripts/milk.py filter
    PYTHONUNBUFFERED=1 uv run python scripts/milk.py score
    PYTHONUNBUFFERED=1 uv run python scripts/milk.py select
    PYTHONUNBUFFERED=1 uv run python scripts/milk.py figure

PYTHONUNBUFFERED=1 uv run python scripts/milk.py activity_figure > logs/milk_activity_figure.log 2>&1
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MILK = ROOT / "results" / "milk"

STANDARD = "ACDEFGHIKLMNPQRSTVWY"
MIN_LENGTH = 8
MAX_LENGTH = 50
SIMILARITY_CEILING = 0.5
HOMOLOGY = "Homologous (%)"
PILOT_COMPONENTS = 30

TARGET = "antibacterial"
HEMOLYTIC = "hemolytic"
SCORED_TASKS = ("antibacterial", "antifungal", "anticancer", "antiinflammatory", "hemolytic")
ACTIVITY_LABELS = {
    "antibacterial": "Antibacterial",
    "antifungal": "Antifungal",
    "anticancer": "Anticancer",
    "antiinflammatory": "Anti-inflammatory",
    "hemolytic": "Hemolytic",
    "antiviral": "Antiviral",
    "cytotoxic": "Cytotoxic",
    "antitubercular": "Antitubercular",
    "antibiofilm": "Antibiofilm",
    "antiparasitic": "Antiparasitic",
}
SCORERS = ("oracle", "ensemble")
SCORER_LABELS = {"oracle": "Oracle", "ensemble": "Ensemble"}

OVERLAP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)

ANTIMICROBIAL_REGIONS = (("Beta-casein", 194, 226), ("Lactotransferrin", 288, 307))

GROUPS = (
    ("Positive", "KRH"),
    ("Negative", "DE"),
    ("Hydrophobic", "AVILMFWP"),
    ("Polar", "STNQ"),
    ("Special", "CGY"),
)
GROUP_LABELS = ("Pos", "Neg", "Hyd", "Pol", "Spc")
RESIDUE_ORDER = "KRHDEAVILMFWPSTNQCGY"
POSITIVE_PKA = {"K": 10.5, "R": 12.4, "H": 6.0}
NEGATIVE_PKA = {"D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1}
TERMINAL_PKA = (9.0, 2.0)

# Descriptive profile of a peptide that is practical to test; counted per candidate and never used as a filter.
PROFILE_NET_CHARGE = 2.0
PROFILE_HYDROPHOBIC_PERCENT = (30.0, 60.0)
PROFILE_MOMENT = 0.722
PROFILE_CYSTEINES = 2

FUNNEL_LABELS = ("All rows", "Human", "Unmodified", "8-50 residues", "Deduplicated", "< 0.5 to corpus")

ACTIVITY_TASKS = (
    "antibacterial", "antifungal", "anticancer", "antiviral",
    "antiinflammatory", "antitubercular", "antibiofilm", "antiparasitic",
)
# Okabe-Ito colors, one per activity
ACTIVITY_COLORS = {
    "antibacterial": "#0072B2",
    "antifungal": "#009E73",
    "anticancer": "#D55E00",
    "antiviral": "#CC79A7",
    "antiinflammatory": "#E69F00",
    "antitubercular": "#56B4E9",
    "antibiofilm": "#F0E442",
    "antiparasitic": "#000000",
}
CANDIDATE_COLOR = "#E4572E"
REST_COLOR = "#2A9D8F"
NEUTRAL_COLOR = "#9A9A9A"
FPR_QUANTILE = 0.95
MIN_VALIDATION_INACTIVES = 30
MIN_CANDIDATES = 10
BASE_FONT = 9.0
TITLE_FONT = 12.0
PIPELINE_TITLE_FONT = 12.5
PIPELINE_FONT = 11.5
PIPELINE_HEADING_FONT = 24.0
PANEL_LETTER_FONT = 28.0
PANEL_LETTER_ADVANCE = 0.36
RADAR_HEADER_RAISE = 0.10
INK_COLOR = "#1A1A1A"
RULE_COLOR = "#E4E4E4"
FIGURES = ROOT / "paper" / "figures"
PERCENT = "\\%"
PANEL_SHIFT = 0.5

GROUP_SHIFT = 0.35
MOTIF_GAP_CUT = 0.30

TABLE_TITLE_FONT = 18.0
TABLE_FONT = 17.0

FIGURE_WIDTH = 9.6
HEAT_LABELS = {"antibacterial": "Bact", "antifungal": "Fung", "anticancer": "Canc", "antiinflammatory": "Infl", "hemolytic": "Hemo"}
MOTIF_MINIMUM = {3: 3, 5: 2}
MOTIF_COUNT = 5

# Chou-Fasman propensities; the structure panel is a propensity estimate, not a structure predictor
HELIX_PROPENSITY = dict(zip("EMALKFQWIVDHRTSCYNPG", (1.51, 1.45, 1.42, 1.21, 1.16, 1.13, 1.11, 1.08, 1.08, 1.06, 1.01, 1.00, 0.98, 0.83, 0.77, 0.70, 0.69, 0.67, 0.57, 0.57)))
SHEET_PROPENSITY = dict(zip("VIYFWLCTQMRNHASGKPDE", (1.70, 1.60, 1.47, 1.38, 1.37, 1.30, 1.19, 1.19, 1.10, 1.05, 0.93, 0.89, 0.87, 0.83, 0.75, 0.75, 0.74, 0.55, 0.54, 0.37)))

# --------------------------------------------------------------------------- table and filters


def read_table(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Table S2", header=None, dtype=object)
    header = int(raw.index[raw[0] == "Sequence"][0])
    table = raw.iloc[header + 1 :].copy()
    table.columns = [str(name).strip() for name in raw.iloc[header]]
    table = table.dropna(subset=["Sequence"]).reset_index(drop=True)
    table["Sequence"] = table["Sequence"].astype(str).str.upper().str.strip()
    table[HOMOLOGY] = pd.to_numeric(table[HOMOLOGY], errors="coerce")

    for column in ("Start", "Stop"):
        table[column] = pd.to_numeric(table[column], errors="coerce")

    return table


def homolog_class(table: pd.DataFrame) -> np.ndarray:
    antimicrobial = np.zeros(len(table), dtype=bool)

    for protein, first, last in ANTIMICROBIAL_REGIONS:
        antimicrobial |= (
            (table["Protein"] == protein) & (table["Start"] <= last) & (table["Stop"] >= first)
        ).to_numpy()

    antimicrobial &= (table["Species"] == "Human").to_numpy()
    labeled = table[HOMOLOGY].notna().to_numpy()

    return np.where(labeled, np.where(antimicrobial, "antimicrobial", "other"), "unlabeled")


def filter_steps(table: pd.DataFrame):
    steps = []

    def record(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        kind = homolog_class(frame)
        steps.append(
            {
                "step": name,
                "peptides": len(frame),
                "homolog": int(frame[HOMOLOGY].notna().sum()),
                "antimicrobial": int((kind == "antimicrobial").sum()),
                "other": int((kind == "other").sum()),
            }
        )

        return frame

    record("all rows", table)
    frame = record("species Human", table[table["Species"] == "Human"])
    frame = record("modifications blank", frame[frame["Modifications"].isna()])

    length = frame["Sequence"].str.len()
    standard = frame["Sequence"].str.fullmatch(f"[{STANDARD}]+")
    frame = record(
        f"{MIN_LENGTH} to {MAX_LENGTH} residues, 20 standard",
        frame[standard & (length >= MIN_LENGTH) & (length <= MAX_LENGTH)],
    )

    frame = frame.sort_values(HOMOLOGY, ascending=False, na_position="last")
    frame = record("deduplicated", frame.drop_duplicates(subset="Sequence"))

    return frame.reset_index(drop=True), steps


def run_filter(args) -> None:
    from ampx.evaluation import load_frame, max_similarity

    workbook = Path(args.workbook)

    if not workbook.exists():
        raise SystemExit(f"{workbook} not found; pass --workbook")

    table = read_table(workbook)
    corpus = load_frame(Path(args.corpus), ("sequence", "split", "component"))["sequence"].tolist()
    print(f"corpus: {len(corpus)} rows, {len(set(corpus))} unique sequences", flush=True)

    MILK.mkdir(parents=True, exist_ok=True)

    frame, steps = filter_steps(table)
    frame["max_corpus_similarity"] = max_similarity(frame["Sequence"].tolist(), corpus)
    frame["class"] = homolog_class(frame)
    kept = frame[frame["max_corpus_similarity"] < SIMILARITY_CEILING].reset_index(drop=True)
    steps.append(
        {
            "step": f"max similarity to corpus below {SIMILARITY_CEILING}",
            "peptides": len(kept),
            "homolog": int(kept[HOMOLOGY].notna().sum()),
            "antimicrobial": int((kept["class"] == "antimicrobial").sum()),
            "other": int((kept["class"] == "other").sum()),
        }
    )

    funnel = pd.DataFrame(steps)
    print(funnel.to_string(index=False), flush=True)

    labeled = frame[HOMOLOGY].notna()
    print("max corpus similarity quantiles (0.05 0.25 0.5 0.75 0.95) before the similarity filter", flush=True)

    for label, part in (("with homology", frame[labeled]), ("without homology", frame[~labeled])):
        if len(part):
            values = part["max_corpus_similarity"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()
            print(f"  {label}: {values}", flush=True)

    frame.to_csv(MILK / "pool.csv", index=False)
    kept.to_csv(MILK / "primary.csv", index=False)
    funnel.to_csv(MILK / "funnel.csv", index=False)
    print(f"wrote pool.csv, primary.csv and funnel.csv in {MILK}", flush=True)


# --------------------------------------------------------------------------- score


def milk_input_features(sequences: list[str], esmc: np.ndarray, esm2: np.ndarray, transform: str, extra: str) -> np.ndarray:
    # the same blocks, transforms and order as input_features in multitask.py
    from ampx.evaluation import composition_matrix, dipeptide_matrix

    series = pd.Series(sequences)
    composition = composition_matrix(series)
    dipeptide = dipeptide_matrix(series)

    if transform == "stabilized":
        composition[:, :-1] = np.sqrt(composition[:, :-1])
        composition[:, -1] = np.log(composition[:, -1])
        dipeptide = np.sqrt(dipeptide)

    blocks = [esmc, composition, dipeptide]

    if extra == "esm2":
        blocks.insert(1, esm2)

    return np.hstack(blocks).astype(np.float32)


def run_score(args) -> None:
    import torch
    from scipy.stats import rankdata

    from ampx.evaluation import (
        BoostedModel,
        combined_features,
        composition_matrix,
        corpus_embeddings,
        dipeptide_matrix,
        fit_oracles,
        load_frame,
        sequence_features,
        split_indices,
        task_selection,
    )
    from ampx.models.data import TASKS
    from ampx.predictor import ESM2Encoder, ESMCEncoder, resolve_device
    from multitask import CLASSIFICATION_TASKS, build_targets, input_features, standardize_inputs, stored_matrix, train_network
    from prepare_corpus import link_graph

    table = pd.read_csv(MILK / "primary.csv")
    sequences = table["Sequence"].tolist()
    corpus = Path(args.corpus)
    device = resolve_device(args.device)

    frame = load_frame(corpus, ("sequence", "split", "component"))
    corpus_sequences = frame["sequence"].tolist()
    split = frame["split"].to_numpy()
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "val")

    # embeddings of the corpus and of the milk peptides
    encoder = ESM2Encoder(device=device)
    corpus_esm2 = corpus_embeddings(corpus_sequences, corpus / "esm_embeddings.npz", device, encoder)
    milk_esm2, milk_combined = sequence_features(sequences, encoder)
    del encoder
    gc.collect()

    esmc_encoder = ESMCEncoder(device=device)
    milk_esmc = esmc_encoder.encode(sequences, batch_size=128)
    del esmc_encoder
    gc.collect()

    if device != "cpu":
        torch.cuda.empty_cache()

    corpus_esmc = stored_matrix(corpus / "esmc_embeddings.npz", corpus_sequences)

    # oracle: boosted trees on ESM-2 embeddings, composition and dipeptide fractions
    oracles, _ = fit_oracles(frame, combined_features(frame["sequence"], corpus_esm2), SCORED_TASKS, matched=True)
    missing = [task for task in SCORED_TASKS if task not in oracles]

    if missing:
        raise SystemExit(f"no oracle for {missing}")

    oracle = {task: oracles[task].score(milk_combined) for task in SCORED_TASKS}

    # boosted ESM-C with composition and dipeptide fractions, the partner of the ensemble
    corpus_esmc_combined = np.hstack(
        [corpus_esmc, composition_matrix(frame["sequence"]), dipeptide_matrix(frame["sequence"])]
    )
    milk_esmc_combined = np.hstack(
        [milk_esmc, composition_matrix(pd.Series(sequences)), dipeptide_matrix(pd.Series(sequences))]
    )
    esmc = {}

    for task in SCORED_TASKS:
        keep, labels, task_split = task_selection(frame, task, True)
        parts = split_indices(task_split, labels)

        if parts is None:
            raise SystemExit(f"{task}: a split lacks one of the two classes")

        task_train, task_val, _ = parts
        model = BoostedModel(corpus_esmc_combined[keep], labels, task_train, task_val)
        esmc[task] = model.score(milk_esmc_combined)
        print(f"{task}: boosted ESM-C fitted, {model.rounds} rounds", flush=True)

    # multi-task network: the milk rows are appended to the corpus rows, carry no labels, and are predicted at the
    # epoch each task was stopped on; standardization and class weights use the training rows only
    count = len(frame)
    raw = np.vstack(
        [
            input_features(corpus, frame, args.transform, args.extra),
            milk_input_features(sequences, milk_esmc, milk_esm2, args.transform, args.extra),
        ]
    )
    features = standardize_inputs(raw, train, args.clip, device)
    del raw
    gc.collect()

    target, observed = build_targets(frame, True)
    target = np.vstack([target, np.zeros((len(sequences), target.shape[1]), dtype=np.float32)])
    observed = np.vstack([observed, np.zeros((len(sequences), observed.shape[1]), dtype=bool)])
    columns = [TASKS.index(task) for task in CLASSIFICATION_TASKS]

    network_args = argparse.Namespace(
        hidden=args.hidden,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        stopping=args.stopping,
    )

    averaged = np.zeros((len(target), len(TASKS)), dtype=np.float64)

    for seed in range(args.seeds):
        scores, value = train_network(features, target, observed, train, val, columns, network_args, device, seed)
        averaged += scores / args.seeds
        print(f"seed {seed}: best validation macro AUROC {value:.4f}", flush=True)

    multitask = {task: averaged[count:, TASKS.index(task)] for task in SCORED_TASKS}

    result = table.copy()
    result["component"] = connected_components(link_graph(sequences, args.identity, 1000, -1), directed=False)[1]
    result["class"] = homolog_class(table)

    for task in SCORED_TASKS:
        result[f"oracle_{task}"] = oracle[task]
        result[f"multitask_{task}"] = multitask[task]
        result[f"esmc_{task}"] = esmc[task]
        result[f"ensemble_{task}"] = 0.5 * (rankdata(esmc[task]) + rankdata(multitask[task])) / len(sequences)

    result.to_csv(MILK / "scores.csv", index=False)
    print(f"wrote scores.csv: {len(result)} peptides in {result['component'].nunique()} components", flush=True)


# --------------------------------------------------------------------------- descriptors


def residue_percent(sequences: list[str]) -> np.ndarray:
    counts = np.array([[sequence.count(residue) for residue in RESIDUE_ORDER] for sequence in sequences], dtype=float)

    return 100.0 * counts / counts.sum(axis=1, keepdims=True)


def isoelectric_points(sequences: list[str]) -> np.ndarray:
    from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint

    return np.array([IsoelectricPoint(sequence).pi() for sequence in sequences], dtype=float)


def peptide_properties(sequences: list[str]):
    from ampx.evaluation import property_matrix

    properties = property_matrix(pd.Series(sequences)).astype(float)
    percent = residue_percent(sequences)
    frame = pd.DataFrame(
        {
            "length": properties[:, 0],
            "net_charge": properties[:, 1],
            "charge_per_residue": properties[:, 2],
            "hydrophobic_moment": properties[:, 3],
            "gravy": properties[:, 4],
            "cysteines": properties[:, 5],
            "kr_percent": percent[:, RESIDUE_ORDER.index("K")] + percent[:, RESIDUE_ORDER.index("R")],
            "isoelectric_point": isoelectric_points(sequences),
        }
    )

    for name, members in GROUPS:
        frame[f"group_{name}"] = percent[:, [RESIDUE_ORDER.index(residue) for residue in members]].sum(axis=1)

    low, high = PROFILE_HYDROPHOBIC_PERCENT
    frame["profile_met"] = (
        (frame["net_charge"] >= PROFILE_NET_CHARGE).astype(int)
        + ((frame["group_Hydrophobic"] >= low) & (frame["group_Hydrophobic"] <= high)).astype(int)
        + (frame["hydrophobic_moment"] >= PROFILE_MOMENT).astype(int)
        + (frame["cysteines"] <= PROFILE_CYSTEINES).astype(int)
    )

    return frame, percent


def top_set(scores: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    # exactly the top share of peptides; ties at the cut are broken at random
    count = int(np.ceil(fraction * len(scores)))
    order = np.lexsort((np.random.default_rng(seed).random(len(scores)), scores))
    chosen = np.zeros(len(scores), dtype=bool)
    chosen[order[-count:]] = True

    return chosen


def component_mean_pvalue(values: np.ndarray, component: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
    # Mann-Whitney on component means, so nested fragments of one region count once
    def means(mask):
        return pd.Series(values[mask]).groupby(component[mask]).mean().to_numpy()

    return float(stats.mannwhitneyu(means(high), means(low), alternative="two-sided").pvalue)


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# --------------------------------------------------------------------------- selection


def candidate_frame(context: dict, rows: np.ndarray) -> pd.DataFrame:
    scores, properties, percentile, raw = context["scores"], context["properties"], context["percentile"], context["raw"]

    frame = pd.DataFrame(
        {
            "sequence": scores["Sequence"].to_numpy()[rows],
            "protein": scores["Protein"].to_numpy()[rows],
            "start": scores["Start"].to_numpy()[rows],
            "stop": scores["Stop"].to_numpy()[rows],
            "component": context["component"][rows],
            "antibacterial_percentile_ensemble": percentile[("ensemble", TARGET)][rows],
            "antibacterial_percentile_oracle": percentile[("oracle", TARGET)][rows],
            "hemolytic_percentile_ensemble": percentile[("ensemble", HEMOLYTIC)][rows],
            "hemolytic_percentile_oracle": percentile[("oracle", HEMOLYTIC)][rows],
            "oracle_probability": raw[("oracle", TARGET)][rows],
            "multitask_probability": scores[f"multitask_{TARGET}"].to_numpy()[rows],
            "esmc_probability": scores[f"esmc_{TARGET}"].to_numpy()[rows],
            "max_corpus_similarity": scores["max_corpus_similarity"].to_numpy()[rows],
            "homology_class": context["kind"][rows],
        }
    )

    for column in properties.columns:
        frame[column] = properties[column].to_numpy()[rows]

    for task in SCORED_TASKS:
        frame[f"{task}_percentile_mean"] = 0.5 * (
            context["percentile"][("ensemble", task)][rows] + context["percentile"][("oracle", task)][rows]
        )
        frame[f"{task}_called"] = context["calls"][task][rows]

    frame["activities_called"] = frame[[f"{task}_called" for task in SCORED_TASKS if task != HEMOLYTIC]].sum(axis=1)
    frame["start"] = frame["start"].astype("Int64")
    frame["stop"] = frame["stop"].astype("Int64")

    return frame

def corpus_activity(corpus: Path) -> pd.DataFrame:
    from ampx.evaluation import load_frame, task_selection
    from ampx.models.data import TASKS

    frame = load_frame(corpus, ("sequence", "split"))
    rows = []

    for task in TASKS:
        state = frame[f"act_{task}_idx"].to_numpy()
        keep, labels, _ = task_selection(frame, task, True)
        rows.append(
            {
                "task": task,
                "corpus": len(frame),
                "labeled": int((state > 0).sum()),
                "active": int((state == 2).sum()),
                "inactive": int((state == 1).sum()),
                "matched_labeled": int(len(keep)),
                "matched_active": int(labels.sum()),
            }
        )

    table = pd.DataFrame(rows)
    table["labeled_share"] = table["labeled"] / table["corpus"]
    table["active_share_of_labeled"] = table["active"] / table["labeled"].clip(lower=1)
    table["active_share_of_corpus"] = table["active"] / table["corpus"]

    return table

def analyse(args) -> dict:
    from ampx.evaluation import auroc, component_groups, percentile_interval, resample_components

    scores = pd.read_csv(MILK / "scores.csv")
    scores["Protein"] = scores["Protein"].fillna("unknown")
    count = len(scores)
    sequences = scores["Sequence"].tolist()
    component = scores["component"].to_numpy()
    kind = scores["class"].to_numpy()
    properties, percent = peptide_properties(sequences)

    raw = {(scorer, task): scores[f"{scorer}_{task}"].to_numpy() for scorer in SCORERS for task in SCORED_TASKS}
    percentile = {key: stats.rankdata(values) / count for key, values in raw.items()}

    top = {scorer: top_set(raw[(scorer, TARGET)], args.top) for scorer in SCORERS}
    consensus = top["oracle"] & top["ensemble"]
    calls = {
        task: top_set(raw[("oracle", task)], args.call) & top_set(raw[("ensemble", task)], args.call)
        for task in SCORED_TASKS
    }
    limit = 1.0 - args.hemolytic_top
    safe = np.all([percentile[(scorer, HEMOLYTIC)] <= limit for scorer in SCORERS], axis=0)
    eligible = consensus & safe
    mean_percentile = np.mean([percentile[(scorer, TARGET)] for scorer in SCORERS], axis=0)

    context = {
        "scores": scores,
        "count": count,
        "component": component,
        "kind": kind,
        "properties": properties,
        "percent": percent,
        "raw": raw,
        "percentile": percentile,
        "top": top,
        "calls": calls,
        "corpus_activity": corpus_activity(Path(args.corpus)),
        "consensus": consensus,
        "safe": safe,
        "eligible": eligible,
    }

    # overlap of the two top sets, with intervals that resample components
    groups = component_groups(component)
    generator = np.random.default_rng(args.seed)
    draws = [resample_components(groups, generator) for _ in range(args.draws)]
    everything = np.arange(count)
    rows = []

    for fraction in sorted(set(OVERLAP_FRACTIONS) | {args.top}):
        first = top_set(raw[("oracle", TARGET)], fraction)
        second = top_set(raw[("ensemble", TARGET)], fraction)

        def share(index):
            return float((first[index] & second[index]).sum() / max(first[index].sum(), 1))

        low, high = percentile_interval([share(index) for index in draws])
        rows.append(
            {
                "fraction": fraction,
                "n_each": int(first.sum()),
                "n_shared": int((first & second).sum()),
                "components_shared": int(np.unique(component[first & second]).size),
                "share": share(everything),
                "lo": low,
                "hi": high,
                "chance": fraction,
            }
        )

    context["overlap"] = pd.DataFrame(rows)

    def spearman(index):
        return float(stats.spearmanr(raw[("oracle", TARGET)][index], raw[("ensemble", TARGET)][index]).statistic)

    rho_low, rho_high = percentile_interval([spearman(index) for index in draws])
    context["spearman"] = (spearman(everything), rho_low, rho_high)

    # agreement with the known antimicrobial homologs, point estimates
    homolog = {}

    for scorer in SCORERS:
        for name in ("antimicrobial", "other"):
            keep = (kind == name) | (kind == "unlabeled")
            homolog[(scorer, name)] = auroc((kind[keep] == name).astype(np.int64), raw[(scorer, TARGET)][keep])

    context["homolog"] = homolog

    # shortlist: one peptide per component, ranked by the mean antibacterial percentile of the two scorers
    chosen = np.flatnonzero(eligible)
    members = pd.Series(component[chosen]).value_counts()
    ranked = (
        pd.DataFrame({"row": chosen, "component": component[chosen], "mean_percentile": mean_percentile[chosen]})
        .sort_values("mean_percentile", ascending=False)
        .drop_duplicates("component")
        .reset_index(drop=True)
    )
    shortlist = candidate_frame(context, ranked["row"].to_numpy())
    shortlist.insert(0, "rank", np.arange(1, len(shortlist) + 1))
    shortlist["members_in_shortlist"] = members.reindex(ranked["component"]).to_numpy()
    context["shortlist"] = shortlist

    # every peptide in the top set of either scorer
    union = np.flatnonzero(top["oracle"] | top["ensemble"])
    union = union[np.argsort(-mean_percentile[union], kind="stable")]
    top_sets = candidate_frame(context, union)
    top_sets["in_oracle_top"] = top["oracle"][union]
    top_sets["in_ensemble_top"] = top["ensemble"][union]
    top_sets["outside_hemolytic_top"] = safe[union]
    context["top_sets"] = top_sets

    return context


def run_select(args) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_colwidth", 60)

    analysis = analyse(args)
    component, kind = analysis["component"], analysis["kind"]
    shortlist = analysis["shortlist"]
    count = analysis["count"]

    print(f"\n{count} milk peptides in {int(np.unique(component).size)} components at similarity {args.identity}", flush=True)

    for scorer in SCORERS:
        top = analysis["top"][scorer]
        print(
            f"{SCORER_LABELS[scorer]}: top {args.top:.0%} = {int(top.sum())} peptides in "
            f"{int(np.unique(component[top]).size)} components",
            flush=True,
        )

    overlap = analysis["overlap"]
    print("\noverlap of the two antibacterial top sets, interval over components", flush=True)
    print(overlap.round(3).to_string(index=False), flush=True)
    rho = analysis["spearman"]
    print(f"Spearman between the two antibacterial scores: {rho[0]:.3f} [{rho[1]:.3f}, {rho[2]:.3f}]", flush=True)

    consensus, eligible = analysis["consensus"], analysis["eligible"]
    print(
        f"\nin the top {args.top:.0%} under both scorers: {int(consensus.sum())}; "
        f"also outside the top {args.hemolytic_top:.0%} hemolytic under both: {int(eligible.sum())}; "
        f"one per component: {len(shortlist)}",
        flush=True,
    )

    if len(shortlist):
        print("\nprofile criteria met by the shortlist (of 4)", flush=True)
        print(shortlist["profile_met"].value_counts().sort_index().to_string(), flush=True)

    labeled = kind != "unlabeled"
    labeled_components = int(np.unique(component[labeled]).size)
    print(
        f"\nhomology-labeled peptides: {int(labeled.sum())} in {labeled_components} components; "
        f"pilot rule (fewer than {PILOT_COMPONENTS}): {labeled_components < PILOT_COMPONENTS}",
        flush=True,
    )

    for name in ("antimicrobial", "other"):
        print(
            f"  {name}: {int((kind == name).sum())} peptides; in both top sets {int(((kind == name) & consensus).sum())}; "
            "AUROC against unlabeled "
            + ", ".join(f"{SCORER_LABELS[scorer]} {analysis['homolog'][(scorer, name)]:.3f}" for scorer in SCORERS),
            flush=True,
        )

    activity = analysis["corpus_activity"]
    activity.to_csv(MILK / "corpus_activity.csv", index=False)
    print("\nlabels in the corpus", flush=True)
    print(activity.round(3).to_string(index=False), flush=True)

    print(f"\npeptides called per activity: top {args.call:.0%} of the milk peptides under both scorers", flush=True)
    counts = pd.DataFrame(
        {
            "milk": {task: int(analysis["calls"][task].sum()) for task in SCORED_TASKS},
            "shortlist": {task: int(shortlist[f"{task}_called"].sum()) for task in SCORED_TASKS},
        }
    )
    print(counts.to_string(), flush=True)

    print("\nshortlist activity percentiles (mean of the two scorers)", flush=True)
    columns = ["rank", "sequence", "activities_called"] + [f"{task}_percentile_mean" for task in SCORED_TASKS]
    print(shortlist[columns].round(2).to_string(index=False), flush=True)
    shortlist.to_csv(MILK / "shortlist.csv", index=False)
    analysis["top_sets"].to_csv(MILK / "top_sets.csv", index=False)

    shown = [
        "rank", "sequence", "protein", "start", "stop", "antibacterial_percentile_ensemble", "antibacterial_percentile_oracle",
        "hemolytic_percentile_ensemble", "hemolytic_percentile_oracle", "net_charge", "hydrophobic_moment", "gravy", "length",
        "profile_met",
    ]
    print(f"\nshortlist, first {args.table}", flush=True)
    print(shortlist.head(args.table)[shown].round(2).to_string(index=False), flush=True)
    print(f"\nwrote shortlist.csv, top_sets.csv and overlap.csv in {MILK}", flush=True)


# --------------------------------------------------------------------------- figure

def windowed_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window)

    return np.convolve(values, kernel, mode="same") / np.convolve(np.ones(len(values)), kernel, mode="same")


def secondary_structure(sequences: list[str]) -> pd.DataFrame:
    # three-state estimate from windowed Chou-Fasman propensities: helix window 6 and threshold 1.03, sheet window 5 and threshold 1.05
    rows = []

    for sequence in sequences:
        helix = windowed_mean(np.array([HELIX_PROPENSITY[residue] for residue in sequence]), 6)
        sheet = windowed_mean(np.array([SHEET_PROPENSITY[residue] for residue in sequence]), 5)
        is_helix = (helix >= 1.03) & (helix >= sheet)
        is_sheet = (sheet >= 1.05) & ~is_helix
        rows.append((is_helix.mean(), is_sheet.mean(), 1.0 - is_helix.mean() - is_sheet.mean()))

    return pd.DataFrame(rows, columns=["helix", "sheet", "coil"])


def motif_enrichment(sequences: list[str], component: np.ndarray, top: np.ndarray, length: int, minimum: int, count: int) -> list:
    # a k-mer is counted once per homology component within each group
    def tally(mask):
        found = {}

        for row in np.flatnonzero(mask):
            sequence = sequences[row]

            for motif in {sequence[start : start + length] for start in range(len(sequence) - length + 1)}:
                found.setdefault(motif, set()).add(component[row])

        return {motif: len(groups) for motif, groups in found.items()}, int(np.unique(component[mask]).size)

    top_counts, top_total = tally(top)
    rest_counts, rest_total = tally(~top)
    rows = []

    for motif, number in top_counts.items():
        if number < minimum:
            continue

        top_share = number / top_total
        rest_share = rest_counts.get(motif, 0) / rest_total
        ratio = top_share / ((rest_counts.get(motif, 0) + 1.0) / (rest_total + 1.0))
        rows.append((motif, top_share, rest_share, number, ratio))

    rows.sort(key=lambda item: (item[4], item[3]), reverse=True)

    return rows[:count]


def run_figure(args) -> None:
    from make_figures_main import (
        BOOSTED, GREEN, GREY, INK, TASK_SHORT,
        bands, bottom_legend, box_axes, group_letter, panel, placer, save, set_style, subtitle,
    )
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    for name in ("scores.csv", "funnel.csv"):
        if not (MILK / name).exists():
            raise SystemExit(f"missing {MILK / name}; run the earlier stages")

    analysis = analyse(args)
    funnel = pd.read_csv(MILK / "funnel.csv")
    activity = analysis["corpus_activity"]
    component = analysis["component"]
    properties, percent = analysis["properties"], analysis["percent"]
    percentile = analysis["percentile"]
    consensus = analysis["consensus"]
    rest = ~consensus
    shortlist = analysis["shortlist"]
    overlap = analysis["overlap"]
    count = analysis["count"]
    sequences = analysis["scores"]["Sequence"].tolist()

    if consensus.sum() < 5:
        raise SystemExit(f"only {int(consensus.sum())} peptides are in both top sets; raise --top")

    if shortlist.empty:
        raise SystemExit("no peptide in both top sets lies outside the hemolytic exclusion; raise --hemolytic-top")

    shown = shortlist.head(args.table)
    shown_count = len(shown)
    structure = 100.0 * secondary_structure(sequences)
    top_components = int(np.unique(component[consensus]).size)

    features = {
        "net_charge": properties["net_charge"].to_numpy(),
        "positive_ratio": properties["kr_percent"].to_numpy() / 100.0,
        "hydrophobic_moment": properties["hydrophobic_moment"].to_numpy(),
        "isoelectric_point": properties["isoelectric_point"].to_numpy(),
        "kr_percent": properties["kr_percent"].to_numpy(),
        "negative_percent": properties["group_Negative"].to_numpy(),
        "hydrophobic_percent": properties["group_Hydrophobic"].to_numpy(),
        "gravy": properties["gravy"].to_numpy(),
        "length": properties["length"].to_numpy(),
        "helix_percent": structure["helix"].to_numpy(),
        "sheet_percent": structure["sheet"].to_numpy(),
        "coil_percent": structure["coil"].to_numpy(),
    }
    print("\nfeatures of the top set against the rest: means, Mann-Whitney on component means", flush=True)

    for name, values in features.items():
        print(
            f"  {name:20s} top {values[consensus].mean():9.3f}  rest {values[rest].mean():9.3f}  "
            f"p {component_mean_pvalue(values, component, consensus, rest):.2g}",
            flush=True,
        )

    motifs = {
        length: motif_enrichment(sequences, component, consensus, length, MOTIF_MINIMUM[length], MOTIF_COUNT)
        for length in (3, 5)
    }

    for length, found in motifs.items():
        print(f"\nenriched {length}-mers: share of components, top against rest", flush=True)

        for motif, top_share, rest_share, number, ratio in found:
            print(f"  {motif}  {number} components  {100 * top_share:.1f}% against {100 * rest_share:.1f}%  enrichment {ratio:.1f}", flush=True)

    set_style()

    width = FIGURE_WIDTH
    height = FIGURE_WIDTH * 9.0 / 16.0
    tick, label_size, note, title_size, letter_size = 6.0, 6.2, 5.4, 6.8, 8.6
    top_one, tall_one = 0.32, 1.15
    top_two, tall_two = 2.02, 1.00
    top_three, pitch = 3.56, 0.112

    fig = plt.figure(figsize=(width, height))
    place = placer(fig, width, height)
    boxed = []

    # A: filters
    ax_a = place(1.00, top_one, 1.05, tall_one)
    steps = np.arange(len(funnel))[::-1].astype(float)
    bands(ax_a, steps)

    for position, step in zip(steps, funnel.itertuples(index=False)):
        ax_a.barh(position, step.peptides, height=0.62, color=GREEN if position == 0.0 else BOOSTED, edgecolor="white", linewidth=0.3)
        ax_a.text(1.03, position, f"{int(step.peptides):,}", va="center", ha="left", fontsize=note, transform=ax_a.get_yaxis_transform())

    ax_a.set_yticks(steps)
    ax_a.set_yticklabels(list(FUNNEL_LABELS[: len(funnel)]))
    ax_a.set_ylim(-0.6, len(funnel) - 0.4)
    ax_a.set_xlim(0, float(funnel["peptides"].max()) * 1.05)
    ax_a.set_xticks([0, 5000, 10000])
    ax_a.set_xlabel("Peptides", fontsize=label_size, labelpad=1.0)
    ax_a.grid(axis="x", color="#E4E4E4", lw=0.4, zorder=0)
    ax_a.set_axisbelow(True)
    panel(ax_a, "A", "Human milk peptides", size=title_size, letter_size=letter_size)
    boxed.append(ax_a)

    # B: amino acid composition, radar and residue groups
    ax_r = place(2.65, top_one, 1.15, tall_one, projection="polar")
    letters = list("ACDEFGHIKLMNPQRSTVWY")
    angles = np.linspace(0, 2 * np.pi, len(letters), endpoint=False)

    for mask, color in ((consensus, GREEN), (rest, GREY)):
        values = np.array([percent[mask, RESIDUE_ORDER.index(letter)].mean() for letter in letters])
        ax_r.plot(np.r_[angles, angles[0]], np.r_[values, values[0]], color=color, lw=0.7, marker="o", ms=1.2)
        ax_r.fill(np.r_[angles, angles[0]], np.r_[values, values[0]], color=color, alpha=0.12)

    ax_r.set_xticks(angles)
    ax_r.set_xticklabels(letters, fontsize=tick)
    ax_r.tick_params(axis="x", pad=-3)
    ax_r.tick_params(axis="y", labelsize=tick - 1.2)
    ax_r.set_rlabel_position(0)
    ax_r.grid(color="#E4E4E4", lw=0.4)
    ax_r.spines["polar"].set_color(GREY)
    ax_r.set_title("Amino acid composition (%)", fontsize=title_size, pad=7)
    group_letter(fig, width, height, "B", 2.65, top_one, letter_size)

    ax_c = place(4.25, top_one, 0.85, tall_one)
    positions = np.arange(len(GROUPS))

    for offset, mask, color in ((-0.2, consensus, GREEN), (0.2, rest, GREY)):
        ax_c.bar(
            positions + offset, [properties[f"group_{name}"].to_numpy()[mask].mean() for name, _ in GROUPS],
            width=0.38, color=color, edgecolor=INK, linewidth=0.3,
        )

    ax_c.set_xticks(positions)
    ax_c.set_xticklabels(GROUP_LABELS)
    ax_c.set_ylabel("Composition (%)", fontsize=label_size, labelpad=1.0)
    ax_c.grid(axis="y", color="#E4E4E4", lw=0.4, zorder=0)
    ax_c.set_axisbelow(True)
    subtitle(ax_c, "Residue groups", title_size)
    boxed.append(ax_c)

    # C: secondary structure estimate
    ax_s = place(5.60, top_one, 0.75, tall_one)
    kinds = (("helix", "Helix"), ("sheet", "Sheet"), ("coil", "Coil"))
    ceiling = 0.0

    for offset, mask, color in ((-0.2, consensus, GREEN), (0.2, rest, GREY)):
        means = [structure[key].to_numpy()[mask].mean() for key, _ in kinds]
        ceiling = max(ceiling, max(means))
        ax_s.bar(np.arange(len(kinds)) + offset, means, width=0.38, color=color, edgecolor=INK, linewidth=0.3)

    for position, (key, _) in enumerate(kinds):
        level = max(structure[key].to_numpy()[consensus].mean(), structure[key].to_numpy()[rest].mean())
        ax_s.text(position, level + 0.02 * ceiling, stars(component_mean_pvalue(structure[key].to_numpy(), component, consensus, rest)),
                  ha="center", va="bottom", fontsize=note)

    ax_s.set_ylim(0, 1.22 * ceiling)
    ax_s.set_xticks(np.arange(len(kinds)))
    ax_s.set_xticklabels([label for _, label in kinds])
    ax_s.set_ylabel("Residues (%)", fontsize=label_size, labelpad=1.0)
    ax_s.grid(axis="y", color="#E4E4E4", lw=0.4, zorder=0)
    ax_s.set_axisbelow(True)
    panel(ax_s, "C", "Structure", size=title_size, letter_size=letter_size)
    boxed.append(ax_s)

    # D: physicochemical properties
    violin_axes = []

    for index, (key, title) in enumerate(
        (("net_charge", "Net charge"), ("positive_ratio", "K+R ratio"), ("hydrophobic_moment", "Moment"), ("isoelectric_point", "pI"))
    ):
        ax = place(6.95 + index * 0.62, top_one, 0.44, tall_one)
        values = features[key]
        data = [values[rest], values[consensus]]
        parts = ax.violinplot(data, positions=[0, 1], widths=0.85, showextrema=False, showmedians=True)

        for body, color in zip(parts["bodies"], (GREY, GREEN)):
            body.set_facecolor(color)
            body.set_edgecolor(INK)
            body.set_linewidth(0.3)
            body.set_alpha(0.75)

        parts["cmedians"].set_color(INK)
        parts["cmedians"].set_linewidth(0.6)
        lower, upper = float(min(map(np.min, data))), float(max(map(np.max, data)))
        span = max(upper - lower, 1e-9)
        level = upper + 0.05 * span
        ax.plot([0, 0, 1, 1], [level, level + 0.03 * span, level + 0.03 * span, level], color=INK, lw=0.4)
        ax.text(0.5, level + 0.02 * span, stars(component_mean_pvalue(values, component, consensus, rest)),
                ha="center", va="bottom", fontsize=note)
        ax.set_ylim(lower - 0.05 * span, upper + 0.2 * span)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([])
        subtitle(ax, title, title_size - 0.4)
        violin_axes.append(ax)

    group_letter(fig, width, height, "D", 6.95, top_one, letter_size)
    boxed.extend(violin_axes)

    # E: enriched k-mers, share of homology components containing the motif
    motif_axes = []

    for ax, (length, left) in zip((place(0.80, top_two, 0.62, tall_two), place(1.95, top_two, 0.62, tall_two)), ((3, 0.80), (5, 1.95))):
        found = motifs[length]
        motif_axes.append(ax)

        if found:
            y = np.arange(len(found))[::-1].astype(float)
            ax.barh(y + 0.2, [100.0 * item[1] for item in found], height=0.36, color=GREEN, edgecolor="white", linewidth=0.3)
            ax.barh(y - 0.2, [100.0 * item[2] for item in found], height=0.36, color=GREY, edgecolor="white", linewidth=0.3)
            ax.set_yticks(y)
            ax.set_yticklabels([item[0] for item in found])
            ax.set_ylim(-0.7, len(found) - 0.3)
            ax.set_xlim(0, 100.0 * max(item[1] for item in found) * 1.15)
            ax.locator_params(axis="x", nbins=3)
        else:
            ax.text(0.5, 0.5, f"none in {MOTIF_MINIMUM[length]}\ncomponents", ha="center", va="center", fontsize=note, color=GREY, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel("Components (%)", fontsize=label_size, labelpad=1.0)
        ax.grid(axis="x", color="#E4E4E4", lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        subtitle(ax, f"{length}-mers", title_size)

    group_letter(fig, width, height, "E", 0.80, top_two, letter_size)
    boxed.extend(motif_axes)

    # F: cationic content
    ax_f = place(3.20, top_two, 0.90, tall_two)
    upper = float(np.ceil(properties["kr_percent"].max() / 5.0) * 5.0)
    edges = np.linspace(0.0, upper, 11)

    for mask, color in ((rest, GREY), (consensus, GREEN)):
        density = np.histogram(properties["kr_percent"].to_numpy()[mask], bins=edges, density=True)[0]
        ax_f.stairs(density, edges, fill=True, color=color, alpha=0.35, linewidth=0)
        ax_f.stairs(density, edges, color=color, linewidth=0.8)

    ax_f.set_xlim(0, upper)
    ax_f.set_ylim(bottom=0)
    ax_f.set_xlabel("K+R, % of residues", fontsize=label_size, labelpad=1.0)
    ax_f.set_ylabel("Density", fontsize=label_size, labelpad=1.0)
    ax_f.grid(color="#E4E4E4", lw=0.4, zorder=0)
    panel(ax_f, "F", "Cationic content", size=title_size, letter_size=letter_size)
    boxed.append(ax_f)

    # G: agreement of the two scorers
    ax_g = place(4.75, top_two, 0.90, tall_two)
    index = np.arange(len(overlap))
    ax_g.fill_between(index, overlap["lo"], overlap["hi"], color=BOOSTED, alpha=0.25, linewidth=0)
    ax_g.plot(index, overlap["share"], color=BOOSTED, lw=1.0, marker="o", ms=2.0)
    ax_g.plot(index, overlap["chance"], color=GREY, lw=0.8, ls="--")
    ax_g.set_xticks(index)
    ax_g.set_xticklabels([f"{100 * value:g}" for value in overlap["fraction"]])
    ax_g.set_ylim(0, 1)
    ax_g.set_xlabel("Top share, %", fontsize=label_size, labelpad=1.0)
    ax_g.set_ylabel("Shared by both scorers", fontsize=label_size, labelpad=1.0)
    rho = analysis["spearman"]
    ax_g.text(0.04, 0.96, f"Spearman {rho[0]:.2f}\n[{rho[1]:.2f}, {rho[2]:.2f}]", transform=ax_g.transAxes,
              ha="left", va="top", fontsize=note)
    ax_g.grid(color="#E4E4E4", lw=0.4, zorder=0)
    panel(ax_g, "G", "Scorer agreement", size=title_size, letter_size=letter_size)
    boxed.append(ax_g)

    # H: antibacterial against hemolytic percentile
    ax_h = place(6.30, top_two, 0.90, tall_two)
    hemolytic_worst = np.maximum(percentile[("ensemble", HEMOLYTIC)], percentile[("oracle", HEMOLYTIC)])
    antibacterial_mean = 0.5 * (percentile[("ensemble", TARGET)] + percentile[("oracle", TARGET)])
    ax_h.scatter(hemolytic_worst[rest], antibacterial_mean[rest], s=0.8, color=GREY, alpha=0.35, linewidths=0, rasterized=True)
    ax_h.scatter(hemolytic_worst[consensus], antibacterial_mean[consensus], s=3.0, color=GREEN, linewidths=0, zorder=3)
    ax_h.axvline(1.0 - args.hemolytic_top, color=INK, lw=0.4, ls=":", zorder=2)
    shown_x = np.maximum(shown["hemolytic_percentile_ensemble"], shown["hemolytic_percentile_oracle"]).to_numpy()
    shown_y = 0.5 * (shown["antibacterial_percentile_ensemble"] + shown["antibacterial_percentile_oracle"]).to_numpy()

    for number, x, y in zip(shown["rank"], shown_x, shown_y):
        ax_h.text(x, y, str(number), fontsize=note - 1.0, ha="center", va="center", color=INK, zorder=4)

    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0.9, 1.005)
    ax_h.set_xlabel("Hemolytic percentile", fontsize=label_size, labelpad=1.0)
    ax_h.set_ylabel("Antibacterial percentile", fontsize=label_size, labelpad=1.0)
    panel(ax_h, "H", "Safety filter", size=title_size, letter_size=letter_size)
    boxed.append(ax_h)

    # I: label coverage of the corpus
    ax_i = place(8.20, top_two, 1.15, tall_two)
    positions_i = np.arange(len(activity))[::-1].astype(float)
    ax_i.barh(positions_i + 0.2, 100.0 * activity["labeled_share"], height=0.36, color=BOOSTED, edgecolor="white", linewidth=0.3)
    ax_i.barh(positions_i - 0.2, 100.0 * activity["active_share_of_labeled"], height=0.36, color=GREEN, edgecolor="white", linewidth=0.3)
    ax_i.set_yticks(positions_i)
    ax_i.set_yticklabels([f"{TASK_SHORT.get(task, task)} {int(labeled):,}" for task, labeled in zip(activity["task"], activity["labeled"])])
    ax_i.set_ylim(-0.7, len(activity) - 0.3)
    ax_i.set_xlim(0, 100)
    ax_i.set_xlabel("Percent", fontsize=label_size, labelpad=1.0)
    ax_i.grid(axis="x", color="#E4E4E4", lw=0.4, zorder=0)
    ax_i.set_axisbelow(True)
    panel(ax_i, "I", f"Corpus labels ({int(activity['corpus'].iloc[0]):,})", size=title_size, letter_size=letter_size)
    boxed.append(ax_i)

    for ax in boxed:
        ax.tick_params(labelsize=tick)

    ax_i.tick_params(axis="y", labelsize=note)

    for ax in motif_axes:
        for text in ax.get_yticklabels():
            text.set_family("monospace")

    box_axes(*boxed)

    # J: shortlist table with the activity heatmap on the same rows
    table_width = 7.30
    ax_t = place(0.25, top_three, table_width, (shown_count + 1) * pitch)
    ax_t.set_xlim(0, table_width)
    ax_t.set_ylim(shown_count + 1, 0)
    ax_t.axis("off")
    bands(ax_t, np.arange(1, shown_count + 1) + 0.5)
    panel(ax_t, "J", f"Shortlist for wet-lab testing: {len(shortlist)} peptides, one per homology component", size=title_size, letter_size=letter_size)

    columns = (
        ("rank", 0.03, "#"), ("sequence", 0.32, "Sequence"), ("region", 1.75, "Parent protein, residues"),
        ("length", 3.20, "Len."), ("charge", 3.60, "Charge"), ("kr", 4.05, "K+R %"), ("moment", 4.52, "Moment"),
        ("gravy", 5.02, "GRAVY"), ("pi", 5.50, "pI"), ("hydrophobic", 5.95, "Hyd. %"), ("profile", 6.45, "Prof."),
        ("called", 6.90, "Act."),
    )

    def shorten(text, limit=30):
        return text if len(text) <= limit else text[: limit - 1] + "."

    for _, x, label in columns:
        ax_t.text(x, 0.5, label, ha="left", va="center", fontsize=note, color=GREY)

    for index, row in enumerate(shown.itertuples(index=False)):
        y = index + 1.5
        region = f"{shorten(row.protein)} {row.start}-{row.stop}" if pd.notna(row.start) else shorten(row.protein)
        cells = {
            "rank": str(row.rank),
            "region": region,
            "length": f"{int(row.length)}",
            "charge": f"{row.net_charge:+.1f}",
            "kr": f"{row.kr_percent:.0f}",
            "moment": f"{row.hydrophobic_moment:.2f}",
            "gravy": f"{row.gravy:+.2f}",
            "pi": f"{row.isoelectric_point:.1f}",
            "hydrophobic": f"{row.group_Hydrophobic:.0f}",
            "profile": f"{int(row.profile_met)}/4",
            "called": f"{int(row.activities_called)}/4",
        }

        for key, x, _ in columns:
            if key == "sequence":
                ax_t.text(x, y, row.sequence, ha="left", va="center", fontsize=note - 0.6, family="monospace", color=INK)
            else:
                ax_t.text(x, y, cells[key], ha="left", va="center", fontsize=note - 0.2, color=INK)

    heat_left, heat_width = 7.70, 1.70
    ax_m = place(heat_left, top_three + pitch, heat_width, shown_count * pitch)
    heat = shown[[f"{task}_percentile_mean" for task in SCORED_TASKS]].to_numpy(dtype=float)
    called = shown[[f"{task}_called" for task in SCORED_TASKS]].to_numpy(dtype=bool)
    ax_m.imshow(heat, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0)

    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            ax_m.text(column_index, row_index, f"{heat[row_index, column_index]:.2f}", ha="center", va="center", fontsize=note - 0.8,
                      color="white" if heat[row_index, column_index] > 0.7 else INK)

            if called[row_index, column_index]:
                ax_m.add_patch(Rectangle((column_index - 0.5, row_index - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=0.8))

    ax_m.xaxis.tick_top()
    ax_m.set_xticks(range(len(SCORED_TASKS)))
    ax_m.set_xticklabels([HEAT_LABELS[task] for task in SCORED_TASKS], fontsize=note)
    ax_m.set_yticks([])
    ax_m.tick_params(axis="x", length=0, pad=1.5)
    box_axes(ax_m)
    fig.text(heat_left / width, 1.0 - (top_three - 0.049) / height, "Predicted activity, percentile", fontsize=title_size, ha="left", va="bottom")

    low, high = PROFILE_HYDROPHOBIC_PERCENT
    bottom_legend(
        fig,
        [
            Patch(facecolor=GREEN, label=f"Top {args.top:.0%} under both scorers (n={int(consensus.sum())}, {top_components} components)"),
            Patch(facecolor=GREY, label=f"Rest of the milk peptides (n={int(rest.sum()):,})"),
            Line2D([], [], lw=0, label="D: charge, K+R ratio, hydrophobic moment, pI; stars, Mann-Whitney on component means"),
            Line2D([], [], lw=0, label="C: Chou-Fasman propensity estimate; E: share of components with the k-mer"),
            Line2D([], [], lw=0, label="G: Oracle against Ensemble; band, 95% interval over components; dashed, chance"),
            Line2D([], [], lw=0, label=f"H: dotted line, top {args.hemolytic_top:.0%} hemolytic excluded; numbers, shortlist ranks"),
            Line2D([], [], lw=0, label="I: blue, share of corpus labeled; green, share of labeled that is active; n labeled"),
            Line2D([], [], lw=0, label="Model predictions only, none tested; charge without termini"),
            Line2D([], [], lw=0, label=f"J: Prof., criteria met of charge >= +{PROFILE_NET_CHARGE:.0f}, hydrophobic {low:.0f}-{high:.0f}%, moment >= {PROFILE_MOMENT}, Cys <= {PROFILE_CYSTEINES}"),
            Line2D([], [], lw=0, label=f"J heatmap: mean percentile of both scorers; outlined, top {args.call:.0%} under both; Act., called of 4"),
        ],
        columns=3, size=note - 0.4,
    )

    save(fig, "milk")



# --------------------------------------------------------------------------- activity scores


def network_scores(args, corpus: Path, frame: pd.DataFrame, sequences: list[str], esm2: np.ndarray, esmc: np.ndarray, device: str) -> np.ndarray:
    # the Multi-task network of the score stage with the given sequences appended as unlabeled rows
    from ampx.models.data import TASKS
    from multitask import CLASSIFICATION_TASKS, build_targets, input_features, standardize_inputs, train_network

    split = frame["split"].to_numpy()
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "val")
    raw = np.vstack(
        [
            input_features(corpus, frame, args.transform, args.extra),
            milk_input_features(sequences, esmc, esm2, args.transform, args.extra),
        ]
    )
    features = standardize_inputs(raw, train, args.clip, device)
    del raw
    gc.collect()

    target, observed = build_targets(frame, True)
    target = np.vstack([target, np.zeros((len(sequences), target.shape[1]), dtype=np.float32)])
    observed = np.vstack([observed, np.zeros((len(sequences), observed.shape[1]), dtype=bool)])
    columns = [TASKS.index(task) for task in CLASSIFICATION_TASKS]

    network_args = argparse.Namespace(
        hidden=args.hidden,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        stopping=args.stopping,
    )

    averaged = np.zeros((len(target), len(TASKS)), dtype=np.float64)

    for seed in range(args.seeds):
        scores, value = train_network(features, target, observed, train, val, columns, network_args, device, seed)
        averaged += scores / args.seeds
        print(f"seed {seed}: best validation macro AUROC {value:.4f}", flush=True)

    return averaged

def tex_escape(text: str) -> str:
    for old, new in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        text = text.replace(old, new)

    return text


def shorten_name(text: str, limit: int = 30) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "."


def signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}".replace("-", "$-$")


def shortlist_table(shortlist: pd.DataFrame, analysis: dict, args) -> str:
    identity, profile = [], []

    for row in shortlist.itertuples(index=False):
        region = tex_escape(shorten_name(row.protein))

        if pd.notna(row.start):
            region += f" {int(row.start)}--{int(row.stop)}"

        identity.append(
            f"{row.rank} & \\texttt{{{row.sequence}}} & {region} & {int(row.length)} & {signed(row.net_charge, 1)} & {row.isoelectric_point:.1f} \\\\"
        )

        cells = [
            f"{row.rank}",
            f"{row.kr_percent:.0f}",
            f"{row.hydrophobic_moment:.2f}",
            signed(row.gravy, 2),
            f"{row.group_Hydrophobic:.0f}",
            f"{int(row.profile_met)}/4",
            f"{int(row.activities_called)}/4",
        ]

        for task in SCORED_TASKS:
            value = getattr(row, f"{task}_percentile_mean")
            text = f"{value:.2f}"

            if getattr(row, f"{task}_called"):
                text = f"\\textbf{{{text}}}"

            cells.append(f"\\cellcolor{{green!{int(round(45 * value))}}}{text}")

        profile.append(" & ".join(cells) + " \\\\")

    first = (
        "\\begin{tabular}{@{}r l l r r r@{}}\n\\toprule\n"
        "Rank & Sequence & Parent protein, residues & Length & Charge & pI \\\\\n\\midrule\n"
        + "\n".join(identity)
        + "\n\\bottomrule\n\\end{tabular}"
    )
    second = (
        "\\begin{tabular}{@{}r*{11}{c}@{}}\n\\toprule\n"
        "Rank & K+R (\\%) & Moment & GRAVY & Hyd. (\\%) & Prof. & Act. & "
        + " & ".join(HEAT_LABELS[task] for task in SCORED_TASKS)
        + " \\\\\n\\midrule\n"
        + "\n".join(profile)
        + "\n\\bottomrule\n\\end{tabular}"
    )

    consensus, eligible, component = analysis["consensus"], analysis["eligible"], analysis["component"]
    low, high = PROFILE_HYDROPHOBIC_PERCENT
    caption = (
        f"Shortlist of {len(shortlist)} milk peptides for wet-lab testing. Of {analysis['count']:,} peptides at similarity below "
        f"{SIMILARITY_CEILING} to the corpus, {int(consensus.sum())} lie in the top {100 * args.top:g}{PERCENT} of antibacterial score "
        f"under both the Oracle and the Ensemble scorer ({int(np.unique(component[consensus]).size)} homology components), and "
        f"{int(eligible.sum())} of them also lie outside the top {100 * args.hemolytic_top:g}{PERCENT} of hemolytic score under both. "
        f"The peptide with the highest mean antibacterial percentile of each component is listed. This selection is separate from the "
        f"candidate calls of Figure~\\ref{{fig:milk}}, which use the Multi-task score, a {100 * (1 - FPR_QUANTILE):g}{PERCENT} "
        f"false-positive threshold and a similarity ceiling of 0.7. Charge is without termini. Prof., criteria met of charge "
        f"$\\ge +{PROFILE_NET_CHARGE:.0f}$, hydrophobic residues {low:.0f} to {high:.0f}{PERCENT}, hydrophobic moment "
        f"$\\ge {PROFILE_MOMENT}$ and at most {PROFILE_CYSTEINES} cysteines, counted and never used as a filter. Act., activities called "
        f"of four, excluding hemolytic. Bact to Hemo are the mean percentile of the two scorers among the {analysis['count']:,} peptides, "
        f"shaded by value; bold marks the top {100 * args.call:g}{PERCENT} under both scorers. All results are model predictions; "
        f"none has been tested."
    )

    return (
        "\\begin{table}[htbp]\n\\caption{" + caption + "}\n\\label{tab:shortlist}\n\\centering\n\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n" + first + "\n\\vspace{6pt}\n" + second + "\n\\end{table}\n"
    )


def agreement_table(analysis: dict, args) -> str:
    lines = [
        f"{100 * row.fraction:g} & {row.n_each} & {row.n_shared} & {row.components_shared} & "
        f"{row.share:.2f} [{row.lo:.2f}, {row.hi:.2f}] & {row.chance:.2f} \\\\"
        for row in analysis["overlap"].itertuples(index=False)
    ]
    body = (
        "\\begin{tabular}{@{}r r r r c r@{}}\n\\toprule\n"
        "Top share (\\%) & Peptides per scorer & Shared & Components shared & Share shared [95\\% interval] & Chance \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}"
    )

    rho = analysis["spearman"]
    kind = analysis["kind"]
    homologs = int((kind == "antimicrobial").sum())
    caption = (
        f"Agreement of the Oracle and Ensemble antibacterial scores on the {analysis['count']:,} milk peptides. Share shared is the "
        f"fraction of one top set that is also in the other, and intervals resample homology components at similarity {args.identity} "
        f"({args.draws:,} draws). The Spearman correlation between the two scores is {rho[0]:.2f} [{rho[1]:.2f}, {rho[2]:.2f}]."
    )

    if homologs:
        caption += (
            f" For the {homologs} peptides homologous to antimicrobial regions of milk proteins against the unlabeled peptides, the AUROC "
            f"of the antibacterial score is {analysis['homolog'][('oracle', 'antimicrobial')]:.2f} for the Oracle and "
            f"{analysis['homolog'][('ensemble', 'antimicrobial')]:.2f} for the Ensemble."
        )

    return (
        "\\begin{table}[htbp]\n\\caption{" + caption + "}\n\\label{tab:agreement}\n\\centering\n\\small\n" + body + "\n\\end{table}\n"
    )


def run_tables(args) -> None:
    analysis = analyse(args)
    shortlist = analysis["shortlist"]

    if shortlist.empty:
        raise SystemExit("no peptide in both top sets lies outside the hemolytic exclusion; raise --hemolytic-top")

    output = ROOT / "paper" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    (output / "milk_shortlist.tex").write_text(shortlist_table(shortlist, analysis, args))
    (output / "milk_agreement.tex").write_text(agreement_table(analysis, args))
    print(f"wrote milk_shortlist.tex and milk_agreement.tex in {output}", flush=True)

def run_activity_scores(args) -> None:
    import torch

    from ampx.evaluation import load_frame, task_selection
    from ampx.models.data import TASKS
    from ampx.predictor import ESM2Encoder, ESMCEncoder, resolve_device

    pool_path = MILK / "pool.csv"

    if not pool_path.exists():
        raise SystemExit(f"missing {pool_path}; run the filter stage")

    pool = pd.read_csv(pool_path)
    sequences = pool["Sequence"].tolist()
    corpus = Path(args.corpus)
    device = resolve_device(args.device)
    frame = load_frame(corpus, ("sequence", "split", "component"))

    encoder = ESM2Encoder(device=device)
    esm2 = encoder.encode(sequences, batch_size=256)
    del encoder
    gc.collect()

    esmc_encoder = ESMCEncoder(device=device)
    esmc = esmc_encoder.encode(sequences, batch_size=128)
    del esmc_encoder
    gc.collect()

    if device != "cpu":
        torch.cuda.empty_cache()

    averaged = network_scores(args, corpus, frame, sequences, esm2, esmc, device)
    count = len(frame)

    scores = pd.DataFrame({"sequence": sequences, "max_sim_corpus": pool["max_corpus_similarity"].to_numpy()})
    thresholds = []

    for task in ACTIVITY_TASKS:
        column = TASKS.index(task)
        scores[f"score_{task}"] = averaged[count:, column]
        keep, labels, task_split = task_selection(frame, task, True)
        inactive = keep[(task_split == "val") & (labels == 0)]
        active = keep[(task_split == "val") & (labels == 1)]
        threshold = float("nan")
        true_positive_rate = float("nan")

        if len(inactive) >= MIN_VALIDATION_INACTIVES:
            threshold = float(np.quantile(averaged[inactive, column], FPR_QUANTILE))
            scores[f"above_{task}"] = scores[f"score_{task}"] > threshold

            if len(active):
                true_positive_rate = float((averaged[active, column] > threshold).mean())

            print(
                f"{task}: threshold {threshold:.4f}, validation inactives {len(inactive)}, validation actives above "
                f"{true_positive_rate:.3f}, milk sequences above {int(scores[f'above_{task}'].sum())} of {len(scores)}",
                flush=True,
            )
        else:
            print(f"{task}: no threshold, validation inactives {len(inactive)} (needs {MIN_VALIDATION_INACTIVES})", flush=True)

        thresholds.append(
            {
                "task": task,
                "threshold": threshold,
                "n_inactive": int(len(inactive)),
                "n_active": int(len(active)),
                "validation_tpr": true_positive_rate,
            }
        )

    metadata = pd.DataFrame(
        {
            "sequence": sequences,
            "protein": pool["Protein"].fillna("unknown").to_numpy(),
            "bioactive": pool["class"].to_numpy(),
        }
    )

    scores.to_csv(MILK / "peptidomics_scores.csv", index=False)
    pd.DataFrame(thresholds).to_csv(MILK / "peptidomics_thresholds.csv", index=False)
    metadata.to_csv(MILK / "peptidomics_metadata.csv", index=False)
    print(f"wrote peptidomics_scores.csv, peptidomics_thresholds.csv and peptidomics_metadata.csv in {MILK}", flush=True)


# --------------------------------------------------------------------------- activity figure

VIOLINS = (
    ("net_charge", "Net\ncharge"),
    ("charge_per_residue", "Charge\nper residue"),
    ("hydrophobic_moment", "Hydrophobic\nmoment"),
    ("isoelectric_point", "Isoelectric\npoint"),
)

# each panel: left, top offset from the row top, width, height, all in inches
LAYOUTS = {
    "wide": {
        "width": 17.5, "pitch": 2.1, "tall": 1.45, "protein_label": "parent proteins", "wrap_name": False, "strip_font": 14,
        "margin": (0.15, 0.0, 1.95, 1.45),
        "radar": (2.45, 0.22, 1.25, 1.25),
        "groups": (4.55, 0.0, 1.45, 1.45),
        "structure": (6.95, 0.0, 1.40, 1.45),
        "violin0": (9.05, 0.0, 0.60, 1.45),
        "violin1": (10.00, 0.0, 0.60, 1.45),
        "violin2": (10.95, 0.0, 0.60, 1.45),
        "violin3": (11.90, 0.0, 0.60, 1.45),
        "motif3": (13.10, 0.0, 0.95, 1.45),
        "motif5": (14.95, 0.0, 0.95, 1.45),
        "kr": (16.55, 0.0, 0.80, 1.45),
    },
    "compact": {
        "width": 7.2, "pitch": 5.75, "tall": 1.15, "protein_label": "proteins", "wrap_name": True, "strip_font": 12,
        "margin": (0.05, 0.0, 1.35, 1.15),
        "radar": (1.50, 0.15, 1.00, 1.00),
        "groups": (3.25, 0.0, 1.10, 1.15),
        "structure": (5.10, 0.0, 1.10, 1.15),
        "violin0": (0.70, 2.05, 0.80, 1.15),
        "violin1": (2.00, 2.05, 0.80, 1.15),
        "violin2": (3.30, 2.05, 0.80, 1.15),
        "violin3": (4.60, 2.05, 0.80, 1.15),
        "motif3": (0.95, 4.10, 1.25, 1.15),
        "motif5": (3.05, 4.10, 1.25, 1.15),
        "kr": (5.35, 4.10, 1.25, 1.15),
    },
}

HEADERS = (
    ("radar", "Amino acid\ncomposition (%)"),
    ("groups", "Residue\ngroups"),
    ("structure", "Secondary\nstructure"),
    ("violin0", VIOLINS[0][1]),
    ("violin1", VIOLINS[1][1]),
    ("violin2", VIOLINS[2][1]),
    ("violin3", VIOLINS[3][1]),
    ("motif3", "3-mers"),
    ("motif5", "5-mers"),
    ("kr", "K+R\ncontent"),
)


def panel_letters(has_structure: bool) -> dict:
    # panel A is the pipeline; the letters of the others follow the panels that are drawn
    names = ["radar"] + (["structure"] if has_structure else []) + ["violin0", "motif3", "kr"]

    return dict(zip(names, "BCDEF"))


def drawing_layout(compact: bool, has_structure: bool) -> dict:
    layout = LAYOUTS["compact" if compact else "wide"]

    if has_structure:
        return layout

    kept = {key: value for key, value in layout.items() if key != "structure"}

    if compact:
        kept["groups"] = (layout["groups"][0], layout["groups"][1], 2.40, layout["groups"][3])

        return kept

    # the panels right of the removed one spread over the freed width
    first = layout["radar"][0]
    last = layout["kr"][0] + layout["kr"][2]
    cut = layout["structure"][0]
    removed = layout["violin0"][0] - cut
    scale = (last - first) / (last - first - removed)

    for key in ("groups", "violin0", "violin1", "violin2", "violin3", "motif3", "motif5", "kr"):
        left, offset, size_x, size_y = layout[key]
        shifted = left - removed if left > cut else left
        kept[key] = (first + (shifted - first) * scale, offset, size_x * scale, size_y)

    # the violins and k-mer panels move left together; the K+R panel keeps its right edge and gains the width
    for key in ("violin0", "violin1", "violin2", "violin3", "motif3", "motif5"):
        left, offset, size_x, size_y = kept[key]
        kept[key] = (left - PANEL_SHIFT, offset, size_x, size_y)

    left, offset, size_x, size_y = kept["kr"]
    kept["kr"] = (left - PANEL_SHIFT, offset, size_x + PANEL_SHIFT, size_y)

    # residue-group bars move toward the radar; their right edge stays, so the bars get wider
    left, offset, size_x, size_y = kept["groups"]
    kept["groups"] = (left - GROUP_SHIFT, offset, size_x + GROUP_SHIFT, size_y)

    # 5-mer panel moves toward the 3-mer panel; the K+R panel again keeps its right edge
    left, offset, size_x, size_y = kept["motif5"]
    kept["motif5"] = (left - MOTIF_GAP_CUT, offset, size_x, size_y)

    left, offset, size_x, size_y = kept["kr"]
    kept["kr"] = (left - MOTIF_GAP_CUT, offset, size_x + MOTIF_GAP_CUT, size_y)

    return kept


def read_inputs(args) -> dict:
    files = {
        "scores": MILK / "peptidomics_scores.csv",
        "thresholds": MILK / "peptidomics_thresholds.csv",
        "metadata": MILK / "peptidomics_metadata.csv",
        "funnel": MILK / "funnel.csv",
    }

    for name, path in files.items():
        if not path.exists():
            stage = "filter" if name == "funnel" else "activity_scores"
            raise SystemExit(f"missing {name} file {path}; run the {stage} stage")

    scores = pd.read_csv(files["scores"]).drop_duplicates("sequence")
    thresholds = pd.read_csv(files["thresholds"]).set_index("task")
    metadata = pd.read_csv(files["metadata"])
    funnel = pd.read_csv(files["funnel"])

    # a sequence is counted once and assigned to its most frequent parent protein
    protein = metadata.groupby("sequence")["protein"].agg(lambda values: values.fillna("unknown").value_counts().idxmax())
    scores["protein"] = scores["sequence"].map(protein).fillna("unknown")

    if args.view == "novel":
        scores = scores[scores["max_sim_corpus"] < args.view_ceiling]
        view_text = f"below {args.view_ceiling:g}"
    else:
        view_text = "any"

    scores = scores.reset_index(drop=True)
    available = [task for task in ACTIVITY_TASKS if f"above_{task}" in scores.columns]
    print(f"view {args.view}: {len(scores)} distinct sequences", flush=True)
    print(f"tasks with an above column: {available}", flush=True)
    print(f"tasks without an above column: {[task for task in ACTIVITY_TASKS if task not in available]}", flush=True)

    structure = None
    path = Path(args.structure)

    if path.exists():
        frame = pd.read_csv(path).drop_duplicates("sequence").set_index("sequence")

        for column in ("ss_helix", "ss_sheet", "ss_coil"):
            if column not in frame.columns:
                raise SystemExit(f"{path} lacks the column {column}")

        absent = ~scores["sequence"].isin(frame.index)

        if absent.any():
            raise SystemExit(f"{path} lacks {int(absent.sum())} sequences of the view")

        structure = {key: frame.loc[scores["sequence"], key].to_numpy(dtype=float) for key in ("ss_helix", "ss_coil", "ss_sheet")}
    else:
        print(f"{path} not found: panel C will read 'structure not computed'", flush=True)

    return {"scores": scores, "thresholds": thresholds, "funnel": funnel, "structure": structure, "view_text": view_text}


def property_table(sequences: list[str]) -> pd.DataFrame:
    properties, _ = peptide_properties(sequences)

    return properties


def composition_table(sequences: list[str]) -> pd.DataFrame:
    return pd.DataFrame(residue_percent(sequences), columns=list(RESIDUE_ORDER))


def motif_table(sequences: list[str], candidate: np.ndarray, proteins: np.ndarray, length: int, count: int) -> list:
    # (k-mer, share of candidate sequences containing it, share of the rest, parent proteins containing it), most frequent first
    def kmers(sequence):
        return {sequence[start : start + length] for start in range(len(sequence) - length + 1)}

    candidate_count, rest_count, parents = {}, {}, {}
    candidate_rows, rest_rows = np.flatnonzero(candidate), np.flatnonzero(~candidate)

    for row in candidate_rows:
        for motif in kmers(sequences[row]):
            candidate_count[motif] = candidate_count.get(motif, 0) + 1
            parents.setdefault(motif, set()).add(proteins[row])

    for row in rest_rows:
        for motif in kmers(sequences[row]):
            rest_count[motif] = rest_count.get(motif, 0) + 1

    ordered = sorted(candidate_count.items(), key=lambda item: (-item[1], item[0]))[:count]

    return [
        (motif, number / len(candidate_rows), rest_count.get(motif, 0) / max(len(rest_rows), 1), len(parents[motif]))
        for motif, number in ordered
    ]


def cluster_bootstrap(matrix: np.ndarray, candidate: np.ndarray, clusters: np.ndarray, draws: int, seed: int):
    # difference of means, candidates minus rest, with an interval that resamples whole clusters (parent proteins)
    from ampx.evaluation import component_groups, percentile_interval, resample_components

    groups = component_groups(clusters)
    generator = np.random.default_rng(seed)
    gaps = []

    for _ in range(draws):
        index = resample_components(groups, generator)
        chosen = candidate[index]

        if not chosen.any() or chosen.all():
            continue

        values = matrix[index]
        gaps.append(values[chosen].mean(axis=0) - values[~chosen].mean(axis=0))

    gaps = np.asarray(gaps)
    bounds = np.array([percentile_interval(gaps[:, column]) for column in range(matrix.shape[1])])
    observed = matrix[candidate].mean(axis=0) - matrix[~candidate].mean(axis=0)

    return observed, bounds[:, 0], bounds[:, 1], len(gaps)


def activity_result(task: str, context: dict, args) -> dict:
    table, thresholds = context["table"], context["thresholds"]
    n_inactive = int(thresholds.loc[task, "n_inactive"]) if task in thresholds.index else 0
    threshold = float(thresholds.loc[task, "threshold"]) if task in thresholds.index else float("nan")
    column = f"above_{task}"
    base = {"task": task, "scored": len(table), "n_inactive": n_inactive, "threshold": threshold}

    if column not in table.columns or not np.isfinite(threshold):
        return {**base, "status": "none", "message": f"no threshold, n validation inactives = {n_inactive}"}

    candidate = table[column].fillna(False).to_numpy(dtype=bool)
    number = int(candidate.sum())
    base.update({"candidates": number, "rest": int(len(table) - number)})

    if number < MIN_CANDIDATES:
        return {**base, "status": "few", "message": f"n candidates = {number}"}

    names = ["net_charge", "charge_per_residue", "hydrophobic_moment", "isoelectric_point", "kr_percent"]
    names += [f"group_{name}" for name, _ in GROUPS]
    matrix = context["properties"][names].to_numpy(dtype=float)

    if context["structure"] is not None:
        structure_names = ["ss_helix", "ss_coil", "ss_sheet"]
        names = names + structure_names
        matrix = np.column_stack([matrix] + [context["structure"][key] for key in structure_names])

    observed, lower, upper, effective = cluster_bootstrap(matrix, candidate, context["clusters"], args.draws, args.seed)
    difference = pd.DataFrame(
        {
            "candidate": matrix[candidate].mean(axis=0),
            "rest": matrix[~candidate].mean(axis=0),
            "difference": observed,
            "lo": lower,
            "hi": upper,
        },
        index=names,
    )
    difference["significant"] = (difference["lo"] > 0) | (difference["hi"] < 0)

    composition = context["composition"]
    radar = pd.DataFrame({"candidate": composition[candidate].mean(), "rest": composition[~candidate].mean()})
    motifs = {length: motif_table(context["sequences"], candidate, context["proteins"], length, 50) for length in (3, 5)}
    cationic = float(
        np.mean([sum(motif.count(residue) for residue in "KR") >= 2 for found in motifs.values() for motif, *_ in found])
    )

    print(f"{task}: {number} candidates, {int(difference['significant'].sum())} of {len(difference)} differences exclude zero", flush=True)

    return {
        **base,
        "status": "ok",
        "candidate": candidate,
        "proteins": int(np.unique(context["clusters"][candidate]).size),
        "difference": difference,
        "radar": radar,
        "motifs": {length: found[:8] for length, found in motifs.items()},
        "cationic_share": cationic,
        "draws": effective,
    }


def value_rows(result: dict, letters: dict) -> list:
    task = result["task"]
    rows = [
        {"task": task, "panel": "A", "item": "scored", "value": result["scored"]},
        {"task": task, "panel": "A", "item": "threshold", "value": result["threshold"]},
        {"task": task, "panel": "A", "item": "validation inactives", "value": result["n_inactive"]},
    ]

    if result["status"] == "none":
        return rows

    rows += [
        {"task": task, "panel": "A", "item": "candidates", "value": result["candidates"]},
        {"task": task, "panel": "A", "item": "rest", "value": result["rest"]},
        {"task": task, "panel": "A", "item": "candidate share", "value": result["candidates"] / result["scored"]},
    ]

    if result["status"] != "ok":
        return rows

    rows.append({"task": task, "panel": "A", "item": "parent proteins of candidates", "value": result["proteins"]})

    for letter, row in result["radar"].iterrows():
        rows.append({"task": task, "panel": letters["radar"], "item": f"composition {letter} (%)", "value": row["candidate"], "rest": row["rest"]})

    for name, row in result["difference"].iterrows():
        if name.startswith("group_"):
            panel = letters["radar"]
        elif name.startswith("ss_"):
            panel = letters["structure"]
        elif name == "kr_percent":
            panel = letters["kr"]
        else:
            panel = letters["violin0"]

        rows.append(
            {
                "task": task, "panel": panel, "item": name, "value": row["candidate"], "rest": row["rest"],
                "difference": row["difference"], "lo": row["lo"], "hi": row["hi"], "excludes_zero": bool(row["significant"]),
            }
        )

    for length, found in result["motifs"].items():
        for motif, share, rest_share, parents in found:
            rows.append({"task": task, "panel": letters["motif3"], "item": f"{length}-mer {motif}", "value": share, "rest": rest_share, "proteins": parents})

    rows.append(
        {"task": task, "panel": letters["motif3"], "item": "share of the 50 most frequent 3-mers and 5-mers with at least 2 K or R", "value": result["cationic_share"]}
    )

    return rows


def write_caption(path: Path, context: dict, results: dict, args) -> None:
    letters = context["letters"]
    funnel = context["funnel"].set_index("step")
    all_rows = int(funnel.loc["all rows", "peptides"])
    distinct = int(funnel.loc["deduplicated", "peptides"])
    scored = len(context["table"])
    drawn = [result for result in results.values() if result["status"] == "ok"]
    empty = [ACTIVITY_LABELS[result["task"]] for result in results.values() if result["status"] != "ok"]

    if args.colors == "shared":
        colors = "Candidates are coral red and the rest teal; the strip at the left carries the color of the activity. "
    else:
        colors = "Candidates carry the color of the activity and the rest is gray. "

    text = (
        f"Mining of milk peptides released in the stomach of preterm infants. "
        f"(A) Pipeline: {all_rows:,} peptidomics identifications, {distinct:,} distinct sequences of 8 to 50 standard residues, "
        f"{scored:,} with similarity to the corpus {context['view_text']}, scored by the multi-task predictor. Candidates of an activity are "
        f"the sequences above the {100 * FPR_QUANTILE:.0f}th percentile of the scores of its validation inactives (a "
        f"{100 * (1 - FPR_QUANTILE):.0f}{PERCENT} false-positive rate); the rest are the other sequences. "
        f"({letters['radar']}) Amino acid composition (radar) and the five residue groups (bars) of candidates and rest. "
    )

    if context["structure"] is not None:
        text += f"({letters['structure']}) Helix, coil and sheet fractions. "

    text += (
        f"({letters['violin0']}) Net charge, charge per residue, hydrophobic moment and isoelectric point. "
        f"({letters['motif3']}) The eight most frequent 3-mers and 5-mers among candidates: bars give the share of candidate sequences "
        "containing the k-mer, open circles the share of the rest. "
        f"({letters['kr']}) Density of the K+R content per peptide, with the mean as a dashed line. "
        f"Asterisks mark differences, candidates minus rest, whose 95{PERCENT} interval over {args.draws:,} bootstrap draws that resample parent "
        f"proteins excludes zero. {colors}"
    )

    for result in drawn:
        row = result["difference"]
        text += (
            f"{ACTIVITY_LABELS[result['task']]}: {result['candidates']:,} candidates of {result['scored']:,} "
            f"({100 * result['candidates'] / result['scored']:.1f}{PERCENT}); net charge {row.loc['net_charge', 'candidate']:+.1f} against "
            f"{row.loc['net_charge', 'rest']:+.1f}, hydrophobic moment {row.loc['hydrophobic_moment', 'candidate']:.2f} against "
            f"{row.loc['hydrophobic_moment', 'rest']:.2f}, isoelectric point {row.loc['isoelectric_point', 'candidate']:.1f} against "
            f"{row.loc['isoelectric_point', 'rest']:.1f}, K+R content {row.loc['kr_percent', 'candidate']:.1f}{PERCENT} against "
            f"{row.loc['kr_percent', 'rest']:.1f}{PERCENT}. "
        )

    if empty:
        text += "No row is drawn for " + ", ".join(empty) + " (too few candidates or no threshold). "

    text += "All results are model predictions; none has been tested."
    path.write_text("\\begin{figure}\n\\caption{" + text + "}\n\\end{figure}\n")


def drawing_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": BASE_FONT,
            "axes.labelsize": BASE_FONT,
            "axes.titlesize": BASE_FONT,
            "xtick.labelsize": BASE_FONT,
            "ytick.labelsize": BASE_FONT,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "axes.labelpad": 2.0,
            "pdf.fonttype": 42,
            "lines.linewidth": 0.9,
        }
    )


def inch_axes(fig, width: float, height: float):
    def place(left, top, size_x, size_y, **kwargs):
        return fig.add_axes([left / width, 1.0 - (top + size_y) / height, size_x / width, size_y / height], **kwargs)

    return place


def horizontal_boxplot(ax, data: list, positions: list, colors: list, width: float) -> None:
    options = {"positions": positions, "widths": width, "patch_artist": True, "showfliers": False, "whis": (5, 95), "manage_ticks": False}

    try:
        parts = ax.boxplot(data, orientation="horizontal", **options)
    except TypeError:
        parts = ax.boxplot(data, vert=False, **options)

    for box, color in zip(parts["boxes"], colors):
        box.set_facecolor(color)
        box.set_edgecolor(INK_COLOR)
        box.set_linewidth(0.6)

    for median in parts["medians"]:
        median.set_color(INK_COLOR)
        median.set_linewidth(1.0)

    for key in ("whiskers", "caps"):
        for line in parts[key]:
            line.set_color(INK_COLOR)
            line.set_linewidth(0.6)


def draw_violin(ax, values: np.ndarray, candidate: np.ndarray, candidate_color: str, rest_color: str, significant: bool) -> None:
    data = [values[~candidate], values[candidate]]
    colors = (rest_color, candidate_color)

    try:
        parts = ax.violinplot(data, positions=[0, 1], widths=0.85, showextrema=False, showmedians=True)

        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(INK_COLOR)
            body.set_linewidth(0.4)
            body.set_alpha(0.8)

        parts["cmedians"].set_color(INK_COLOR)
        parts["cmedians"].set_linewidth(0.9)
    except (np.linalg.LinAlgError, ValueError):
        for position, group, color in zip((0, 1), data, colors):
            ax.hlines(np.median(group), position - 0.3, position + 0.3, color=color, linewidth=2.0)

    lower, upper = float(min(map(np.min, data))), float(max(map(np.max, data)))
    span = max(upper - lower, 1e-9)

    if significant:
        ax.text(0.5, upper + 0.02 * span, "*", ha="center", va="bottom", fontsize=BASE_FONT)

    ax.set_ylim(lower - 0.05 * span, upper + 0.18 * span)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([])


def draw_headers(fig, layout: dict, top: float, width: float, height: float, letters: dict) -> None:
    for key, title in HEADERS:
        if key not in layout:
            continue

        left, offset, _, _ = layout[key]
        # the radar axes start lower than the other panels; its title is anchored at the row top and raised
        header_offset = -RADAR_HEADER_RAISE if key == "radar" else offset
        x = left / width
        y = 1.0 - (top + header_offset - 0.07) / height
        letter = letters.get(key)

        if letter:
            # va="bottom" includes the font descent, so the larger letter is lowered by the descent difference to share the title baseline
            drop = 0.2 * (PANEL_LETTER_FONT - TITLE_FONT) / 72.0 / height
            fig.text(x, y - drop, letter, fontsize=PANEL_LETTER_FONT, fontweight="bold", ha="left", va="bottom")
            x += PANEL_LETTER_ADVANCE / width

        fig.text(x, y, title, fontsize=TITLE_FONT, ha="left", va="bottom", linespacing=1.0)

ICON_BLUE, ICON_GREEN, ICON_ORANGE, ICON_OUTLINE, ICON_GREY = "#286AA5", "#2a9d5c", "#d98e04", "#4a4a4a", "#8A8A8A"
ICON_PEPTIDE = ("#f4c06a", "#7fb0e3", "#e88a8a", "#9bd29b")


def icon_tint(color, strength):
    from matplotlib.colors import to_rgb

    return tuple(1.0 - (1.0 - value) * strength for value in to_rgb(color))


def draw_milk_icon(ax, cx, y0):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

    bx = cx - 0.62
    ax.add_patch(Polygon([(bx - 0.07, y0 + 0.14), (bx + 0.07, y0 + 0.14), (bx + 0.045, y0 + 0.03), (bx - 0.045, y0 + 0.03)], closed=True,
                         facecolor=icon_tint(ICON_ORANGE, 0.25), edgecolor=ICON_ORANGE, linewidth=0.6))
    ax.add_patch(Rectangle((bx - 0.2, y0 + 0.14), 0.4, 0.07, facecolor=icon_tint(ICON_BLUE, 0.45), edgecolor=ICON_BLUE, linewidth=0.6))
    ax.add_patch(FancyBboxPatch((bx - 0.17, y0 + 0.21), 0.34, 0.37, boxstyle="round,pad=0,rounding_size=0.05", facecolor="white",
                                edgecolor=ICON_BLUE, linewidth=0.7))
    ax.add_patch(Rectangle((bx - 0.165, y0 + 0.36), 0.33, 0.215, facecolor=icon_tint(ICON_BLUE, 0.16), edgecolor="none"))

    for step in range(3):
        y = y0 + 0.30 + 0.09 * step
        ax.plot([bx + 0.04, bx + 0.14], [y, y], color=ICON_BLUE, linewidth=0.4, zorder=2)

    ax.add_patch(FancyArrowPatch((bx + 0.30, y0 + 0.38), (bx + 0.50, y0 + 0.38), arrowstyle="-|>", mutation_scale=7, color=ICON_OUTLINE, linewidth=0.7))

    sx, base = cx - 0.08, y0 + 0.58
    ax.plot([sx, sx + 0.95], [base, base], color=ICON_OUTLINE, linewidth=0.6, zorder=2)

    for dx, height in ((0.05, 0.14), (0.15, 0.30), (0.25, 0.18), (0.37, 0.50), (0.47, 0.22), (0.58, 0.38), (0.68, 0.16), (0.78, 0.26), (0.88, 0.12)):
        ax.plot([sx + dx, sx + dx], [base, base - height], color=ICON_ORANGE if dx == 0.37 else ICON_BLUE, linewidth=1.0, solid_capstyle="butt", zorder=2)


def draw_filter_icon(ax, cx, y0):
    from matplotlib.patches import Circle, Ellipse, Polygon

    half, stem = 0.62, 0.07
    rim, stem_top, stem_bottom = y0 + 0.17, y0 + 0.40, y0 + 0.53

    ax.add_patch(Polygon([(cx - half, rim), (cx + half, rim), (cx + stem, stem_top), (cx + stem, stem_bottom), (cx - stem, stem_bottom), (cx - stem, stem_top)],
                         closed=True, facecolor=icon_tint(ICON_BLUE, 0.2), edgecolor=ICON_BLUE, linewidth=0.8, joinstyle="round"))
    ax.add_patch(Ellipse((cx, rim), 2.0 * half, 0.10, facecolor=icon_tint(ICON_BLUE, 0.4), edgecolor=ICON_BLUE, linewidth=0.7))

    mesh = y0 + 0.30
    width = half + (mesh - rim) / (stem_top - rim) * (stem - half)
    ax.plot([cx - width, cx + width], [mesh, mesh], color=ICON_ORANGE, linewidth=0.8, linestyle=(0, (2.0, 1.2)), zorder=2)

    for dx, color in ((-0.38, ICON_PEPTIDE[0]), (-0.13, ICON_PEPTIDE[1]), (0.13, ICON_PEPTIDE[2]), (0.38, ICON_PEPTIDE[3])):
        ax.add_patch(Circle((cx + dx, y0 + 0.06), 0.045, facecolor=color, edgecolor=ICON_OUTLINE, linewidth=0.3, zorder=3))

    for dx, color in ((-0.26, ICON_PEPTIDE[2]), (0.0, ICON_PEPTIDE[0]), (0.26, ICON_PEPTIDE[1])):
        ax.add_patch(Circle((cx + dx, y0 + 0.235), 0.045, facecolor=color, edgecolor=ICON_OUTLINE, linewidth=0.3, zorder=3))

    for y in (y0 + 0.465, y0 + 0.585):
        ax.add_patch(Circle((cx, y), 0.045, facecolor=ICON_PEPTIDE[3], edgecolor=ICON_GREEN, linewidth=0.9, zorder=3))


def draw_model_icon(ax, cx, y0):
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    yc = y0 + 0.30
    xs = [cx - 0.83 + 0.30 * index for index in range(3)]
    layers = [[yc + pitch * (index - (count - 1) / 2) for index in range(count)] for count, pitch in zip((3, 4, 3), (0.18, 0.15, 0.18))]

    for column in range(2):
        for ya in layers[column]:
            for yb in layers[column + 1]:
                ax.plot([xs[column], xs[column + 1]], [ya, yb], color="#a9b4c0", linewidth=0.3, zorder=2)

    for x, ys in zip(xs, layers):
        for y in ys:
            ax.add_patch(Circle((x, y), 0.045, facecolor=icon_tint(ICON_BLUE, 0.35), edgecolor=ICON_BLUE, linewidth=0.5, zorder=3))

    ax.add_patch(FancyArrowPatch((cx - 0.15, yc), (cx + 0.03, yc), arrowstyle="-|>", mutation_scale=7, color=ICON_OUTLINE, linewidth=0.7))

    for row, states in enumerate(("AIAUU", "AIUAU")):
        for column, state in enumerate(states):
            rect = (cx + 0.08 + 0.17 * column, yc - 0.18 + 0.20 * row)

            if state == "A":
                ax.add_patch(Rectangle(rect, 0.13, 0.16, facecolor=ICON_GREEN, edgecolor=ICON_OUTLINE, linewidth=0.4))
            elif state == "I":
                ax.add_patch(Rectangle(rect, 0.13, 0.16, facecolor="#c0392b", edgecolor=ICON_OUTLINE, linewidth=0.4))
            else:
                ax.add_patch(Rectangle(rect, 0.13, 0.16, facecolor="white", edgecolor="#bdbdbd", linewidth=0.4, hatch="////"))


def draw_ranking_icon(ax, cx, y0):
    import math

    from matplotlib.patches import FancyBboxPatch, Polygon

    left, threshold = cx - 0.75, 0.82
    ax.plot([left + threshold, left + threshold], [y0 + 0.03, y0 + 0.60], color=ICON_GREY, linewidth=0.5, linestyle=(0, (2.0, 1.5)), zorder=2)

    for index, length in enumerate((1.25, 1.0, 0.7, 0.45)):
        color = ICON_GREEN if length > threshold else ICON_BLUE
        ax.add_patch(FancyBboxPatch((left, y0 + 0.07 + 0.14 * index), length, 0.10, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=icon_tint(color, 0.6 if color == ICON_GREEN else 0.4), edgecolor=color, linewidth=0.5))

    sx, sy = left + 1.39, y0 + 0.12
    points = []

    for index in range(10):
        radius = 0.10 if index % 2 == 0 else 0.042
        angle = math.radians(90 + index * 36)
        points.append((sx + radius * math.cos(angle), sy - radius * math.sin(angle)))

    ax.add_patch(Polygon(points, closed=True, facecolor=ICON_ORANGE, edgecolor="none"))


def draw_pipeline(fig, place, width: float, top: float, height: float, context: dict, results: dict) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    span = width - 0.30
    ax = place(0.15, top, span, height)
    ax.set_xlim(0, span)
    ax.set_ylim(height, 0)
    ax.axis("off")

    funnel = context["funnel"].set_index("step")
    all_rows = int(funnel.loc["all rows", "peptides"])
    distinct = int(funnel.loc["deduplicated", "peptides"])
    scored = len(context["table"])
    with_threshold = sum(result["status"] != "none" for result in results.values())
    view = context["view_text"].replace("below ", "")

    boxes = (
        ("Milk peptidomics", f"{all_rows:,} identifications\npreterm infant stomachs", draw_milk_icon),
        ("Filter", f"8 to 50 residues\nsimilarity below {view}\n{distinct:,} distinct, {scored:,} kept", draw_filter_icon),
        ("Multi-task predictor", f"ESM C and ESM-2\nten tasks\n{scored:,} scored", draw_model_icon),
        ("Candidates per activity", f"5% false-positive-rate\nthreshold\n{with_threshold} of {len(results)} activities", draw_ranking_icon),
    )

    ax.text(0.0, 0.22, "A", fontsize=PIPELINE_HEADING_FONT + 4.0, fontweight="bold", ha="left", va="center")
    ax.text(0.45, 0.22, "Pipeline", fontsize=PIPELINE_HEADING_FONT, ha="left", va="center")

    box_width, box_pitch = 2.35, 2.65

    for index, (title, body, draw_icon) in enumerate(boxes):
        x = 0.30 + index * box_pitch
        ax.add_patch(
            FancyBboxPatch((x, 0.60), box_width, 1.60, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#F2F2F2",
                           edgecolor="#8A8A8A", linewidth=0.8)
        )
        draw_icon(ax, x + box_width / 2, 0.70)
        ax.text(x + box_width / 2, 1.46, title, ha="center", va="center", fontsize=PIPELINE_TITLE_FONT, fontweight="bold")
        ax.text(x + box_width / 2, 1.60, body, ha="center", va="top", fontsize=PIPELINE_FONT, linespacing=1.15, color=INK_COLOR)

        if index:
            ax.add_patch(
                FancyArrowPatch((x - 0.28, 1.40), (x - 0.03, 1.40), arrowstyle="-|>", mutation_scale=11, color=INK_COLOR, linewidth=0.9)
            )

    left = 11.00
    columns = {"scored": 14.50, "candidates": 16.00, "share": 17.15}
    ax.text(left, 0.32, "Activity", ha="left", va="center", fontsize=TABLE_TITLE_FONT, fontweight="bold")

    for key, label in (("scored", "Scored"), ("candidates", "Candidates"), ("share", "Share")):
        ax.text(columns[key], 0.32, label, ha="right", va="center", fontsize=TABLE_TITLE_FONT, fontweight="bold")

    ax.plot([left, columns["share"]], [0.54, 0.54], color=INK_COLOR, linewidth=0.7)

    listed = [task for task, result in results.items() if result["status"] != "none"]

    for index, task in enumerate(listed):
        result = results[task]
        y = 0.82 + 0.29 * index
        ax.add_patch(Rectangle((left, y - 0.085), 0.17, 0.17, facecolor=ACTIVITY_COLORS[task], edgecolor=INK_COLOR, linewidth=0.4))
        ax.text(left + 0.30, y, ACTIVITY_LABELS[task], ha="left", va="center", fontsize=TABLE_FONT)
        ax.text(columns["scored"], y, f"{result['scored']:,}", ha="right", va="center", fontsize=TABLE_FONT)
        ax.text(columns["candidates"], y, f"{result['candidates']:,}", ha="right", va="center", fontsize=TABLE_FONT)
        ax.text(columns["share"], y, f"{100 * result['candidates'] / result['scored']:.1f}%", ha="right", va="center", fontsize=TABLE_FONT)


def draw_row(fig, place, layout: dict, top: float, result: dict, context: dict, scheme: str) -> None:
    task = result["task"]
    candidate_color = CANDIDATE_COLOR if scheme == "shared" else ACTIVITY_COLORS[task]
    rest_color = REST_COLOR if scheme == "shared" else NEUTRAL_COLOR
    candidate = result["candidate"]
    difference = result["difference"]
    properties = context["properties"]

    # name strip
    left, offset, size_x, size_y = layout["margin"]
    strip = place(left, top + offset, size_x, size_y)
    strip.axis("off")
    strip.set_xlim(0, 1)
    strip.set_ylim(1, 0)
    strip.add_patch(plt_rectangle((0.0, 0.0), 0.035, 1.0, ACTIVITY_COLORS[task]))

    name = ACTIVITY_LABELS[task]
    name_lines = 1

    if layout["wrap_name"] and "-" in name:
        name = name.replace("-", "-\n")
        name_lines = 2

    strip_font = layout["strip_font"]
    line = 1.25 * (strip_font + 1.0) / 72.0 / size_y
    counts = (
        f"{result['candidates']:,} candidates\n{100 * result['candidates'] / result['scored']:.1f}% of {result['scored']:,}"
        f"\n{result['proteins']:,} {layout['protein_label']}"
    )
    strip.text(0.09, 0.04, name, ha="left", va="top", fontsize=strip_font + 1.0, fontweight="bold", linespacing=1.1)
    strip.text(0.09, 0.04 + line * name_lines + 0.03, counts, ha="left", va="top", fontsize=strip_font, color=INK_COLOR, linespacing=1.3)

    # composition radar
    left, offset, size_x, size_y = layout["radar"]
    radar = place(left, top + offset, size_x, size_y, projection="polar")
    angles = np.linspace(0.0, 2.0 * np.pi, len(RESIDUE_ORDER), endpoint=False)
    closed = np.append(angles, angles[0])
    table = result["radar"].loc[list(RESIDUE_ORDER)]

    for column, color in (("rest", rest_color), ("candidate", candidate_color)):
        values = np.append(table[column].to_numpy(), table[column].iloc[0])
        radar.plot(closed, values, color=color, linewidth=1.0)
        radar.fill(closed, values, color=color, alpha=0.25)

    radar.set_theta_zero_location("N")
    radar.set_theta_direction(-1)
    radar.set_xticks(angles)
    radar.set_xticklabels(list(RESIDUE_ORDER), fontsize=BASE_FONT)
    radar.tick_params(axis="x", pad=1)
    radar.set_yticklabels([])
    radar.set_ylim(0, float(table.to_numpy().max()) * 1.05)
    radar.grid(color=RULE_COLOR, linewidth=0.5)
    radar.spines["polar"].set_visible(False)

    # residue groups
    left, offset, size_x, size_y = layout["groups"]
    ax = place(left, top + offset, size_x, size_y)
    names = [f"group_{name}" for name, _ in GROUPS]
    positions = np.arange(len(GROUPS))
    ax.bar(positions - 0.2, difference.loc[names, "rest"], 0.38, color=rest_color, edgecolor=INK_COLOR, linewidth=0.4)
    ax.bar(positions + 0.2, difference.loc[names, "candidate"], 0.38, color=candidate_color, edgecolor=INK_COLOR, linewidth=0.4)
    peak = float(difference.loc[names, ["candidate", "rest"]].to_numpy().max())

    for position, group in zip(positions, names):
        if difference.loc[group, "significant"]:
            high = max(difference.loc[group, "candidate"], difference.loc[group, "rest"])
            ax.text(position, high + 0.02 * peak, "*", ha="center", va="bottom", fontsize=BASE_FONT)

    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_LABELS, rotation=45 if size_x < 1.3 else 0, ha="right" if size_x < 1.3 else "center")
    ax.set_ylim(0, peak * 1.2)
    ax.set_ylabel("% of residues")

    # secondary structure, drawn only when the structure file was read
    if "structure" in layout:
        left, offset, size_x, size_y = layout["structure"]
        ax = place(left, top + offset, size_x, size_y)
        keys = (("ss_helix", "Helix"), ("ss_coil", "Coil"), ("ss_sheet", "Sheet"))
        data, places, colors = [], [], []

        for index, (key, _) in enumerate(keys):
            values = context["structure"][key]
            data += [values[~candidate], values[candidate]]
            places += [3 * index, 3 * index + 1]
            colors += [rest_color, candidate_color]

        horizontal_boxplot(ax, data, places, colors, 0.8)
        limit = max(float(np.percentile(group, 95)) for group in data)
        limit = limit * 1.2 if limit > 0 else 1.0
        ax.set_xlim(0, limit)
        ax.set_ylim(7.6, -0.8)
        ax.set_yticks([0.5, 3.5, 6.5])
        ax.set_yticklabels([label for _, label in keys])

        for index, (key, _) in enumerate(keys):
            if difference.loc[key, "significant"]:
                ax.text(0.97 * limit, 3 * index + 0.5, "*", ha="right", va="center", fontsize=BASE_FONT)

    # violins
    for index, (name, _) in enumerate(VIOLINS):
        left, offset, size_x, size_y = layout[f"violin{index}"]
        ax = place(left, top + offset, size_x, size_y)
        draw_violin(ax, properties[name].to_numpy(dtype=float), candidate, candidate_color, rest_color, bool(difference.loc[name, "significant"]))

    # most frequent k-mers of the candidates
    for length in (3, 5):
        left, offset, size_x, size_y = layout[f"motif{length}"]
        ax = place(left, top + offset, size_x, size_y)
        found = result["motifs"][length]
        rows = np.arange(len(found))
        shares = np.array([100.0 * item[1] for item in found])
        ax.barh(rows, shares, height=0.65, color=candidate_color, edgecolor=INK_COLOR, linewidth=0.4)
        ax.set_yticks(rows)
        ax.set_yticklabels([item[0] for item in found], family="monospace", fontsize=BASE_FONT)
        ax.set_ylim(len(found) - 0.4, -0.6)
        ax.set_xlim(0, float(shares.max()) * 1.08)
        ax.set_xlabel("% of sequences")
        ax.locator_params(axis="x", nbins=3)

    # K+R density
    left, offset, size_x, size_y = layout["kr"]
    ax = place(left, top + offset, size_x, size_y)
    values = properties["kr_percent"].to_numpy(dtype=float)
    edges = np.linspace(0.0, max(float(values.max()), 1.0), 21)

    for group, color in ((~candidate, rest_color), (candidate, candidate_color)):
        density, _ = np.histogram(values[group], bins=edges, density=True)
        ax.stairs(density, edges, color=color, fill=True, alpha=0.25)
        ax.stairs(density, edges, color=color, linewidth=1.2)
        ax.axvline(values[group].mean(), color=color, linestyle="--", linewidth=1.0)

    if difference.loc["kr_percent", "significant"]:
        ax.text(0.97, 0.97, "*", transform=ax.transAxes, ha="right", va="top", fontsize=BASE_FONT)

    ax.set_xlim(0, edges[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("K+R (%)")
    ax.set_ylabel("density")
    ax.locator_params(axis="both", nbins=3)


def plt_rectangle(xy: tuple, width: float, height: float, color: str):
    from matplotlib.patches import Rectangle

    return Rectangle(xy, width, height, facecolor=color, edgecolor="none")


def build_figure(context: dict, results: dict, scheme: str, compact: bool):
    import textwrap

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    drawing_style()
    layout = drawing_layout(compact, context["structure"] is not None)
    letters = context["letters"]
    width = layout["width"]
    drawn = [task for task, result in results.items() if result["status"] == "ok"]
    pipeline_height = 0.0 if compact else 2.30
    start = pipeline_height + 0.85

    height = start + len(drawn) * layout["pitch"]

    fig = plt.figure(figsize=(width, height))
    place = inch_axes(fig, width, height)

    if not compact:
        draw_pipeline(fig, place, width, 0.0, pipeline_height, context, results)

    draw_headers(fig, layout, start, width, height, letters)

    for index, task in enumerate(drawn):
        top = start + index * layout["pitch"]

        if index:
            y = 1.0 - (top - 0.12) / height
            fig.add_artist(Line2D([0.15 / width, 1.0 - 0.15 / width], [y, y], color=RULE_COLOR, linewidth=0.6, transform=fig.transFigure))

        draw_row(fig, place, layout, top, results[task], context, scheme)

    if scheme == "shared":
        fig.legend(
            handles=[
                Patch(facecolor=CANDIDATE_COLOR, edgecolor=INK_COLOR, linewidth=0.4, label="Candidates"),
                Patch(facecolor=REST_COLOR, edgecolor=INK_COLOR, linewidth=0.4, label="Rest"),
            ],
            loc="lower left", bbox_to_anchor=(0.15 / width, 0.06 / height), ncol=2, frameon=False, fontsize=BASE_FONT,
        )

    return fig


def run_activity_figure(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inputs = read_inputs(args)
    table = inputs["scores"]
    sequences = table["sequence"].tolist()
    proteins = table["protein"].to_numpy()

    context = {
        "table": table,
        "thresholds": inputs["thresholds"],
        "funnel": inputs["funnel"],
        "structure": inputs["structure"],
        "view_text": inputs["view_text"],
        "letters": panel_letters(inputs["structure"] is not None),
        "sequences": sequences,
        "proteins": proteins,
        "clusters": pd.factorize(proteins)[0],
        "properties": property_table(sequences),
        "composition": composition_table(sequences),
        "draws": args.draws,
    }
    print(f"{len(sequences)} sequences from {len(set(proteins))} parent proteins", flush=True)

    tasks = (args.task,) if args.task else ACTIVITY_TASKS
    results = {task: activity_result(task, context, args) for task in tasks}
    drawn = [task for task, result in results.items() if result["status"] == "ok"]

    if not drawn:
        raise SystemExit("; ".join(f"{task}: {result['message']}" for task, result in results.items()))

    print(f"rows drawn: {drawn}", flush=True)
    print(f"rows left out: {[task for task in results if task not in drawn]}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.task}" if args.task else ""

    fig = build_figure(context, results, args.colors, compact=bool(args.task))
    path = output / f"fig_milk{suffix}.pdf"
    fig.savefig(path, format="pdf")
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {path} and {path.with_suffix('.png')}", flush=True)

    columns = ["task", "panel", "item", "value", "rest", "difference", "lo", "hi", "excludes_zero", "proteins"]
    rows = [row for result in results.values() for row in value_rows(result, context["letters"])]
    pd.DataFrame(rows, columns=columns).to_csv(output / f"fig_milk_values{suffix}.csv", index=False)
    write_caption(output / f"fig_milk_caption{suffix}.tex", context, results, args)
    print(f"wrote fig_milk_values{suffix}.csv and fig_milk_caption{suffix}.tex in {output}", flush=True)

# --------------------------------------------------------------------------- command line


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workbook", default=str(ROOT / "scripts" / "pr8b00604_si_002.xlsx"))
    common.add_argument("--corpus", default=str(ROOT / "data" / "processed" / "corpus"))
    common.add_argument("--device", default=None)
    common.add_argument("--hemolytic-top", type=float, default=0.25, help="share of the most hemolytic peptides excluded")
    common.add_argument("--identity", type=float, default=0.7, help="similarity that links peptides into one component")
    common.add_argument("--top", type=float, default=0.05, help="share of peptides called top-scored")
    common.add_argument("--call", type=float, default=0.20, help="top share per task called active under both scorers")
    common.add_argument("--draws", type=int, default=1000)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--table", type=int, default=12, help="shortlist rows drawn in the figure")
    common.add_argument("--seeds", type=int, default=5, help="Multi-task networks averaged")
    common.add_argument("--transform", choices=["standard", "stabilized"], default="stabilized")
    common.add_argument("--extra", choices=["none", "esm2"], default="esm2")
    common.add_argument("--clip", type=float, default=5.0)
    common.add_argument("--input-dropout", type=float, default=0.0)
    common.add_argument("--stopping", choices=["macro", "per_task"], default="per_task")
    common.add_argument("--epochs", type=int, default=100)
    common.add_argument("--patience", type=int, default=10)
    common.add_argument("--batch-size", type=int, default=512)
    common.add_argument("--hidden", type=int, default=512)
    common.add_argument("--dropout", type=float, default=0.3)
    common.add_argument("--lr", type=float, default=1e-3)
    common.add_argument("--weight-decay", type=float, default=1e-2)
    common.add_argument("--view", choices=["novel", "all"], default="novel", help="activity figure: sequences below --view-ceiling or all")
    common.add_argument("--view-ceiling", type=float, default=0.7, help="maximum similarity to the corpus in the novel view")
    common.add_argument("--task", choices=ACTIVITY_TASKS, default=None, help="activity figure: one activity in a 7.2 inch figure")
    common.add_argument("--structure", default=str(MILK / "peptidomics_structure.csv"))
    common.add_argument("--output", default=str(FIGURES))
    common.add_argument("--colors", choices=["activity", "shared"], default="activity", help="activity figure: candidate color per activity, or one shared candidate color")

    parser = argparse.ArgumentParser(description="Antibacterial candidates among human milk peptides.")
    stages = parser.add_subparsers(dest="stage", required=True)

    for name in ("filter", "score", "select", "figure", "activity_scores", "activity_figure", "tables", "all"):
        stages.add_parser(name, parents=[common])

    args = parser.parse_args()
    run = {
        "filter": (run_filter,),
        "score": (run_score,),
        "select": (run_select,),
        "figure": (run_figure,),
        "activity_scores": (run_activity_scores,),
        "activity_figure": (run_activity_figure,),
        "tables": (run_tables,),
        "all": (run_filter, run_score, run_select, run_figure),
    }

    for function in run[args.stage]:
        function(args)


if __name__ == "__main__":
    main()