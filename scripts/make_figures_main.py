"""
Figures of the main paper, drawn from results/tables, logs and the corpus files. Each figure is drawn on its own; a failure
in one is printed and the others still run. Writes paper/figures/fig-<name>.png. Panel letters are capital and bold.

    generator    A  requested against generated properties for ours, OmegAMP and HydrAMP
                 B  control by arm and copy rate
                 C  control on minus off by condition, and the matched control, which share the generator rows
                 D  length, charge, GRAVY and moment of the samples against the test split
                 E  unconditional samples of every generator against the test split (fourteen panels); ours is the seed
                    closest to the resampled training row, or the mean of the three seeds where that row is not defined
    design       A to G, 16:10: evidence behind the labels, independent units and label thresholds on the same task rows,
                 component sizes, sources, and ESM C with and without synthetic inactives
    results      A to H, 16:10: macro AUROC on matched subsets and all labels, per-fold macro with the fixed split,
                 threshold sensitivity, provenance against sequence models, nearest neighbor under two splits, and the
                 released classifiers

The design and results figures are drawn at WIDTH inches with axes placed in inches, so nothing moves when the text changes;
their text is 5 to 6 pt on the page. The generator figure is drawn on a 8.8 in canvas and scaled to the text width. The method
drawing, the likelihood by reveal order, the paired differences and the per-task AUROC are in make_figures_appendix.py.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/make_figures_main.py > logs/make_figures_main.log 2>&1
    PYTHONUNBUFFERED=1 uv run python scripts/make_figures_main.py --only generator
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ampx.evaluation import (  # noqa: E402
    auroc, component_groups, load_frame, percentile_interval, resample_components, task_selection,
)
from ampx.compliance import read_fasta  # noqa: E402
from ampx.models.data import compute_sequence_features, is_valid_sequence  # noqa: E402

TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "logs"
CORPUS = ROOT / "data" / "processed" / "corpus"
SAMPLES = ROOT / "results" / "samples"
RUNS = ROOT / "runs" / "unified"
OUTPUT = ROOT / "paper" / "figures"

WIDTH = 6.5

TICK = 7.0
NOTE = 6.5
LABEL = 7.5
TITLE = 8.0
LETTER = 9.5

INK = "#1A1A1A"
GREY = "#8A8A8A"
RULE = "#E4E4E4"
NETWORK = GREY
BOOSTED = "#286AA5"
PROVENANCE = "#E49D18"
SKY = "#65A3CE"
GREEN = "#2D925D"
PURPLE = "#AF80B8"

OURS_DARK = "#176A44"
OURS_MID = "#55AA7C"
OURS_LIGHT = "#8AC9AA"

ESM_DARK = "#164D83"
ESM_BLUE = "#286AA5"
ESM_MID = "#3F83B8"
ESM_LIGHT = "#65A3CE"
ESM_PALE = "#88B7D7"

OMEGAMP = "#C18A15"
OMEGAMP_RETRAINED = "#E2B84D"
HYDRAMP = "#B33E48"
HYDRAMP_RETRAINED = "#D36E75"


SIX_TASKS = ("antibacterial", "antifungal", "anticancer", "antiinflammatory", "hemolytic", "antiviral")
SMALL_TASKS = ("cytotoxic", "antitubercular", "antibiofilm", "antiparasitic")
ALL_TASKS = SIX_TASKS + SMALL_TASKS

TASK_LABELS = {
    "antibacterial": "Antibacterial", "antifungal": "Antifungal", "anticancer": "Anticancer",
    "antiinflammatory": "Antiinflammatory", "hemolytic": "Hemolytic", "antiviral": "Antiviral",
    "cytotoxic": "Cytotoxic", "antitubercular": "Antitubercular", "antibiofilm": "Antibiofilm",
    "antiparasitic": "Antiparasitic",
}
TASK_SHORT = {
    "antibacterial": "Antibact.", "antifungal": "Antifung.", "anticancer": "Anticancer",
    "antiinflammatory": "Antiinflam.", "hemolytic": "Hemolytic", "antiviral": "Antiviral",
    "cytotoxic": "Cytotoxic", "antitubercular": "Antitub.", "antibiofilm": "Antibiofilm",
    "antiparasitic": "Antiparas.",
}

# key in the score files, label, color; the color is the only thing that distinguishes a method

MODELS = (
    ("blend_network_esmc", "Ensemble", OURS_DARK),
    ("multitask_pertask", "Multi-task", GREEN),
    ("esm2_esmc_combined", "ESM C + ESM-2", ESM_DARK),
    ("esmc_combined", "ESM C", ESM_BLUE),
    ("combined", "ESM-2", ESM_MID),
    ("esmc", "ESM C emb", ESM_LIGHT),
    ("esm2", "ESM-2 emb", ESM_PALE),
    ("omegamp_xgb", "OmegAMP", OMEGAMP),
    ("hydramp_retrained", "HydrAMP", HYDRAMP_RETRAINED),
    ("composition", "Composition", INK),
    ("properties", "Properties", INK),
    ("nearest_neighbor", "Nearest neighbor", INK),
    ("length", "Length", INK),
)

RESULT_MODELS = tuple(item for item in MODELS if item[0] != "hydramp_retrained")
LABELS = {key: label for key, label, _ in MODELS}
COLORS = {key: color for key, _, color in MODELS}
MARKER_SCALE = {"o": 1.0, "^": 1.25, "s": 0.9}

OMEGAMP_RUNS = (
    ("omegamp_released_s0", "OmegAMP released", OMEGAMP),
    ("omegamp_retrained_s0", "OmegAMP retrained", OMEGAMP_RETRAINED),
)

OURS_COLOR = GREEN
HYDRAMP_RUNS = (
    ("released", "HydrAMP released", HYDRAMP),
    ("retrained", "HydrAMP retrained", HYDRAMP_RETRAINED),
)

# models scored on the peptides of at most 25 residues next to the HydrAMP head
RESTRICTED_KEYS = ("multitask_pertask", "esmc_combined", "hydramp_retrained", "composition", "properties")
MAIN_POSITIONS = np.arange(len(RESULT_MODELS))[::-1].astype(float)
RESTRICTED_POSITIONS = -2.0 - np.arange(len(RESTRICTED_KEYS)).astype(float)

# name of a method in the fixed-split macro table
FIXED_KEY = {"multitask_pertask": "multitask"}

CELL = re.compile(r"^(?:(\S+)\s+)?(-?\d*\.?\d+)\s+\[\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+)\s*\]$")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": LABEL,
            "axes.titlesize": TITLE,
            "axes.labelsize": LABEL,
            "axes.labelpad": 2.0,
            "axes.titlepad": 3.0,
            "xtick.labelsize": TICK,
            "ytick.labelsize": TICK,
            "legend.fontsize": TICK,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.major.pad": 1.8,
            "ytick.major.pad": 1.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "figure.constrained_layout.h_pad": 0.03,
            "figure.constrained_layout.w_pad": 0.03,
            "figure.constrained_layout.hspace": 0.02,
            "figure.constrained_layout.wspace": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "lines.linewidth": 0.8,
        }
    )


def panel(ax, letter: str, title: str, size: float = TITLE, letter_size: float = LETTER) -> None:
    letter = letter.upper()
    ax.annotate(
        letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
        fontsize=letter_size, fontweight="bold", ha="left", va="bottom",
    )
    ax.annotate(
        title, xy=(0, 1), xycoords="axes fraction", xytext=((4.5 + 6.5 * len(letter)) * letter_size / LETTER, 3.5),
        textcoords="offset points", fontsize=size, ha="left", va="bottom",
    )


def subtitle(ax, text: str, size: float) -> None:
    # title without a letter, for the panels of a group that carries one letter
    ax.annotate(
        text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3.5), textcoords="offset points",
        fontsize=size, ha="left", va="bottom",
    )


def placer(fig, width: float, height: float):
    # axes placed by their left edge and top edge in inches
    def place(left, top, size_x, size_y, **kwargs):
        return fig.add_axes([left / width, 1.0 - (top + size_y) / height, size_x / width, size_y / height], **kwargs)

    return place


def group_letter(fig, width: float, height: float, letter: str, left: float, top: float, size: float) -> None:
    # bold letter just left of the top-left corner of the first axes of a group, on the baseline of the titles
    fig.text(
        (left - 0.04) / width, 1.0 - (top - 0.05) / height, letter, fontsize=size, fontweight="bold", ha="right",
        va="bottom",
    )


def bottom_legend(fig, handles, columns: int, size: float) -> None:
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=columns, frameon=False, fontsize=size,
        handletextpad=0.3, columnspacing=1.0, labelspacing=0.2, borderaxespad=0.1,
    )


def save(fig, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"fig-{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def legend_line(fig, handles, columns: int, size: float | None = None) -> None:
    fig.legend(
        handles=handles, loc="outside lower center", ncol=columns, frameon=False,
        handletextpad=0.3, columnspacing=1.2, labelspacing=0.3, fontsize=size,
    )


def dot(color: str, label: str, filled: bool = True, alpha: float = 1.0, marker: str = "o") -> Line2D:
    return Line2D(
        [], [], marker=marker, color=color, mfc=color if filled else "white", lw=0,
        ms=3.8 * MARKER_SCALE.get(marker, 1.0), alpha=alpha, label=label,
    )


def interval(
    ax, centre, position, low, high, color, filled=True, size=3.4, alpha=1.0, width=0.8, cap=1.3, vertical=False,
    marker="o",
) -> None:
    error = [[centre - low], [high - centre]]
    x, y = (position, centre) if vertical else (centre, position)
    ax.errorbar(
        x, y, xerr=None if vertical else error, yerr=error if vertical else None, fmt=marker, color=color,
        mfc=color if filled else "white", ms=size * MARKER_SCALE.get(marker, 1.0), lw=width, capsize=cap,
        capthick=0.5, alpha=alpha, zorder=3,
    )


def bands(ax, positions) -> None:
    for index, position in enumerate(positions):
        if index % 2 == 0:
            ax.axhspan(position - 0.5, position + 0.5, color="#F5F5F5", zorder=0, lw=0)


def parse_cell(text):
    match = CELL.match(str(text).strip())

    if match is None:
        return None

    name, value, low, high = match.groups()

    return name, float(value), float(low), float(high)


def effective_units(components: np.ndarray) -> tuple[int, int, float]:
    sizes = pd.Series(components).value_counts().to_numpy(dtype=float)
    share = sizes / sizes.sum()

    return int(sizes.sum()), len(sizes), float(1.0 / (share**2).sum())


def hide_row_labels(ax) -> None:
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", length=0)


def box_axes(*axes) -> None:
    for ax in axes:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.5)


def run_figures(figures: dict, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--only", nargs="+", choices=sorted(figures), default=None)
    args = parser.parse_args()

    set_style()
    failed = []

    for name, function in figures.items():
        if args.only and name not in args.only:
            continue

        try:
            function()
        except (Exception, SystemExit):
            failed.append(name)
            print(f"figure {name} failed:", flush=True)
            traceback.print_exc()
            sys.stderr.flush()
            plt.close("all")

    print("failed figures:", failed if failed else "none", flush=True)


def macro_table(mode: str, tag: str | None = None) -> pd.DataFrame:
    # six-task macro of every method: the rank average file holds all models but OmegAMP, which has its own file
    stem = f"macro_benchmark_{mode}_cv" if tag is None else f"macro_benchmark_{mode}_{tag}"
    blend = pd.read_csv(TABLES / f"{stem}_blend_full.csv").set_index("model")
    omegamp = pd.read_csv(TABLES / f"{stem}_omegamp_full.csv").set_index("model")

    return pd.concat([blend, omegamp.loc[["omegamp_xgb"]]])


def restricted_table(mode: str, tag: str | None = None) -> pd.DataFrame:
    name = "macro_restricted_hydramp_retrained_full.csv" if tag is None else f"macro_restricted_hydramp_retrained_{tag}_full.csv"
    table = pd.read_csv(TABLES / name)

    return table[table["mode"] == mode].set_index("model")


def draw_macro(ax, table, style, offset=0.0, alpha=1.0, family=True) -> None:
    for position, (key, _, color) in zip(MAIN_POSITIONS, RESULT_MODELS):
        row = table.loc[key]
        interval(
            ax, row["peptide_macro"], position + offset, row["peptide_lo"], row["peptide_hi"], color, alpha=alpha,
            size=style["size"], width=style["line"], cap=style["cap"],
        )

        if family:
            ax.scatter(
                row["family_macro"], position + offset, marker="D", s=style["diamond"], facecolors="white",
                edgecolors=color, linewidths=0.7, zorder=4,
            )


def draw_restricted(ax, table, style, offset=0.0, alpha=1.0) -> None:
    for position, key in zip(RESTRICTED_POSITIONS, RESTRICTED_KEYS):
        row = table.loc[key]
        interval(
            ax, row["macro"], position + offset, row["lo"], row["hi"], COLORS[key], alpha=alpha,
            size=style["size"], width=style["line"], cap=style["cap"],
        )


def frame_rows(ax, note: float, labels: bool = True, divider: bool = True) -> None:
    ax.set_yticks(np.concatenate([MAIN_POSITIONS, RESTRICTED_POSITIONS]))
    ax.set_ylim(RESTRICTED_POSITIONS[-1] - 0.6, MAIN_POSITIONS[0] + 0.6)

    if labels:
        ax.set_yticklabels([label for _, label, _ in RESULT_MODELS] + [LABELS[key] for key in RESTRICTED_KEYS])
        ax.text(
            0.03, -1.5, "At most 25 residues", fontsize=note, color=GREY, va="center",
            transform=ax.get_yaxis_transform(),
        )

    if divider:
        ax.axhline(-1.0, color=GREY, lw=0.5, zorder=1)


GENERATOR_RUNS = ("gen_s0", "gen_s1", "gen_s2")
ARMS = (
    ("", "Ours: label only"),
    ("_sc", "Ours: train props,\nown length + label"),
    ("_sc_cl", "Ours: train props\n+ label"),
    ("_pf_state", "Ours: density\n+ label"),
    ("_pf_state_nolabel", "Ours: density,\nno label"),
    ("_pf_pooled", "Ours: density\npooled + label"),
)
SAMPLING_SEED_ARMS = ("_pf_state", "_pf_state_nolabel")
ORACLE_FAMILIES = (
    ("steering_matched", "ESM-2 oracles, matched", BOOSTED),
    ("steering_alllabels", "ESM-2 oracles, all labels", SKY),
    ("steering_esmc", "ESM C oracles", PURPLE),
)


def arm_groups(suffix: str) -> list[list[str]]:
    groups = []

    for run in GENERATOR_RUNS:
        names = [f"{run}{suffix}"]

        if run == GENERATOR_RUNS[0] and suffix in SAMPLING_SEED_ARMS:
            names += [f"{run}{suffix}_s{seed}" for seed in (100, 200)]

        groups.append(names)

    return groups


def group_values(table, column: str, suffix: str) -> list[float]:
    # one value per trained seed: the mean over its sampling seeds
    values = []

    for names in arm_groups(suffix):
        present = [float(table.loc[name, column]) for name in names if name in table.index]

        if present:
            values.append(float(np.mean(present)))

    return values


OMEGAMP_SAMPLES = ROOT / "results" / "samples_omegamp"
PUBLISHED_SAMPLES = ROOT / "results" / "samples_published"
FLAG_SAMPLES = ROOT / "results" / "samples_flags"

PUBLISHED_SETS = (
    ("OmegAMP released", PUBLISHED_SAMPLES / "omegamp__unconditional.fasta", OMEGAMP),
    ("OmegAMP retrained", PUBLISHED_SAMPLES / "omegamp_retrained__unconditional.fasta", OMEGAMP_RETRAINED),
    ("HydrAMP released", PUBLISHED_SAMPLES / "hydramp__unconditional.fasta", HYDRAMP),
    ("HydrAMP retrained", PUBLISHED_SAMPLES / "hydramp_retrained__unconditional.fasta", HYDRAMP_RETRAINED),
    ("Ours, seed 0", SAMPLES / "gen_s0__unconditional.fasta", OURS_MID),
)

GENERATOR_ROWS = (
    ("reference_test", "Test split", INK),
    ("reference_train", "Training (resampled)", GREY),
    ("ours_s0", "Ours, seed 0", GREEN),
    ("ours_s1", "Ours, seed 1", GREEN),
    ("ours_s2", "Ours, seed 2", GREEN),
    ("hydramp", "HydrAMP released", HYDRAMP),
    ("hydramp_retrained", "HydrAMP retrained", HYDRAMP_RETRAINED),
    ("omegamp", "OmegAMP released", OMEGAMP),
    ("omegamp_retrained", "OmegAMP retrained", OMEGAMP_RETRAINED),
)
COMPARISON_ROWS = (
    ("reference_test", "Test split", INK),
    ("reference_train", "Training (resampled)", GREY),
    ("ours", "Ours", GREEN),
    ("hydramp", "HydrAMP released", HYDRAMP),
    ("hydramp_retrained", "HydrAMP retrained", HYDRAMP_RETRAINED),
    ("omegamp", "OmegAMP released", OMEGAMP),
    ("omegamp_retrained", "OmegAMP retrained", OMEGAMP_RETRAINED),
)
GENERATOR_PANELS = (
    ("exact", "Exact copy", "% of samples", 100.0),
    ("ge070", "\u22650.7, corpus", "% of samples", 100.0),
    ("ge070_test", "\u22650.7, test", "% of samples", 100.0),
    ("frechet", "Fr\u00e9chet", "distance", 1.0),
    ("coverage", "Coverage", "share of test", 1.0),
    ("precision", "Precision", "share", 1.0),
    ("recall", "Recall", "share", 1.0),
    ("diversity", "Diversity", "edit distance", 1.0),
    ("charge2", "Charge \u22652", "% of samples", 100.0),
    ("js_composition", "JS composition", "bits", 1.0),
    ("js_dipeptide", "JS dipeptide", "bits", 1.0),
    ("w1_charge_per_residue", "W1 charge", "SD of test", 1.0),
    ("w1_mu_h", "W1 moment", "SD of test", 1.0),
    ("w1_gravy_raw", "W1 GRAVY", "SD of test", 1.0),
)


def sequence_properties(sequences) -> pd.DataFrame:
    features = compute_sequence_features(list(sequences))
    length = features["length"].to_numpy(dtype=np.float64)

    return pd.DataFrame(
        {
            "length": length,
            "charge_per_residue": features["net_charge"].to_numpy(dtype=np.float64) / length,
            "gravy_raw": features["gravy_raw"].to_numpy(dtype=np.float64),
            "mu_h": features["mu_h"].to_numpy(dtype=np.float64),
        }
    )


def fasta_properties(path: Path) -> pd.DataFrame:
    names = (str(item).upper().strip() for item in read_fasta(path)[1])

    return sequence_properties([item for item in dict.fromkeys(names) if is_valid_sequence(item) and 8 <= len(item) <= 50])


def charge_pairs(paths, requested_charge_column: bool):
    # requested and generated charge per residue; ours requests a continuous value, OmegAMP an integer net charge
    frame = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True).dropna(subset=["sequence"])
    frame = frame[(frame["sequence"].str.len() > 0) & (frame["sequence"].str.len() <= 50)].reset_index(drop=True)
    features = compute_sequence_features(frame["sequence"].tolist())
    obtained = features["net_charge"].to_numpy() / features["length"].to_numpy()

    if requested_charge_column:
        wanted = frame["charge_per_residue"].to_numpy()
    else:
        wanted = (frame["net_requested"] / frame["length_requested"]).to_numpy()

    return wanted, obtained


def binned_median(x: np.ndarray, y: np.ndarray, bins: int = 12):
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    index = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
    used = [i for i in range(bins) if (index == i).any()]

    return np.array([x[index == i].mean() for i in used]), np.array([np.median(y[index == i]) for i in used])


def flag_pairs(which: str):
    # charge per residue of the same latent draw decoded with both HydrAMP flags off and on, from the files scored in the
    # control table; the two files are paired line by line, peptides of 8 to 25 residues on both sides
    on_path = FLAG_SAMPLES / f"hydramp_{which}_s0__on.fasta"
    off_path = FLAG_SAMPLES / f"hydramp_{which}_s0__off.fasta"

    if not (on_path.exists() and off_path.exists()):
        return None

    on, off = read_fasta(on_path)[1], read_fasta(off_path)[1]

    if len(on) != len(off):
        raise SystemExit(f"hydramp {which}: {len(on)} on and {len(off)} off sequences are not paired")

    kept = [
        (first, second) for first, second in zip(on, off)
        if 8 <= len(first) <= 25 and 8 <= len(second) <= 25 and is_valid_sequence(first) and is_valid_sequence(second)
    ]
    first = sequence_properties([item[0] for item in kept])
    second = sequence_properties([item[1] for item in kept])

    return second["charge_per_residue"].to_numpy(), first["charge_per_residue"].to_numpy()


def generator_name(name: str) -> str:
    for prefix in ("ours_gen_s", "gen_s"):
        if name.startswith(prefix):
            return "ours_s" + name[len(prefix):]

    return name


def generator_values() -> pd.DataFrame:
    comparison = pd.read_csv(TABLES / "generator_comparison.csv").set_index("model")
    manifold = pd.read_csv(TABLES / "generator_manifold.csv").set_index("model")
    novelty = pd.read_csv(TABLES / "published_generators_novelty.csv").set_index("model")
    novelty.index = [generator_name(name) for name in novelty.index]
    names = [name for name in ("sample_evaluation_published.csv", "sample_evaluation_binned.csv") if (LOGS / name).exists()]
    evaluation = pd.concat([pd.read_csv(LOGS / name) for name in names], ignore_index=True)
    evaluation = evaluation[evaluation["condition"].isin(["unconditional", "all"])].drop_duplicates("model", keep="last")
    evaluation = evaluation.assign(model=[generator_name(name) for name in evaluation["model"]]).set_index("model")

    table = pd.DataFrame(index=[key for key, _, _ in GENERATOR_ROWS])
    table["exact"], table["ge070"], table["ge070_test"] = novelty["exact_copy_corpus"], novelty["ge_070_corpus"], novelty["ge_070_test"]
    table["frechet"], table["coverage"] = evaluation["frechet_test"], evaluation["coverage_ge_050"]
    table["precision"], table["recall"] = manifold["precision"], manifold["recall"]
    table["diversity"], table["charge2"] = comparison["diversity"], comparison["share_charge_ge_2"]

    for name in ("js_composition", "js_dipeptide", "w1_charge_per_residue", "w1_mu_h", "w1_gravy_raw"):
        table[name] = comparison[name]

    return table


def ours_row(table: pd.DataFrame) -> pd.DataFrame:
    # one row for ours: in each column the seed closest to the resampled training row, or the mean of the three seeds
    # where the training row is not defined; the choice is printed so the caption can state it
    names = ["ours_s0", "ours_s1", "ours_s2"]
    seeds = table.loc[names]
    row = {}

    for column in table.columns:
        values = seeds[column].dropna()
        reference = table.loc["reference_train", column]

        if values.empty:
            row[column] = np.nan
        elif pd.notna(reference):
            closest = int((values - reference).abs().to_numpy().argmin())
            row[column] = float(values.iloc[closest])
            print(f"ours, {column}: {values.index[closest]}", flush=True)
        else:
            row[column] = float(values.mean())
            print(f"ours, {column}: mean of the three seeds", flush=True)

    result = table.drop(index=names)
    result.loc["ours"] = pd.Series(row)

    return result


def figure_generator() -> None:
    paths = sorted(SAMPLES.glob("gen_s0_pf_state__*.requests.tsv"))

    if not paths:
        raise SystemExit("no gen_s0_pf_state requests files in results/samples")

    frame = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True)
    frame = frame[frame["sequence"].fillna("").str.len() > 0].reset_index(drop=True)

    generated = compute_sequence_features(frame["sequence"].tolist())
    frame["gen_charge_per_residue"] = generated["net_charge"].to_numpy() / generated["length"].to_numpy()

    hit = pd.read_csv(TABLES / "property_hit.csv")
    hit = hit[hit["model"] == "gen_s0_pf_state"].set_index("property")
    omegamp = pd.read_csv(TABLES / "omegamp_property_hit.csv").set_index("model")

    sample = frame.sample(min(40000, len(frame)), random_state=0)

    values = {name: pd.read_csv(TABLES / f"{name}.csv").set_index("model") for name, _, _ in ORACLE_FAMILIES}
    copy = pd.read_csv(TABLES / "generator_copy_rates.csv").set_index("model")
    comparison = ours_row(generator_values())
    arms = tuple(item for item in ARMS if item[0] != "")

    flag_tables = [pd.read_csv(TABLES / "generator_flags.csv")]
    reference_path = TABLES / "generator_flags_reference.csv"

    if reference_path.exists():
        flag_tables.append(pd.read_csv(reference_path))

    flags = pd.concat(flag_tables, ignore_index=True)
    flags["family"] = flags["label"].str.replace(r"_s\d+$", "", regex=True)
    has_matched = "matched_control" in flags.columns
    has_unmatched = "control" in flags.columns

    families = (
        ("ours_trainprops_ownlength", "Ours: train props,\nown length + label", GREEN),
        ("hydramp_released", "HydrAMP\nreleased", HYDRAMP),
        ("hydramp_retrained", "HydrAMP\nretrained", HYDRAMP_RETRAINED),
        ("omegamp_targeted_released", "OmegAMP\nreleased", OMEGAMP),
        ("omegamp_targeted_retrained", "OmegAMP\nretrained", OMEGAMP_RETRAINED),
    )

    families = tuple(item for item in families if (flags["family"] == item[0]).any())

    corpus = None
    sets = [("Test split", pd.read_csv(ROOT / "reference" / "test_properties.csv"), INK)]
    sets += [(label, fasta_properties(path), color) for label, path, color in PUBLISHED_SETS if path.exists()]

    x = sample["charge_per_residue"].to_numpy()
    y = sample["gen_charge_per_residue"].to_numpy()
    low, high = np.quantile(np.concatenate([x, y]), [0.005, 0.995])
    rho, error = hit.loc["charge_per_residue", "spearman"], hit.loc["charge_per_residue", "mean_abs_error_sd"]
    items = [
        (
            "Ours, seed 0\ncharge per residue", x, y, low, high, f"\u03c1 {rho:.2f}\nerror {error:.2f} SD", OURS_COLOR,
            "Requested",
        )
    ]

    for prefix, label, color in OMEGAMP_RUNS:
        x, y = charge_pairs(sorted(OMEGAMP_SAMPLES.glob(f"{prefix}__*.tsv")), False)
        row = omegamp.loc[prefix]
        text = f"\u03c1 {row['spearman']:.2f}\nerror {row['mean_abs_error_sd']:.2f} SD"
        items.append((f"{label}\ncharge per residue", x, y, low, high, text, color, "Requested"))

    for which, label, color in HYDRAMP_RUNS:
        pair = flag_pairs(which)

        if pair is None:
            raise SystemExit(f"no hydramp_{which}_s0 on and off files in results/samples_flags")

        x, y = pair
        text = f"\u03c1 {stats.spearmanr(x, y).statistic:.2f}\nshift {np.mean(y - x):+.3f}"
        items.append((f"{label}\ncharge per residue", x, y, low, high, text, color, "Flags off"))

    # canvas wider than the text width; included at the text width it is scaled by text width / fig_width, so the font sizes
    # below are chosen for that scaling
    fig_width = 8.8
    tick, label_size, note, title_size, letter_size = 6.2, 6.4, 6.0, 6.6, 8.5

    count = len(items)
    side, title_zone = 1.15, 0.24
    margin_left, margin_right = 0.5, 0.3
    gap = (fig_width - margin_left - margin_right - count * side) / (count - 1)
    lefts = [margin_left + index * (side + gap) for index in range(count)]
    control_tall = 1.22

    top_scatter = 0.05 + 0.36
    top_scatter_legend = top_scatter + side + 0.40
    top_control = top_scatter + side + 0.7 + title_zone
    top_legend = top_control + control_tall + 0.45

    # control row: arm and copy rate (B), switching and matched control sharing the generator rows (C), distributions (D)
    arm_left, arm_side = 0.85, 0.95
    copy_left, copy_side = 1.90, 0.80
    switch_left, switch_side = 3.55, 0.80
    matched_left, matched_side = 4.42, 0.80
    cdf_left, cdf_side, cdf_gap = 5.70, 0.64, 0.08

    # unconditional samples of every generator: two rows of seven panels below the legends (E)
    app_tall = 0.88
    app_pitch = app_tall + title_zone + 0.38
    app_left, app_gap = 1.15, 0.10
    app_side = (fig_width - app_left - margin_right - 6 * app_gap) / 7.0
    top_app = top_legend + 0.60 + title_zone
    app_bottom = top_app + app_pitch + app_tall
    height = app_bottom + 0.65

    fig = plt.figure(figsize=(fig_width, height))
    place = placer(fig, fig_width, height)

    scatter_axes = [place(left, top_scatter, side, side) for left in lefts]

    for ax, (title, x, y, lo, hi, text, color, xlabel) in zip(scatter_axes, items):
        ax.hexbin(x, y, gridsize=26, bins="log", cmap="Blues", mincnt=1, extent=(lo, hi, lo, hi), linewidths=0.0)
        centres, medians = binned_median(x, y)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.5, ls="--")
        ax.plot(centres, medians, color=color, lw=0.9, marker="o", ms=2.0, mec=INK, mew=0.3, zorder=5)
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes, va="top", fontsize=note,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.locator_params(nbins=4)
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=1.5)
        ax.set_ylabel("Flags on" if xlabel == "Flags off" else "Generated", fontsize=label_size, labelpad=1.0)
        subtitle(ax, title, title_size)

    ax_arm = place(arm_left, top_control, arm_side, control_tall)
    ax_copy = place(copy_left, top_control, copy_side, control_tall, sharey=ax_arm)
    ax_switch = place(switch_left, top_control, switch_side, control_tall)
    ax_matched = place(matched_left, top_control, matched_side, control_tall, sharey=ax_switch)
    cdf_axes = [place(cdf_left + index * (cdf_side + cdf_gap), top_control, cdf_side, control_tall) for index in range(4)]

    y = np.arange(len(arms))[::-1].astype(float)
    models = [model for model, _ in arms]

    bands(ax_arm, y)
    bands(ax_copy, y)

    for index, (name, _, color) in enumerate(ORACLE_FAMILIES):
        offset = (1 - index) * 0.25

        for model, position in zip(models, y):
            seeds = group_values(values[name], "mean_six_tasks", model)
            ax_arm.scatter(seeds, [position + offset] * len(seeds), s=5, color=color, alpha=0.45, linewidths=0, zorder=3)
            ax_arm.scatter([np.mean(seeds)], [position + offset], s=16, color=color, edgecolors="white", linewidths=0.4, zorder=4)

    ax_arm.set_yticks(y)
    ax_arm.set_yticklabels([title for _, title in arms])
    ax_arm.set_ylim(-0.6, len(arms) - 0.4)
    ax_arm.set_xlim(0, 0.09)
    ax_arm.set_xticks([0.0, 0.04, 0.08])
    ax_arm.set_xlabel("Control, six-task mean\n(higher = stronger)", fontsize=label_size, labelpad=1.5)
    ax_arm.grid(axis="x", color=RULE, lw=0.4, zorder=0)
    subtitle(ax_arm, "Control by arm", title_size)

    for model, position in zip(models, y):
        means = group_values(copy, "mean_ge_070", model)

        if not means:
            continue

        mean_rate = float(np.mean(means))
        worst_rate = max(
            float(copy.loc[name, "max_ge_070"]) for names in arm_groups(model) for name in names if name in copy.index
        )
        ax_copy.hlines(position, mean_rate, worst_rate, color=GREY, lw=0.9, zorder=2)
        ax_copy.scatter(mean_rate, position, s=14, color=INK, zorder=3)
        ax_copy.scatter(worst_rate, position, s=14, facecolors="white", edgecolors=INK, linewidths=0.7, zorder=3)

    ax_copy.axvline(0.15, color=NETWORK, lw=0.7, ls="--")
    ax_copy.set_xlim(0, 0.22)
    ax_copy.set_xticks([0.0, 0.1, 0.2])
    ax_copy.set_xlabel("Copy rate at \u2265 0.7\n(lower = more novel)", fontsize=label_size, labelpad=1.5)
    ax_copy.grid(axis="x", color=RULE, lw=0.4, zorder=0)
    hide_row_labels(ax_copy)
    subtitle(ax_copy, "Copy rate", title_size)

    rows = np.arange(len(families))[::-1].astype(float)
    tasks = list(SIX_TASKS)

    bands(ax_switch, rows)
    bands(ax_matched, rows)

    for position, (family, _, color) in zip(rows, families):
        part = flags[flags["family"] == family]
        per_length = part.pivot_table(index="label", columns="task", values="strat_control")
        per_task = per_length[tasks].mean(axis=0).to_numpy()
        seeds = per_length[tasks].mean(axis=1).to_numpy()
        ax_switch.scatter(
            per_task, np.full(len(per_task), position) + np.linspace(-0.2, 0.2, len(per_task)),
            s=4, color=color, alpha=0.5, linewidths=0, zorder=3,
        )
        ax_switch.scatter([seeds.mean()], [position], s=18, color=color, edgecolors=INK, linewidths=0.4, zorder=4)

        if has_unmatched:
            unmatched = part.pivot_table(index="label", columns="task", values="control")
            ax_matched.scatter([unmatched[tasks].mean(axis=1).mean()], [position + 0.28], marker="s", s=12, color=color, edgecolors=INK, linewidths=0.4, zorder=4)

        ax_matched.scatter([per_length[tasks].mean(axis=1).mean()], [position], s=18, color=color, edgecolors=INK, linewidths=0.4, zorder=4)
        ax_matched.scatter([per_length["antibacterial"].mean()], [position], s=14, facecolors="white", edgecolors=color, linewidths=0.8, zorder=5)

        if has_matched:
            both = part.pivot_table(index="label", columns="task", values="matched_control")
            ax_matched.scatter([float(np.nanmean(both[tasks].to_numpy()))], [position - 0.28], marker="D", s=12, color=color, edgecolors=INK, linewidths=0.4, zorder=4)

    for ax in (ax_switch, ax_matched):
        ax.axvline(0.0, color=INK, lw=0.4, ls=":", zorder=1)
        ax.set_yticks(rows)
        ax.set_ylim(-0.7, len(families) - 0.4)
        ax.set_xlim(-0.025, 0.09)
        ax.set_xticks([0.0, 0.04, 0.08])
        ax.grid(axis="x", color=RULE, lw=0.4, zorder=0)

    ax_switch.set_yticklabels([label for _, label, _ in families])
    ax_switch.set_xlabel("On minus off,\nhigher = stronger", fontsize=label_size, labelpad=1.5)
    ax_matched.set_xlabel("On minus off,\n8 to 25 residues", fontsize=label_size, labelpad=1.5)
    hide_row_labels(ax_matched)
    subtitle(ax_switch, "Condition switch", title_size)
    subtitle(ax_matched, "Matched", title_size)

    test = sets[0][1]
    columns = (
        ("length", "Length", "Length\n(residues)"),
        ("charge_per_residue", "Charge", "Charge per\nresidue"),
        ("gravy_raw", "GRAVY", "Hydropathy\n(GRAVY)"),
        ("mu_h", "Moment", "Hydrophobic\nmoment"),
    )

    for index, (ax, (name, title, label)) in enumerate(zip(cdf_axes, columns)):
        for set_name, table, color in sets:
            data = np.sort(table[name].to_numpy())
            ax.plot(
                data, np.arange(1, len(data) + 1) / len(data), color=color,
                lw=1.2 if set_name == "Test split" else 0.8, drawstyle="steps-post",
            )

        if name == "length":
            ax.set_xlim(8, 50)
            ax.set_xticks([10, 30, 50])
        else:
            ax.set_xlim(*np.quantile(test[name], [0.005, 0.995]))
            ax.locator_params(axis="x", nbins=3)

        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlabel(label, fontsize=label_size, labelpad=1.5)
        ax.grid(color=RULE, lw=0.4, zorder=0)

        if index == 0:
            ax.set_ylabel("Cumulative share", fontsize=label_size, labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)

        subtitle(ax, title, title_size)

    app_rows = np.arange(len(COMPARISON_ROWS))[::-1].astype(float)
    app_axes = []

    for index, (column, title, unit, scale) in enumerate(GENERATOR_PANELS):
        row, position_in_row = divmod(index, 7)
        ax = place(
            app_left + position_in_row * (app_side + app_gap), top_app + row * app_pitch, app_side, app_tall,
            sharey=app_axes[0] if app_axes else None,
        )
        app_axes.append(ax)
        bands(ax, app_rows)

        if column.startswith(("js_", "w1_")) and pd.notna(comparison.loc["reference_train", column]):
            ax.axvline(comparison.loc["reference_train", column] * scale, color=GREY, lw=0.6, ls=":", zorder=1)

        for position, (key, _, color) in zip(app_rows, COMPARISON_ROWS):
            value = comparison.loc[key, column]

            if pd.notna(value):
                ax.scatter([value * scale], [position], s=16, color=color, edgecolors=INK, linewidths=0.4, zorder=3)
            else:
                ax.text(
                    0.5, position, "reference" if key == "reference_test" else "not defined", ha="center", va="center",
                    fontsize=note - 0.2, color=GREY, style="italic", transform=ax.get_yaxis_transform(),
                )

        ax.set_yticks(app_rows)
        ax.set_ylim(-0.7, len(COMPARISON_ROWS) - 0.3)

        if position_in_row == 0:
            ax.set_yticklabels([label for _, label, _ in COMPARISON_ROWS])
        else:
            ax.tick_params(labelleft=False)

        ax.set_xlabel(unit, fontsize=label_size, labelpad=1.0)
        ax.grid(axis="x", color=RULE, lw=0.4, zorder=0)
        ax.locator_params(axis="x", nbins=3)
        subtitle(ax, title, title_size)

    for ax in (*scatter_axes, ax_arm, ax_copy, ax_switch, ax_matched, *cdf_axes, *app_axes):
        ax.tick_params(labelsize=tick)

    box_axes(*scatter_axes, ax_arm, ax_copy, ax_switch, ax_matched, *cdf_axes, *app_axes)

    group_letter(fig, fig_width, height, "A", lefts[0], top_scatter, letter_size)
    group_letter(fig, fig_width, height, "B", arm_left, top_control, letter_size)
    group_letter(fig, fig_width, height, "C", switch_left, top_control, letter_size)
    group_letter(fig, fig_width, height, "D", cdf_left, top_control, letter_size)
    group_letter(fig, fig_width, height, "E", app_left, top_app, letter_size)

    def legend_at(handles, x, top, loc, columns=1):
        fig.legend(
            handles=handles, loc=loc, bbox_to_anchor=(x / fig_width, 1.0 - top / height), ncol=columns,
            frameon=False, fontsize=note, handletextpad=0.4, columnspacing=1.0, labelspacing=0.4, borderaxespad=0.0,
        )

    # A: one row centred under the scatter panels
    legend_at(
        [
            Patch(facecolor="#6BAED6", label="hexagon count, log"),
            Line2D([], [], color=GREY, lw=0.9, marker="o", ms=2.0, mec=INK, mew=0.3, label="binned median"),
            Line2D([], [], color=INK, lw=0.5, ls="--", label="identity"),
        ],
        (lefts[0] + lefts[-1] + side) / 2.0, top_scatter_legend, "upper center", columns=3,
    )

    # B: oracle families, then the copy-rate markers
    legend_at(
        [dot(BOOSTED, "ESM-2, matched labels"), dot(SKY, "ESM-2, all labels"), dot(PURPLE, "ESM C")],
        arm_left + arm_side + 0.10, top_legend, "upper right",
    )
    legend_at(
        [
            dot(INK, "mean of 12 conditions"), dot(INK, "worst condition", filled=False),
            Line2D([], [], color=NETWORK, lw=0.7, ls="--", label="0.15 limit"),
        ],
        copy_left + 0.10, top_legend, "upper left",
    )

    # C: the markers of the condition switch, then those of the matched control, in two columns
    legend_at(
        [
            Line2D([], [], marker="o", color=GREY, alpha=0.5, lw=0, ms=2.2, label="one task"),
            Line2D([], [], marker="o", color=GREY, mec=INK, mew=0.4, lw=0, ms=3.6, label="mean"),
            Line2D([], [], marker="s", color=GREY, mec=INK, mew=0.4, lw=0, ms=3.2, label="unmatched"),
            Line2D([], [], marker="o", color=GREY, mec=INK, mew=0.4, lw=0, ms=3.6, label="length-matched, six tasks"),
            Line2D([], [], marker="o", mfc="white", mec=GREY, color=GREY, lw=0, ms=3.4, label="length-matched, antibacterial"),
            Line2D([], [], marker="D", color=GREY, mec=INK, mew=0.4, lw=0, ms=3.4, label="also matched on charge, GRAVY"),
        ],
        switch_left - 0.15, top_legend, "upper left", columns=2,
    )

    # D: two columns of three, centred under the four panels
    legend_at(
        [Line2D([], [], color=color, lw=1.2, label=label) for label, _, color in sets],
        cdf_left + (4 * cdf_side + 3 * cdf_gap) / 2.0, top_legend, "upper center", columns=2,
    )

    # E: the dotted line is the same corpus split in two
    legend_at(
        [Line2D([], [], color=GREY, lw=0.8, ls=":", label="dotted line: two splits of the corpus (training row), in the JS and W1 panels")],
        fig_width / 2.0, app_bottom + 0.42, "upper center",
    )

    save(fig, "generator")


def fold_score_frame() -> pd.DataFrame:
    # out-of-fold scores of every method on matched labels, six macro tasks
    names = (
        "benchmark_scores_cv.csv", "benchmark_scores_cv_tasks.csv",
        "multitask_scores_cv_subset_pertask.csv", "multitask_scores_cv_subset_pertask_tasks.csv",
        "blend_scores_cv_subset.csv", "blend_scores_cv_subset_tasks.csv",
        "baseline_omegamp_xgb_scores_cv_subset.csv", "baseline_omegamp_xgb_scores_cv_subset_tasks.csv",
    )
    frame = pd.concat([pd.read_csv(LOGS / name) for name in names], ignore_index=True)

    return frame[(frame["mode"] == "subset") & frame["task"].isin(SIX_TASKS)]


def fold_macros(frame: pd.DataFrame) -> np.ndarray:
    values = []

    for fold in range(10):
        part = frame[frame["fold"] == fold]
        values.append(np.mean([auroc(group["target"].to_numpy(), group["score"].to_numpy()) for _, group in part.groupby("task")]))

    return np.array(values)


def fixed_split_ensemble() -> float:
    # equal-weight rank average of the network and ESM C combined on the fixed split, macro over six tasks
    reference = pd.read_csv(LOGS / "fixed_ten" / "benchmark_scores_final.csv")
    reference = reference[(reference["mode"] == "subset") & (reference["model"] == "esmc_combined")]
    network = pd.concat(
        [
            pd.read_csv(LOGS / "multitask_scores_subset_pertask.csv"),
            pd.read_csv(LOGS / "fixed_ten" / "multitask_scores_subset_pertask_tasks.csv"),
        ]
    )
    network = network[network["mode"] == "subset"]
    values = []

    for task in SIX_TASKS:
        x = reference[reference["task"] == task].reset_index(drop=True)
        y = network[network["task"] == task].reset_index(drop=True)

        if not (np.array_equal(x["target"], y["target"]) and np.array_equal(x["component"], y["component"])):
            raise SystemExit(f"{task}: fixed-split score rows differ between the network and ESM C combined")

        values.append(auroc(y["target"].to_numpy(), stats.rankdata(x["score"]) + stats.rankdata(y["score"])))

    return float(np.mean(values))


def fixed_split_values() -> dict:
    # methods without a fixed-split run have no entry
    table = pd.read_csv(TABLES / "macro_benchmark_subset_split_full.csv").set_index("model")
    values = {
        key: float(table.loc[FIXED_KEY.get(key, key), "peptide_macro"])
        for key, _, _ in RESULT_MODELS if FIXED_KEY.get(key, key) in table.index
    }
    values["blend_network_esmc"] = fixed_split_ensemble()

    return values


def draw_results(ax_a, ax_b, ax_c, style) -> None:
    matched, everything = macro_table("subset"), macro_table("all")
    scores = fold_score_frame()
    fixed = fixed_split_values()
    folds = {key: fold_macros(scores[scores["model"] == key]) for key, _, _ in RESULT_MODELS}

    for ax, table, mode, letter, title in (
        (ax_a, matched, "subset", "a", "Matched subsets"), (ax_b, everything, "all", "b", "All labels")
    ):
        draw_macro(ax, table, style)
        draw_restricted(ax, restricted_table(mode), style)
        ax.set_xlim(0.56, 0.80)
        ax.set_xticks([0.6, 0.7])
        ax.set_xlabel("Macro AUROC")
        ax.grid(axis="x", color=RULE, lw=0.4, zorder=0)
        panel(ax, letter, title, style["title"], style["letter"])

    generator = np.random.default_rng(0)

    for position, (key, _, color) in zip(MAIN_POSITIONS, RESULT_MODELS):
        values = folds[key]

        try:
            body = ax_c.violinplot(values, positions=[position], orientation="horizontal", widths=0.9, showextrema=False)["bodies"][0]
        except TypeError:
            body = ax_c.violinplot(values, positions=[position], vert=False, widths=0.9, showextrema=False)["bodies"][0]

        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.22)

        spread = position + generator.uniform(-0.2, 0.2, len(values))
        ax_c.scatter(values[1:], spread[1:], color=color, s=style["dot"], zorder=3)
        ax_c.scatter(values[:1], spread[:1], facecolors="white", edgecolors=color, linewidths=0.7, s=style["dot"] + 1, zorder=3)
        ax_c.vlines(values.mean(), position - 0.35, position + 0.35, color=color, lw=1.0, zorder=2)

        if key in fixed:
            ax_c.scatter([fixed[key]], [position], marker="D", s=style["diamond"] + 6, color=color, edgecolors="white", linewidths=0.4, zorder=4)

    everything_c = np.concatenate(list(folds.values()) + [np.array(list(fixed.values()))])
    ax_c.set_xlim(everything_c.min() - 0.01, everything_c.max() + 0.01)
    ax_c.set_xticks([0.60, 0.70, 0.80])
    ax_c.set_xlabel("Mean AUROC, six tasks")
    ax_c.grid(axis="x", color=RULE, lw=0.4, zorder=0)
    panel(ax_c, "c", "Folds, fixed split", style["title"], style["letter"])

    frame_rows(ax_a, style["note"])
    frame_rows(ax_b, style["note"], labels=False)
    frame_rows(ax_c, style["note"], labels=False, divider=False)
    hide_row_labels(ax_b)
    hide_row_labels(ax_c)


def draw_shortcuts(ax_a, ax_b, ax_c, style) -> None:
    table = pd.read_csv(TABLES / "source_probe.csv")
    lookup = {(row.mode, row.task): row for row in table.itertuples()}
    counts = pd.read_csv(TABLES / "corpus_tasks.csv").set_index("task")
    protocol = pd.read_csv(TABLES / "split_protocol.csv").set_index("task")
    matched_tasks = ("antibacterial", "antifungal", "anticancer", "antiviral")
    rows = np.arange(len(ALL_TASKS))[::-1].astype(float)
    note = style["note"]

    def draw_probe(ax, mode: str, tasks) -> None:
        bands(ax, rows)

        for position, task in zip(rows, ALL_TASKS):
            if task not in tasks:
                continue

            row = lookup.get((mode, task))

            if row is None:
                ax.text(
                    0.5, position, f"not evaluable ({int(counts.loc[task, 'test_inactive'])} inactives)",
                    va="center", ha="center", fontsize=note, color=GREY, transform=ax.get_yaxis_transform(),
                )
                continue

            source, best, gap = parse_cell(row.source), parse_cell(row.best_reference), parse_cell(row.source_minus_best)
            interval(ax, source[1], position + 0.17, source[2], source[3], PROVENANCE, size=2.2, width=0.6, cap=0.6)
            interval(ax, best[1], position - 0.17, best[2], best[3], BOOSTED, size=2.2, width=0.6, cap=0.6)
            clear = gap is not None and (gap[2] > 0 or gap[3] < 0)
            ax.text(
                1.08, position, f"{source[1] - best[1]:+.2f}", va="center", ha="left", fontsize=note,
                fontweight="bold" if clear else "normal",
            )

        ax.axhline(rows[len(SIX_TASKS) - 1] - 0.5, color=GREY, lw=0.5, zorder=1)
        ax.axvline(0.5, color=INK, lw=0.4, ls=":", zorder=1)
        ax.set_xlim(0.3, 1.36)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Test AUROC")
        # ax.text(1.08, rows[0] + 0.6, "gap", va="bottom", ha="left", fontsize=note, clip_on=False)

    draw_probe(ax_a, "all", ALL_TASKS)
    draw_probe(ax_b, "subset", matched_tasks)
    panel(ax_a, "e", "Provenance, all labels", style["title"], style["letter"])
    panel(ax_b, "f", "Provenance, matched", style["title"], style["letter"])

    bands(ax_c, rows)

    for position, task in zip(rows, ALL_TASKS):
        if task not in protocol.index or pd.isna(protocol.loc[task, "nearest_neighbor_homology"]):
            ax_c.text(0.675, position, "not evaluable", va="center", ha="center", fontsize=note, color=GREY)
            continue

        disjoint = protocol.loc[task, "nearest_neighbor_homology"]
        random_split = protocol.loc[task, "nearest_neighbor_random"]
        ax_c.scatter(disjoint, position, s=9, color="#4D4D4D", zorder=3)

        if pd.isna(random_split):
            continue

        ax_c.hlines(position, min(disjoint, random_split), max(disjoint, random_split), color=RULE, lw=0.9, zorder=1)
        ax_c.scatter(random_split, position, s=9, color=PURPLE, zorder=3)
        ax_c.text(1.0, position, f"{protocol.loc[task, 'random_test_fraction_ge_070']:.0%}", va="center", ha="left", fontsize=note)

    ax_c.axhline(rows[len(SIX_TASKS) - 1] - 0.5, color=GREY, lw=0.5, zorder=1)
    ax_c.axvline(0.5, color=INK, lw=0.4, ls=":", zorder=1)
    ax_c.set_xlim(0.4, 1.22)
    ax_c.set_xticks([0.4, 0.6, 0.8])
    ax_c.set_xlabel("Nearest-neighbor AUROC")
    # ax_c.text(1.22, rows[0] + 0.6, "with relative", va="bottom", ha="right", fontsize=note, clip_on=False)
    panel(ax_c, "g", "Random split", style["title"], style["letter"])

    ax_a.set_yticks(rows)
    ax_a.set_yticklabels([TASK_LABELS[task] for task in ALL_TASKS])
    ax_a.set_ylim(-0.6, len(ALL_TASKS) - 0.4)
    hide_row_labels(ax_b)
    hide_row_labels(ax_c)


def draw_sensitivity(ax, ax_b, style) -> None:
    scales = (("0.5", "thr_half", "o"), ("1", None, "^"), ("2", "thr_double", "s"))
    offsets = (-0.27, 0.0, 0.27)
    main_x = np.arange(len(RESULT_MODELS)).astype(float)
    restricted_x = len(RESULT_MODELS) + 0.8 + np.arange(len(RESTRICTED_KEYS)).astype(float)
    note = style["note"]

    released = (
        ("omegamp:broad", "OmegAMP\nantimicrobial"),
        ("hydramp:p_mic", "HydrAMP\nE. coli MIC"),
        ("hydramp:p_amp", "HydrAMP\nantimicrobial"),
    )
    views = (
        ("subset", "all rows", "matched labels", "o"),
        ("subset", "outside published data", "matched labels, outside data files", "^"),
        ("all", "all rows", "all labels (includes annotations)", "s"),
    )

    tables = {mode: pd.read_csv(TABLES / f"published_classifiers_{mode}.csv") for mode in ("subset", "all")}

    for (_, tag, marker), offset in zip(scales, offsets):
        table, head = macro_table("subset", tag), restricted_table("subset", tag)

        for position, (key, _, color) in zip(main_x, RESULT_MODELS):
            row = table.loc[key]
            interval(
                ax, row["peptide_macro"], position + offset, row["peptide_lo"], row["peptide_hi"], color,
                size=style["size"], width=0.6, cap=0.6, vertical=True, marker=marker,
            )

        for position, key in zip(restricted_x, RESTRICTED_KEYS):
            row = head.loc[key]
            interval(
                ax, row["macro"], position + offset, row["lo"], row["hi"], COLORS[key],
                size=style["size"], width=0.6, cap=0.6, vertical=True, marker=marker,
            )

    ax.axvline(len(RESULT_MODELS) + 0.15, color=GREY, lw=0.5, zorder=1)
    ax.text(len(RESULT_MODELS) + 0.3, 0.558, "At most 25 residues", fontsize=note, color=GREY, va="bottom", ha="left")
    ax.set_xticks(np.concatenate([main_x, restricted_x]))
    ax.set_xticklabels(
        [label for _, label, _ in RESULT_MODELS] + [LABELS[key] for key in RESTRICTED_KEYS],
        rotation=40, ha="right", rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, restricted_x[-1] + 0.6)
    ax.set_ylim(0.55, 0.81)
    ax.set_ylabel("Macro AUROC")
    ax.grid(axis="y", color=RULE, lw=0.4, zorder=0)
    ax.legend(
        handles=[dot(GREY, f"\u00d7{name}", marker=marker) for (name, _, marker) in scales],
        loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8,
        fontsize=note + 0.4, borderaxespad=0.1,
    )
    panel(ax, "d", "Thresholds, matched labels", style["title"], style["letter"])

    for group, (key, _) in enumerate(released):
        released_color = OMEGAMP if key.startswith("omegamp") else HYDRAMP

        for offset, (mode, view, _, marker) in zip((-0.3, 0.0, 0.3), views):
            table = tables[mode]
            part = table[(table["task"] == "antibacterial") & (table["score"] == key) & (table["view"] == view)]

            for model, color, shift, filled in ((key, released_color, -0.07, True), ("multitask_pertask", GREEN, 0.07, False)):
                row = part[part["model"] == model].iloc[0]
                interval(
                    ax_b, row["auroc"], group + offset + shift, row["lo"], row["hi"], color, filled=filled,
                    size=style["size"] - 0.2, width=0.6, cap=0.6, vertical=True, marker=marker,
                )

    ax_b.set_xticks(range(len(released)))
    ax_b.set_xticklabels([label for _, label in released], rotation=40, ha="right", rotation_mode="anchor")
    ax_b.set_xlim(-0.55, len(released) - 0.45)
    ax_b.set_ylim(0.45, 0.95)
    ax_b.axhline(0.5, color=INK, lw=0.4, ls=":", zorder=1)
    ax_b.set_ylabel("AUROC, antibact.")
    ax_b.grid(axis="y", color=RULE, lw=0.4, zorder=0)
    panel(ax_b, "h", "Released", style["title"], style["letter"])


def figure_results() -> None:
    width, height = WIDTH, 4.1
    tick, label_size = 5.4, 5.7
    style = {"note": 5.0, "title": 6.2, "letter": 8.0, "size": 2.4, "line": 0.6, "cap": 0.6, "diamond": 8, "dot": 5}

    fig = plt.figure(figsize=(width, height))
    place = placer(fig, width, height)

    # top band: the three results panels share the model rows; threshold sensitivity beside them
    top_one, tall_one = 0.25, 1.55
    ax_a = place(0.74, top_one, 0.95, tall_one)
    ax_b = place(1.80, top_one, 0.72, tall_one, sharey=ax_a)
    ax_c = place(2.64, top_one, 0.88, tall_one, sharey=ax_a)
    ax_d = place(3.95, top_one, 2.43, 1.35)

    # bottom band: provenance and nearest neighbor share the task rows; the released classifiers beside them
    top_two, tall_two = 2.52, 0.92
    ax_e = place(0.74, top_two, 1.30, tall_two)
    ax_f = place(2.16, top_two, 1.05, tall_two, sharey=ax_e)
    ax_g = place(3.33, top_two, 1.20, tall_two, sharey=ax_e)
    ax_h = place(4.92, top_two, 1.46, 0.68)

    draw_results(ax_a, ax_b, ax_c, style)
    draw_shortcuts(ax_e, ax_f, ax_g, style)
    draw_sensitivity(ax_d, ax_h, style)

    for ax in fig.axes:
        ax.tick_params(labelsize=tick)
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)

    box_axes(*fig.axes)

    def legend_at(handles, x, top, loc, columns):
        fig.legend(
            handles=handles, loc=loc, bbox_to_anchor=(x / width, 1.0 - top / height), ncol=columns, frameon=False,
            fontsize=5.0, handletextpad=0.3, columnspacing=1.0, labelspacing=0.2, borderaxespad=0.0,
        )

    # A to C, under them
    legend_at(
        [
            Line2D([], [], marker="o", color=GREY, lw=0.8, ms=3.2, label="peptide-weighted, 95% interval"),
            Line2D([], [], marker="D", mfc="white", mec=GREY, color=GREY, lw=0, ms=3.2, label="family-weighted"),
            Line2D([], [], marker="o", color=GREY, lw=0, ms=3.2, label="test fold"),
            Line2D([], [], marker="o", mfc="white", mec=GREY, color=GREY, lw=0, ms=3.2, label="fold 0, large component"),
            Patch(facecolor=GREY, alpha=0.3, label="fold density"),
            Line2D([], [], marker="|", color=GREY, lw=0, ms=6, mew=1.1, label="fold mean"),
            Line2D([], [], marker="D", color=GREY, lw=0, ms=3.2, label="fixed split"),
        ],
        0.74, 2.09, "upper left", 4,
    )

    # E to G, under them
    legend_at(
        [
            dot(PROVENANCE, "provenance only"),
            dot(BOOSTED, "best sequence model"),
            dot("#4D4D4D", "similarity-disjoint"),
            dot(PURPLE, "random split"),
        ],
        0.74, 3.74, "upper left", 4,
    )

    # H, under it and right aligned with it
    legend_at(
        [
            dot(OMEGAMP, "OmegAMP released"),
            dot(HYDRAMP, "HydrAMP released"),
            dot(GREEN, "Multi-task", filled=False),
            dot(GREY, "matched labels", marker="o"),
            dot(GREY, "matched, outside data", marker="^"),
            dot(GREY, "all labels", marker="s"),
        ],
        6.40, 3.70, "upper right", 2,
    )

    save(fig, "results")


FIGURES = {
    "generator": figure_generator,
    "results": figure_results,
}


def main() -> None:
    run_figures(FIGURES, "Main figures of the paper.")


if __name__ == "__main__":
    main()