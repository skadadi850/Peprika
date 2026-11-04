"""
Recomputes and redraws the main figures of the paper for Peprika from the released weights.

generate   Samples seed 0 of the generator (unconditional and the two property arms, 5,000 per condition), scores the
           samples with the three oracle families, and replaces the seed 0 rows of the shipped tables. The shipped
           values are printed next to the new ones.
milk       Rescores the milk peptides with the released Multi-task networks and the stored 5% false-positive thresholds.
figures    Draws the overview, generator, results and milk figures into figures/.

Everything that needs corpus sequences, the other methods and the cross-validation scores of the predictors is read from
the shipped tables.

Usage:
    python scripts/download_weights.py
    PYTHONUNBUFFERED=1 python scripts/reproduce.py > logs/reproduce.log 2>&1
    PYTHONUNBUFFERED=1 python scripts/reproduce.py --skip-generate --skip-milk > logs/reproduce_figures.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_generator as eg  # noqa: E402
from ampx.compliance import read_fasta  # noqa: E402
from ampx.evaluation import combined_features  # noqa: E402
from ampx.models.data import TASKS, is_valid_sequence  # noqa: E402
from ampx.models.masked_diffusion import MaskedDiffusionModel  # noqa: E402
from ampx.predictor import ESM2Encoder, ESMCEncoder, condition_matrix, resolve_device, sample_library  # noqa: E402
from multitask import MultiTaskNetwork, predict  # noqa: E402

WEIGHTS = ROOT / "weights"
RUN = "gen_s0"
FAMILIES = ("esm2_matched", "esm2_alllabels", "esmc_matched")
STEERING = {"esm2_matched": "steering_matched.csv", "esm2_alllabels": "steering_alllabels.csv", "esmc_matched": "steering_esmc.csv"}
SIX = ("antibacterial", "antifungal", "anticancer", "antiinflammatory", "hemolytic", "antiviral")
FOUR = ("antibacterial", "antifungal", "anticancer", "hemolytic")
HIT_COLUMNS = ["mean_abs_error_sd", "shuffled_mean_abs_error_sd", "bias_sd", "spearman", "exact_fraction"]


class SavedOracle:
    def __init__(self, path: Path, rounds: int):
        self.booster = xgb.Booster()
        self.booster.load_model(str(path))
        self.rounds = int(rounds)

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(features), iteration_range=(0, self.rounds))


def load_oracles(family: str) -> dict:
    directory = WEIGHTS / "oracles" / family
    rounds = json.loads((directory / "rounds.json").read_text())

    return {task: SavedOracle(directory / f"{task}.ubj", count) for task, count in rounds.items()}


def update_steering(path: Path, model: str, control: dict) -> float:
    table = pd.read_csv(path).set_index("model")
    before = float(table.loc[model, "mean_six_tasks"]) if model in table.index else float("nan")
    row = {}

    for column in table.columns:
        if column in control:
            row[column] = round(float(control[column]), 3)
        elif column == "mean_four_tasks":
            row[column] = round(float(np.mean([control[task] for task in FOUR])), 4)
        elif column == "mean_six_tasks":
            row[column] = round(float(np.mean([control[task] for task in SIX])), 4)
        else:
            row[column] = np.nan

    table.loc[model] = pd.Series(row)
    table.to_csv(path)

    return before


def update_property_hit(path: Path, rows: list) -> dict:
    new = pd.DataFrame(rows).groupby(["model", "property"])[HIT_COLUMNS].mean().round(3).reset_index()
    old = pd.read_csv(path)
    columns = list(old.columns)
    before = old.set_index(["model", "property"])["spearman"].to_dict()
    pd.concat([old[~old["model"].isin(new["model"])], new], ignore_index=True)[columns].to_csv(path, index=False)

    return before


def sample_arms(args, device: str, samples_dir: Path, comparison: list) -> list:
    train = pd.read_csv(ROOT / "reference" / "train_properties.csv")
    reference = train[["length", "charge_per_residue", "mu_h", "gravy_raw"]]
    pool = reference["length"].to_numpy().astype(np.int64)
    model = MaskedDiffusionModel.load(str(WEIGHTS / "generator" / "masked_diffusion.pt"), device=device)
    vocabulary = json.loads((WEIGHTS / "generator" / "species_vocab.json").read_text())
    dtype = torch.bfloat16 if device == "cuda" else None
    rows = []

    for name, labels in ((f"{RUN}_pf_state", True), (f"{RUN}_pf_state_nolabel", False)):
        index = 0
        identical = []

        for task_index, task in enumerate(eg.BENCHMARK_TASKS):
            lengths = np.random.default_rng(task_index).choice(pool, args.count)

            for state in ("active", "inactive"):
                density = eg.fit_density(reference[(train[f"act_{task}_idx"] == eg.STATE_CODES[state]).to_numpy()], None)
                vectors = eg.draw_vectors(density, args.count, task_index)
                condition = condition_matrix(eg.build_requests(task, state, vectors, lengths, labels), [1] * args.count, vocabulary, device)
                samples = sample_library(
                    model, condition, guidance_weight=2.0, steps=128, temperature=1.0, batch_size=512, seed=index,
                    autocast_dtype=dtype, guidance_null_axes=None,
                )
                index += 1
                label = f"{task}_{state}"
                eg.write_samples(samples_dir, name, label, samples)

                saved = pd.DataFrame(vectors, columns=eg.HIT_PROPERTIES)
                saved.insert(0, "length", lengths)
                saved["sequence"] = samples
                saved.to_csv(samples_dir / f"{name}__{label}.requests.tsv", sep="\t", index=False)

                shipped = ROOT / "results" / "samples" / f"{name}__{label}.requests.tsv"

                if shipped.exists():
                    old = pd.read_csv(shipped, sep="\t")["sequence"].fillna("").tolist()

                    if len(old) == len(samples):
                        identical.append(float(np.mean([first == second for first, second in zip(old, samples)])))

                for row in eg.property_hit(vectors, lengths, eg.property_table(samples), reference, 0):
                    rows.append({"model": name, "task": task, "state": state, **row})

        if identical:
            comparison.append({"quantity": f"{name}: share of sequences identical to the shipped samples", "shipped": 1.0, "regenerated": float(np.mean(identical))})

    samples = eg.generate(model, vocabulary, {"is_amp": "amp"}, pool, args.count, (2.0, None), 128, 512, 0, device)
    eg.write_samples(samples_dir, RUN, "unconditional", samples)
    del model

    if device == "cuda":
        torch.cuda.empty_cache()

    return rows


def score_arms(device: str, samples_dir: Path) -> dict:
    encoder = ESM2Encoder(device=device)
    esmc_encoder = ESMCEncoder(device=device)
    families = {family: load_oracles(family) for family in FAMILIES}
    control = {family: {} for family in FAMILIES}

    for name in (f"{RUN}_pf_state", f"{RUN}_pf_state_nolabel"):
        means = {family: {} for family in FAMILIES}

        for task in eg.BENCHMARK_TASKS:
            for state in ("active", "inactive"):
                raw = read_fasta(samples_dir / f"{name}__{task}_{state}.fasta")[1]
                cleaned = (str(item).upper().strip() for item in raw)
                unique = list(dict.fromkeys(item for item in cleaned if is_valid_sequence(item)))

                if len(unique) < eg.MIN_SEQUENCES:
                    raise SystemExit(f"{name} {task} {state}: {len(unique)} valid unique sequences")

                series = pd.Series(unique)
                features = {"esm2_matched": combined_features(series, encoder.encode(unique, batch_size=256))}
                features["esm2_alllabels"] = features["esm2_matched"]
                features["esmc_matched"] = combined_features(series, esmc_encoder.encode(unique, batch_size=128))

                for family in FAMILIES:
                    means[family][(task, state)] = float(families[family][task].score(features[family]).mean())

            print(f"{name} {task} scored", flush=True)

        for family in FAMILIES:
            control[family][name] = {task: means[family][(task, "active")] - means[family][(task, "inactive")] for task in eg.BENCHMARK_TASKS}

    return control


def generate(args, device: str, tables_dir: Path, samples_dir: Path) -> None:
    comparison = []
    rows = sample_arms(args, device, samples_dir, comparison)
    before = update_property_hit(tables_dir / "property_hit.csv", rows)
    now = pd.read_csv(tables_dir / "property_hit.csv").set_index(["model", "property"])["spearman"]
    key = (f"{RUN}_pf_state", "charge_per_residue")
    comparison.append({"quantity": "charge per residue, Spearman", "shipped": before.get(key, float("nan")), "regenerated": float(now.loc[key])})

    control = score_arms(device, samples_dir)

    for family in FAMILIES:
        for name, values in control[family].items():
            shipped = update_steering(tables_dir / STEERING[family], name, values)
            comparison.append({"quantity": f"{family} {name}: control, six-task mean", "shipped": shipped, "regenerated": float(np.mean([values[task] for task in SIX]))})

    table = pd.DataFrame(comparison)
    table.to_csv(ROOT / "logs" / "reproduce_comparison.csv", index=False)
    print(table.round(4).to_string(index=False), flush=True)


def rescore_milk(args, device: str, milk_dir: Path) -> None:
    import milk

    config = json.loads((WEIGHTS / "multitask" / "config.json").read_text())
    shipped = pd.read_csv(ROOT / "results" / "milk" / "peptidomics_scores.csv")
    sequences = shipped["sequence"].tolist()

    esm2 = ESM2Encoder(device=device).encode(sequences, batch_size=256)
    esmc = ESMCEncoder(device=device).encode(sequences, batch_size=128)
    raw = milk.milk_input_features(sequences, esmc, esm2, config["transform"], config["extra"])
    standardization = np.load(WEIGHTS / "multitask" / "standardization.npz")
    scaled = np.clip((raw - standardization["mean"]) / standardization["spread"], -config["clip"], config["clip"]).astype(np.float32)
    features = torch.as_tensor(scaled, device=device)

    network = MultiTaskNetwork(config["inputs"], config["outputs"], config["hidden"], config["dropout"], 0.0).to(device)
    averaged = np.zeros((len(sequences), config["outputs"]), dtype=np.float64)

    for seed in range(config["seeds"]):
        states = torch.load(WEIGHTS / "multitask" / f"seed{seed}.pt", map_location=device)
        network.load_state_dict(states["macro"])
        merged = predict(network, features)

        for column, state in states["columns"].items():
            network.load_state_dict(state)
            merged[:, column] = predict(network, features)[:, column]

        averaged += merged / config["seeds"]

    thresholds = pd.read_csv(WEIGHTS / "multitask" / "thresholds.csv")
    result = pd.DataFrame({"sequence": sequences, "max_sim_corpus": shipped["max_sim_corpus"].to_numpy()})
    comparison = []

    for task in milk.ACTIVITY_TASKS:
        result[f"score_{task}"] = averaged[:, TASKS.index(task)]
        threshold = float(thresholds.set_index("task").loc[task, "threshold"])

        if np.isfinite(threshold):
            result[f"above_{task}"] = result[f"score_{task}"] > threshold
            comparison.append(
                {
                    "task": task,
                    "spearman_to_shipped": float(stats.spearmanr(result[f"score_{task}"], shipped[f"score_{task}"]).statistic),
                    "above_shipped": int(shipped[f"above_{task}"].sum()),
                    "above_regenerated": int(result[f"above_{task}"].sum()),
                }
            )

    milk_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(milk_dir / "peptidomics_scores.csv", index=False)
    thresholds.to_csv(milk_dir / "peptidomics_thresholds.csv", index=False)

    for name in ("peptidomics_metadata.csv", "funnel.csv"):
        shutil.copy2(ROOT / "results" / "milk" / name, milk_dir / name)

    table = pd.DataFrame(comparison)
    table.to_csv(ROOT / "logs" / "reproduce_milk_comparison.csv", index=False)
    print(table.round(4).to_string(index=False), flush=True)


def draw(args, tables_dir: Path, samples_dir: Path, milk_dir: Path) -> list:
    failed = []

    def run(name, function):
        try:
            function()
        except (Exception, SystemExit):
            failed.append(name)
            print(f"figure {name} failed:", flush=True)
            traceback.print_exc()
            sys.stderr.flush()

    import make_figures_main as figures

    figures.set_style()
    figures.TABLES = tables_dir
    figures.SAMPLES = samples_dir
    figures.OUTPUT = ROOT / "figures"
    figures.PUBLISHED_SETS = tuple(
        (label, samples_dir / "gen_s0__unconditional.fasta" if label == "Ours, seed 0" else path, color)
        for label, path, color in figures.PUBLISHED_SETS
    )
    run("generator", figures.figure_generator)
    run("results", figures.figure_results)

    def overview():
        import make_figure_overview

        make_figure_overview.main()

    def milk_figure():
        import milk

        milk.MILK = milk_dir
        milk.run_activity_figure(
            argparse.Namespace(
                view="novel", view_ceiling=0.7, structure=str(milk_dir / "peptidomics_structure.csv"), task=None,
                draws=1000, seed=0, colors="activity", output=str(ROOT / "figures"),
            )
        )

    run("overview", overview)
    run("milk", milk_figure)
    print("failed figures:", failed if failed else "none", flush=True)

    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw the main figures of Peprika.")
    parser.add_argument("--count", type=int, default=5000, help="samples per condition")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-milk", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    if not (args.skip_generate and args.skip_milk) and not WEIGHTS.exists():
        raise SystemExit("weights/ is missing; run scripts/download_weights.py")

    rerun = ROOT / "results" / "rerun"
    tables_dir, samples_dir, milk_dir = ROOT / "results" / "tables", ROOT / "results" / "samples", ROOT / "results" / "milk"

    if not args.skip_generate:
        tables_dir, samples_dir = rerun / "tables", rerun / "samples"
        shutil.copytree(ROOT / "results" / "tables", tables_dir, dirs_exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        generate(args, device, tables_dir, samples_dir)

    if not args.skip_milk:
        milk_dir = rerun / "milk"
        rescore_milk(args, device, milk_dir)

    draw(args, tables_dir, samples_dir, milk_dir)


if __name__ == "__main__":
    main()
