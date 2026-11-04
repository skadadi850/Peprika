"""
Train the masked discrete diffusion model

Usage:
    uv run python scripts/train_generator.py \
        --metadata data/processed/corpus/metadata.csv \
        --species-vocab data/processed/corpus/species_vocab.json \
        --splits data/processed/corpus/splits.csv \
        --output runs/final

When the split table has a split column (train, val, test) it defines the training and
validation rows and --holdout-fold is ignored; test rows are never trained on.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from ampx.compliance import read_sequences
from ampx.models.data import (
    LEGACY_CONTINUOUS_AXES,
    TASKS,
    WHEEL_PITCHES,
    WHEEL_SIZE,
    TokenDataset,
    clean_sequence,
    cluster_split,
    condition_from_batch,
    fit_wheel_prior,
    fit_wheel_thresholds,
    is_amp_sampler_weights,
    stratified_sampler_weights,
    task_sampler_weights,
)
from ampx.models.masked_diffusion import PAD_TOKEN, MaskedDiffusionModel

# Shape mismatches --init-from is allowed to hit. The species embedding
# table is data-dependent (pretrain and conditioned stages build their own
# species_vocab.json from different corpora). The condition encoder's output
# projection changes width whenever a conditioning axis is added, since it
# consumes one embedding block per axis; the new axis's own parameters are
# absent from an older checkpoint rather than mismatched, and land in
# `missing` instead.
EXPECTED_INIT_FROM_SKIPS = {
    "condition_encoder.categorical.species.weight",
    "condition_encoder.output_proj.0.weight",
}

# Validation draws its timesteps and masks from a fixed seed, restored afterwards, so the
# validation loss is comparable across epochs and does not consume the training RNG stream.
VALIDATION_SEED = 0


def load_compatible_state_dict(model: MaskedDiffusionModel, checkpoint_path: str, device: str, reinit_condition_proj: bool = False) -> None:
    """
    Load a checkpoint's state dict, keeping only keys whose shape matches
    the freshly constructed model. Used for stage 5b's --init-from, where
    num_species (and therefore the species embedding shape) differs between
    the pretrain and conditioned species vocabularies.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = dict(checkpoint["model_state_dict"])
    target_state = model.state_dict()
    if reinit_condition_proj:
        # Stage 5a pretrains with every sparse axis absent, so the condition
        # encoder's output projection learns to ignore those input slots.
        # Inheriting it into stage 5b starts the fine-tune from weights that
        # already discard MIC/HC50/species, and a short low-LR fine-tune does
        # not undo that. Dropping it forces the projection to relearn the
        # mapping with all axes present.
        dropped = [k for k in source_state if k.startswith("condition_encoder.output_proj")]
        for key in dropped:
            source_state.pop(key)
        print(f"--reinit-condition-proj: dropped {len(dropped)} keys: {dropped}")
    compatible = {}
    skipped = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing = [key for key in target_state if key not in compatible]
    model.load_state_dict(compatible, strict=False)
    print(f"--init-from {checkpoint_path}: loaded {len(compatible)} keys, skipped {len(skipped)}")
    if skipped:
        print(f"skipped keys (shape mismatch or absent in checkpoint): {skipped}")
    print(f"keys left at fresh initialization (not present in checkpoint): {missing}")
    if reinit_condition_proj:
        skipped = [k for k in skipped if not k.startswith("condition_encoder.output_proj")]
    unexpected = set(skipped) - EXPECTED_INIT_FROM_SKIPS
    if unexpected:
        raise SystemExit(
            "STOP: --init-from skipped keys other than the species embedding: "
            f"{sorted(unexpected)}. This is the flagged stop-and-report condition -- "
            "investigate before training from this checkpoint."
        )


class WeightAverage:
    # Exponential moving average of the parameters. Validation, task gain and every saved
    # checkpoint use the averaged copy; training updates the raw model.
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.steps = 0
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        self.steps += 1
        decay = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        for shadow, current in zip(self.model.parameters(), model.parameters()):
            shadow.mul_(decay).add_(current.detach(), alpha=1.0 - decay)


def task_gain(model, loader, device) -> float:
    model.eval()
    devices = [torch.cuda.current_device()] if torch.device(device).type == "cuda" else []
    total = 0.0
    batches = 0
    with torch.random.fork_rng(devices=devices):
        for index, batch in enumerate(loader):
            tokens = batch["tokens"].to(device)
            condition = condition_from_batch(batch, device)
            with torch.no_grad():
                torch.manual_seed(VALIDATION_SEED + index)
                with_task = model(tokens, condition, p_axis=0.0, p_all=0.0, aux_weight=0.0)
                torch.manual_seed(VALIDATION_SEED + index)
                without_task = model(tokens, condition, p_axis=0.0, p_all=0.0, aux_weight=0.0, p_tasks=1.0)
            total += float(without_task.item() - with_task.item())
            batches += 1
    return total / max(batches, 1)


def exact_epoch(model, loader, device) -> float:
    model.eval()
    total_log_probability = 0.0
    total_residues = 0
    for batch in tqdm(loader, desc="exact", leave=False):
        tokens = batch["tokens"].to(device)
        condition = condition_from_batch(batch, device)
        log_probability, _ = model.exact_log_likelihood(tokens, condition)
        total_log_probability += float(log_probability.sum().item())
        total_residues += int((tokens != PAD_TOKEN).sum().item())
    return -total_log_probability / max(total_residues, 1)


def run_epoch(
    model, loader, optimizer, device, p_axis, p_all, aux_weight, train,
    p_tasks=0.0, average=None, p_properties=0.0, p_measurements=0.0,
):
    model.train(train)
    total = 0.0
    batches = 0
    devices = [torch.cuda.current_device()] if torch.device(device).type == "cuda" else []
    context = contextlib.nullcontext() if train else torch.random.fork_rng(devices=devices)
    with context:
        if not train:
            torch.manual_seed(VALIDATION_SEED)
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
            tokens = batch["tokens"].to(device)
            condition = condition_from_batch(batch, device)
            if train:
                loss = model(
                    tokens, condition, p_axis=p_axis, p_all=p_all, aux_weight=aux_weight,
                    p_tasks=p_tasks, p_properties=p_properties, p_measurements=p_measurements,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if average is not None:
                    average.update(model)
            else:
                with torch.no_grad():
                    loss = model(tokens, condition, p_axis=0.0, p_all=0.0, aux_weight=0.0)
            total += float(loss.item())
            batches += 1
    return total / max(batches, 1)


def family_sampler_weights(metadata, splits, base, train_index, power) -> np.ndarray:
    # Rows are down-weighted by (rows in their homology component) ** -power. Rows without a
    # component, the background peptides, keep their weight. The total weight of the labeled
    # training rows is preserved, so the mix between labeled and background rows is unchanged.
    if splits is None or "component" not in splits:
        raise SystemExit("--balance-families needs --splits with a component column")
    keyed = metadata[["sequence"]].merge(
        splits[["sequence", "component"]].drop_duplicates(subset=["sequence"]), on="sequence", how="left"
    )
    component = keyed["component"].to_numpy(dtype=np.float64)
    in_family = ~np.isnan(component)
    labels, inverse, counts = np.unique(component[in_family], return_inverse=True, return_counts=True)
    rows = counts[inverse].astype(np.float64)
    base = np.asarray(base, dtype=np.float64)
    adjusted = base.copy()
    adjusted[in_family] = adjusted[in_family] * rows ** (-float(power))
    on_train = np.zeros(len(adjusted), dtype=bool)
    on_train[train_index] = True
    on_train &= in_family
    adjusted[in_family] *= base[on_train].sum() / adjusted[on_train].sum()
    in_largest = on_train & (component == labels[int(np.argmax(counts))])
    print(
        "family balancing: the largest component holds "
        f"{base[in_largest].sum() / base[on_train].sum():.3f} of the labeled sampling mass before "
        f"and {adjusted[in_largest].sum() / adjusted[on_train].sum():.3f} after",
        flush=True,
    )
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/processed/metadata.csv")
    parser.add_argument("--species-vocab", default="checkpoint/species_vocab.json")
    parser.add_argument("--splits", default=None)
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument(
        "--output", default="checkpoints",
        help="written into on every run, so training with the default overwrites the shipped "
             "masked_diffusion_best.pt and species_vocab.json. Pass an experiment directory to keep them.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--p-axis", type=float, default=0.15)
    parser.add_argument("--p-all", type=float, default=0.10)
    parser.add_argument(
        "--p-tasks", type=float, default=0.0,
        help="probability of hiding every task label of a row, so the model also learns the "
             "task-unconditional distribution that classifier-free guidance contrasts against",
    )
    parser.add_argument(
        "--p-properties", type=float, default=0.0,
        help="probability of hiding charge per residue, moment, GRAVY and cysteine class of a row "
             "together, so the model also learns what a task label implies without properties",
    )
    parser.add_argument(
        "--p-measurements", type=float, default=0.0,
        help="probability of hiding every measurement and species axis of a row together",
    )
    parser.add_argument(
        "--max-cysteines", type=int, default=8,
        help="generation limit for cys_class 3 and unset; cys_class 1 allows none and 2 allows up to MAX_CYSTEINES",
    )
    parser.add_argument(
        "--exact-limit", type=int, default=2000,
        help="number of validation rows used for the exact held-out likelihood",
    )
    parser.add_argument(
        "--balance-tasks", action="store_true",
        help="sample rows so that rare tasks are seen more often than their natural share",
    )
    parser.add_argument(
        "--task-balance-power", type=float, default=0.5,
        help="a task bucket of n rows receives sampling mass n ** power; 1 is the natural mix, 0 is uniform over buckets",
    )
    parser.add_argument(
        "--aux-weight", type=float, default=1.0,
        help="weight of the skip-ahead term; 0 trains the exact frontier objective only",
    )
    parser.add_argument(
        "--order", choices=["wheel", "forward", "random"], default="wheel",
        help="order in which residues are revealed",
    )
    parser.add_argument(
        "--wheel-thresholds", default=None,
        help="json file with one moment threshold per pitch; fitted on the is_amp_idx == 1 "
             "rows and written to the output directory when omitted",
    )
    parser.add_argument(
        "--wheel-negatives", default=None,
        help="FASTA of general peptides used to fit the wheel thresholds when the metadata "
             "has no is_amp_idx == 1 rows",
    )
    parser.add_argument("--wheel-quantile", type=float, default=0.9)
    parser.add_argument("--skip-exact", action="store_true", help="skip the exact held-out likelihood evaluation after training")
    parser.add_argument("--balance-species", action="store_true")
    parser.add_argument("--balance-is-amp", action="store_true", help="~3:1 negative:positive sampler by is_amp_idx, for pretraining")
    parser.add_argument(
        "--legacy-continuous-axes", action="store_true",
        help="train with the continuous axis list as it stood before gram_selectivity was added, "
             "so a metadata table built before that axis existed can still be used",
    )
    parser.add_argument(
        "--reinit-condition-proj", action="store_true",
        help="do not inherit the condition encoder's output projection from --init-from; "
             "reinitialize it so it must relearn how to use the sparse axes that were absent during pretraining",
    )
    parser.add_argument(
        "--init-from", default=None,
        help="checkpoint .pt to initialize from; shape-mismatched keys (expected: only the species embedding) are skipped",
    )
    parser.add_argument(
        "--samples-per-epoch", type=int, default=None,
        help="number of rows drawn per epoch by a weighted sampler; defaults to the number of training rows",
    )
    parser.add_argument(
        "--save-every", type=int, default=0,
        help="also write epoch_<n>.pt every this many epochs, so checkpoints can be compared on generation metrics",
    )
    parser.add_argument(
        "--balance-families", action="store_true",
        help="with --balance-tasks, weight rows by (rows in their homology component) ** "
             "-family-balance-power, using the component column of --splits",
    )
    parser.add_argument("--family-balance-power", type=float, default=0.5)
    parser.add_argument(
        "--ema-decay", type=float, default=0.0,
        help="decay of an exponential moving average of the weights; 0 disables it. Validation, "
             "task gain and all saved checkpoints then use the averaged weights",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.balance_families and not args.balance_tasks:
        raise SystemExit("--balance-families is applied inside --balance-tasks; pass both")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    species_vocab = json.loads(Path(args.species_vocab).read_text())

    if args.wheel_thresholds:
        wheel_thresholds = tuple(float(value) for value in json.loads(Path(args.wheel_thresholds).read_text()))
        if len(wheel_thresholds) != len(WHEEL_PITCHES):
            raise SystemExit(f"--wheel-thresholds must hold {len(WHEEL_PITCHES)} values")
    elif args.wheel_negatives:
        negatives = read_sequences(args.wheel_negatives)
        if len(negatives) > 50000:
            chosen = np.random.default_rng(0).choice(len(negatives), 50000, replace=False)
            negatives = [negatives[index] for index in chosen]
        print(f"fitting wheel thresholds on {len(negatives)} sequences from {args.wheel_negatives}")
        wheel_thresholds = fit_wheel_thresholds(negatives, args.wheel_quantile)
    else:
        if "is_amp_idx" in metadata and (metadata["is_amp_idx"] == 1).any():
            negatives = metadata.loc[metadata["is_amp_idx"] == 1, "sequence"].map(clean_sequence)
            print(f"fitting wheel thresholds on {len(negatives)} is_amp_idx == 1 rows")
        else:
            sources = [clean_sequence(sequence) for sequence in metadata["sequence"]]
            sources = [sequence for sequence in sources if sequence]
            if len(sources) > 50000:
                chosen = np.random.default_rng(0).choice(len(sources), 50000, replace=False)
                sources = [sources[index] for index in chosen]
            shuffler = np.random.default_rng(0)
            negatives = ["".join(shuffler.permutation(list(sequence))) for sequence in sources]
            print(f"no general-peptide rows; fitting wheel thresholds on {len(negatives)} shuffled copies of the training sequences")
        wheel_thresholds = fit_wheel_thresholds(negatives, args.wheel_quantile)

    (output / "wheel_thresholds.json").write_text(json.dumps(list(wheel_thresholds)))
    print(f"wheel thresholds: {wheel_thresholds}")

    splits = pd.read_csv(args.splits) if args.splits else None
    train_index, val_index = cluster_split(metadata, splits=splits, holdout_fold=args.holdout_fold)
    print(f"train {len(train_index)} rows, val {len(val_index)} rows")

    continuous_axes = LEGACY_CONTINUOUS_AXES if args.legacy_continuous_axes else None
    dataset = TokenDataset(metadata, species_vocab, continuous_axes=continuous_axes, wheel_thresholds=wheel_thresholds)

    classes = dataset.metadata["wheel_idx"].to_numpy(dtype=np.int64)
    mu_h = dataset.metadata["mu_h"].to_numpy(dtype=np.float64)
    wheel_prior = fit_wheel_prior(classes[train_index], mu_h[train_index], wheel_thresholds[0])
    print(f"wheel class counts: {np.bincount(classes[train_index], minlength=WHEEL_SIZE).tolist()}")

    epoch_size = args.samples_per_epoch or len(train_index)
    sampler = None
    shuffle = True
    if args.balance_species:
        weights = stratified_sampler_weights(dataset.metadata)[train_index]
        sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=epoch_size, replacement=True)
        shuffle = False
    elif args.balance_is_amp:
        weights = is_amp_sampler_weights(dataset.metadata)[train_index]
        sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=epoch_size, replacement=True)
        shuffle = False
    elif args.balance_tasks:
        weights = task_sampler_weights(dataset.metadata, args.task_balance_power)
        if args.balance_families:
            weights = family_sampler_weights(dataset.metadata, splits, weights, train_index, args.family_balance_power)
        weights = weights[train_index]
        sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=epoch_size, replacement=True)
        shuffle = False

    for name, rows in (("train", train_index), ("val", val_index)):
        subset = dataset.metadata.iloc[rows]
        counts = {
            task: (int((subset[f"act_{task}_idx"] == 2).sum()), int((subset[f"act_{task}_idx"] == 1).sum()))
            for task in TASKS
        }
        print(f"{name} rows (active, inactive) per task: {counts}")

    train_loader = DataLoader(
        Subset(dataset, train_index.tolist()),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_index.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = MaskedDiffusionModel(
        num_species=len(species_vocab) + 1,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_cysteines=args.max_cysteines,
        wheel_prior=wheel_prior,
        wheel_threshold=wheel_thresholds[0],
        order=args.order,
        continuous_axes=continuous_axes,
        device=args.device,
    )
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.init_from:
        load_compatible_state_dict(model, args.init_from, args.device, reinit_condition_proj=args.reinit_condition_proj)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(args.warmup_epochs, 1)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-6
    )

    average = WeightAverage(model, args.ema_decay) if args.ema_decay > 0 else None
    evaluated = average.model if average is not None else model

    history = {"train": [], "val": [], "gain": []}
    best = float("inf")
    best_gain = float("-inf")
    stale = 0
    track_gain = args.p_tasks > 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, args.device, args.p_axis, args.p_all,
            args.aux_weight, True, p_tasks=args.p_tasks, average=average,
            p_properties=args.p_properties, p_measurements=args.p_measurements,
        )
        val_loss = run_epoch(evaluated, val_loader, optimizer, args.device, 0.0, 0.0, 0.0, False)
        gain = task_gain(evaluated, val_loader, args.device) if track_gain else 0.0
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["gain"].append(gain)
        if epoch <= args.warmup_epochs:
            warmup.step()
        else:
            cosine.step()
        print(
            f"epoch {epoch}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"gain {gain:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        if args.save_every and epoch % args.save_every == 0:
            evaluated.save(str(output / f"epoch_{epoch:03d}.pt"))
        improved = False
        if val_loss < best:
            best = val_loss
            improved = True
            evaluated.save(str(output / "masked_diffusion_best.pt"))
        if track_gain and gain > best_gain:
            best_gain = gain
            improved = True
            evaluated.save(str(output / "masked_diffusion_best_gain.pt"))
        if improved:
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    evaluated.save(str(output / "masked_diffusion_final.pt"))
    (output / "species_vocab.json").write_text(json.dumps(species_vocab, indent=2))
    with open(output / "masked_diffusion_history.pkl", "wb") as handle:
        pickle.dump({"history": history, "args": vars(args), "best_val": best}, handle)

    print(f"best val loss {best:.4f}")
    print(f"wrote {output / 'masked_diffusion_best.pt'}")

    if not args.skip_exact:
        checkpoint = torch.load(str(output / "masked_diffusion_best.pt"), map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        exact_loader = DataLoader(
            Subset(dataset, val_index[: args.exact_limit].tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        exact = exact_epoch(model, exact_loader, args.device)
        print(f"exact held-out NLL per residue, conditional on length and condition: {exact:.4f}")


if __name__ == "__main__":
    main()