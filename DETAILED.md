# Peprika: detailed documentation

This document describes everything needed to run, retrain and extend the release. `README.md` is the short version.

## 1. Repository contents

```
scripts/
  download.sh              weights and corpus from Zenodo, checksum verified
  download_weights.py      weights only, Python fallback of download.sh
  train_all.sh             end-to-end training (pretrain, fine-tune, select, export, predictors)
  train_generator.py       generator training (pretraining and fine-tuning)
  train_predictors.py      oracles and Multi-task networks
  evaluate_generator.py    sampling, property request, scoring, checkpoint selection, likelihood
  multitask.py             Multi-task network and macro benchmark code
  milk.py                  milk peptide screen and its figure
  reproduce.py             recomputes and redraws the main figures
  make_figures_main.py     generator figure and results figure
  make_figure_overview.py  overview figure
src/ampx/                  data.py (axes, features), masked_diffusion.py (model), evaluation.py, predictor.py, compliance.py
reference/                 train_properties.csv, test_properties.csv, corpus_summary.json
results/tables/            shipped tables read by the figures
results/samples*/          shipped samples (ours, OmegAMP, HydrAMP)
results/milk/              scores, thresholds, metadata and funnel of the milk screen
logs/                      cross-validation score files and the comparison files of reproduce.py
weights.sha256             checksum of peprika_weights.zip
data.sha256                checksum of peprika_data.zip
```

## 2. Environment

Tested with Python 3.10, one NVIDIA RTX 4090 (24 GB), and the versions in `requirements.txt`:
torch 2.10.0, transformers 5.8.1, esm 3.2.0, xgboost 3.2.0, numpy 1.26.4, pandas 2.2.2, scipy 1.14.1, rapidfuzz 3.14.5,
matplotlib 3.10.8, biopython 1.88. Install a torch build whose CUDA version matches your driver.

ESM-2 (`facebook/esm2_t12_35M_UR50D`, a pinned revision set in `src/ampx/predictor.py`) and ESM C (`esmc_300m`) are downloaded
from Hugging Face on first use, so the first run needs internet access. They are not redistributed.

GPU memory: one fine-tuning run needs about 8.6 GB. Sampling and scoring need less. Do not run other GPU jobs at the same time;
an out-of-memory error in training is almost always another process.

## 3. Data (Zenodo, `peprika_data.zip`)

`bash scripts/download.sh data` unpacks into `data/corpus/`:

| File | Content |
| --- | --- |
| `labels.csv` | one row per peptide: `sequence`, `act_<task>_idx`, `evidence_<task>`, `source_<task>`, `hc50_um`, `cc50_um`, `gram_selectivity_raw`, `is_amp_idx` |
| `sequences.csv` | `sequence`, `sources` (semicolon-separated source names) |
| `measurements.csv` | one row per deduplicated measurement, concentrations in micromolar |
| `splits.csv` | `sequence`, `component`, `split` (train, val, test), `max_train_similarity` |
| `folds.csv` | `sequence`, `component`, `fold` (0 to 9), `max_other_similarity` |
| `species_vocab.json` | species table of the generator (226 rows with the unobserved row) |
| `metadata_pretrain.csv` | corpus peptides and filtered bacterial small ORFs (background) for pretraining |
| `metadata_finetune.csv` | corpus measurement rows and the background, with the conditioning columns |
| `esm_embeddings.npz`, `esmc_embeddings.npz` | arrays `sequences` and `embeddings` for the corpus (ESM-2 35M, 480 dimensions, ESM C 300M, 960 dimensions) |
| `../MANIFEST.json`, `../corpus_sources.json` | counts, file hashes, source licences and the label rules |

The corpus has 47,119 peptides of 8 to 50 residues from eight public resources (DBAASP, DRAMP, GRAMPA, MLAMP_db, HemoPI-2,
AntiCP 2.0, AIPpred, AVPpred) with 115,450 deduplicated measurements.

**Label states.** `act_<task>_idx` is 0 for unknown, 1 for inactive and 2 for active. For hemolytic and cytotoxic, active means
harmful. An unknown label is never treated as inactive.

**Evidence codes** (`evidence_<task>`): 0 no label, 1 positive-only annotation, 2 measurement under the task rule, 3 explicit
curated label, 4 measured inactive overriding a positive-only annotation. A measured inactive overrides an annotation; a
contradiction gives unknown.

**Matched labels.** The benchmark and the oracles also use evidence-matched subsets: antibacterial and antifungal keep labels with
evidence 2 or 4, anticancer keeps peptides from the AntiCP 2.0 main set, antiviral keeps peptides from the AVPpred lists, and
the other tasks use all labels.

**Splits and folds.** Similarity is the normalized Levenshtein similarity with unit costs. Peptides at similarity 0.7 or more are
linked, and connected components are assigned whole. `splits.csv` is the fixed 80/10/10 split (37,670, 4,738 and 4,711 peptides).
`folds.csv` holds ten folds; fold k tests on fold k and stops on fold k+1. An exact audit found a maximum similarity below 0.7
between any held-out peptide and the training peptides, and between each fold and the others.

**Licences of the sources.** MLAMP_db is CC0; DBAASP and DRAMP are CC BY 4.0; HemoPI-2 and AntiCP 2.0 are GPL-3.0; GRAMPA
aggregates sources with mixed terms; the terms of AIPpred and AVPpred are not confirmed. `data/MANIFEST.json` records the
clearance of each source and counts, per task and label state, the labels supported only by sources with an open licence, by both
open and other sources, and only by other sources. Check it before redistributing any part of the corpus.

The background of `metadata_pretrain.csv` (268,879 bacterial small ORFs from sORFdb) cannot be regenerated from this repository.

## 4. Weights (Zenodo, `peprika_weights.zip`)

`bash scripts/download.sh weights` unpacks into `weights/`:

```
weights/generator/masked_diffusion.pt     generator checkpoint, fine-tuning seed 0, checkpoint chosen by the selection rule
weights/generator/species_vocab.json
weights/generator/wheel_thresholds.json   90th percentile of the wheel moment of background peptides, 100 and 180 degrees
weights/oracles/esm2_matched/<task>.ubj   boosted trees, six benchmark tasks, plus rounds.json
weights/oracles/esm2_alllabels/<task>.ubj
weights/oracles/esmc_matched/<task>.ubj
weights/multitask/seed0.pt ... seed4.pt   five networks, weights of the best macro epoch and of the best epoch of each task
weights/multitask/standardization.npz     mean and spread of the training rows
weights/multitask/config.json
weights/multitask/thresholds.csv          5% false-positive thresholds on validation inactives
```

The generator has 59,765,270 parameters: 12 adaptive-layer-norm transformer blocks (width 512, 8 heads, feed-forward width 2048),
a learned position embedding on 50 positions, and a conditioning network over 26 axes of 128 dimensions each.

## 5. Reproducing the figures

`python scripts/reproduce.py [--skip-generate] [--skip-milk] [--count N] [--device cuda]`

| Stage | What it does |
| --- | --- |
| generate | samples the unconditional set and the two property arms (density with label, density without label) for the six benchmark tasks, 5,000 per task and state, with the released generator; scores them with the three oracle families; replaces the seed 0 rows of copies of the shipped tables in `results/rerun/` |
| milk | rescores the milk peptides with the released networks and the stored thresholds into `results/rerun/milk/` |
| figures | draws the overview, the generator figure, the results figure (panels A to H) and the milk figure into `figures/` |

Outputs: `figures/overview.png|pdf`, `figures/fig-generator.png`, `figures/fig-results.png`, `figures/fig_milk.png|pdf` with
`fig_milk_values.csv` and `fig_milk_caption.tex`. The comparison of shipped and regenerated values is written to
`logs/reproduce_comparison.csv` and `logs/reproduce_milk_comparison.csv`.

With `--skip-generate --skip-milk` the figures are drawn from the shipped tables only. No weights, GPU or corpus are needed.

**Agreement with the paper** (full run on an RTX 4090): charge-per-residue Spearman 0.987 (shipped 0.987); six-task control with
the label 0.0218 and without 0.0210 for the ESM-2 matched oracles (shipped 0.0219 and 0.0212); the milk scores correlate with the
shipped scores at 0.99999999 and the numbers of peptides above each threshold are identical. About 96% of regenerated sequences
are identical to the shipped samples. The rest differ because sampling in bfloat16 is not bit-exact across kernels; the control
statistics are unaffected.

**What is not recomputed.** The generator figure and the results figure contain values of other methods: retrained and released
OmegAMP and HydrAMP, the baseline predictors (composition, properties, length, nearest neighbor, ESM embeddings with boosted
trees) and the cross-validated scores of the Multi-task network and the Ensemble. These are read from `results/tables` and
`logs`. Panels I and J of the results figure (ESMFold structures) are not drawn here; they need ESMFold 2 and the samples of every
generator.

## 6. Training from scratch

`bash scripts/train_all.sh` runs five stages and skips a stage whose result exists. `SEEDS="0 1 2"` trains three fine-tuning
seeds; the first seed becomes `weights/generator`, the others `weights/generator_s<seed>`. Logs go to `logs/`.

| Stage | Command (abridged) | Result |
| --- | --- | --- |
| pretrain | `train_generator.py --metadata metadata_pretrain.csv --epochs 60 --patience 15 --batch-size 256 --balance-is-amp --max-cysteines 8 --skip-exact --seed 0` | `runs/pretrain` |
| finetune | `train_generator.py --metadata metadata_finetune.csv --init-from runs/pretrain/masked_diffusion_best.pt --reinit-condition-proj --epochs 150 --patience 15 --lr 1e-4 --warmup-epochs 5 --balance-tasks --samples-per-epoch 60000 --save-every 2 --p-tasks 0.25 --ema-decay 0.999 --balance-families --family-balance-power 0.25 --seed <seed>` | `runs/final_s<seed>` |
| select | `evaluate_generator.py select`, then `choose` | `runs/selected_ckpt.txt` |
| export | copies the selected checkpoint | `weights/generator` |
| predictors | `train_predictors.py` | `weights/oracles`, `weights/multitask` |

**Pretraining** uses the training-split peptides and the background, AdamW with learning rate 3e-4, weight decay 0.01, 10 warmup
epochs with cosine decay, and fits the wheel thresholds. **Fine-tuning** reinitializes the output projection of the conditioning
network, drops activity labels jointly with probability 0.25, samples rows with task balance (a bucket of n rows receives mass
proportional to n to the power 0.5) and family balance (rows weighted by component size to the power -0.25), and averages the
weights with decay 0.999. Every axis except the antimicrobial flag is dropped with probability 0.15 and all axes jointly with
probability 0.10.

**Selection rule.** Every saved checkpoint is sampled (500 sequences per condition at guidance 2) and scored. Among checkpoints
with at most 2% exact copies of corpus peptides and at most 15% of samples at similarity 0.7 or above to a training peptide, the
one with the lowest validation loss is chosen per seed. The test split is never used to train, stop or select the generator.

**Reproducibility.** Seeds fix initialization, batch order and sampling, but GPU kernels are not deterministic and the
dropout masks of the current code draw extra random numbers. A retrained generator follows the paper's recipe and lands close to
the paper's validation loss (the paper's seed 0 stopped after 22 epochs at 2.1979), but its weights differ from the released
ones. Use the released weights to reproduce the paper's numbers.

## 7. Predictors

**Features** (1,861 dimensions): ESM C 960, ESM-2 480, composition and length 21, dipeptide fractions 400 (square-root
fractions, log length, standardized on the training rows and clipped at 5).

**Multi-task network:** hidden layers of 512 and 256 units (GELU, dropout 0.3), one logit per task for all ten tasks. The loss is a
class-balanced cross-entropy per task, averaged over tasks, using only observed labels. AdamW, learning rate 1e-3, weight decay
0.01, batches of 512, at most 100 epochs, patience 10. Five networks are averaged. Antibacterial, antifungal, anticancer,
anti-inflammatory and hemolytic each take the epoch of their own best validation AUROC; the other tasks take the epoch of the best
validation macro AUROC.

**Oracles:** boosted trees (depth 4, learning rate 0.05, subsample 0.8, column fraction 0.5, up to 600 rounds, early stopping
after 30 rounds on validation AUROC) on ESM-2 or ESM C embeddings with composition and dipeptide fractions, trained on the training
split. Test AUROCs of the ESM-2 matched family: antibacterial 0.891, antifungal 0.683, anticancer 0.804, anti-inflammatory 0.737,
hemolytic 0.790, antiviral 0.636. They measure resemblance to the labels of the corpus, not experimental activity.

## 8. Generator evaluation

For each task, 5,000 samples are requested as active and 5,000 as inactive. Control is the mean oracle probability of active for the
first set minus the second, averaged over six tasks. The arms differ only in the request: label only; density with label (charge
per residue, hydrophobic moment and GRAVY drawn from a kernel density fitted to the training peptides of the requested state);
density without label; pooled density with label. The label increment is the difference between the second and third arms.
`reference/train_properties.csv` holds the property vectors of the training split that the density is fitted on.

## 9. Milk peptide screen

The input is Table S2 of the supplement of the preterm infant gastric peptidomics paper (ACS, `pr8b00604_si_002.xlsx`). The
workbook is not redistributed. `results/milk/` holds the derived peptide table, the funnel counts, the scores and the thresholds.
The filters are fixed before scoring: species human, unmodified, 8 to 50 residues of the 20 standard amino acids, deduplicated,
and a maximum similarity below 0.7 to every corpus peptide for the candidate calls (below 0.5 for the experimental shortlist). A
peptide is a candidate for an activity if its score exceeds the 95th percentile of the scores of validation inactives, which needs
at least 30 validation inactives. All results are model predictions; none has been tested.

To rerun the filters from the workbook, download it from the publisher and use `python scripts/milk.py filter --workbook <path>`
with a corpus in `data/corpus`.

## 10. Limitations

- Every charge value uses a definition without termini, with histidine at +0.1 and no pH.
- The label thresholds are design choices. An active label needs one measurement at or below its threshold and an inactive label
  needs all measurements at or above it.
- Folds are disjoint at similarity 0.7, which does not exclude shared short k-mers.
- The Multi-task configuration was selected on validation folds that rotate into other test folds.
- Oracle-scored control reflects length, charge and hydropathy; it is not experimental activity.

## 11. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `the Zenodo record id is not set` | edit `RECORD` in `scripts/download.sh` or run `ZENODO_RECORD=<id> bash scripts/download.sh` |
| `checksum mismatch` | the download was incomplete or changed; delete `dist/` and run again |
| CUDA out of memory | another process holds the GPU; check `nvidia-smi`; fine-tuning needs about 8.6 GB |
| ESM download fails | the first run needs access to Hugging Face; later runs can set `HF_HUB_OFFLINE=1` |
| a figure fails in `reproduce.py` | the traceback is in the log; the other figures are still drawn |
| `missing data/corpus/...` in `train_all.sh` | run `bash scripts/download.sh data` |

## 12. Citation

See `README.md`. Cite the Zenodo record for the weights and the corpus, and the sources of the corpus for the labels you use.
