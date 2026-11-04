# Peprika

Multi-axis masked diffusion in helical-wheel order for bioactive peptide generation, with an evidence-aware corpus and a
similarity-disjoint evaluation.

Peprika is a masked discrete diffusion generator for peptides of 8 to 50 residues. One network is conditioned on ten
activity labels (antibacterial, antifungal, anticancer, antiviral, anti-inflammatory, hemolytic, cytotoxic,
antitubercular, antibiofilm, antiparasitic) and on charge per residue, hydrophobic moment and GRAVY. It reveals residues in
helical-wheel order. This repository holds the code that recomputes the main figures of the paper, the code that trains the
generator and the predictors from scratch, and the tables behind the figures. Weights and the corpus are on Zenodo.

Paper: Multi-Axis Masked Diffusion in Helical-Wheel Order for Bioactive Peptide Generation.
Data and weights: https://doi.org/10.5281/zenodo.ZENODO_RECORD_ID

## What it reproduces

| Output | Command | Needs |
| --- | --- | --- |
| Figure 1 (overview), generator figure, results figure A to H, milk figure from the shipped tables | `python scripts/reproduce.py --skip-generate --skip-milk` | repository only |
| Generator control and property hit rebuilt from the released weights, milk screen rescored | `python scripts/reproduce.py` | weights, one GPU |
| Generator, oracles and Multi-task networks trained from the corpus | `bash scripts/train_all.sh` | corpus, one GPU |

Values of the other methods (OmegAMP, HydrAMP and the baseline predictors) are stored in `results/tables` and `logs`; they are
not recomputed here.

## Quick start

```bash
git clone https://github.com/skadadi850/Peprika.git
cd peprika
python -m venv .venv && source .venv/bin/activate     # Python 3.10 or newer
pip install -r requirements.txt                       # install the CUDA build of torch that matches your driver
bash scripts/download.sh weights                      # about 344 MB, checksum verified
python scripts/reproduce.py                           # figures are written to figures/
```

The first run downloads ESM-2 (35M) and ESM C (300M) from Hugging Face. A GPU with 24 GB (RTX 4090) was used for everything
here.

## Train from scratch

```bash
bash scripts/download.sh data      # the corpus, about 47,000 peptides with labels, splits, folds and embeddings
bash scripts/train_all.sh          # pretrain, fine-tune, select, export, predictors
python scripts/reproduce.py
```

Pretraining takes about two hours and one fine-tuning run about half an hour on an RTX 4090. Retraining follows the recipe of
the paper; it does not reproduce the released weights bit for bit.

## Layout

| Path | Content |
| --- | --- |
| `scripts/` | download, training, reproduction and figure scripts |
| `src/ampx/` | model, features, evaluation and sampling code |
| `reference/` | property vectors of the training and test splits and corpus counts, without sequences |
| `results/` | shipped tables, samples and milk scores that the figures read |
| `logs/` | cross-validation scores and the comparison files written by `reproduce.py` |
| `weights.sha256`, `data.sha256` | checksums of the Zenodo archives |

`DETAILED.md` documents the data, the weights, every script and its options, the figure inputs and the known limitations.

## Citation

```bibtex
@misc{kadadi2026peprika,
  title  = {Multi-Axis Masked Diffusion in Helical-Wheel Order for Bioactive Peptide Generation},
  author = {Kadadi, Simran and Verma, Shourya and Wang, Mengbo and Jayasundara, Shyaman and Grama, Ananth},
  year   = {2026},
  note   = {Code, corpus and weights: https://doi.org/10.5281/zenodo.ZENODO_RECORD_ID}
}
```

Update the entry with the venue once the paper is published. Please also cite the sources of the corpus that your work uses;
they are listed in `DETAILED.md`.

## License

See `LICENSE` for the code. The corpus combines public resources with their own terms, listed in `DETAILED.md` and in
`data/MANIFEST.json` after `bash scripts/download.sh data`.
