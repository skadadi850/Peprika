#!/usr/bin/env bash
# End-to-end training of Peprika from the corpus in data/corpus (bash scripts/download.sh data).
#   bash scripts/train_all.sh                  every stage, generator seed 0
#   SEEDS="0 1 2" bash scripts/train_all.sh    three generator seeds
#   bash scripts/train_all.sh select export    only the named stages
# Stages, in order:
#   pretrain    generator on corpus peptides and bacterial small ORFs            about 2 hours on an RTX 4090
#   finetune    one run per seed from the pretrained checkpoint                  about 30 minutes per seed
#   select      every saved checkpoint of every seed is sampled and scored; the rule picks one per seed
#   export      copies the selected checkpoint of the first seed to weights/generator
#   predictors  boosted-tree oracles and the Multi-task networks into weights/
# A stage whose result exists is skipped; remove the result to rerun it.
# Retraining reproduces the recipe of the paper, not its weights bit for bit.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1 PYTHONPATH=src
PY=${PYTHON:-python}
C=data/corpus
SEEDS=${SEEDS:-"0"}
FT_EXTRA=${FT_EXTRA-"--ema-decay 0.999 --balance-families --family-balance-power 0.25"}

mkdir -p logs runs weights
exec > >(tee -a "logs/train_all_$(date +%Y%m%d_%H%M%S).log") 2>&1

for name in labels.csv sequences.csv splits.csv folds.csv species_vocab.json metadata_pretrain.csv metadata_finetune.csv; do
  [ -f "$C/$name" ] || { echo "missing $C/$name; run bash scripts/download.sh data"; exit 1; }
done

stage_pretrain() {
  if [ -f runs/pretrain/masked_diffusion_history.pkl ]; then echo "pretraining finished, skipped"; return; fi

  $PY scripts/train_generator.py --metadata $C/metadata_pretrain.csv --species-vocab $C/species_vocab.json --splits $C/splits.csv \
    --output runs/pretrain --epochs 60 --patience 15 --batch-size 256 --num-workers 4 --balance-is-amp --max-cysteines 8 \
    --skip-exact --seed 0 > logs/pretrain.log 2>&1
}

stage_finetune() {
  local s
  for s in $SEEDS; do
    if [ -f runs/final_s$s/masked_diffusion_history.pkl ]; then echo "seed $s finished, skipped"; continue; fi

    $PY scripts/train_generator.py --metadata $C/metadata_finetune.csv --species-vocab $C/species_vocab.json --splits $C/splits.csv \
      --output runs/final_s$s --wheel-thresholds runs/pretrain/wheel_thresholds.json \
      --init-from runs/pretrain/masked_diffusion_best.pt --reinit-condition-proj --epochs 150 --patience 15 --batch-size 256 \
      --num-workers 4 --lr 1e-4 --warmup-epochs 5 --balance-tasks --samples-per-epoch 60000 --save-every 2 --p-tasks 0.25 \
      --max-cysteines 8 --seed $s $FT_EXTRA > logs/finetune_s$s.log 2>&1
  done
}

stage_select() {  # the rule: lowest validation loss within the copy limits (at most 2% exact copies, at most 15% at similarity 0.7 or above)
  local s f list="" runs=""

  for s in $SEEDS; do
    runs="$runs $s=runs/final_s$s"

    for f in runs/final_s$s/epoch_*.pt runs/final_s$s/masked_diffusion_best.pt runs/final_s$s/masked_diffusion_final.pt; do
      list="$list $s:$(basename $f .pt)=$f"
    done
  done

  mkdir -p logs/select
  $PY scripts/evaluate_generator.py select --corpus $C --count 500 --settings 2.0:all --logs logs/select --checkpoints $list
  $PY scripts/evaluate_generator.py choose --evaluation logs/select/generator_evaluation.csv --runs $runs --output runs/selected_ckpt.txt
}

stage_export() {  # the first seed becomes weights/generator, which scripts/reproduce.py reads; the others go to weights/generator_s<seed>
  local s ck dest first=1

  [ -f runs/selected_ckpt.txt ] || { echo "runs/selected_ckpt.txt is missing; run the select stage"; exit 1; }

  while read -r s ck; do
    if [ $first = 1 ]; then dest=weights/generator; first=0; else dest=weights/generator_s$s; fi

    mkdir -p $dest
    cp "$ck" $dest/masked_diffusion.pt
    cp "$(dirname "$ck")/species_vocab.json" "$(dirname "$ck")/wheel_thresholds.json" $dest/
    echo "seed $s: $ck -> $dest"
  done < runs/selected_ckpt.txt
}

stage_predictors() {
  if [ -f weights/multitask/thresholds.csv ]; then echo "predictors finished, skipped"; return; fi

  $PY scripts/train_predictors.py --corpus $C --output weights
}

ALL="pretrain finetune select export predictors"

for n in ${@:-$ALL}; do
  echo "== $n"
  "stage_$n"
done

echo "== done; python scripts/reproduce.py recomputes the figures from weights/"
