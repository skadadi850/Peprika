#!/usr/bin/env bash
# Downloads the weights and the corpus of Peprika from Zenodo, checks their checksums and unpacks them.
#   bash scripts/download.sh            weights and corpus
#   bash scripts/download.sh weights    weights only (about 344 MB)
#   bash scripts/download.sh data       corpus only
# The record id is set once in the next line, or per call with ZENODO_RECORD=<id> bash scripts/download.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RECORD=${ZENODO_RECORD:-ZENODO_RECORD_ID}
WHAT=${1:-all}

case "$RECORD" in
  ''|*[!0-9]*)
    echo "the Zenodo record id is not set: edit RECORD in this script or run ZENODO_RECORD=<id> bash scripts/download.sh"
    exit 1
    ;;
esac

BASE="https://zenodo.org/records/$RECORD/files"

checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

unpack() {
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$1"
  else
    python3 -m zipfile -e "$1" .
  fi
}

fetch() {  # $1 archive name, $2 checksum file in the repository
  local name=$1 sums=$2 expected actual

  [ -f "$sums" ] || { echo "missing $sums"; exit 1; }

  expected=$(cut -d' ' -f1 "$sums")
  mkdir -p dist

  if [ ! -f "dist/$name" ]; then
    echo "downloading $name"
    curl -fL --retry 3 -o "dist/$name.part" "$BASE/$name?download=1"
    mv "dist/$name.part" "dist/$name"
  fi

  actual=$(checksum "dist/$name")

  if [ "$actual" != "$expected" ]; then
    echo "checksum mismatch for $name"
    echo "expected $expected"
    echo "actual   $actual"
    rm -f "dist/$name"
    exit 1
  fi

  unpack "dist/$name"
  echo "$name unpacked"
}

case "$WHAT" in
  weights) fetch peprika_weights.zip weights.sha256 ;;
  data) fetch peprika_data.zip data.sha256 ;;
  all)
    fetch peprika_weights.zip weights.sha256
    fetch peprika_data.zip data.sha256
    ;;
  *)
    echo "usage: bash scripts/download.sh [weights|data|all]"
    exit 1
    ;;
esac
