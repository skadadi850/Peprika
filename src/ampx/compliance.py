from __future__ import annotations

from pathlib import Path


def read_fasta(path):
    headers, sequences, current = [], [], None

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()

        if not line:
            continue

        if line.startswith(">"):
            if current is not None:
                sequences.append("".join(current))

            headers.append(line[1:])
            current = []
        elif current is not None:
            current.append(line)

    if current is not None:
        sequences.append("".join(current))

    return headers, sequences


def write_fasta(sequences, path, prefix="seq") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f">{prefix}{index}\n{sequence}\n" for index, sequence in enumerate(sequences)))


def read_sequences(path):
    return read_fasta(path)[1]
