"""Dataset discovery and label encoding.

Two layouts are supported so you can bring data for *any* animal:

1. Directory layout (recommended)::

       dataset/
         cat/
           happy/     clip1.wav clip2.wav ...
           angry/     ...
         dog/
           alert/     ...
         frog/
           mating_call/ ...

2. Filename layout::

       cat__happy__whiskers_web.wav      # species__intent__free-text

Both produce ``(filepath, species, intent)`` samples that the trainer turns
into features and one-hot labels using the taxonomy's global label spaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .config import Taxonomy

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


@dataclass
class Sample:
    path: Path
    species: str
    intent: str


class LabelEncoder:
    """Maps species/intent strings to integer indices and back.

    The label spaces come from the taxonomy, so the model output layers are
    sized for every species/intent known to the taxonomy — add a new animal to
    the taxonomy and the encoder grows automatically.
    """

    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy
        self.species_classes = taxonomy.species_labels()
        self.intent_classes = taxonomy.intent_labels()
        self._species_index = {s: i for i, s in enumerate(self.species_classes)}
        self._intent_index = {t: i for i, t in enumerate(self.intent_classes)}

    @property
    def num_species(self) -> int:
        return len(self.species_classes)

    @property
    def num_intents(self) -> int:
        return len(self.intent_classes)

    def encode_species(self, species: str) -> int:
        return self._species_index[species]

    def encode_intent(self, intent: str) -> int:
        return self._intent_index[intent]

    def decode_species(self, index: int) -> str:
        return self.species_classes[index]

    def decode_intent(self, index: int) -> str:
        return self.intent_classes[index]

    def one_hot_species(self, species: str) -> np.ndarray:
        vec = np.zeros(self.num_species, dtype=np.float32)
        vec[self.encode_species(species)] = 1.0
        return vec

    def one_hot_intent(self, intent: str) -> np.ndarray:
        vec = np.zeros(self.num_intents, dtype=np.float32)
        vec[self.encode_intent(intent)] = 1.0
        return vec


def _parse_filename(path: Path) -> Sample | None:
    """Parse the ``species__intent__free-text`` filename convention."""
    parts = path.stem.split("__")
    if len(parts) < 2:
        return None
    species, intent = parts[0].strip().lower(), parts[1].strip().lower()
    if not species or not intent:
        return None
    return Sample(path=path, species=species, intent=intent)


def discover_samples(root: str | Path, taxonomy: Taxonomy | None = None) -> List[Sample]:
    """Find all labeled audio samples under ``root``.

    Directory layout is tried first (``root/species/intent/file``); files that
    do not fit are parsed with the filename convention. Samples whose labels
    are unknown to the taxonomy are skipped with the caller able to validate.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples: List[Sample] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        # Directory layout: .../<species>/<intent>/<file>
        rel = path.relative_to(root)
        if len(rel.parts) >= 3:
            species = rel.parts[-3].lower()
            intent = rel.parts[-2].lower()
            samples.append(Sample(path=path, species=species, intent=intent))
            continue

        parsed = _parse_filename(path)
        if parsed is not None:
            samples.append(parsed)

    if taxonomy is not None:
        samples = _filter_known(samples, taxonomy)
    return samples


def _filter_known(samples: List[Sample], taxonomy: Taxonomy) -> List[Sample]:
    """Keep only samples whose species/intent exist in the taxonomy."""
    known_species = set(taxonomy.species_labels())
    known_intents = set(taxonomy.intent_labels())
    kept = []
    for sample in samples:
        if sample.species in known_species and sample.intent in known_intents:
            kept.append(sample)
    return kept


def summarize(samples: List[Sample]) -> dict:
    """Return a per-species / per-intent count summary for logging."""
    summary: dict = {}
    for sample in samples:
        summary.setdefault(sample.species, {})
        summary[sample.species].setdefault(sample.intent, 0)
        summary[sample.species][sample.intent] += 1
    return summary
