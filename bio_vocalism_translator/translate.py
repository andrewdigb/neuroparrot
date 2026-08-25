"""Inference and translation.

Loads a trained bundle and turns an audio clip from *any* animal into a
human-readable sentence, along with the detected species, intent, and
confidence scores.

A key detail: once the species head predicts an animal, the intent head is
*masked* to only the intents that are valid for that species. This keeps the
translation coherent (a frog can't be "purring") and lets one shared model
serve wildly different animals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json
import numpy as np

from .config import FeatureConfig, Taxonomy, UNKNOWN_INTENT
from .dataset import LabelEncoder
from .features import extract_features
from .train import FEATURE_FILENAME, MODEL_FILENAME, TAXONOMY_FILENAME


@dataclass
class TranslationResult:
    """The outcome of translating a single vocalization."""

    species: str
    species_display: str
    species_confidence: float
    intent: str
    intent_confidence: float
    sentence: str
    top_intents: List[tuple[str, float]]

    def __str__(self) -> str:
        return (
            f"[{self.species_display} "
            f"{self.species_confidence * 100:.0f}%] "
            f"{self.sentence} "
            f"(intent={self.intent}, {self.intent_confidence * 100:.0f}%)"
        )


class Translator:
    """High-level inference wrapper around a trained bundle."""

    def __init__(self, model, feature_config: FeatureConfig,
                 taxonomy: Taxonomy):
        self.model = model
        self.feature_config = feature_config
        self.taxonomy = taxonomy
        self.encoder = LabelEncoder(taxonomy)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, bundle_dir: str | Path) -> "Translator":
        """Load a translator from a bundle directory produced by training."""
        bundle_dir = Path(bundle_dir)
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TensorFlow is required for inference. "
                "Install it with `pip install tensorflow`."
            ) from exc

        model = tf.keras.models.load_model(bundle_dir / MODEL_FILENAME)
        feature_config = FeatureConfig.from_dict(
            json.loads((bundle_dir / FEATURE_FILENAME).read_text(encoding="utf-8"))
        )
        taxonomy = Taxonomy.load(bundle_dir / TAXONOMY_FILENAME)
        return cls(model, feature_config, taxonomy)

    # -------------------------------------------------------------- inference
    def _predict_arrays(self, features: np.ndarray):
        species_probs, intent_probs = self.model.predict(
            features[np.newaxis, ...], verbose=0
        )
        return species_probs[0], intent_probs[0]

    def _mask_intents_to_species(self, species: str,
                                 intent_probs: np.ndarray) -> np.ndarray:
        """Zero out intents that are invalid for the detected species."""
        valid = set(self.taxonomy.valid_intents_for(species))
        valid.add(UNKNOWN_INTENT)
        masked = np.zeros_like(intent_probs)
        for i, intent in enumerate(self.encoder.intent_classes):
            if intent in valid:
                masked[i] = intent_probs[i]
        total = masked.sum()
        if total <= 0:  # No overlap — fall back to the raw distribution.
            return intent_probs
        return masked / total

    def translate(self, audio_path: str | Path, top_k: int = 3) -> TranslationResult:
        """Translate a single audio file into a human-readable result."""
        features = extract_features(audio_path, self.feature_config)
        species_probs, intent_probs = self._predict_arrays(features)

        species_idx = int(np.argmax(species_probs))
        species = self.encoder.decode_species(species_idx)
        species_conf = float(species_probs[species_idx])

        masked = self._mask_intents_to_species(species, intent_probs)
        intent_idx = int(np.argmax(masked))
        intent = self.encoder.decode_intent(intent_idx)
        intent_conf = float(masked[intent_idx])

        order = np.argsort(masked)[::-1][:top_k]
        top_intents = [
            (self.encoder.decode_intent(int(i)), float(masked[int(i)]))
            for i in order if masked[int(i)] > 0
        ]

        sentence = self.taxonomy.translate(species, intent)
        return TranslationResult(
            species=species,
            species_display=self.taxonomy.display_name(species),
            species_confidence=species_conf,
            intent=intent,
            intent_confidence=intent_conf,
            sentence=sentence,
            top_intents=top_intents,
        )

    def translate_many(self, audio_paths: List[str | Path]) -> Dict[str, TranslationResult]:
        """Translate several files, keyed by path."""
        return {str(path): self.translate(path) for path in audio_paths}
