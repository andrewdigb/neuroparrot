"""Training pipeline.

Discovers a labeled dataset, extracts log-mel features, and trains the
multi-task CNN. The result is saved as a self-contained *bundle* directory
holding the model weights, the feature configuration, and the taxonomy, so
inference can reproduce the exact preprocessing for any animal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .config import FeatureConfig, Taxonomy, TAXONOMY
from .dataset import LabelEncoder, Sample, discover_samples, summarize
from .features import extract_features
from .model import build_model

MODEL_FILENAME = "model.keras"
FEATURE_FILENAME = "feature_config.json"
TAXONOMY_FILENAME = "taxonomy.json"


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    validation_split: float = 0.2
    base_filters: int = 32
    seed: int = 1337


def _build_arrays(samples: List[Sample], encoder: LabelEncoder,
                  feature_config: FeatureConfig):
    """Extract features and one-hot labels for every sample."""
    features, species_labels, intent_labels = [], [], []
    for sample in samples:
        features.append(extract_features(sample.path, feature_config))
        species_labels.append(encoder.one_hot_species(sample.species))
        intent_labels.append(encoder.one_hot_intent(sample.intent))
    X = np.stack(features, axis=0)
    y_species = np.stack(species_labels, axis=0)
    y_intent = np.stack(intent_labels, axis=0)
    return X, y_species, y_intent


def _split(n: int, validation_split: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * validation_split)) if n > 1 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return train_idx, val_idx


def save_bundle(model, feature_config: FeatureConfig, taxonomy: Taxonomy,
                out_dir: str | Path) -> Path:
    """Persist model + preprocessing + taxonomy into one directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / MODEL_FILENAME)
    import json
    (out_dir / FEATURE_FILENAME).write_text(
        json.dumps(feature_config.to_dict(), indent=2), encoding="utf-8"
    )
    taxonomy.save(out_dir / TAXONOMY_FILENAME)
    return out_dir


def train(
    dataset_dir: str | Path,
    out_dir: str | Path = "translator_model",
    taxonomy: Taxonomy | None = None,
    feature_config: FeatureConfig | None = None,
    train_config: TrainConfig | None = None,
):
    """Train the translator on a dataset directory and save a bundle.

    Parameters
    ----------
    dataset_dir:
        Root of the labeled dataset (see :mod:`dataset` for layouts).
    out_dir:
        Where to write the trained model bundle.
    taxonomy:
        Species/intent taxonomy. Defaults to the built-in multi-species one;
        pass a custom taxonomy to support additional animals.
    """
    taxonomy = taxonomy or TAXONOMY
    feature_config = feature_config or FeatureConfig()
    train_config = train_config or TrainConfig()

    samples = discover_samples(dataset_dir, taxonomy)
    if not samples:
        raise ValueError(
            f"No labeled audio samples found under {dataset_dir}. "
            "Check your directory/filename layout and taxonomy."
        )

    print(f"Discovered {len(samples)} samples:")
    for species, intents in summarize(samples).items():
        total = sum(intents.values())
        print(f"  {species}: {total} clips across {len(intents)} intents")

    encoder = LabelEncoder(taxonomy)
    X, y_species, y_intent = _build_arrays(samples, encoder, feature_config)

    train_idx, val_idx = _split(len(samples), train_config.validation_split,
                                train_config.seed)

    model = build_model(
        num_species=encoder.num_species,
        num_intents=encoder.num_intents,
        config=feature_config,
        base_filters=train_config.base_filters,
    )
    model.summary()

    validation_data = None
    if len(val_idx) > 0:
        validation_data = (
            X[val_idx],
            {"species": y_species[val_idx], "intent": y_intent[val_idx]},
        )

    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_intent_accuracy" if validation_data else "intent_accuracy",
            mode="max", patience=8, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if validation_data else "loss",
            factor=0.5, patience=4, min_lr=1e-5,
        ),
    ]

    model.fit(
        X[train_idx],
        {"species": y_species[train_idx], "intent": y_intent[train_idx]},
        validation_data=validation_data,
        epochs=train_config.epochs,
        batch_size=train_config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    bundle = save_bundle(model, feature_config, taxonomy, out_dir)
    print(f"Saved translator bundle to: {bundle}")
    return bundle
