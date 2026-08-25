"""Multi-task convolutional neural network.

The original design was a single-head cat-sound CNN. To translate *any*
animal we use one shared convolutional backbone with two output heads:

* ``species`` — which animal is vocalizing (softmax over all species)
* ``intent``  — what it means (softmax over the global intent vocabulary)

Sharing the backbone lets low-level acoustic features transfer across
species, while the two heads let a single model both identify the animal and
decode its message. New species/intents only change the head sizes.
"""
from __future__ import annotations

from .config import FeatureConfig
from .features import input_shape

try:  # TensorFlow is heavy; import lazily with a clear error.
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:  # pragma: no cover
    tf = None
    layers = None
    models = None


def _require_tf() -> None:
    if tf is None:
        raise ImportError(
            "TensorFlow is required to build/train the model. "
            "Install it with `pip install tensorflow`."
        )


def _conv_block(x, filters: int):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool -> Dropout."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    return x


def build_model(
    num_species: int,
    num_intents: int,
    config: FeatureConfig,
    base_filters: int = 32,
):
    """Build and compile the multi-task CNN.

    Returns an uncompiled-then-compiled ``tf.keras.Model`` with two outputs
    named ``species`` and ``intent``.
    """
    _require_tf()

    inputs = layers.Input(shape=input_shape(config), name="logmel")

    x = _conv_block(inputs, base_filters)
    x = _conv_block(x, base_filters * 2)
    x = _conv_block(x, base_filters * 4)
    x = _conv_block(x, base_filters * 8)

    # Global pooling keeps the head sizes independent of input length so clips
    # of different animals/durations all funnel into the same representation.
    x = layers.GlobalAveragePooling2D()(x)
    shared = layers.Dense(256, activation="relu")(x)
    shared = layers.Dropout(0.4)(shared)

    species_head = layers.Dense(128, activation="relu")(shared)
    species_out = layers.Dense(num_species, activation="softmax", name="species")(
        species_head
    )

    intent_head = layers.Dense(128, activation="relu")(shared)
    intent_out = layers.Dense(num_intents, activation="softmax", name="intent")(
        intent_head
    )

    model = models.Model(inputs=inputs, outputs=[species_out, intent_out],
                         name="bio_vocalism_translator")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "species": "categorical_crossentropy",
            "intent": "categorical_crossentropy",
        },
        # Intent decoding is the harder, more valuable task, so weight it higher.
        loss_weights={"species": 0.4, "intent": 1.0},
        metrics={"species": "accuracy", "intent": "accuracy"},
    )
    return model
