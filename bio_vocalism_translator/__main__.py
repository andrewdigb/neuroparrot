"""Command-line interface for the Bio-Digital Vocalism Translator.

Examples
--------
Train on a labeled dataset::

    python -m bio_vocalism_translator train --data ./dataset --out ./translator_model

Translate a clip from any animal::

    python -m bio_vocalism_translator translate --model ./translator_model --audio meow.wav

List the species/intents the taxonomy knows about::

    python -m bio_vocalism_translator taxonomy
"""
from __future__ import annotations

import argparse
import sys

from .config import TAXONOMY, FeatureConfig


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    feature_config = FeatureConfig(
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
    )
    train(
        dataset_dir=args.data,
        out_dir=args.out,
        feature_config=feature_config,
        train_config=train_config,
    )
    return 0


def _cmd_translate(args: argparse.Namespace) -> int:
    from .translate import Translator

    translator = Translator.load(args.model)
    result = translator.translate(args.audio, top_k=args.top_k)

    print(f"\nSpecies : {result.species_display} "
          f"({result.species_confidence * 100:.1f}%)")
    print(f"Meaning : {result.sentence}")
    print(f"Intent  : {result.intent} "
          f"({result.intent_confidence * 100:.1f}%)")
    if len(result.top_intents) > 1:
        print("Alternatives:")
        for intent, prob in result.top_intents[1:]:
            print(f"  - {intent}: {prob * 100:.1f}%")
    return 0


def _cmd_taxonomy(_: argparse.Namespace) -> int:
    for species in TAXONOMY.species_labels():
        tax = TAXONOMY.species[species]
        print(f"\n{tax.display_name} ({species}):")
        for intent, translation in tax.intents.items():
            print(f"  {intent:<16} -> {translation}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bio_vocalism_translator",
        description="Translate animal vocalizations into human language "
                    "using a multi-species neural network.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the translator on a dataset.")
    p_train.add_argument("--data", required=True, help="Dataset root directory.")
    p_train.add_argument("--out", default="translator_model",
                         help="Output bundle directory.")
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--val-split", type=float, default=0.2)
    p_train.add_argument("--sample-rate", type=int, default=22050)
    p_train.add_argument("--duration", type=float, default=4.0)
    p_train.set_defaults(func=_cmd_train)

    p_tr = sub.add_parser("translate", help="Translate an audio clip.")
    p_tr.add_argument("--model", required=True, help="Trained bundle directory.")
    p_tr.add_argument("--audio", required=True, help="Audio file to translate.")
    p_tr.add_argument("--top-k", type=int, default=3)
    p_tr.set_defaults(func=_cmd_translate)

    p_tax = sub.add_parser("taxonomy", help="Show the known species and intents.")
    p_tax.set_defaults(func=_cmd_taxonomy)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
