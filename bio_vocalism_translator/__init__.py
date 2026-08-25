"""Bio-Digital Vocalism Translator.

A neural-network powered translator that converts animal vocalizations into
human-readable intent/emotion for *any* animal species, not just cats.

The pipeline is:

    audio -> features (log-mel spectrogram) -> multi-task CNN -> (species, intent)
          -> human-readable translation

Public API::

    from bio_vocalism_translator import Translator

    translator = Translator.load("model.keras")
    result = translator.translate("meow.wav")
    print(result.sentence)
"""
from .config import TAXONOMY, Taxonomy, SpeciesTaxonomy, FeatureConfig
from .translate import Translator, TranslationResult

__all__ = [
    "TAXONOMY",
    "Taxonomy",
    "SpeciesTaxonomy",
    "FeatureConfig",
    "Translator",
    "TranslationResult",
]

__version__ = "1.0.0"
