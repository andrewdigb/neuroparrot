"""Species-agnostic taxonomy and feature configuration.

The original project was cat-only. To support *any* animal we model the
problem as a taxonomy that maps every species to the set of vocalization
"intents" it can express, plus a human-readable translation for each
(species, intent) pair.

The taxonomy is data — not code — so new animals can be added without
touching the model. It can be extended at runtime or loaded from JSON, which
means the same neural network architecture generalizes to any species you
have data for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import json

# A special intent bucket used when a vocalization does not match a known
# class for the detected species. Keeps inference honest instead of forcing a
# wrong label.
UNKNOWN_INTENT = "unknown"


@dataclass
class FeatureConfig:
    """Audio feature-extraction settings shared by training and inference.

    The same configuration must be used at train and inference time, so it is
    serialized alongside the model.
    """

    sample_rate: int = 22050
    duration_seconds: float = 4.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int | None = None  # defaults to sample_rate / 2 when None

    @property
    def n_frames(self) -> int:
        """Number of time frames after fixed-length padding/cropping."""
        samples = int(self.sample_rate * self.duration_seconds)
        return samples // self.hop_length + 1

    def to_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureConfig":
        return cls(**data)


@dataclass
class SpeciesTaxonomy:
    """Vocalization intents for a single species.

    ``intents`` maps an intent key to a natural-language translation, e.g.
    ``{"hungry": "I'm hungry, feed me."}``.
    """

    species: str
    display_name: str
    intents: Dict[str, str] = field(default_factory=dict)

    def intent_keys(self) -> List[str]:
        return list(self.intents.keys())


@dataclass
class Taxonomy:
    """Full multi-species taxonomy.

    Provides the label spaces the neural network needs (one head for species,
    one head for intent) and the lookup used to turn predictions into a human
    sentence.
    """

    species: Dict[str, SpeciesTaxonomy] = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    def add_species(self, taxonomy: SpeciesTaxonomy) -> None:
        self.species[taxonomy.species] = taxonomy

    def add_intent(self, species: str, intent: str, translation: str,
                   display_name: str | None = None) -> None:
        """Register a new (species, intent) pair, creating the species if new.

        This is what makes the translator work for *any* animal: call it for a
        new species/intent and retrain, no architectural changes required.
        """
        if species not in self.species:
            self.species[species] = SpeciesTaxonomy(
                species=species,
                display_name=display_name or species.capitalize(),
            )
        self.species[species].intents[intent] = translation

    # ------------------------------------------------------------- label sets
    def species_labels(self) -> List[str]:
        return sorted(self.species.keys())

    def intent_labels(self) -> List[str]:
        """Global, sorted set of every intent across all species."""
        intents = {UNKNOWN_INTENT}
        for taxonomy in self.species.values():
            intents.update(taxonomy.intent_keys())
        return sorted(intents)

    def valid_intents_for(self, species: str) -> List[str]:
        taxonomy = self.species.get(species)
        if taxonomy is None:
            return [UNKNOWN_INTENT]
        return taxonomy.intent_keys() or [UNKNOWN_INTENT]

    # --------------------------------------------------------------- translate
    def translate(self, species: str, intent: str) -> str:
        taxonomy = self.species.get(species)
        if taxonomy is None:
            return f"Unrecognized species '{species}'."
        translation = taxonomy.intents.get(intent)
        if translation is None:
            return f"{taxonomy.display_name} vocalization (meaning unknown)."
        return translation

    def display_name(self, species: str) -> str:
        taxonomy = self.species.get(species)
        return taxonomy.display_name if taxonomy else species

    # ------------------------------------------------------------- (de)serialize
    def to_dict(self) -> dict:
        return {
            "species": {
                key: {
                    "species": tax.species,
                    "display_name": tax.display_name,
                    "intents": tax.intents,
                }
                for key, tax in self.species.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Taxonomy":
        taxonomy = cls()
        for key, tax in data.get("species", {}).items():
            taxonomy.species[key] = SpeciesTaxonomy(
                species=tax["species"],
                display_name=tax.get("display_name", key.capitalize()),
                intents=dict(tax.get("intents", {})),
            )
        return taxonomy

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Taxonomy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _default_taxonomy() -> Taxonomy:
    """A seed taxonomy spanning several species.

    Cat classes are kept (and mapped) from the original project; other species
    demonstrate that the design is not cat-specific. Extend freely.
    """
    taxonomy = Taxonomy()

    # ------------------------------------------------------------------- cat
    cat = SpeciesTaxonomy("cat", "Cat", {
        "warning": "I'm warning you — back off.",
        "angry": "I'm angry.",
        "leave_me_alone": "Leave me alone.",
        "fight": "Want to fight?",
        "happy": "I'm happy and content.",
        "hunt_play": "I want to hunt or play.",
        "seek_mate": "I'm looking for a mate.",
        "call_mother": "Mama! Mama!",
        "pain_hunger": "It hurts, or I'm hungry.",
        "comfortable": "I'm comfortable and relaxed.",
    })
    taxonomy.add_species(cat)

    # ------------------------------------------------------------------- dog
    dog = SpeciesTaxonomy("dog", "Dog", {
        "alert": "Alert! Something's here.",
        "playful": "Let's play!",
        "distress": "I'm anxious or in distress.",
        "aggressive": "Stay back, I'm serious.",
        "happy": "I'm happy to see you.",
        "attention": "Pay attention to me.",
    })
    taxonomy.add_species(dog)

    # ------------------------------------------------------------------- bird
    bird = SpeciesTaxonomy("bird", "Bird", {
        "song": "Courtship song — I'm advertising myself.",
        "alarm": "Predator alarm!",
        "contact": "Where are you? Staying in touch.",
        "begging": "Feed me, I'm a chick.",
        "territorial": "This is my territory.",
    })
    taxonomy.add_species(bird)

    # ------------------------------------------------------------------- cow
    cow = SpeciesTaxonomy("cow", "Cow", {
        "hunger": "I'm hungry.",
        "distress": "I'm distressed or separated.",
        "calf_call": "Calling my calf.",
        "content": "I'm calm and content.",
    })
    taxonomy.add_species(cow)

    # ------------------------------------------------------------------- frog
    frog = SpeciesTaxonomy("frog", "Frog", {
        "mating_call": "Mating call.",
        "release_call": "Let go — wrong partner.",
        "distress": "Distress scream.",
        "territorial": "Territorial call.",
    })
    taxonomy.add_species(frog)

    return taxonomy


# Default taxonomy used when no custom one is provided.
TAXONOMY = _default_taxonomy()
