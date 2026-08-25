"""Audio feature extraction.

Converts a raw audio waveform (any animal, any recording) into a fixed-size
log-mel spectrogram suitable for a convolutional neural network. Using a
mel-spectrogram rather than raw MFCCs keeps enough spectral detail for the CNN
to learn species-specific patterns, and works uniformly across the very
different frequency ranges of, say, a frog vs. a bird.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .config import FeatureConfig

try:  # librosa is optional at import time so the package can be inspected.
    import librosa
except ImportError:  # pragma: no cover - handled with a clear runtime error.
    librosa = None


def _require_librosa() -> None:
    if librosa is None:
        raise ImportError(
            "librosa is required for audio feature extraction. "
            "Install it with `pip install librosa soundfile`."
        )


def load_waveform(path: str | Path, config: FeatureConfig) -> np.ndarray:
    """Load an audio file as a mono waveform at the configured sample rate."""
    _require_librosa()
    waveform, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    return waveform.astype(np.float32)


def fix_length(waveform: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Pad with silence or center-crop to the configured fixed duration."""
    target = int(config.sample_rate * config.duration_seconds)
    if len(waveform) == target:
        return waveform
    if len(waveform) < target:
        pad = target - len(waveform)
        left = pad // 2
        right = pad - left
        return np.pad(waveform, (left, right), mode="constant")
    # Longer than target: take the center window (usually the loudest content).
    start = (len(waveform) - target) // 2
    return waveform[start:start + target]


def waveform_to_logmel(waveform: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Compute a normalized log-mel spectrogram of shape (n_mels, n_frames, 1)."""
    _require_librosa()
    waveform = fix_length(waveform, config)

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax or config.sample_rate // 2,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Per-sample standardization keeps recordings with different gain/volume on
    # the same scale — important when mixing crowd-sourced clips of many animals.
    mean = log_mel.mean()
    std = log_mel.std()
    log_mel = (log_mel - mean) / (std + 1e-6)

    return log_mel[..., np.newaxis].astype(np.float32)


def extract_features(path: str | Path, config: FeatureConfig) -> np.ndarray:
    """End-to-end: file path -> CNN-ready log-mel spectrogram."""
    waveform = load_waveform(path, config)
    return waveform_to_logmel(waveform, config)


def batch_extract(paths: Iterable[str | Path], config: FeatureConfig) -> np.ndarray:
    """Extract features for many files into a single stacked array."""
    features = [extract_features(path, config) for path in paths]
    return np.stack(features, axis=0)


def input_shape(config: FeatureConfig) -> tuple[int, int, int]:
    """The CNN input shape for a given feature config."""
    return (config.n_mels, config.n_frames, 1)
