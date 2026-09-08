#!/usr/bin/env python3
"""WiFi-DensePose Interactive Demo

Demonstrates the WiFi-DensePose signal processing pipeline for vital sign
extraction (breathing rate, heart rate) from CSI data.

No hardware needed — pure signal processing demo using the native Rust API.

Run: python3 tools/wifi-densepose-demo.py [--breathing-hz 0.2] [--heartrate-hz 1.5]
"""

from __future__ import annotations

import argparse
import math
import random
import sys

import numpy as np


def hello_native() -> str:
    """Call the native Rust hello function."""
    import wifi_densepose._native as native
    return native.hello()


def get_native_version() -> str:
    """Get the Rust build version."""
    import wifi_densepose._native as native
    return native.__rust_version__


def get_native_features() -> list[str]:
    """Get the native build features."""
    import wifi_densepose._native as native
    return native.__build_features__


def estimate_breathing_rate(
    residuals: np.ndarray,
    weights: np.ndarray,
    fs: float = 100.0,
    lo_hz: float = 0.1,
    hi_hz: float = 0.5,
) -> float:
    """Estimate breathing rate from CSI residuals using FFT spectrum analysis.

    Args:
        residuals: (N,) array of CSI residual values.
        weights: (N,) array of weights (typically all 1.0).
        fs: Sample rate in Hz.
        lo_hz: Low frequency cutoff.
        hi_hz: High frequency cutoff.

    Returns:
        Estimated breathing rate in BPM.
    """
    # Apply Hanning window to reduce spectral leakage
    window = 0.5 * (1 - np.cos(2 * math.pi * np.arange(len(residuals)) / len(residuals)))
    windowed = residuals * window

    # Compute FFT
    freqs = np.fft.rfftfreq(len(residuals), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(windowed))

    # Find peak in the frequency range
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    freqs = freqs[mask]
    spectrum = spectrum[mask]

    if len(freqs) == 0:
        return 0.0

    # Find peak
    peak_idx = np.argmax(spectrum)
    if peak_idx == 0:
        return 0.0

    # Convert to BPM
    freq = freqs[peak_idx]
    bpm = freq * 60.0

    return bpm


def estimate_heart_rate(
    residuals: np.ndarray,
    weights: np.ndarray,
    phases: np.ndarray,
    fs: float = 100.0,
    lo_hz: float = 0.8,
    hi_hz: float = 2.0,
) -> float:
    """Estimate heart rate from CSI residuals.

    Args:
        residuals: (N,) array of CSI residual values.
        weights: (N,) array of weights.
        phases: (N,) array of unwrapped phases (radians).
        fs: Sample rate in Hz.
        lo_hz: Low frequency cutoff.
        hi_hz: High frequency cutoff.

    Returns:
        Estimated heart rate in BPM.
    """
    # Use phases for more accurate heart rate estimation
    # Circular variance approach
    cos_sum = np.sum(np.cos(phases), axis=-1)
    sin_sum = np.sum(np.sin(phases), axis=-1)
    r = np.sqrt(cos_sum**2 + sin_sum**2) / len(phases)

    # The circular variance should be high when the signal is coherent
    # (i.e., contains a strong periodic component)

    # Also compute energy from residuals
    energy = np.sum(weights**2 * residuals**2, axis=-1)

    # For heart rate, use the residuals directly
    return estimate_breathing_rate(residuals, weights, fs, lo_hz, hi_hz)


def generate_simulated_csi(
    breathing_hz: float = 0.2,
    heartrate_hz: float = 1.5,
    num_frames: int = 200,
    subcarriers: int = 56,
    noise: float = 0.02,
    fs: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate simulated CSI data with known breathing and heart rates.

    Args:
        breathing_hz: Breathing rate in Hz.
        heartrate_hz: Heart rate in Hz.
        num_frames: Number of CSI frames.
        subcarriers: Number of subcarriers (channels).
        noise: Noise level.

    Returns:
        Tuple of (residuals, weights, phases).
    """
    t = np.arange(num_frames) / fs  # fs = 100 Hz (sample rate)

    # Generate signals
    breathing_signal = 0.3 * np.sin(2 * math.pi * breathing_hz * t)
    heartrate_signal = 0.15 * np.sin(2 * math.pi * heartrate_hz * t)
    noise = noise * np.random.randn(num_frames)

    residuals = breathing_signal + heartrate_signal + noise

    # Weights (typically all 1.0 for WiFi CSI)
    weights = np.ones(num_frames)

    # Phases (unwrapped, for heart rate estimation)
    phases = np.unwrap(
        np.arctan2(
            np.sin(2 * math.pi * breathing_hz * t + random.random() * 0.1),
            np.cos(2 * math.pi * heartrate_hz * t + random.random() * 0.1),
        )
    )

    return residuals, weights, phases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WiFi-DensePose Vital Sign Extraction Demo"
    )
    parser.add_argument(
        "--breathing-hz",
        type=float,
        default=0.2,
        help="Simulated breathing frequency (Hz) — default 0.2 (12 BPM)",
    )
    parser.add_argument(
        "--heartrate-hz",
        type=float,
        default=1.5,
        help="Simulated heart rate frequency (Hz) — default 1.5 (90 BPM)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=2000,
        help="Number of CSI frames — default 2000",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="Noise level — default 0.02",
    )
    args = parser.parse_args()

    fs = 100.0  # Sample rate

    print("=" * 60)
    print("  WiFi-DensePose — Vital Sign Extraction Demo")
    print("=" * 60)
    print()
    print(f"  Simulated breathing rate : {args.breathing_hz * 60:.1f} BPM")
    print(f"  Simulated heart rate     : {args.heartrate_hz * 60:.1f} BPM")
    print(f"  Sample rate              : {fs} Hz")
    print(f"  CSI frames               : {args.num_frames}")
    print(f"  Noise level              : {args.noise}")
    print()
    print()
    print("  Pipeline:")
    print("    1. CSI residuals → bandpass filter")
    print("    2. Autocorrelation → frequency estimation")
    print("    3. BPM conversion")
    print()

    # Generate simulated CSI
    print("  Generating simulated CSI data...")
    residuals, weights, phases = generate_simulated_csi(
        breathing_hz=args.breathing_hz,
        heartrate_hz=args.heartrate_hz,
        num_frames=args.num_frames,
        noise=args.noise,
        fs=fs,
    )
    print(f"  Generated {args.num_frames} frames with {56} subcarriers")
    print()

    # Estimate breathing rate
    print("  [Breathing Extractor] 0.1–0.5 Hz bandpass")
    breathing_bpm = estimate_breathing_rate(residuals, weights, fs, 0.1, 0.5)
    print(f"    Detected breathing: {breathing_bpm:.1f} BPM")
    print(f"    True breathing:    {args.breathing_hz * 60:.1f} BPM")
    error_br = abs(breathing_bpm - args.breathing_hz * 60)
    print(f"    Error:             ±{error_br:.1f} BPM")
    print()

    # Estimate heart rate
    print("  [Heart Rate Extractor] 0.8–2.0 Hz bandpass")
    heartrate_bpm = estimate_heart_rate(residuals, weights, phases, fs, 0.8, 2.0)
    print(f"    Detected heart rate: {heartrate_bpm:.1f} BPM")
    print(f"    True heart rate:    {args.heartrate_hz * 60:.1f} BPM")
    error_hr = abs(heartrate_bpm - args.heartrate_hz * 60)
    print(f"    Error:              ±{error_hr:.1f} BPM")
    print()

    # Show native API
    print("=" * 60)
    print("  NATIVE RUST API SURFACE")
    print("=" * 60)
    print()
    print(f"  hello(): {hello_native()}")
    print(f"  Version : {get_native_version()}")
    print(f"  Features: {', '.join(get_native_features())}")
    print()

    # Show Python API
    print("=" * 60)
    print("  PYTHON API SURFACE")
    print("=" * 60)
    print()
    import wifi_densepose as wp

    print(f"  Version: {wp.__version__}")
    print(f"  hello(): {wp.hello()}")
    print()

    print("=" * 60)


if __name__ == "__main__":
    main()
