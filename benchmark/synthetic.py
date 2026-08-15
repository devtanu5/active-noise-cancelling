"""
Synthetic Audio and Noise Generator for Benchmarking
Author: shadcy (https://github.com/shadcy)
"""

import numpy as np

NOISE_TYPES = ["white", "pink", "brown", "mains_hum", "chirp", "composite"]


def generate_noise(noise_type: str, n_samples: int, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Generates synthetic background noise of various acoustic profiles."""
    rng = np.random.default_rng(seed)
    
    if noise_type == "white":
        return rng.standard_normal(n_samples).astype(np.float32)

    elif noise_type == "pink":
        # 1/f noise generation via FFT spectral shaping
        white = rng.standard_normal(n_samples)
        X = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)
        freqs[0] = freqs[1]  # avoid division by zero
        scaling = 1.0 / np.sqrt(freqs)
        X_pink = X * scaling
        pink = np.fft.irfft(X_pink, n_samples)
        return (pink / (np.std(pink) + 1e-8)).astype(np.float32)

    elif noise_type == "brown":
        # 1/f^2 Brownian noise
        white = rng.standard_normal(n_samples)
        brown = np.cumsum(white)
        # remove DC drift with leaky integrator
        brown = brown - np.mean(brown)
        return (brown / (np.std(brown) + 1e-8)).astype(np.float32)

    elif noise_type == "mains_hum":
        # 50 Hz / 60 Hz hum with harmonics + rumble
        t = np.arange(n_samples) / sr
        hum = (
            0.60 * np.sin(2 * np.pi * 50.0 * t) +
            0.25 * np.sin(2 * np.pi * 100.0 * t) +
            0.15 * np.sin(2 * np.pi * 150.0 * t) +
            0.10 * np.sin(2 * np.pi * 200.0 * t) +
            0.05 * rng.standard_normal(n_samples)
        )
        return (hum / (np.std(hum) + 1e-8)).astype(np.float32)

    elif noise_type == "chirp":
        # Linear frequency modulated swept interference
        t = np.arange(n_samples) / sr
        f0, f1 = 200.0, 3500.0
        T = n_samples / sr
        phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * (t ** 2))
        chirp = np.sin(phase) + 0.1 * rng.standard_normal(n_samples)
        return (chirp / (np.std(chirp) + 1e-8)).astype(np.float32)

    elif noise_type == "composite":
        # Realistic office / room mix (pink noise + hum + occasional burst)
        pink = generate_noise("pink", n_samples, sr, seed)
        hum = generate_noise("mains_hum", n_samples, sr, seed)
        composite = 0.7 * pink + 0.3 * hum
        return (composite / (np.std(composite) + 1e-8)).astype(np.float32)

    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from {NOISE_TYPES}")


def generate_speech(n_samples: int, sr: int = 16000, speech_mask: np.ndarray = None) -> np.ndarray:
    """
    Generates multi-harmonic pseudo-speech with vowel-like formant structure and amplitude envelope.
    """
    t = np.arange(n_samples) / sr
    speech = np.zeros(n_samples, dtype=np.float32)
    
    if speech_mask is None:
        speech_mask = np.ones(n_samples, dtype=bool)

    # Fundamental frequencies with natural vibrato
    f0 = 220.0 + 3.0 * np.sin(2 * np.pi * 5.0 * t)  # 220Hz fundamental with 5Hz vibrato
    phase_f0 = 2 * np.pi * np.cumsum(f0) / sr

    # Harmonics with formant-like weights
    harmonics = (
        0.50 * np.sin(phase_f0) +                  # F0 (220 Hz)
        0.35 * np.sin(2 * phase_f0) +              # 2nd harmonic (440 Hz - F1 region)
        0.25 * np.sin(3 * phase_f0) +              # 3rd harmonic (660 Hz)
        0.20 * np.sin(4 * phase_f0) +              # 4th harmonic (880 Hz)
        0.15 * np.sin(6 * phase_f0) +              # 6th harmonic (1320 Hz - F2 region)
        0.10 * np.sin(11 * phase_f0)               # 11th harmonic (2420 Hz - F3 region)
    )

    # Smooth trapezoidal amplitude envelope for active segments
    envelope = np.zeros(n_samples, dtype=np.float32)
    active_indices = np.where(speech_mask)[0]
    if len(active_indices) > 0:
        fade_len = min(int(0.1 * sr), len(active_indices) // 4)
        active_env = np.ones(len(active_indices), dtype=np.float32)
        if fade_len > 0:
            fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
            fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))
            active_env[:fade_len] = fade_in
            active_env[-fade_len:] = fade_out
        envelope[active_indices] = active_env

    speech = harmonics * envelope
    return speech.astype(np.float32)


def generate_benchmark_signal(
    sr: int = 16000,
    duration: float = 6.0,
    calib_duration: float = 1.5,
    speech_start: float = 1.8,
    speech_end: float = 4.2,
    target_snr_db: float = 5.0,
    noise_type: str = "white",
    seed: int = 42,
) -> dict:
    """
    Constructs a calibrated multi-phase benchmark audio test signal.

    Timeline:
    - [0.0s, calib_duration]: Calibration region (Noise only)
    - [calib_duration, speech_start]: Pre-speech ambient silence
    - [speech_start, speech_end]: Active speech + Noise at specified SNR
    - [speech_end, duration]: Post-speech silence (Noise only) for measuring Noise Attenuation.
    """
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr

    speech_mask = (t >= speech_start) & (t <= speech_end)
    silence_mask = (t >= speech_end + 0.3) & (t <= duration)
    calib_mask = t <= calib_duration

    # Generate components
    raw_speech = generate_speech(n_samples, sr=sr, speech_mask=speech_mask)
    raw_noise = generate_noise(noise_type, n_samples, sr=sr, seed=seed)

    # Scale noise relative to speech to achieve exact target SNR in active speech region
    speech_power = np.mean(raw_speech[speech_mask] ** 2)
    noise_power = np.mean(raw_noise[speech_mask] ** 2)
    
    desired_noise_power = speech_power / (10.0 ** (target_snr_db / 10.0))
    noise_scaling = np.sqrt(desired_noise_power / (noise_power + 1e-12))
    scaled_noise = raw_noise * noise_scaling

    noisy_signal = raw_speech + scaled_noise

    return {
        "speech": raw_speech,
        "noise": scaled_noise,
        "noisy": noisy_signal.astype(np.float32),
        "t": t,
        "sr": sr,
        "duration": duration,
        "speech_mask": speech_mask,
        "silence_mask": silence_mask,
        "calib_mask": calib_mask,
        "calib_duration": calib_duration,
        "target_snr_db": target_snr_db,
        "noise_type": noise_type,
    }
