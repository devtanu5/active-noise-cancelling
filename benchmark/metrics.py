"""
DSP and Audio Quality Metrics for Noise Cancellation Benchmarking
Author: shadcy (https://github.com/shadcy)
"""

import numpy as np
from scipy.signal import stft


def calculate_delay_compensated_snr(
    clean: np.ndarray,
    noisy: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    delay_samples: int,
) -> dict:
    """
    Calculates input and output SNR, and SNR Improvement (dB) accounting for
    algorithmic buffering/OLA delay.

    Parameters:
    - clean: Ground truth clean signal.
    - noisy: Noisy input signal.
    - processed: Processed output signal from NoiseSuppressor.
    - mask: Boolean mask indicating active speech segments in the original timeline.
    - delay_samples: Algorithmic delay in samples (typically frame_len = 2 * hop).

    Returns:
    - Dict with snr_in, snr_out, snr_improvement, delay_samples.
    """
    total_len = len(clean)
    
    # Active indices within valid range accounting for delay
    valid_indices = np.where(mask)[0]
    valid_indices = valid_indices[
        (valid_indices >= 0) & (valid_indices + delay_samples < total_len) & (valid_indices + delay_samples < len(processed))
    ]
    
    if len(valid_indices) == 0:
        return {
            "snr_in": 0.0,
            "snr_out": 0.0,
            "snr_improvement": 0.0,
            "delay_samples": delay_samples,
        }

    s_clean = clean[valid_indices]
    s_noisy = noisy[valid_indices]
    s_proc = processed[valid_indices + delay_samples]

    clean_power = np.mean(s_clean ** 2)
    noise_in_power = np.mean((s_noisy - s_clean) ** 2)
    error_out_power = np.mean((s_proc - s_clean) ** 2)

    snr_in = 10.0 * np.log10(clean_power / (noise_in_power + 1e-12))
    snr_out = 10.0 * np.log10(clean_power / (error_out_power + 1e-12))
    snr_improvement = snr_out - snr_in

    return {
        "snr_in": float(snr_in),
        "snr_out": float(snr_out),
        "snr_improvement": float(snr_improvement),
        "delay_samples": int(delay_samples),
    }


def calculate_noise_attenuation(
    noisy: np.ndarray,
    processed: np.ndarray,
    silence_mask: np.ndarray,
    delay_samples: int = 0,
) -> float:
    """
    Calculates Noise Attenuation (NA in dB) during silent / ambient noise regions.
    Higher NA means more background noise was removed.
    """
    valid_indices = np.where(silence_mask)[0]
    valid_indices = valid_indices[
        (valid_indices >= 0) & (valid_indices + delay_samples < len(noisy)) & (valid_indices + delay_samples < len(processed))
    ]
    
    if len(valid_indices) == 0:
        return 0.0

    in_power = np.mean(noisy[valid_indices] ** 2)
    out_power = np.mean(processed[valid_indices + delay_samples] ** 2)

    return float(10.0 * np.log10((in_power + 1e-12) / (out_power + 1e-12)))


def calculate_log_spectral_distortion(
    clean: np.ndarray,
    processed: np.ndarray,
    sr: int,
    mask: np.ndarray = None,
    delay_samples: int = 0,
    n_fft: int = 512,
    hop_length: int = 128,
) -> float:
    """
    Calculates Log-Spectral Distortion (LSD in dB) between clean speech and
    the denoised output (with delay compensation). Lower LSD = lower speech distortion.
    """
    total_len = min(len(clean), len(processed) - delay_samples)
    if total_len <= n_fft:
        return 0.0

    s_clean = clean[:total_len]
    s_proc = processed[delay_samples : delay_samples + total_len]

    # STFT
    _, _, Z_clean = stft(s_clean, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, Z_proc = stft(s_proc, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    P_clean = np.abs(Z_clean) ** 2 + 1e-12
    P_proc = np.abs(Z_proc) ** 2 + 1e-12

    # Log spectral difference in dB
    diff_db = 10.0 * np.log10(P_clean) - 10.0 * np.log10(P_proc)
    
    # Root mean squared difference across frequency bins for each frame
    lsd_per_frame = np.sqrt(np.mean(diff_db ** 2, axis=0))

    # Optional speech-masking
    if mask is not None:
        frame_times = np.arange(len(lsd_per_frame)) * hop_length / sr
        frame_mask = np.interp(frame_times, np.arange(len(mask)) / sr, mask.astype(float)) > 0.5
        if np.any(frame_mask):
            lsd_per_frame = lsd_per_frame[frame_mask]

    return float(np.mean(lsd_per_frame))


def calculate_nmse(
    clean: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    delay_samples: int = 0,
) -> float:
    """
    Calculates Normalized Mean Squared Error (NMSE) on the active speech segment.
    """
    valid_indices = np.where(mask)[0]
    valid_indices = valid_indices[
        (valid_indices >= 0) & (valid_indices + delay_samples < len(clean)) & (valid_indices + delay_samples < len(processed))
    ]
    if len(valid_indices) == 0:
        return 0.0

    s_clean = clean[valid_indices]
    s_proc = processed[valid_indices + delay_samples]

    norm = np.max(np.abs(s_clean)) + 1e-8
    return float(np.mean(((s_clean / norm) - (s_proc / norm)) ** 2))


def verify_ola_reconstruction(
    sr: int = 16000,
    frame_ms: int = 20,
    duration: float = 1.0,
) -> dict:
    """
    Verifies that the Hann-sqrt Overlap-Add (OLA) engine satisfies the Constant
    Overlap-Add (COLA) property with near-zero mathematical reconstruction error.
    """
    from dsp import NoiseSuppressor

    # Initialize suppressor in pure identity bypass mode
    ns = NoiseSuppressor(
        sr=sr,
        frame_ms=frame_ms,
        beta=0.0,
        noise_floor=0.0,
        gain_smooth=0.0,
        highpass_hz=0.0,
        algo="spec_sub",
    )

    t = np.arange(int(sr * duration)) / sr
    test_sig = (
        0.5 * np.sin(2 * np.pi * 300 * t) +
        0.3 * np.cos(2 * np.pi * 750 * t)
    ).astype(np.float32)

    hop = ns.hop
    delay_samples = 2 * hop  # Total algorithmic delay (1 frame)
    out = np.zeros(len(test_sig), dtype=np.float32)

    for i in range(0, len(test_sig) - hop, hop):
        chunk = test_sig[i : i + hop]
        out[i : i + hop] = ns.process(chunk)

    # Compare aligned segments (excluding initial warm-up and boundary tails)
    eval_start = 3 * delay_samples
    eval_end = len(test_sig) - 2 * delay_samples

    ref_segment = test_sig[eval_start : eval_end]
    proc_segment = out[eval_start + delay_samples : eval_end + delay_samples]

    abs_err = np.abs(ref_segment - proc_segment)
    max_err = float(np.max(abs_err))
    mse = float(np.mean(abs_err ** 2))
    passed = max_err < 0.05 and mse < 1e-4

    return {
        "passed": bool(passed),
        "max_error": max_err,
        "mse": mse,
        "delay_samples": delay_samples,
        "delay_ms": (delay_samples / sr) * 1000.0,
        "sr": sr,
        "frame_ms": frame_ms,
        "hop_samples": hop,
        "hop_ms": (hop / sr) * 1000.0,
    }
