#!/usr/bin/env python3
"""
DSP Comparative Benchmark and Quick Test
Originally contributed with Multi-Algorithm support;
Extended and Benchmark Suite created by: shadcy (https://github.com/shadcy)
"""

import os
import sys

# Auto-detect and switch to local virtual environment if dependencies are missing
try:
    import numpy
except ImportError:
    venv_py = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "bin", "python3"))
    if os.path.exists(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)
    else:
        print("Error: numpy is not installed. Please create .venv and run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

import time
import numpy as np
from dsp import NoiseSuppressor


def generate_synthetic_audio(sr=16000, duration=5.0):
    """
    Generates synthetic audio containing speech-like segments and noise:
    - 0.0s to 1.5s: Noise only (calibration region)
    - 1.5s to 3.5s: Speech (sine wave mixture) + Noise
    - 3.5s to 5.0s: Noise only
    """
    t = np.arange(int(sr * duration)) / sr
    
    # Clean speech signal: sum of fundamental (440Hz) and harmonics (880Hz, 1320Hz)
    speech = np.zeros_like(t)
    speech_mask = (t >= 1.5) & (t <= 3.5)
    speech[speech_mask] = (
        0.50 * np.sin(2.0 * np.pi * 440.0 * t[speech_mask]) +
        0.25 * np.sin(2.0 * np.pi * 880.0 * t[speech_mask]) +
        0.15 * np.sin(2.0 * np.pi * 1320.0 * t[speech_mask])
    )
    
    # Background noise (White Gaussian Noise)
    np.random.seed(42)
    noise = 0.12 * np.random.randn(len(t))
    
    noisy = speech + noise
    return speech, noisy, t, speech_mask


def run_benchmark():
    sr = 16000
    speech, noisy, t, speech_mask = generate_synthetic_audio(sr)
    
    algos = ["spec_sub", "wiener", "gate"]
    results = {}
    
    print("=" * 95)
    print(f"Starting DSP Comparative Benchmark (duration={len(t)/sr:.1f}s, SR={sr}Hz)")
    print("Benchmarking & Performance Framework contributed by: shadcy (https://github.com/shadcy)")
    print("=" * 95)
    
    for algo in algos:
        # Create suppressor
        ns = NoiseSuppressor(
            sr=sr,
            frame_ms=20,
            beta=1.2,
            noise_floor=0.02,
            ema_alpha=0.96,
            gain_smooth=0.8,
            highpass_hz=0.0,  # disabled highpass for synthetic benchmark
            algo=algo,
            wiener_alpha=0.98,
            gate_threshold_db=5.0,
            gate_attenuation_db=-18.0,
        )
        
        hop = ns.hop
        delay_samples = ns.frame_len  # 2 * hop algorithmic buffering delay
        num_samples = len(noisy)
        output = np.zeros(num_samples, dtype=np.float32)
        
        # 1. Calibration on first 1.0 second
        calib_samples = int(sr * 1.0)
        for i in range(0, calib_samples - hop, hop):
            chunk = noisy[i : i + hop]
            ns.calibrate_noise(chunk)
            
        # 2. Main real-time style processing loop
        start_time = time.perf_counter()
        for i in range(0, num_samples - hop, hop):
            chunk = noisy[i : i + hop]
            out_chunk = ns.process(chunk)
            output[i : i + hop] = out_chunk
        elapsed = time.perf_counter() - start_time
        
        # 3. Calculate Performance Metrics (with delay compensation)
        
        # Segment masks with delay alignment
        active_idx = np.where(speech_mask)[0]
        valid_idx = active_idx[(active_idx + delay_samples < num_samples)]
        
        speech_active = speech[valid_idx]
        noisy_active = noisy[valid_idx]
        noise_active = noisy_active - speech_active
        output_active = output[valid_idx + delay_samples]
        
        # Input SNR in active segment
        snr_in = 10.0 * np.log10(np.mean(speech_active**2) / (np.mean(noise_active**2) + 1e-12))
        
        # Output SNR (error between delayed output and true clean signal)
        error_active = output_active - speech_active
        snr_out = 10.0 * np.log10(np.mean(speech_active**2) / (np.mean(error_active**2) + 1e-12))
        
        # SNR Improvement
        snr_imp = snr_out - snr_in
        
        # Noise Attenuation (NA) during silence (4.0s to 5.0s)
        silence_idx = np.where((t >= 4.0) & (t <= 5.0))[0]
        valid_silence = silence_idx[(silence_idx + delay_samples < num_samples)]
        noise_in_silence = noisy[valid_silence]
        noise_out_silence = output[valid_silence + delay_samples]
        noise_att = 10.0 * np.log10(np.mean(noise_in_silence**2) / (np.mean(noise_out_silence**2) + 1e-12))
        
        # Signal Distortion (normalized Mean Squared Error in active segment)
        norm_speech = speech_active / (np.max(np.abs(speech_active)) + 1e-8)
        norm_output = output_active / (np.max(np.abs(output_active)) + 1e-8)
        sig_dist = np.mean((norm_speech - norm_output)**2)
        
        # Processing Speed (represented as Real-time Factor RTF: CPU time / audio duration)
        rtf = elapsed / (len(t) / sr)
        
        results[algo] = {
            "snr_imp": snr_imp,
            "noise_att": noise_att,
            "sig_dist": sig_dist,
            "rtf": rtf,
            "elapsed_ms": elapsed * 1000.0,
        }
    
    # Print Table
    print(f"{'Algorithm':<18} | {'SNR Imp. (dB)':<13} | {'Noise Att. (dB)':<15} | {'Distortion (MSE)':<16} | {'RTF (CPU/Audio)':<15}")
    print("-" * 95)
    for algo, res in results.items():
        name = "Spec. Subtraction" if algo == "spec_sub" else ("Wiener Filter" if algo == "wiener" else "Spectral Gating")
        print(f"{name:<18} | {res['snr_imp']:>12.2f}  | {res['noise_att']:>14.2f}  | {res['sig_dist']:>15.6f}  | {res['rtf']:>14.5f}")
    print("=" * 95)
    print("Metrics Explanation:")
    print(" - SNR Imp. (dB): Output SNR minus Input SNR (delay-compensated). Higher is better.")
    print(" - Noise Att. (dB): Attenuation level in silent/noise-only regions. Higher is better.")
    print(" - Distortion: Mean Squared Error of normalized active signal. Lower is better.")
    print(" - RTF (Real-Time Factor): Processing time divided by audio duration (< 0.05 is highly efficient).")
    print("\nFor the full multi-dimensional benchmark suite (latency, OLA, noise types, JSON export), run:")
    print("  python benchmark.py --all")


if __name__ == "__main__":
    run_benchmark()
