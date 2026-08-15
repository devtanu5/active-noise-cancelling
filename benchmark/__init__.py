"""
Real-Time Active Noise Cancelling Benchmarking Suite
Author / Contributor: shadcy (https://github.com/shadcy)

Comprehensive benchmarking framework for measuring:
- Latency & OLA Buffering mechanics (algorithmic delay, per-hop execution timing, jitter)
- CPU utilization, Real-Time Factor (RTF), and Memory footprint
- DSP noise suppression quality (Delay-compensated SNR Improvement, Noise Attenuation, LSD, NMSE)
- Algorithm comparisons (Spectral Subtraction, Wiener Filter, Spectral Gating)
"""

__author__ = "shadcy (https://github.com/shadcy)"
__version__ = "1.0.0"

from benchmark.metrics import (
    calculate_delay_compensated_snr,
    calculate_noise_attenuation,
    calculate_log_spectral_distortion,
    calculate_nmse,
    verify_ola_reconstruction,
)
from benchmark.synthetic import generate_benchmark_signal, NOISE_TYPES
from benchmark.profiler import LatencyProfiler, CpuMemoryProfiler
from benchmark.suite import BenchmarkSuite

__all__ = [
    "calculate_delay_compensated_snr",
    "calculate_noise_attenuation",
    "calculate_log_spectral_distortion",
    "calculate_nmse",
    "verify_ola_reconstruction",
    "generate_benchmark_signal",
    "NOISE_TYPES",
    "LatencyProfiler",
    "CpuMemoryProfiler",
    "BenchmarkSuite",
    "__author__",
    "__version__",
]
