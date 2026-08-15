"""
Complete Multi-Dimensional Benchmarking Suite Engine
Author: shadcy (https://github.com/shadcy)
"""

from typing import Dict, Any, List, Optional
import numpy as np

from dsp import NoiseSuppressor
from benchmark.synthetic import generate_benchmark_signal, NOISE_TYPES
from benchmark.profiler import CpuMemoryProfiler
from benchmark.metrics import (
    calculate_delay_compensated_snr,
    calculate_noise_attenuation,
    calculate_log_spectral_distortion,
    calculate_nmse,
    verify_ola_reconstruction,
)


class BenchmarkSuite:
    """
    Main benchmark orchestration suite for Real-Time Active Noise Cancelling.
    """

    def __init__(self, author: str = "shadcy (https://github.com/shadcy)"):
        self.author = author
        self.profiler = CpuMemoryProfiler()

    def run_algorithm_comparison(
        self,
        sr: int = 16000,
        frame_ms: int = 20,
        duration: float = 6.0,
        noise_type: str = "white",
        target_snr_db: float = 5.0,
        algorithms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive DSP, latency, and CPU benchmarks comparing algorithms.
        """
        if algorithms is None:
            algorithms = ["spec_sub", "wiener", "gate"]

        # Generate standard test signal
        sig_data = generate_benchmark_signal(
            sr=sr,
            duration=duration,
            calib_duration=1.5,
            speech_start=1.8,
            speech_end=4.2,
            target_snr_db=target_snr_db,
            noise_type=noise_type,
        )

        results = {}

        for algo in algorithms:
            # Create instance
            ns = NoiseSuppressor(
                sr=sr,
                frame_ms=frame_ms,
                beta=1.2,
                noise_floor=0.02,
                ema_alpha=0.96,
                gain_smooth=0.8,
                highpass_hz=0.0,
                algo=algo,
                wiener_alpha=0.98,
                gate_threshold_db=5.0,
                gate_attenuation_db=-18.0,
            )

            # Profile calibration + processing
            calib_signal = sig_data["noisy"][sig_data["calib_mask"]]
            exec_profile = self.profiler.profile_execution(
                suppressor=ns,
                noisy_signal=sig_data["noisy"],
                calib_signal=calib_signal,
                warmup_runs=1,
            )

            output = exec_profile["output"]
            delay_samples = ns.frame_len  # 2 * hop

            # Calculate DSP quality metrics
            snr_res = calculate_delay_compensated_snr(
                clean=sig_data["speech"],
                noisy=sig_data["noisy"],
                processed=output,
                mask=sig_data["speech_mask"],
                delay_samples=delay_samples,
            )

            na_db = calculate_noise_attenuation(
                noisy=sig_data["noisy"],
                processed=output,
                silence_mask=sig_data["silence_mask"],
                delay_samples=delay_samples,
            )

            lsd_db = calculate_log_spectral_distortion(
                clean=sig_data["speech"],
                processed=output,
                sr=sr,
                mask=sig_data["speech_mask"],
                delay_samples=delay_samples,
            )

            nmse = calculate_nmse(
                clean=sig_data["speech"],
                processed=output,
                mask=sig_data["speech_mask"],
                delay_samples=delay_samples,
            )

            lat_stats = exec_profile["latency_stats"]

            results[algo] = {
                "algorithm": algo,
                "snr_in_db": snr_res["snr_in"],
                "snr_out_db": snr_res["snr_out"],
                "snr_improvement_db": snr_res["snr_improvement"],
                "noise_attenuation_db": na_db,
                "log_spectral_distortion_db": lsd_db,
                "nmse_active": nmse,
                "rtf": exec_profile["rtf"],
                "cpu_usage_pct": exec_profile["cpu_usage_pct"],
                "speedup_x": exec_profile["speedup_x"],
                "throughput_ksamples_sec": exec_profile["throughput_ksamples_per_sec"],
                "mean_latency_us": lat_stats.get("mean_latency_us", 0.0),
                "p95_latency_ms": lat_stats.get("p95_latency_ms", 0.0),
                "p99_latency_ms": lat_stats.get("p99_latency_ms", 0.0),
                "headroom_pct": lat_stats.get("headroom_pct", 0.0),
                "algorithmic_delay_ms": lat_stats.get("algorithmic_delay_ms", 0.0),
                "mem_delta_mb": exec_profile["mem_delta_mb"],
            }

        return {
            "test_config": {
                "sr": sr,
                "frame_ms": frame_ms,
                "duration_sec": duration,
                "noise_type": noise_type,
                "target_snr_db": target_snr_db,
            },
            "algorithms": results,
        }

    def run_latency_ola_benchmark(
        self,
        sample_rates: Optional[List[int]] = None,
        frame_sizes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Sweeps sample rates and frame sizes to measure OLA buffering mechanics,
        COLA reconstruction error, and per-hop execution timing.
        """
        if sample_rates is None:
            sample_rates = [8000, 16000, 32000, 44100, 48000]
        if frame_sizes is None:
            frame_sizes = [10, 20, 30, 40]

        ola_verification_results = []
        latency_sweep_results = []

        # 1. OLA Constant Overlap-Add reconstruction verification
        for sr in sample_rates:
            for f_ms in frame_sizes:
                ola_res = verify_ola_reconstruction(sr=sr, frame_ms=f_ms, duration=1.0)
                ola_verification_results.append(ola_res)

        # 2. Timing latency sweep
        for sr in sample_rates:
            for f_ms in frame_sizes:
                ns = NoiseSuppressor(sr=sr, frame_ms=f_ms, algo="spec_sub")
                hop = ns.hop
                test_sig = np.random.randn(int(sr * 2.0)).astype(np.float32)

                exec_profile = self.profiler.profile_execution(
                    suppressor=ns,
                    noisy_signal=test_sig,
                    warmup_runs=1,
                )

                lat = exec_profile["latency_stats"]
                latency_sweep_results.append({
                    "sr": sr,
                    "frame_ms": f_ms,
                    "hop_ms": lat["hop_duration_ms"],
                    "algorithmic_delay_ms": lat["algorithmic_delay_ms"],
                    "mean_latency_us": lat["mean_latency_us"],
                    "p95_latency_ms": lat["p95_latency_ms"],
                    "p99_latency_ms": lat["p99_latency_ms"],
                    "max_latency_ms": lat["max_latency_ms"],
                    "headroom_pct": lat["headroom_pct"],
                    "rtf": exec_profile["rtf"],
                    "cpu_usage_pct": exec_profile["cpu_usage_pct"],
                })

        return {
            "ola_verification": ola_verification_results,
            "latency_sweep": latency_sweep_results,
        }

    def run_noise_robustness_benchmark(
        self,
        sr: int = 16000,
        frame_ms: int = 20,
        noise_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Tests algorithm performance across different noise types.
        """
        if noise_types is None:
            noise_types = NOISE_TYPES

        robustness_results = {}
        for ntype in noise_types:
            res = self.run_algorithm_comparison(
                sr=sr,
                frame_ms=frame_ms,
                duration=5.0,
                noise_type=ntype,
                target_snr_db=5.0,
            )
            robustness_results[ntype] = res["algorithms"]

        return robustness_results

    def run_full_suite(self) -> Dict[str, Any]:
        """Runs the entire multi-dimensional benchmark suite."""
        print("▶ Running Algorithm Comparison Benchmark...")
        algo_comp = self.run_algorithm_comparison(sr=16000, frame_ms=20, duration=6.0)

        print("▶ Running Latency & OLA Buffering Analysis...")
        lat_ola = self.run_latency_ola_benchmark(
            sample_rates=[8000, 16000, 32000, 48000],
            frame_sizes=[10, 20, 30, 40],
        )

        print("▶ Running Noise Robustness Benchmark...")
        noise_rob = self.run_noise_robustness_benchmark(
            sr=16000,
            frame_ms=20,
            noise_types=["white", "pink", "brown", "mains_hum", "composite"],
        )

        return {
            "benchmark_author": self.author,
            "algorithm_comparison": algo_comp,
            "latency_and_ola": lat_ola,
            "noise_robustness": noise_rob,
        }
