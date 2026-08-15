"""
Latency, OLA Buffering, and CPU/Memory Profiler
Author: shadcy (https://github.com/shadcy)
"""

import time
import os
import numpy as np
import psutil
from typing import Dict, Any, List


class LatencyProfiler:
    """
    Profiles high-precision per-hop execution latency and computes theoretical
    buffering delay and real-time safety margins.
    """

    def __init__(self, sr: int, frame_ms: int):
        self.sr = sr
        self.frame_ms = frame_ms
        self.frame_len = int(sr * frame_ms / 1000)
        self.hop_len = self.frame_len // 2
        self.hop_ms = (self.hop_len / self.sr) * 1000.0

        # Algorithmic delay components
        # 1. Framing history tail buffer: frame_len - hop_len = hop_len
        # 2. OLA synthesis overlap shift: hop_len
        # Total algorithmic buffering delay = 2 * hop_len = frame_len
        self.algorithmic_delay_samples = 2 * self.hop_len
        self.algorithmic_delay_ms = (self.algorithmic_delay_samples / self.sr) * 1000.0

        self.hop_durations_ns: List[int] = []

    def record_hop(self, duration_ns: int):
        """Records the time taken (nanoseconds) for a single process() call."""
        self.hop_durations_ns.append(duration_ns)

    def compute_stats(self) -> Dict[str, Any]:
        """Calculates statistical distribution of execution latency and headroom."""
        if not self.hop_durations_ns:
            return {}

        durations_ms = np.array(self.hop_durations_ns) / 1e6  # convert ns to ms
        durations_us = np.array(self.hop_durations_ns) / 1e3  # convert ns to us

        mean_ms = float(np.mean(durations_ms))
        median_ms = float(np.median(durations_ms))
        min_ms = float(np.min(durations_ms))
        max_ms = float(np.max(durations_ms))
        p90_ms = float(np.percentile(durations_ms, 90))
        p95_ms = float(np.percentile(durations_ms, 95))
        p99_ms = float(np.percentile(durations_ms, 99))
        std_ms = float(np.std(durations_ms))

        # Deadline and real-time safety margin
        deadline_ms = self.hop_ms
        missed_deadlines = int(np.sum(durations_ms > deadline_ms))
        headroom_pct = float(max(0.0, (deadline_ms - mean_ms) / deadline_ms * 100.0))
        headroom_p99_pct = float(max(0.0, (deadline_ms - p99_ms) / deadline_ms * 100.0))

        return {
            "sr": self.sr,
            "frame_ms": self.frame_ms,
            "frame_samples": self.frame_len,
            "hop_samples": self.hop_len,
            "hop_duration_ms": self.hop_ms,
            "algorithmic_delay_samples": self.algorithmic_delay_samples,
            "algorithmic_delay_ms": self.algorithmic_delay_ms,
            "total_hops": len(durations_ms),
            "mean_latency_ms": mean_ms,
            "mean_latency_us": mean_ms * 1000.0,
            "median_latency_ms": median_ms,
            "min_latency_ms": min_ms,
            "max_latency_ms": max_ms,
            "p90_latency_ms": p90_ms,
            "p95_latency_ms": p95_ms,
            "p99_latency_ms": p99_ms,
            "jitter_std_ms": std_ms,
            "headroom_pct": headroom_pct,
            "headroom_p99_pct": headroom_p99_pct,
            "missed_deadlines": missed_deadlines,
        }


class CpuMemoryProfiler:
    """
    Profiles CPU usage, Real-Time Factor (RTF), memory footprint, and throughput.
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def profile_execution(
        self,
        suppressor,
        noisy_signal: np.ndarray,
        calib_signal: np.ndarray = None,
        warmup_runs: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes noise cancellation pipeline over input signal and measures
        precise CPU time, RTF, and memory delta.
        """
        hop = suppressor.hop
        sr = suppressor.sr
        num_samples = len(noisy_signal)
        audio_duration_sec = num_samples / sr

        # Warmup if requested
        if warmup_runs > 0:
            for _ in range(warmup_runs):
                for i in range(0, min(len(noisy_signal), 10 * hop), hop):
                    suppressor.process(noisy_signal[i : i + hop])

        # Calibration phase (if provided)
        if calib_signal is not None:
            for i in range(0, len(calib_signal) - hop, hop):
                suppressor.calibrate_noise(calib_signal[i : i + hop])

        # Memory baseline
        mem_before = self.process.memory_info().rss / (1024 * 1024)

        # Processing loop with high-resolution per-hop timing
        output = np.zeros(num_samples, dtype=np.float32)
        latency_prof = LatencyProfiler(sr=sr, frame_ms=int(round((suppressor.frame_len / sr) * 1000)))

        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        for i in range(0, num_samples - hop, hop):
            chunk = noisy_signal[i : i + hop]
            
            t0 = time.perf_counter_ns()
            out_chunk = suppressor.process(chunk)
            t1 = time.perf_counter_ns()
            
            latency_prof.record_hop(t1 - t0)
            output[i : i + hop] = out_chunk

        elapsed_cpu_sec = time.process_time() - start_cpu
        elapsed_wall_sec = time.perf_counter() - start_wall

        # Memory after
        mem_after = self.process.memory_info().rss / (1024 * 1024)

        # Metrics
        rtf = elapsed_wall_sec / audio_duration_sec
        cpu_rtf = elapsed_cpu_sec / audio_duration_sec
        cpu_usage_pct = rtf * 100.0  # Core utilization ratio during continuous streaming
        audio_speedup = 1.0 / (rtf + 1e-12)
        throughput_kps = (num_samples / 1000.0) / (elapsed_wall_sec + 1e-12)

        lat_stats = latency_prof.compute_stats()

        return {
            "output": output,
            "audio_duration_sec": audio_duration_sec,
            "elapsed_wall_sec": elapsed_wall_sec,
            "elapsed_cpu_sec": elapsed_cpu_sec,
            "rtf": float(rtf),
            "cpu_rtf": float(cpu_rtf),
            "cpu_usage_pct": float(cpu_usage_pct),
            "speedup_x": float(audio_speedup),
            "throughput_ksamples_per_sec": float(throughput_kps),
            "mem_before_mb": float(mem_before),
            "mem_after_mb": float(mem_after),
            "mem_delta_mb": float(mem_after - mem_before),
            "latency_stats": lat_stats,
        }
