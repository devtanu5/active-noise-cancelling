# Active Noise Cancelling (Real-Time Mic Noise Suppression)

> Low-latency, real-time microphone noise suppression in Python using spectral subtraction + adaptive noise tracking and 50% overlap-add (OLA). Works cross-platform with PortAudio via `sounddevice`.

https://github.com/devtanu5
https://github.com/shadcy (Benchmarking & Performance Suite by @shadcy)

---

## ✨ Features

- **Real-time** mic noise suppression (20 ms frames, 50% overlap).
- **Multiple Algorithms**: Spectral Subtraction (`spec_sub`), Decision-Directed Wiener Filter (`wiener`), and Spectral Gating (`gate`).
- **Adaptive noise estimate** using exponential moving average (EMA).
- **Spectral subtraction + Wiener-style gain**, smoothed per band.
- **High-pass filter** (optional) to reduce rumble / hum.
- **Calibration step** (1–2 s) to capture baseline ambient noise.
- **Low latency** (frame = 20 ms, hop = 10 ms).
- **Comprehensive Benchmarking Suite** (contributed by `@shadcy`).
- **Pure Python + NumPy/SciPy**. No heavyweight ML runtime required.

> This is **noise suppression** (post-filtering of mic input), not feedforward/feedback “anti-noise” (phase-inversion) for headphones.

---

## 📦 Install

```bash
git clone https://github.com/DevtanuBarman111/active-noise-cancelling.git
cd active-noise-cancelling
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
# .venv\Scripts\activate                            # (Windows)
pip install -r requirements.txt
```
## Dependencies

- `sounddevice` (PortAudio binding)
- `numpy`
- `scipy`
- `pyyaml` (for reading config.yaml, optional)
- `psutil` (for CPU/memory benchmarking)

Windows: if PortAudio devices are not listed, update your audio drivers or install WASAPI loopback support (typically not needed).  
Linux: ensure your user is in audio group and PulseAudio/PipeWire is running.  
macOS: grant microphone permission to the terminal/IDE.

## ▶️ Run

Quick start (defaults are fine):

```bash
python main.py
```

Useful options:

```bash
python main.py \
  --samplerate 16000 \
  --frame_ms 20 \
  --calib_sec 1.0 \
  --algo spec_sub \
  --device_in default \
  --device_out default \
  --highpass 80
```

- `--samplerate` : 16000 or 48000 are common.
- `--frame_ms` : 20 ms frame (10 ms hop with 50% overlap).
- `--calib_sec` : initial ambient-noise calibration duration (seconds).
- `--algo` : `spec_sub` (Spectral Subtraction), `wiener` (Wiener Filter), or `gate` (Spectral Gating).
- `--highpass` : cut below N Hz (0 to disable).
- `--output` : save output audio to a WAV file.

Press Ctrl+C to stop.

## ⚙️ How it works (DSP)

### Framing & Windowing
Signal is split into frames (e.g., 20 ms) with 50% overlap and Sqrt-Hann analysis and synthesis windows (providing perfect Constant Overlap-Add reconstruction).

### Noise Spectrum Estimation (EMA)
Magnitude spectrum of noise is tracked with an exponential moving average. During low-energy segments it adapts faster.

### Spectral Subtraction + Gain Smoothing
We estimate clean magnitude per bin: $|X|_{\text{clean}} = \max(|X| - \beta \cdot |N|, \text{floor} \cdot |N|)$.
Then we compute a Wiener-like gain and smooth it over time to avoid musical noise.

### High-Pass
Optional IIR HPF removes rumble and low-frequency DC offset.

### OLA Synthesis
Inverse FFT → Sqrt-Hann synthesis window → Overlap-Add. The output hop is emitted with minimal delay.

## 📁 Config

`config.yaml` (optional; CLI args override file):

```yaml
samplerate: 16000
frame_ms: 20
calib_sec: 1.0
highpass_hz: 80
noise_beta: 1.0
noise_floor: 0.02
ema_alpha: 0.96
gain_smooth: 0.8
device_in: default
device_out: default
algo: "spec_sub"
wiener_alpha: 0.98
gate_threshold_db: 6.0
gate_attenuation_db: -20.0
```

---

## 🧪 Benchmarking & Performance Suite

> Benchmarking framework and profiling suite developed and contributed by **[shadcy](https://github.com/shadcy)** (`@shadcy`).

The project includes an extensible, multi-dimensional benchmarking suite measuring **Latency**, **OLA Buffering mechanics**, **CPU load**, **Real-Time Factor (RTF)**, and **DSP noise cancellation metrics** (delay-compensated SNR improvement, Noise Attenuation, Log-Spectral Distortion, NMSE) across algorithms and noise profiles.

### Running Benchmarks

```bash
# Run full benchmark suite (Algorithms + Latency/OLA + Noise Robustness)
python benchmark.py --all

# Run specific benchmarks
python benchmark.py --latency-only
python benchmark.py --cpu-only
python benchmark.py --dsp-only

# Export results to JSON or Markdown
python benchmark.py --all --json-out results.json --md-out results.md

# Quick comparative test
python test_algorithms.py
```

### 1. Latency & OLA Buffering Breakdown

| Latency Component | Duration (at 16 kHz, 20 ms Frame) | Description |
| :--- | :--- | :--- |
| **Input Packetization Delay** | **10.0 ms** (1 Hop = 160 samples) | Time to accumulate 1 hop of mic input |
| **Framing Tail Buffer Delay** | **10.0 ms** (1 Hop = 160 samples) | Preceding hop buffer for 50% overlap analysis |
| **OLA Synthesis Buffer Delay**| **10.0 ms** (1 Hop = 160 samples) | Overlap-add reconstruction summation delay |
| **Total Algorithmic Buffering Delay** | **20.0 ms** (1 Full Frame = 320 samples) | Exact mathematical end-to-end algorithmic latency ($2 \times T_{\text{hop}}$) |
| **Per-Hop CPU Processing Time** | **~0.07 – 0.09 ms** (70–90 µs) | Actual execution time of `NoiseSuppressor.process()` |
| **Real-Time Safety Margin / Headroom**| **> 99.1%** | $(T_{\text{hop}} - \tau_{\text{proc}}) / T_{\text{hop}}$ available for OS & Audio I/O |
| **COLA Reconstruction Fidelity** | **Exact** ($\text{MSE} < 10^{-5}$) | Sqrt-Hann 50% overlap-add satisfies COLA property |

### 2. Algorithm Comparison & CPU Usage

Benchmarked at 16,000 Hz sample rate with 20 ms frame length (10 ms hop):

| Algorithm | Δ SNR (dB) | Noise Attenuation (dB) | Log-Spectral Dist. (dB) | NMSE (Active) | Per-Hop Latency | Real-Time Factor (RTF) | CPU Load (Single Core) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spectral Subtraction** | -4.38 dB | 18.31 dB | 52.05 dB | 0.2296 | ~70 µs | 0.0071 | **~0.71%** |
| **Wiener Filter (DD)** | -4.43 dB | **31.40 dB** | **41.96 dB** | 0.2322 | ~81 µs | 0.0083 | **~0.83%** |
| **Spectral Gating** | **-1.96 dB** | 9.84 dB | 60.13 dB | **0.1315** | ~84 µs | 0.0085 | **~0.85%** |

*Note: Real-Time Factor (RTF) $< 0.01$ indicates that 1 second of audio is processed in under 8.5 milliseconds (over 115× faster than real-time).*

### 3. Noise Profile Robustness

- **White Noise**: Highest noise attenuation with Wiener filtering (> 31 dB attenuation).
- **Pink & Brownian Noise**: Smooth suppression with high spectral comfort.
- **50 Hz / 60 Hz Mains Hum & Rumble**: Strong suppression when combined with `--highpass 80`.
- **Composite Acoustic Noise**: Robust suppression across mixed office/room background noise.

---

## 📜 License

MIT

## 🙌 Acknowledgements

- **[devtanu5](https://github.com/devtanu5)** - Initial project author and core real-time ANC implementation.
- **[shadcy](https://github.com/shadcy)** (`@shadcy`) - Benchmarking & Performance Suite, latency and OLA buffering analysis, CPU profiler, and multi-dimensional DSP metrics.
- `sounddevice` (PortAudio) for reliable cross-platform audio I/O.
- Classic spectral subtraction and Wiener filtering literature for single-channel speech enhancement.

---
