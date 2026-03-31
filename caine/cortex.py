"""
CAINE - Module 3: Cortical Architecture
=========================================
Ten cortical and subcortical regions built from populations of Hodgkin-Huxley
neurons, connected by white matter tracts with axonal transmission delays and
STDP-based plasticity.

Regions
-------
  3.1  V1 - Primary Visual Cortex
         Orientation columns (0/45/90/135 deg), DoG receptive fields (numpy only),
         simple cells + complex cells (motion-sensitive, position-invariant).
         Input: raw pixel frames via pixel_to_drive() or orientation angle via
         compute_drive() for the stimulus-schedule demo.

  3.2  Visual Association (V2, V4, MT, IT)
         V2:  contour / illusory-boundary detection (combines adjacent V1 columns)
         V4:  color opponency (R-G, G-B, B-R) + V2 shape input
         MT:  optic-flow / motion direction columns (frame-to-frame V1 delta)
         IT:  Hebbian object identity -- no categories hardcoded; Hebbian binding
              of V2/V4/MT features builds stable representations through exposure.

  3.3  A1 - Primary Auditory Cortex
         Mel filterbank (128 bands, FFT of mic buffer) via mel_filterbank() +
         mel_filterbank_drive(); tonotopic organisation; onset/offset detectors;
         amplitude modulation detectors critical for speech rhythm.
         Fallback: compute_drive(freq_hz) for the demo stimulus schedule.

  3.4  Auditory Association (A2, STG)
         A2:  spectrotemporal pattern detection -- integrates A1 history
         STG: emergent phoneme populations via STDP + Hebbian reinforcement;
              no phonemes hardcoded; dopamine/oxytocin gating by caller.

  3.5  S1 - Somatosensory Cortex
         Joint-angle proprioception, body-part columns sized to motor importance
         (hand=12, arm=6, head=6, torso=4, foot=4), efference-copy from M1.

  3.7  PFC - Prefrontal Cortex
         Developmentally gated -- suppressed until Stage 2, myelination grows
         gradually.  Working-memory persistent loops (500ms-5s decay).
         Inhibitory-control sub-region.  Self-model sub-region (mPFC, neurons 0-7).
         PCC proxy sub-region (neurons 24-31) for DMN tracking.

  3.9  DMN - Default Mode Network (emergent)
         Medial PFC + PCC + Angular Gyrus correlated resting-state activity.
         Not hardcoded -- emerges from correlation of these three regions.
         First sustained appearance logged as a developmental milestone.

  3.10 Thalamus
         Relay nuclei for visual (LGN), auditory (MGN), somatosensory (VPL).
         Thalamocortical feedback loops: cortex gates its own thalamic input.
         Pulvinar: cross-modal integration and attention spotlight.

White-matter tract (V1 -> A1): topographic Gaussian weights, 5 ms axonal
delay, AMPA conductance kinetics, STDP modulated by neurochemicals.

Single file.  Builds on Modules 1, 2, chemicals.
No ML frameworks -- only numpy.
"""

import os
import sys as _sys
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
try:
    from caine.gpu import xp as _xp, to_numpy as _to_numpy, GPU_AVAILABLE as _GPU
except ImportError:
    _xp = np
    _to_numpy = np.asarray
    _GPU = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from collections import deque

import caine.paths as _paths
_OUTPUT_DIR = _paths.OUTPUT_DIR

# ---------------------------------------------------------------------------
# Import primitives from earlier modules
# ---------------------------------------------------------------------------
from caine.neuron import (
    Cm, gNa, gK, gL, ENa, EK, EL, V_rest,
    alpha_m, beta_m,
    alpha_h, beta_h,
    alpha_n, beta_n,
    gate_steady_state,
)
from caine.synapse import (
    AMPA_TAU, AMPA_E_REV,
    STDP_A_PLUS, STDP_A_MINUS, STDP_TAU_PLUS, STDP_TAU_MINUS,
)
from caine.chemicals import (
    NeurochemicalSystem, NeurochemicalEvent, EventType,
)


# ===========================================================================
# SECTION 1 -- SIMULATION PARAMETERS
# ===========================================================================

# --- Primary population sizes ---
N_V1 = 20        # 5 neurons per orientation column (4 columns)
N_A1 = 20        # tonotopic neurons

# --- V1 orientation columns ---
V1_ORIENTATIONS = [0.0, 45.0, 90.0, 135.0]   # degrees
N_PER_COL       = N_V1 // len(V1_ORIENTATIONS)  # 5

# --- A1 frequency range ---
A1_F_LOW        = 200.0    # Hz
A1_F_HIGH       = 6000.0   # Hz
A1_SIGMA_OCT    = 0.5      # tuning width in octaves

# --- Association cortex sizes ---
N_V2  = 12   # V2: 3 contour types x 4
N_V4  = 12   # V4: 3 color-opponent channels x 4
N_MT  = 12   # MT: 4 motion-direction columns x 3
N_IT  = 24   # IT: Hebbian object identity
N_A2  = 16   # A2: spectrotemporal
N_STG = 24   # STG: emergent phonemes
N_S1  = 32   # S1: somatosensory (body-part columns)
N_PFC = 32   # PFC: prefrontal (sub-regions within)
N_AG  = 16   # Angular Gyrus

# --- Stimulus drive levels ---
I_BASE          = 3.0      # uA/cm^2 -- constant subthreshold tonic drive
I_STIM_V1       = 8.5      # uA/cm^2 -- peak drive to a perfectly tuned V1 neuron
I_STIM_A1       = 8.5      # uA/cm^2 -- peak drive to a perfectly tuned A1 neuron

# --- White matter tract ---
WM_DELAY_MS     = 5.0      # axonal conduction delay (ms)
WM_INIT_WEIGHT  = 0.18     # initial weight
WM_CONN_WIDTH   = 0.35     # topographic spread (fraction of population width)
WM_G_PEAK       = 0.5      # AMPA peak conductance per spike (mS/cm^2)

# --- Spike detection ---
SPIKE_THR       = 0.0      # mV
REFRACTORY_MS   = 2.0

# --- Simulation ---
DT_MS           = 0.05     # 50 us -- accurate enough for HH
DURATION_MS     = 500.0

# --- Rate estimator decay ---
RATE_TAU_MS     = 20.0     # EMA time constant for firing rate estimates


# ===========================================================================
# SECTION 2 -- UTILITY FUNCTIONS AND BASE CLASS
# ===========================================================================

def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """Normalised 1-D Gaussian kernel (numpy only)."""
    x = np.arange(size) - size // 2
    g = np.exp(-x**2 / (2.0 * sigma**2))
    return g / g.sum()


def dog_convolve(frame: np.ndarray,
                 sigma1: float = 1.0, sigma2: float = 2.0) -> np.ndarray:
    """
    Difference-of-Gaussians convolution -- ON-center / OFF-surround receptive
    fields.  Separable 1-D passes for speed.  numpy only.

    frame  : (H, W) luminance image, values in [0, 1]
    Returns: (H, W) DoG response
    """
    k1 = _gaussian_kernel_1d(7, sigma1)
    k2 = _gaussian_kernel_1d(7, sigma2)

    def _conv2d_sep(img: np.ndarray, k: np.ndarray) -> np.ndarray:
        r = np.apply_along_axis(lambda row: np.convolve(row, k, mode='same'), 1, img)
        r = np.apply_along_axis(lambda col: np.convolve(col, k, mode='same'), 0, r)
        return r

    return _conv2d_sep(frame, k1) - _conv2d_sep(frame, k2)


def mel_filterbank(fft_magnitude: np.ndarray, sr: int = 16000,
                   n_mels: int = 128) -> np.ndarray:
    """
    Compute n_mels Mel-filterbank energies from a one-sided FFT magnitude
    spectrum.  numpy only (no librosa / torchaudio).

    fft_magnitude : shape (n_fft_bins,) -- one-sided magnitudes
    Returns       : shape (n_mels,) filterbank energies
    """
    n_fft_bins = len(fft_magnitude)
    n_fft      = (n_fft_bins - 1) * 2

    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_low  = _hz_to_mel(80.0)
    mel_high = _hz_to_mel(sr / 2.0)
    mel_pts  = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_pts   = np.array([_mel_to_hz(m) for m in mel_pts])
    bins     = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins     = np.clip(bins, 0, n_fft_bins - 1)

    fb = np.zeros(n_mels)
    for m in range(n_mels):
        lo, ctr, hi = bins[m], bins[m + 1], bins[m + 2]
        for k in range(lo, ctr):
            if ctr > lo:
                fb[m] += fft_magnitude[k] * (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi > ctr:
                fb[m] += fft_magnitude[k] * (hi - k) / (hi - ctr)
    return fb


class CorticalPopulation:
    """
    Base class: vectorised HH neuron population for all cortical regions.
    V1Population and A1Population do NOT inherit from this (backward-compat),
    but all new regions do.
    """

    def __init__(self, n: int):
        self.n = n
        m0, h0, n0 = gate_steady_state(V_rest)
        self.V       = np.full(n, V_rest)
        self.m       = np.full(n, m0)
        self.h       = np.full(n, h0)
        self.n_gate  = np.full(n, n0)
        self.last_spike  = np.full(n, -np.inf)
        self.spike_times = [[] for _ in range(n)]
        self._V_prev     = self.V.copy()
        self.rate_est    = np.zeros(n)   # EMA firing rate estimate (Hz)

    def step(self, dt: float, I_ext: np.ndarray,
             I_syn: np.ndarray | None = None) -> None:
        """Vectorised Euler step."""
        I_total = I_ext.copy()
        if I_syn is not None:
            I_total += I_syn
        I_Na = gNa * self.m**3 * self.h * (self.V - ENa)
        I_K  = gK  * self.n_gate**4     * (self.V - EK)
        I_L  = gL                       * (self.V - EL)
        self.V       = np.clip(self.V + dt * (I_total - I_Na - I_K - I_L) / Cm, -150.0, 150.0)
        self.m      += dt * (alpha_m(self.V) * (1 - self.m)      - beta_m(self.V) * self.m)
        self.h      += dt * (alpha_h(self.V) * (1 - self.h)      - beta_h(self.V) * self.h)
        self.n_gate += dt * (alpha_n(self.V) * (1 - self.n_gate) - beta_n(self.V) * self.n_gate)

    def detect_spikes(self, t: float) -> np.ndarray:
        """Vectorised spike detection with refractory period."""
        crossed    = (self._V_prev < SPIKE_THR) & (self.V >= SPIKE_THR)
        not_refrac = (t - self.last_spike) > REFRACTORY_MS
        fired      = crossed & not_refrac
        self.last_spike[fired] = t
        for i in np.where(fired)[0]:
            self.spike_times[i].append(float(t))
        self._V_prev = self.V.copy()
        return fired

    def update_rate_est(self, fired: np.ndarray, dt_ms: float = DT_MS) -> None:
        """Exponential moving average firing rate estimate (Hz)."""
        self.rate_est *= np.exp(-dt_ms / RATE_TAU_MS)
        self.rate_est[fired] += 1000.0 / RATE_TAU_MS


# ===========================================================================
# SECTION 3 -- V1 POPULATION
# ===========================================================================

class V1Population:
    """
    Primary Visual Cortex -- 20 HH neurons in 4 orientation columns.

    Orientation selectivity: cosine-squared tuning (Hubel & Wiesel).
    Pixel pipeline (pixel_to_drive): 64x64 frame -> luminance -> DoG ->
      oriented gradient responses -> complex-cell motion component.
    Demo pipeline (compute_drive): takes a scalar stimulus angle directly.
    """

    def __init__(self, n_neurons: int = N_V1):
        assert n_neurons % 4 == 0
        self.n    = n_neurons
        n_col     = n_neurons // 4

        self.pref_orient = np.array(
            [ang for ang in V1_ORIENTATIONS for _ in range(n_col)], dtype=float
        )
        self.column = np.array(
            [c for c in range(4) for _ in range(n_col)], dtype=int
        )

        m0, h0, n0 = gate_steady_state(V_rest)
        self.V       = np.full(n_neurons, V_rest)
        self.m       = np.full(n_neurons, m0)
        self.h       = np.full(n_neurons, h0)
        self.n_gate  = np.full(n_neurons, n0)

        self.last_spike  = np.full(n_neurons, -np.inf)
        self.spike_times = [[] for _ in range(n_neurons)]
        self._V_prev     = self.V.copy()

        # Rate estimator and pixel pipeline state
        self.rate_est  = np.zeros(n_neurons)
        self._prev_dog = None   # stored DoG response for complex/motion cells

    # ------------------------------------------------------------------
    def compute_drive(self, stim_angle_deg: float,
                      I_max: float, gain_mod: float = 1.0) -> np.ndarray:
        """
        Demo drive from a scalar orientation angle.
        I = I_BASE + I_max * cos^2(pref - stim) * gain_mod
        """
        delta_rad = np.radians(self.pref_orient - stim_angle_deg)
        tuning    = np.cos(delta_rad) ** 2
        return I_BASE + I_max * tuning * gain_mod

    # ------------------------------------------------------------------
    def pixel_to_drive(self, frame: np.ndarray,
                       gain_mod: float = 1.0) -> np.ndarray:
        """
        Full pixel input pipeline for a 64x64 frame.

        1. Convert to luminance (rec. 709 weights for colour, pass-through for grey)
        2. DoG convolution -> ON-center / OFF-surround responses
        3. Oriented gradient for each column (0/45/90/135 deg simple cells)
        4. Complex cell motion component from frame-to-frame DoG difference
        Returns drive array for all V1 neurons.
        """
        # Luminance
        if frame.ndim == 3:
            lum = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
        else:
            lum = frame.astype(float)
        lum = lum / (lum.max() + 1e-8)

        dog = dog_convolve(lum)

        # Oriented gradient responses
        dy = np.gradient(dog, axis=0)
        dx = np.gradient(dog, axis=1)

        orient_energy: dict = {}
        for ang in V1_ORIENTATIONS:
            theta = np.radians(ang)
            resp  = dx * np.sin(theta) + dy * np.cos(theta)
            orient_energy[ang] = float(np.mean(np.abs(resp)))

        # Complex cells: motion from frame-to-frame DoG delta
        if self._prev_dog is not None:
            motion_energy = float(np.mean(np.abs(dog - self._prev_dog)))
        else:
            motion_energy = 0.0
        self._prev_dog = dog.copy()

        # Per-neuron drive
        drive = np.empty(self.n)
        for i in range(self.n):
            ang     = self.pref_orient[i]
            simple  = orient_energy[ang]
            drive[i] = I_BASE + (simple + 0.3 * motion_energy) * I_STIM_V1 * gain_mod
        return drive

    # ------------------------------------------------------------------
    def step(self, dt: float, I_ext: np.ndarray,
             I_syn: np.ndarray | None = None) -> None:
        """Vectorised Euler step for all V1 neurons."""
        I_total = I_ext.copy()
        if I_syn is not None:
            I_total += I_syn
        I_Na = gNa * self.m**3 * self.h * (self.V - ENa)
        I_K  = gK  * self.n_gate**4     * (self.V - EK)
        I_L  = gL                       * (self.V - EL)
        self.V       = np.clip(self.V + dt * (I_total - I_Na - I_K - I_L) / Cm, -150.0, 150.0)
        self.m      += dt * (alpha_m(self.V) * (1 - self.m)      - beta_m(self.V) * self.m)
        self.h      += dt * (alpha_h(self.V) * (1 - self.h)      - beta_h(self.V) * self.h)
        self.n_gate += dt * (alpha_n(self.V) * (1 - self.n_gate) - beta_n(self.V) * self.n_gate)

    # ------------------------------------------------------------------
    def detect_spikes(self, t: float) -> np.ndarray:
        """Vectorised spike detection with refractory period."""
        crossed    = (self._V_prev < SPIKE_THR) & (self.V >= SPIKE_THR)
        not_refrac = (t - self.last_spike) > REFRACTORY_MS
        fired      = crossed & not_refrac
        self.last_spike[fired] = t
        for i in np.where(fired)[0]:
            self.spike_times[i].append(float(t))
        self._V_prev = self.V.copy()
        return fired

    # ------------------------------------------------------------------
    def update_rate_est(self, fired: np.ndarray, dt_ms: float = DT_MS) -> None:
        """EMA firing rate estimator (Hz)."""
        self.rate_est *= np.exp(-dt_ms / RATE_TAU_MS)
        self.rate_est[fired] += 1000.0 / RATE_TAU_MS


# ===========================================================================
# SECTION 4 -- A1 POPULATION
# ===========================================================================

class A1Population:
    """
    Primary Auditory Cortex -- 20 HH neurons arranged tonotopically.

    Tonotopic: log-spaced preferred frequencies (cochlear scale).
    Tuning: Gaussian in log-frequency space.
    Onset/offset detectors: transient current on stimulus change.
    AM detectors: track rate-of-change of spectral energy.
    Pixel/FFT pipeline: mel_filterbank_drive() for real microphone input.
    Demo pipeline: compute_drive(freq_hz) for stimulus-schedule.
    """

    def __init__(self, n_neurons: int = N_A1):
        self.n = n_neurons

        self.pref_freq = np.logspace(
            np.log10(A1_F_LOW), np.log10(A1_F_HIGH), n_neurons
        )

        m0, h0, n0 = gate_steady_state(V_rest)
        self.V       = np.full(n_neurons, V_rest)
        self.m       = np.full(n_neurons, m0)
        self.h       = np.full(n_neurons, h0)
        self.n_gate  = np.full(n_neurons, n0)

        self.last_spike  = np.full(n_neurons, -np.inf)
        self.spike_times = [[] for _ in range(n_neurons)]
        self._V_prev     = self.V.copy()

        # Onset transient (existing)
        self._onset_current = np.zeros(n_neurons)
        self._onset_tau_ms  = 10.0
        self._prev_stim_hz  = 0.0

        # New: offset detector, Mel pipeline state
        self._offset_current  = np.zeros(n_neurons)
        self._offset_tau_ms   = 8.0
        self._prev_mel_active = np.zeros(n_neurons, dtype=bool)
        self._prev_mel_energy = None
        self._am_energy_prev  = None

        # Rate estimator
        self.rate_est = np.zeros(n_neurons)

    # ------------------------------------------------------------------
    def compute_drive(self, stim_freq_hz: float,
                      I_max: float, gain_mod: float = 1.0) -> np.ndarray:
        """
        Demo drive from a scalar stimulus frequency.
        Gaussian tuning in log-frequency space + onset transient.
        """
        if stim_freq_hz <= 0:
            return np.full(self.n, I_BASE) + self._onset_current

        octave_dist = np.log2(self.pref_freq / stim_freq_hz)
        tuning      = np.exp(-octave_dist**2 / (2 * A1_SIGMA_OCT**2))

        if abs(stim_freq_hz - self._prev_stim_hz) > 1.0:
            self._onset_current  = 4.0 * tuning * gain_mod
            self._offset_current[self._prev_mel_active] += 2.0
            self._prev_stim_hz   = stim_freq_hz

        return I_BASE + I_max * tuning * gain_mod + self._onset_current

    # ------------------------------------------------------------------
    def mel_filterbank_drive(self, mel_energies: np.ndarray,
                             gain_mod: float = 1.0) -> np.ndarray:
        """
        Full auditory input pipeline from 128-band Mel filterbank energies.

        mel_energies: shape (128,) from mel_filterbank()
        Maps each neuron's preferred frequency to a Gaussian-weighted sum of
        nearby Mel bands.  Also updates onset/offset and AM detector state.
        """
        n_mels    = len(mel_energies)
        mel_low   = 2595.0 * np.log10(1.0 + 80.0   / 700.0)
        mel_high  = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
        mel_ctrs  = np.linspace(mel_low, mel_high, n_mels)
        sigma_mel = (mel_high - mel_low) / n_mels * 2.5

        pref_mel = 2595.0 * np.log10(
            1.0 + np.clip(self.pref_freq, 80.0, 8000.0) / 700.0
        )

        # Gaussian-weighted Mel energy per neuron
        neuron_energy = np.array([
            float(
                np.dot(np.exp(-(mel_ctrs - pm)**2 / (2 * sigma_mel**2)), mel_energies)
                / (np.exp(-(mel_ctrs - pm)**2 / (2 * sigma_mel**2)).sum() + 1e-8)
            )
            for pm in pref_mel
        ])

        max_e = neuron_energy.max()
        if max_e > 1e-8:
            neuron_energy /= max_e

        # Onset / offset detection
        new_active = neuron_energy > 0.3
        if self._prev_mel_energy is not None:
            onset  = new_active & ~self._prev_mel_active
            offset = ~new_active & self._prev_mel_active
            self._onset_current[onset]   += 3.5
            self._offset_current[offset] += 2.0
        self._prev_mel_active = new_active.copy()
        self._prev_mel_energy = neuron_energy.copy()

        # AM detector: rate-of-change of overall spectral energy
        total_energy = float(mel_energies.mean())
        if self._am_energy_prev is not None:
            am_rate     = abs(total_energy - self._am_energy_prev) / (DT_MS * 1e-3)
            am_boost    = np.clip(am_rate * 0.1, 0.0, 2.0)
            neuron_energy = np.clip(neuron_energy + am_boost * 0.05, 0.0, 1.0)
        self._am_energy_prev = total_energy

        return (I_BASE + I_STIM_A1 * neuron_energy * gain_mod
                + self._onset_current + self._offset_current)

    # ------------------------------------------------------------------
    def decay_onset(self, dt: float) -> None:
        """Exponential decay of onset and offset transient currents."""
        self._onset_current  *= np.exp(-dt / self._onset_tau_ms)
        self._offset_current *= np.exp(-dt / self._offset_tau_ms)

    # ------------------------------------------------------------------
    def step(self, dt: float, I_ext: np.ndarray,
             I_syn: np.ndarray | None = None) -> None:
        """Vectorised Euler step for all A1 neurons."""
        I_total = I_ext.copy()
        if I_syn is not None:
            I_total += I_syn
        I_Na = gNa * self.m**3 * self.h * (self.V - ENa)
        I_K  = gK  * self.n_gate**4     * (self.V - EK)
        I_L  = gL                       * (self.V - EL)
        self.V       = np.clip(self.V + dt * (I_total - I_Na - I_K - I_L) / Cm, -150.0, 150.0)
        self.m      += dt * (alpha_m(self.V) * (1 - self.m)      - beta_m(self.V) * self.m)
        self.h      += dt * (alpha_h(self.V) * (1 - self.h)      - beta_h(self.V) * self.h)
        self.n_gate += dt * (alpha_n(self.V) * (1 - self.n_gate) - beta_n(self.V) * self.n_gate)

    # ------------------------------------------------------------------
    def detect_spikes(self, t: float) -> np.ndarray:
        """Vectorised spike detection with refractory period."""
        crossed    = (self._V_prev < SPIKE_THR) & (self.V >= SPIKE_THR)
        not_refrac = (t - self.last_spike) > REFRACTORY_MS
        fired      = crossed & not_refrac
        self.last_spike[fired] = t
        for i in np.where(fired)[0]:
            self.spike_times[i].append(float(t))
        self._V_prev = self.V.copy()
        return fired

    # ------------------------------------------------------------------
    def update_rate_est(self, fired: np.ndarray, dt_ms: float = DT_MS) -> None:
        """EMA firing rate estimator (Hz)."""
        self.rate_est *= np.exp(-dt_ms / RATE_TAU_MS)
        self.rate_est[fired] += 1000.0 / RATE_TAU_MS


# ===========================================================================
# SECTION 5 -- VISUAL ASSOCIATION CORTICES (V2, V4, MT, IT)
# ===========================================================================

class V2Population(CorticalPopulation):
    """
    V2: Combines V1 orientation column outputs to detect contours and
    illusory boundaries.  Three contour types x 4 neurons each:
      Type 0 (H-contour): V1 col-0 (0deg) + col-1 (45deg)
      Type 1 (V-contour): V1 col-2 (90deg) + col-3 (135deg)
      Type 2 (D-contour): V1 col-1 (45deg) + col-2 (90deg)
    """
    N_TYPES = 3

    def __init__(self, n: int = N_V2):
        super().__init__(n)
        assert n % self.N_TYPES == 0
        n_per = n // self.N_TYPES
        self._type_idx = np.array([t for t in range(self.N_TYPES) for _ in range(n_per)],
                                  dtype=int)

    def compute_drive(self, v1_rates: np.ndarray,
                      gain_mod: float = 1.0) -> np.ndarray:
        """v1_rates: shape (N_V1,) EMA firing rates from V1Population."""
        col_rates = np.array([
            v1_rates[c * N_PER_COL:(c + 1) * N_PER_COL].mean()
            for c in range(4)
        ])
        contour = np.array([
            (col_rates[0] + col_rates[1]) * 0.5,   # H-contour
            (col_rates[2] + col_rates[3]) * 0.5,   # V-contour
            (col_rates[1] + col_rates[2]) * 0.5,   # D-contour
        ])
        max_c = contour.max()
        if max_c > 1e-8:
            contour /= max_c

        drive = np.empty(self.n)
        for i in range(self.n):
            drive[i] = I_BASE + contour[self._type_idx[i]] * I_STIM_V1 * gain_mod
        return drive


class V4Population(CorticalPopulation):
    """
    V4: Color opponency neurons + complex shape selectivity.
    Three color-opponent channels x 4 neurons each:
      Ch 0: R - G  (red-green)
      Ch 1: G - B  (green-blue)
      Ch 2: B - R  (blue-red)
    Shape input: V2 rates boost all channels (form-colour binding).
    """
    N_CHANNELS = 3

    def __init__(self, n: int = N_V4):
        super().__init__(n)
        assert n % self.N_CHANNELS == 0
        n_per = n // self.N_CHANNELS
        self._ch_idx = np.array([ch for ch in range(self.N_CHANNELS) for _ in range(n_per)],
                                dtype=int)

    def compute_drive(self, rgb_mean: np.ndarray | None = None,
                      v2_rates: np.ndarray | None = None,
                      gain_mod: float = 1.0) -> np.ndarray:
        """
        rgb_mean : (3,) mean RGB of scene, values in [0, 1].  None -> neutral grey.
        v2_rates : (N_V2,) V2 EMA rates for shape input.
        """
        if rgb_mean is not None and len(rgb_mean) == 3:
            r, g, b = float(rgb_mean[0]), float(rgb_mean[1]), float(rgb_mean[2])
        else:
            r = g = b = 0.33

        opponent = np.array([
            max(0.0, r - g),   # R-G
            max(0.0, g - b),   # G-B
            max(0.0, b - r),   # B-R
        ])

        if v2_rates is not None and v2_rates.max() > 1e-8:
            shape_boost = np.clip(v2_rates.mean() / 100.0, 0.0, 0.5)
            opponent    = np.clip(opponent + shape_boost, 0.0, 1.0)

        drive = np.empty(self.n)
        for i in range(self.n):
            drive[i] = I_BASE + opponent[self._ch_idx[i]] * I_STIM_V1 * gain_mod
        return drive


class MTPopulation(CorticalPopulation):
    """
    MT/V5: Motion direction columns.  Computes optic flow from frame-to-frame
    V1 rate changes.  4 direction preferences x 3 neurons each:
      Dir 0: 'up'    -- positive delta in V1 col-0
      Dir 1: 'down'  -- negative delta in V1 col-1
      Dir 2: 'left'  -- positive delta in V1 col-2
      Dir 3: 'right' -- negative delta in V1 col-3
    """
    DIRECTIONS = ['up', 'down', 'left', 'right']

    def __init__(self, n: int = N_MT):
        super().__init__(n)
        assert n % 4 == 0
        n_per = n // 4
        self._dir_idx      = np.array([d for d in range(4) for _ in range(n_per)], dtype=int)
        self._prev_v1_rates = None

    def compute_drive(self, v1_rates: np.ndarray,
                      gain_mod: float = 1.0) -> np.ndarray:
        """v1_rates: (N_V1,) EMA firing rates from V1."""
        if self._prev_v1_rates is None:
            self._prev_v1_rates = v1_rates.copy()
            return np.full(self.n, I_BASE)

        delta = v1_rates - self._prev_v1_rates
        self._prev_v1_rates = v1_rates.copy()

        col_delta = np.array([
            delta[c * N_PER_COL:(c + 1) * N_PER_COL].mean()
            for c in range(4)
        ])
        dir_resp = np.array([
            max(0.0,  col_delta[0]),   # up
            max(0.0, -col_delta[1]),   # down
            max(0.0,  col_delta[2]),   # left
            max(0.0, -col_delta[3]),   # right
        ])
        max_d = dir_resp.max()
        if max_d > 1e-8:
            dir_resp /= max_d

        drive = np.empty(self.n)
        for i in range(self.n):
            drive[i] = I_BASE + dir_resp[self._dir_idx[i]] * I_STIM_V1 * gain_mod
        return drive


class ITPopulation(CorticalPopulation):
    """
    IT: Inferior Temporal Cortex.  Object identity via Hebbian binding.

    No categories are hardcoded.  W_hebb (N_IT x N_input) is initialised to
    small random values.  Repeated co-activation of V2/V4/MT patterns with
    IT neurons strengthens the binding (BCM-variant Hebbian rule).
    Each stable IT attractor becomes one emergent object representation.
    """
    HEBB_LR       = 4e-4
    HEBB_BCM_COEF = 0.10   # homeostatic depression coefficient

    def __init__(self, n: int = N_IT,
                 n_input: int = N_V2 + N_V4 + N_MT):
        super().__init__(n)
        self.n_input = n_input
        rng = np.random.default_rng(42)
        W_cpu = rng.uniform(0.0, 0.01, (n, n_input)).astype(np.float32)
        self.W_hebb = _xp.asarray(W_cpu)   # on GPU if available

    def compute_drive(self, input_rates: np.ndarray,
                      gain_mod: float = 1.0) -> np.ndarray:
        """input_rates: (N_V2+N_V4+N_MT,) concatenated EMA rates."""
        inp = _xp.asarray(input_rates.astype(np.float32))
        inp_norm = inp / (float(_xp.linalg.norm(inp)) + 1e-8)
        raw      = self.W_hebb @ inp_norm
        max_r    = float(raw.max())
        if max_r > 1e-8:
            raw = raw / max_r
        result = _to_numpy(raw)
        return I_BASE + result * I_STIM_V1 * gain_mod

    def hebbian_update(self, it_rates: np.ndarray,
                       input_rates: np.ndarray) -> None:
        """
        BCM-variant Hebbian update.
        dW = lr * post * pre^T - lr * BCM_coef * post^2 * pre^T
        Called once per simulation step after rate estimates are updated.
        """
        inp = _xp.asarray(input_rates.astype(np.float32))
        itr = _xp.asarray(it_rates.astype(np.float32))
        inp_norm = inp / (float(_xp.linalg.norm(inp)) + 1e-8)
        it_norm  = itr / (float(_xp.linalg.norm(itr)) + 1e-8)
        dW = self.HEBB_LR * (
            _xp.outer(it_norm, inp_norm)
            - self.HEBB_BCM_COEF * _xp.outer(it_norm**2, inp_norm)
        )
        self.W_hebb = _xp.clip(self.W_hebb + dW, 0.0, 1.0)


# ===========================================================================
# SECTION 6 -- AUDITORY ASSOCIATION CORTEX (A2, STG)
# ===========================================================================

class A2Population(CorticalPopulation):
    """
    A2: Auditory Association.  Complex spectrotemporal pattern detection.
    Integrates a rolling history of A1 firing rates across N_HISTORY steps,
    learning to respond to combinations of frequency and time.
    """
    N_HISTORY = 8

    def __init__(self, n: int = N_A2, n_a1: int = N_A1):
        super().__init__(n)
        self.n_a1    = n_a1
        self._a1_buf = deque(maxlen=self.N_HISTORY)
        rng = np.random.default_rng(123)
        self.W_temp  = rng.uniform(0.0, 0.05, (n, n_a1 * self.N_HISTORY))

    def compute_drive(self, a1_rates: np.ndarray,
                      gain_mod: float = 1.0) -> np.ndarray:
        """a1_rates: (N_A1,) EMA firing rates from A1."""
        self._a1_buf.append(a1_rates.copy())
        if len(self._a1_buf) < 2:
            return np.full(self.n, I_BASE)

        hist = np.zeros(self.n_a1 * self.N_HISTORY)
        for j, h in enumerate(self._a1_buf):
            hist[j * self.n_a1:(j + 1) * self.n_a1] = h
        hist_norm = hist / (np.linalg.norm(hist) + 1e-8)

        raw   = self.W_temp @ hist_norm
        max_r = raw.max()
        if max_r > 1e-8:
            raw /= max_r
        return I_BASE + raw * I_STIM_A1 * gain_mod


class STGPopulation(CorticalPopulation):
    """
    STG: Superior Temporal Gyrus.  Emergent phoneme-like representations.

    No phonemes hardcoded.  STDP + Hebbian reinforcement cause STG neuron
    populations to self-organise around recurring spectrotemporal patterns.
    Dopamine/oxytocin neuro_scale (provided by caller) gates learning.
    Language acquisition (word 'CAINE') lives here -- every co-occurrence
    with limbic reward strengthens the binding.
    """
    STDP_A_PLUS_STG  = 0.005
    STDP_TAU_STG_MS  = 25.0
    HEBB_LR          = 3e-4

    def __init__(self, n: int = N_STG, n_a2: int = N_A2):
        super().__init__(n)
        self.n_a2 = n_a2
        rng = np.random.default_rng(456)
        W_in_cpu    = rng.uniform(0.0, 0.02,  (n, n_a2)).astype(np.float32)
        W_recur_cpu = rng.uniform(0.0, 0.005, (n, n)).astype(np.float32)
        np.fill_diagonal(W_recur_cpu, 0.0)
        self.W_in    = _xp.asarray(W_in_cpu)
        self.W_recur = _xp.asarray(W_recur_cpu)
        self._last_stg_spike = np.full(n, -np.inf)

    def compute_drive(self, a2_rates: np.ndarray,
                      gain_mod: float = 1.0) -> np.ndarray:
        """a2_rates: (N_A2,) EMA firing rates from A2."""
        inp = _xp.asarray(a2_rates.astype(np.float32))
        inp_norm = inp / (float(_xp.linalg.norm(inp)) + 1e-8)
        rate_dev = _xp.asarray(self.rate_est.astype(np.float32))
        feedfwd  = self.W_in    @ inp_norm
        recur    = self.W_recur @ (rate_dev / (float(rate_dev.max()) + 1e-8))
        raw = feedfwd + 0.3 * recur
        max_r = float(raw.max())
        if max_r > 1e-8:
            raw = raw / max_r
        result = _to_numpy(raw)
        return I_BASE + result * I_STIM_A1 * gain_mod

    def stdp_update(self, stg_fired: np.ndarray, t: float,
                    a2_rates: np.ndarray, neuro_scale: float = 1.0) -> None:
        """
        STDP: when an STG neuron fires, strengthen W_in weights that
        correspond to currently active A2 inputs.
        Called after each step.
        """
        if not stg_fired.any():
            return
        inp_act = a2_rates / (a2_rates.max() + 1e-8)
        inp_dev = _xp.asarray(inp_act.astype(np.float32))
        for j in np.where(stg_fired)[0]:
            self._last_stg_spike[j] = t
            dw = self.STDP_A_PLUS_STG * inp_dev * neuro_scale
            self.W_in[j, :] = _xp.clip(self.W_in[j, :] + dw, 0.0, 1.0)


# ===========================================================================
# SECTION 7 -- SOMATOSENSORY CORTEX (S1)
# ===========================================================================

class S1Population(CorticalPopulation):
    """
    S1: Somatosensory Cortex.  Proprioceptive input from avatar joint angles.

    Body-part columns, sized proportional to motor importance:
      hand=12, arm=6, head=6, torso=4, foot=4  (total 32)

    Efference copy: M1 sends a proprioceptive prediction before movement.
    S1 computes expected vs actual discrepancy -> extra drive.
    S1 activity is the foundation of the self/world boundary.
    """
    BODY_PARTS = {'hand': 12, 'arm': 6, 'head': 6, 'torso': 4, 'foot': 4}

    def __init__(self):
        n = sum(self.BODY_PARTS.values())   # 32
        super().__init__(n)
        self._part_slice: dict = {}
        idx = 0
        for part, size in self.BODY_PARTS.items():
            self._part_slice[part] = slice(idx, idx + size)
            idx += size
        self._efference_copy = np.full(n, I_BASE)

    def compute_drive(self, joint_angles: dict,
                      gain_mod: float = 1.0) -> np.ndarray:
        """
        joint_angles: dict {part_name: activation in [0,1]}
        Missing parts default to 0.0 (at rest).
        """
        drive = np.full(self.n, I_BASE)
        for part, sl in self._part_slice.items():
            act       = float(joint_angles.get(part, 0.0))
            drive[sl] = I_BASE + act * I_STIM_V1 * gain_mod

        # Efference copy: prediction-error drives saliency
        pred_error = np.abs(drive - self._efference_copy)
        drive     += 0.15 * pred_error * gain_mod
        return drive

    def receive_efference_copy(self, m1_prediction: np.ndarray) -> None:
        """M1 calls this before each motor command to set expected proprioception."""
        if len(m1_prediction) == self.n:
            self._efference_copy = m1_prediction.copy()


# ===========================================================================
# SECTION 8 -- PREFRONTAL CORTEX (PFC)
# ===========================================================================

class PFCPopulation(CorticalPopulation):
    """
    PFC: Prefrontal Cortex.  Developmentally gated from Stage 2.

    Sub-regions within the 32-neuron population:
      [0:8]   mPFC -- self-model (self-referential processing)
      [8:16]  inhibitory control / error monitoring
      [16:24] working memory / planning / action simulation
      [24:32] PCC proxy -- self-related memory retrieval (for DMN)

    Developmental gating:
      stage < 2 -> myelination_factor = 0.0  (connections suppressed)
      stage 2   -> 0.5 (partially myelinated)
      stage >= 3 -> 1.0 (fully functional)

    Working memory: 500ms decay recurrent loop.  Loaded by strong input;
    freely decays without refresh.
    """
    SUBREGION_MPFC  = slice(0,  8)
    SUBREGION_INHIB = slice(8,  16)
    SUBREGION_WM    = slice(16, 24)
    SUBREGION_PCC   = slice(24, 32)

    WM_DECAY_TAU_MS  = 600.0   # working memory persistence ~600ms
    WM_RECUR_GAIN    = 0.20    # recurrent excitation strength

    def __init__(self, n: int = N_PFC, stage: int = 0):
        super().__init__(n)
        self.stage = stage
        rng = np.random.default_rng(789)
        self.W_recur = rng.uniform(0.0, 0.008, (n, n))
        np.fill_diagonal(self.W_recur, 0.0)
        self._wm_trace = np.zeros(n)   # persistent activity trace

    @property
    def myelination_factor(self) -> float:
        """Fraction [0, 1] of PFC output connections that are myelinated."""
        if self.stage < 2:
            return 0.0
        return float(np.clip((self.stage - 2.0) + 0.5, 0.0, 1.0))

    def compute_drive(self, limbic_rates: np.ndarray | None = None,
                      gain_mod: float = 1.0) -> np.ndarray:
        """
        limbic_rates: (N,) amygdala / hippocampus firing rates (optional).
        WM recurrent loop adds self-sustaining excitation.
        PFC is suppressed until Stage 2 (myelination_factor = 0).
        """
        mf    = self.myelination_factor
        drive = np.full(self.n, I_BASE * (0.5 + 0.5 * mf))

        # WM recurrent excitation
        recur  = self.W_recur @ self._wm_trace
        drive += self.WM_RECUR_GAIN * recur * mf

        # Limbic input -> mPFC and inhibitory control
        if limbic_rates is not None and len(limbic_rates) > 0:
            lim = np.clip(float(limbic_rates.mean()) / 100.0, 0.0, 1.0)
            drive[self.SUBREGION_MPFC]  += lim * I_STIM_V1 * gain_mod * mf
            drive[self.SUBREGION_INHIB] += lim * I_STIM_V1 * gain_mod * mf

        return drive

    def update_wm(self, fired: np.ndarray, dt_ms: float) -> None:
        """Update working memory trace after each step."""
        self._wm_trace *= np.exp(-dt_ms / self.WM_DECAY_TAU_MS)
        self._wm_trace[fired] += 0.08
        self._wm_trace = np.clip(self._wm_trace, 0.0, 1.0)

    @property
    def mPFC_rates(self) -> np.ndarray:
        """mPFC sub-region activity trace (used by DMNMonitor)."""
        return self._wm_trace[self.SUBREGION_MPFC]

    @property
    def PCC_rates(self) -> np.ndarray:
        """PCC proxy sub-region activity trace (used by DMNMonitor)."""
        return self._wm_trace[self.SUBREGION_PCC]


# ===========================================================================
# SECTION 9 -- DEFAULT MODE NETWORK
# ===========================================================================

class AngularGyrusPopulation(CorticalPopulation):
    """
    Angular Gyrus: Integration of self and social information.
    Receives from STG (language) and mPFC (self-model).
    Together with mPFC and PCC forms the Default Mode Network.
    """
    AG_DECAY_TAU_MS = 200.0

    def __init__(self, n: int = N_AG):
        super().__init__(n)
        self._ag_trace = np.zeros(n)

    def compute_drive(self, stg_rates: np.ndarray | None = None,
                      mpfc_rates: np.ndarray | None = None,
                      gain_mod: float = 1.0) -> np.ndarray:
        drive = np.full(self.n, I_BASE)
        if stg_rates is not None and stg_rates.max() > 1e-8:
            boost  = np.clip(float(stg_rates.mean()) / 100.0, 0.0, 1.0)
            drive += boost * I_STIM_A1 * gain_mod
        if mpfc_rates is not None and mpfc_rates.max() > 1e-8:
            boost  = np.clip(float(mpfc_rates.mean()), 0.0, 1.0) * 0.3
            drive += boost * I_STIM_V1 * gain_mod
        return drive

    def update_trace(self, fired: np.ndarray, dt_ms: float) -> None:
        """EMA trace for DMN correlation tracking."""
        self._ag_trace *= np.exp(-dt_ms / self.AG_DECAY_TAU_MS)
        self._ag_trace[fired] += 0.08
        self._ag_trace = np.clip(self._ag_trace, 0.0, 1.0)

    @property
    def ag_rates(self) -> np.ndarray:
        return self._ag_trace


class DMNMonitor:
    """
    Default Mode Network monitor.

    Tracks correlated resting-state activity across three regions:
      - Medial PFC    (PFCPopulation.mPFC_rates)
      - PCC           (PFCPopulation.PCC_rates)
      - Angular Gyrus (AngularGyrusPopulation.ag_rates)

    During active sensory engagement these are suppressed.  During rest they
    become the most active regions.  Their emergence -- sustained correlated
    activity (r >= CORR_THRESHOLD for SUSTAIN_MS) -- is a developmental
    milestone logged to output/dmn_log.jsonl and printed.
    """
    CORR_THRESHOLD = 0.55   # min pairwise correlation for DMN detection
    SUSTAIN_MS     = 150.0  # how long correlations must be sustained
    BUFFER_MS      = 400.0  # rolling window for correlation computation

    def __init__(self, pfc: PFCPopulation, ang: AngularGyrusPopulation,
                 dt_ms: float = DT_MS):
        self.pfc    = pfc
        self.ang    = ang
        self.dt_ms  = dt_ms
        buf_len     = max(20, int(self.BUFFER_MS / dt_ms))
        self._mpfc_buf = deque(maxlen=buf_len)
        self._pcc_buf  = deque(maxlen=buf_len)
        self._ag_buf   = deque(maxlen=buf_len)
        self._sustain_ms = 0.0
        self.dmn_active  = False
        self.dmn_emerged = False
        self.dmn_emerge_t: float | None = None
        self.corr_log: list  = []   # list of (t_ms, avg_corr)

    def update(self, t: float, is_resting: bool = True) -> bool:
        """
        Push current sub-region activity, compute rolling correlations.
        Returns True when DMN is active.
        """
        self._mpfc_buf.append(float(self.pfc.mPFC_rates.mean()))
        self._pcc_buf.append(float(self.pfc.PCC_rates.mean()))
        self._ag_buf.append(float(self.ang.ag_rates.mean()))

        if len(self._mpfc_buf) < 20:
            return False

        mpfc_arr = np.array(self._mpfc_buf)
        pcc_arr  = np.array(self._pcc_buf)
        ag_arr   = np.array(self._ag_buf)

        corrs = []
        for a, b in [(mpfc_arr, pcc_arr), (mpfc_arr, ag_arr), (pcc_arr, ag_arr)]:
            if a.std() > 1e-9 and b.std() > 1e-9:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
            else:
                corrs.append(0.0)

        avg_corr = float(np.mean(corrs))
        dmn_now  = (avg_corr >= self.CORR_THRESHOLD) and is_resting

        if dmn_now:
            self._sustain_ms += self.dt_ms
            if self._sustain_ms >= self.SUSTAIN_MS:
                self.dmn_active = True
                if not self.dmn_emerged:
                    self.dmn_emerged  = True
                    self.dmn_emerge_t = t
                    self._log_emergence(t, avg_corr)
                    print(f'[DMN] *** MILESTONE: Default Mode Network emerged at '
                          f't={t:.1f}ms  corr={avg_corr:.3f} ***')
        else:
            self._sustain_ms = max(0.0, self._sustain_ms - self.dt_ms * 3)
            if self._sustain_ms < self.SUSTAIN_MS * 0.3:
                self.dmn_active = False

        if len(self.corr_log) % 200 == 0:
            self.corr_log.append((t, avg_corr))

        return self.dmn_active

    def _log_emergence(self, t: float, corr: float) -> None:
        import json
        entry    = {'event': 'dmn_emergence', 't_ms': round(t, 2),
                    'correlation': round(corr, 4)}
        log_path = _paths.DMN_LOG
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except OSError:
            pass


# ===========================================================================
# SECTION 10 -- THALAMUS
# ===========================================================================

class ThalamusRelay:
    """
    Thalamus: Sensory relay nuclei with thalamocortical feedback gating.

    Relay nuclei:
      VISUAL      -- LGN: relays V1 input (and cross-modal from Pulvinar)
      AUDITORY    -- MGN: relays A1 input
      SOMATOSENS  -- VPL: relays S1 joint-angle input

    Thalamocortical loop: cortex sends feedback -> gates own relay gain.
    High cortical activity -> gate increases (amplify signal).
    Sustained high activity -> slight suppression (habituation).

    Pulvinar: detects cross-modal co-activation (visual + auditory) and
    boosts both streams -- attention spotlight on multimodal events.
    """
    VISUAL      = 'visual'
    AUDITORY    = 'auditory'
    SOMATOSENS  = 'somatosensory'

    BASELINE_GATE  = 0.85
    FB_RATE        = 0.008   # how fast gate tracks cortical feedback
    PULVINAR_BOOST = 0.12

    def __init__(self):
        self._gates = {
            self.VISUAL:     self.BASELINE_GATE,
            self.AUDITORY:   self.BASELINE_GATE,
            self.SOMATOSENS: self.BASELINE_GATE,
        }
        self._pulvinar_vis = 0.0
        self._pulvinar_aud = 0.0

    def relay(self, modality: str, input_rates: np.ndarray) -> np.ndarray:
        """Apply relay nucleus gain to incoming sensory rates."""
        gate = self._gates.get(modality, self.BASELINE_GATE)
        return input_rates * gate

    def cortical_feedback(self, modality: str,
                          cortex_mean_rate_hz: float) -> None:
        """
        Cortex calls this each step with its current mean firing rate.
        Thalamic gate tracks cortical activity: attention amplifies,
        habituation slightly suppresses sustained input.
        """
        act   = np.clip(cortex_mean_rate_hz / 80.0, 0.0, 1.5)
        delta = self.FB_RATE * (act - self._gates[modality])
        self._gates[modality] = float(np.clip(
            self._gates[modality] + delta, 0.2, 1.3
        ))

    def pulvinar(self, v1_rates: np.ndarray,
                 a1_rates: np.ndarray) -> tuple:
        """
        Pulvinar nucleus: cross-modal integration.
        Co-active visual + auditory -> amplify both (attention spotlight).
        Returns (boosted_v1_rates, boosted_a1_rates).
        """
        vis = float(v1_rates.mean())
        aud = float(a1_rates.mean())
        self._pulvinar_vis = self._pulvinar_vis * 0.9 + vis * 0.1
        self._pulvinar_aud = self._pulvinar_aud * 0.9 + aud * 0.1

        cross   = float(np.sqrt(self._pulvinar_vis * self._pulvinar_aud + 1e-8))
        boost   = 1.0 + self.PULVINAR_BOOST * cross
        return v1_rates * boost, a1_rates * boost


# ===========================================================================
# SECTION 11 -- WHITE MATTER TRACT (V1 -> A1)
# ===========================================================================

class WhiteMatterTract:
    """
    Long-range axonal projection from one cortical region to another.

    - Topographic Gaussian connectivity (nearby neurons connect more strongly)
    - AMPA conductance kinetics with exponential decay (tau = 5 ms)
    - Axonal conduction delay via a deque-based spike buffer
    - Nearest-neighbour STDP, gated and scaled by neurochemicals
    """

    def __init__(self, n_pre: int, n_post: int,
                 delay_ms: float    = WM_DELAY_MS,
                 init_weight: float = WM_INIT_WEIGHT,
                 conn_width: float  = WM_CONN_WIDTH):
        self.n_pre    = n_pre
        self.n_post   = n_post
        self.delay_ms = delay_ms

        pre_pos  = np.linspace(0, 1, n_pre)
        post_pos = np.linspace(0, 1, n_post)
        dist     = np.abs(pre_pos[:, None] - post_pos[None, :])
        self.weights = init_weight * np.exp(-dist**2 / (2 * conn_width**2))

        self.g_ampa  = np.zeros((n_pre, n_post))
        self.delay_buffer: deque = deque()

        self.last_pre_spike  = np.full(n_pre,  -np.inf)
        self.last_post_spike = np.full(n_post, -np.inf)

        self.neuro_stdp_scale = 1.0
        self.neuro_gate       = 1.0

        self.mean_weight_log: list = []

    def on_pre_spikes(self, t: float, fired: np.ndarray) -> None:
        for i in np.where(fired)[0]:
            self.delay_buffer.append((t + self.delay_ms, int(i)))
            self.last_pre_spike[int(i)] = t

    def deliver_and_stdp_depress(self, t: float) -> None:
        while self.delay_buffer and self.delay_buffer[0][0] <= t:
            _, pre_i = self.delay_buffer.popleft()
            self.g_ampa[pre_i, :] += self.weights[pre_i, :] * WM_G_PEAK

            dt_stdp = t - self.last_post_spike
            valid   = (dt_stdp > 0) & (dt_stdp < 300)
            dw      = np.where(
                valid,
                -STDP_A_MINUS * np.exp(-dt_stdp / STDP_TAU_MINUS)
                * self.neuro_stdp_scale * self.neuro_gate,
                0.0
            )
            self.weights[pre_i, :] = np.clip(self.weights[pre_i, :] + dw, 0.0, 1.0)

    def on_post_spikes(self, t: float, fired: np.ndarray) -> None:
        for j in np.where(fired)[0]:
            self.last_post_spike[int(j)] = t
            dt_stdp = t - self.last_pre_spike
            valid   = (dt_stdp > 0) & (dt_stdp < 300)
            dw      = np.where(
                valid,
                STDP_A_PLUS * np.exp(-dt_stdp / STDP_TAU_PLUS)
                * self.neuro_stdp_scale * self.neuro_gate,
                0.0
            )
            self.weights[:, int(j)] = np.clip(self.weights[:, int(j)] + dw, 0.0, 1.0)

    def get_synaptic_currents(self, V_post: np.ndarray) -> np.ndarray:
        g_total       = self.g_ampa.sum(axis=0)
        I_syn_outward = g_total * (V_post - AMPA_E_REV)
        return -I_syn_outward

    def update(self, dt: float) -> None:
        self.g_ampa *= np.exp(-dt / AMPA_TAU)
        self.mean_weight_log.append(float(self.weights.mean()))


# ===========================================================================
# SECTION 12 -- STIMULUS SCHEDULE
# ===========================================================================

def _build_stimulus_schedule(duration_ms: float, dt_ms: float) -> tuple:
    """
    Returns (stim_v1, stim_a1, neuro_events):
      stim_v1[i]   = (angle_deg, I_max)  -- angle=-1 means no visual input
      stim_a1[i]   = (freq_hz,   I_max)  -- freq=0  means silence
      neuro_events = {timestep_index: [NeurochemicalEvent, ...]}
    """
    N       = int(duration_ms / dt_ms) + 1
    stim_v1 = [(-1.0, 0.0)] * N
    stim_a1 = [(0.0,  0.0)] * N

    def _idx(t_ms):
        return int(round(t_ms / dt_ms))

    for i in range(min(_idx(50),  N), min(_idx(200), N)):
        stim_v1[i] = (45.0, I_STIM_V1)

    for i in range(min(_idx(200), N), min(_idx(350), N)):
        stim_v1[i] = (90.0,   I_STIM_V1)
        stim_a1[i] = (2000.0, I_STIM_A1)

    neuro_events = {}

    def _ev(t_ms, etype, mag=1.0):
        neuro_events.setdefault(_idx(t_ms), []).append(
            NeurochemicalEvent(etype, mag)
        )

    _ev(50.0,  EventType.NOVEL_STIMULUS,    1.0)
    _ev(50.0,  EventType.NOVEL_ENVIRONMENT, 0.7)
    _ev(200.0, EventType.NOVEL_STIMULUS,    0.8)
    _ev(200.0, EventType.STARTLE,           0.5)
    _ev(300.0, EventType.DIRECTED_GAZE,     1.0)
    _ev(320.0, EventType.REWARD,            0.7)

    return stim_v1, stim_a1, neuro_events


# ===========================================================================
# SECTION 13 -- MAIN SIMULATION
# ===========================================================================

def run_cortical_simulation(
    duration_ms: float = DURATION_MS,
    dt_ms:       float = DT_MS,
    pfc_stage:   int   = 0,
) -> dict:
    """
    Simulate all Module 3 cortical regions with neurochemical modulation.

    Returns a dict with all traces.  All existing keys are preserved.
    New keys added for association cortices, S1, PFC, DMN, Thalamus.
    """
    N       = int(duration_ms / dt_ms) + 1
    t_array = np.linspace(0.0, duration_ms, N)

    # --- Instantiate all regions ---
    v1    = V1Population(N_V1)
    a1    = A1Population(N_A1)
    wm    = WhiteMatterTract(N_V1, N_A1)
    neuro = NeurochemicalSystem()

    v2    = V2Population()
    v4    = V4Population()
    mt    = MTPopulation()
    it    = ITPopulation()
    a2    = A2Population()
    stg   = STGPopulation()
    s1    = S1Population()
    pfc   = PFCPopulation(stage=pfc_stage)
    ag    = AngularGyrusPopulation()
    dmn   = DMNMonitor(pfc, ag, dt_ms=dt_ms)
    thal  = ThalamusRelay()

    stim_v1, stim_a1, neuro_sched = _build_stimulus_schedule(duration_ms, dt_ms)

    # --- Trace storage ---
    v1_trace_idx = [5, 10]
    a1_trace_idx = [2, 13]
    V_v1_sample  = {i: np.empty(N) for i in v1_trace_idx}
    V_a1_sample  = {i: np.empty(N) for i in a1_trace_idx}

    v1_pop_fired = np.zeros(N, dtype=int)
    a1_pop_fired = np.zeros(N, dtype=int)

    chem_names  = ['dopamine', 'serotonin', 'norepinephrine', 'acetylcholine', 'cortisol']
    chem_traces = {name: np.empty(N) for name in chem_names}
    modulation  = {'stdp_scale':    np.empty(N),
                   'learning_gate': np.empty(N),
                   'global_gain':   np.empty(N)}

    wm_weight_trace = np.empty(N)
    dmn_corr_trace  = np.zeros(N)

    print("=" * 70)
    print("CAINE - Module 3: Cortical Architecture")
    print("=" * 70)
    print(f"  V1 neurons  : {N_V1}  ({N_PER_COL} per orientation column)")
    print(f"  A1 neurons  : {N_A1}  (tonotopic {A1_F_LOW:.0f}-{A1_F_HIGH:.0f} Hz)")
    print(f"  Assoc.cortex: V2={N_V2} V4={N_V4} MT={N_MT} IT={N_IT} "
          f"A2={N_A2} STG={N_STG}")
    print(f"  S1={N_S1}  PFC={N_PFC} (stage={pfc_stage}, "
          f"myelination={pfc.myelination_factor:.1f})  AG={N_AG}")
    print(f"  WM delay    : {WM_DELAY_MS} ms  |  WM init weight: {WM_INIT_WEIGHT}")
    print(f"  Duration    : {duration_ms} ms  |  dt: {dt_ms} ms  |  steps: {N}")
    print("-" * 70)

    # -----------------------------------------------------------------------
    # Main integration loop
    # -----------------------------------------------------------------------
    for i, t in enumerate(t_array):

        # Neurochemical events
        events_now = neuro_sched.get(i)
        if events_now:
            for ev in events_now:
                print(f"[EVENT] t={t:7.1f} ms  {ev.event_type.name:<28} "
                      f"mag={ev.magnitude:.2f}")
        neuro.update(dt_ms, events=events_now, current_time=t)

        gain_mod = neuro.global_gain()
        stdp_s   = neuro.stdp_scale()
        ach_gate = neuro.learning_gate()

        wm.neuro_stdp_scale = stdp_s
        wm.neuro_gate       = ach_gate

        # --- Visual stimulus (demo: scalar angle) ---
        stim_angle, I_v1_max = stim_v1[i]
        if stim_angle >= 0:
            I_v1 = v1.compute_drive(stim_angle, I_v1_max, gain_mod=gain_mod)
        else:
            I_v1 = np.full(N_V1, I_BASE * gain_mod)

        # --- Auditory stimulus (demo: scalar frequency) ---
        stim_freq, I_a1_max = stim_a1[i]
        I_a1 = a1.compute_drive(stim_freq, I_a1_max, gain_mod=gain_mod)
        a1.decay_onset(dt_ms)

        # --- Thalamus: relay + cross-modal Pulvinar ---
        I_v1_relayed = thal.relay(ThalamusRelay.VISUAL,   I_v1)
        I_a1_relayed = thal.relay(ThalamusRelay.AUDITORY, I_a1)
        I_v1_relayed, I_a1_relayed = thal.pulvinar(I_v1_relayed, I_a1_relayed)

        # --- White matter synaptic current into A1 ---
        I_wm = wm.get_synaptic_currents(a1.V)

        # --- V1 + A1 step ---
        v1.step(dt_ms, I_v1_relayed)
        a1.step(dt_ms, I_a1_relayed, I_syn=I_wm)

        v1_fired = v1.detect_spikes(t)
        a1_fired = a1.detect_spikes(t)

        v1.update_rate_est(v1_fired, dt_ms)
        a1.update_rate_est(a1_fired, dt_ms)

        wm.on_pre_spikes(t, v1_fired)
        wm.deliver_and_stdp_depress(t)
        wm.on_post_spikes(t, a1_fired)
        wm.update(dt_ms)

        # Thalamic cortical feedback
        thal.cortical_feedback(ThalamusRelay.VISUAL,   float(v1.rate_est.mean()))
        thal.cortical_feedback(ThalamusRelay.AUDITORY, float(a1.rate_est.mean()))

        # --- Visual association cortex ---
        I_v2 = v2.compute_drive(v1.rate_est, gain_mod=gain_mod)
        I_v4 = v4.compute_drive(v2_rates=v2.rate_est, gain_mod=gain_mod)
        I_mt = mt.compute_drive(v1.rate_est, gain_mod=gain_mod)
        I_it = it.compute_drive(
            np.concatenate([v2.rate_est, v4.rate_est, mt.rate_est]),
            gain_mod=gain_mod
        )

        v2.step(dt_ms, I_v2);  v2_fired = v2.detect_spikes(t);  v2.update_rate_est(v2_fired, dt_ms)
        v4.step(dt_ms, I_v4);  v4_fired = v4.detect_spikes(t);  v4.update_rate_est(v4_fired, dt_ms)
        mt.step(dt_ms, I_mt);  mt_fired = mt.detect_spikes(t);  mt.update_rate_est(mt_fired, dt_ms)
        it.step(dt_ms, I_it);  it_fired = it.detect_spikes(t);  it.update_rate_est(it_fired, dt_ms)
        it.hebbian_update(it.rate_est,
                          np.concatenate([v2.rate_est, v4.rate_est, mt.rate_est]))

        # --- Auditory association cortex ---
        I_a2  = a2.compute_drive(a1.rate_est, gain_mod=gain_mod)
        I_stg = stg.compute_drive(a2.rate_est, gain_mod=gain_mod)

        a2.step(dt_ms, I_a2);   a2_fired  = a2.detect_spikes(t);  a2.update_rate_est(a2_fired,  dt_ms)
        stg.step(dt_ms, I_stg); stg_fired = stg.detect_spikes(t); stg.update_rate_est(stg_fired, dt_ms)
        stg.stdp_update(stg_fired, t, a2.rate_est, neuro_scale=stdp_s)

        # --- S1 (no avatar joints in demo -- at rest) ---
        I_s1 = s1.compute_drive({}, gain_mod=gain_mod)
        s1.step(dt_ms, I_s1)
        s1_fired = s1.detect_spikes(t)
        s1.update_rate_est(s1_fired, dt_ms)

        # --- PFC ---
        I_pfc = pfc.compute_drive(gain_mod=gain_mod)
        pfc.step(dt_ms, I_pfc)
        pfc_fired = pfc.detect_spikes(t)
        pfc.update_rate_est(pfc_fired, dt_ms)
        pfc.update_wm(pfc_fired, dt_ms)

        # --- Angular Gyrus ---
        I_ag = ag.compute_drive(stg_rates=stg.rate_est,
                                 mpfc_rates=pfc.mPFC_rates,
                                 gain_mod=gain_mod)
        ag.step(dt_ms, I_ag)
        ag_fired = ag.detect_spikes(t)
        ag.update_rate_est(ag_fired, dt_ms)
        ag.update_trace(ag_fired, dt_ms)

        # --- DMN: resting when no strong external stimulus ---
        is_resting = (stim_angle < 0) and (stim_freq <= 0)
        dmn.update(t, is_resting=is_resting)

        # --- Record traces ---
        for ni in v1_trace_idx:
            V_v1_sample[ni][i] = v1.V[ni]
        for ni in a1_trace_idx:
            V_a1_sample[ni][i] = a1.V[ni]

        v1_pop_fired[i] = int(v1_fired.sum())
        a1_pop_fired[i] = int(a1_fired.sum())

        snap = neuro.snapshot()
        for name in chem_names:
            chem_traces[name][i] = snap[name]
        modulation['stdp_scale'][i]    = stdp_s
        modulation['learning_gate'][i] = ach_gate
        modulation['global_gain'][i]   = gain_mod

        wm_weight_trace[i] = wm.weights.mean()

    # --- Summary ---
    v1_total  = sum(len(st) for st in v1.spike_times)
    a1_total  = sum(len(st) for st in a1.spike_times)
    it_total  = sum(len(st) for st in it.spike_times)
    stg_total = sum(len(st) for st in stg.spike_times)
    print("-" * 70)
    print(f"[CAINE] V1 total spikes : {v1_total}")
    print(f"[CAINE] A1 total spikes : {a1_total}")
    print(f"[CAINE] IT total spikes : {it_total}  |  STG: {stg_total}")
    print(f"[CAINE] WM mean weight  : init={WM_INIT_WEIGHT:.4f}  "
          f"final={wm.weights.mean():.4f}  "
          f"delta={wm.weights.mean() - WM_INIT_WEIGHT:+.4f}")
    print(f"[CAINE] Peak NE gain    : {modulation['global_gain'].max():.3f}")
    print(f"[CAINE] DMN emerged     : {dmn.dmn_emerged}  "
          f"(t={dmn.dmn_emerge_t})")
    print("=" * 70)

    return {
        # --- Primary regions (backward-compat keys) ---
        't':               t_array,
        'v1':              v1,
        'a1':              a1,
        'wm':              wm,
        'V_v1_sample':     V_v1_sample,
        'V_a1_sample':     V_a1_sample,
        'v1_pop_fired':    v1_pop_fired,
        'a1_pop_fired':    a1_pop_fired,
        'chem_traces':     chem_traces,
        'modulation':      modulation,
        'wm_weight_trace': wm_weight_trace,
        'stim_v1':         stim_v1,
        'stim_a1':         stim_a1,
        # --- Association cortex ---
        'v2': v2, 'v4': v4, 'mt': mt, 'it': it,
        'a2': a2, 'stg': stg,
        # --- Other regions ---
        's1': s1, 'pfc': pfc, 'ag': ag,
        # --- Infrastructure ---
        'dmn':  dmn,
        'thal': thal,
        'dmn_corr_trace': dmn_corr_trace,
    }


# ===========================================================================
# SECTION 14 -- HELPERS
# ===========================================================================

def _smooth_rate(fired_array: np.ndarray, t_array: np.ndarray,
                 win_ms: float = 10.0, dt_ms: float = DT_MS) -> np.ndarray:
    """Convert per-step spike counts to smoothed population firing rate (Hz)."""
    win_steps = max(1, int(win_ms / dt_ms))
    kernel    = np.ones(win_steps) / (win_ms * 1e-3)
    return np.convolve(fired_array, kernel, mode='same')


def _stimulus_shade(ax, stim_list, t_array, color, alpha=0.08):
    """Shade timesteps where a stimulus is active."""
    active = np.array([
        (s[0] >= 0 if len(s) == 2 and isinstance(s[0], float) else s[0] > 0)
        for s in stim_list
    ], dtype=bool)
    in_block = False
    t0 = 0.0
    for idx, a in enumerate(active):
        if a and not in_block:
            t0       = t_array[idx]
            in_block = True
        elif not a and in_block:
            ax.axvspan(t0, t_array[idx], color=color, alpha=alpha, linewidth=0)
            in_block = False
    if in_block:
        ax.axvspan(t0, t_array[-1], color=color, alpha=alpha, linewidth=0)


# ===========================================================================
# SECTION 15 -- PLOTTING
# ===========================================================================

def plot_cortex(results: dict) -> None:
    """
    Figure 1 -- Activity overview (6 panels):
      V1 raster | A1 raster | V1 voltage traces | A1 voltage traces
      Population firing rate | White matter mean weight

    Figure 2 -- Neurochemical state (4 panels):
      Dopamine | Norepinephrine | Acetylcholine (+ STDP gate overlay) | Cortisol
    """
    t           = results['t']
    v1          = results['v1']
    a1          = results['a1']
    V_v1_sample = results['V_v1_sample']
    V_a1_sample = results['V_a1_sample']
    chem        = results['chem_traces']
    mod         = results['modulation']
    wm_w        = results['wm_weight_trace']
    stim_v1     = results['stim_v1']
    stim_a1     = results['stim_a1']
    dmn         = results.get('dmn')

    col_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44']
    freq_cmap  = plt.cm.plasma

    # -------------------------------------------------------------------
    # Figure 1: Activity
    # -------------------------------------------------------------------
    fig1 = plt.figure(figsize=(15, 13))
    fig1.suptitle(
        "CAINE - Module 3: Cortical Architecture\n"
        f"V1 ({N_V1} neurons, 4 orientation cols)  +  "
        f"A1 ({N_A1} neurons, tonotopic {A1_F_LOW:.0f}-{A1_F_HIGH:.0f} Hz)  "
        f"via {WM_DELAY_MS:.0f}ms white-matter tract",
        fontsize=11, fontweight='bold'
    )

    gs   = gridspec.GridSpec(6, 1, hspace=0.55, top=0.91, bottom=0.06)
    axes = [fig1.add_subplot(gs[r]) for r in range(6)]

    # V1 raster
    ax = axes[0]
    for ni in range(N_V1):
        col = col_colors[v1.column[ni]]
        for st in v1.spike_times[ni]:
            ax.plot(st, ni, '|', color=col, markersize=5, markeredgewidth=1.0)
    _stimulus_shade(ax, stim_v1, t, 'cornflowerblue', alpha=0.10)
    for c_idx, (ang, col) in enumerate(zip(V1_ORIENTATIONS, col_colors)):
        ax.plot([], [], '|', color=col, markersize=8,
                label=f'{ang:.0f} deg  (n {c_idx*N_PER_COL}-{(c_idx+1)*N_PER_COL-1})')
    ax.set_yticks([0, 4, 9, 14, 19])
    ax.set_ylim(-0.5, N_V1 - 0.5)
    ax.set_ylabel('Neuron')
    ax.set_title('V1 Raster  (shading = visual stimulus period)', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.18)

    # A1 raster
    ax   = axes[1]
    norm = mcolors.LogNorm(vmin=A1_F_LOW, vmax=A1_F_HIGH)
    for ni in range(N_A1):
        col = freq_cmap(norm(a1.pref_freq[ni]))
        for st in a1.spike_times[ni]:
            ax.plot(st, ni, '|', color=col, markersize=5, markeredgewidth=1.0)
    _stimulus_shade(ax, stim_a1, t, 'salmon', alpha=0.12)
    ax2r = ax.twinx()
    ax2r.set_ylim(-0.5, N_A1 - 0.5)
    freq_ticks = [0, 5, 10, 15, 19]
    ax2r.set_yticks(freq_ticks)
    ax2r.set_yticklabels([f'{a1.pref_freq[i]:.0f}Hz' for i in freq_ticks], fontsize=7)
    ax2r.set_ylabel('Preferred frequency', fontsize=8)
    ax.set_ylim(-0.5, N_A1 - 0.5)
    ax.set_ylabel('Neuron')
    ax.set_title('A1 Raster  (colour = preferred freq, shading = auditory stimulus)', fontsize=9)
    ax.grid(True, alpha=0.18)

    # V1 voltage traces
    ax       = axes[2]
    lmap_v1  = {5: 'V1 n5 (45 deg col)', 10: 'V1 n10 (90 deg col)'}
    cols_v1  = ['#4477AA', '#228833']
    for ni, col in zip([5, 10], cols_v1):
        ax.plot(t, V_v1_sample[ni], color=col, linewidth=0.6,
                label=lmap_v1[ni], alpha=0.9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylim(-90, 65)
    ax.set_ylabel('V (mV)')
    ax.set_title('Sample V1 voltage traces', fontsize=9)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.18)

    # A1 voltage traces
    ax      = axes[3]
    lmap_a1 = {2:  f'A1 n2 ({a1.pref_freq[2]:.0f} Hz)',
               13: f'A1 n13 ({a1.pref_freq[13]:.0f} Hz, ~2 kHz)'}
    cols_a1 = ['#9933CC', '#CC4400']
    for ni, col in zip([2, 13], cols_a1):
        ax.plot(t, V_a1_sample[ni], color=col, linewidth=0.6,
                label=lmap_a1[ni], alpha=0.9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylim(-90, 65)
    ax.set_ylabel('V (mV)')
    ax.set_title('Sample A1 voltage traces', fontsize=9)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.18)

    # Population firing rate
    ax      = axes[4]
    rate_v1 = _smooth_rate(results['v1_pop_fired'], t, win_ms=10.0)
    rate_a1 = _smooth_rate(results['a1_pop_fired'], t, win_ms=10.0)
    ax.plot(t, rate_v1, color='steelblue', linewidth=1.1, label='V1 population rate')
    ax.plot(t, rate_a1, color='tomato',    linewidth=1.1, label='A1 population rate')
    if dmn is not None and dmn.dmn_emerge_t is not None:
        ax.axvline(dmn.dmn_emerge_t, color='gold', linewidth=1.2,
                   linestyle='--', alpha=0.8, label='DMN emerged')
    ax.set_ylabel('Spikes/s')
    ax.set_title('Population firing rate  (smoothed 10 ms window)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.18)

    # White matter weight
    ax = axes[5]
    ax.plot(t, wm_w, color='darkorchid', linewidth=1.1, label='WM mean weight')
    ax.axhline(WM_INIT_WEIGHT, color='darkorchid', linewidth=0.7,
               linestyle=':', alpha=0.5, label=f'Initial ({WM_INIT_WEIGHT:.3f})')
    ax.set_ylim(max(0, wm_w.min() - 0.005), wm_w.max() + 0.005)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Time (ms)')
    ax.set_title('White matter mean synaptic weight  (STDP-modulated)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.18)

    for ax in axes:
        ax.set_xlim(0, DURATION_MS)

    plt.savefig(os.path.join(_OUTPUT_DIR, 'caine_module3_cortex_activity.png'),
                dpi=150, bbox_inches='tight')
    print('[CAINE] Plot saved -> output/caine_module3_cortex_activity.png')

    # -------------------------------------------------------------------
    # Figure 2: Neurochemicals
    # -------------------------------------------------------------------
    fig2, ax2s = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    fig2.suptitle(
        "CAINE - Module 3: Neurochemical Modulation During Cortical Activity",
        fontsize=11, fontweight='bold'
    )

    neuro_layout = [
        ('dopamine',       'Dopamine (DA)',       'royalblue'),
        ('norepinephrine', 'Norepinephrine (NE)', 'darkorange'),
        ('acetylcholine',  'Acetylcholine (ACh)', 'steelblue'),
        ('cortisol',       'Cortisol (CORT)',     'tomato'),
    ]
    for ax, (name, label, color) in zip(ax2s, neuro_layout):
        bl = chem[name][0]
        ax.plot(t, chem[name], color=color, linewidth=1.0, label=label)
        ax.axhline(bl, color=color, linewidth=0.6, linestyle=':', alpha=0.5,
                   label=f'baseline ({bl:.2f})')
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel('Concentration')
        ax.set_title(label, fontsize=9)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.20)

    ax_twin = ax2s[2].twinx()
    ax_twin.plot(t, mod['learning_gate'], color='navy', linewidth=0.8,
                 linestyle='--', alpha=0.7, label='Learning gate')
    ax_twin.plot(t, mod['stdp_scale'], color='purple', linewidth=0.8,
                 linestyle='-.', alpha=0.7, label='STDP scale')
    ax_twin.set_ylim(-0.05, 3.1)
    ax_twin.set_ylabel('Gate / Scale', fontsize=8)
    ax_twin.legend(loc='upper left', fontsize=7)

    ax2s[-1].set_xlabel('Time (ms)')
    for ax in ax2s:
        ax.set_xlim(0, DURATION_MS)

    plt.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, 'caine_module3_neurochemicals.png'),
                dpi=150, bbox_inches='tight')
    print('[CAINE] Plot saved -> output/caine_module3_neurochemicals.png')

    plt.show()


# ===========================================================================
# SECTION 16 -- ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    results = run_cortical_simulation(
        duration_ms = DURATION_MS,
        dt_ms       = DT_MS,
        pfc_stage   = 0,   # PFC suppressed in Stage 0
    )
    results['neuro_sched_times'] = [50.0, 200.0, 300.0, 320.0]
    plot_cortex(results)
