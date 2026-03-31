"""
CAINE Sensory Layer — Module 5
================================
Bridges raw world signals into the cortical populations built in Modules 1-4.

Sensory streams
---------------
  1. VISION   — 64×64 RGB frame  → DoG (ON/OFF)  → 4 orientation bands → V1
  2. AUDIO    — 20ms PCM frames  → FFT → 128-band Mel → A1 → A2 → STG
  3. PROPRIO  — joint angles     → S1 body-part columns (32 neurons)

New in this module
------------------
  stg_lateral_inhibit()    — winner-take-more competition across STG
  S1Encoder                — body-part columns: hand/arm/head/torso/foot (32 n)
  VocalTract               — Kelly-Lochbaum 44-tube waveguide (Caine geometry)
  RVCLayer                 — RVC voice conversion stub (pass-through)
  ClipLearningQueue        — Caine clip queue: vocal/behavioral/self-recognition

Requires only numpy / scipy (+ optional PyAudio for live mic input).

Usage
-----
    from caine.sensory  import SensoryLayer
    from caine.cortex   import V1Population, A1Population, A2Population, STGPopulation
    from caine.chemicals import NeurochemicalSystem

    v1    = V1Population()
    a1    = A1Population()
    a2    = A2Population()
    stg   = STGPopulation()
    neuro = NeurochemicalSystem()
    sense = SensoryLayer(v1, a1, neuro, a2=a2, stg=stg)

    # tick every ~20 ms
    frame        = env.get_camera_feed()          # (64,64,3) uint8
    joint_angles = np.zeros(6)                    # radians
    sense.update(frame, joint_angles, dt_ms=20.0)

    sense.visualize()                             # 4-panel matplotlib figure
"""

import os
import sys as _sys
# Allow running this file directly (python caine/sensory.py) by ensuring the
# project root is on sys.path so 'caine' package resolves correctly.
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

# -- optional PyAudio -------------------------------------------------------
try:
    import pyaudio
    _PYAUDIO_OK = True
except ImportError:
    _PYAUDIO_OK = False

# -- matplotlib (Agg-safe) --------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -- intra-package imports ---------------------------------------------------
from caine.cortex    import V1Population, A1Population
from caine.chemicals import NeurochemicalSystem

# A2Population and STGPopulation are available if cortex.py has been updated
try:
    from caine.cortex import A2Population, STGPopulation
    _A2_STG_AVAILABLE = True
except ImportError:
    _A2_STG_AVAILABLE = False

_OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE  = 22050    # Hz — matches PyAudio default
FRAME_SAMPLES = 441     # ~20 ms @ 22 050 Hz
FFT_SIZE     = 512
MEL_BANDS    = 128
MEL_F_MIN    = 80.0     # Hz
MEL_F_MAX    = 8000.0   # Hz
N_JOINTS     = 6        # proprioceptive channels
N_S1_NEURONS = 20       # S1 population size
HH_DT_MS     = 0.025    # HH Euler step inside each 20ms sensory frame (matches cortex.py)

# ---------------------------------------------------------------------------
# Mel filterbank (pure numpy)
# ---------------------------------------------------------------------------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def build_mel_filterbank(n_bands: int = MEL_BANDS,
                          f_min: float = MEL_F_MIN,
                          f_max: float = MEL_F_MAX,
                          fft_size: int = FFT_SIZE,
                          sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Construct a (n_bands, fft_size//2+1) triangular Mel filterbank matrix.
    Each row is one band; multiply by the magnitude spectrum to get energies.
    """
    n_fft  = fft_size // 2 + 1
    mel_lo = _hz_to_mel(f_min)
    mel_hi = _hz_to_mel(f_max)

    # n_bands+2 uniformly spaced Mel points → Hz → FFT bin indices
    mel_pts  = np.linspace(mel_lo, mel_hi, n_bands + 2)
    hz_pts   = _mel_to_hz(mel_pts)
    bin_pts  = np.floor((fft_size + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_bands, n_fft), dtype=np.float32)
    for m in range(1, n_bands + 1):
        lo, ctr, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo, ctr):
            if ctr > lo:
                fb[m - 1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi > ctr:
                fb[m - 1, k] = (hi - k) / (hi - ctr)
    return fb


# ---------------------------------------------------------------------------
# DoG filter (Difference of Gaussians — ON-centre / OFF-surround)
# ---------------------------------------------------------------------------

def _dog_filter(frame_gray: np.ndarray,
                sigma_center: float = 1.0,
                sigma_surround: float = 3.0) -> np.ndarray:
    """
    Apply a DoG to a single-channel image.

    Returns a float32 array normalised to [-1, 1] where:
      +1  =  bright centre on dark surround  (ON response)
      -1  =  dark centre on bright surround  (OFF response)
    """
    center   = gaussian_filter(frame_gray.astype(np.float32), sigma_center)
    surround = gaussian_filter(frame_gray.astype(np.float32), sigma_surround)
    dog      = center - surround
    peak     = np.abs(dog).max()
    if peak > 1e-6:
        dog /= peak
    return dog


# ---------------------------------------------------------------------------
# Orientation energy (Gabor-like via oriented DoG sums)
# ---------------------------------------------------------------------------

def _orientation_energy(dog: np.ndarray,
                         angles_deg=(0, 45, 90, 135)) -> np.ndarray:
    """
    Estimate orientation energy at 4 angles using directional derivative pairs.

    Returns array shape (4,) — mean energy per orientation band, normalised.
    These values are used to drive the corresponding V1 orientation columns.
    """
    h, w  = dog.shape
    energies = []
    for ang in angles_deg:
        rad  = np.radians(ang)
        dx   = np.cos(rad)
        dy   = np.sin(rad)

        # Horizontal gradient ≈ difference of shifted columns
        shift_x = int(round(dx))
        shift_y = int(round(dy))

        # Compute directional derivative via shifted difference
        shifted = np.roll(np.roll(dog, shift_x, axis=1), shift_y, axis=0)
        deriv   = dog - shifted
        energies.append(float(np.mean(deriv ** 2)))

    energies = np.array(energies, dtype=np.float32)
    total = energies.sum()
    if total > 1e-8:
        energies /= total
    return energies


# ---------------------------------------------------------------------------
# STG lateral inhibition — winner-take-more phoneme competition
# ---------------------------------------------------------------------------

def stg_lateral_inhibit(stg_rates: np.ndarray,
                         inhibit_strength: float = 0.6,
                         inhibit_radius: int = 3) -> np.ndarray:
    """
    Winner-take-more lateral inhibition across STG phoneme population.

    Neurons that fire more strongly suppress their neighbours, sharpening
    phoneme cluster selectivity.  Implements competitive interneuron dynamics:
        output_i = rate_i − inhibit_strength × sum(rate_j, j ∈ neighbourhood)

    Parameters
    ----------
    stg_rates        : (N_STG,) EMA firing-rate estimate array
    inhibit_strength : inhibitory weight from neighbours [0, 1]
    inhibit_radius   : neighbourhood half-width in neuron index space

    Returns
    -------
    inhibited_rates : (N_STG,) non-negative rates after lateral competition
    """
    n = len(stg_rates)
    rates = stg_rates.astype(np.float32)
    output = rates.copy()
    for i in range(n):
        lo = max(0, i - inhibit_radius)
        hi = min(n, i + inhibit_radius + 1)
        nb_sum = rates[lo:hi].sum() - rates[i]
        output[i] = rates[i] - inhibit_strength * nb_sum
    return np.clip(output, 0.0, None)


# ---------------------------------------------------------------------------
# S1 body-part columns — somatosensory population
# ---------------------------------------------------------------------------

class S1Encoder:
    """
    Somatosensory S1 population with body-part columns.

    Neuron allocation (32 total):
        hand  : 12 neurons  (digits + palm, highest tactile resolution)
        arm   :  6 neurons
        head  :  6 neurons
        torso :  4 neurons
        foot  :  4 neurons

    Each body-part column is assigned a joint channel and covers [−π, π]
    with uniformly spaced circular Gaussian tuning curves.

    Preserves the ``encode(joint_angles)`` and ``efference_copy()`` API.
    """

    BODY_PARTS = {'hand': 12, 'arm': 6, 'head': 6, 'torso': 4, 'foot': 4}

    def __init__(self, n_joints: int = N_JOINTS):
        self.n_joints  = n_joints
        self.n_neurons = sum(self.BODY_PARTS.values())  # 32

        # Build per-neuron preferred joint index and preferred angle
        pref_joint: list = []
        pref_angle: list = []
        body_part_ranges: dict = {}
        idx = 0
        joint_alloc = list(range(n_joints))  # one joint per body part (cycle)

        for bp_i, (bp_name, bp_n) in enumerate(self.BODY_PARTS.items()):
            body_part_ranges[bp_name] = (idx, idx + bp_n)
            j = joint_alloc[bp_i % len(joint_alloc)]
            angles = np.linspace(-np.pi, np.pi, bp_n, endpoint=False)
            for ang in angles:
                pref_joint.append(j)
                pref_angle.append(float(ang))
            idx += bp_n

        self.pref_joint        = np.array(pref_joint, dtype=int)
        self.pref_angle        = np.array(pref_angle, dtype=np.float32)
        self.body_part_ranges  = body_part_ranges
        self._sigma            = 0.5
        self.rate_history: list = []

    def encode(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Convert joint angles (radians) to population firing rates.

        Parameters
        ----------
        joint_angles : (N_JOINTS,) array

        Returns
        -------
        rates : (32,) array, values in [0, 1]
        """
        rates = np.zeros(self.n_neurons, dtype=np.float32)
        for i in range(self.n_neurons):
            j   = self.pref_joint[i]
            ang = float(joint_angles[j]) if j < len(joint_angles) else 0.0
            diff = ang - self.pref_angle[i]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [−π, π]
            rates[i] = np.exp(-0.5 * (diff / self._sigma) ** 2)
        self.rate_history.append(rates.copy())
        return rates

    def efference_copy(self, joint_angles: np.ndarray) -> np.ndarray:
        """Efference copy — identical to encode() for now."""
        return self.encode(joint_angles)


# Backward compatibility: old code using S1Population still works
S1Population = S1Encoder


# ---------------------------------------------------------------------------
# VocalTract — Kelly-Lochbaum 44-tube waveguide
# ---------------------------------------------------------------------------

VOCAL_SR = 44100  # Hz — synthesis sample rate


class VocalTract:
    """
    Kelly-Lochbaum digital waveguide vocal tract model.

    Models Caine's vocal anatomy as 44 cylindrical tube sections from glottis
    (section 0) to lips (section 43).  Cross-sectional areas are initialised
    to Caine's anatomy: extra-wide oral cavity (large mouth).

    Motor cortex varies ``areas`` in real time; the waveguide propagates a
    glottal source pulse through the tube geometry and returns PCM audio.

    K-L scattering at each junction k  (Γ_k = (A_k − A_{k+1})/(A_k + A_{k+1})):
        new_fwd[k+1] = (1 + Γ_k) * p_fwd[k] − Γ_k * p_bwd[k+1]
        new_bwd[k]   =  Γ_k * p_fwd[k] + (1 − Γ_k) * p_bwd[k+1]
    Lip radiation:
        p_bwd[−1] = r_rad * new_fwd[−1]   (r_rad ≈ −0.9)
    Output sample: new_fwd[−1] (forward pressure at lips)
    """

    # Caine vocal tract geometry — 44 sections, cross-sectional area (cm²)
    # fmt: off
    _DEFAULT_AREAS = np.array([
        # 0-9: sub-glottal trachea / lower pharynx (narrow → widening)
        0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62, 0.60,
        # 10-19: upper pharynx
        0.80, 0.88, 0.96, 1.04, 1.10, 1.15, 1.18, 1.20, 1.20, 1.18,
        # 20-33: oral cavity — EXTRA WIDE (Caine's large mouth)
        2.00, 2.30, 2.60, 2.90, 3.10, 3.30, 3.50, 3.65,
        3.80, 3.90, 4.00, 3.95, 3.85, 3.70,
        # 34-41: front oral tapering toward lips
        3.50, 3.20, 2.80, 2.40, 2.00, 1.70, 1.50, 1.30,
        # 42-43: lips
        0.80, 0.40,
    ], dtype=np.float64)
    # fmt: on

    def __init__(self,
                 areas: np.ndarray = None,
                 pitch_hz: float = 120.0,
                 glottal_open: float = 0.8,
                 sr: int = VOCAL_SR):
        self.areas        = areas.copy() if areas is not None else self._DEFAULT_AREAS.copy()
        assert len(self.areas) == 44, "VocalTract requires exactly 44 tube sections"
        self.pitch_hz     = float(pitch_hz)
        self.glottal_open = float(glottal_open)  # 0=closed, 1=fully open
        self.sr           = int(sr)

        # Traveling wave state
        self._p_fwd = np.zeros(44, dtype=np.float64)
        self._p_bwd = np.zeros(44, dtype=np.float64)

        # Glottal oscillator (sample counter for phase)
        self._glottal_phase = 0.0

        # Lip radiation reflection coefficient (negative = phase flip at open end)
        self._r_rad = -0.9

    # ------------------------------------------------------------------
    # Internal: one-sample waveguide step
    # ------------------------------------------------------------------

    def _glottal_sample(self) -> float:
        """Periodic triangle-wave glottal pulse at pitch_hz, scaled by glottal_open."""
        period = self.sr / max(self.pitch_hz, 50.0)
        frac   = (self._glottal_phase % period) / period
        tri    = frac / 0.6 if frac < 0.6 else (1.0 - frac) / 0.4
        self._glottal_phase += 1.0
        return tri * self.glottal_open

    def _scatter(self, g_in: float) -> float:
        """
        Advance waveguide one sample.  Injects glottal source at tube 0.
        Returns output pressure sample at lips (tube 43).
        """
        areas = self.areas
        p_f   = self._p_fwd
        p_b   = self._p_bwd

        # Reflection coefficients at 43 junctions
        gamma = (areas[:-1] - areas[1:]) / (areas[:-1] + areas[1:] + 1e-12)

        new_f = np.empty(44, dtype=np.float64)
        new_b = np.empty(44, dtype=np.float64)

        # Inject glottal source into section 0 (forward wave only)
        new_f[0] = g_in

        # Scattering at each junction (vectorised)
        new_f[1:] = (1.0 + gamma) * p_f[:-1] - gamma * p_b[1:]
        new_b[:-1] =        gamma  * p_f[:-1] + (1.0 - gamma) * p_b[1:]

        # Lip radiation: reflection back into tube 43
        new_b[-1] = self._r_rad * new_f[-1]

        self._p_fwd = new_f
        self._p_bwd = new_b

        return float(new_f[-1])  # output = lip forward pressure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize_frame(self, duration_ms: float) -> np.ndarray:
        """
        Synthesise one frame of PCM audio.

        Parameters
        ----------
        duration_ms : frame duration in milliseconds

        Returns
        -------
        pcm : float32 array of length round(sr * duration_ms / 1000),
              normalised amplitude in [−1, 1]
        """
        n_samples = max(1, round(self.sr * duration_ms / 1000.0))
        out = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            out[i] = self._scatter(self._glottal_sample())

        # Normalise to prevent clipping
        peak = np.abs(out).max()
        if peak > 1e-8:
            out /= peak
        return out.astype(np.float32)

    def reset(self):
        """Zero traveling wave state (use between utterances)."""
        self._p_fwd[:] = 0.0
        self._p_bwd[:] = 0.0
        self._glottal_phase = 0.0


# ---------------------------------------------------------------------------
# RVCLayer — voice conversion stub
# ---------------------------------------------------------------------------

class RVCLayer:
    """
    RVC (Retrieval Voice Conversion) stub.

    Sits between VocalTract output and audio playback.  When an RVC model
    trained on show audio is available it will shift the timbre toward
    Caine's canonical voice while preserving timing and emotional cadence.

    Until then: pass-through (returns PCM unchanged).
    """

    def __init__(self, model_path: str = None):
        self._model_path = model_path
        self._model      = None
        if model_path is not None and os.path.isfile(model_path):
            # Future: load model weights here
            print(f"[RVCLayer] Model file found at {model_path} — loading not yet implemented.")
        else:
            print("[RVCLayer] No RVC model — pass-through mode active.")

    def convert(self, pcm: np.ndarray, sr: int = VOCAL_SR) -> np.ndarray:
        """
        Apply voice conversion to PCM audio.

        Parameters
        ----------
        pcm : float32 PCM array
        sr  : sample rate of the input

        Returns
        -------
        float32 PCM array (same shape) — pass-through until model is loaded
        """
        if self._model is None:
            return pcm.astype(np.float32)
        # Future: return self._model.infer(pcm, sr)
        return pcm.astype(np.float32)

    @property
    def is_active(self) -> bool:
        """True when a real RVC model is loaded and conversion is live."""
        return self._model is not None


# ---------------------------------------------------------------------------
# ClipLearningQueue — Caine video clip learning
# ---------------------------------------------------------------------------

class ClipLearningQueue:
    """
    Queue of Caine video clips used for three learning objectives:

    1. **Vocal modeling**   — STG builds cadence/rhythm representations that
                              motor learning targets for speech production.
    2. **Behavioral modeling** — observing Caine interact with people teaches
                                 social response patterns.
    3. **Self recognition** — if CAINE's self-model matches IT visual
                              representation of Caine-in-clip, that constitutes
                              a mirror-test equivalent and is logged as a major
                              developmental milestone.

    Each queued clip is a dict:
        { 'path': str, 'category': 'vocal'|'behavioral'|'self',
          'loaded': bool, 'processed_frames': int }
    """

    SELF_RECOGNITION_LOG = os.path.join(
        _OUTPUT_DIR, 'self_recognition_log.jsonl')

    def __init__(self):
        self._queue: list  = []
        self._processed    = 0
        self._self_recognized = False

    def enqueue(self, path: str,
                category: str = 'vocal') -> None:
        """Add a clip to the learning queue."""
        assert category in ('vocal', 'behavioral', 'self'), \
            f"Unknown category '{category}'; use vocal/behavioral/self"
        self._queue.append({
            'path':             path,
            'category':         category,
            'loaded':           False,
            'processed_frames': 0,
        })

    def next_frame(self) -> dict:
        """
        Return metadata for the next unprocessed clip frame.
        Returns None when queue is empty.
        """
        if not self._queue:
            return None
        clip = self._queue[0]
        clip['processed_frames'] += 1
        clip['loaded'] = True
        return clip

    def mark_done(self) -> None:
        """Mark the current front clip as fully consumed."""
        if self._queue:
            self._queue.pop(0)
            self._processed += 1

    def check_self_recognition(self,
                                pfc_self_model: np.ndarray,
                                it_visual_repr: np.ndarray,
                                threshold: float = 0.75,
                                t_ms: float = 0.0) -> bool:
        """
        Test whether PFC self-model matches IT visual representation of Caine.

        Uses cosine similarity.  When similarity exceeds ``threshold`` for the
        first time, logs a developmental milestone and returns True.

        Parameters
        ----------
        pfc_self_model  : (N,) rate vector from PFC self-model sub-region
        it_visual_repr  : (N,) rate vector from IT population
        threshold       : cosine similarity threshold [0, 1]
        t_ms            : current simulation time for the log
        """
        na = np.linalg.norm(pfc_self_model)
        nb = np.linalg.norm(it_visual_repr)
        if na < 1e-8 or nb < 1e-8:
            return False

        sim = float(np.dot(pfc_self_model, it_visual_repr) / (na * nb))
        if sim >= threshold and not self._self_recognized:
            self._self_recognized = True
            self._log_milestone(t_ms, sim)
            return True
        return False

    def _log_milestone(self, t_ms: float, similarity: float) -> None:
        import json, time as _time
        entry = {
            'event':      'self_recognition',
            't_ms':       t_ms,
            'similarity': round(similarity, 4),
            'wall_time':  _time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        with open(self.SELF_RECOGNITION_LOG, 'a') as fh:
            fh.write(json.dumps(entry) + '\n')
        print(f"[ClipLearningQueue] MILESTONE — self-recognition at t={t_ms:.1f} ms "
              f"(cosine sim={similarity:.3f})")

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def total_processed(self) -> int:
        return self._processed


# ---------------------------------------------------------------------------
# Audio stream manager
# ---------------------------------------------------------------------------

class AudioStream:
    """
    Manages live microphone input via PyAudio, with fallback to generated tones.

    Call read_frame() each sensory tick to get a FRAME_SAMPLES float32 array.
    """

    def __init__(self, sr: int = SAMPLE_RATE,
                 frame_samples: int = FRAME_SAMPLES):
        self._sr    = sr
        self._n     = frame_samples
        self._phase = 0.0  # for fallback tone generation
        self._stream = None

        if _PYAUDIO_OK:
            try:
                self._pa = pyaudio.PyAudio()
                self._stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=sr,
                    input=True,
                    frames_per_buffer=frame_samples,
                )
                print("[sensory] Live microphone stream opened.")
            except Exception as e:
                print(f"[sensory] Mic unavailable ({e}) — using generated tones.")
                self._stream = None
        else:
            print("[sensory] PyAudio not installed — using generated tones.")

        # Sweep frequency for demo fallback (Hz)
        self._sweep_freq = 440.0

    @property
    def mic_ok(self) -> bool:
        """True when a live microphone stream is open and active."""
        return self._stream is not None

    def read_frame(self) -> np.ndarray:
        """Return one frame of audio as a normalised float32 array."""
        if self._stream is not None:
            try:
                raw = self._stream.read(self._n, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.float32).copy()
                if len(data) == self._n:
                    return data
            except Exception:
                pass

        # Fallback: generate a pure sine tone (sweep 200–4000 Hz)
        t = np.arange(self._n) / self._sr + self._phase
        self._sweep_freq = 200.0 + 1900.0 * (0.5 + 0.5 * np.sin(
            2 * np.pi * 0.2 * (self._phase + self._n / self._sr)))
        wave = np.sin(2 * np.pi * self._sweep_freq * t).astype(np.float32)
        self._phase += self._n / self._sr
        return wave

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
                self._pa.terminate()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# SensoryLayer — main class
# ---------------------------------------------------------------------------

class SensoryLayer:
    """
    The top-level sensory processing pipeline for CAINE.

    Connects raw world signals (camera feed, microphone, proprioception) to
    the cortical populations from Module 4 via biologically-inspired
    preprocessing.
    """

    def __init__(self,
                 v1: V1Population,
                 a1: A1Population,
                 neuro: NeurochemicalSystem,
                 a2=None,
                 stg=None,
                 use_mic: bool = True):
        self.v1    = v1
        self.a1    = a1
        self.neuro = neuro
        self.s1    = S1Encoder()

        # Optional higher auditory cortex populations (A2 + STG)
        self.a2  = a2   # A2Population or None
        self.stg = stg  # STGPopulation or None

        # Mel filterbank (built once)
        self.mel_fb = build_mel_filterbank()

        # Audio
        self.audio = AudioStream() if use_mic else AudioStream.__new__(AudioStream)
        if not use_mic:
            # Minimal stub for generated tones only
            self.audio._sr = SAMPLE_RATE
            self.audio._n  = FRAME_SAMPLES
            self.audio._phase = 0.0
            self.audio._stream = None
            self.audio._sweep_freq = 440.0

        # History buffers for visualisation
        self._frame_history:     list = []   # raw frames
        self._dog_history:       list = []   # DoG outputs
        self._audio_history:     list = []   # raw waveforms
        self._fft_history:       list = []   # FFT magnitudes
        self._mel_history:       list = []   # Mel energies
        self._v1_rate_history:   list = []   # V1 mean firing rates
        self._a1_rate_history:   list = []   # A1 mean firing rates
        self._a2_rate_history:   list = []   # A2 EMA rates (if a2 present)
        self._stg_rate_history:  list = []   # STG lateral-inhibited rates
        self._s1_rate_history:   list = []   # S1 firing rates
        self._neuro_history:     list = []   # neurochemical snapshots

        self._tick = 0  # frame counter

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self,
               frame_rgb: np.ndarray,
               joint_angles: np.ndarray,
               dt_ms: float = 20.0,
               injected_audio: np.ndarray = None) -> dict:
        """
        Process one sensory frame (~20 ms wall-clock equivalent).

        Parameters
        ----------
        frame_rgb      : (H, W, 3) uint8 — from env.get_camera_feed()
        joint_angles   : (6,) float array of joint angles in radians
        dt_ms          : sensory frame duration (ms)
        injected_audio : optional (N,) float32 array to bypass microphone

        Returns
        -------
        dict with keys:
            dog, mel_energy, orient_energy, s1_rates,
            v1_spikes, a1_spikes, neuro_snapshot,
            a2_rates   (None if no A2Population provided),
            stg_rates  (None if no STGPopulation provided; post-lateral-inhibition),
            stg_spikes (None if no STGPopulation provided)
        """
        self._tick += 1

        # ---- 1. Visual processing ----------------------------------------
        gray = self._to_gray(frame_rgb)
        dog  = _dog_filter(gray)
        orient_energy = _orientation_energy(dog)

        # Map orientation energies to V1 drive currents
        # Each of the 4 energy values drives its orientation column in V1
        # Scale: max energy → I_max, modulated by neurochemical gain
        gain_mod = self.neuro.global_gain()

        # Run V1 for dt_ms (multiple HH steps)
        n_steps = max(1, int(dt_ms / HH_DT_MS))
        v1_spikes_accum = np.zeros(self.v1.n, dtype=bool)
        # Pass raw DoG so _orient_to_v1_current can use RMS contrast as stimulus magnitude
        I_v1 = self._orient_to_v1_current(orient_energy, gain_mod, dog=dog)
        for step_i in range(n_steps):
            self.v1.step(HH_DT_MS, I_v1)
            spk = self.v1.detect_spikes(self._tick * dt_ms + step_i * HH_DT_MS)
            v1_spikes_accum |= spk

        # ---- 2. Audio processing -----------------------------------------
        if injected_audio is not None:
            audio_frame = injected_audio[:FRAME_SAMPLES].astype(np.float32)
            if len(audio_frame) < FRAME_SAMPLES:
                audio_frame = np.pad(audio_frame,
                                     (0, FRAME_SAMPLES - len(audio_frame)))
        else:
            audio_frame = self.audio.read_frame()

        # FFT magnitude spectrum
        window     = np.hanning(len(audio_frame))
        fft_mag    = np.abs(np.fft.rfft(audio_frame * window, n=FFT_SIZE))
        mel_energy = self.mel_fb @ fft_mag   # (128,) raw Mel energies

        # Log compression (simulates cochlear dynamic range)
        mel_log = np.log1p(mel_energy)
        mel_max = mel_log.max()
        if mel_max > 1e-8:
            mel_norm = mel_log / mel_max
        else:
            mel_norm = mel_log

        # Map Mel energies → A1 tonotopic drive
        # A1 has 20 neurons tuned to log-spaced frequencies 200–6000 Hz
        a1_drive = self._mel_to_a1_current(mel_norm, gain_mod)

        a1_spikes_accum = np.zeros(self.a1.n, dtype=bool)
        for step_i in range(n_steps):
            self.a1.decay_onset(HH_DT_MS)
            self.a1.step(HH_DT_MS, a1_drive)
            spk = self.a1.detect_spikes(self._tick * dt_ms + step_i * HH_DT_MS)
            a1_spikes_accum |= spk

        # ---- 3. A2 temporal pattern detection (if available) -------------
        a2_rates   = None
        a2_spikes  = None
        if self.a2 is not None:
            # Convert A1 spike accumulator to Hz estimate for this frame
            a1_hz = a1_spikes_accum.astype(np.float32) * (1000.0 / dt_ms)
            I_a2  = self.a2.compute_drive(a1_hz)
            a2_spikes_accum = np.zeros(self.a2.n, dtype=bool)
            for step_i in range(n_steps):
                self.a2.step(HH_DT_MS, I_a2)
                spk = self.a2.detect_spikes(self._tick * dt_ms + step_i * HH_DT_MS)
                a2_spikes_accum |= spk
            self.a2.update_rate_est(a2_spikes_accum, HH_DT_MS)
            a2_rates  = self.a2.rate_est.copy()
            a2_spikes = a2_spikes_accum

        # ---- 4. STG phoneme emergence (if available) ---------------------
        stg_rates_inhibited = None
        stg_spikes          = None
        if self.stg is not None and a2_rates is not None:
            I_stg = self.stg.compute_drive(a2_rates)
            stg_spikes_accum = np.zeros(self.stg.n, dtype=bool)
            for step_i in range(n_steps):
                self.stg.step(HH_DT_MS, I_stg)
                spk = self.stg.detect_spikes(self._tick * dt_ms + step_i * HH_DT_MS)
                stg_spikes_accum |= spk
            # STDP update gated by oxytocin scale
            ot_scale = self.neuro.oxytocin_stg_scale() if hasattr(self.neuro, 'oxytocin_stg_scale') else 1.0
            t_now = self._tick * dt_ms
            self.stg.stdp_update(stg_spikes_accum, t_now, a2_rates, neuro_scale=ot_scale)
            self.stg.update_rate_est(stg_spikes_accum, HH_DT_MS)
            # Winner-take-more lateral inhibition
            stg_rates_inhibited = stg_lateral_inhibit(self.stg.rate_est)
            stg_spikes          = stg_spikes_accum

        # ---- 5. Proprioception -------------------------------------------
        s1_rates = self.s1.encode(joint_angles)

        # ---- 6. Neurochemical update (motor/prediction signals) ----------
        neuro_snapshot = self.neuro.snapshot()

        # ---- 7. Store histories for visualisation ------------------------
        self._frame_history.append(frame_rgb.copy())
        self._dog_history.append(dog.copy())
        self._audio_history.append(audio_frame.copy())
        self._fft_history.append(fft_mag.copy())
        self._mel_history.append(mel_norm.copy())
        self._v1_rate_history.append(v1_spikes_accum.astype(float))
        self._a1_rate_history.append(a1_spikes_accum.astype(float))
        if a2_rates is not None:
            self._a2_rate_history.append(a2_rates.copy())
        if stg_rates_inhibited is not None:
            self._stg_rate_history.append(stg_rates_inhibited.copy())
        self._s1_rate_history.append(s1_rates.copy())
        self._neuro_history.append(neuro_snapshot)

        # Keep rolling window (last 50 frames)
        for buf in (self._frame_history, self._dog_history,
                    self._audio_history, self._fft_history,
                    self._mel_history, self._v1_rate_history,
                    self._a1_rate_history, self._a2_rate_history,
                    self._stg_rate_history, self._s1_rate_history,
                    self._neuro_history):
            if len(buf) > 50:
                buf.pop(0)

        return {
            'dog':            dog,
            'mel_energy':     mel_norm,
            'orient_energy':  orient_energy,
            's1_rates':       s1_rates,
            'v1_spikes':      v1_spikes_accum,
            'a1_spikes':      a1_spikes_accum,
            'a2_rates':       a2_rates,           # None if a2 not provided
            'stg_rates':      stg_rates_inhibited, # None if stg not provided
            'stg_spikes':     stg_spikes,          # None if stg not provided
            'neuro_snapshot': neuro_snapshot,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame_rgb: np.ndarray) -> np.ndarray:
        """Convert (H,W,3) uint8 to float32 luminance [0,1]."""
        if frame_rgb.ndim == 2:
            return frame_rgb.astype(np.float32) / 255.0
        # Rec. 709 luminance
        r, g, b = frame_rgb[:,:,0], frame_rgb[:,:,1], frame_rgb[:,:,2]
        return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32) / 255.0

    def _orient_to_v1_current(self, orient_energy: np.ndarray,
                               gain_mod: float,
                               dog: np.ndarray = None) -> np.ndarray:
        """
        Map 4 orientation energies to per-neuron V1 drive currents.

        BUG FIX: previous version used normalised orient_energy, which lost all
        information about whether a stimulus was actually present.  A blank frame
        and a high-contrast frame both produced the same normalised vector, so V1
        fired identically regardless of visual input.

        Fix: use DoG RMS contrast as a 'stimulus present' scalar (0→1).  The
        normalised orientation vector distributes drive across columns; the RMS
        scalar gates whether any drive is applied at all.

        Blank frame  → dog_rms ≈ 0 → I = I_BASE (3 µA, subthreshold) → no spikes
        Strong edges → dog_rms ≈ 0.15 → stimulus_mag ≈ 1.0 → I up to 11.5 µA → spikes
        """
        from caine.cortex import I_BASE, I_STIM_V1

        # ---- Orientation distribution (normalised to sum=1) ----
        pref_deg = np.array(self.v1.pref_orient, dtype=float) % 180.0
        refs     = np.array([0., 45., 90., 135.])

        orient_per_neuron = np.zeros(self.v1.n, dtype=np.float32)
        for i, ref in enumerate(refs):
            diff   = np.radians(pref_deg - ref)
            tuning = np.cos(diff) ** 2
            orient_per_neuron += orient_energy[i] * tuning
        # orient_per_neuron peaks at ~0.7 for a perfectly tuned neuron; normalise to 1.0
        peak = orient_per_neuron.max()
        if peak > 1e-8:
            orient_per_neuron /= peak

        # ---- Stimulus magnitude from DoG RMS contrast ----
        # DoG RMS of ~0.12 corresponds to a moderate-contrast edge; clamp at 1.0.
        if dog is not None:
            dog_rms = float(np.sqrt(np.mean(dog ** 2)))
            stimulus_mag = min(1.0, dog_rms / 0.12)
        else:
            # Fallback: use raw orient_energy magnitude (pre-fix behaviour)
            stimulus_mag = float(min(1.0, orient_energy.max() * 4.0))

        # ---- Final current ----
        # I_BASE   = 3.0 µA  (constant subthreshold tonic drive)
        # I_STIM_V1= 8.5 µA  (peak stimulus-driven component, from cortex.py)
        # Total range: 3.0 (silent) → 11.5 µA (maximally driven) — matches cortex demo
        I_per_neuron = I_BASE + orient_per_neuron * I_STIM_V1 * stimulus_mag * gain_mod

        # Reset any neurons that diverged to NaN (numerical safety)
        nan_mask = ~np.isfinite(self.v1.V)
        if nan_mask.any():
            from caine.neuron import V_rest, gate_steady_state
            m0, h0, n0 = gate_steady_state(V_rest)
            self.v1.V[nan_mask]       = V_rest
            self.v1.m[nan_mask]       = m0
            self.v1.h[nan_mask]       = h0
            self.v1.n_gate[nan_mask]  = n0
            self.v1._V_prev[nan_mask] = V_rest  # prevent NaN in spike-crossing check

        return I_per_neuron.astype(np.float32)

    def _mel_to_a1_current(self, mel_norm: np.ndarray,
                            gain_mod: float) -> np.ndarray:
        """
        Map 128-band Mel energies to 20-neuron A1 currents.

        A1 neurons are log-spaced 200–6000 Hz; each receives a weighted sum
        of Mel bands whose centre frequency is close to the neuron's preference.
        """
        # A1 preferred frequencies (log-spaced)
        pref_hz  = np.logspace(np.log10(200), np.log10(6000),
                               self.a1.n)
        pref_mel = _hz_to_mel(pref_hz)

        # Mel band centre frequencies
        mel_lo  = _hz_to_mel(MEL_F_MIN)
        mel_hi  = _hz_to_mel(MEL_F_MAX)
        band_mel = np.linspace(mel_lo, mel_hi, MEL_BANDS)

        sigma_mel = (mel_hi - mel_lo) / MEL_BANDS * 3.0  # bandwidth

        I_per_neuron = np.zeros(self.a1.n, dtype=np.float32)
        for i in range(self.a1.n):
            weights = np.exp(-0.5 * ((band_mel - pref_mel[i]) / sigma_mel) ** 2)
            weights /= (weights.sum() + 1e-8)
            I_per_neuron[i] = float(np.dot(weights, mel_norm))

        # Scale: 8 µA/cm² peak — matches safe operating range for HH neurons
        # I_BASE (3 µA/cm²) + Mel-driven component. Same scale budget as V1.
        I_per_neuron = 3.0 + I_per_neuron * 10.0 * gain_mod

        # Reset any A1 neurons that diverged to NaN
        nan_mask = ~np.isfinite(self.a1.V)
        if nan_mask.any():
            from caine.neuron import V_rest, gate_steady_state
            m0, h0, n0 = gate_steady_state(V_rest)
            self.a1.V[nan_mask]       = V_rest
            self.a1.m[nan_mask]       = m0
            self.a1.h[nan_mask]       = h0
            self.a1.n_gate[nan_mask]  = n0
            self.a1._V_prev[nan_mask] = V_rest

        return I_per_neuron.astype(np.float32)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, save_path: str = None) -> str:
        """
        Render a 4-panel matplotlib figure showing the current sensory state.

        Panels
        ------
        1  Top-left  : Raw camera frame + DoG overlay
        2  Top-right : Audio waveform, FFT magnitude, Mel bar chart
        3  Bottom-left  : V1 / A1 / S1 population firing rates (heatmaps)
        4  Bottom-right : Neurochemical state from Module 4

        Returns the path to the saved PNG.
        """
        if not self._frame_history:
            warnings.warn("No data in history — call update() first.")
            return ''

        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # ---- Panel 1: Vision --------------------------------------------
        ax_cam = fig.add_subplot(gs[0, 0])
        self._draw_vision_panel(ax_cam)

        # ---- Panel 2: Audio ---------------------------------------------
        ax_aud = fig.add_subplot(gs[0, 1])
        self._draw_audio_panel(ax_aud)

        # ---- Panel 3: Cortical populations ------------------------------
        ax_cort = fig.add_subplot(gs[1, 0])
        self._draw_cortex_panel(ax_cort)

        # ---- Panel 4: Neurochemicals ------------------------------------
        ax_neuro = fig.add_subplot(gs[1, 1])
        self._draw_neuro_panel(ax_neuro)

        fig.suptitle("CAINE Sensory Layer — Module 5", fontsize=14, y=1.01)

        if save_path is None:
            save_path = os.path.join(_OUTPUT_DIR, 'caine_module5_sensory.png')

        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[sensory] Visualisation saved to {save_path}")
        return save_path

    def _draw_vision_panel(self, ax):
        """Panel 1: most-recent camera frame with DoG overlay."""
        n = len(self._frame_history)
        frame = self._frame_history[-1]
        dog   = self._dog_history[-1]

        # Create side-by-side subpanels within the axes area
        ax.axis('off')
        ax.set_title("Vision: Camera | DoG", fontsize=11, pad=4)
        parent_fig = ax.get_figure()

        # Use inset axes
        bbox = ax.get_position()
        x0, y0, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height

        ax_raw = parent_fig.add_axes([x0,          y0, w * 0.48, h])
        ax_dog = parent_fig.add_axes([x0 + w * 0.52, y0, w * 0.48, h])

        ax_raw.imshow(frame, interpolation='nearest')
        ax_raw.set_title("raw", fontsize=9)
        ax_raw.axis('off')

        ax_dog.imshow(dog, cmap='RdBu_r', vmin=-1, vmax=1,
                      interpolation='nearest')
        ax_dog.set_title("DoG", fontsize=9)
        ax_dog.axis('off')

    def _draw_audio_panel(self, ax):
        """Panel 2: waveform, FFT, Mel energies in three horizontal sub-panels."""
        audio = self._audio_history[-1]
        fft   = self._fft_history[-1]
        mel   = self._mel_history[-1]

        ax.axis('off')
        ax.set_title("Audio: Waveform | FFT | Mel", fontsize=11, pad=4)
        parent_fig = ax.get_figure()
        bbox = ax.get_position()
        x0, y0, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height

        ax_wav = parent_fig.add_axes([x0,           y0, w * 0.30, h])
        ax_fft = parent_fig.add_axes([x0 + w * 0.35, y0, w * 0.30, h])
        ax_mel = parent_fig.add_axes([x0 + w * 0.70, y0, w * 0.30, h])

        t_ms = np.linspace(0, len(audio) / SAMPLE_RATE * 1000, len(audio))
        ax_wav.plot(t_ms, audio, lw=0.7, color='steelblue')
        ax_wav.set_xlim(0, t_ms[-1])
        ax_wav.set_ylim(-1.1, 1.1)
        ax_wav.set_title("waveform", fontsize=9)
        ax_wav.set_xlabel("ms", fontsize=7)
        ax_wav.tick_params(labelsize=7)

        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        ax_fft.plot(freqs / 1000, fft, lw=0.7, color='darkorange')
        ax_fft.set_xlim(0, SAMPLE_RATE / 2 / 1000)
        ax_fft.set_title("FFT mag", fontsize=9)
        ax_fft.set_xlabel("kHz", fontsize=7)
        ax_fft.tick_params(labelsize=7)

        ax_mel.barh(np.arange(MEL_BANDS), mel,
                    height=0.9, color='mediumpurple', linewidth=0)
        ax_mel.set_xlim(0, 1.05)
        ax_mel.set_title("Mel", fontsize=9)
        ax_mel.set_xlabel("norm. energy", fontsize=7)
        ax_mel.invert_yaxis()
        ax_mel.tick_params(labelsize=7)

    def _draw_cortex_panel(self, ax):
        """Panel 3: V1 / A1 / A2 / STG / S1 firing rate heatmaps over time."""
        # Build the list of rows to display (always V1/A1/S1; add A2/STG if present)
        rows = []
        rows.append((self._v1_rate_history,  'V1'))
        rows.append((self._a1_rate_history,  'A1'))
        if self._a2_rate_history:
            rows.append((self._a2_rate_history,  'A2'))
        if self._stg_rate_history:
            rows.append((self._stg_rate_history, 'STG'))
        rows.append((self._s1_rate_history,  'S1'))

        labels = ' / '.join(r[1] for r in rows)
        ax.set_title(f"Cortex: {labels} firing rates", fontsize=11)
        ax.axis('off')
        parent_fig = ax.get_figure()
        bbox = ax.get_position()
        x0, y0, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height

        def _make_heatmap(buf):
            if not buf:
                return np.zeros((1, 1))
            return np.array(buf).T  # (neurons, time)

        n_rows  = len(rows)
        row_h   = 0.95 / n_rows  # fractional height per row (with small gap)
        gap     = 0.02

        sub_axes = []
        for i, (buf, label) in enumerate(rows):
            mat   = _make_heatmap(buf)
            y_pos = y0 + h * (1.0 - (i + 1) * row_h)
            sub_ax = parent_fig.add_axes([x0, y_pos, w, h * (row_h - gap)])
            sub_ax.imshow(mat, aspect='auto', interpolation='nearest',
                          cmap='hot', vmin=0, vmax=1)
            sub_ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=18)
            sub_ax.set_xticks([])
            sub_ax.tick_params(labelsize=7)
            sub_axes.append(sub_ax)

        if sub_axes:
            sub_axes[-1].set_xlabel("time (frames)", fontsize=8)

    def _draw_neuro_panel(self, ax):
        """Panel 4: neurochemical state over time."""
        ax.set_title("Neurochemicals", fontsize=11)

        if not self._neuro_history:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes)
            return

        chem_names = list(self._neuro_history[0].keys())
        times      = np.arange(len(self._neuro_history))

        colors = ['#e74c3c', '#9b59b6', '#3498db',
                  '#e67e22', '#2ecc71', '#1abc9c']

        for i, name in enumerate(chem_names):
            vals = [snap[name] for snap in self._neuro_history]
            ax.plot(times, vals, label=name,
                    color=colors[i % len(colors)], lw=1.5)

        ax.set_xlabel("frame", fontsize=9)
        ax.set_ylabel("level (norm.)", fontsize=9)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.set_xlim(0, max(1, len(times) - 1))
        ax.tick_params(labelsize=8)
        ax.set_ylim(0, 0.4)
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

def run_sensory_demo(n_frames: int = 30, dt_ms: float = 20.0):
    """
    Run the sensory pipeline for n_frames with synthetic inputs:
      - Random noise frame (simulates undefined visual input)
      - Synthetic audio tone sweep (no microphone needed)
      - Sinusoidal joint angles
    """
    from caine.chemicals import NeurochemicalSystem, NeurochemicalEvent, EventType

    # Build cortical populations + neuro system
    v1    = V1Population(n_neurons=20)
    a1    = A1Population(n_neurons=20)
    neuro = NeurochemicalSystem()

    # Wire A2 + STG if available (updated cortex.py)
    a2_pop  = None
    stg_pop = None
    if _A2_STG_AVAILABLE:
        a2_pop  = A2Population()
        stg_pop = STGPopulation()
        print("[sensory] A2 + STG populations wired in.")
    else:
        print("[sensory] A2/STG not available — running V1/A1/S1 only.")

    # Build sensory layer (no mic needed)
    sense = SensoryLayer(v1, a1, neuro, a2=a2_pop, stg=stg_pop, use_mic=False)

    print(f"[sensory] Running demo: {n_frames} frames x {dt_ms} ms each")

    rng = np.random.default_rng(42)

    for f in range(n_frames):
        t_s = f * dt_ms / 1000.0

        # Synthetic 64x64 RGB frame — concentric rings with slight noise
        y, x = np.ogrid[-32:32, -32:32]
        r     = np.sqrt(x**2 + y**2) / 32.0 * 255
        noise = rng.integers(0, 40, (64, 64), dtype=np.uint8)
        frame = np.stack([
            np.clip(r + noise, 0, 255).astype(np.uint8),
            np.clip(255 - r + noise, 0, 255).astype(np.uint8),
            noise,
        ], axis=-1)

        # Synthetic audio: pure sine tone sweeping 200 → 3000 Hz
        freq = 200.0 + 2800.0 * (f / max(1, n_frames - 1))
        t_audio = np.arange(FRAME_SAMPLES) / SAMPLE_RATE
        audio   = np.sin(2 * np.pi * freq * t_audio).astype(np.float32)

        # Sinusoidal joint angles
        joints = np.array([
            0.5 * np.sin(2 * np.pi * 0.5 * t_s + i * np.pi / 3)
            for i in range(N_JOINTS)
        ])

        # Inject a reward event mid-sequence to see neuro response
        events = []
        if f == n_frames // 2:
            from caine.chemicals import NeurochemicalEvent, EventType
            events = [NeurochemicalEvent(EventType.REWARD, 1.0)]

        # Update neurochemicals
        neuro.update(dt_ms, events=events)

        # Update sensory layer
        result = sense.update(frame, joints, dt_ms=dt_ms,
                              injected_audio=audio)

        if f % 10 == 0:
            n_v1  = result['v1_spikes'].sum()
            n_a1  = result['a1_spikes'].sum()
            a2_str  = (f"  A2 rate_max={result['a2_rates'].max():.2f}"
                       if result['a2_rates'] is not None else '')
            stg_str = (f"  STG top={result['stg_rates'].max():.2f}"
                       if result['stg_rates'] is not None else '')
            print(f"  frame {f:3d}: V1={n_v1:2d}  A1={n_a1:2d}"
                  f"{a2_str}{stg_str}"
                  f"  orient={result['orient_energy'].round(3)}")

    # Final visualisation
    path = sense.visualize()
    print(f"[sensory] Demo complete. Figure: {path}")
    return sense


if __name__ == '__main__':
    run_sensory_demo()
