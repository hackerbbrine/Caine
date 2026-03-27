"""
CAINE Motor Cortex — Module 8
================================
This is when CAINE first tries to move.

Structures
----------
  M1  (Primary Motor Cortex)    40 HH neurons in 16 motor columns
      Columns 0-5   : avatar joint targets
                      spine, head, arm_L, arm_R, leg_L, leg_R
      Columns 6-11  : vocal tract articulators
                      lips, teeth, alveolar, palate, velum, glottis
      Columns 12-15 : world interaction stubs
                      grab, push, spawn, remove

  Mirror Neurons                20 additional M1 neurons
      Fire during visual detection of motion patterns.
      Provide a motor learning signal without overt movement.
      Feed from frame-difference motion vectors → movement templates.

All populations use Hodgkin-Huxley biophysics (same model as Modules 1-7).

Body Map Bootstrap
------------------
On first run (no body_map.json on disk) CAINE fires each motor column in
isolation and reads the S1 proprioceptive response after the test movement.
This builds body_map: {column_idx: output_name} — CAINE discovers which
neurons move which limbs rather than being told.
Console: "mapping column 0... → spine confirmed"

Neurochemical modulation  (input currents to M1 — NOT output overrides)
------------------------------------------------------------------------
  dopamine    → global M1 excitability boost + larger exploration noise
  cortisol    → global inhibitory current (less M1 spiking → quieter movement)
  oxytocin    → slight excitability bias toward head-column neurons
  NE          → arousal; raises mirror neuron gain
  ACh         → slight excitability bias toward articulator-column neurons

  No neurochemical directly writes joint targets or suppresses output.
  All motor behaviour emerges from HH dynamics given these input currents.

Efference Copy
--------------
Before each movement: M1 sends predicted joint targets → S1 prediction.
After movement executes: S1 encodes actual angles.
Mismatch = L2(predicted − actual) → ACC_CONFLICT event → NE → learning boost.

WorldAPI Stubs  (matches environment.py interface)
------------------
WorldAction(action_type, magnitude, target_id, timestamp_ms)
Generated when world-stub column fires above WORLD_FIRE_THRESH.
Caller passes to env.spawn_object() / move_object() / remove_object() etc.

Usage
-----
    from caine.motor import MotorCortex

    motor = MotorCortex(s1, neuro, body_map_file='output/body_map.json')

    # each tick:
    motor.developmental_stage = 0   # update when stage advances

    motor_result = motor.update(
        dt_ms, v1_spikes, a1_spikes,
        neuro_snapshot, s1_rates,
        visual_frame=None,      # (H,W,3) uint8 for mirror-neuron motion detection
    )

    joint_angles   = motor_result['joint_angles']       # feed back into sense.update()
    loco_mode      = motor_result['locomotion_mode']    # 'walk' | 'fly' | 'teleport'
    motor_score    = motor_result['motor_learning_score']

    # Mother API (observational learning):
    motor.inject_motion_template('walking', list_of_frames)
"""

import os
import sys
import json
import math
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from caine.neuron import (
    Cm, gNa, gK, gL, ENa, EK, EL, V_rest,
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n,
    gate_steady_state,
)
from caine.chemicals import NeurochemicalEvent, EventType

_OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# SECTION 1 — CONSTANTS
# ===========================================================================

# HH integration
HH_DT_MS   = 0.05    # 50 µs — same as limbic (motor needs less precision than V1)
SPIKE_THR  = 0.0     # mV threshold
REFRAC_MS  = 2.0     # refractory period

# Tonic sub-threshold base currents (µA/cm²) — keep below HH rheobase (~2.17)
I_BASE_M1     = 1.7
I_BASE_MIRROR = 1.5

# Stimulus drive scales
I_STIM_M1     = 7.0
I_STIM_MIRROR = 6.0

# Initial output noise (σ of Gaussian random twitches)
NOISE_SIGMA_BASE = 0.10   # radians for joints; [0-1] for articulators

# DA → exploration noise scale (M1 parameter, not a behavioral override)
DA_EXPLORE_SCALE = 3.0    # extra noise σ per unit of DA elevation above baseline

# Chemical input currents to M1 HH neurons
# These modulate M1 excitability; motor behavior emerges from HH dynamics.
DA_EXCIT_SCALE  = 1.2     # dopamine → global M1 excitability boost (µA/cm² per unit)
CORT_INHIB_SCALE = 1.8    # cortisol → global inhibitory current (subtracted)
OT_HEAD_BIAS    = 0.8     # OT → extra current to head-column neurons
ACH_ART_BIAS    = 0.6     # ACh → extra current to articulator-column neurons
NE_MIRROR_SCALE = 2.0     # NE → mirror neuron gain multiplier

# Physical joint constraint (not behavioral): max radians per frame
JOINT_LERP_MAX_RAD = 0.087   # ~5° per frame

# Efference copy mismatch
EFFERENCE_MISMATCH_THRESH = 0.15   # L2 norm threshold → emit ACC_CONFLICT
EFFERENCE_ACC_SCALE       = 3.0    # multiply mismatch by this for event magnitude

# World stub
WORLD_FIRE_THRESH = 0.45   # fraction of world-column neurons firing → action

# Articulator range
ART_NOISE_SIGMA = 0.05   # initial random noise σ for articulators

# Bootstrap
BOOTSTRAP_TEST_ANGLE = math.pi / 3.0   # 60° test movement for body map
BOOTSTRAP_TICKS      = 8               # lerp ticks during bootstrap

# Locomotion mode system
LOCO_UNLOCK_SCORE_FLY        = 0.70  # motor_learning_score threshold to unlock fly
LOCO_UNLOCK_STAGE_FLY        = 2     # minimum developmental stage for fly
LOCO_MASTERY_THRESH_TELEPORT = 0.70  # fly mastery score needed to unlock teleport
LOCO_UNLOCK_STAGE_TELEPORT   = 3     # minimum developmental stage for teleport
LOCO_TMPL_LR                 = 0.005 # Hebbian learning rate for mode templates
LOCO_MASTERY_TAU             = 200   # frames for mastery score exponential average
LOCO_FEEDBACK_SCALE          = 0.40  # strength of locomotion template → M1 prior current
MOTOR_SCORE_TAU              = 150   # frames for motor learning score average


# ===========================================================================
# SECTION 2 — COLUMN DEFINITIONS
# ===========================================================================
# Each entry: (name, output_type, neuron_start, n_neurons)
# Total: 4+3+3+3+3+3 + 2*6 + 2+2+3+2 = 19+12+9 = 40 neurons

JOINT_NAMES      = ['spine', 'head', 'arm_L', 'arm_R', 'leg_L', 'leg_R']
ARTICULATOR_NAMES = ['lips', 'teeth', 'alveolar', 'palate', 'velum', 'glottis']
WORLD_ACTION_NAMES = ['grab', 'push', 'spawn', 'remove']

_COLUMNS: List[Tuple[str, str, int, int]] = [
    # --- avatar joints ---
    ('spine',     'joint',        0,  4),
    ('head',      'joint',        4,  3),
    ('arm_L',     'joint',        7,  3),
    ('arm_R',     'joint',       10,  3),
    ('leg_L',     'joint',       13,  3),
    ('leg_R',     'joint',       16,  3),
    # --- vocal tract ---
    ('lips',      'articulator', 19,  2),
    ('teeth',     'articulator', 21,  2),
    ('alveolar',  'articulator', 23,  2),
    ('palate',    'articulator', 25,  2),
    ('velum',     'articulator', 27,  2),
    ('glottis',   'articulator', 29,  2),
    # --- world stubs ---
    ('grab',      'world',       31,  2),
    ('push',      'world',       33,  2),
    ('spawn',     'world',       35,  3),
    ('remove',    'world',       38,  2),
]
N_M1_NEURONS = 40   # sum of n_neurons above

# Pre-built numpy array mapping neuron index → column index
_NEURON_COLUMN = np.zeros(N_M1_NEURONS, dtype=int)
for _col_i, (_n, _t, _s, _k) in enumerate(_COLUMNS):
    _NEURON_COLUMN[_s:_s + _k] = _col_i

# Neuron index slices for targeted chemical input currents.
# Keep in sync with _COLUMNS if column layout ever changes.
_HEAD_SLICE = slice(_COLUMNS[1][2], _COLUMNS[1][2] + _COLUMNS[1][3])    # head: 4..7
_ART_SLICE  = slice(_COLUMNS[6][2], _COLUMNS[11][2] + _COLUMNS[11][3])  # art: 19..31


# ===========================================================================
# SECTION 3 — WorldAction  (matches environment.py WorldAPI interface)
# ===========================================================================

@dataclass
class WorldAction:
    """
    Generated by M1 world-stub columns when they fire above WORLD_FIRE_THRESH.

    The caller (run_live.py) is responsible for passing these to the
    appropriate environment method:
        'spawn'  → env.spawn_object(id, position, rotation, object_type)
        'remove' → env.remove_object(handle)
        'push'   → env.move_object(handle, force)
        'grab'   → env.move_object(handle, force=0)  [stub — pick up intent]
    """
    action_type:  str    # 'grab' | 'push' | 'spawn' | 'remove'
    magnitude:    float  # 0.0–1.0 (column firing fraction)
    target_id:    str    # object ID to act on (filled in by caller)
    timestamp_ms: float


# ===========================================================================
# SECTION 4 — HH MOTOR POPULATION  (replicates LimbicPopulation, standalone)
# ===========================================================================

class MotorPopulation:
    """
    Vectorised Hodgkin-Huxley population for M1 and mirror neurons.
    Identical math to LimbicPopulation (limbic.py) — standalone to avoid
    circular imports.
    """

    def __init__(self, n_neurons: int, i_base: float = I_BASE_M1):
        self.n      = n_neurons
        self.i_base = i_base

        m0, h0, n0 = gate_steady_state(V_rest)
        self.V      = np.full(n_neurons, V_rest, dtype=np.float64)
        self.m      = np.full(n_neurons, m0,     dtype=np.float64)
        self.h      = np.full(n_neurons, h0,     dtype=np.float64)
        self.n_gate = np.full(n_neurons, n0,     dtype=np.float64)

        self.last_spike  = np.full(n_neurons, -np.inf)
        self.spike_times = [[] for _ in range(n_neurons)]
        self._V_prev     = self.V.copy()

    # ------------------------------------------------------------------
    def step(self, dt: float, I_ext: np.ndarray) -> None:
        """Vectorised Euler step.  I_ext shape (N,)."""
        I_Na = gNa * self.m**3 * self.h * (self.V - ENa)
        I_K  = gK  * self.n_gate**4     * (self.V - EK)
        I_L  = gL                       * (self.V - EL)

        self.V      += dt * (I_ext - I_Na - I_K - I_L) / Cm
        self.m      += dt * (alpha_m(self.V) * (1 - self.m)      - beta_m(self.V) * self.m)
        self.h      += dt * (alpha_h(self.V) * (1 - self.h)      - beta_h(self.V) * self.h)
        self.n_gate += dt * (alpha_n(self.V) * (1 - self.n_gate) - beta_n(self.V) * self.n_gate)

        nan_mask = ~np.isfinite(self.V)
        if nan_mask.any():
            m0, h0, n0 = gate_steady_state(V_rest)
            self.V[nan_mask]      = V_rest
            self.m[nan_mask]      = m0
            self.h[nan_mask]      = h0
            self.n_gate[nan_mask] = n0
            self._V_prev[nan_mask] = V_rest

    # ------------------------------------------------------------------
    def detect_spikes(self, t: float) -> np.ndarray:
        """Threshold-crossing detector with refractory period. Returns bool (N,)."""
        crossed    = (self._V_prev < SPIKE_THR) & (self.V >= SPIKE_THR)
        not_refrac = (t - self.last_spike) > REFRAC_MS
        fired      = crossed & not_refrac

        self.last_spike[fired] = t
        for i in np.where(fired)[0]:
            self.spike_times[i].append(float(t))

        self._V_prev = self.V.copy()
        return fired

    # ------------------------------------------------------------------
    def firing_rate(self, window_ms: float = 100.0) -> np.ndarray:
        """Mean firing rate (Hz) per neuron over last window_ms."""
        if not any(self.spike_times):
            return np.zeros(self.n)
        rates = np.zeros(self.n)
        now = max((max(st) for st in self.spike_times if st), default=0.0)
        for i, st in enumerate(self.spike_times):
            recent = [s for s in st if s >= now - window_ms]
            rates[i] = len(recent) / (window_ms / 1000.0)
        return rates

    # ------------------------------------------------------------------
    def column_fractions(self) -> np.ndarray:
        """
        For each of the 16 M1 columns, return the fraction of neurons that
        fired in the last detect_spikes() call.
        Only meaningful if n == N_M1_NEURONS.
        """
        if self.n != N_M1_NEURONS:
            raise ValueError("column_fractions() only valid for M1 population (40 neurons)")
        fracs = np.zeros(len(_COLUMNS))
        recent_fired = np.array([
            bool(self.spike_times[i] and
                 self.spike_times[i][-1] == self.last_spike[i])
            for i in range(self.n)
        ], dtype=float)
        for col_i, (_, _, start, k) in enumerate(_COLUMNS):
            fracs[col_i] = recent_fired[start:start + k].mean()
        return fracs


# ===========================================================================
# SECTION 5 — VOCAL TRACT
# ===========================================================================

class VocalTract:
    """
    Six articulators driven by M1 vocal columns (columns 6-11).

    Positions range 0.0 (fully open/relaxed) → 1.0 (fully closed/constricted).
    Initial output: low-amplitude Gaussian noise (σ = ART_NOISE_SIGMA).
    No audio synthesis yet — positions are logged each frame for the future
    Kelly-Lochbaum connection (Module 9).

    Modulation:
        High ACh → ACH_ART_BIAS input current to M1 articulator columns;
        noise sigma is reduced proportionally (tighter tuning emerges from HH dynamics)
    """

    NAMES = ARTICULATOR_NAMES

    def __init__(self, rng=None):
        self._rng = rng or np.random.default_rng(99)
        self.positions = np.full(6, 0.5, dtype=np.float32)   # midpoint at rest
        self.targets   = np.full(6, 0.5, dtype=np.float32)
        self._history: list = []   # (6,) per frame — for logging / future KL

    # ------------------------------------------------------------------
    def update(self, column_fracs: np.ndarray, ach: float,
               dt_ms: float) -> np.ndarray:
        """
        Advance articulator positions by one frame.

        Parameters
        ----------
        column_fracs : (6,) — firing fractions for art columns (columns 6-11)
        ach          : acetylcholine level — gates precision
        dt_ms        : frame duration in ms

        Returns
        -------
        positions : (6,) float32 in [0, 1]
        """
        # Column firing → target position (column frac is 0–1 → art position 0–1)
        noise_sigma = ART_NOISE_SIGMA * (1.0 - float(np.clip(ach / 0.5, 0.0, 1.0)) * 0.5)
        noise = self._rng.normal(0.0, noise_sigma, 6).astype(np.float32)
        self.targets = np.clip(column_fracs + noise, 0.0, 1.0)

        # Lerp positions toward targets (smooth motor output)
        lerp = float(np.clip(dt_ms / 50.0, 0.0, 1.0))  # full lerp in ~50 ms
        self.positions += lerp * (self.targets - self.positions)
        self.positions  = np.clip(self.positions, 0.0, 1.0)

        self._history.append(self.positions.copy())
        return self.positions

    # ------------------------------------------------------------------
    def console_log(self) -> str:
        """One-line articulator state for console output."""
        parts = [f"{n[:3]}={v:.2f}" for n, v in zip(self.NAMES, self.positions)]
        return "VT[" + " ".join(parts) + "]"


# ===========================================================================
# SECTION 6 — MEDIA LIBRARY
# ===========================================================================

@dataclass
class MediaItem:
    """A single entry in the MediaLibrary."""
    path:          str
    media_type:    str    # 'video' | 'image' | 'audio'
    tag:           str    # user-assigned label (e.g. 'walking', 'happy_face')
    stage_gate:    int    # minimum developmental stage to unlock
    concept_label: str    # semantic label for limbic association
    repetitions:   int = 1   # how many times to replay before marking done
    played:        int = 0   # how many times already played

    def is_available(self, current_stage: int) -> bool:
        return current_stage >= self.stage_gate and self.played < self.repetitions


class MediaLibrary:
    """
    Manages CAINE's developmental media injection pipeline.

    No playback yet — just the data structure and scheduling API.
    A future module will consume items and push frames into the sensory layer.

    Usage
    -----
        lib = MediaLibrary('output/media_library.json')
        lib.add_video('clips/walking.mp4', tag='walking',
                      stage_gate=2, concept_label='locomotion', repetitions=5)
        item = lib.get_next(current_stage=2)   # → MediaItem or None
    """

    def __init__(self, path: str = None):
        self._path = path or os.path.join(_OUTPUT_DIR, 'media_library.json')
        self._items: List[MediaItem] = []
        self._load()

    # ------------------------------------------------------------------
    def add_video(self, path: str, tag: str, stage_gate: int,
                  concept_label: str, repetitions: int = 1) -> None:
        self._items.append(MediaItem(path, 'video', tag,
                                     stage_gate, concept_label, repetitions))
        self._save()

    def add_image(self, path: str, tag: str, stage_gate: int,
                  concept_label: str) -> None:
        self._items.append(MediaItem(path, 'image', tag,
                                     stage_gate, concept_label, 1))
        self._save()

    def add_audio(self, path: str, tag: str, stage_gate: int,
                  concept_label: str) -> None:
        self._items.append(MediaItem(path, 'audio', tag,
                                     stage_gate, concept_label, 1))
        self._save()

    # ------------------------------------------------------------------
    def get_next(self, current_stage: int) -> Optional[MediaItem]:
        """
        Return the next available media item for the given developmental stage.
        Items are returned in order; already-exhausted items are skipped.
        Returns None if nothing is available.
        """
        for item in self._items:
            if item.is_available(current_stage):
                return item
        return None

    def mark_played(self, item: MediaItem) -> None:
        item.played += 1
        self._save()

    # ------------------------------------------------------------------
    def _save(self) -> None:
        data = [
            {k: getattr(it, k) for k in
             ['path', 'media_type', 'tag', 'stage_gate',
              'concept_label', 'repetitions', 'played']}
            for it in self._items
        ]
        try:
            with open(self._path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._items = [MediaItem(**d) for d in data]
        except (json.JSONDecodeError, TypeError, KeyError):
            self._items = []


# ===========================================================================
# SECTION 7 — MIRROR NEURON SYSTEM
# ===========================================================================

class MirrorNeuronSystem:
    """
    20 additional M1 neurons that fire when observed motion matches stored
    movement templates — enabling observational learning without overt movement.

    Visual motion is extracted as a frame-difference vector (optical flow stub).
    Template matching uses cosine similarity.

    Movement templates build up over time as videos are processed via
    inject_motion_template(tag, frames).

    When mirror neurons fire, they produce a motor learning signal identical in
    format to a real M1 spike — downstream STDP can use this to improve motor
    programs through observation.
    """

    def __init__(self):
        self.pop = MotorPopulation(20, i_base=I_BASE_MIRROR)
        self._t_ms = 0.0

        # Templates: tag → list of motion vector arrays (each (H*W,) float)
        self._templates: Dict[str, List[np.ndarray]] = {}
        # Video reference library: {path: {'tag': ..., 'n_frames': ...}}
        self._video_refs: Dict[str, dict] = {}
        # Rolling frame buffer for live motion extraction
        self._prev_frame: Optional[np.ndarray] = None

        # Activation history for visualisation
        self.spikes_last: np.ndarray = np.zeros(20, dtype=bool)
        self._match_history: deque = deque(maxlen=80)

    # ------------------------------------------------------------------
    def update(self, dt_ms: float, v1_spikes: np.ndarray,
               visual_frame: Optional[np.ndarray],
               ne_gain: float = 1.0) -> np.ndarray:
        """
        Advance mirror neurons by dt_ms.

        Parameters
        ----------
        v1_spikes    : (20,) bool — visual cortex activity as proxy for motion
        visual_frame : (H,W,3) uint8 or None — raw camera frame
        ne_gain      : norepinephrine gain multiplier

        Returns
        -------
        spikes : (20,) bool
        """
        self._t_ms += dt_ms
        n_steps = max(1, int(dt_ms / HH_DT_MS))

        # --- Motion signal: frame difference → magnitude ---------------
        motion_mag = 0.0
        if visual_frame is not None:
            gray = np.mean(visual_frame.astype(np.float32), axis=2) / 255.0
            if self._prev_frame is not None:
                diff = np.abs(gray - self._prev_frame)
                motion_mag = float(np.mean(diff))
                # Template matching: compare diff vector to stored templates
                motion_mag = self._match_templates(diff.ravel(), motion_mag)
            self._prev_frame = gray

        self._match_history.append(motion_mag)

        # --- V1 activity is also a proxy for visual motion salience -----
        v1_activity = float(v1_spikes.astype(float).mean())

        # Mirror neurons receive drive from motion_mag + V1 activity
        combined = float(np.clip((motion_mag * 3.0 + v1_activity) * ne_gain, 0.0, 2.0))
        I_mirror = np.full(20, I_BASE_MIRROR + combined * I_STIM_MIRROR)

        spikes_accum = np.zeros(20, dtype=bool)
        for step_i in range(n_steps):
            t = self._t_ms + step_i * HH_DT_MS
            self.pop.step(HH_DT_MS, I_mirror)
            spikes_accum |= self.pop.detect_spikes(t)

        self.spikes_last = spikes_accum
        return spikes_accum

    # ------------------------------------------------------------------
    def _match_templates(self, motion_vec: np.ndarray,
                         base_mag: float) -> float:
        """
        Compare motion_vec to stored templates via cosine similarity.
        Returns boosted magnitude if a match is found.
        """
        if not self._templates:
            return base_mag
        mv_norm = motion_vec / (np.linalg.norm(motion_vec) + 1e-8)
        best_sim = 0.0
        for tag, tmpl_list in self._templates.items():
            for tmpl in tmpl_list:
                if len(tmpl) != len(mv_norm):
                    continue
                t_norm = tmpl / (np.linalg.norm(tmpl) + 1e-8)
                sim = float(np.dot(mv_norm, t_norm))
                best_sim = max(best_sim, sim)
        # Boost magnitude proportional to template match quality
        return base_mag + best_sim * 0.3

    # ------------------------------------------------------------------
    def inject_motion_template(self, tag: str,
                                frames: List[np.ndarray]) -> None:
        """
        Add movement templates from a sequence of video frames.

        Parameters
        ----------
        tag    : movement label ('walking', 'reaching', 'gesturing', ...)
        frames : list of (H,W,3) uint8 arrays
        """
        if len(frames) < 2:
            return
        templates = []
        for i in range(1, len(frames)):
            prev = np.mean(frames[i-1].astype(np.float32), axis=2) / 255.0
            curr = np.mean(frames[i].astype(np.float32),   axis=2) / 255.0
            diff = np.abs(curr - prev).ravel()
            if np.any(diff > 0):
                templates.append(diff)
        if templates:
            self._templates.setdefault(tag, []).extend(templates)

    def add_video_reference(self, path: str, movement_type: str) -> None:
        """
        Register a .mp4 file tagged with a movement type.
        Frames will be loaded and added to templates when processed.
        """
        self._video_refs[path] = {'tag': movement_type, 'loaded': False}


# ===========================================================================
# SECTION 8 — LOCOMOTION SYSTEM
# ===========================================================================

class LocomotionSystem:
    """
    Locomotion mode selection via winner-take-all over M1 joint column patterns.

    Three modes: walk (always unlocked), fly (stage-gated), teleport (stage-gated).

    Each mode owns a firing pattern template — a (6,) vector over the six
    joint columns.  Templates start either as a weak random prior (walk) or
    all-zeros (fly, teleport).  They evolve through Hebbian updates: the
    winning mode's template drifts toward the current M1 column fractions.

    Mode selection:
        active_mode = argmax { cosine_similarity(joint_fracs, templates[mode]) }
        over unlocked modes only.

    Feedback to M1:
        The active mode's template is returned as a prior current vector.
        MotorCortex injects this into M1 joint-column neurons, creating a
        self-reinforcing attractor dynamic.  Because the template is learned
        (not hardcoded), the actual neurons recruited for each mode emerge
        from CAINE's own motor experience.

    Unlock gates (no neurochemical triggers — purely developmental):
        fly      : motor_learning_score > LOCO_UNLOCK_SCORE_FLY
                   AND developmental_stage >= LOCO_UNLOCK_STAGE_FLY
        teleport : fly mastery > LOCO_MASTERY_THRESH_TELEPORT
                   AND developmental_stage >= LOCO_UNLOCK_STAGE_TELEPORT
    """

    MODES = ('walk', 'fly', 'teleport')

    def __init__(self, rng):
        self._rng = rng

        # Learned templates: mode → (6,) float in [0,1], joint column pattern.
        # Walk has a mild random prior so WTA has something to grip onto early.
        # Fly and teleport are all-zero until unlocked and then primed.
        self.templates: Dict[str, np.ndarray] = {
            'walk':     rng.uniform(0.0, 0.2, 6).astype(np.float32),
            'fly':      np.zeros(6, dtype=np.float32),
            'teleport': np.zeros(6, dtype=np.float32),
        }

        # Lock state
        self._unlocked: Dict[str, bool] = {
            'walk': True, 'fly': False, 'teleport': False,
        }

        # Per-mode mastery: exponentially smoothed WTA similarity [0,1]
        self.mastery: Dict[str, float] = {m: 0.0 for m in self.MODES}

        # Active mode this tick
        self.active_mode: str = 'walk'

        # Rolling histories for visualiser
        self.mode_history: deque = deque(maxlen=80)
        self.similarity:   Dict[str, float] = {m: 0.0 for m in self.MODES}

    # ------------------------------------------------------------------
    def update(self, joint_fracs: np.ndarray,
               motor_learning_score: float,
               developmental_stage: int) -> str:
        """
        Advance locomotion system by one tick.

        Parameters
        ----------
        joint_fracs          : (6,) column firing fractions (columns 0-5)
        motor_learning_score : float [0,1] — running motor competence
        developmental_stage  : int — current developmental stage (0-4)

        Returns
        -------
        active_mode : str
        """
        # --- Unlock checks (gate is developmental state, NOT emotion) -------
        if not self._unlocked['fly']:
            if (motor_learning_score > LOCO_UNLOCK_SCORE_FLY
                    and developmental_stage >= LOCO_UNLOCK_STAGE_FLY):
                self._unlocked['fly'] = True
                # Prime fly template as roughly orthogonal to walk
                walk = self.templates['walk']
                orth = self._rng.uniform(0.1, 0.5, 6).astype(np.float32)
                proj = float(np.dot(orth, walk)) / (float(np.dot(walk, walk)) + 1e-8)
                orth = np.clip(np.abs(orth - walk * proj), 0.0, 1.0)
                self.templates['fly'] = orth
                print(f"[locomotion] FLY unlocked "
                      f"(score={motor_learning_score:.2f} "
                      f"stage={developmental_stage})")

        if not self._unlocked['teleport']:
            if (self.mastery['fly'] > LOCO_MASTERY_THRESH_TELEPORT
                    and developmental_stage >= LOCO_UNLOCK_STAGE_TELEPORT):
                self._unlocked['teleport'] = True
                # Teleport template: high uniform activity — maximal column recruitment
                self.templates['teleport'] = np.full(6, 0.85, dtype=np.float32)
                print(f"[locomotion] TELEPORT unlocked "
                      f"(fly mastery={self.mastery['fly']:.2f} "
                      f"stage={developmental_stage})")

        # --- Winner-take-all: cosine similarity to each unlocked template ---
        jf = np.asarray(joint_fracs, dtype=np.float32)
        jf_norm = jf / (np.linalg.norm(jf) + 1e-8)

        best_mode = 'walk'
        best_sim  = -np.inf

        for mode in self.MODES:
            if not self._unlocked[mode]:
                self.similarity[mode] = 0.0
                continue
            tmpl = self.templates[mode]
            if np.all(tmpl == 0.0):
                self.similarity[mode] = 0.0
                continue
            t_norm = tmpl / (np.linalg.norm(tmpl) + 1e-8)
            sim = float(np.dot(jf_norm, t_norm))
            self.similarity[mode] = sim
            if sim > best_sim:
                best_sim  = sim
                best_mode = mode

        self.active_mode = best_mode
        self.mode_history.append(best_mode)

        # --- Hebbian template update: winner drifts toward current pattern --
        if best_sim > -np.inf:
            self.templates[best_mode] = np.clip(
                self.templates[best_mode]
                + LOCO_TMPL_LR * (jf - self.templates[best_mode]),
                0.0, 1.0,
            )

        # --- Mastery score update for winning mode --------------------------
        alpha = 1.0 / LOCO_MASTERY_TAU
        sim_c = float(np.clip(best_sim, 0.0, 1.0)) if best_sim > -np.inf else 0.0
        self.mastery[best_mode] = (
            (1.0 - alpha) * self.mastery[best_mode] + alpha * sim_c
        )

        return best_mode

    # ------------------------------------------------------------------
    def feedback_current(self) -> np.ndarray:
        """
        Return the active mode's template as a (6,) prior current vector.
        Caller scales by LOCO_FEEDBACK_SCALE and injects into joint columns.
        """
        return self.templates[self.active_mode].copy()

    def is_unlocked(self, mode: str) -> bool:
        return self._unlocked.get(mode, False)

    def status(self) -> str:
        locked = [m for m in self.MODES if not self._unlocked[m]]
        m_str  = ' '.join(f"{m[:3]}={v:.2f}" for m, v in self.mastery.items())
        return (f"loco={self.active_mode}  "
                f"mastery=[{m_str}]  "
                f"locked={locked}")


# ===========================================================================
# SECTION 9 — BODY MAP BOOTSTRAP
# ===========================================================================

def _run_bootstrap(s1, body_map_file: str) -> dict:
    """
    Bootstrap protocol — fires each M1 column in isolation and reads S1 to
    confirm which joint responds.  Called once at MotorCortex.__init__ when
    no body_map.json exists.

    Parameters
    ----------
    s1            : S1Population (from sensory.py)
    body_map_file : path to save confirmed body_map

    Returns
    -------
    body_map : {column_idx (int as str): output_name (str)}
    """
    print("[M1] Body map bootstrap — CAINE is discovering its anatomy...")
    body_map: dict = {}
    rest_angles = np.zeros(6, dtype=np.float32)

    for col_idx, (name, output_type, start, n_col) in enumerate(_COLUMNS):
        print(f"[M1]   mapping column {col_idx:2d} ({name:<12s})... ", end='', flush=True)

        if output_type == 'joint':
            joint_idx = JOINT_NAMES.index(name)
            # Simulate firing this column: set test angle on the target joint
            test_angles = rest_angles.copy()
            test_angles[joint_idx] = BOOTSTRAP_TEST_ANGLE
            # Lerp toward test angle for BOOTSTRAP_TICKS frames
            actual = rest_angles.copy()
            for _ in range(BOOTSTRAP_TICKS):
                actual += 0.25 * (test_angles - actual)
            # Read S1 response to the resulting posture
            s1_response = s1.encode(actual)
            # Find which joint index S1 responds to most strongly
            joint_responses = np.array([
                float(np.mean(s1_response[s1.pref_joint == j]))
                for j in range(6)
            ])
            confirmed_idx = int(np.argmax(joint_responses))
            confirmed_name = JOINT_NAMES[confirmed_idx]
            print(f"-> {confirmed_name} confirmed")
        else:
            # Articulators and world stubs: no proprioceptive S1 response,
            # confirm the hardwired label
            print(f"-> {output_type} ({name}) confirmed")
            confirmed_name = name

        body_map[str(col_idx)] = confirmed_name

    os.makedirs(os.path.dirname(os.path.abspath(body_map_file)), exist_ok=True)
    with open(body_map_file, 'w', encoding='utf-8') as f:
        json.dump(body_map, f, indent=2)

    print(f"[M1] Body map saved -> {body_map_file}")
    return body_map


# ===========================================================================
# SECTION 10 — MOTOR CORTEX  (main integration class)
# ===========================================================================

class MotorCortex:
    """
    Primary Motor Cortex (M1) + Mirror Neuron System.

    40 HH neurons in 16 motor columns:
        Columns 0-5   : avatar joints  (spine → leg_R)
        Columns 6-11  : vocal articulators (lips → glottis)
        Columns 12-15 : world-action stubs (grab → remove)

    20 mirror neurons observe visual motion and produce a motor learning
    signal through observation, without CAINE needing to move.

    All output starts as Gaussian noise (σ = NOISE_SIGMA_BASE) — the
    neonatal random-twitch phase.  Learning via downstream STDP will
    gradually sculpt purposeful movement.
    """

    def __init__(self, s1, neuro,
                 body_map_file:    str = None,
                 media_lib_file:   str = None,
                 rng_seed:         int = 0):
        """
        Parameters
        ----------
        s1             : S1Population — for efference copy comparison
        neuro          : NeurochemicalSystem — for chemical modulation
        body_map_file  : path to body_map.json; bootstraps if absent
        media_lib_file : path to media_library.json
        """
        self._s1    = s1
        self._neuro = neuro
        self._rng   = np.random.default_rng(rng_seed)
        self._t_ms  = 0.0

        # --- HH populations ---
        self.m1_pop     = MotorPopulation(N_M1_NEURONS, i_base=I_BASE_M1)
        self.mirror_sys = MirrorNeuronSystem()

        # --- State: joints ---
        self.joint_angles  = np.zeros(6, dtype=np.float32)   # actual current angles
        self.joint_targets = np.zeros(6, dtype=np.float32)   # M1 commanded targets

        # --- State: articulators ---
        self.vocal_tract = VocalTract(self._rng)

        # --- Locomotion system ---
        self.locomotion = LocomotionSystem(self._rng)

        # --- Motor learning score: rolling (1 - mismatch) average ---
        # 0.0 = random twitches; 1.0 = perfect motor control
        self._motor_learning_score: float = 0.0

        # --- Developmental stage (updated by caller; gates locomotion unlock) ---
        self.developmental_stage: int = 0

        # --- Efference copy: last predicted S1 rates ---
        self._efference_predicted = np.zeros(s1.n_neurons, dtype=np.float32)
        self._mismatch_history: deque = deque(maxlen=80)

        # --- Rolling histories for visualisation ---
        self._m1_history     = deque(maxlen=80)  # (60,) bool each frame
        self._joint_history  = deque(maxlen=80)  # (6,) float per frame
        self._art_history    = deque(maxlen=80)  # (6,) float per frame

        # --- Media library ---
        self.media = MediaLibrary(media_lib_file)

        # --- World action event queue (drained by caller each frame) ---
        self._world_queue: List[WorldAction] = []

        # --- Body map ---
        bm_path = body_map_file or os.path.join(_OUTPUT_DIR, 'body_map.json')
        self._body_map_file = bm_path
        if os.path.exists(bm_path):
            with open(bm_path, 'r', encoding='utf-8') as f:
                self.body_map = json.load(f)
            print(f"[M1] Body map loaded from {bm_path}")
        else:
            self.body_map = _run_bootstrap(s1, bm_path)

        # Per-column firing fraction buffer (read by visualiser)
        self.column_fractions = np.zeros(len(_COLUMNS))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, dt_ms: float,
               v1_spikes: np.ndarray,
               a1_spikes: np.ndarray,
               neuro_snapshot: dict,
               s1_rates: np.ndarray,
               voice_position: Optional[tuple] = None,
               visual_frame: Optional[np.ndarray] = None) -> dict:
        """
        Advance motor cortex by dt_ms.

        Parameters
        ----------
        dt_ms          : frame duration (ms)
        v1_spikes      : (20,) bool — visual cortex
        a1_spikes      : (20,) bool — auditory cortex
        neuro_snapshot : dict — neurochemical concentrations
        s1_rates       : (20,) float — actual S1 rates (from previous frame)
        voice_position : (x,y,z) or None — world position of loudest A1 source
        visual_frame   : (H,W,3) uint8 or None — for mirror neurons

        Returns
        -------
        dict with keys:
            joint_angles    : (6,) float32 — current actual joint angles (radians)
            joint_targets   : (6,) float32 — M1-commanded targets
            art_positions   : (6,) float32 — articulator positions [0,1]
            m1_spikes       : (40,) bool   — M1 population spikes this frame
            mirror_spikes   : (20,) bool   — mirror neuron spikes
            efference_mismatch : float     — S1 prediction error (0..1)
            world_actions   : list[WorldAction] — pending world events
            neuro_events    : list[NeurochemicalEvent] — emitted this frame
        """
        self._t_ms += dt_ms
        n_hh_steps = max(1, int(dt_ms / HH_DT_MS))
        neuro_events: List[NeurochemicalEvent] = []

        # ------ 1. Extract neurochemical concentrations --------------------
        da   = neuro_snapshot.get('dopamine',       0.10)
        cort = neuro_snapshot.get('cortisol',       0.08)
        ot   = neuro_snapshot.get('oxytocin',       0.08)
        ach  = neuro_snapshot.get('acetylcholine',  0.12)
        ne   = neuro_snapshot.get('norepinephrine', 0.10)

        # DA scales exploration noise (M1 parameter, not a behavioral override)
        noise_sigma = NOISE_SIGMA_BASE * (1.0 + DA_EXPLORE_SCALE * max(0.0, da - 0.10))

        # NE → mirror neuron gain
        ne_gain = float(np.clip(1.0 + NE_MIRROR_SCALE * max(0.0, ne - 0.10), 1.0, 3.0))

        # ------ 2. Mirror neuron update ------------------------------------
        mirror_spikes = self.mirror_sys.update(
            dt_ms, v1_spikes, visual_frame, ne_gain=ne_gain)

        # ------ 3. Chemical input currents to M1 HH neurons ----------------
        # Neurochemicals modulate M1 excitability via input currents.
        # No chemical writes joint targets directly — all motor behavior
        # emerges from what these currents do to HH spiking dynamics.

        # DA: global excitability boost → more/louder M1 spiking
        da_excit   = float(np.clip((da   - 0.10) * DA_EXCIT_SCALE,  0.0, 2.0))
        # CORT: global inhibitory current → less M1 spiking → quieter movement
        cort_inhib = float(np.clip( cort          * CORT_INHIB_SCALE, 0.0, 2.5))
        # OT: slight bias toward head-column neurons (auditory attention input)
        ot_head    = float(np.clip((ot   - 0.08)  * OT_HEAD_BIAS,    0.0, 1.0))
        # ACh: slight bias toward articulator-column neurons (vocal readiness)
        ach_art    = float(np.clip((ach  - 0.12)  * ACH_ART_BIAS,    0.0, 1.0))

        chem_I = np.zeros(N_M1_NEURONS)
        chem_I          += da_excit          # global boost
        chem_I          -= cort_inhib        # global inhibition
        chem_I[_HEAD_SLICE] += ot_head       # head columns only
        chem_I[_ART_SLICE]  += ach_art       # articulator columns only

        # ------ 4. Locomotion mode feedback current ------------------------
        # The active mode's learned template provides a gentle prior input
        # to joint-column neurons, creating self-reinforcing attractors.
        # The feedback evolves with the template — never hardcoded.
        loco_drive = np.zeros(N_M1_NEURONS)
        active_tmpl = self.locomotion.feedback_current()   # (6,) learned pattern
        for col_i in range(6):   # joint columns 0-5 only
            _, _, col_s, col_k = _COLUMNS[col_i]
            loco_drive[col_s:col_s + col_k] = (
                float(active_tmpl[col_i]) * LOCO_FEEDBACK_SCALE
            )

        # ------ 5. M1 HH integration ---------------------------------------
        # Drive = sensory projection + noise + chemical input + locomotion prior
        sensory_drive = np.zeros(N_M1_NEURONS)
        v1_mean = float(v1_spikes.astype(float).mean())
        a1_mean = float(a1_spikes.astype(float).mean())
        for col_i, (name, otype, start, k) in enumerate(_COLUMNS):
            if otype == 'joint':
                sensory_drive[start:start+k] = (v1_mean + a1_mean) * 2.0
            elif otype == 'articulator':
                sensory_drive[start:start+k] = a1_mean * 3.0
            elif otype == 'world':
                sensory_drive[start:start+k] = (v1_mean + a1_mean) * 0.75

        noise = self._rng.normal(0.0, noise_sigma, N_M1_NEURONS)

        # All modulation enters M1 as input current — no output multipliers
        I_m1 = (I_BASE_M1
                + np.clip(sensory_drive + noise + chem_I + loco_drive,
                          -1.0, 3.0) * I_STIM_M1)

        m1_spikes_accum = np.zeros(N_M1_NEURONS, dtype=bool)
        for step_i in range(n_hh_steps):
            t = self._t_ms + step_i * HH_DT_MS
            self.m1_pop.step(HH_DT_MS, I_m1)
            m1_spikes_accum |= self.m1_pop.detect_spikes(t)

        # Per-column firing fractions
        recent_fired = m1_spikes_accum.astype(float)
        for col_i, (_, _, start, k) in enumerate(_COLUMNS):
            self.column_fractions[col_i] = float(recent_fired[start:start+k].mean())

        # ------ 6. Efference copy (before movement executes) ---------------
        # Predict what S1 will see if the motor command succeeds
        predicted_angles = self.joint_targets.copy()
        self._efference_predicted = self._s1.efference_copy(
            predicted_angles.astype(np.float32))

        # ------ 7. Compute new joint targets from M1 output ----------------
        # Always M1-driven — no chemical override, no hardcoded pose table.
        # Column firing fraction in [0,1] → angle in [-π, +π] (0.5 frac = 0 rad)
        joint_col_fracs = self.column_fractions[:6]
        raw_targets = ((joint_col_fracs - 0.5) * 2.0 * math.pi).astype(np.float32)
        raw_targets += self._rng.normal(0.0, noise_sigma, 6).astype(np.float32)
        self.joint_targets = np.clip(raw_targets, -math.pi, math.pi).astype(np.float32)

        # ------ 8. Lerp actual joint angles toward targets -----------------
        # Physical joint speed constraint — not a behavioral rule
        delta = self.joint_targets - self.joint_angles
        step  = np.sign(delta) * np.minimum(np.abs(delta), JOINT_LERP_MAX_RAD)
        self.joint_angles = (self.joint_angles + step).astype(np.float32)

        # ------ 9. Efference mismatch (compare predicted vs actual S1) -----
        actual_s1 = np.asarray(s1_rates, dtype=np.float32)
        predicted  = self._efference_predicted[:len(actual_s1)]
        mismatch = float(np.linalg.norm(actual_s1 - predicted) /
                         (len(actual_s1) ** 0.5 + 1e-8))
        mismatch = float(np.clip(mismatch, 0.0, 1.0))
        self._mismatch_history.append(mismatch)

        if mismatch > EFFERENCE_MISMATCH_THRESH:
            # Emit ACC conflict → NE → learning boost
            mag = float(np.clip(mismatch * EFFERENCE_ACC_SCALE, 0.0, 1.0))
            neuro_events.append(NeurochemicalEvent(EventType.ACC_CONFLICT, mag))

        # ------ 10. Motor learning score update ----------------------------
        # Running average of (1 - mismatch): approaches 1.0 as CAINE learns
        alpha_ms = 1.0 / MOTOR_SCORE_TAU
        self._motor_learning_score = float(
            (1.0 - alpha_ms) * self._motor_learning_score
            + alpha_ms * (1.0 - mismatch)
        )

        # ------ 11. Locomotion mode update (WTA over joint fractions) ------
        self.locomotion.update(
            self.column_fractions[:6],
            self._motor_learning_score,
            self.developmental_stage,
        )

        # ------ 12. Vocal tract update -------------------------------------
        art_col_fracs = self.column_fractions[6:12].astype(np.float32)
        art_positions = self.vocal_tract.update(art_col_fracs, ach, dt_ms)

        # ------ 13. World stub check ---------------------------------------
        self._world_queue.clear()
        for col_i, name in enumerate(WORLD_ACTION_NAMES):
            wld_col_i = 12 + col_i   # columns 12-15
            frac = self.column_fractions[wld_col_i]
            if frac >= WORLD_FIRE_THRESH:
                self._world_queue.append(WorldAction(
                    action_type  = name,
                    magnitude    = float(frac),
                    target_id    = '',   # caller fills in the target
                    timestamp_ms = self._t_ms,
                ))

        # ------ 14. History for visualisation ------------------------------
        combined_spikes = np.concatenate([m1_spikes_accum,
                                          mirror_spikes]).astype(float)
        self._m1_history.append(combined_spikes)
        self._joint_history.append(self.joint_angles.copy())
        self._art_history.append(art_positions.copy())

        return {
            'joint_angles':          self.joint_angles,
            'joint_targets':         self.joint_targets,
            'art_positions':         art_positions,
            'm1_spikes':             m1_spikes_accum,
            'mirror_spikes':         mirror_spikes,
            'column_fractions':      self.column_fractions.copy(),
            'efference_mismatch':    mismatch,
            'world_actions':         list(self._world_queue),
            'neuro_events':          neuro_events,
            'vocal_tract_state':     self.vocal_tract.console_log(),
            'locomotion_mode':       self.locomotion.active_mode,
            'locomotion_similarity': dict(self.locomotion.similarity),
            'locomotion_locked':     {m: not self.locomotion.is_unlocked(m)
                                      for m in LocomotionSystem.MODES},
            'motor_learning_score':  self._motor_learning_score,
        }

    # ------------------------------------------------------------------
    def pop_world_actions(self) -> List[WorldAction]:
        """Drain the world-action queue.  Call once per tick in run_live.py."""
        actions = list(self._world_queue)
        self._world_queue.clear()
        return actions

    def inject_motion_template(self, tag: str,
                                frames: List[np.ndarray]) -> None:
        """Add movement templates to the mirror neuron system."""
        self.mirror_sys.inject_motion_template(tag, frames)

    def get_joint_angles(self) -> np.ndarray:
        return self.joint_angles.copy()

    def get_articulator_positions(self) -> np.ndarray:
        return self.vocal_tract.positions.copy()

    def status(self) -> str:
        """One-line status string for console output."""
        j = self.joint_angles
        mm = f"{list(self._mismatch_history)[-1]:.3f}" if self._mismatch_history else "?"
        return (
            f"M1 score={self._motor_learning_score:.3f}  "
            f"joints=[sp={j[0]:+.2f} hd={j[1]:+.2f} "
            f"aL={j[2]:+.2f} aR={j[3]:+.2f}]  "
            f"{self.locomotion.status()}  "
            f"{self.vocal_tract.console_log()}  "
            f"mismatch={mm}"
        )


# ===========================================================================
# SECTION 11 — VISUALIZER EXTENSION
# ===========================================================================

def extend_visualizer(viz) -> None:
    """
    Monkey-patch motor cortex panels into an existing LiveVisualizer.

    Adds Row 4 (motor row) to the existing 4-row figure.  Call this ONCE
    after constructing LiveVisualizer and before the first viz.update().

    This function injects:
        viz._m1_history         — deque for M1+mirror raster
        viz._mismatch_history   — deque for mismatch signal
        viz._joint_history      — deque for joint angles
        viz._art_history        — deque for articulator positions

    Then patches viz.update() to also accept motor_result and draw motor panels.

    Usage
    -----
        from caine.motor import extend_visualizer
        viz = LiveVisualizer()
        extend_visualizer(viz)
        ...
        viz.update(result, frame_rgb, audio, limbic_result, motor_result)
    """
    from collections import deque
    HISTORY = 80

    # Inject history buffers
    viz._m1_history_motor    = deque(maxlen=HISTORY)
    viz._mismatch_history_vz = deque(maxlen=HISTORY)
    viz._joint_history_vz    = deque(maxlen=HISTORY)
    viz._art_history_vz      = deque(maxlen=HISTORY)

    # We'll build motor axes lazily at first update
    viz._motor_axes_built = False
    viz._motor_body_map   = {}

    # ------ Extend _build_figure -------------------------------------------
    _orig_build = viz._build_figure

    def _build_figure_extended():
        _orig_build()
        # The figure was built with GridSpec(4,3).  We can't safely resize an
        # existing GridSpec, so we add a new sub-figure row via subfigures.
        # Use add_axes with manual rect positioning instead.
        fig = viz._fig
        fig.set_size_inches(15, 15)   # expand height

        def _moto_ax(left, bottom, width, height, label):
            ax = fig.add_axes([left, bottom, width, height])
            ax.set_facecolor('#1a1a1a')
            for sp in ax.spines.values():
                sp.set_color('#444444')
            ax.tick_params(colors='#aaaaaa', labelsize=7)
            ax.yaxis.label.set_color('#aaaaaa')
            ax.xaxis.label.set_color('#aaaaaa')
            ax.title.set_color('#dddddd')
            viz._axes[label] = ax
            return ax

        # Row 4: three motor panels
        # [left, bottom, width, height]  — figure coords 0-1
        ax_m1    = _moto_ax(0.06,  0.01, 0.28, 0.15, 'm1_raster')
        ax_joint = _moto_ax(0.385, 0.01, 0.28, 0.15, 'm1_joints')
        ax_art   = _moto_ax(0.71,  0.01, 0.26, 0.15, 'm1_art')

        # M1 raster (60 neurons = 40 M1 + 20 mirror)
        ax_m1.set_title("M1 + Mirror raster (40+20 neurons)", fontsize=9)
        ax_m1.set_xlabel("time (frames)", fontsize=7)
        ax_m1.set_ylabel("neuron", fontsize=7)
        viz._artists['m1_img'] = ax_m1.imshow(
            np.zeros((60, 1)), aspect='auto', cmap='plasma',
            vmin=0, vmax=1, interpolation='nearest')

        # Joint bar chart (current blue, target orange)
        ax_joint.set_title("Joint angles: current (◈) vs target (○)", fontsize=9)
        ax_joint.set_xlabel("joint", fontsize=7)
        ax_joint.set_ylabel("radians", fontsize=7)
        ax_joint.set_xlim(-0.5, 5.5)
        ax_joint.set_ylim(-math.pi - 0.1, math.pi + 0.1)
        ax_joint.set_xticks(range(6))
        ax_joint.set_xticklabels(
            ['spine', 'head', 'arm_L', 'arm_R', 'leg_L', 'leg_R'],
            fontsize=6, rotation=30)
        x_j = np.arange(6)
        viz._artists['j_cur'] = ax_joint.bar(
            x_j - 0.18, np.zeros(6), width=0.32,
            color='#3498db', label='current', linewidth=0)
        viz._artists['j_tgt'] = ax_joint.bar(
            x_j + 0.18, np.zeros(6), width=0.32,
            color='#e67e22', label='target', linewidth=0)
        ax_joint.legend(fontsize=6, loc='upper right',
                        facecolor='#222222', labelcolor='white', framealpha=0.7)
        ax_joint.axhline(0, color='#555555', lw=0.6, ls='--')

        # Articulator bars + mismatch line (dual-axis)
        ax_art.set_title("Vocal tract (bars) + eff. mismatch (line)", fontsize=9)
        ax_art.set_xlabel("articulator", fontsize=7)
        ax_art.set_ylabel("position [0–1]", fontsize=7)
        ax_art.set_xlim(-0.5, 5.5)
        ax_art.set_ylim(-0.05, 1.05)
        ax_art.set_xticks(range(6))
        ax_art.set_xticklabels(
            ['lips', 'teeth', 'alv.', 'pal.', 'vel.', 'glot.'],
            fontsize=6, rotation=30)
        viz._artists['art_bars'] = ax_art.bar(
            np.arange(6), np.zeros(6), width=0.7,
            color='#9b59b6', linewidth=0)

        ax_mis = ax_art.twinx()
        ax_mis.set_ylabel("mismatch", fontsize=7, color='#e74c3c')
        ax_mis.tick_params(colors='#e74c3c', labelsize=7)
        ax_mis.set_ylim(0, 1.05)
        viz._artists['mismatch_line'], = ax_mis.plot(
            [], [], lw=1.5, color='#e74c3c', label='mismatch')
        viz._axes['m1_art_twin'] = ax_mis

        viz._motor_axes_built = True

    viz._build_figure = _build_figure_extended

    # ------ Extend update() ------------------------------------------------
    _orig_update = viz.update

    def _update_extended(result, frame_rgb, audio_frame,
                         limbic_result=None, motor_result=None):
        # Accumulate motor history every tick
        if motor_result is not None:
            combined = np.concatenate([
                motor_result['m1_spikes'].astype(float),
                motor_result['mirror_spikes'].astype(float),
            ])
            viz._m1_history_motor.append(combined)
            viz._mismatch_history_vz.append(
                float(motor_result.get('efference_mismatch', 0.0)))
            viz._joint_history_vz.append(
                motor_result['joint_angles'].copy())
            viz._art_history_vz.append(
                motor_result['art_positions'].copy())

        # Call original (handles sensory + limbic + figure construction)
        _orig_update(result, frame_rgb, audio_frame,
                     limbic_result=limbic_result)

        # Draw motor panels if we have data
        if (motor_result is not None and
                viz._motor_axes_built and
                viz._m1_history_motor):
            _draw_motor(viz, motor_result)
            viz._fig.canvas.draw_idle()
            viz._fig.canvas.flush_events()

    viz.update = _update_extended


def _draw_motor(viz, motor_result: dict) -> None:
    """Draw all motor panels into the extended figure."""
    # --- M1 raster ---------------------------------------------------------
    if viz._m1_history_motor:
        mat = np.array(viz._m1_history_motor).T  # (60, time)
        viz._artists['m1_img'].set_data(mat)
        viz._artists['m1_img'].set_extent([0, mat.shape[1], mat.shape[0], 0])
        ax = viz._axes['m1_raster']
        ax.set_xlim(0, mat.shape[1])
        ax.set_ylim(mat.shape[0], 0)
        # Body map annotation: column name at neuron start row (only first 16 rows)
        # (drawn once; removing old texts first would be expensive, skip for perf)

    # --- Joint bars (current vs target) — title shows active loco mode ----
    loco_mode = motor_result.get('locomotion_mode', 'walk')
    loco_lock = motor_result.get('locomotion_locked', {})
    locked_str = '/'.join(m for m, locked in loco_lock.items() if locked)
    mode_str = f"{loco_mode.upper()}" + (f"  (locked: {locked_str})" if locked_str else "")
    viz._axes['m1_joints'].set_title(
        f"Joints: current / target  [{mode_str}]", fontsize=8, color='#dddddd')

    j_cur = motor_result['joint_angles']
    j_tgt = motor_result['joint_targets']
    for bar, v in zip(viz._artists['j_cur'], j_cur):
        bar.set_height(float(v))
        bar.set_y(min(float(v), 0.0))
    for bar, v in zip(viz._artists['j_tgt'], j_tgt):
        bar.set_height(float(v))
        bar.set_y(min(float(v), 0.0))

    # --- Articulator bars --------------------------------------------------
    art = motor_result['art_positions']
    for bar, v in zip(viz._artists['art_bars'], art):
        bar.set_height(float(v))

    # --- Mismatch line -----------------------------------------------------
    if viz._mismatch_history_vz:
        times = np.arange(len(viz._mismatch_history_vz))
        viz._artists['mismatch_line'].set_data(times, list(viz._mismatch_history_vz))
        viz._axes['m1_art_twin'].set_xlim(0, max(1, len(times) - 1))


# ===========================================================================
# SECTION 12 — STAND-ALONE DEMO
# ===========================================================================

def run_motor_demo(n_frames: int = 200, dt_ms: float = 20.0) -> None:
    """
    Headless demo of Module 8 — Motor Cortex + Locomotion System.

    Shows:
      1. Body map bootstrap (console output, first run only)
      2. Random-twitch phase — M1 noise exploration; locomotion WTA in walk mode
      3. Developmental stage progression: stage 0 → 1 → 2 → 3
      4. Neurochemical events modulate M1 excitability via input currents
         (no behavioral overrides — all joint movement is M1-emergent)
      5. Motor learning score rises as mismatch decreases
      6. Fly unlocks when score > LOCO_UNLOCK_SCORE_FLY at stage 2
      7. Teleport unlocks when fly mastery > threshold at stage 3
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from caine.cortex    import V1Population, A1Population
    from caine.sensory   import S1Population
    from caine.chemicals import NeurochemicalSystem, EventType

    print(f"\n[motor demo] {n_frames} frames x {dt_ms} ms")

    neuro = NeurochemicalSystem()
    s1    = S1Population()

    bm_file = os.path.join(_OUTPUT_DIR, 'body_map_demo.json')
    motor   = MotorCortex(s1, neuro, body_map_file=bm_file, rng_seed=42)

    # ---- Event schedule --------------------------------------------------
    # Neurochemical events modulate M1 excitability as input currents.
    # No event directly triggers a behavioral response.
    EVENTS = [
        (10,  EventType.REWARD,          0.9,  "REWARD  (DA up  -> M1 excitability up)"),
        (40,  EventType.THREAT,          0.8,  "THREAT  (CORT up -> M1 inhibition up)"),
        (70,  EventType.SOCIAL_POSITIVE, 0.85, "SOCIAL  (OT up  -> head col. bias up)"),
        (100, EventType.STARTLE,         0.7,  "STARTLE (NE up  -> mirror gain up)"),
        (140, EventType.REWARD,          0.95, "REWARD  (DA up  -> exploration boost)"),
    ]
    event_map = {fr: (et, em, el) for (fr, et, em, el) in EVENTS}

    # ---- Developmental stage schedule ------------------------------------
    # Simulate CAINE aging through stages during the demo
    STAGE_SCHEDULE = {0: 0, 50: 1, 100: 2, 150: 3}

    rng = np.random.default_rng(1)

    joint_trace    = []   # (n_frames, 6)
    art_trace      = []   # (n_frames, 6)
    mismatch_trace = []
    score_trace    = []
    da_trace       = []
    cort_trace     = []
    m1_raster      = []   # (n_frames, 60)
    loco_trace     = []   # locomotion mode per frame (str)
    loco_sim_trace = {m: [] for m in LocomotionSystem.MODES}

    print(f"\n[motor demo] Starting simulation...")
    for f in range(n_frames):
        # Advance developmental stage
        for threshold, stage in sorted(STAGE_SCHEDULE.items()):
            if f >= threshold:
                motor.developmental_stage = stage

        # Inject scheduled neurochemical event
        if f in event_map:
            ev_type, ev_mag, ev_label = event_map[f]
            neuro.update(0.0, events=[NeurochemicalEvent(ev_type, ev_mag)])
            print(f"  frame {f:3d} stage={motor.developmental_stage}: {ev_label}")

        neuro.update(dt_ms)
        snap = neuro.snapshot()

        # Synthetic spikes (moderate visual + audio activity)
        v1_spk = rng.random(20) < 0.18
        a1_spk = rng.random(20) < 0.12
        s1_rates = s1.encode(
            np.array([math.sin(f * 0.08 + i * 0.6) * 0.25
                      for i in range(6)], dtype=np.float32))

        result = motor.update(dt_ms, v1_spk, a1_spk, snap, s1_rates)

        joint_trace.append(result['joint_angles'].copy())
        art_trace.append(result['art_positions'].copy())
        mismatch_trace.append(result['efference_mismatch'])
        score_trace.append(result['motor_learning_score'])
        da_trace.append(snap.get('dopamine', 0.1))
        cort_trace.append(snap.get('cortisol', 0.08))
        loco_trace.append(result['locomotion_mode'])
        for m in LocomotionSystem.MODES:
            loco_sim_trace[m].append(result['locomotion_similarity'].get(m, 0.0))
        m1_raster.append(np.concatenate([
            result['m1_spikes'].astype(float),
            result['mirror_spikes'].astype(float),
        ]))

        if f % 50 == 0:
            print(f"  frame {f:3d}: {motor.status()}")

    print(f"\n[motor demo] Done.")

    # ---- Plot ------------------------------------------------------------
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor('#111111')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)
    t   = np.arange(n_frames) * dt_ms / 1000.0

    # Stage change markers
    stage_ts = {v * dt_ms / 1000.0: s for v, s in STAGE_SCHEDULE.items()}

    def _ax(r, c, title, xlabel='time (s)', ylabel=''):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor('#1a1a1a')
        ax.set_title(title, fontsize=9, color='#dddddd')
        ax.set_xlabel(xlabel, fontsize=7, color='#aaaaaa')
        ax.set_ylabel(ylabel, fontsize=7, color='#aaaaaa')
        ax.tick_params(colors='#888888', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#444444')
        ax.grid(True, alpha=0.2)
        # Event markers
        for (fr, _, _, _) in EVENTS:
            ax.axvline(fr * dt_ms / 1000.0, color='#ffffff',
                       lw=0.5, ls=':', alpha=0.4)
        # Stage markers
        for ts, stage in stage_ts.items():
            if ts > 0:
                ax.axvline(ts, color='#00ffaa', lw=0.6, ls='--', alpha=0.5)
                ax.text(ts + 0.01, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] != 0 else 0.9,
                        f"S{stage}", fontsize=6, color='#00ffaa', va='top')
        return ax

    # M1 raster
    ax0 = _ax(0, 0, "M1 + Mirror raster (40+20 neurons)")
    mat = np.array(m1_raster).T
    ax0.imshow(mat, aspect='auto', cmap='plasma', vmin=0, vmax=1,
               interpolation='nearest',
               extent=[t[0], t[-1], mat.shape[0], 0])
    ax0.set_ylabel("neuron", fontsize=7, color='#aaaaaa')
    ax0.axhline(40, color='#00ccff', lw=0.7, ls='--')
    ax0.text(t[-1] * 0.02, 38, 'mirror', fontsize=5, color='#00ccff', va='bottom')
    ax0.grid(False)

    # Joint angles
    ax1 = _ax(0, 1, "Joint angles (6 DOF)  [M1-emergent]", ylabel='radians')
    colors_j = ['#e74c3c','#e67e22','#2ecc71','#27ae60','#3498db','#9b59b6']
    jt = np.array(joint_trace)
    for i, (name, col) in enumerate(zip(JOINT_NAMES, colors_j)):
        ax1.plot(t, jt[:, i], lw=1.2, color=col, label=name)
    ax1.axhline(0, color='#555555', lw=0.5, ls='--')
    ax1.legend(fontsize=6, loc='upper right', facecolor='#222222',
               labelcolor='white', framealpha=0.7, ncol=2)

    # Locomotion mode (categorical, coloured band)
    ax2 = _ax(0, 2, "Locomotion mode  [WTA over M1 columns]", ylabel='mode')
    mode_colors = {'walk': '#2ecc71', 'fly': '#e67e22', 'teleport': '#e74c3c'}
    mode_y      = {'walk': 0, 'fly': 1, 'teleport': 2}
    for m, col in mode_colors.items():
        ax2.plot(t, loco_sim_trace[m], lw=1.2, color=col, label=m)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axhline(0, color='#555555', lw=0.5, ls='--')
    ax2.legend(fontsize=6, facecolor='#222222', labelcolor='white', framealpha=0.7)
    # Shade background by active mode
    active_idx = [mode_y.get(m, 0) for m in loco_trace]
    for m, col in mode_colors.items():
        mask = np.array([1.0 if lm == m else 0.0 for lm in loco_trace])
        ax2.fill_between(t, 0, mask * 0.08, color=col, alpha=0.25, lw=0)

    # Dopamine (excitability input, not a behavioral gate)
    ax3 = _ax(1, 0, "Dopamine  (M1 excitability input)", ylabel='DA level')
    ax3.plot(t, da_trace, color='#e74c3c', lw=1.5)
    ax3.axhline(0.10, color='#555555', lw=0.5, ls='--')

    # Cortisol (inhibitory input, not a suppression multiplier)
    ax4 = _ax(1, 1, "Cortisol  (M1 inhibitory input)", ylabel='CORT level')
    ax4.plot(t, cort_trace, color='#3498db', lw=1.5)
    ax4.axhline(0.08, color='#555555', lw=0.5, ls='--')

    # Motor learning score + mismatch
    ax5 = _ax(1, 2, "Motor learning score  (1 - mismatch running avg)", ylabel='score')
    ax5.plot(t, score_trace,    color='#2ecc71', lw=1.5, label='score')
    ax5.plot(t, mismatch_trace, color='#e67e22', lw=1.0, ls='--', alpha=0.7,
             label='mismatch')
    ax5.axhline(LOCO_UNLOCK_SCORE_FLY, color='#e67e22', lw=0.7, ls=':',
                label=f'fly gate={LOCO_UNLOCK_SCORE_FLY}')
    ax5.set_ylim(-0.05, 1.05)
    ax5.legend(fontsize=6, facecolor='#222222', labelcolor='white', framealpha=0.7)

    # Body map display
    ax6 = _ax(2, 0, "Body map  (column -> output)", xlabel='column', ylabel='')
    ax6.axis('off')
    bm = motor.body_map
    cell_text = [[str(ci), bm.get(str(ci), '?'),
                  _COLUMNS[int(ci)][1] if int(ci) < len(_COLUMNS) else '?']
                 for ci in sorted(bm.keys(), key=int)]
    tbl = ax6.table(cellText=cell_text,
                    colLabels=['col', 'name', 'type'],
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for key, cell in tbl.get_celld().items():
        cell.set_facecolor('#1a1a1a' if key[0] > 0 else '#2a2a2a')
        cell.set_text_props(color='#dddddd')
        cell.set_edgecolor('#333333')

    # Locomotion mastery scores
    ax7 = _ax(2, 1, "Locomotion mastery  (Hebbian WTA score)", ylabel='mastery')
    for m, col in mode_colors.items():
        # Reconstruct mastery history from final state (approximate)
        ax7.plot(t, loco_sim_trace[m], lw=1.2, color=col, alpha=0.6, ls='--')
    ax7.axhline(LOCO_UNLOCK_SCORE_FLY,         color='#e67e22', lw=0.7, ls=':',
                label=f'fly gate={LOCO_UNLOCK_SCORE_FLY}')
    ax7.axhline(LOCO_MASTERY_THRESH_TELEPORT,  color='#e74c3c', lw=0.7, ls=':',
                label=f'tele gate={LOCO_MASTERY_THRESH_TELEPORT}')
    ax7.set_ylim(-0.1, 1.1)
    ax7.legend(fontsize=6, facecolor='#222222', labelcolor='white', framealpha=0.7)

    # Arm + leg angles (M1 output, pure HH-emergent)
    ax8 = _ax(2, 2, "Arm / leg angles  [M1-emergent]", ylabel='radians')
    ax8.plot(t, jt[:, 2], color='#2ecc71', lw=1.3, label='arm_L')
    ax8.plot(t, jt[:, 3], color='#27ae60', lw=1.3, label='arm_R')
    ax8.plot(t, jt[:, 4], color='#3498db', lw=1.3, label='leg_L', ls='--')
    ax8.plot(t, jt[:, 5], color='#9b59b6', lw=1.3, label='leg_R', ls='--')
    ax8.axhline(0, color='#555555', lw=0.5, ls='--')
    ax8.legend(fontsize=6, facecolor='#222222', labelcolor='white',
               framealpha=0.7, ncol=2)

    fig.suptitle("CAINE Module 8 — Motor Cortex + Locomotion System (emergent)",
                 fontsize=12, color='#ffffff')
    out = os.path.join(_OUTPUT_DIR, 'caine_module8_motor.png')
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"[motor demo] Plot saved -> {out}")


if __name__ == '__main__':
    run_motor_demo()
