"""
CAINE Limbic System — Module 6
================================
This is when CAINE starts having feelings.

Structures
----------
  Amygdala
    BLA  (Basolateral Amygdala)   20 neurons  — threat detection, fear conditioning
    CeA  (Central Amygdala)       20 neurons  — reward salience, approach drive

  Hippocampus
    CA3  (pattern completion)     30 neurons  — recurrent collaterals, Hopfield recall
    CA1  (output / encoding)      20 neurons  — Schaffer collateral input, episodic gate

  ACC  (Anterior Cingulate Cortex) 20 neurons — conflict detection, prediction error

  Insula                           15 neurons — interoception, internal-state awareness

All populations use Hodgkin-Huxley biophysics (same model as Modules 1-4).
Episodic memories are persisted to output/episodes.json across runs.

Usage
-----
    from caine.limbic import LimbicSystem

    limbic = LimbicSystem(v1, a1, neuro)

    # each sensory tick:
    limbic.set_active_stimulus('ball_red')          # what CAINE sees right now
    result = limbic.update(dt_ms, v1_spikes, a1_spikes, neuro_snapshot)

    # Mother API — fire neurochemical events while a stimulus is visible.
    # Valence associations emerge through STDP; objects cannot be directly tagged.
    limbic.trigger_event(EventType.REWARD, 0.8)     # something good happened
    limbic.trigger_event(EventType.THREAT, 0.6)     # danger signal
"""

import os
import sys
import json
import math
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime

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
HH_DT_MS    = 0.05      # 50 µs step — coarser than sensory (limbic needs less precision)
SPIKE_THR   = 0.0       # mV
REFRAC_MS   = 2.0       # refractory period

# Tonic subthreshold drive for each structure (µA/cm²).
# HH rheobase (these parameters) ≈ 2.17 µA — all I_BASE values must stay
# below that so populations rest silently and only fire on meaningful input.
I_BASE_AMYG  = 1.6
I_BASE_HIPPO = 1.5
I_BASE_ACC   = 1.5
I_BASE_INS   = 1.6

# Stimulus-driven current scale (µA/cm²) — added on top of I_BASE
I_STIM_AMYG  = 7.5
I_STIM_HIPPO = 7.0
I_STIM_ACC   = 7.5
I_STIM_INS   = 8.0

# Amygdala — spike thresholds for neuromodulator release
BLA_FIRE_THRESH   = 0.40   # fraction of BLA neurons firing → cortisol release
CEA_FIRE_THRESH   = 0.35   # fraction of CeA neurons firing → DA + 5HT release

# STDP parameters for amygdala conditioning
AMYG_STDP_A_PLUS  = 0.025  # LTP amplitude (fear/reward conditioning)
AMYG_STDP_A_MINUS = 0.018
AMYG_STDP_TAU     = 25.0   # ms
AMYG_STDP_LR      = 0.015  # learning rate for emergent valence updates per tick
AMYG_NOVELTY_MAG  = 0.6    # magnitude of NOVEL_STIMULUS event on first encounter

# Hippocampal encoding gate
CORTISOL_BASELINE = 0.08
CORTISOL_SUPPRESS = 0.25    # above this → encoding fully suppressed

# CA3 recurrent collateral weight (Hopfield-style)
CA3_RECURRENT_SCALE = 0.04

# ACC conflict detection
ACC_MISMATCH_THRESH = 0.20   # minimum prediction error to trigger NE release
ACC_NE_MAG          = 0.40   # norepinephrine release magnitude on conflict

# Insula chemical tuning widths
INS_SIGMA = 0.08             # concentration sigma for Gaussian tuning

# Episode persistence
EPISODE_SAVE_INTERVAL = 10   # write JSON every N encoded episodes
EPISODE_MAX_MEMORY    = 500  # rolling window kept in RAM


# ===========================================================================
# SECTION 2 — BASE HH POPULATION (shared by all limbic structures)
# ===========================================================================

class LimbicPopulation:
    """
    Generic Hodgkin-Huxley population used by all limbic structures.

    State arrays are vectorised (numpy) for efficient Euler integration.
    Spike detection uses a threshold-crossing detector with refractory period,
    identical in convention to V1Population / A1Population in cortex.py.
    """

    def __init__(self, n_neurons: int, i_base: float = 2.8):
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

        # Eligibility trace for STDP (pre-synaptic side)
        self.eligibility = np.zeros(n_neurons)

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

        # Numerical guard — reset any diverged neurons
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
        """Threshold-crossing detector with refractory period.
        Returns bool array shape (N,)."""
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
        """Estimated mean firing rate (Hz) over the last window_ms for each neuron."""
        if not any(self.spike_times):
            return np.zeros(self.n)
        rates = np.zeros(self.n)
        now = max((max(st) for st in self.spike_times if st), default=0.0)
        for i, st in enumerate(self.spike_times):
            recent = [s for s in st if s >= now - window_ms]
            rates[i] = len(recent) / (window_ms / 1000.0)  # spikes/s
        return rates

    # ------------------------------------------------------------------
    def decay_eligibility(self, dt: float, tau: float = AMYG_STDP_TAU) -> None:
        """Exponential decay of STDP eligibility trace."""
        self.eligibility *= math.exp(-dt / tau)


# ===========================================================================
# SECTION 3 — AMYGDALA  (BLA + CeA)
# ===========================================================================

class Amygdala:
    """
    Amygdala with emergent valence learning — no hardcoded labels.

    BLA  (Basolateral Amygdala, 20 neurons)
        Receives direct V1/A1 input.  Learns to fire for stimuli that
        have previously been followed by threat-related neurochemicals
        (cortisol, NE elevation above baseline).

    CeA  (Central Amygdala, 20 neurons)
        Learns to fire for stimuli that have previously been followed by
        reward-related neurochemicals (DA, 5HT, OT elevation).

    Valence is never injected from outside.  It emerges from STDP:

      Teacher signal  =  (DA + 0.5*5HT + 0.3*OT) - (CORT + 0.3*NE_stress)
                         measured as elevation above each chemical's baseline.

      ACh gates learning: CAINE must be attending (high ACh) for an
      association to form.  Novelty itself raises ACh (and DA) briefly,
      making the first encounter with a stimulus a high-plasticity window.

      Novelty: first time a stimulus_id is seen → NOVEL_STIMULUS event
               (DA+NE+ACh burst, magnitude AMYG_NOVELTY_MAG).  All novel
               stimuli start with learned_valence = 0.0.

    Mother API: cannot tag objects.  Use LimbicSystem.trigger_event() to
    fire neurochemical events that CAINE experiences while viewing a
    stimulus — those events then shape valence via STDP naturally.
    """

    # Resting baselines for each chemical — used to compute elevation.
    # Must match chemicals.py ChemicalProfile baseline values.
    _CHEM_BASE = {
        'dopamine':       0.10,
        'serotonin':      0.10,
        'cortisol':       0.08,
        'oxytocin':       0.08,
        'norepinephrine': 0.10,
        'acetylcholine':  0.12,
    }

    def __init__(self, n_v1: int, n_a1: int):
        self.bla = LimbicPopulation(20, i_base=I_BASE_AMYG)
        self.cea = LimbicPopulation(20, i_base=I_BASE_AMYG)

        self.n_v1 = n_v1
        self.n_a1 = n_a1
        n_sensory  = n_v1 + n_a1

        rng = np.random.default_rng(42)

        # Feedforward weights (V1 ++ A1) → BLA / CeA
        # ~20% sparse connectivity; shaped over time by STDP
        self._w_v1a1_bla = rng.uniform(0, 0.25, (20, n_sensory)) * (
            rng.random((20, n_sensory)) < 0.20)
        self._w_v1a1_cea = rng.uniform(0, 0.20, (20, n_sensory)) * (
            rng.random((20, n_sensory)) < 0.15)

        # BLA → CeA pathway (fear state also drives CeA)
        self._w_bla_cea = rng.uniform(0.10, 0.30, (20, 20)) * (
            rng.random((20, 20)) < 0.35)
        np.fill_diagonal(self._w_bla_cea, 0)

        # STDP pre-trace: exponentially accumulates sensory activity during exposure
        self._pre_trace = np.zeros(n_sensory)

        # Emergent valence memory: stimulus_id → learned valence in [-1, +1]
        # Starts empty; new stimuli default to 0.0 (neutral) on first encounter.
        self._learned_valence: dict = {}

        # Novelty registry: stimulus_ids seen at least once before
        self._seen_stimuli: set = set()

        # Running valence signal (smoothed from learned values; decays ~500ms)
        self._current_valence = 0.0
        self._valence_tau_ms  = 500.0

        # Fast-adapting chemical baseline (tau ~120ms).
        # Teacher signal uses (current − running_avg) so it responds to
        # TRANSIENT spikes, not chronic elevated/suppressed states.
        # This prevents residual cortisol from one encounter poisoning the
        # next encounter's teacher signal.
        self._chem_avg = dict(self._CHEM_BASE)

        # Population firing fractions (read by LimbicSystem for display/neuromod)
        self.bla_fraction = 0.0
        self.cea_fraction = 0.0

        self._t_ms = 0.0

    # ------------------------------------------------------------------
    def get_valence(self, stimulus_id: str) -> float:
        """Return the currently learned valence for a stimulus_id."""
        return self._learned_valence.get(stimulus_id, 0.0)

    # ------------------------------------------------------------------
    def _compute_teacher(self, snap: dict, dt_ms: float) -> tuple:
        """
        Compute a signed teaching signal from the TRANSIENT neurochemical state.

        Uses a fast-adapting running average (_chem_avg, tau ~120ms) so the
        teacher responds to sharp concentration CHANGES rather than absolute
        levels.  Prevents chronic cortisol from a previous threat encounter
        from biasing the teacher signal during a later reward encounter.

        Returns (teacher, ach_gate):
          teacher  : float in [-1, +1]  positive = reward, negative = threat
          ach_gate : float in [0, 1]   attention gate; 0.5 at rest ACh
        """
        # Update fast-adapting baseline (~120ms)
        TAU_RUNNING = 120.0
        decay_r = math.exp(-dt_ms / TAU_RUNNING)
        for k in self._CHEM_BASE:
            curr = snap.get(k, self._CHEM_BASE[k])
            self._chem_avg[k] = self._chem_avg[k] * decay_r + curr * (1.0 - decay_r)

        # Transient = current concentration - running average
        da   = snap.get('dopamine',       0.10) - self._chem_avg['dopamine']
        ser  = snap.get('serotonin',      0.10) - self._chem_avg['serotonin']
        cort = snap.get('cortisol',       0.08) - self._chem_avg['cortisol']
        ot   = snap.get('oxytocin',       0.08) - self._chem_avg['oxytocin']
        ne   = snap.get('norepinephrine', 0.10) - self._chem_avg['norepinephrine']
        ach  = snap.get('acetylcholine',  0.12)

        # Reward: transient DA + social chemicals above running average
        reward = da + 0.5 * ser + 0.3 * ot

        # Threat: transient cortisol.  NE contributes only with co-elevated cortisol.
        ne_stress = max(0.0, ne) * (1.0 if cort > 0.005 else 0.2)
        threat = cort + 0.3 * ne_stress

        teacher = float(np.clip(reward - threat, -1.0, 1.0))

        # ACh gates learning (sigmoid, 0.5 at resting ACh; rises with attention/novelty)
        ach_gate = 1.0 / (1.0 + math.exp(-20.0 * (ach - 0.12)))

        return teacher, ach_gate

    # ------------------------------------------------------------------
    def update(self, dt_ms: float,
               v1_spikes: np.ndarray,
               a1_spikes: np.ndarray,
               neuro_snapshot: dict,
               active_stimulus: str = None) -> tuple:
        """
        Advance amygdala by dt_ms.

        Parameters
        ----------
        v1_spikes       : (n_v1,)  bool
        a1_spikes       : (n_a1,)  bool
        neuro_snapshot  : dict of current chemical concentrations
        active_stimulus : str — which stimulus_id is currently in view (or None)

        Returns
        -------
        (bla_spikes, cea_spikes, valence_signal, neuro_events)
        """
        self._t_ms += dt_ms
        n_hh_steps = max(1, int(dt_ms / HH_DT_MS))
        neuro_events = []

        # --- Sensory vector -----------------------------------------------
        sensory = np.concatenate([v1_spikes.astype(float),
                                   a1_spikes.astype(float)])

        # --- Pre-trace: accumulate sensory activity during this exposure ---
        # Decays with tau = AMYG_STDP_TAU so recent spikes matter more
        self._pre_trace *= math.exp(-dt_ms / AMYG_STDP_TAU)
        self._pre_trace += sensory

        # --- Novelty detection --------------------------------------------
        if active_stimulus is not None and active_stimulus not in self._seen_stimuli:
            self._seen_stimuli.add(active_stimulus)
            self._learned_valence.setdefault(active_stimulus, 0.0)
            # DA + NE + ACh burst: raises learning gate for the first encounter
            neuro_events.append(
                NeurochemicalEvent(EventType.NOVEL_STIMULUS, AMYG_NOVELTY_MAG))

        # --- Emergent valence STDP update ---------------------------------
        # Teacher = neurochemical state right now; ACh gates plasticity.
        # The pre-trace provides the eligibility: how active was this stimulus's
        # sensory pattern during the recent window?
        if active_stimulus is not None:
            teacher, ach_gate = self._compute_teacher(neuro_snapshot, dt_ms)
            exposure = float(np.clip(np.mean(self._pre_trace), 0.0, 1.0))

            delta = AMYG_STDP_LR * teacher * ach_gate * exposure
            prev  = self._learned_valence.get(active_stimulus, 0.0)
            self._learned_valence[active_stimulus] = float(
                np.clip(prev + delta, -1.0, 1.0))

        # --- Current valence: blend learned value into running signal -----
        self._current_valence *= math.exp(-dt_ms / self._valence_tau_ms)
        if active_stimulus is not None:
            v = self._learned_valence.get(active_stimulus, 0.0)
            # Soft blend toward the learned value so the signal is smooth
            self._current_valence += (v - self._current_valence) * 0.20

        # --- STDP weight update on feedforward connections ----------------
        # Hebbian: if current valence is significant, strengthen the weights
        # that connect active sensory patterns to whichever population fired.
        if abs(self._current_valence) > 0.05:
            bla_post = (self.bla.last_spike > self._t_ms - 2 * dt_ms).astype(float)
            cea_post = (self.cea.last_spike > self._t_ms - 2 * dt_ms).astype(float)
            lr_w = 0.001

            if self._current_valence < 0:
                dw = lr_w * abs(self._current_valence) * np.outer(
                    bla_post, self._pre_trace)
                self._w_v1a1_bla = np.clip(
                    self._w_v1a1_bla + dw * AMYG_STDP_A_PLUS, 0.0, 0.8)
            else:
                dw = lr_w * self._current_valence * np.outer(
                    cea_post, self._pre_trace)
                self._w_v1a1_cea = np.clip(
                    self._w_v1a1_cea + dw * AMYG_STDP_A_PLUS, 0.0, 0.8)

        # --- Drive currents -----------------------------------------------
        gain = neuro_snapshot.get('norepinephrine', 0.1) / 0.1
        gain = float(np.clip(gain, 0.5, 2.5))

        bla_input = self._w_v1a1_bla @ sensory
        I_bla = I_BASE_AMYG + np.clip(bla_input, 0, 1) * I_STIM_AMYG * gain
        if self._current_valence < 0:
            # Learned threat association boosts BLA excitability
            I_bla += abs(self._current_valence) * 3.0

        cea_input = (self._w_bla_cea @ (self.bla.V > SPIKE_THR).astype(float)
                     + self._w_v1a1_cea @ sensory)
        I_cea = I_BASE_AMYG + np.clip(cea_input, 0, 1.5) * I_STIM_AMYG * gain
        if self._current_valence > 0:
            # Learned reward association boosts CeA excitability
            I_cea += self._current_valence * 3.0

        # --- HH integration -----------------------------------------------
        bla_spikes_accum = np.zeros(20, dtype=bool)
        cea_spikes_accum = np.zeros(20, dtype=bool)

        for step_i in range(n_hh_steps):
            t = self._t_ms + step_i * HH_DT_MS
            self.bla.step(HH_DT_MS, I_bla)
            self.cea.step(HH_DT_MS, I_cea)
            bla_spikes_accum |= self.bla.detect_spikes(t)
            cea_spikes_accum |= self.cea.detect_spikes(t)

        self.bla_fraction = float(bla_spikes_accum.sum()) / 20.0
        self.cea_fraction = float(cea_spikes_accum.sum()) / 20.0

        # --- Neuromodulator release gated by learned valence ---------------
        # BLA and CeA only release chemicals once an association has formed.
        # Before learning (valence ≈ 0), the populations can still fire
        # (driven by raw sensory input) but the downstream neuromodulator
        # cascade is suppressed.  This prevents early random BLA firing
        # from flooding cortisol and corrupting the very STDP that is trying
        # to establish valence in the first place.
        lv = self._learned_valence.get(active_stimulus, 0.0) if active_stimulus else 0.0

        if self.bla_fraction >= BLA_FIRE_THRESH and lv < -0.05:
            # Learned threat: BLA burst → cortisol + NE (scales with learned strength)
            mag = float(np.clip(self.bla_fraction * 1.5 * abs(lv) * 5.0, 0.2, 1.0))
            neuro_events.append(NeurochemicalEvent(EventType.AMYGDALA_BLA, mag))

        if self.cea_fraction >= CEA_FIRE_THRESH and lv > 0.05:
            # Learned reward: CeA burst → DA + 5HT (scales with learned strength)
            mag = float(np.clip(self.cea_fraction * 1.5 * lv * 5.0, 0.2, 1.0))
            neuro_events.append(NeurochemicalEvent(EventType.REWARD, mag * 0.5))

        return bla_spikes_accum, cea_spikes_accum, self._current_valence, neuro_events


# ===========================================================================
# SECTION 4 — HIPPOCAMPUS  (CA3 + CA1)
# ===========================================================================

class Hippocampus:
    """
    Hippocampal system with two linked populations:

    CA3  (30 neurons, pattern completion)
        Receives sensory input via simulated perforant path.
        Recurrent Schaffer collaterals implement Hopfield-style associative
        memory: partial input patterns drive full-episode recall.
        Weights are updated (slowly) by Hebbian learning as episodes are encoded.

    CA1  (20 neurons, output / encoding gate)
        Receives input from CA3 (Schaffer collateral pathway).
        Cortisol suppresses CA1 spiking → stress impairs episodic encoding.

    Episodic Memory
        Each frame that CA1 is active above threshold gets stored as a
        compressed episode: {timestamp, V1_pattern, A1_pattern, neuro, valence}.
        Episodes are persisted to output/episodes.json every N encodings.

    Replay
        Called during low-activity frames.  CA3 is driven by a stored episode
        pattern, replaying the activation trajectory without sensory input.
    """

    def __init__(self, episode_file: str = None):
        self.ca3 = LimbicPopulation(30, i_base=I_BASE_HIPPO)
        self.ca1 = LimbicPopulation(20, i_base=I_BASE_HIPPO)

        rng = np.random.default_rng(7)

        # Perforant path: sensory → CA3 (sparse, ~15% connectivity)
        # Size determined at update() time once we know n_v1, n_a1
        self._w_sensory_ca3 = None  # built lazily on first update

        # CA3 recurrent collaterals (Hopfield memory matrix, 30×30)
        # Initialised small; strengthened by Hebbian learning during encoding
        self._w_ca3_ca3 = np.zeros((30, 30))

        # Schaffer collaterals: CA3 → CA1
        self._w_ca3_ca1 = rng.uniform(0.05, 0.25, (20, 30)) * (
            rng.random((20, 30)) < 0.40)

        # Episode store (in-RAM rolling buffer)
        self._episodes: list = []
        self._episode_file = episode_file or os.path.join(_OUTPUT_DIR, 'episodes.json')
        self._encode_count = 0
        self._t_ms = 0.0

        # Load existing episodes from disk
        self._load_episodes()

        # Replay state
        self._replaying = False
        self._replay_pattern_ca3 = None

    # ------------------------------------------------------------------
    def _load_episodes(self) -> None:
        """Load persisted episodes from JSON (if file exists)."""
        if os.path.exists(self._episode_file):
            try:
                with open(self._episode_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._episodes = data[-EPISODE_MAX_MEMORY:]
                # Rebuild Hopfield weight matrix from stored patterns
                self._rebuild_ca3_weights()
                print(f"[hippocampus] Loaded {len(self._episodes)} episodes "
                      f"from {self._episode_file}")
            except Exception as e:
                print(f"[hippocampus] Could not load episodes: {e}")

    def _save_episodes(self) -> None:
        """Persist current episode buffer to JSON."""
        try:
            with open(self._episode_file, 'w', encoding='utf-8') as f:
                json.dump(self._episodes[-EPISODE_MAX_MEMORY:], f,
                          indent=None, separators=(',', ':'))
        except Exception as e:
            print(f"[hippocampus] Save failed: {e}")

    def _rebuild_ca3_weights(self) -> None:
        """
        Rebuild the CA3 recurrent Hopfield weight matrix from stored episodes.
        w_ij = (1/N) * sum_k( x_i^k * x_j^k ) — Hebbian outer-product rule.
        """
        patterns = [ep['v1_pattern'][:30] for ep in self._episodes
                    if 'v1_pattern' in ep and len(ep['v1_pattern']) >= 30]
        if not patterns:
            return
        N = len(patterns)
        W = np.zeros((30, 30))
        for p in patterns:
            x = np.array(p[:30], dtype=float) * 2 - 1   # bipolar {-1, +1}
            W += np.outer(x, x)
        W /= N
        np.fill_diagonal(W, 0)
        self._w_ca3_ca3 = np.clip(W * CA3_RECURRENT_SCALE, -0.1, 0.1)

    # ------------------------------------------------------------------
    def encode(self, t_ms: float,
               v1_spikes: np.ndarray,
               a1_spikes: np.ndarray,
               neuro_snapshot: dict,
               valence: float) -> bool:
        """
        Attempt to encode a new episodic memory.

        Returns True if encoding succeeded (not blocked by cortisol).
        """
        # High cortisol suppresses CA1 encoding (stress-induced memory impairment)
        cortisol = neuro_snapshot.get('cortisol', CORTISOL_BASELINE)
        gate = 1.0 - np.clip(
            (cortisol - CORTISOL_BASELINE) / (CORTISOL_SUPPRESS - CORTISOL_BASELINE),
            0.0, 1.0)

        if gate < 0.15:
            return False   # fully suppressed

        episode = {
            'time_ms':   round(float(t_ms), 1),
            'wall_time': datetime.utcnow().isoformat(),
            'v1_pattern': v1_spikes.astype(int).tolist(),
            'a1_pattern': a1_spikes.astype(int).tolist(),
            'neuro':     {k: round(float(v), 4) for k, v in neuro_snapshot.items()},
            'valence':   round(float(valence), 3),
            'cortisol_gate': round(float(gate), 3),
        }
        self._episodes.append(episode)
        if len(self._episodes) > EPISODE_MAX_MEMORY:
            self._episodes = self._episodes[-EPISODE_MAX_MEMORY:]

        self._encode_count += 1

        # Hebbian update to CA3 recurrent weights with the new pattern
        x = v1_spikes[:30].astype(float) * 2 - 1  # bipolar
        x = np.pad(x, (0, max(0, 30 - len(x))))[:30]
        dW = np.outer(x, x)
        np.fill_diagonal(dW, 0)
        self._w_ca3_ca3 = np.clip(
            self._w_ca3_ca3 + dW * CA3_RECURRENT_SCALE * 0.01 * gate,
            -0.15, 0.15)

        if self._encode_count % EPISODE_SAVE_INTERVAL == 0:
            self._save_episodes()

        return True

    # ------------------------------------------------------------------
    def replay(self, dt_ms: float,
               v1_spikes: np.ndarray,
               a1_spikes: np.ndarray,
               neuro_snapshot: dict) -> tuple:
        """
        Drive CA3 with a recalled episode pattern (memory replay).

        Called when current sensory activity is low.  CA3 completes the pattern
        via its recurrent weights, then drives CA1.

        Returns (ca3_spikes, ca1_spikes, replayed_episode_or_None)
        """
        if not self._episodes:
            return (np.zeros(30, dtype=bool), np.zeros(20, dtype=bool), None)

        # Pick a recent episode (biased toward recent; slight random)
        idx = int(np.clip(
            len(self._episodes) - 1 - int(np.random.exponential(10)),
            0, len(self._episodes) - 1))
        episode = self._episodes[idx]

        # Construct CA3 drive from stored V1 pattern
        stored_v1 = np.array(episode['v1_pattern'], dtype=float)
        stored_v1 = np.pad(stored_v1, (0, max(0, 30 - len(stored_v1))))[:30]

        I_ca3_replay = I_BASE_HIPPO + stored_v1 * I_STIM_HIPPO * 0.7

        return self._run_ca3_ca1(dt_ms, I_ca3_replay, neuro_snapshot, episode)

    # ------------------------------------------------------------------
    def update(self, dt_ms: float,
               v1_spikes: np.ndarray,
               a1_spikes: np.ndarray,
               neuro_snapshot: dict,
               valence: float,
               force_replay: bool = False) -> tuple:
        """
        Advance hippocampus by dt_ms.

        Returns (ca3_spikes, ca1_spikes, encoded)
        """
        self._t_ms += dt_ms

        # Build perforant path weight matrix lazily
        n_sensory = len(v1_spikes) + len(a1_spikes)
        if (self._w_sensory_ca3 is None
                or self._w_sensory_ca3.shape[1] != n_sensory):
            rng = np.random.default_rng(13)
            self._w_sensory_ca3 = rng.uniform(0, 0.20, (30, n_sensory)) * (
                rng.random((30, n_sensory)) < 0.15)

        # Decide: encode or replay?
        total_spikes = v1_spikes.sum() + a1_spikes.sum()
        do_replay    = force_replay or (total_spikes == 0 and len(self._episodes) > 5)

        if do_replay:
            ca3_spikes, ca1_spikes, _ = self.replay(
                dt_ms, v1_spikes, a1_spikes, neuro_snapshot)
            encoded = False
        else:
            # Normal encoding path
            sensory = np.concatenate([v1_spikes.astype(float),
                                       a1_spikes.astype(float)])

            # Perforant path: sensory → CA3
            ca3_input = self._w_sensory_ca3 @ sensory
            # Recurrent: CA3 → CA3
            ca3_recurrent = self._w_ca3_ca3 @ (self.ca3.V > SPIKE_THR).astype(float)
            I_ca3 = I_BASE_HIPPO + np.clip(ca3_input + ca3_recurrent, 0, 1.5) * I_STIM_HIPPO

            ca3_spikes, ca1_spikes, _ = self._run_ca3_ca1(
                dt_ms, I_ca3, neuro_snapshot)

            encoded = self.encode(self._t_ms, v1_spikes, a1_spikes,
                                   neuro_snapshot, valence)

        return ca3_spikes, ca1_spikes, False if do_replay else True

    # ------------------------------------------------------------------
    def _run_ca3_ca1(self, dt_ms, I_ca3, neuro_snapshot, episode=None):
        """Shared HH integration for CA3 → CA1."""
        n_steps = max(1, int(dt_ms / HH_DT_MS))

        # Cortisol gates CA1 excitability
        cortisol = neuro_snapshot.get('cortisol', CORTISOL_BASELINE)
        ca1_gate = float(np.clip(
            1.0 - (cortisol - CORTISOL_BASELINE) /
            (CORTISOL_SUPPRESS - CORTISOL_BASELINE), 0.1, 1.0))

        ca3_spikes_accum = np.zeros(30, dtype=bool)
        ca1_spikes_accum = np.zeros(20, dtype=bool)

        for step_i in range(n_steps):
            t = self._t_ms + step_i * HH_DT_MS

            self.ca3.step(HH_DT_MS, I_ca3)
            ca3_fired = self.ca3.detect_spikes(t)
            ca3_spikes_accum |= ca3_fired

            # Schaffer collaterals: CA3 → CA1
            ca3_rate = ca3_fired.astype(float)
            ca3_to_ca1 = self._w_ca3_ca1 @ ca3_rate
            I_ca1 = (I_BASE_HIPPO * ca1_gate +
                     np.clip(ca3_to_ca1, 0, 1.5) * I_STIM_HIPPO * ca1_gate)

            self.ca1.step(HH_DT_MS, I_ca1)
            ca1_spikes_accum |= self.ca1.detect_spikes(t)

        return ca3_spikes_accum, ca1_spikes_accum, episode

    # ------------------------------------------------------------------
    @property
    def n_episodes(self) -> int:
        return len(self._episodes)


# ===========================================================================
# SECTION 5 — ANTERIOR CINGULATE CORTEX  (ACC)
# ===========================================================================

class ACC(LimbicPopulation):
    """
    Anterior Cingulate Cortex — conflict / prediction-error detection.

    Tracks the recent outcome history for each stimulus type.  When the
    current outcome deviates significantly from the running mean, ACC fires
    and triggers norepinephrine release (learning-rate boost).

    Conflict signal feeds back to the amygdala and hippocampus to flag
    that the current episode should be encoded with elevated salience.
    """

    def __init__(self, neuro):
        super().__init__(20, i_base=I_BASE_ACC)
        self._neuro = neuro
        # Rolling outcome history per stimulus_id
        self._outcome_history: dict = defaultdict(lambda: deque(maxlen=5))
        self._conflict_level  = 0.0   # running estimate (decays)
        self._t_ms            = 0.0
        self._ne_refractory   = 0.0   # cooldown between NE releases

    # ------------------------------------------------------------------
    def record_outcome(self, stimulus_id: str, outcome: float) -> None:
        """Log a signed outcome for a stimulus (-1 bad, +1 good, 0 neutral)."""
        self._outcome_history[stimulus_id].append(float(outcome))

    # ------------------------------------------------------------------
    def compute_conflict(self, stimulus_id: str,
                         actual_outcome: float) -> float:
        """
        Compare actual_outcome to recent history for this stimulus.
        Returns conflict magnitude ∈ [0, 1].
        """
        history = list(self._outcome_history.get(stimulus_id, []))
        if len(history) < 2:
            return 0.0
        expected = float(np.mean(history[:-1]))
        conflict = float(np.abs(actual_outcome - expected))
        return float(np.clip(conflict, 0.0, 1.0))

    # ------------------------------------------------------------------
    def update(self, dt_ms: float,
               conflict_signal: float,
               amygdala_valence: float,
               neuro_snapshot: dict) -> tuple:
        """
        Advance ACC by dt_ms.

        Parameters
        ----------
        conflict_signal   : prediction-error magnitude [0, 1]
        amygdala_valence  : current amygdala valence signal

        Returns
        -------
        (acc_spikes, conflict_level, neuro_events)
        """
        self._t_ms  += dt_ms
        self._ne_refractory = max(0, self._ne_refractory - dt_ms)

        # Decay conflict estimate
        self._conflict_level = (self._conflict_level * math.exp(-dt_ms / 200.0)
                                + conflict_signal * (1 - math.exp(-dt_ms / 200.0)))

        # Drive ACC proportional to conflict + valence mismatch
        ne_mod = neuro_snapshot.get('norepinephrine', 0.1) / 0.1
        I_acc = I_BASE_ACC + self._conflict_level * I_STIM_ACC * float(np.clip(ne_mod, 0.5, 2.0))

        # If amygdala says positive but prediction says negative → extra conflict
        valence_surprise = abs(amygdala_valence) * (1 - abs(self._conflict_level))
        I_acc += valence_surprise * 2.0

        I_acc_arr = np.full(20, I_acc)

        n_steps = max(1, int(dt_ms / HH_DT_MS))
        acc_spikes_accum = np.zeros(20, dtype=bool)
        for step_i in range(n_steps):
            t = self._t_ms + step_i * HH_DT_MS
            self.step(HH_DT_MS, I_acc_arr)
            acc_spikes_accum |= self.detect_spikes(t)

        neuro_events = []
        fire_frac = float(acc_spikes_accum.sum()) / 20.0

        # NE release when conflict is high and ACC fires strongly
        if (fire_frac > 0.4
                and self._conflict_level > ACC_MISMATCH_THRESH
                and self._ne_refractory <= 0):
            neuro_events.append(
                NeurochemicalEvent(EventType.ACC_CONFLICT,
                                   float(np.clip(fire_frac * ACC_NE_MAG, 0.1, 1.0))))
            self._ne_refractory = 500.0  # 500ms cooldown

        return acc_spikes_accum, self._conflict_level, neuro_events


# ===========================================================================
# SECTION 6 — INSULA  (interoception / internal state)
# ===========================================================================

class Insula(LimbicPopulation):
    """
    Insula — CAINE's awareness of how he feels.

    15 neurons tuned to different configurations of the neurochemical space:
      Neurons 0-5   : each tuned to one of the 6 chemicals (peak at that
                      chemical's resting concentration + moderate elevation)
      Neurons 6-11  : tuned to pairwise combinations (DA+NE, 5HT+OT, CORT+NE, ...)
      Neurons 12-14 : global arousal, valence positivity, stress index

    Output feeds into the amygdala (as interoceptive context) and PFC stub.
    Insula firing patterns represent CAINE's felt sense of his own state.
    """

    # Chemical order matches chemicals.py snapshot() keys
    _CHEM_ORDER = [
        'dopamine', 'serotonin', 'cortisol',
        'oxytocin', 'norepinephrine', 'acetylcholine',
    ]

    # Preferred concentration for each of the 6 single-chemical neurons
    # (slightly above baseline to detect meaningful elevation)
    _PREF_CONC = {
        'dopamine':       0.18,
        'serotonin':      0.16,
        'cortisol':       0.14,
        'oxytocin':       0.14,
        'norepinephrine': 0.18,
        'acetylcholine':  0.20,
    }

    def __init__(self, neuro):
        super().__init__(15, i_base=I_BASE_INS)
        self._neuro = neuro
        self._t_ms  = 0.0

        # Readout of internal state as a named scalar
        self.felt_valence = 0.0   # positive = good, negative = bad
        self.felt_arousal = 0.0   # 0..1

    # ------------------------------------------------------------------
    def _chem_to_drive(self, snap: dict) -> np.ndarray:
        """
        Map neurochemical snapshot to per-neuron drive currents.

        Neurons 0-5  : Gaussian tuning to individual chemicals
        Neurons 6-11 : product of two chemical activations
        Neurons 12-14: computed composite states
        """
        I = np.zeros(15, dtype=float)

        levels = np.array([snap.get(c, 0.08) for c in self._CHEM_ORDER])

        # --- Single-chemical neurons (0-5) ---
        for i, chem in enumerate(self._CHEM_ORDER):
            pref  = self._PREF_CONC[chem]
            diff  = (levels[i] - pref) / INS_SIGMA
            tuning = math.exp(-0.5 * diff ** 2)
            I[i] = I_BASE_INS + tuning * I_STIM_INS

        # --- Pairwise neurons (6-11) ---
        pairs = [(0, 4), (1, 3), (2, 4), (0, 1), (2, 3), (4, 5)]
        for j, (a, b) in enumerate(pairs):
            combined = math.sqrt(max(0, levels[a]) * max(0, levels[b]))
            I[6 + j] = I_BASE_INS + combined * I_STIM_INS * 5.0

        # --- Composite state neurons (12-14) ---
        da, ser, cort, ot, ne, ach = levels

        # 12: global arousal (NE + DA - CORT)
        arousal = float(np.clip((ne + da - cort) * 3.0, 0, 1))
        I[12]   = I_BASE_INS + arousal * I_STIM_INS
        self.felt_arousal = arousal

        # 13: positive valence (DA + 5HT + OT - CORT)
        pos_val = float(np.clip((da + ser + ot - cort) * 2.0, 0, 1))
        I[13]   = I_BASE_INS + pos_val * I_STIM_INS

        # 14: stress / distress index (CORT + NE - OT - 5HT)
        stress  = float(np.clip((cort + ne - ot - ser) * 2.0, 0, 1))
        I[14]   = I_BASE_INS + stress * I_STIM_INS

        # Update felt valence composite
        self.felt_valence = float(np.clip(pos_val - stress, -1, 1))

        return I.astype(np.float64)

    # ------------------------------------------------------------------
    def update(self, dt_ms: float, neuro_snapshot: dict) -> np.ndarray:
        """
        Advance insula by dt_ms.

        Returns spikes array (15,) bool.
        """
        self._t_ms += dt_ms
        n_steps = max(1, int(dt_ms / HH_DT_MS))
        I_ins   = self._chem_to_drive(neuro_snapshot)

        spikes_accum = np.zeros(15, dtype=bool)
        for step_i in range(n_steps):
            t = self._t_ms + step_i * HH_DT_MS
            self.step(HH_DT_MS, I_ins)
            spikes_accum |= self.detect_spikes(t)

        return spikes_accum


# ===========================================================================
# SECTION 7 — LIMBIC SYSTEM  (integration / public API)
# ===========================================================================

class LimbicSystem:
    """
    Top-level integration of all limbic structures.

    Usage
    -----
        limbic = LimbicSystem(v1, a1, neuro)

        # each sensory tick:
        limbic.set_active_stimulus('object_name')   # what CAINE is looking at
        result = limbic.update(dt_ms, v1_spikes, a1_spikes, neuro_snapshot)

    Mother API (no hardcoded valence labels):
        limbic.trigger_event(EventType.REWARD,  0.8)  # fire reward into CAINE
        limbic.trigger_event(EventType.THREAT,  0.6)  # fire threat signal
        limbic.record_outcome('approach', -0.5)       # ACC conflict tracking

        Mother cannot directly tag objects — valence associations form
        purely through STDP between sensory exposure and the neurochemical
        state that follows.  Use trigger_event() to create that state.
    """

    def __init__(self, v1, a1, neuro,
                 episode_file: str = None):
        self.v1    = v1
        self.a1    = a1
        self.neuro = neuro

        self.amygdala    = Amygdala(v1.n, a1.n)
        self.hippocampus = Hippocampus(episode_file)
        self.acc         = ACC(neuro)
        self.insula      = Insula(neuro)

        self._t_ms            = 0.0
        self._active_stimulus = None

        # Rolling histories for visualisation
        self._bla_history    = deque(maxlen=80)
        self._cea_history    = deque(maxlen=80)
        self._ca1_history    = deque(maxlen=80)
        self._acc_history    = deque(maxlen=80)
        self._insula_history = deque(maxlen=80)
        self._valence_history = deque(maxlen=80)
        self._felt_history    = deque(maxlen=80)   # felt_valence over time

    # ------------------------------------------------------------------
    # Mother API
    # ------------------------------------------------------------------

    def trigger_event(self, event_type, magnitude: float = 1.0) -> None:
        """
        Mother API: fire a neurochemical event into CAINE's environment.

        This is the only way the Mother can influence valence — by creating
        the neurochemical context that CAINE experiences while viewing a
        stimulus.  STDP then gradually associates that context with the
        stimulus over repeated exposures.

        Examples
        --------
            from caine.chemicals import EventType
            limbic.trigger_event(EventType.REWARD, 0.8)   # something good happened
            limbic.trigger_event(EventType.THREAT, 0.6)   # danger signal
            limbic.trigger_event(EventType.SOCIAL_POSITIVE, 1.0)

        Parameters
        ----------
        event_type : EventType  (from caine.chemicals)
        magnitude  : float in [0.0, 1.0]
        """
        self.neuro.update(0.0, events=[
            NeurochemicalEvent(event_type, float(np.clip(magnitude, 0.0, 1.0)))])

    def set_active_stimulus(self, stimulus_id: str) -> None:
        """Tell the limbic system which stimulus is currently in view."""
        self._active_stimulus = stimulus_id

    def record_outcome(self, stimulus_id: str, outcome: float) -> None:
        """Log an outcome (-1..+1) for ACC conflict tracking."""
        self.acc.record_outcome(stimulus_id, outcome)

    def get_learned_valences(self) -> dict:
        """Return a copy of all currently learned valence associations."""
        return dict(self.amygdala._learned_valence)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, dt_ms: float,
               v1_spikes: np.ndarray,
               a1_spikes: np.ndarray,
               neuro_snapshot: dict,
               conflict_signal: float = 0.0) -> dict:
        """
        Advance all limbic structures by dt_ms.

        Parameters
        ----------
        v1_spikes, a1_spikes : bool arrays from the sensory layer
        neuro_snapshot       : dict from NeurochemicalSystem.snapshot()
        conflict_signal      : optional external prediction-error signal

        Returns
        -------
        dict with all limbic spike arrays, valence, and neuro events
        """
        self._t_ms += dt_ms
        all_neuro_events = []

        # ---- 1. Insula reads internal chemical state ----
        insula_spikes = self.insula.update(dt_ms, neuro_snapshot)

        # ---- 2. Amygdala processes sensory input ----
        bla_spikes, cea_spikes, valence, amyg_events = self.amygdala.update(
            dt_ms, v1_spikes, a1_spikes, neuro_snapshot,
            active_stimulus=self._active_stimulus)
        all_neuro_events.extend(amyg_events)

        # ---- 3. ACC detects conflict ----
        acc_spikes, conflict_level, acc_events = self.acc.update(
            dt_ms, conflict_signal, valence, neuro_snapshot)
        all_neuro_events.extend(acc_events)

        # ---- 4. Hippocampus encodes / replays ----
        ca3_spikes, ca1_spikes, encoded = self.hippocampus.update(
            dt_ms, v1_spikes, a1_spikes, neuro_snapshot, valence)

        # ---- 5. Apply neuromodulator events to neurochemical system ----
        if all_neuro_events:
            self.neuro.update(0.0, events=all_neuro_events)

        # ---- 6. Store histories ----
        self._bla_history.append(bla_spikes.astype(float))
        self._cea_history.append(cea_spikes.astype(float))
        self._ca1_history.append(ca1_spikes.astype(float))
        self._acc_history.append(acc_spikes.astype(float))
        self._insula_history.append(insula_spikes.astype(float))
        self._valence_history.append(float(valence))
        self._felt_history.append(float(self.insula.felt_valence))

        return {
            'bla_spikes':       bla_spikes,
            'cea_spikes':       cea_spikes,
            'ca3_spikes':       ca3_spikes,
            'ca1_spikes':       ca1_spikes,
            'acc_spikes':       acc_spikes,
            'insula_spikes':    insula_spikes,
            'valence':          valence,
            'conflict':         conflict_level,
            'felt_valence':     self.insula.felt_valence,
            'felt_arousal':     self.insula.felt_arousal,
            'bla_fraction':     self.amygdala.bla_fraction,
            'cea_fraction':     self.amygdala.cea_fraction,
            'n_episodes':       self.hippocampus.n_episodes,
            'neuro_events':     all_neuro_events,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def status(self) -> str:
        """One-line status string for console output."""
        # Show up to 3 learned valences (most extreme ones)
        lv = self.amygdala._learned_valence
        if lv:
            top = sorted(lv.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            lv_str = '  '.join(f"{k}={v:+.2f}" for k, v in top)
        else:
            lv_str = "none"
        return (
            f"BLA={self.amygdala.bla_fraction:.2f}  "
            f"CeA={self.amygdala.cea_fraction:.2f}  "
            f"felt={self.insula.felt_valence:+.2f}  "
            f"arousal={self.insula.felt_arousal:.2f}  "
            f"ep={self.hippocampus.n_episodes}  "
            f"valences=[{lv_str}]"
        )


# ===========================================================================
# SECTION 8 — STAND-ALONE DEMO
# ===========================================================================

def run_limbic_demo(n_frames: int = 150, dt_ms: float = 20.0):
    """
    Headless demo of emergent valence learning.

    No valence is hardcoded.  Instead:
      - CAINE is exposed to each object in sustained 12-frame windows.
      - During each window, Mother fires a neurochemical event (THREAT or REWARD).
      - STDP accumulates across the whole window (cortisol from THREAT persists
        ~8s, so the teacher signal remains active for many frames after the event).
      - Red_ball becomes negatively valenced; blue_ball positively — emergently.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from caine.cortex    import V1Population, A1Population
    from caine.chemicals import NeurochemicalSystem, EventType

    v1    = V1Population()
    a1    = A1Population()
    neuro = NeurochemicalSystem()

    limbic = LimbicSystem(v1, a1, neuro,
                          episode_file=os.path.join(_OUTPUT_DIR,
                                                    'episodes_demo.json'))

    # ---- Encounter windows -----------------------------------------------
    # (window_start, window_end, stimulus_id, event_frame, EventType, magnitude)
    # CAINE sees the stimulus for the whole window; event fires mid-window.
    # Cortisol/DA from the event persists across the remaining window frames,
    # driving STDP for all frames where the stimulus is still visible.
    WINDOWS = [
        ( 5,  18, 'red_ball',   8, EventType.THREAT,  0.75),  # fear 1
        (25,  40, 'blue_ball', 28, EventType.REWARD,  0.85),  # reward 1
        (50,  65, 'red_ball',  53, EventType.THREAT,  0.85),  # fear 2
        (75,  92, 'blue_ball', 78, EventType.REWARD,  0.90),  # reward 2
        (105, 120, 'red_ball', 108, EventType.THREAT, 0.80),  # fear 3 (test)
        (128, 145, 'blue_ball',131, EventType.REWARD, 0.85),  # reward 3 (test)
    ]
    # Fast lookup: frame → event to fire this frame (event may lag window start)
    event_frames = {ew: (stim, et, em) for (_, _, stim, ew, et, em) in WINDOWS}
    # Stimulus presence: frame → stimulus_id
    stim_at = {}
    for (ws, we, stim, *_) in WINDOWS:
        for fr in range(ws, we):
            stim_at[fr] = stim

    rng = np.random.default_rng(0)
    bla_trace, cea_trace, ca1_trace, felt_trace, neuro_da = [], [], [], [], []
    valence_red, valence_blue = [], []

    print(f"[limbic] Emergent valence demo: {n_frames} frames x {dt_ms} ms")
    print(f"[limbic] No labels set — learning from STDP + neurochemical context")

    for f in range(n_frames):
        # --- Active stimulus for this frame --------------------------------
        stim_id = stim_at.get(f)
        limbic.set_active_stimulus(stim_id)

        # --- Mother fires event if scheduled this frame --------------------
        if f in event_frames:
            _, ev_type, ev_mag = event_frames[f]
            limbic.trigger_event(ev_type, ev_mag)

        # --- Synthetic spikes: high during encounter windows ---------------
        if stim_id is not None:
            v1_spk = rng.random(20) < 0.70   # clear sensory response
            a1_spk = rng.random(20) < 0.50
        else:
            v1_spk = rng.random(20) < 0.05   # background noise
            a1_spk = rng.random(20) < 0.05

        neuro.update(dt_ms)
        snap   = neuro.snapshot()
        result = limbic.update(dt_ms, v1_spk, a1_spk, snap)

        bla_trace.append(result['bla_fraction'])
        cea_trace.append(result['cea_fraction'])
        ca1_trace.append(result['ca1_spikes'].sum() / 20.0)
        felt_trace.append(result['felt_valence'])
        neuro_da.append(snap.get('dopamine', 0.1))
        valence_red.append(limbic.amygdala.get_valence('red_ball'))
        valence_blue.append(limbic.amygdala.get_valence('blue_ball'))

        if f % 30 == 0:
            print(f"  frame {f:3d}: {limbic.status()}")

    # ---- Summary ---------------------------------------------------------
    print(f"\n[limbic] Learned valences after {n_frames} frames:")
    for stim_id, val in limbic.get_learned_valences().items():
        print(f"  {stim_id:20s}  {val:+.4f}")

    # ---- Plot ------------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.3)
    t   = np.arange(n_frames) * dt_ms / 1000.0

    def _ax(r, c, title, ylabel):
        ax = fig.add_subplot(gs[r, c])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        # Shade encounter windows
        for (ws, we, stim, *_) in WINDOWS:
            col = '#e74c3c' if 'red' in stim else '#2ecc71'
            ax.axvspan(ws * dt_ms / 1000.0, we * dt_ms / 1000.0,
                       color=col, alpha=0.08, lw=0)
        return ax

    ax1 = _ax(0, 0, "Emergent learned valence (no labels set)",
              "valence [-1..+1]")
    ax1.plot(t, valence_red,  color='#e74c3c', lw=1.8, label='red_ball (THREAT)')
    ax1.plot(t, valence_blue, color='#2ecc71', lw=1.8, label='blue_ball (REWARD)')
    ax1.axhline(0, color='gray', lw=0.7, ls='--')
    ax1.set_ylim(-1.05, 1.05)
    ax1.legend(fontsize=8)

    ax2 = _ax(0, 1, "Dopamine (REWARD signal)", "DA level")
    ax2.plot(t, neuro_da, color='#e67e22', lw=1.5)

    ax3 = _ax(1, 0, "BLA firing fraction (threat response)", "fraction")
    ax3.plot(t, bla_trace, color='#c0392b', lw=1.5)
    ax3.set_ylim(0, 1)

    ax4 = _ax(1, 1, "CeA firing fraction (reward response)", "fraction")
    ax4.plot(t, cea_trace, color='#27ae60', lw=1.5)
    ax4.set_ylim(0, 1)

    ax5 = _ax(2, 0, "CA1 output (episodic encoding)", "fraction")
    ax5.plot(t, ca1_trace, color='#3498db', lw=1.5)
    ax5.set_ylim(0, 1)

    ax6 = _ax(2, 1, "Felt valence (Insula)", "valence")
    ax6.plot(t, felt_trace, color='#9b59b6', lw=1.5)
    ax6.axhline(0, color='gray', lw=0.7, ls='--')
    ax6.set_ylim(-1.05, 1.05)

    fig.suptitle("CAINE Module 6 — Emergent Valence Learning (no hardcoded labels)",
                 fontsize=12)
    out = os.path.join(_OUTPUT_DIR, 'caine_module6_limbic.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()

    limbic.hippocampus._save_episodes()
    print(f"[limbic] Plot: {out}")
    print(f"[limbic] Episodes: {limbic.hippocampus.n_episodes}")
    return limbic


if __name__ == '__main__':
    run_limbic_demo()
