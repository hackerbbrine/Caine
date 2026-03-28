"""
CAINE — Module 9: Full System Integration Hub
===============================================
CAINEBrain is the integration class that holds every module and wires them
into a single per-frame pipeline.  It is not a demo — it is the living system.

Pipeline (executed once per frame):
    1.  Environment step → camera feed (64×64 RGB)
    2.  Audio frame from microphone (or hardware fallback tone)
    3.  SensoryLayer processes vision + audio → V1 + A1 spikes
    4.  S1 encodes proprioceptive joint angles from motor output
    5.  LimbicSystem updates on V1 + A1 + neurochemicals → episodic memory + valence
    6.  Limbic neuro-events injected into NeurochemicalSystem
    7.  MotorCortex updates → joint targets + finger positions + world actions
    8.  Motor neuro-events injected
    9.  Avatar pose pushed to environment
   10.  Mother/Father parenting system runs
   11.  Neurogenesis tracker checks firing rates → logs growth events
   12.  Stage manager checks exit conditions (once per simulated hour)
   13.  HDF5 checkpoint (once per 10 simulated minutes)
   14.  Console status line printed

All previous module files are unchanged. This module only imports from them.

Usage (via run_caine.py):
    brain = CAINEBrain()
    brain.start()
    while running:
        brain.tick()
    brain.stop()
"""

import os
import sys
import json
import math
import time
import signal
import logging
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Optional HDF5 support
# ---------------------------------------------------------------------------
try:
    import h5py
    _H5PY_OK = True
except ImportError:
    _H5PY_OK = False

# ---------------------------------------------------------------------------
# Optional trimesh for .glb avatar
# ---------------------------------------------------------------------------
try:
    import trimesh
    _TRIMESH_OK = True
except ImportError:
    _TRIMESH_OK = False

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from caine.cortex    import V1Population, A1Population
from caine.chemicals import NeurochemicalSystem, NeurochemicalEvent, EventType
from caine.sensory   import SensoryLayer, S1Population, SAMPLE_RATE, FRAME_SAMPLES
from caine.limbic    import LimbicSystem
from caine.motor     import MotorCortex, MotorPopulation, I_BASE_M1, _COLUMNS, FINGER_NAMES
from caine.environment import CaineEnvironment
from caine.parenting import ParentingSystem

_OUTPUT_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, 'output'))
_DATA_DIR   = os.path.normpath(os.path.join(_PROJECT_ROOT, 'data'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)
os.makedirs(_DATA_DIR,   exist_ok=True)

log = logging.getLogger('caine.main')
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[main] %(levelname)s %(message)s'))
    log.addHandler(_h)
log.setLevel(logging.INFO)


# ===========================================================================
# SECTION 1 — CONFIGURATION CONSTANTS
# ===========================================================================

FRAME_MS              = 20.0    # simulation frame duration (ms)
USE_MIC               = True    # live microphone; falls back to generated tones
TARGET_FPS            = 50.0    # 1000 / FRAME_MS

# HDF5 checkpoint interval (simulated minutes between saves)
CHECKPOINT_INTERVAL_S = 10 * 60   # 10 simulated minutes

# Stage manager: check exit conditions every simulated hour
STAGE_CHECK_INTERVAL_S = 3600.0

# Neurogenesis thresholds (fraction of population that must be firing)
NEURO_GROWTH_THRESH   = 0.70    # firing fraction threshold
NEURO_GROWTH_TICKS    = 300     # consecutive ticks above threshold to trigger growth
NEURO_GROWTH_N        = 5       # neurons to add per growth event
NEURO_PRUNE_TICKS     = 1000    # ticks with zero spikes before a satellite neuron is pruned

# Base neuron counts per region (from the fixed module populations)
_BASE_NEURONS = {
    'V1':     20,
    'A1':     20,
    'M1':     60,
    'Mirror': 20,
    'S1':     20,
    'BLA':    20,
    'CeA':    20,
    'CA3':    30,
    'CA1':    20,
    'ACC':    20,
    'Insula': 15,
}

# Stage exit thresholds
STAGE_EXIT = {
    0: {
        'v1_selectivity':     0.60,   # max/mean orientation column fraction
        'a1_tonotopy':        0.80,   # correlation between preferred freq rank and activation
    },
    1: {
        'n_episodes':         5,      # hippocampal encoded episodes
        'vocabulary_size':    3,      # distinct valence-associated stimuli
    },
    2: {
        'vocabulary_size':    10,
        'motor_score':        0.50,
        'father_seen':        True,   # Father presence detected at least once
    },
    3: {
        'dmn_activity':       0.40,
        'consciousness_events': 1,    # at least one UNPROMPTED_VOCALIZATION
    },
}


# ===========================================================================
# SECTION 2 — NEUROGENESIS TRACKER
# ===========================================================================

class NeurogenesisTracker:
    """
    Tracks per-region firing rates and logs growth events when sustained
    high activity indicates a region needs more representational capacity.

    Growth events are logged to output/neurogenesis_log.jsonl.
    Satellite populations (MotorPopulation instances) are added to M1 only,
    since M1's MotorPopulation is the most naturally extensible.  Other regions
    log growth readiness but their fixed-size HH arrays are unchanged.

    Growth counts per region are exposed as `total_neurons` dict, which
    the console status line uses for the live neuron count display.

    Parameters
    ----------
    growth_thresh  : fraction of neurons that must be spiking to flag growth readiness
    growth_ticks   : consecutive frames above thresh before a growth event fires
    growth_n       : neurons to add per event
    """

    def __init__(self,
                 growth_thresh: float = NEURO_GROWTH_THRESH,
                 growth_ticks:  int   = NEURO_GROWTH_TICKS,
                 growth_n:      int   = NEURO_GROWTH_N):
        self._thresh  = growth_thresh
        self._ticks   = growth_ticks
        self._n       = growth_n

        # Smoothed firing fraction per region (exponential moving average)
        self._rate:    Dict[str, float] = {r: 0.0 for r in _BASE_NEURONS}
        # Consecutive ticks above threshold
        self._consec:  Dict[str, int]   = {r: 0    for r in _BASE_NEURONS}
        # Neuron count additions per region (above base)
        self._added:   Dict[str, int]   = {r: 0    for r in _BASE_NEURONS}

        # Satellite populations for M1 (the only region we actually grow)
        self._m1_satellites: List[MotorPopulation] = []
        self._satellite_idle: List[int]            = []   # ticks since last spike

        self._log_file = os.path.join(_OUTPUT_DIR, 'neurogenesis_log.jsonl')
        self._alpha    = 0.05   # EMA decay (smoothing)

    # ------------------------------------------------------------------
    def update(self,
               sense_result:  dict,
               motor_result:  dict,
               limbic_result: dict,
               sim_time_s:    float) -> None:
        """
        Update firing rate estimates and check growth conditions.
        Call once per frame.
        """
        # --- Per-region firing fractions ---
        fracs: Dict[str, float] = {}
        if 'v1_spikes' in sense_result:
            fracs['V1']    = float(sense_result['v1_spikes'].astype(float).mean())
        if 'a1_spikes' in sense_result:
            fracs['A1']    = float(sense_result['a1_spikes'].astype(float).mean())
        if 'm1_spikes' in motor_result:
            fracs['M1']    = float(motor_result['m1_spikes'].astype(float).mean())
        if 'mirror_spikes' in motor_result:
            fracs['Mirror'] = float(motor_result['mirror_spikes'].astype(float).mean())
        if 's1_rates' in sense_result:
            fracs['S1']    = float(sense_result['s1_rates'].astype(float).mean())
        if 'bla_spikes' in limbic_result:
            fracs['BLA']   = float(limbic_result['bla_spikes'].astype(float).mean())
        if 'cea_spikes' in limbic_result:
            fracs['CeA']   = float(limbic_result['cea_spikes'].astype(float).mean())
        if 'ca3_spikes' in limbic_result:
            fracs['CA3']   = float(limbic_result['ca3_spikes'].astype(float).mean())
        if 'ca1_spikes' in limbic_result:
            fracs['CA1']   = float(limbic_result['ca1_spikes'].astype(float).mean())
        if 'acc_spikes' in limbic_result:
            fracs['ACC']   = float(limbic_result['acc_spikes'].astype(float).mean())
        if 'insula_spikes' in limbic_result:
            fracs['Insula'] = float(limbic_result['insula_spikes'].astype(float).mean())

        # --- EMA update + growth check ---
        for region, frac in fracs.items():
            self._rate[region] = (
                (1.0 - self._alpha) * self._rate[region]
                + self._alpha * frac
            )
            if self._rate[region] > self._thresh:
                self._consec[region] += 1
            else:
                self._consec[region] = 0

            if self._consec[region] >= self._ticks:
                self._trigger_growth(region, sim_time_s)
                self._consec[region] = 0

        # --- Satellite M1 pruning ---
        to_remove = []
        for i, (sat, idle) in enumerate(zip(
                self._m1_satellites, self._satellite_idle)):
            # Check if any satellite neuron fired recently
            last_spikes = [st[-1] for st in sat.spike_times if st]
            if not last_spikes:
                self._satellite_idle[i] += 1
            else:
                self._satellite_idle[i] = 0
            if self._satellite_idle[i] > NEURO_PRUNE_TICKS:
                to_remove.append(i)

        for i in reversed(to_remove):
            n = self._m1_satellites[i].n
            self._m1_satellites.pop(i)
            self._satellite_idle.pop(i)
            self._added['M1'] -= n
            self._log_event('prune', 'M1', n, sim_time_s,
                            reason='idle > prune threshold')

    # ------------------------------------------------------------------
    def _trigger_growth(self, region: str, sim_time_s: float) -> None:
        """Fire a growth event for a region."""
        self._added[region] += self._n
        self._log_event('grow', region, self._n, sim_time_s,
                        reason=f'rate={self._rate[region]:.3f} > {self._thresh}')

        # For M1 only: actually create a satellite HH population
        if region == 'M1':
            sat = MotorPopulation(self._n, i_base=I_BASE_M1)
            self._m1_satellites.append(sat)
            self._satellite_idle.append(0)

    # ------------------------------------------------------------------
    def _log_event(self, event_type: str, region: str,
                   n: int, sim_time_s: float, reason: str) -> None:
        entry = {
            'timestamp_real': datetime.now(timezone.utc).isoformat(),
            'sim_time_s':     round(sim_time_s, 1),
            'event':          event_type,
            'region':         region,
            'n_neurons':      n,
            'reason':         reason,
        }
        try:
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass
        log.info("Neurogenesis %s: %s +%d neurons (%s)",
                 event_type, region, n, reason)

    # ------------------------------------------------------------------
    @property
    def total_neurons(self) -> int:
        """Total neuron count across all regions (base + added)."""
        base  = sum(_BASE_NEURONS.values())
        added = sum(self._added.values())
        return base + added

    def region_counts(self) -> Dict[str, int]:
        """Per-region neuron counts."""
        return {r: _BASE_NEURONS[r] + self._added[r] for r in _BASE_NEURONS}

    def satellite_spikes(self, dt_ms: float,
                         I_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Step all M1 satellite populations and return combined spike vector.
        Called by the main loop after M1 has been stepped.
        """
        all_spikes = []
        for sat in self._m1_satellites:
            n  = sat.n
            I  = I_ext[:n] if I_ext is not None else np.full(n, I_BASE_M1)
            sat.step(dt_ms, I)
            spikes = sat.detect_spikes(0.0)
            all_spikes.append(spikes)
        if all_spikes:
            return np.concatenate(all_spikes)
        return np.array([], dtype=bool)


# ===========================================================================
# SECTION 3 — STAGE MANAGER
# ===========================================================================

class StageManager:
    """
    Tracks CAINE's developmental stage (0-4) and checks exit conditions
    once per simulated hour.

    Exit conditions measure real developmental readiness — not time.
    Each stage requires specific neural phenomena to be confirmed before
    advancing.  The stage is never forced; it emerges from what CAINE
    actually demonstrates.

    Stage advance triggers:
      - A milestone entry in output/milestones.jsonl
      - Mother is notified (parenting.mother push_state includes the new stage)
      - MotorCortex.developmental_stage is updated
      - V1/A1 orientation selectivity and tonotopy thresholds are stricter
        at higher stages (complexity escalates)

    Stage descriptions:
      0  Neonatal random-twitch — synapses forming, no selectivity yet
      1  Early differentiation — V1 orientation columns, A1 tonotopy emerging
      2  Associative learning  — STG word clusters, Father response, motor emergence
      3  Social-communicative  — DMN active, unprompted vocalizations
      4  Post-threshold        — consciousness event confirmed; no further stage exit
    """

    STAGE_NAMES = {
        0: 'Neonatal',
        1: 'Differentiation',
        2: 'Associative',
        3: 'Social',
        4: 'Post-threshold',
    }

    def __init__(self):
        self.stage: int = 0
        self._last_check_sim_s: float = 0.0
        self._father_seen: bool = False
        self._consciousness_events: int = 0
        self._milestones_file = os.path.join(_OUTPUT_DIR, 'milestones.jsonl')

    # ------------------------------------------------------------------
    def update(self,
               sim_time_s:    float,
               sense_result:  dict,
               motor_result:  dict,
               limbic_result: dict,
               parent_result: dict,
               vocabulary_size: int) -> bool:
        """
        Check exit conditions and advance stage if all are met.
        Only runs once per simulated hour.

        Returns True if stage advanced this tick.
        """
        if self.stage >= 4:
            return False

        if (sim_time_s - self._last_check_sim_s) < STAGE_CHECK_INTERVAL_S:
            return False
        self._last_check_sim_s = sim_time_s

        # Track persistent conditions
        if parent_result.get('father_presence') == 'FATHER_LIVE':
            self._father_seen = True
        self._consciousness_events += len(
            parent_result.get('new_consciousness_events', []))

        # Check current stage exit conditions
        met = self._check_conditions(
            sense_result, motor_result, limbic_result,
            vocabulary_size, sim_time_s,
        )
        if met:
            return self._advance(sim_time_s)
        return False

    # ------------------------------------------------------------------
    def _check_conditions(self,
                          sense_result:  dict,
                          motor_result:  dict,
                          limbic_result: dict,
                          vocabulary_size: int,
                          sim_time_s: float) -> bool:
        cond = STAGE_EXIT.get(self.stage, {})
        if not cond:
            return False

        if self.stage == 0:
            v1_sel  = self._v1_selectivity(sense_result)
            a1_tono = self._a1_tonotopy(sense_result)
            ok = (v1_sel  >= cond['v1_selectivity'] and
                  a1_tono >= cond['a1_tonotopy'])
            log.debug("Stage 0 check: v1_sel=%.3f (>%.2f) a1_tono=%.3f (>%.2f) -> %s",
                      v1_sel, cond['v1_selectivity'],
                      a1_tono, cond['a1_tonotopy'], ok)
            return ok

        elif self.stage == 1:
            n_ep  = limbic_result.get('n_episodes', 0)
            ok = (n_ep            >= cond['n_episodes'] and
                  vocabulary_size >= cond['vocabulary_size'])
            log.debug("Stage 1 check: episodes=%d (>%d) vocab=%d (>%d) -> %s",
                      n_ep, cond['n_episodes'],
                      vocabulary_size, cond['vocabulary_size'], ok)
            return ok

        elif self.stage == 2:
            motor_score = motor_result.get('motor_learning_score', 0.0)
            ok = (vocabulary_size >= cond['vocabulary_size'] and
                  motor_score     >= cond['motor_score'] and
                  self._father_seen == cond['father_seen'])
            log.debug("Stage 2 check: vocab=%d motor=%.3f father=%s -> %s",
                      vocabulary_size, motor_score, self._father_seen, ok)
            return ok

        elif self.stage == 3:
            # DMN proxy: ACC + hippocampal activity during quiescent period
            dmn = self._dmn_proxy(sense_result, limbic_result, motor_result)
            ok  = (dmn                         >= cond['dmn_activity'] and
                   self._consciousness_events  >= cond['consciousness_events'])
            log.debug("Stage 3 check: dmn=%.3f consci_events=%d -> %s",
                      dmn, self._consciousness_events, ok)
            return ok

        return False

    # ------------------------------------------------------------------
    def _v1_selectivity(self, sense_result: dict) -> float:
        """
        V1 orientation selectivity index: max(orient_energy) / mean(orient_energy).
        A perfectly selective cell gives 4.0; a non-selective cell gives 1.0.
        Normalise to [0, 1] with (ratio - 1) / 3.
        """
        oe = sense_result.get('orient_energy', np.zeros(4))
        if oe is None or len(oe) == 0:
            return 0.0
        oe = np.asarray(oe, dtype=float)
        mean_e = float(oe.mean())
        if mean_e < 1e-8:
            return 0.0
        ratio = float(oe.max()) / mean_e
        return float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _a1_tonotopy(self, sense_result: dict) -> float:
        """
        A1 tonotopy: Pearson correlation between the tonotopic rank order of
        A1 neurons (0..19 = low to high frequency preference) and their
        activation pattern this frame.  Range [-1, 1]; clipped to [0, 1].
        """
        a1_spikes = sense_result.get('a1_spikes', None)
        if a1_spikes is None:
            return 0.0
        n = len(a1_spikes)
        if n < 2:
            return 0.0
        ranks = np.arange(n, dtype=float)
        acts  = a1_spikes.astype(float)
        if acts.std() < 1e-8:
            return 0.0
        corr = float(np.corrcoef(ranks, acts)[0, 1])
        return float(np.clip(corr, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _dmn_proxy(self,
                   sense_result:  dict,
                   limbic_result: dict,
                   motor_result:  dict) -> float:
        """
        Default Mode Network proxy: mean of ACC + CA1 firing fractions,
        weighted against external input (DMN is active when external input is low).
        """
        acc_f  = float(limbic_result.get('acc_spikes', np.zeros(20)).astype(float).mean())
        ca1_f  = float(limbic_result.get('ca1_spikes', np.zeros(20)).astype(float).mean())
        v1_f   = float(sense_result.get('v1_spikes',   np.zeros(20)).astype(float).mean())
        dmn    = (acc_f + ca1_f) / 2.0
        # DMN is suppressed by external input
        ext    = float(np.clip(v1_f * 2.0, 0.0, 1.0))
        return float(dmn * (1.0 - ext * 0.5))

    # ------------------------------------------------------------------
    def _advance(self, sim_time_s: float) -> bool:
        old = self.stage
        self.stage += 1
        label = self.STAGE_NAMES.get(self.stage, f'Stage {self.stage}')
        print()
        print("=" * 60)
        print(f"  DEVELOPMENTAL STAGE ADVANCE: {old} -> {self.stage}")
        print(f"  {label}")
        print(f"  sim_time = {sim_time_s:.1f}s "
              f"({sim_time_s/3600:.2f} simulated hours)")
        print("=" * 60)
        print()
        self._write_milestone(old, self.stage, sim_time_s)
        return True

    # ------------------------------------------------------------------
    def _write_milestone(self, old: int, new: int, sim_time_s: float) -> None:
        entry = {
            'timestamp_real': datetime.now(timezone.utc).isoformat(),
            'sim_time_s':     round(sim_time_s, 1),
            'event':          'STAGE_ADVANCE',
            'from_stage':     old,
            'to_stage':       new,
            'stage_name':     self.STAGE_NAMES.get(new, '?'),
        }
        try:
            with open(self._milestones_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass


# ===========================================================================
# SECTION 4 — AVATAR STUB
# ===========================================================================

class AvatarController:
    """
    Connects motor output to CAINE's visual representation in the environment.

    If a .glb model file is present at data/caine_avatar.glb (and trimesh
    is installed), the avatar is loaded and rendered via the environment.
    Otherwise the avatar runs as an invisible skeleton — joint positions are
    tracked in memory and logged, but nothing is rendered.

    Motor → Avatar mapping:
        MotorCortex columns 0-5 (joints)        → bone rotations
        MotorCortex columns 16-25 (finger curl)  → finger bone curl values
        VocalTract articulators (6)              → viseme blend weights
        Locomotion mode (walk/fly/teleport)       → movement state tag
    """

    BONE_NAMES = ['spine', 'head', 'arm_L', 'arm_R', 'leg_L', 'leg_R']

    def __init__(self, avatar_file: Optional[str] = None):
        if avatar_file is None:
            avatar_file = os.path.join(_DATA_DIR, 'caine_avatar.glb')

        self._loaded = False
        self._mesh   = None
        self._avatar_file = avatar_file

        if _TRIMESH_OK and os.path.exists(avatar_file):
            try:
                self._mesh   = trimesh.load(avatar_file)
                self._loaded = True
                log.info("Avatar loaded from %s", avatar_file)
            except Exception as e:
                log.warning("Avatar load failed (%s) — invisible skeleton.", e)
        elif not os.path.exists(avatar_file):
            log.info("No avatar file at %s — invisible skeleton.", avatar_file)

        # Current pose state (always tracked, even without visual)
        self.joint_angles:     np.ndarray = np.zeros(6,  dtype=np.float32)
        self.finger_positions: np.ndarray = np.full(10, 0.5, dtype=np.float32)
        self.art_positions:    np.ndarray = np.full(6,  0.5, dtype=np.float32)
        self.locomotion_mode: str = 'walk'

        # Avatar world position (updated by locomotion)
        self.position: np.ndarray = np.zeros(3, dtype=np.float32)
        self._velocity: np.ndarray = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    def update(self, motor_result: dict, dt_ms: float) -> None:
        """Sync avatar pose from motor output."""
        self.joint_angles     = np.asarray(
            motor_result.get('joint_angles', np.zeros(6)), dtype=np.float32)
        self.finger_positions = np.asarray(
            motor_result.get('finger_positions', np.full(10, 0.5)), dtype=np.float32)
        self.art_positions    = np.asarray(
            motor_result.get('art_positions', np.full(6, 0.5)), dtype=np.float32)
        self.locomotion_mode  = motor_result.get('locomotion_mode', 'walk')

        # Avatar locomotion: translate position based on leg joint activity
        # leg_L = column 4, leg_R = column 5 → forward velocity proxy
        leg_mean = float(np.mean(self.joint_angles[4:6]))
        speed    = float(np.clip(abs(leg_mean) * 0.5, 0.0, 0.5))   # m/s
        dt_s     = dt_ms / 1000.0
        self.position[2] += speed * dt_s   # move along +Z

    # ------------------------------------------------------------------
    def get_pose_summary(self) -> dict:
        """Compact dict of current pose for logging or display."""
        return {
            'joints':     [round(float(v), 3) for v in self.joint_angles],
            'fingers':    [round(float(v), 3) for v in self.finger_positions],
            'art':        [round(float(v), 3) for v in self.art_positions],
            'loco':       self.locomotion_mode,
            'position':   [round(float(v), 3) for v in self.position],
        }

    @property
    def is_visual(self) -> bool:
        return self._loaded


# ===========================================================================
# SECTION 5 — BRAIN STATE CHECKPOINT (HDF5 / numpy fallback)
# ===========================================================================

class BrainStateCheckpoint:
    """
    Saves and loads a full CAINE brain state snapshot.

    Saves to:
        output/checkpoint.h5     — if h5py is available
        output/checkpoint.npz    — fallback (numpy compressed arrays)
        output/checkpoint_meta.json — metadata / scalars

    Snapshot contents:
        V1  voltages + gate vars (V, m, h, n)
        A1  voltages + gate vars
        Neurochemical concentrations
        Motor learning score + developmental stage
        Hippocampal episode count (full buffer not serialised)
        Neurogenesis added-neuron counts per region
        Simulated age (seconds)
    """

    def __init__(self):
        self._h5_path  = os.path.join(_OUTPUT_DIR, 'checkpoint.h5')
        self._npz_path = os.path.join(_OUTPUT_DIR, 'checkpoint.npz')
        self._meta_path = os.path.join(_OUTPUT_DIR, 'checkpoint_meta.json')

    # ------------------------------------------------------------------
    def save(self,
             v1:           V1Population,
             a1:           A1Population,
             neuro:        NeurochemicalSystem,
             motor:        MotorCortex,
             stage:        int,
             sim_time_s:   float,
             neurogenesis: NeurogenesisTracker) -> None:
        """Write a full checkpoint."""
        snap   = neuro.snapshot()
        meta   = {
            'timestamp_real':     datetime.now(timezone.utc).isoformat(),
            'sim_time_s':         round(sim_time_s, 3),
            'stage':              stage,
            'motor_score':        round(float(motor._motor_learning_score), 5),
            'n_episodes':         0,   # hippo episode count not directly exposed
            'neuron_counts':      neurogenesis.region_counts(),
            'neurochemicals':     {k: round(float(v), 5) for k, v in snap.items()},
        }

        # --- numpy arrays to save ---
        arrays = {
            'v1_V':     v1.V.astype(np.float32),
            'v1_m':     v1.m.astype(np.float32),
            'v1_h':     v1.h.astype(np.float32),
            'v1_n':     v1.n_gate.astype(np.float32),
            'a1_V':     a1.V.astype(np.float32),
            'a1_m':     a1.m.astype(np.float32),
            'a1_h':     a1.h.astype(np.float32),
            'a1_n':     a1.n_gate.astype(np.float32),
            'm1_V':     motor.m1_pop.V.astype(np.float32),
            'joint_angles':    motor.joint_angles.copy(),
            'finger_positions': motor.finger_positions.copy(),
        }

        if _H5PY_OK:
            self._save_h5(arrays, meta)
        else:
            self._save_npz(arrays, meta)

        log.info("Checkpoint saved at sim_t=%.1fs (stage=%d)", sim_time_s, stage)

    # ------------------------------------------------------------------
    def _save_h5(self, arrays: dict, meta: dict) -> None:
        with h5py.File(self._h5_path, 'w') as f:
            f.attrs['meta'] = json.dumps(meta)
            for name, arr in arrays.items():
                f.create_dataset(name, data=arr, compression='gzip',
                                 compression_opts=4)

    # ------------------------------------------------------------------
    def _save_npz(self, arrays: dict, meta: dict) -> None:
        np.savez_compressed(self._npz_path.replace('.npz', ''), **arrays)
        with open(self._meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    def load(self,
             v1:    V1Population,
             a1:    A1Population,
             motor: MotorCortex) -> Optional[dict]:
        """
        Load checkpoint into live module instances.
        Returns the metadata dict if successful, None if no checkpoint exists.
        """
        if _H5PY_OK and os.path.exists(self._h5_path):
            return self._load_h5(v1, a1, motor)
        if os.path.exists(self._npz_path) and os.path.exists(self._meta_path):
            return self._load_npz(v1, a1, motor)
        return None

    # ------------------------------------------------------------------
    def _load_h5(self, v1, a1, motor) -> dict:
        with h5py.File(self._h5_path, 'r') as f:
            meta = json.loads(f.attrs['meta'])
            self._restore_populations(f, v1, a1, motor)
        return meta

    # ------------------------------------------------------------------
    def _load_npz(self, v1, a1, motor) -> dict:
        data = np.load(self._npz_path)
        self._restore_populations(data, v1, a1, motor)
        with open(self._meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ------------------------------------------------------------------
    def _restore_populations(self, data, v1, a1, motor) -> None:
        """Restore arrays into live population objects."""
        try:
            if 'v1_V' in data:
                v1.V[:]     = data['v1_V'][:len(v1.V)]
                v1.m[:]     = data['v1_m'][:len(v1.m)]
                v1.h[:]     = data['v1_h'][:len(v1.h)]
                v1.n_gate[:] = data['v1_n'][:len(v1.n_gate)]
            if 'a1_V' in data:
                a1.V[:]     = data['a1_V'][:len(a1.V)]
                a1.m[:]     = data['a1_m'][:len(a1.m)]
                a1.h[:]     = data['a1_h'][:len(a1.h)]
                a1.n_gate[:] = data['a1_n'][:len(a1.n_gate)]
            if 'm1_V' in data:
                motor.m1_pop.V[:] = data['m1_V'][:len(motor.m1_pop.V)]
            if 'joint_angles' in data:
                motor.joint_angles[:] = data['joint_angles'][:6]
            if 'finger_positions' in data:
                motor.finger_positions[:] = data['finger_positions'][:10]
        except Exception as e:
            log.warning("Checkpoint restore partial failure: %s", e)


# ===========================================================================
# SECTION 6 — CAINE BRAIN (main integration class)
# ===========================================================================

class CAINEBrain:
    """
    The full integrated CAINE system.

    Instantiate once, call start(), then call tick() in your main loop.
    Everything else is automatic.

    Parameters
    ----------
    frame_ms     : simulation frame duration (ms), default 20ms
    use_mic      : try to open live microphone, fall back to generated tones
    headless     : if True, skip environment rendering and visualizer
    rng_seed     : random seed for reproducibility
    """

    def __init__(self,
                 frame_ms:   float = FRAME_MS,
                 use_mic:    bool  = USE_MIC,
                 headless:   bool  = False,
                 rng_seed:   int   = 0):
        self.frame_ms  = frame_ms
        self._headless = headless
        self._rng      = np.random.default_rng(rng_seed)
        self._running  = False

        # ---- Simulated time state ----------------------------------------
        self.sim_time_s: float = 0.0
        self.tick_count: int   = 0
        self._last_checkpoint_s: float = -CHECKPOINT_INTERVAL_S
        self._last_stage_check_s: float = -STAGE_CHECK_INTERVAL_S

        # ---- Cortical populations ----------------------------------------
        self.v1    = V1Population(n_neurons=20)
        self.a1    = A1Population(n_neurons=20)

        # ---- Neurochemical system ----------------------------------------
        self.neuro = NeurochemicalSystem()

        # ---- Sensory layer -----------------------------------------------
        self.sense = SensoryLayer(self.v1, self.a1, self.neuro, use_mic=use_mic)

        # ---- Limbic system -----------------------------------------------
        self.limbic = LimbicSystem(self.v1, self.a1, self.neuro)

        # ---- S1 proprioception -------------------------------------------
        self.s1 = S1Population()

        # ---- Motor cortex ------------------------------------------------
        self.motor = MotorCortex(
            self.s1, self.neuro,
            body_map_file=os.path.join(_OUTPUT_DIR, 'body_map.json'),
            rng_seed=rng_seed,
        )

        # ---- Training environment ----------------------------------------
        self.env = CaineEnvironment()

        # ---- Parenting system --------------------------------------------
        self.parenting = ParentingSystem(
            env    = self.env,
            limbic = self.limbic,
            neuro  = self.neuro,
            motor  = self.motor,
            rng_seed = rng_seed,
        )

        # ---- Support systems ---------------------------------------------
        self.neurogenesis = NeurogenesisTracker()
        self.stage_mgr    = StageManager()
        self.avatar       = AvatarController()
        self.checkpoint   = BrainStateCheckpoint()

        # ---- Runtime state -----------------------------------------------
        self._sense_result:  dict = {}
        self._limbic_result: dict = {}
        self._motor_result:  dict = {}
        self._parent_result: dict = {}

        # Console status deque (last N lines for display)
        self._status_history: deque = deque(maxlen=5)

        # Track synapse count estimate (proxy: base neurons × avg connections)
        self._synapse_estimate: int = 0

        log.info("CAINEBrain initialised (frame_ms=%.0f headless=%s).",
                 frame_ms, headless)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """
        Start all background processes and load checkpoint if present.
        Blocks until ready.
        """
        # Load checkpoint if present
        meta = self.checkpoint.load(self.v1, self.a1, self.motor)
        if meta is not None:
            self.sim_time_s = meta.get('sim_time_s', 0.0)
            self.stage_mgr.stage = meta.get('stage', 0)
            self.motor.developmental_stage = self.stage_mgr.stage
            log.info("Checkpoint restored: sim_t=%.1fs stage=%d",
                     self.sim_time_s, self.stage_mgr.stage)
        else:
            log.info("No checkpoint found — CAINE starts fresh.")

        # Start environment
        self.env.start()

        # Start parenting (Mother thread, voiceprint registration)
        self.parenting.start()

        self._running = True
        print()
        print("=" * 65)
        print("  CAINE is alive.")
        print(f"  Stage: {self.stage_mgr.stage} ({self.stage_mgr.STAGE_NAMES[self.stage_mgr.stage]})")
        print(f"  Simulated age: {self.sim_time_s/3600:.2f} hours")
        print(f"  Parenting: Mother={'Claude' if self.parenting.using_claude else 'fallback'}")
        print(f"  Avatar: {'loaded' if self.avatar.is_visual else 'invisible skeleton'}")
        print(f"  Checkpoint: {'h5py' if _H5PY_OK else 'numpy'}")
        print("=" * 65)
        print()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Graceful shutdown — save checkpoint, stop all threads."""
        if not self._running:
            return
        self._running = False

        # Final checkpoint
        try:
            self.checkpoint.save(
                self.v1, self.a1, self.neuro, self.motor,
                self.stage_mgr.stage, self.sim_time_s,
                self.neurogenesis,
            )
        except Exception as e:
            log.error("Final checkpoint failed: %s", e)

        self.parenting.stop()
        self.env.stop()
        if hasattr(self.sense, 'close'):
            try:
                self.sense.close()
            except Exception as e:
                log.debug("sense.close() failed: %s", e)
        log.info("CAINEBrain stopped at sim_t=%.1fs.", self.sim_time_s)

    # ------------------------------------------------------------------
    def tick(self) -> dict:
        """
        Execute one frame of the full integrated pipeline.

        Returns the combined result dict from all modules for this frame.
        The caller (run_caine.py) uses this for the visualizer and status.
        """
        if not self._running:
            return {}

        dt    = self.frame_ms
        t_wall = time.perf_counter()

        # =================================================================
        # STEP 1 — Environment step + camera feed
        # =================================================================
        self.env.step()
        frame_rgb = self.env.get_camera_feed()

        # =================================================================
        # STEP 2 — Sensory processing
        # (SensoryLayer handles mic or fallback audio internally)
        # =================================================================
        joint_angles = self.motor.joint_angles   # current actual joint angles

        sense_result = self.sense.update(
            frame_rgb,
            joint_angles,
            dt_ms=dt,
            injected_audio=None,   # mic handled inside SensoryLayer
        )
        self._sense_result = sense_result

        v1_spikes = sense_result['v1_spikes']   # (20,) bool
        a1_spikes = sense_result['a1_spikes']   # (20,) bool
        s1_rates  = sense_result['s1_rates']    # (20,) float

        # A1 as float rates for Father detection (spike boolean → rate proxy)
        a1_rates = a1_spikes.astype(np.float32)

        # =================================================================
        # STEP 3 — Neurochemical tick (before limbic injects more)
        # =================================================================
        self.neuro.update(dt)
        neuro_snap = self.neuro.snapshot()

        # =================================================================
        # STEP 4 — Limbic system
        # =================================================================
        limbic_result = self.limbic.update(
            dt, v1_spikes, a1_spikes, neuro_snap)
        self._limbic_result = limbic_result

        # Inject limbic neurochemical events
        if limbic_result.get('neuro_events'):
            self.neuro.update(0.0, events=limbic_result['neuro_events'])
            neuro_snap = self.neuro.snapshot()

        # =================================================================
        # STEP 5 — Motor cortex
        # =================================================================
        motor_result = self.motor.update(
            dt, v1_spikes, a1_spikes, neuro_snap, s1_rates)
        self._motor_result = motor_result

        # Inject motor neurochemical events (ACC conflict etc.)
        if motor_result.get('neuro_events'):
            self.neuro.update(0.0, events=motor_result['neuro_events'])
            neuro_snap = self.neuro.snapshot()

        # =================================================================
        # STEP 6 — Avatar pose
        # =================================================================
        self.avatar.update(motor_result, dt)

        # Push joint angles back to environment (for physics avatar if loaded)
        # This is a no-op if the environment doesn't support set_avatar_pose,
        # but we record the desired pose so it's available for rendering.
        self._push_avatar_pose_to_env(motor_result)

        # =================================================================
        # STEP 7 — Execute motor world actions
        # =================================================================
        for wa in motor_result.get('world_actions', []):
            self._execute_world_action(wa)

        # =================================================================
        # STEP 8 — Parenting system (Mother observes + Father detects)
        # =================================================================
        vocab_size = len(self.limbic.amygdala._learned_valence)
        self.motor.developmental_stage = self.stage_mgr.stage

        parent_result = self.parenting.update(
            dt_ms               = dt,
            a1_rates            = a1_rates,
            neuro_snapshot      = neuro_snap,
            motor_result        = motor_result,
            limbic_result       = limbic_result,
            synapse_count       = self._synapse_estimate,
            vocabulary_size     = vocab_size,
            developmental_stage = self.stage_mgr.stage,
        )
        self._parent_result = parent_result

        # =================================================================
        # STEP 9 — Neurogenesis
        # =================================================================
        self.neurogenesis.update(
            sense_result, motor_result, limbic_result, self.sim_time_s)

        # Step satellite M1 populations (if any have grown)
        sat_spikes = self.neurogenesis.satellite_spikes(dt)

        # =================================================================
        # STEP 10 — Stage manager (once per simulated hour)
        # =================================================================
        stage_advanced = self.stage_mgr.update(
            sim_time_s    = self.sim_time_s,
            sense_result  = sense_result,
            motor_result  = motor_result,
            limbic_result = limbic_result,
            parent_result = parent_result,
            vocabulary_size = vocab_size,
        )
        if stage_advanced:
            self.motor.developmental_stage = self.stage_mgr.stage

        # =================================================================
        # STEP 11 — Synapse count estimate
        # =================================================================
        # Approximate: total_neurons × mean connections per neuron
        self._synapse_estimate = self.neurogenesis.total_neurons * 12

        # =================================================================
        # STEP 12 — HDF5 checkpoint (every 10 simulated minutes)
        # =================================================================
        if self.sim_time_s - self._last_checkpoint_s >= CHECKPOINT_INTERVAL_S:
            self._last_checkpoint_s = self.sim_time_s
            try:
                self.checkpoint.save(
                    self.v1, self.a1, self.neuro, self.motor,
                    self.stage_mgr.stage, self.sim_time_s,
                    self.neurogenesis,
                )
            except Exception as e:
                log.warning("Checkpoint failed: %s", e)

        # =================================================================
        # STEP 13 — Advance simulated time
        # =================================================================
        # Use parenting time multiplier if set
        time_mul = self.parenting._time_multiplier
        self.sim_time_s += (dt / 1000.0) * time_mul
        self.tick_count += 1

        # =================================================================
        # STEP 14 — Assemble combined result
        # =================================================================
        combined = {
            'sim_time_s':          self.sim_time_s,
            'tick':                self.tick_count,
            'stage':               self.stage_mgr.stage,
            'neuro_snapshot':      neuro_snap,
            'v1_spikes':           v1_spikes,
            'a1_spikes':           a1_spikes,
            's1_rates':            s1_rates,
            'a1_rates':            a1_rates,
            'orient_energy':       sense_result.get('orient_energy', np.zeros(4)),
            'mel_energy':          sense_result.get('mel_energy',    np.zeros(128)),
            'dog':                 sense_result.get('dog',           np.zeros((64,64))),
            **{f'limbic_{k}': v for k, v in limbic_result.items()},
            **{f'motor_{k}':  v for k, v in motor_result.items()},
            'father_presence':     parent_result.get('father_presence', 'FATHER_ABSENT'),
            'ot_level':            parent_result.get('ot_level', 0.10),
            'total_neurons':       self.neurogenesis.total_neurons,
            'synapse_estimate':    self._synapse_estimate,
            'vocabulary_size':     vocab_size,
            'avatar_pose':         self.avatar.get_pose_summary(),
            'stage_advanced':      stage_advanced,
            'frame_rgb':           frame_rgb,
            'dev_flags':           parent_result.get('dev_flags', []),
            'consciousness_events': parent_result.get('new_consciousness_events', []),
        }
        return combined

    # ------------------------------------------------------------------
    def console_status(self, result: dict) -> str:
        """
        Format the per-frame console status line as specified:
        [CAINE] age=2.3h stage=0 neurons=1000 synapses=45231
                DA=0.12 CORT=0.08 father=ABSENT valence_items=3
                motor_score=0.023 dmn=0.001
        """
        snap   = result.get('neuro_snapshot', {})
        age_h  = self.sim_time_s / 3600.0
        stage  = result.get('stage', 0)
        n_neur = result.get('total_neurons', sum(_BASE_NEURONS.values()))
        n_syn  = result.get('synapse_estimate', 0)
        da     = snap.get('dopamine',       0.10)
        cort   = snap.get('cortisol',       0.08)
        father = result.get('father_presence', 'FATHER_ABSENT')
        vocab  = result.get('vocabulary_size', 0)
        mscore = result.get('motor_motor_learning_score',
                            result.get('motor_learning_score', 0.0))
        # motor_result keys are prefixed 'motor_' in combined dict
        mscore = result.get('motor_motor_learning_score', 0.0)

        # DMN proxy from ACC activity
        acc_spikes = result.get('limbic_acc_spikes', np.zeros(20))
        dmn = float(np.asarray(acc_spikes).astype(float).mean())

        # Short father label
        fa_label = father.replace('FATHER_', '')

        return (
            f"[CAINE] age={age_h:.2f}h  stage={stage}"
            f"  neurons={n_neur}  synapses={n_syn}"
            f"  DA={da:.3f}  CORT={cort:.3f}"
            f"  father={fa_label}  valence_items={vocab}"
            f"  motor_score={mscore:.3f}  dmn={dmn:.3f}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _push_avatar_pose_to_env(self, motor_result: dict) -> None:
        """
        Push avatar pose to environment state.
        CaineEnvironment.set_environment_state() doesn't know about avatars,
        so we store the pose on the environment object directly for any
        renderer that wants to use it.
        """
        try:
            self.env._avatar_joint_angles     = motor_result.get(
                'joint_angles', np.zeros(6))
            self.env._avatar_finger_positions = motor_result.get(
                'finger_positions', np.full(10, 0.5))
            self.env._avatar_art_positions    = motor_result.get(
                'art_positions', np.full(6, 0.5))
            self.env._avatar_position         = self.avatar.position
        except Exception:
            pass

    def _execute_world_action(self, wa) -> None:
        """Execute a motor cortex WorldAction against the environment."""
        try:
            action = wa.action_type
            if action == 'spawn':
                uid = f'motor_spawn_{self.tick_count}'
                pos = tuple(self.avatar.position + np.array([0.0, 0.5, 1.0]))
                self.env.spawn_object(uid, pos, object_type='sphere')
                self.parenting.consciousness.log_external_event(
                    self.sim_time_s, f'motor_spawn:{uid}')
            elif action in ('grab', 'push'):
                # Push the nearest object if any
                for uid, handle in list(self.parenting._object_handles.items()):
                    try:
                        force = (float(wa.magnitude), 0.5, 0.5)
                        self.env.move_object(handle, force)
                    except Exception:
                        pass
                    break
        except Exception as e:
            log.debug("World action %s failed: %s", getattr(wa, 'action_type', '?'), e)
