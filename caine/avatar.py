"""
CAINE Avatar — Module 8
========================
CAINE has a body from birth.  Not a body that appears when he is ready —
a body that is always present, that he gradually learns to inhabit.

Expression is not scripted.  It is a continuous mapping from the
neurochemical state vector to bone rotation targets, interpolated at 60 fps.

Mouth movement is not driven by a TTS system.  Viseme blend weights are
computed from the vocal tract motor output (VocalTract.areas).  The mouth
moves because CAINE is trying to produce sound.

Architecture
------------
  PoseTarget          — dataclass: all controllable bone + shape-key targets
  emotional_to_pose() — NeurochemicalSystem → PoseTarget (README eq.)
  VisemeMapper        — VocalTract.areas → 15 viseme blend weights
  HatPhysicsChain     — 3-bone spring simulation for secondary hat motion
  AvatarController    — 60 fps update loop; integrates all of the above

Rig inventory (Blender armature)
---------------------------------
  Spine:   spine_01 … spine_05
  Neck/head: neck, head
  L/R:     shoulder, upper_arm, lower_arm, hand
  Fingers: 3 bones each × 5 fingers × 2 hands  (30 bones)
  Hips:    hips
  L/R leg: upper_leg, lower_leg, foot
  Face:    jaw, eye_L, eye_R
           lid_upper_L, lid_lower_L, lid_upper_R, lid_lower_R
  Hat:     hat_tilt, hat_chain_01, hat_chain_02, hat_chain_03

Visemes (15):
  sil  PP  FF  TH  DD  kk  CH  SS  nn  RR  aa  E  ih  oh  ou

Usage
-----
    from caine.avatar import AvatarController
    from caine.chemicals import NeurochemicalSystem
    from caine.sensory  import VocalTract

    av = AvatarController()
    av.update(dt_ms=16.67, neuro=neuro, vocal_tract=vt, stg_rates=stg_rates)
    state = av.current_pose
"""

import os
import sys as _sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Constants — bone names matching the Blender armature spec
# ---------------------------------------------------------------------------

SPINE_BONES: List[str] = [
    'spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05']

NECK_HEAD_BONES: List[str] = ['neck', 'head']

ARM_BONES: List[str] = [
    'shoulder_L', 'upper_arm_L', 'lower_arm_L', 'hand_L',
    'shoulder_R', 'upper_arm_R', 'lower_arm_R', 'hand_R',
]

# 3 bones per finger × 5 fingers × 2 hands
FINGER_BONES: List[str] = [
    f'finger_{hand}_{finger}_{seg}'
    for hand in ('L', 'R')
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky')
    for seg in ('01', '02', '03')
]

LEG_BONES: List[str] = [
    'hips',
    'upper_leg_L', 'lower_leg_L', 'foot_L',
    'upper_leg_R', 'lower_leg_R', 'foot_R',
]

FACE_BONES: List[str] = [
    'jaw',
    'eye_L', 'eye_R',
    'lid_upper_L', 'lid_lower_L',
    'lid_upper_R', 'lid_lower_R',
]

HAT_BONES: List[str] = [
    'hat_tilt',
    'hat_chain_01', 'hat_chain_02', 'hat_chain_03',
]

ALL_BONES: List[str] = (
    SPINE_BONES + NECK_HEAD_BONES + ARM_BONES +
    FINGER_BONES + LEG_BONES + FACE_BONES + HAT_BONES
)

# 15 viseme shape keys (minimum per spec)
VISEMES: List[str] = [
    'sil', 'PP', 'FF', 'TH', 'DD', 'kk', 'CH',
    'SS', 'nn', 'RR', 'aa', 'E', 'ih', 'oh', 'ou',
]

# Vocal-tract tube index boundaries for region-based viseme features
# (44-tube Kelly-Lochbaum waveguide from VocalTract)
_VT_VELAR_SLICE    = slice(25, 29)   # soft palate (kk)
_VT_PALATAL_SLICE  = slice(30, 34)   # hard palate (CH, RR)
_VT_ALVEOLAR_SLICE = slice(35, 38)   # alveolar ridge (DD, SS, nn)
_VT_DENTAL_SLICE   = slice(38, 42)   # upper teeth / TH zone
_VT_LIP_SLICE      = slice(42, 44)   # lips (PP, FF)
_VT_MID_ORAL_SLICE = slice(20, 34)   # main oral cavity (vowels)
_VT_BACK_SLICE     = slice(20, 25)   # tongue back (aa low, ih high)
_VT_FRONT_SLICE    = slice(28, 33)   # tongue front height


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation; t clamped to [0, 1]."""
    return float(a) + (float(b) - float(a)) * float(np.clip(t, 0.0, 1.0))


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax."""
    x = np.asarray(x, dtype=np.float64) / max(temperature, 1e-8)
    x -= x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# PoseTarget — all controllable degrees of freedom
# ---------------------------------------------------------------------------

@dataclass
class PoseTarget:
    """
    Target values for every controllable bone and shape key on CAINE's rig.

    Bone rotation values are in normalised [-1, 1] space where the Blender
    driver maps 0 → rest pose and ±1 → maximum rotation.

    Viseme weights are in [0, 1] and should sum to <= 1.0 at any given time
    (the rig driver normalises them).

    Expression is continuous — these values are interpolated at 60 fps by
    AvatarController toward the current emotional/phonemic target.
    """

    # ---- Emotional expression (driven by neurochemicals) ----
    eye_width:    float = 0.65    # DA → pupil dilation / eye openness
    spine_curl:   float = 0.0     # CORT → forward curl (submission, stress)
    head_tilt:    float = 0.0     # ACh → curious forward tilt
    hat_tilt:     float = 0.1     # NE → hat forward=curious, back=surprised

    # ---- Gaze (driven by STG / social attention) ----
    # World-space XYZ point that both eyes converge on
    gaze_target:  Tuple[float, float, float] = (0.0, 1.0, 3.0)

    # ---- Eye & eyelid ----
    blink_left:   float = 0.0     # 0=open, 1=closed
    blink_right:  float = 0.0
    squint_left:  float = 0.0
    squint_right: float = 0.0

    # ---- Jaw / mouth ----
    jaw_open:     float = 0.0     # 0=closed, 1=fully open

    # ---- Viseme shape key blend weights ----
    # Dict[viseme_name → weight], values in [0, 1]
    viseme_weights: Dict[str, float] = field(
        default_factory=lambda: {v: 0.0 for v in VISEMES})

    def as_dict(self) -> dict:
        """Flat dict for serialisation / Blender bridge."""
        d = {
            'eye_width':    self.eye_width,
            'spine_curl':   self.spine_curl,
            'head_tilt':    self.head_tilt,
            'hat_tilt':     self.hat_tilt,
            'gaze_target':  list(self.gaze_target),
            'blink_left':   self.blink_left,
            'blink_right':  self.blink_right,
            'squint_left':  self.squint_left,
            'squint_right': self.squint_right,
            'jaw_open':     self.jaw_open,
        }
        d.update({f'viseme_{k}': v for k, v in self.viseme_weights.items()})
        return d


# ---------------------------------------------------------------------------
# emotional_to_pose  — README equation
# ---------------------------------------------------------------------------

def emotional_to_pose(neurochemicals,
                       gaze_target: Tuple[float, float, float] = (0.0, 1.0, 3.0)
                       ) -> PoseTarget:
    """
    Map the current neurochemical state to a PoseTarget.

    Implements the README equation exactly:

        eye_width  = lerp(0.3,  1.0,  dopamine)
        spine_curl = lerp(0.0, -0.3,  cortisol)
        head_tilt  = lerp(0.0,  0.2,  acetylcholine)
        hat_tilt   = lerp(-0.1, 0.3,  norepinephrine)
        gaze_target = social_attention_target()   ← passed in as gaze_target

    Additional mappings (not in README but needed for full expression):
        blink      = random occasional blink gated by acetylcholine
        squint     = lerp(0.0, 0.4, cortisol)   ← stress narrowing
        jaw_open   = 0.0  (driven separately by VisemeMapper)

    Parameters
    ----------
    neurochemicals : NeurochemicalSystem — live chemical state
    gaze_target    : (x, y, z) world-space social attention target;
                     caller should derive this from STG voice-detection results

    Returns
    -------
    PoseTarget with emotional expression fields filled; viseme_weights zeroed
    (filled in by VisemeMapper separately).
    """
    da   = float(neurochemicals.dopamine)
    cort = float(neurochemicals.cortisol)
    ach  = float(neurochemicals.acetylcholine)
    ne   = float(neurochemicals.norepinephrine)
    ot   = float(neurochemicals.oxytocin)

    return PoseTarget(
        eye_width   = lerp(0.3,   1.0,  da),
        spine_curl  = lerp(0.0,  -0.3,  cort),
        head_tilt   = lerp(0.0,   0.2,  ach),
        hat_tilt    = lerp(-0.1,  0.3,  ne),
        gaze_target = gaze_target,
        # Supplemental expression:
        squint_left  = lerp(0.0,  0.4,  cort),
        squint_right = lerp(0.0,  0.4,  cort),
        # OT softens the gaze — less intense stare, warmer presence
        blink_left   = 0.0,
        blink_right  = 0.0,
        jaw_open     = 0.0,
        viseme_weights = {v: 0.0 for v in VISEMES},
    )


# ---------------------------------------------------------------------------
# VisemeMapper — VocalTract areas → 15 viseme blend weights
# ---------------------------------------------------------------------------

class VisemeMapper:
    """
    Converts a VocalTract area array (44 tubes) to 15 viseme blend weights.

    The mapping is physiologically motivated: each viseme corresponds to a
    characteristic constriction pattern at a specific location in the vocal tract.
    Weights are derived by extracting regional area statistics and projecting
    them onto viseme prototype vectors, then normalising.

    Region-to-viseme correspondences
    ---------------------------------
    Bilabial  (PP)       — complete lip closure             (tubes 42-43)
    Labio-dental (FF)    — lip–teeth partial contact        (tubes 40-43)
    Dental    (TH)       — tongue tip at teeth              (tubes 38-41)
    Alveolar  (DD, nn)   — tongue tip at alveolar ridge     (tubes 35-38)
    Velar     (kk)       — tongue back at soft palate       (tubes 25-29)
    Palatal   (CH, RR)   — tongue body raised               (tubes 30-33)
    Sibilant  (SS)       — narrow front groove              (tubes 35-38 narrow)
    Vowels    (aa, E, ih, oh, ou) — open tract, tongue height gradient
    Silence   (sil)      — near-resting area pattern
    """

    # Default areas at rest (from VocalTract._DEFAULT_AREAS statistics)
    _REST_LIP_AREA    = 0.60   # cm² at rest
    _REST_ORAL_AREA   = 2.50   # cm² midpoint of oral cavity at rest

    def __call__(self, areas: np.ndarray) -> Dict[str, float]:
        """
        Compute viseme blend weights from a 44-element area array.

        Parameters
        ----------
        areas : (44,) float — VocalTract cross-sectional areas (cm²)

        Returns
        -------
        dict mapping each viseme name → blend weight in [0, 1]
        """
        if len(areas) != 44:
            return {v: 0.0 for v in VISEMES}

        a = np.asarray(areas, dtype=np.float64)

        # ---- Extract region features ----
        lip_area      = a[_VT_LIP_SLICE].mean()
        dental_area   = a[_VT_DENTAL_SLICE].mean()
        alveolar_area = a[_VT_ALVEOLAR_SLICE].mean()
        velar_area    = a[_VT_VELAR_SLICE].mean()
        palatal_area  = a[_VT_PALATAL_SLICE].mean()
        mid_oral_area = a[_VT_MID_ORAL_SLICE].mean()
        back_area     = a[_VT_BACK_SLICE].mean()
        front_area    = a[_VT_FRONT_SLICE].mean()

        # Constriction score: 0=open, 1=closed (inverted, normalised)
        lip_close     = float(np.clip(1.0 - lip_area / 0.8,         0.0, 1.0))
        dental_close  = float(np.clip(1.0 - dental_area / 1.2,      0.0, 1.0))
        alv_close     = float(np.clip(1.0 - alveolar_area / 1.5,    0.0, 1.0))
        velar_close   = float(np.clip(1.0 - velar_area / 1.5,       0.0, 1.0))
        palatal_raise = float(np.clip(palatal_area / 3.5,            0.0, 1.0))

        # Oral openness (0=closed, 1=fully open) — drives vowel weights
        oral_open     = float(np.clip(mid_oral_area / 3.8, 0.0, 1.0))

        # Tongue height gradient (front relative to back)
        tongue_raise  = float(np.clip((front_area - back_area) / 2.0, 0.0, 1.0))

        # Lip rounding (back-oral area relative to front)
        lip_round     = float(np.clip((back_area - front_area) / 1.5, 0.0, 1.0))

        # Silence score: all areas near rest, low acoustic energy
        silence_score = float(np.clip(
            1.0 - np.abs(a - self._REST_ORAL_AREA).mean() / 2.0, 0.0, 1.0))

        # ---- Map features to raw viseme activations ----
        raw = {
            'sil': silence_score * (1.0 - oral_open) * (1.0 - lip_close),
            'PP':  lip_close * 0.95,                         # bilabial stop
            'FF':  lip_close * 0.4 + dental_close * 0.5,    # labio-dental
            'TH':  dental_close * 0.8,                      # dental fricative
            'DD':  alv_close * 0.85,                        # alveolar stop
            'kk':  velar_close * 0.9,                       # velar stop
            'CH':  palatal_raise * 0.85,                    # palatal affricate
            'SS':  alv_close * 0.5 + dental_close * 0.3,   # sibilant fricative
            'nn':  alv_close * 0.6,                         # alveolar nasal
            'RR':  palatal_raise * 0.5 + tongue_raise * 0.4,  # retroflex
            'aa':  oral_open * (1.0 - tongue_raise) * 0.9,    # open vowel
            'E':   oral_open * 0.5 + tongue_raise * 0.4,      # mid-front vowel
            'ih':  tongue_raise * 0.85,                        # high-front vowel
            'oh':  oral_open * lip_round * 0.9,               # mid-back vowel
            'ou':  lip_round * (1.0 - lip_close) * 0.8,       # back rounded
        }

        # Normalise: largest weight drives; others are proportional blend
        weights_arr = np.array([raw[v] for v in VISEMES], dtype=np.float32)
        total = weights_arr.sum()
        if total > 1e-6:
            weights_arr /= total
        else:
            weights_arr[0] = 1.0   # silence

        return {v: float(w) for v, w in zip(VISEMES, weights_arr)}

    def jaw_aperture(self, areas: np.ndarray) -> float:
        """
        Derive jaw open value [0, 1] from lip + front-oral areas.
        Used to drive the jaw bone independently of viseme weights.
        """
        if len(areas) < 44:
            return 0.0
        lip_avg  = float(np.asarray(areas)[_VT_LIP_SLICE].mean())
        oral_avg = float(np.asarray(areas)[_VT_MID_ORAL_SLICE].mean())
        # Large lip + oral area → jaw open; near-zero → closed
        return float(np.clip((lip_avg + oral_avg * 0.3) / (0.8 + 3.8 * 0.3), 0.0, 1.0))


# ---------------------------------------------------------------------------
# HatPhysicsChain — 3-bone spring simulation
# ---------------------------------------------------------------------------

class HatPhysicsChain:
    """
    Secondary physics for the 3-bone hat chain (hat_chain_01/02/03).

    Models each bone as a damped spring attached to the bone above it.
    The root bone (hat_tilt) is driven by the emotional mapping; the chain
    lags behind with realistic inertia.

    Physics parameters
    ------------------
    stiffness  : spring constant (higher = stiffer chain, less lag)
    damping    : velocity damping (higher = less oscillation)
    mass       : effective bone mass (higher = more inertia / lag)
    """

    N_CHAIN = 3   # hat_chain_01, hat_chain_02, hat_chain_03

    def __init__(self,
                 stiffness: float = 12.0,
                 damping:   float = 4.0,
                 mass:      float = 0.08):
        self.stiffness = stiffness
        self.damping   = damping
        self.mass      = mass

        # Per-bone state: angle (radians), angular velocity
        self._angles = np.zeros(self.N_CHAIN, dtype=np.float64)
        self._vels   = np.zeros(self.N_CHAIN, dtype=np.float64)

    def update(self, dt_ms: float, root_angle: float) -> np.ndarray:
        """
        Advance spring chain by one step.

        Parameters
        ----------
        dt_ms      : time step in milliseconds
        root_angle : current hat_tilt angle (in normalised rig units)

        Returns
        -------
        angles : (3,) array — hat_chain_01/02/03 angles in normalised rig units
        """
        dt_s = dt_ms / 1000.0

        # Each chain bone springs toward the bone above it
        targets = np.empty(self.N_CHAIN, dtype=np.float64)
        targets[0] = root_angle          # chain_01 follows hat_tilt
        targets[1] = self._angles[0]     # chain_02 follows chain_01
        targets[2] = self._angles[1]     # chain_03 follows chain_02

        # Semi-implicit Euler spring integration
        for i in range(self.N_CHAIN):
            displacement = self._angles[i] - targets[i]
            spring_force = -self.stiffness * displacement
            damp_force   = -self.damping   * self._vels[i]
            accel        = (spring_force + damp_force) / self.mass

            self._vels[i]   += accel * dt_s
            self._angles[i] += self._vels[i] * dt_s

        # Clamp to sane range
        self._angles = np.clip(self._angles, -0.8, 0.8)
        return self._angles.astype(np.float32)

    def reset(self) -> None:
        self._angles[:] = 0.0
        self._vels[:]   = 0.0


# ---------------------------------------------------------------------------
# AvatarController — 60 fps integration class
# ---------------------------------------------------------------------------

class AvatarController:
    """
    Drives CAINE's avatar at 60 fps.

    On each call to ``update()`` it:
    1. Calls ``emotional_to_pose(neuro)`` to get the emotional target.
    2. Calls ``VisemeMapper(vocal_tract.areas)`` to get viseme targets.
    3. Advances the hat spring chain.
    4. Interpolates the current pose toward the target at ``lerp_speed``.
    5. Returns the current AvatarState dict (consumable by Blender bridge).

    Parameters
    ----------
    lerp_speed : fraction of gap to close per second (0.1 = slow/dreamy,
                 1.0 = instant snap).  Applied as exponential approach.
    """

    # Periodic blink parameters
    BLINK_INTERVAL_S  = 4.5    # mean seconds between blinks
    BLINK_DURATION_MS = 120.0  # ms for a full blink cycle

    def __init__(self, lerp_speed: float = 6.0):
        self.lerp_speed     = lerp_speed
        self._viseme_mapper = VisemeMapper()
        self._hat_chain     = HatPhysicsChain()

        # Current interpolated pose
        self._current = PoseTarget()

        # Blink state
        self._blink_timer_ms:    float = 0.0
        self._next_blink_ms:     float = self.BLINK_INTERVAL_S * 1000.0
        self._blink_progress_ms: float = -1.0   # -1 = not blinking

        # History for visualisation
        self._history: list = []   # list of PoseTarget.as_dict()

    # ------------------------------------------------------------------
    def update(self,
               dt_ms:       float,
               neuro,
               vocal_tract  = None,
               stg_rates:   np.ndarray = None) -> dict:
        """
        Advance avatar by one frame.

        Parameters
        ----------
        dt_ms        : frame duration in ms (16.67 for 60 fps)
        neuro        : NeurochemicalSystem — live neurochemical state
        vocal_tract  : VocalTract or None — provides areas for viseme mapping
        stg_rates    : (N_STG,) float or None — STG firing rates for gaze

        Returns
        -------
        dict — current pose values (identical to PoseTarget.as_dict())
        """
        # --- Compute emotional target ---
        gaze = self._social_gaze(stg_rates)
        target = emotional_to_pose(neuro, gaze_target=gaze)

        # --- Compute viseme target from vocal tract ---
        if vocal_tract is not None:
            target.viseme_weights = self._viseme_mapper(vocal_tract.areas)
            target.jaw_open       = self._viseme_mapper.jaw_aperture(vocal_tract.areas)
        else:
            target.viseme_weights = {v: 0.0 for v in VISEMES}
            target.viseme_weights['sil'] = 1.0

        # --- Advance hat spring chain ---
        chain_angles = self._hat_chain.update(dt_ms, target.hat_tilt)

        # --- Blink update ---
        blink_val = self._update_blink(dt_ms, neuro)
        target.blink_left  = blink_val
        target.blink_right = blink_val

        # --- Exponential approach toward target ---
        alpha = float(np.clip(1.0 - np.exp(-self.lerp_speed * dt_ms / 1000.0),
                               0.0, 1.0))
        self._current = self._interpolate(self._current, target, alpha)

        # Build output dict (includes hat chain angles)
        out = self._current.as_dict()
        for i, ang in enumerate(chain_angles):
            out[f'hat_chain_{i+1:02d}'] = float(ang)

        self._history.append(out)
        if len(self._history) > 300:   # keep last 5 s at 60 fps
            self._history.pop(0)

        return out

    # ------------------------------------------------------------------
    def _social_gaze(self, stg_rates: np.ndarray) -> Tuple[float, float, float]:
        """
        Derive a gaze target from STG population activity.

        When STG has a strong peak (phoneme detected → voice present),
        gaze shifts toward the social target position.  No STG → forward gaze.
        """
        if stg_rates is None or len(stg_rates) == 0:
            return (0.0, 1.0, 3.0)   # default: looking slightly forward/up

        peak = float(stg_rates.max())
        if peak < 0.5:
            return (0.0, 1.0, 3.0)   # nothing strong — forward gaze

        # Winner neuron index → lateral gaze offset
        winner  = int(np.argmax(stg_rates))
        n       = len(stg_rates)
        x_shift = lerp(-1.2, 1.2, winner / max(n - 1, 1))
        return (x_shift, 1.0, 3.0)

    # ------------------------------------------------------------------
    def _update_blink(self, dt_ms: float, neuro) -> float:
        """
        Periodic blink gated by acetylcholine.

        High ACh (alert, attending) → less frequent blinking.
        Low ACh  (drowsy/resting)   → more frequent blinking.
        """
        ach = float(getattr(neuro, 'acetylcholine', 0.12))
        # Adjust effective interval: high ACh lengthens it
        effective_interval_ms = self.BLINK_INTERVAL_S * 1000.0 * (0.5 + 1.5 * ach)

        self._blink_timer_ms += dt_ms

        # Start a blink
        if (self._blink_progress_ms < 0 and
                self._blink_timer_ms >= effective_interval_ms):
            self._blink_timer_ms    = 0.0
            self._blink_progress_ms = 0.0

        # Advance blink animation
        if self._blink_progress_ms >= 0:
            self._blink_progress_ms += dt_ms
            if self._blink_progress_ms >= self.BLINK_DURATION_MS:
                self._blink_progress_ms = -1.0   # done
                return 0.0
            # Triangle wave: close then open
            half = self.BLINK_DURATION_MS / 2.0
            if self._blink_progress_ms < half:
                return float(self._blink_progress_ms / half)
            else:
                return float(1.0 - (self._blink_progress_ms - half) / half)

        return 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _interpolate(current: PoseTarget,
                     target:  PoseTarget,
                     alpha:   float) -> PoseTarget:
        """Lerp every scalar field and viseme weight toward the target."""
        def _l(c, t):
            return c + (t - c) * alpha

        # Viseme interpolation
        vis = {}
        for v in VISEMES:
            c_w = current.viseme_weights.get(v, 0.0)
            t_w = target.viseme_weights.get(v, 0.0)
            vis[v] = _l(c_w, t_w)

        # Gaze — direct snap (no interpolation; gaze should be reactive)
        return PoseTarget(
            eye_width    = _l(current.eye_width,    target.eye_width),
            spine_curl   = _l(current.spine_curl,   target.spine_curl),
            head_tilt    = _l(current.head_tilt,    target.head_tilt),
            hat_tilt     = _l(current.hat_tilt,     target.hat_tilt),
            gaze_target  = target.gaze_target,
            blink_left   = target.blink_left,
            blink_right  = target.blink_right,
            squint_left  = _l(current.squint_left,  target.squint_left),
            squint_right = _l(current.squint_right, target.squint_right),
            jaw_open     = _l(current.jaw_open,     target.jaw_open),
            viseme_weights = vis,
        )

    # ------------------------------------------------------------------
    @property
    def current_pose(self) -> PoseTarget:
        return self._current

    # ------------------------------------------------------------------
    def visualize(self, save_path: str = None) -> str:
        """
        Render a figure showing avatar state history.

        Panels
        ------
        1  Emotional expression scalars over time
        2  Top-5 viseme weights over time
        3  Hat tilt + spring chain angles
        4  Jaw aperture + blink
        """
        if not self._history:
            warnings.warn("[avatar] No history — call update() first.")
            return ''

        n = len(self._history)
        t = np.arange(n) / 60.0   # seconds at 60 fps

        def _get(key):
            return [frame.get(key, 0.0) for frame in self._history]

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("CAINE Avatar — Module 8", fontsize=13)

        # Panel 1: Emotional expression
        ax = axes[0]
        ax.plot(t, _get('eye_width'),  label='eye_width (DA)',    color='#3498db', lw=1.5)
        ax.plot(t, _get('spine_curl'), label='spine_curl (CORT)', color='#e74c3c', lw=1.5)
        ax.plot(t, _get('head_tilt'),  label='head_tilt (ACh)',   color='#2ecc71', lw=1.5)
        ax.plot(t, _get('hat_tilt'),   label='hat_tilt (NE)',     color='#f39c12', lw=1.5)
        ax.set_ylabel('normalised', fontsize=8)
        ax.set_title('Emotional expression (neurochemical mapping)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.45, 1.1)

        # Panel 2: Top-5 visemes
        ax = axes[1]
        vis_names = list(VISEMES)
        vis_mat = np.array([[frame.get(f'viseme_{v}', 0.0) for v in vis_names]
                             for frame in self._history])
        mean_weights = vis_mat.mean(axis=0)
        top5_idx = np.argsort(mean_weights)[::-1][:5]
        colors5 = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#f39c12']
        for ci, vi in enumerate(top5_idx):
            ax.plot(t, vis_mat[:, vi], label=vis_names[vi],
                    color=colors5[ci], lw=1.2)
        ax.set_ylabel('weight', fontsize=8)
        ax.set_title('Viseme blend weights (top 5 by mean)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right', ncol=3)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Panel 3: Hat physics chain
        ax = axes[2]
        ax.plot(t, _get('hat_tilt'),     label='hat_tilt (root)', color='#f39c12', lw=1.5)
        ax.plot(t, _get('hat_chain_01'), label='chain_01',        color='#e67e22', lw=1.0, ls='--')
        ax.plot(t, _get('hat_chain_02'), label='chain_02',        color='#d35400', lw=1.0, ls=':')
        ax.plot(t, _get('hat_chain_03'), label='chain_03',        color='#c0392b', lw=1.0, ls='-.')
        ax.set_ylabel('angle (rig units)', fontsize=8)
        ax.set_title('Hat secondary physics chain', fontsize=9)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)

        # Panel 4: Jaw + blink
        ax = axes[3]
        ax.plot(t, _get('jaw_open'),   label='jaw_open',   color='#1abc9c', lw=1.5)
        ax.plot(t, _get('blink_left'), label='blink',      color='#95a5a6', lw=1.2)
        ax.set_xlabel('time (s)', fontsize=9)
        ax.set_ylabel('value', fontsize=8)
        ax.set_title('Jaw aperture + blink', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(_OUTPUT_DIR, 'caine_module8_avatar.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[avatar] Visualisation saved to {save_path}")
        return save_path


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

def run_avatar_demo(n_frames: int = 300, dt_ms: float = 16.67) -> AvatarController:
    """
    Run the avatar controller for n_frames using synthetic neurochemical
    and vocal tract inputs.

    Demonstrates:
      - Emotional expression changing as neurochemicals evolve
      - Viseme mapping as vocal tract areas vary
      - Hat spring chain lag behind hat_tilt
      - Periodic blinks gated by ACh
    """
    from caine.chemicals import NeurochemicalSystem, NeurochemicalEvent, EventType
    from caine.sensory   import VocalTract

    neuro = NeurochemicalSystem()
    vt    = VocalTract(pitch_hz=120.0, glottal_open=0.8)
    av    = AvatarController(lerp_speed=5.0)

    print(f"[avatar] Demo: {n_frames} frames x {dt_ms:.2f} ms ({n_frames * dt_ms / 1000:.1f} s)")

    rng = np.random.default_rng(99)

    for f in range(n_frames):
        t_s = f * dt_ms / 1000.0

        # Simulate neurochemical dynamics
        events = []
        if f == 60:    # dopamine spike at 1s — exciting moment
            events.append(NeurochemicalEvent(EventType.REWARD, 0.9))
        if f == 120:   # stress event at 2s
            events.append(NeurochemicalEvent(EventType.THREAT, 0.6))
        if f == 200:   # social warmth at ~3.3s
            events.append(NeurochemicalEvent(EventType.VOICE_MATCH, 0.8))

        neuro.update(dt_ms, events=events)

        # Simulate vocal tract articulation (slow oscillation)
        vt.areas = vt._DEFAULT_AREAS.copy()
        # Lip area varies sinusoidally (mouth opening/closing)
        lip_mod = 0.4 + 0.4 * np.sin(2 * np.pi * 1.5 * t_s)
        vt.areas[42] = float(np.clip(lip_mod, 0.05, 0.8))
        vt.areas[43] = float(np.clip(lip_mod * 0.5, 0.02, 0.4))
        # Tongue position cycles through vowel space
        tongue_phase = 2 * np.pi * 0.5 * t_s
        for i in range(20, 34):
            vt.areas[i] = float(np.clip(
                vt._DEFAULT_AREAS[i] + 1.5 * np.sin(tongue_phase + i * 0.2),
                0.1, 5.0))

        # Synthetic STG rates (phoneme detector activity)
        stg_rates = rng.uniform(0.0, 1.0, 24).astype(np.float32)
        stg_rates *= (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t_s))

        # Update avatar
        state = av.update(dt_ms, neuro, vt, stg_rates)

        if f % 60 == 0:
            top_vis = max(state, key=lambda k: state[k]
                         if k.startswith('viseme_') else -1)
            print(f"  frame {f:3d} | "
                  f"eye_w={state['eye_width']:.2f}  "
                  f"spine={state['spine_curl']:.3f}  "
                  f"hat={state['hat_tilt']:.3f}  "
                  f"jaw={state['jaw_open']:.3f}  "
                  f"top_vis={top_vis.replace('viseme_','')}")

    path = av.visualize()
    print(f"[avatar] Demo complete. Figure: {path}")
    return av


if __name__ == '__main__':
    run_avatar_demo()
