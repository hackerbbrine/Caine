"""
CAINE Media Learning System — Module 6
========================================
Structured external media for accelerated concept acquisition.

Once CAINE reaches sufficient developmental maturity he gains access to
flashcards and learning videos.  This is analogous to educational content
for developing children.

Key principle: media does NOT inject knowledge.  Flashcards create highly
reliable co-activation between IT object representations and STG phoneme
patterns; the concept forms through Hebbian association.  Premature
presentation produces cortisol (confusion/failure signal) and wastes
developmental time.

Classes
-------
  Flashcard            — paired multi-modal stimulus (image + audio label)
  LearningVideo        — temporal concept video with synchronized captions
  GateEvaluator        — checks unlock conditions for each content type
  MediaLearningSystem  — queue management, gate enforcement, presentation

Safety gates (Table 6.3)
------------------------
  Object flashcards        → IT differential response confirmed
  Action flashcards        → M1 voluntary movement detected
  Social behavior videos   → DMN activity detectable
  Mental state videos      → PFC working memory span > 2 s
  Abstract concept videos  → Self-model representation stable in PFC

Usage
-----
    from caine.media import MediaLearningSystem, Flashcard, LearningVideo
    from caine.media import ContentType

    mls = MediaLearningSystem()

    mls.enqueue(Flashcard(
        image_path='data/apple.png',
        audio_label='apple',
        repetitions=5,
        neurochemical_boost='acetylcholine',
    ))

    # Each simulation tick:
    result = mls.tick(cortex_state, sensory_layer, neuro, dt_ms=20.0, t_ms=t)
"""

import os
import sys as _sys
import json
import time as _time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

# -- matplotlib (Agg-safe) --------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import caine.paths as _paths
_OUTPUT_DIR = _paths.OUTPUT_DIR
_MEDIA_LOG  = _paths.MEDIA_LOG


# ---------------------------------------------------------------------------
# Content type enum
# ---------------------------------------------------------------------------

class ContentType(Enum):
    """Category of learning content — each maps to a safety gate."""
    OBJECT_FLASHCARD        = 'object_flashcard'
    ACTION_FLASHCARD        = 'action_flashcard'
    SOCIAL_VIDEO            = 'social_video'
    MENTAL_STATE_VIDEO      = 'mental_state_video'
    ABSTRACT_CONCEPT_VIDEO  = 'abstract_concept_video'


# ---------------------------------------------------------------------------
# 6.1  Flashcard
# ---------------------------------------------------------------------------

@dataclass
class Flashcard:
    """
    Paired multi-modal stimulus presented with controlled spaced repetition.

    A flashcard is NOT a knowledge injection.  It creates highly reliable
    co-activation between IT object representations and STG phoneme patterns
    so that Hebbian association forms quickly.

    Fields
    ------
    image_path          : path to image file displayed in CAINE's visual field
    audio_label         : spoken word played simultaneously (TTS or WAV path)
    repetitions         : total number of spaced presentations
    interval_seconds    : minimum inter-presentation gap (seconds sim time)
    neurochemical_boost : optional chemical to prime attention ('acetylcholine',
                          'dopamine', or '' for none)
    motor_context       : optional animation tag to pair with action words
                          (e.g. 'reach', 'grasp') — used with ACTION_FLASHCARD
    content_type        : controls which safety gate applies
    """
    image_path:           str
    audio_label:          str
    repetitions:          int            = 3
    interval_seconds:     float          = 5.0
    neurochemical_boost:  str            = ''
    motor_context:        Optional[str]  = None
    content_type:         ContentType    = ContentType.OBJECT_FLASHCARD

    # Internal tracking (not part of public API)
    _reps_done:           int   = field(default=0, repr=False, init=False)
    _last_presented_ms:   float = field(default=-1e9, repr=False, init=False)

    @property
    def is_complete(self) -> bool:
        """True when all scheduled repetitions have been presented."""
        return self._reps_done >= self.repetitions

    @property
    def reps_remaining(self) -> int:
        return max(0, self.repetitions - self._reps_done)

    def ready_to_present(self, t_ms: float) -> bool:
        """True when enough simulated time has passed since last presentation."""
        elapsed_ms = t_ms - self._last_presented_ms
        return elapsed_ms >= self.interval_seconds * 1000.0

    def mark_presented(self, t_ms: float) -> None:
        self._reps_done += 1
        self._last_presented_ms = t_ms


# ---------------------------------------------------------------------------
# 6.2  LearningVideo
# ---------------------------------------------------------------------------

@dataclass
class LearningVideo:
    """
    Video providing temporal concept learning (sequences, causality, behavior).

    CAINE watches through his normal visual pipeline; captions are played as
    synchronized audio through the auditory pipeline.  Over repeated viewings,
    the visual temporal pattern of the concept binds to its verbal label in STG.

    Content sub-types and their gates:
      Behavioral videos       → DMN activity detectable (Stage 3)
      Mental state videos     → PFC WM span > 2 s
      Abstract concept videos → Self-model stable in PFC (late Stage 3)

    Fields
    ------
    video_path          : path to video file (played frame-by-frame into V1)
    caption_track       : timed TTS captions (spoken, not displayed as text)
    concept_label       : human-readable description ('someone thinking', etc.)
    pre_attention_cue   : if True, play a 440 Hz attention tone before start
    repetitions         : total full-video playback count
    content_type        : maps to safety gate (SOCIAL_VIDEO / MENTAL_STATE_VIDEO
                          / ABSTRACT_CONCEPT_VIDEO)
    """
    video_path:         str
    caption_track:      str
    concept_label:      str
    pre_attention_cue:  bool          = True
    repetitions:        int           = 1
    content_type:       ContentType   = ContentType.SOCIAL_VIDEO

    # Internal tracking
    _reps_done:         int   = field(default=0, repr=False, init=False)
    _frame_index:       int   = field(default=0, repr=False, init=False)

    @property
    def is_complete(self) -> bool:
        return self._reps_done >= self.repetitions

    @property
    def reps_remaining(self) -> int:
        return max(0, self.repetitions - self._reps_done)

    def mark_rep_complete(self) -> None:
        self._reps_done += 1
        self._frame_index = 0

    def advance_frame(self) -> int:
        """Increment internal frame counter and return new index."""
        self._frame_index += 1
        return self._frame_index


# ---------------------------------------------------------------------------
# 6.3  GateEvaluator — safety gate logic
# ---------------------------------------------------------------------------

class GateEvaluator:
    """
    Evaluates whether developmental prerequisites are met for each content type.

    All checks consume a ``cortex_state`` dict that the main loop should
    populate each tick.  Expected keys:

        it_rate_history     : list of (N_IT,) np.ndarray — recent IT rates
        m1_output_history   : list of float — recent M1 max output values
        dmn_emerged         : bool — from DMNMonitor.update()
        wm_trace_history    : list of (N_PFC,) np.ndarray — PFC WM trace
        pfc_mpfc_history    : list of (8,) np.ndarray — mPFC [0:8] rates
        dt_ms               : float — simulation step duration

    Methods return True when the gate is open (content may be shown).
    """

    # Tunable thresholds
    IT_VAR_THRESHOLD        = 0.04   # mean per-neuron variance across frames
    M1_MOVE_THRESHOLD       = 0.25   # normalised M1 output amplitude
    WM_SPAN_MS              = 2000.0 # PFC WM must stay active for this long
    WM_ACTIVE_RATE_HZ       = 5.0    # min mean WM rate (Hz) to count as active
    PFC_STABILITY_THRESHOLD = 0.60   # correlation coefficient for stable mPFC

    def it_differential_confirmed(self, it_rate_history: list) -> bool:
        """
        IT cortex shows stable differential object responses.
        Measured as mean per-neuron firing-rate variance across recent frames.
        Requires at least 10 frames of history.
        """
        if len(it_rate_history) < 10:
            return False
        mat = np.array(it_rate_history[-30:], dtype=np.float32)  # (T, N_IT)
        return float(mat.var(axis=0).mean()) > self.IT_VAR_THRESHOLD

    def m1_movement_detected(self, m1_output_history: list) -> bool:
        """
        M1 voluntary movement: at least one recent M1 output exceeds threshold.
        """
        if not m1_output_history:
            return False
        recent = m1_output_history[-20:]
        return float(max(recent)) > self.M1_MOVE_THRESHOLD

    def dmn_active(self, dmn_emerged: bool) -> bool:
        """DMN coherence detectable (returned by DMNMonitor)."""
        return bool(dmn_emerged)

    def pfc_wm_span_sufficient(self,
                                wm_trace_history: list,
                                dt_ms: float = 20.0) -> bool:
        """
        PFC working memory trace must remain active (mean WM rate > threshold)
        for at least WM_SPAN_MS of continuous simulated time.
        """
        if not wm_trace_history:
            return False
        frames_needed = max(1, int(self.WM_SPAN_MS / dt_ms))
        if len(wm_trace_history) < frames_needed:
            return False
        # Check last `frames_needed` frames: WM sub-region neurons [16:24]
        active_frames = 0
        for trace in wm_trace_history[-frames_needed:]:
            trace = np.asarray(trace)
            # WM sub-region is indices 16:24 if full PFC vector, else whole trace
            wm_region = trace[16:24] if len(trace) >= 24 else trace
            if wm_region.mean() >= self.WM_ACTIVE_RATE_HZ:
                active_frames += 1
        return active_frames >= frames_needed

    def pfc_self_model_stable(self, pfc_mpfc_history: list) -> bool:
        """
        mPFC self-model sub-region must show stable, consistent activation.
        Measured as mean pairwise Pearson correlation over recent frames >= threshold.
        """
        if len(pfc_mpfc_history) < 20:
            return False
        mat = np.array(pfc_mpfc_history[-30:], dtype=np.float32)  # (T, 8)
        # Mean pairwise correlation across time-steps
        T = mat.shape[0]
        # Normalise each frame
        mu  = mat.mean(axis=1, keepdims=True)
        std = mat.std(axis=1, keepdims=True) + 1e-8
        norm = (mat - mu) / std
        corr_matrix = (norm @ norm.T) / mat.shape[1]  # (T, T)
        # Mean off-diagonal
        mask = ~np.eye(T, dtype=bool)
        mean_corr = float(corr_matrix[mask].mean())
        return mean_corr >= self.PFC_STABILITY_THRESHOLD

    def check_all(self, cortex_state: dict) -> dict:
        """
        Evaluate all gates and return an unlock-status dict.

        Parameters
        ----------
        cortex_state : dict with keys described in class docstring

        Returns
        -------
        dict mapping ContentType → bool
        """
        it_hist   = cortex_state.get('it_rate_history',   [])
        m1_hist   = cortex_state.get('m1_output_history', [])
        dmn       = cortex_state.get('dmn_emerged',       False)
        wm_hist   = cortex_state.get('wm_trace_history',  [])
        mpfc_hist = cortex_state.get('pfc_mpfc_history',  [])
        dt_ms     = cortex_state.get('dt_ms',             20.0)

        return {
            ContentType.OBJECT_FLASHCARD:       self.it_differential_confirmed(it_hist),
            ContentType.ACTION_FLASHCARD:       self.m1_movement_detected(m1_hist),
            ContentType.SOCIAL_VIDEO:           self.dmn_active(dmn),
            ContentType.MENTAL_STATE_VIDEO:     self.pfc_wm_span_sufficient(wm_hist, dt_ms),
            ContentType.ABSTRACT_CONCEPT_VIDEO: self.pfc_self_model_stable(mpfc_hist),
        }


# ---------------------------------------------------------------------------
# MediaLearningSystem — main class
# ---------------------------------------------------------------------------

class MediaLearningSystem:
    """
    Manages the full media learning pipeline for CAINE.

    Responsibilities
    ----------------
    1. Maintain queues of Flashcard and LearningVideo items.
    2. Each tick: evaluate safety gates against current cortex state.
    3. If gate is open: present next item (inject into sensory pipeline).
    4. If gate is closed: skip item, optionally apply cortisol penalty.
    5. Apply neurochemical boosts (acetylcholine for attention) before presentation.
    6. Log all presentation events, gate transitions, and cortisol penalties.

    The system does NOT modify cortical weights directly — all learning
    happens through the existing Hebbian/STDP mechanisms already in the
    cortex populations.
    """

    # Cortisol penalty for premature content presentation
    PREMATURE_CORTISOL_BOOST = 0.08

    def __init__(self):
        self._flashcard_queue:  list = []   # List[Flashcard]
        self._video_queue:      list = []   # List[LearningVideo]
        self._gate_evaluator    = GateEvaluator()

        # Gate status history for visualisation
        self._gate_history:     list = []   # list of {ContentType: bool}
        self._event_log:        list = []   # presentation events
        self._cortisol_events:  list = []   # premature-presentation cortisol hits

        # Cumulative statistics
        self._total_presented   = 0
        self._total_skipped     = 0

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def enqueue(self, item) -> None:
        """
        Add a Flashcard or LearningVideo to the appropriate queue.

        Items are presented in FIFO order within each queue.
        Flashcards are interleaved with videos (flashcard takes priority
        when both are due at the same tick).
        """
        if isinstance(item, Flashcard):
            self._flashcard_queue.append(item)
        elif isinstance(item, LearningVideo):
            self._video_queue.append(item)
        else:
            raise TypeError(f"Expected Flashcard or LearningVideo, got {type(item)}")

    def enqueue_batch(self, items: list) -> None:
        """Convenience: enqueue a list of Flashcard / LearningVideo items."""
        for item in items:
            self.enqueue(item)

    @property
    def flashcard_queue_length(self) -> int:
        return len(self._flashcard_queue)

    @property
    def video_queue_length(self) -> int:
        return len(self._video_queue)

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def tick(self,
             cortex_state: dict,
             sensory_layer,
             neuro,
             dt_ms: float = 20.0,
             t_ms: float = 0.0) -> dict:
        """
        Process one simulation tick.

        Evaluates gates, then attempts to present one item (flashcard first,
        then video if no flashcard was presented this tick).

        Parameters
        ----------
        cortex_state  : dict of cortex state (see GateEvaluator for keys)
        sensory_layer : SensoryLayer instance (for injecting stimuli)
        neuro         : NeurochemicalSystem instance (for boosts)
        dt_ms         : simulation step duration
        t_ms          : current simulation time in ms

        Returns
        -------
        dict with keys:
            gate_status      : {ContentType: bool}
            presented        : Flashcard/LearningVideo or None
            cortisol_penalty : float (0.0 unless premature presentation attempted)
            event            : str description of what happened this tick
        """
        cortex_state.setdefault('dt_ms', dt_ms)

        # Evaluate safety gates
        gates = self._gate_evaluator.check_all(cortex_state)
        self._gate_history.append(gates.copy())

        presented       = None
        cortisol_penalty = 0.0
        event           = 'idle'

        # Try flashcard first
        if self._flashcard_queue:
            fc = self._flashcard_queue[0]
            if not fc.ready_to_present(t_ms):
                event = f'flashcard waiting (interval not elapsed)'
            elif gates.get(fc.content_type, False):
                presented = fc
                self._present_flashcard(fc, sensory_layer, neuro, t_ms)
                event = f'flashcard: {fc.audio_label!r} rep {fc._reps_done}/{fc.repetitions}'
                if fc.is_complete:
                    self._flashcard_queue.pop(0)
            else:
                # Gate closed — cortisol penalty
                cortisol_penalty = self.PREMATURE_CORTISOL_BOOST
                self._apply_cortisol(neuro, cortisol_penalty)
                self._total_skipped += 1
                event = (f'flashcard BLOCKED ({fc.content_type.value} gate closed) '
                         f'-> cortisol +{cortisol_penalty:.3f}')

        # Try video if no flashcard was presented
        elif self._video_queue:
            vid = self._video_queue[0]
            if gates.get(vid.content_type, False):
                presented = vid
                self._present_video_frame(vid, sensory_layer, neuro, t_ms)
                event = (f'video: {vid.concept_label!r} '
                         f'frame {vid._frame_index} rep {vid._reps_done}/{vid.repetitions}')
                if vid.is_complete:
                    self._video_queue.pop(0)
            else:
                cortisol_penalty = self.PREMATURE_CORTISOL_BOOST
                self._apply_cortisol(neuro, cortisol_penalty)
                self._total_skipped += 1
                event = (f'video BLOCKED ({vid.content_type.value} gate closed) '
                         f'-> cortisol +{cortisol_penalty:.3f}')

        if presented is not None:
            self._total_presented += 1

        # Log to file every time something non-idle happens
        if event != 'idle':
            self._log_event(t_ms, event, gates, cortisol_penalty)

        self._event_log.append({
            't_ms':            t_ms,
            'event':           event,
            'cortisol_penalty': cortisol_penalty,
        })

        return {
            'gate_status':       gates,
            'presented':         presented,
            'cortisol_penalty':  cortisol_penalty,
            'event':             event,
        }

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def _present_flashcard(self, fc: Flashcard,
                            sensory_layer, neuro, t_ms: float) -> None:
        """
        Inject flashcard stimuli into the sensory pipeline.

        1. Optionally boost neurochemical (e.g. acetylcholine for attention).
        2. Load and resize image → inject as visual frame.
        3. Generate audio label waveform → inject into auditory pipeline.
        4. Mark the flashcard as presented.
        """
        # Neurochemical boost
        if fc.neurochemical_boost:
            self._apply_neuro_boost(neuro, fc.neurochemical_boost)

        # Visual: load image or synthesise a placeholder
        frame_rgb = self._load_image_frame(fc.image_path)

        # Audio: synthesise spoken label as a tone burst (TTS stub)
        audio_pcm = self._label_to_audio(fc.audio_label)

        # Inject into sensory layer
        if sensory_layer is not None:
            sensory_layer.update(
                frame_rgb,
                np.zeros(6),          # no proprioceptive input during flashcard
                dt_ms=20.0,
                injected_audio=audio_pcm,
            )

        fc.mark_presented(t_ms)

    def _present_video_frame(self, vid: LearningVideo,
                              sensory_layer, neuro, t_ms: float) -> None:
        """
        Inject one video frame + caption audio into the sensory pipeline.

        On the first frame of a new repetition, optionally play a pre-attention
        cue tone through the auditory pipeline.
        """
        frame_idx = vid._frame_index

        # Pre-attention cue on first frame of each rep
        if frame_idx == 0 and vid.pre_attention_cue:
            cue_pcm = self._attention_cue_tone()
            if sensory_layer is not None:
                sensory_layer.update(
                    np.zeros((64, 64, 3), dtype=np.uint8),
                    np.zeros(6),
                    dt_ms=20.0,
                    injected_audio=cue_pcm,
                )

        # Video frame (stub: synthetic pattern keyed to frame_idx)
        frame_rgb = self._load_video_frame(vid.video_path, frame_idx)

        # Caption audio (stub: tone burst representing spoken caption)
        caption_pcm = self._label_to_audio(vid.caption_track)

        if sensory_layer is not None:
            sensory_layer.update(
                frame_rgb,
                np.zeros(6),
                dt_ms=20.0,
                injected_audio=caption_pcm,
            )

        # Advance frame; wrap into new rep when video ends
        MAX_VIDEO_FRAMES = 150   # ~3 s at 20 ms/frame
        new_idx = vid.advance_frame()
        if new_idx >= MAX_VIDEO_FRAMES:
            vid.mark_rep_complete()

    def _apply_neuro_boost(self, neuro, chemical: str) -> None:
        """Apply a small neurochemical boost before presenting content."""
        if neuro is None:
            return
        try:
            from caine.chemicals import NeurochemicalEvent, EventType
            _MAP = {
                'acetylcholine': EventType.ATTENTION,
                'dopamine':      EventType.REWARD,
                'oxytocin':      EventType.SOCIAL_TOUCH,
            }
            et = _MAP.get(chemical.lower())
            if et is not None:
                neuro.update(0.0, events=[NeurochemicalEvent(et, 0.5)])
        except Exception:
            pass

    def _apply_cortisol(self, neuro, amount: float) -> None:
        """Inject a cortisol stress signal for premature content presentation."""
        if neuro is None:
            return
        try:
            from caine.chemicals import NeurochemicalEvent, EventType
            neuro.update(0.0, events=[NeurochemicalEvent(EventType.THREAT, amount)])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Media loading stubs
    # ------------------------------------------------------------------

    def _load_image_frame(self, image_path: str) -> np.ndarray:
        """
        Load an image file and resize to 64×64 RGB.
        Falls back to a synthetic pattern when the file is unavailable.
        """
        if os.path.isfile(image_path):
            try:
                # Optional PIL/Pillow
                from PIL import Image as _PIL_Image
                img = _PIL_Image.open(image_path).convert('RGB').resize((64, 64))
                return np.array(img, dtype=np.uint8)
            except ImportError:
                pass
            except Exception:
                pass

        # Fallback: checkered pattern (distinct from blank frame)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                if (i // 8 + j // 8) % 2 == 0:
                    frame[i, j] = [200, 180, 100]
                else:
                    frame[i, j] = [60, 80, 140]
        return frame

    def _load_video_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """
        Extract a single frame from a video file.
        Falls back to a synthetic animated pattern when unavailable.
        """
        if os.path.isfile(video_path):
            try:
                import cv2 as _cv2
                cap = _cv2.VideoCapture(video_path)
                cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                    frame_rgb = _cv2.resize(frame_rgb, (64, 64))
                    return frame_rgb.astype(np.uint8)
            except ImportError:
                pass
            except Exception:
                pass

        # Fallback: animated sine-wave pattern keyed to frame index
        phase = frame_idx * 0.15
        y, x  = np.ogrid[0:64, 0:64]
        r_ch  = (128 + 100 * np.sin((x + y) / 8.0 + phase)).astype(np.uint8)
        g_ch  = (128 + 100 * np.sin((x - y) / 8.0 + phase * 1.3)).astype(np.uint8)
        b_ch  = (128 + 100 * np.cos(x / 6.0 + phase * 0.7)).astype(np.uint8)
        return np.stack([r_ch, g_ch, b_ch], axis=-1)

    @staticmethod
    def _label_to_audio(label: str, sr: int = 22050,
                         duration_ms: float = 200.0) -> np.ndarray:
        """
        Stub TTS: map a label string to a tone burst at a label-specific pitch.

        Each unique label gets a distinct fundamental frequency derived from
        its character codes, making labels acoustically distinguishable.
        When a real TTS engine is available, replace this method.
        """
        n_samples = int(sr * duration_ms / 1000.0)
        t = np.arange(n_samples, dtype=np.float32) / sr

        # Derive a reproducible pitch in [200, 1200] Hz from the label
        seed   = sum(ord(c) for c in (label or 'x')) % 997
        pitch  = 200.0 + (seed / 997.0) * 1000.0

        # Short attack/release envelope to avoid clicks
        env     = np.ones(n_samples, dtype=np.float32)
        ramp    = int(sr * 0.01)  # 10 ms
        env[:ramp]  = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)

        return (np.sin(2 * np.pi * pitch * t) * env).astype(np.float32)

    @staticmethod
    def _attention_cue_tone(sr: int = 22050, duration_ms: float = 80.0,
                             freq: float = 880.0) -> np.ndarray:
        """440 Hz double-beep attention cue tone."""
        n = int(sr * duration_ms / 1000.0)
        t = np.arange(n, dtype=np.float32) / sr
        env = np.ones(n, dtype=np.float32)
        r   = int(sr * 0.005)
        env[:r]  = np.linspace(0, 1, r)
        env[-r:] = np.linspace(1, 0, r)
        return (0.6 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_event(self, t_ms: float, event: str,
                   gates: dict, cortisol_penalty: float) -> None:
        """Append a JSON event record to the media learning log."""
        entry = {
            'event':            event,
            't_ms':             t_ms,
            'cortisol_penalty': round(cortisol_penalty, 4),
            'gates':            {ct.value: bool(v) for ct, v in gates.items()},
            'wall_time':        _time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        try:
            with open(_MEDIA_LOG, 'a') as fh:
                fh.write(json.dumps(entry) + '\n')
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Gate status summary (for diagnostics)
    # ------------------------------------------------------------------

    def gate_status_string(self, cortex_state: dict) -> str:
        """Return a human-readable one-liner of current gate status."""
        gates = self._gate_evaluator.check_all(cortex_state)
        parts = []
        icons = {True: '[open]', False: '[closed]'}
        for ct, open_ in gates.items():
            parts.append(f"{ct.value.replace('_', ' ')} {icons[open_]}")
        return ' | '.join(parts)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, save_path: str = None) -> str:
        """
        Render a figure showing gate history and event timeline.

        Panels
        ------
        1  Gate open/closed history per content type (raster)
        2  Cortisol penalty events over time
        3  Presentation count by content type (bar)

        Returns the path to the saved PNG.
        """
        if not self._gate_history:
            warnings.warn("[media] No gate history — call tick() first.")
            return ''

        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        fig.suptitle("CAINE Media Learning System — Module 6", fontsize=14)

        # ---- Panel 1: Gate history raster --------------------------------
        ax = axes[0]
        labels = [ct.value.replace('_', ' ') for ct in ContentType]
        n_ticks = len(self._gate_history)
        gate_mat = np.zeros((len(ContentType), n_ticks), dtype=np.float32)
        for ti, g in enumerate(self._gate_history):
            for ci, ct in enumerate(ContentType):
                gate_mat[ci, ti] = 1.0 if g.get(ct, False) else 0.0

        ax.imshow(gate_mat, aspect='auto', interpolation='nearest',
                  cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("tick", fontsize=9)
        ax.set_title("Safety Gate Status (green=open, red=closed)", fontsize=10)
        ax.tick_params(labelsize=8)

        # ---- Panel 2: Cortisol penalty timeline --------------------------
        ax = axes[1]
        if self._event_log:
            times   = [e['t_ms'] for e in self._event_log]
            cortisol = [e['cortisol_penalty'] for e in self._event_log]
            ax.bar(range(len(times)), cortisol, color='#e74c3c', width=0.8)
        ax.set_xlabel("event index", fontsize=9)
        ax.set_ylabel("cortisol\npenalty", fontsize=9)
        ax.set_title("Cortisol Penalties (premature content presentation)", fontsize=10)
        ax.tick_params(labelsize=8)

        # ---- Panel 3: Presentation counts --------------------------------
        ax = axes[2]
        ct_labels = [ct.value.replace('_', '\n') for ct in ContentType]
        counts = [0] * len(ContentType)
        for ev in self._event_log:
            for ci, ct in enumerate(ContentType):
                if ct.value in ev['event'] and 'BLOCKED' not in ev['event']:
                    counts[ci] += 1
        bars = ax.bar(ct_labels, counts,
                      color=['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c'])
        ax.set_ylabel("presentations", fontsize=9)
        ax.set_title("Content Presentations by Type", fontsize=10)
        ax.tick_params(labelsize=8)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        str(count), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(_OUTPUT_DIR, 'caine_module6_media.png')

        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[media] Visualisation saved to {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def print_stats(self) -> None:
        print(f"[MediaLearningSystem] presented={self._total_presented}  "
              f"skipped={self._total_skipped}  "
              f"fc_queue={self.flashcard_queue_length}  "
              f"vid_queue={self.video_queue_length}")


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

def run_media_demo(n_ticks: int = 60, dt_ms: float = 20.0):
    """
    Demonstrate the media learning system against a synthetic cortex state.

    Simulates developmental progression:
      ticks  0-19  : immature  (all gates closed)
      ticks 20-39  : IT comes online  (object flashcards unlock)
      ticks 40-59  : M1 + DMN come online  (action flashcards + social videos)

    Shows gate transitions, presentations, and cortisol penalties.
    """
    from caine.chemicals import NeurochemicalSystem

    mls   = MediaLearningSystem()
    neuro = NeurochemicalSystem()

    # Build a small curriculum
    flashcards = [
        Flashcard('data/apple.png',  'apple',  repetitions=3,
                  neurochemical_boost='acetylcholine',
                  content_type=ContentType.OBJECT_FLASHCARD),
        Flashcard('data/ball.png',   'ball',   repetitions=2,
                  content_type=ContentType.OBJECT_FLASHCARD),
        Flashcard('data/reach.png',  'reach',  repetitions=2,
                  motor_context='reach',
                  content_type=ContentType.ACTION_FLASHCARD),
    ]
    videos = [
        LearningVideo('data/social.mp4',   'person waving hello',
                      concept_label='social greeting',
                      pre_attention_cue=True,
                      content_type=ContentType.SOCIAL_VIDEO),
        LearningVideo('data/thinking.mp4', 'thinking deciding trying',
                      concept_label='internal mental state',
                      content_type=ContentType.MENTAL_STATE_VIDEO),
    ]
    mls.enqueue_batch(flashcards)
    mls.enqueue_batch(videos)

    print(f"[media] Demo: {n_ticks} ticks x {dt_ms} ms")
    print(f"  Queue: {mls.flashcard_queue_length} flashcards, "
          f"{mls.video_queue_length} videos")

    rng = np.random.default_rng(7)

    for tick in range(n_ticks):
        t_ms = tick * dt_ms

        # Synthetic cortex state — gates open progressively
        # ticks 20+: IT comes online (add variance)
        it_history = []
        if tick >= 20:
            for _ in range(30):
                it_history.append(rng.uniform(0.0, 12.0, size=24).astype(np.float32))

        # ticks 40+: M1 movement + DMN emerge
        m1_history  = [0.4] * 20 if tick >= 40 else [0.0] * 20
        dmn_emerged = (tick >= 40)

        cortex_state = {
            'it_rate_history':   it_history,
            'm1_output_history': m1_history,
            'dmn_emerged':       dmn_emerged,
            'wm_trace_history':  [],
            'pfc_mpfc_history':  [],
            'dt_ms':             dt_ms,
        }

        neuro.update(dt_ms)

        result = mls.tick(
            cortex_state,
            sensory_layer=None,   # no real sensory layer needed for this demo
            neuro=neuro,
            dt_ms=dt_ms,
            t_ms=t_ms,
        )

        if tick % 10 == 0 or result['cortisol_penalty'] > 0:
            gates_open = sum(v for v in result['gate_status'].values())
            print(f"  tick {tick:3d} | gates_open={gates_open}/5 | "
                  f"cort={result['cortisol_penalty']:.3f} | {result['event'][:60]}")

    mls.print_stats()
    path = mls.visualize()
    print(f"[media] Demo complete. Figure: {path}")
    return mls


if __name__ == '__main__':
    run_media_demo()
