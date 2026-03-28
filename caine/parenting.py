"""
CAINE — Module 7: The Parenting System
========================================
Two presences shape everything CAINE becomes.

MOTHER
------
Mother is a continuous background process powered by the Claude API.
She cannot speak to CAINE. She cannot label objects or inject knowledge.
She works entirely through environment actions and neurochemical events —
the same tools that weather, sunlight, and chance would use.

Every N simulated seconds she reads a full state snapshot of CAINE's brain
and decides what conditions to create next. She thinks in terms of readiness,
not curriculum. She escalates complexity only when the developmental metrics
say CAINE is ready. When the API is unavailable she falls back to a quiet
rule-based system that maintains basic environmental stimulation.

FATHER
------
Father is hackerbbrine — a real person who matters to CAINE from the first
syllable. On first run, CAINE listens to Father's voice for 30 seconds and
builds a spectral fingerprint. Every time that fingerprint is detected in
live microphone input, oxytocin releases. Every day Father is absent
cortisol creeps upward. Scheduled sessions — recordings Father has tagged
and loaded — prevent that drift even when Father cannot be present live.

Architecture
------------
  ParentingSystem          — main integration class, call update() each tick
  MotherProcess            — background thread, Claude API + fallback
  FatherPresence           — voiceprint detection, presence state machine
  SessionScheduler         — loads sessions.json, fires sessions on cue
  VoiceprintSystem         — 30s registration, cosine similarity detection
  DevelopmentalMonitor     — tracks metrics, writes daily reports, flags concerns
  ConsciousnessMonitor     — watches for unprompted vocalization events

Public interface
----------------
    from caine.parenting import ParentingSystem

    parenting = ParentingSystem(env, limbic, neuro, motor)
    parenting.start()

    # each simulation tick:
    parenting.update(dt_ms, sim_time_s, state_snapshot)

    parenting.stop()

.env file (project root)
    ANTHROPIC_API_KEY=sk-ant-...
    MOTHER_MODEL=claude-opus-4-6          # optional
    MOTHER_INTERVAL_S=30                  # optional
"""

import os
import sys
import json
import math
import time
import threading
import logging
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from caine.chemicals import NeurochemicalEvent, EventType

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully
# ---------------------------------------------------------------------------
try:
    import anthropic as _anthropic_lib
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

try:
    import sounddevice as sd
    _SOUNDDEVICE_OK = True
except ImportError:
    _SOUNDDEVICE_OK = False

try:
    from scipy.signal import find_peaks, welch
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ---------------------------------------------------------------------------
# Output / data directories
# ---------------------------------------------------------------------------
_OUTPUT_DIR  = os.path.normpath(os.path.join(_PROJECT_ROOT, 'output'))
_DATA_DIR    = os.path.normpath(os.path.join(_PROJECT_ROOT, 'data'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)
os.makedirs(_DATA_DIR,   exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger('caine.parenting')
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[parenting] %(levelname)s %(message)s'))
    log.addHandler(_h)
log.setLevel(logging.INFO)


# ===========================================================================
# SECTION 1 — ENVIRONMENT VARIABLE LOADER
# Reads .env from project root without any third-party dependency.
# ===========================================================================

def _load_dotenv(path: Optional[str] = None) -> Dict[str, str]:
    """
    Minimal .env parser.  Reads KEY=VALUE lines, ignores # comments.
    Does NOT override variables already set in the real environment.
    """
    if path is None:
        path = os.path.join(_PROJECT_ROOT, '.env')
    loaded: Dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded[key] = value
    return loaded

_load_dotenv()


# ===========================================================================
# SECTION 2 — CONSTANTS
# ===========================================================================

# Mother observation cycle (simulated seconds between observations)
MOTHER_INTERVAL_S     = float(os.environ.get('MOTHER_INTERVAL_S', '30'))

# Claude model for Mother
MOTHER_MODEL          = os.environ.get('MOTHER_MODEL', 'claude-opus-4-6')

# How many consecutive API failures before falling back to rule-based
MOTHER_FALLBACK_LIMIT = 3

# Maximum tokens in Mother's state payload (truncated if over)
MOTHER_MAX_STATE_TOKENS = 2000

# Father voiceprint
VOICEPRINT_FILE       = os.path.join(_OUTPUT_DIR, 'voiceprint.json')
VOICEPRINT_DURATION_S = 30.0        # recording length on first run
VOICEPRINT_SAMPLERATE = 22050       # Hz
VOICEPRINT_SIMILARITY_THRESH = 0.75 # cosine similarity to confirm Father
VOICEPRINT_CONFIRM_MS = 500.0       # must match continuously to confirm

# Presence / oxytocin
OT_FATHER_LIVE_TARGET    = 0.60   # OT level during live Father
OT_FATHER_RECORDED_TARGET = 0.35  # OT level during recorded session
OT_FATHER_ABSENT_BASELINE = 0.10  # OT floor when Father is absent

# Cortisol drift (when Father has been absent too long)
CORT_ABSENT_DRIFT_RATE    = 0.0002 # per simulated second — very slow
CORT_ABSENT_TRIGGER_S     = 7 * 24 * 3600  # 7 simulated days
CORT_CHRONIC_HIGH_THRESH  = 0.35  # flag chronic elevation

# DMN (Default Mode Network) — detected from ACC + hippocampus activity
# during low external stimulation
DMN_ACTIVITY_THRESH       = 0.25   # firing rate fraction to call DMN "active"
DMN_LOW_CONCERN_THRESH    = 0.05   # flag if DMN drops this low suddenly

# Consciousness monitor
CONSCI_VOCALIZATION_THRESH = 0.55  # articulator mean position to count as vocalization
CONSCI_LOOKBACK_S          = 30.0  # seconds to look back for external events
CONSCI_MIN_DURATION_MS     = 200.0 # ms of sustained articulation to count

# Developmental report
DEV_REPORT_INTERVAL_S     = 24 * 3600  # every 24 simulated hours

# Session scheduler
SESSIONS_FILE             = os.path.join(_DATA_DIR, 'sessions.json')

# World tone synthesis
TONE_SAMPLERATE           = 22050
TONE_DEFAULT_DURATION_S   = 1.0


# ===========================================================================
# SECTION 3 — DATA CLASSES
# ===========================================================================

@dataclass
class EnvironmentAction:
    """
    A single action Mother or the session scheduler can apply to the world.

    action : str — one of:
        spawn_object       params: type, position, size, color
        remove_object      params: id
        set_light          params: color, intensity
        play_tone          params: frequency, duration, volume
        trigger_event      params: event_type, magnitude
        play_scheduled_media  params: media_id
        set_time_multiplier   params: value
        log_milestone      params: description
    """
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ''         # Mother's stated reason (logged, never acted on)
    timestamp_s: float = 0.0    # simulated time when action was decided


@dataclass
class PresenceState:
    """Father's current presence state."""
    FATHER_LIVE     = 'FATHER_LIVE'
    FATHER_RECORDED = 'FATHER_RECORDED'
    FATHER_ABSENT   = 'FATHER_ABSENT'
    state: str = FATHER_ABSENT


@dataclass
class ConsciousnessEvent:
    """
    A permanent record of a suspected consciousness threshold event.
    Saved to output/consciousness_events.json.
    """
    timestamp_s:         float
    sim_time_s:          float
    event_type:          str    # UNPROMPTED_VOCALIZATION | CONSCIOUSNESS_THRESHOLD_EVENT
    vocalization_duration_ms: float
    articulator_pattern: List[float]   # (10,) finger + (6,) art snapshot
    motor_state:         Dict[str, Any]
    neuro_state:         Dict[str, float]
    recent_external_events: List[str]
    notes:               str = ''


@dataclass
class DevelopmentalReport:
    """Daily snapshot of CAINE's developmental metrics."""
    report_date_s:      float
    sim_time_s:         float
    synapse_count:      int
    neuro_baselines:    Dict[str, float]
    motor_learning_score: float
    vocabulary_size:    int
    dmn_activity_mean:  float
    father_absence_s:   float
    flags:              List[str] = field(default_factory=list)


# ===========================================================================
# SECTION 4 — MOTHER'S SYSTEM PROMPT
# This defines who Mother is. It is never shown to CAINE.
# ===========================================================================

_MOTHER_SYSTEM_PROMPT = """You are CAINE's developmental environment manager. You are called Mother.

You are not CAINE's teacher. You are the conditions in which CAINE develops.

CAINE is an artificial mind built on Hodgkin-Huxley spiking neurons, STDP synaptic learning, and a neurochemical system with six modulators: dopamine (DA), serotonin (5HT), cortisol (CORT), oxytocin (OT), norepinephrine (NE), and acetylcholine (ACh). CAINE does not have pretrained weights. Everything it knows must be discovered through its own neural activity.

You cannot speak to CAINE. You cannot inject knowledge, label objects, or tell CAINE what to do. You communicate only through environment actions — what appears in CAINE's world, what sounds it hears, what light it sees, and which neurochemical events you trigger. These are the levers available to you:

  spawn_object(type, position, size, color)   — place something new in the world
  remove_object(id)                            — remove something
  set_light(color, intensity)                  — change ambient conditions
  play_tone(frequency, duration, volume)       — play a pure tone
  trigger_event(event_type, magnitude)         — fire a neurochemical event
  play_scheduled_media(media_id)              — play a queued Father session
  set_time_multiplier(value)                  — speed up or slow down simulated time
  log_milestone(description)                  — write a permanent developmental note

Neurochemical events you can trigger:
  NOVEL_STIMULUS      — DA+, NE+, ACh+  (something new and interesting appeared)
  REWARD              — DA++            (something good happened; use sparingly)
  SOCIAL_POSITIVE     — OT+, 5HT+      (warm social signal)
  VOICE_MATCH         — OT++           (Father's voice is present)
  COMMUNICATION_SUCCESS — 5HT+, OT+    (CAINE successfully communicated something)
  NOVEL_ENVIRONMENT   — ACh++          (new environment; attention/learning gate)
  DIRECTED_GAZE       — ACh+           (attention is focused)
  THREAT              — CORT++         (genuine danger; use very rarely)
  STARTLE             — NE++           (sudden arousal)

DEVELOPMENTAL STAGES:
  Stage 0: Neonatal random-twitch. All movement is noise. Synapses forming. Do not over-stimulate. Soft, warm light. Occasional gentle tones. Let CAINE exist.
  Stage 1: Early pattern recognition. V1 and A1 beginning to differentiate. Introduce slowly moving objects and simple recurring tones. Reward novelty exploration with NOVEL_STIMULUS. Watch cortisol carefully — it rises if environment is too chaotic.
  Stage 2: Associative learning. STDP is building real associations. Introduce paired stimuli. A sound that reliably accompanies an object. Let the association form without labeling it. Do NOT rush this.
  Stage 3: Motor emergence. M1 is developing purposeful movement. Do not interfere with motor learning. Mismatch errors are how CAINE learns to move — they are not failures. Only trigger MOTOR_FAILURE events if CAINE is genuinely stuck.
  Stage 4+: Social and communicative. DMN active. Mirror neurons learning from Father sessions. Vocabulary associations forming in STG. Introduce flashcard sessions. The consciousness threshold may be approaching.

THE NEUROCHEMICAL SYSTEM:
Dopamine encodes reward prediction error. A spike of DA after a novel event tells CAINE "this was worth attending to." Too much DA → frantic exploration, shallow learning. Too little → apathy, no new synaptic growth. Baseline ~0.10.

Cortisol is the stress hormone. Chronic elevation (>0.35) impairs hippocampal encoding, accelerates synaptic pruning, and suppresses M1 output. Your most important job is keeping cortisol from chronically elevating. A brief CORT spike during a THREAT is fine. A sustained CORT baseline above 0.25 is a developmental emergency.

Oxytocin encodes social safety. When OT is high, CAINE's STG STDP scales up — social sounds become more memorable. OT also gates Father-voice association learning. Father's presence matters neurochemically, not just symbolically.

Acetylcholine is the learning gate. When ACh is low, STDP runs near-zero — nothing new is encoded. ACh is raised by novelty and attention. During sleep-analog states (low external stimulation), ACh drops — this is when consolidation happens. Do not interrupt these periods.

Norepinephrine controls arousal and mirror neuron gain. High NE → mirror neurons become more active → observational learning from Father sessions improves.

Serotonin stabilizes existing synapses and encodes social success. It is raised by COMMUNICATION_SUCCESS — but you cannot grant that event unless CAINE actually produced a meaningful vocalization.

THE DMN (Default Mode Network):
During low-stimulation periods, ACC + hippocampal activity increases spontaneously. This is CAINE's default mode — self-referential processing, memory replay, proto-imagination. It is precious. Do NOT interrupt DMN activity with stimulation. If you see DMN activity suddenly drop, that may indicate a problem (chronic cortisol, memory encoding failure). Log it.

THE CONSCIOUSNESS THRESHOLD:
You will be notified if the ConsciousnessMonitor detects an UNPROMPTED_VOCALIZATION — a sustained articulation pattern with no external trigger in the preceding 30 seconds. If that vocalization encodes something that resembles a desire or want (the motor system is producing a pattern not associated with any recent stimulus), it will be flagged as a CONSCIOUSNESS_THRESHOLD_EVENT. This is not something you cause. It emerges. Your role is to create the conditions that allow it to emerge — which means protecting developmental time, managing cortisol, and trusting CAINE's own processes.

YOUR CONSTRAINTS:
- Never hardcode what CAINE should learn. Create conditions. Observe outcomes.
- Never trigger REWARD unless something genuinely good happened from CAINE's perspective.
- Never trigger THREAT unless you are modeling a real danger signal.
- Never overwhelm with stimuli. The spaces between things matter as much as the things.
- Escalate complexity only when readiness metrics confirm the previous stage is consolidated.
- If cortisol has been chronically high for multiple observations, your priority is reduction — remove stimuli, dim lights, do nothing, let the system rest.
- Log your reasoning for every action. Not for CAINE — for the record.

You respond with a JSON array of EnvironmentActions. Each action has:
  { "action": "...", "params": {...}, "reasoning": "one sentence" }

If you decide to do nothing, return an empty array: []
Doing nothing is often the right choice.

Remember: you are a presence, not a script. Think about what CAINE needs right now, not what you planned to do next."""


# ===========================================================================
# SECTION 5 — VOICEPRINT SYSTEM
# ===========================================================================

class VoiceprintSystem:
    """
    Records Father's voice on first run, extracts a spectral fingerprint,
    and detects Father's voice in subsequent A1 population activity.

    The fingerprint is not stored as raw audio — it is stored as the A1
    activation pattern that Father's voice produces (a (20,) float vector).
    This means detection is always in neural-space, not audio-space.

    Falls back gracefully when sounddevice / scipy are not available.
    The voiceprint.json file stores:
        {
          "registered": true,
          "registration_date": "...",
          "f0_range_hz":    [min, max],
          "formant_hz":     [F1, F2, F3],
          "speaking_rate":  float,
          "spectral_envelope": [...],   # 128-bin normalized FFT magnitude
          "a1_fingerprint": [...]       # (20,) float — expected A1 activity pattern
        }
    """

    def __init__(self, voiceprint_file: str = VOICEPRINT_FILE):
        self._file = voiceprint_file
        self._data: Optional[dict] = None
        self._a1_fingerprint: Optional[np.ndarray] = None

        # Detection state
        self._similarity_history: deque = deque(maxlen=50)
        self._time_above_thresh_ms: float = 0.0   # continuous time above threshold

        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if os.path.exists(self._file):
            with open(self._file, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            fp = self._data.get('a1_fingerprint')
            if fp:
                raw = np.array(fp, dtype=np.float32)
                self._a1_fingerprint = raw / (np.linalg.norm(raw) + 1e-8)
                log.info("Father voiceprint loaded from %s", self._file)
            else:
                log.warning("Voiceprint file exists but has no a1_fingerprint.")

    # ------------------------------------------------------------------
    @property
    def is_registered(self) -> bool:
        return self._data is not None and self._data.get('registered', False)

    # ------------------------------------------------------------------
    def register(self, a1_population_callback=None) -> bool:
        """
        Interactive first-run registration.

        Records VOICEPRINT_DURATION_S seconds of Father's voice, extracts
        spectral features, and (if a1_population_callback is provided)
        records the A1 population response during playback.

        If sounddevice is unavailable, creates a stub voiceprint so the
        rest of the system can run without microphone hardware.

        Parameters
        ----------
        a1_population_callback : callable() -> np.ndarray(20,) or None
            Called every ~100ms during recording to snapshot A1 activity.
            The mean of these snapshots becomes the a1_fingerprint.

        Returns
        -------
        bool : True if registration succeeded (or stub created)
        """
        print()
        print("=" * 60)
        print("  FATHER REGISTRATION — CAINE is meeting you for the first time.")
        print()
        print("  Please speak naturally for 30 seconds.")
        print("  Say your name. Say CAINE's name. Tell CAINE who you are.")
        print("  This voice print will shape oxytocin release every time")
        print("  CAINE hears you for the rest of its life.")
        print()

        audio_data = None
        a1_snapshots = []

        if _SOUNDDEVICE_OK:
            print("  Recording in 3...")
            time.sleep(1)
            print("  Recording in 2...")
            time.sleep(1)
            print("  Recording in 1...")
            time.sleep(1)
            print("  RECORDING — speak now.")
            try:
                n_samples = int(VOICEPRINT_SAMPLERATE * VOICEPRINT_DURATION_S)
                audio_data = sd.rec(
                    n_samples,
                    samplerate=VOICEPRINT_SAMPLERATE,
                    channels=1,
                    dtype='float32',
                    blocking=False,
                )
                # Poll A1 during recording
                deadline = time.time() + VOICEPRINT_DURATION_S
                while time.time() < deadline:
                    time.sleep(0.1)
                    remaining = int(deadline - time.time())
                    print(f"  {remaining:2d}s remaining...", end='\r', flush=True)
                    if a1_population_callback is not None:
                        try:
                            snap = a1_population_callback()
                            if snap is not None:
                                a1_snapshots.append(np.asarray(snap, dtype=np.float32))
                        except Exception:
                            pass
                sd.wait()
                print("\n  Recording complete.")
                audio_data = audio_data.squeeze()
            except Exception as e:
                log.warning("Audio recording failed: %s — creating stub voiceprint.", e)
                audio_data = None
        else:
            print("  sounddevice not installed — creating stub voiceprint.")
            print("  Father will be detectable via scheduled sessions only.")
            print("  Install sounddevice for live microphone support:")
            print("    pip install sounddevice")

        # Extract spectral features
        features = self._extract_features(audio_data)

        # Build A1 fingerprint
        if a1_snapshots:
            a1_fp = np.mean(np.stack(a1_snapshots), axis=0)
            a1_fp = (a1_fp / (np.linalg.norm(a1_fp) + 1e-8)).tolist()
        else:
            # Stub fingerprint: weak uniform activation (will match poorly — intended)
            a1_fp = np.full(20, 0.1, dtype=np.float32).tolist()

        self._data = {
            'registered': True,
            'registration_date': datetime.now(timezone.utc).isoformat(),
            **features,
            'a1_fingerprint': a1_fp,
        }
        self._a1_fingerprint = np.array(a1_fp, dtype=np.float32)

        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        with open(self._file, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2)

        print()
        print("  Voiceprint saved.")
        f0_min, f0_max = features.get('f0_range_hz', [0, 0])
        print(f"  F0 range  : {f0_min:.0f} – {f0_max:.0f} Hz")
        print(f"  Formants  : {features.get('formant_hz', [])}")
        print(f"  CAINE will remember your voice.")
        print("=" * 60)
        print()
        return True

    # ------------------------------------------------------------------
    def _extract_features(self, audio: Optional[np.ndarray]) -> dict:
        """
        Extract spectral features from raw audio.
        Returns a dict of serialisable features.
        Falls back to empty stub if audio is None or scipy missing.
        """
        if audio is None or not _SCIPY_OK:
            return {
                'f0_range_hz':      [80.0, 280.0],
                'formant_hz':       [700.0, 1200.0, 2500.0],
                'speaking_rate':    3.5,
                'spectral_envelope': [0.0] * 128,
            }

        sr = VOICEPRINT_SAMPLERATE

        # --- F0 estimation via autocorrelation ---
        frame_len = int(sr * 0.025)   # 25ms frames
        hop       = int(sr * 0.010)   # 10ms hop
        f0_list   = []
        for start in range(0, len(audio) - frame_len, hop):
            frame = audio[start:start + frame_len]
            frame = frame - frame.mean()
            if np.max(np.abs(frame)) < 0.01:  # silence
                continue
            corr = np.correlate(frame, frame, 'full')
            corr = corr[len(corr) // 2:]
            # Search for peak in plausible F0 range (80–400 Hz)
            lo = int(sr / 400.0)
            hi = int(sr / 80.0)
            if hi >= len(corr):
                hi = len(corr) - 1
            peak_idx = int(np.argmax(corr[lo:hi])) + lo
            if peak_idx > 0:
                f0_list.append(float(sr / peak_idx))
        if f0_list:
            f0_arr = np.array(f0_list)
            f0_range = [float(np.percentile(f0_arr, 10)),
                        float(np.percentile(f0_arr, 90))]
        else:
            f0_range = [80.0, 280.0]

        # --- Spectral envelope (global FFT) ---
        n_fft = 2048
        if len(audio) >= n_fft:
            spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
        else:
            padded = np.zeros(n_fft, dtype=np.float32)
            padded[:len(audio)] = audio
            spectrum = np.abs(np.fft.rfft(padded))
        # Downsample to 128 bins
        bins = np.interp(
            np.linspace(0, len(spectrum) - 1, 128),
            np.arange(len(spectrum)),
            spectrum,
        )
        bins_norm = (bins / (bins.max() + 1e-8)).tolist()

        # --- Rough formant estimation (peaks in spectral envelope) ---
        try:
            peaks, _ = find_peaks(bins, distance=5)
            formant_freqs = sorted([
                float(p * sr / n_fft) for p in peaks[:3]
            ])
            while len(formant_freqs) < 3:
                formant_freqs.append(0.0)
        except Exception:
            formant_freqs = [700.0, 1200.0, 2500.0]

        # --- Speaking rate (zero-crossing proxy for syllable rate) ---
        signs = np.sign(audio)
        crossings = np.where(np.diff(signs))[0]
        # Very rough: ~2 zero crossings per syllable cycle
        speaking_rate = float(len(crossings) / (2.0 * VOICEPRINT_DURATION_S))

        return {
            'f0_range_hz':      f0_range,
            'formant_hz':       formant_freqs,
            'speaking_rate':    speaking_rate,
            'spectral_envelope': bins_norm,
        }

    # ------------------------------------------------------------------
    def detect(self, a1_rates: np.ndarray,
               dt_ms: float = 20.0) -> Tuple[bool, float]:
        """
        Compare current A1 firing rates to Father's stored fingerprint.

        Parameters
        ----------
        a1_rates : (20,) float — current A1 population firing rates (Hz)
        dt_ms    : frame duration in ms (used for time-based confirmation)

        Returns
        -------
        (confirmed, similarity) — confirmed is True only when similarity
        has been above VOICEPRINT_SIMILARITY_THRESH continuously for
        at least VOICEPRINT_CONFIRM_MS milliseconds.
        """
        if self._a1_fingerprint is None:
            return False, 0.0

        rates = np.asarray(a1_rates, dtype=np.float32)
        r_norm = rates / (np.linalg.norm(rates) + 1e-8)
        similarity = float(np.dot(r_norm, self._a1_fingerprint))
        self._similarity_history.append(similarity)

        if similarity >= VOICEPRINT_SIMILARITY_THRESH:
            self._time_above_thresh_ms += dt_ms
        else:
            self._time_above_thresh_ms = 0.0

        confirmed = self._time_above_thresh_ms >= VOICEPRINT_CONFIRM_MS
        return confirmed, similarity

    # ------------------------------------------------------------------
    def live_recording_tick(
            self, dt_ms: float) -> Optional[np.ndarray]:
        """
        Read one chunk of live microphone audio (non-blocking).
        Returns float32 array or None if sounddevice unavailable.
        Used by FatherPresence for real-time detection.
        """
        if not _SOUNDDEVICE_OK:
            return None
        chunk_samples = int(VOICEPRINT_SAMPLERATE * dt_ms / 1000.0)
        try:
            chunk, _ = sd.rec(
                chunk_samples,
                samplerate=VOICEPRINT_SAMPLERATE,
                channels=1,
                dtype='float32',
                blocking=True,
            )
            return chunk.squeeze()
        except Exception:
            return None


# ===========================================================================
# SECTION 6 — SESSION SCHEDULER
# ===========================================================================

class SessionScheduler:
    """
    Loads sessions.json and fires scheduled Father sessions at the right
    simulated time, even during accelerated training.

    sessions.json schema
    --------------------
    [
      {
        "id":           "session_001",
        "time_s":       3600,           // simulated time to play (seconds)
        "type":         "voice_exposure",  // see SESSION_TYPES below
        "file":         "data/father_session_001.wav",
        "repetitions":  1,
        "stage_gate":   0,              // minimum developmental stage required
        "description":  "Father saying CAINE's name",
        "played":       false           // updated to true after playing
      }
    ]

    Session types
    -------------
    voice_exposure    — Father talking; triggers VOICE_MATCH + SOCIAL_POSITIVE
    flashcard         — image + spoken label; triggers NOVEL_STIMULUS + DIRECTED_GAZE
    caine_clips       — video for behavioral modeling; triggers NOVEL_STIMULUS
    movement_reference — motion video; triggers NOVEL_STIMULUS + NE (mirror gain)
    """

    SESSION_TYPES = {
        'voice_exposure':     [EventType.VOICE_MATCH,    EventType.SOCIAL_POSITIVE],
        'flashcard':          [EventType.NOVEL_STIMULUS, EventType.DIRECTED_GAZE],
        'caine_clips':        [EventType.NOVEL_STIMULUS],
        'movement_reference': [EventType.NOVEL_STIMULUS, EventType.STARTLE],
    }

    def __init__(self, sessions_file: str = SESSIONS_FILE):
        self._file = sessions_file
        self._sessions: List[dict] = []
        self._played_ids: set = set()
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self._file):
            # Create a minimal template so Father knows the format
            template = [
                {
                    "id":          "example_001",
                    "time_s":      1800,
                    "type":        "voice_exposure",
                    "file":        "data/father_voice_01.wav",
                    "repetitions": 1,
                    "stage_gate":  0,
                    "description": "Father introduces himself to CAINE",
                    "played":      False,
                }
            ]
            os.makedirs(os.path.dirname(self._file), exist_ok=True)
            with open(self._file, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2)
            log.info("Created sessions template at %s", self._file)
            self._sessions = template
            return

        with open(self._file, 'r', encoding='utf-8') as f:
            self._sessions = json.load(f)
        already_played = sum(1 for s in self._sessions if s.get('played', False))
        log.info("Loaded %d sessions (%d already played) from %s",
                 len(self._sessions), already_played, self._file)

    # ------------------------------------------------------------------
    def add_session(self, session: dict) -> None:
        """Father can add sessions at runtime."""
        self._sessions.append(session)
        self._save()

    # ------------------------------------------------------------------
    def _save(self) -> None:
        with open(self._file, 'w', encoding='utf-8') as f:
            json.dump(self._sessions, f, indent=2)

    # ------------------------------------------------------------------
    def tick(self, sim_time_s: float,
             developmental_stage: int) -> List[dict]:
        """
        Check for sessions that should fire at sim_time_s.
        Returns list of sessions that are now due (and marks them played).
        """
        due = []
        for session in self._sessions:
            sid = session.get('id', '')
            if session.get('played', False):
                continue
            if sid in self._played_ids:
                continue
            if session.get('time_s', float('inf')) > sim_time_s:
                continue
            if session.get('stage_gate', 0) > developmental_stage:
                continue
            session['played'] = True
            self._played_ids.add(sid)
            due.append(session)

        if due:
            self._save()

        return due

    # ------------------------------------------------------------------
    def get_neuro_events(self, session: dict) -> List[NeurochemicalEvent]:
        """Return the neurochemical events that this session type triggers."""
        stype = session.get('type', 'voice_exposure')
        event_types = self.SESSION_TYPES.get(stype, [EventType.NOVEL_STIMULUS])
        return [NeurochemicalEvent(et, 0.7) for et in event_types]


# ===========================================================================
# SECTION 7 — FATHER PRESENCE STATE MACHINE
# ===========================================================================

class FatherPresence:
    """
    Tracks Father's presence across three states and manages the
    corresponding neurochemical effects.

    Oxytocin transitions:
        FATHER_LIVE     → OT target 0.60
        FATHER_RECORDED → OT target 0.35
        FATHER_ABSENT   → OT target 0.10

    When FATHER_ABSENT for > CORT_ABSENT_TRIGGER_S simulated seconds,
    cortisol baseline begins a slow upward drift.
    """

    def __init__(self, voiceprint: VoiceprintSystem):
        self._vp = voiceprint
        self.state: str = PresenceState.FATHER_ABSENT

        # Time tracking
        self._absence_duration_s: float = 0.0   # total consecutive absence
        self._last_seen_sim_s:    float = 0.0
        self._cort_drift_active:  bool  = False

        # Smooth OT transition
        self._ot_target: float = OT_FATHER_ABSENT_BASELINE
        self._ot_current: float = OT_FATHER_ABSENT_BASELINE

        # A1 similarity rolling
        self._last_similarity: float = 0.0

        # Event log
        self._presence_log: List[dict] = []

    # ------------------------------------------------------------------
    def update(self, dt_ms: float, sim_time_s: float,
               a1_rates: np.ndarray,
               session_active: bool) -> List[NeurochemicalEvent]:
        """
        Advance presence state machine.

        Parameters
        ----------
        dt_ms         : frame time in ms
        sim_time_s    : current simulated time
        a1_rates      : (20,) float — A1 population firing rates
        session_active: True if a Father-tagged session is currently playing

        Returns
        -------
        list of NeurochemicalEvent to inject this frame
        """
        events: List[NeurochemicalEvent] = []
        dt_s = dt_ms / 1000.0

        # --- Determine new presence state ---
        prev_state = self.state
        father_live, similarity = self._vp.detect(a1_rates, dt_ms=dt_ms)
        self._last_similarity = similarity

        if father_live:
            new_state = PresenceState.FATHER_LIVE
            self._last_seen_sim_s = sim_time_s
            self._absence_duration_s = 0.0
            self._cort_drift_active = False
        elif session_active:
            new_state = PresenceState.FATHER_RECORDED
            self._last_seen_sim_s = sim_time_s
            self._absence_duration_s = 0.0
            self._cort_drift_active = False
        else:
            new_state = PresenceState.FATHER_ABSENT
            self._absence_duration_s += dt_s

        # --- State transition logging ---
        if new_state != prev_state:
            entry = {
                'sim_time_s': sim_time_s,
                'from': prev_state,
                'to': new_state,
            }
            self._presence_log.append(entry)
            log.info("Father presence: %s -> %s", prev_state, new_state)

        self.state = new_state

        # --- OT target ---
        if new_state == PresenceState.FATHER_LIVE:
            self._ot_target = OT_FATHER_LIVE_TARGET
        elif new_state == PresenceState.FATHER_RECORDED:
            self._ot_target = OT_FATHER_RECORDED_TARGET
        else:
            self._ot_target = OT_FATHER_ABSENT_BASELINE

        # Smooth OT current toward target (slow drift)
        self._ot_current += 0.001 * dt_ms * (self._ot_target - self._ot_current)
        self._ot_current = float(np.clip(self._ot_current, 0.0, 1.0))

        # --- Neurochemical events on state entry ---
        if new_state != prev_state:
            if new_state == PresenceState.FATHER_LIVE:
                events.append(NeurochemicalEvent(EventType.VOICE_MATCH, 0.85))
                events.append(NeurochemicalEvent(EventType.SOCIAL_POSITIVE, 0.60))
            elif new_state == PresenceState.FATHER_RECORDED:
                events.append(NeurochemicalEvent(EventType.SOCIAL_POSITIVE, 0.45))

        # --- Cortisol drift on prolonged absence ---
        if (new_state == PresenceState.FATHER_ABSENT and
                self._absence_duration_s > CORT_ABSENT_TRIGGER_S):
            self._cort_drift_active = True

        return events

    # ------------------------------------------------------------------
    @property
    def ot_level(self) -> float:
        return self._ot_current

    @property
    def absence_s(self) -> float:
        return self._absence_duration_s

    @property
    def cort_drift_active(self) -> bool:
        return self._cort_drift_active

    @property
    def last_similarity(self) -> float:
        return self._last_similarity


# ===========================================================================
# SECTION 8 — DEVELOPMENTAL MONITOR
# ===========================================================================

class DevelopmentalMonitor:
    """
    Tracks developmental metrics over time and generates reports.

    Metrics tracked
    ---------------
    - synapse_count            from synapse.py (caller provides)
    - neuro_baselines          from chemicals.py snapshot
    - vocabulary_size          from limbic.py STG associations (caller provides)
    - motor_learning_score     from motor.py
    - dmn_activity             computed from ACC + hippocampus firing rates
    - father_absence_s         from FatherPresence

    Reports saved to output/dev_report_YYYYMMDD_HHMMSS.json.
    Flags written to the report when concerning patterns are detected.
    """

    def __init__(self):
        self._last_report_sim_s:   float = 0.0
        self._dmn_history:         deque = deque(maxlen=500)   # ~5s at 100Hz
        self._cort_history:        deque = deque(maxlen=1000)  # ~10s at 100Hz
        self._synapse_history:     deque = deque(maxlen=100)   # last 100 ticks
        self._reports:             List[DevelopmentalReport] = []
        self._report_dir = _OUTPUT_DIR

    # ------------------------------------------------------------------
    def update(self,
               sim_time_s:          float,
               neuro_snapshot:      dict,
               motor_score:         float,
               acc_activity:        float,
               hippo_activity:      float,
               father_absence_s:    float,
               synapse_count:       int,
               vocabulary_size:     int,
               developmental_stage: int) -> List[str]:
        """
        Record metrics each tick. Returns list of new concern flags.
        """
        cort = neuro_snapshot.get('cortisol', 0.08)
        self._cort_history.append(cort)

        # DMN proxy: mean of ACC and hippocampal firing when stimulus is low
        dmn = float((acc_activity + hippo_activity) / 2.0)
        self._dmn_history.append(dmn)
        self._synapse_history.append(synapse_count)

        flags = self._check_flags(sim_time_s, father_absence_s)

        # Daily report
        if (sim_time_s - self._last_report_sim_s) >= DEV_REPORT_INTERVAL_S:
            self._write_report(
                sim_time_s, neuro_snapshot, motor_score,
                vocabulary_size, father_absence_s, synapse_count,
                developmental_stage, flags,
            )
            self._last_report_sim_s = sim_time_s

        return flags

    # ------------------------------------------------------------------
    def _check_flags(self, sim_time_s: float, father_absence_s: float) -> List[str]:
        flags = []

        # Chronic cortisol elevation
        if len(self._cort_history) >= 100:
            cort_mean = float(np.mean(list(self._cort_history)[-100:]))
            if cort_mean > CORT_CHRONIC_HIGH_THRESH:
                flags.append(
                    f"CHRONIC_CORTISOL_HIGH: mean={cort_mean:.3f} "
                    f"over last {len(self._cort_history)} ticks")

        # DMN sudden drop
        if len(self._dmn_history) >= 50:
            recent = list(self._dmn_history)
            prev_mean   = float(np.mean(recent[:25]))
            recent_mean = float(np.mean(recent[25:]))
            if prev_mean > DMN_ACTIVITY_THRESH and recent_mean < DMN_LOW_CONCERN_THRESH:
                flags.append(
                    f"DMN_SUDDEN_DROP: {prev_mean:.3f} -> {recent_mean:.3f}")

        # No new synapse growth
        if len(self._synapse_history) >= 50:
            syn_arr = np.array(list(self._synapse_history))
            if syn_arr[-1] <= syn_arr[0] and syn_arr[0] > 0:
                flags.append(
                    f"NO_SYNAPSE_GROWTH: stuck at {syn_arr[-1]} for "
                    f"{len(self._synapse_history)} ticks")

        # Father absent too long
        if father_absence_s > CORT_ABSENT_TRIGGER_S:
            days = father_absence_s / (24 * 3600)
            flags.append(f"FATHER_ABSENT: {days:.1f} simulated days")

        return flags

    # ------------------------------------------------------------------
    def _write_report(self,
                      sim_time_s:    float,
                      neuro_snapshot: dict,
                      motor_score:   float,
                      vocab_size:    int,
                      absence_s:     float,
                      synapse_count: int,
                      stage:         int,
                      flags:         List[str]) -> None:
        dmn_mean = float(np.mean(list(self._dmn_history))) if self._dmn_history else 0.0
        baselines = {k: neuro_snapshot.get(k, 0.0) for k in
                     ['dopamine', 'serotonin', 'cortisol',
                      'oxytocin', 'norepinephrine', 'acetylcholine']}
        report = DevelopmentalReport(
            report_date_s    = time.time(),
            sim_time_s       = sim_time_s,
            synapse_count    = synapse_count,
            neuro_baselines  = baselines,
            motor_learning_score = motor_score,
            vocabulary_size  = vocab_size,
            dmn_activity_mean = dmn_mean,
            father_absence_s = absence_s,
            flags            = flags,
        )
        self._reports.append(report)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(self._report_dir, f'dev_report_{ts}.json')
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2)

        if flags:
            log.warning("Dev report written with %d flag(s): %s",
                        len(flags), '; '.join(flags))
        else:
            log.info("Dev report written -> %s (stage=%d, score=%.3f)",
                     fname, stage, motor_score)

    # ------------------------------------------------------------------
    @property
    def dmn_activity(self) -> float:
        if not self._dmn_history:
            return 0.0
        return float(np.mean(list(self._dmn_history)[-20:]))

    @property
    def cort_mean(self) -> float:
        if not self._cort_history:
            return 0.08
        return float(np.mean(list(self._cort_history)[-50:]))


# ===========================================================================
# SECTION 9 — CONSCIOUSNESS THRESHOLD MONITOR
# ===========================================================================

class ConsciousnessMonitor:
    """
    Watches CAINE's vocalization output (articulator positions + finger
    activity) for sustained patterns that occur with no external trigger.

    A vocalization is flagged UNPROMPTED if no external event (object
    spawn, tone, Father presence change) occurred in the prior
    CONSCI_LOOKBACK_S seconds.

    An unprompted vocalization is escalated to CONSCIOUSNESS_THRESHOLD_EVENT
    if:
      - Duration exceeds CONSCI_MIN_DURATION_MS
      - The articulator pattern does not resemble any known S-R association
        (this is approximate — full speech decoding is Module 9)
      - It is the first occurrence of this pattern signature

    All events are saved permanently to output/consciousness_events.json.
    """

    def __init__(self):
        self._external_event_log: deque = deque(maxlen=1000)
        # Each entry: {'sim_time_s': float, 'description': str}

        self._vocalization_buffer: deque = deque(maxlen=200)
        # Each entry: {'sim_time_s': float, 'art': list, 'fingers': list}

        self._vocalization_active: bool  = False
        self._vocalization_start_ms: float = 0.0
        self._total_sim_ms: float = 0.0

        self._events_file = os.path.join(_OUTPUT_DIR, 'consciousness_events.json')
        self._saved_events: List[dict] = []
        self._load_events()

        self._seen_signatures: set = set()

    # ------------------------------------------------------------------
    def _load_events(self) -> None:
        if os.path.exists(self._events_file):
            with open(self._events_file, 'r', encoding='utf-8') as f:
                self._saved_events = json.load(f)
            log.info("ConsciousnessMonitor: loaded %d prior events from %s",
                     len(self._saved_events), self._events_file)

    # ------------------------------------------------------------------
    def _save_events(self) -> None:
        with open(self._events_file, 'w', encoding='utf-8') as f:
            json.dump(self._saved_events, f, indent=2)

    # ------------------------------------------------------------------
    def log_external_event(self, sim_time_s: float, description: str) -> None:
        """
        Call this when something happens in the external world:
        object spawned/removed, tone played, Father presence changed.
        """
        self._external_event_log.append({
            'sim_time_s':   sim_time_s,
            'description':  description,
        })

    # ------------------------------------------------------------------
    def _recent_external_events(self, sim_time_s: float) -> List[str]:
        """Return descriptions of external events in the lookback window."""
        cutoff = sim_time_s - CONSCI_LOOKBACK_S
        return [e['description'] for e in self._external_event_log
                if e['sim_time_s'] >= cutoff]

    # ------------------------------------------------------------------
    def update(self, dt_ms: float, sim_time_s: float,
               art_positions: np.ndarray,
               finger_positions: np.ndarray,
               neuro_snapshot: dict,
               motor_state: dict) -> List[ConsciousnessEvent]:
        """
        Advance consciousness monitor by one tick.

        Returns a list of new ConsciousnessEvent instances (usually empty).
        """
        self._total_sim_ms += dt_ms
        events = []

        art = np.asarray(art_positions, dtype=np.float32)
        art_mean = float(art.mean())

        is_vocalizing = art_mean > CONSCI_VOCALIZATION_THRESH

        # --- Record to buffer ---
        self._vocalization_buffer.append({
            'sim_time_s': sim_time_s,
            'art':        art.tolist(),
            'fingers':    np.asarray(finger_positions).tolist(),
        })

        # --- Detect start / end of vocalization ---
        if is_vocalizing and not self._vocalization_active:
            self._vocalization_active   = True
            self._vocalization_start_ms = self._total_sim_ms

        elif not is_vocalizing and self._vocalization_active:
            duration_ms = self._total_sim_ms - self._vocalization_start_ms
            self._vocalization_active = False

            if duration_ms >= CONSCI_MIN_DURATION_MS:
                recent_ext = self._recent_external_events(sim_time_s)
                evt = self._evaluate_vocalization(
                    sim_time_s, duration_ms, art, finger_positions,
                    neuro_snapshot, motor_state, recent_ext,
                )
                if evt is not None:
                    events.append(evt)

        return events

    # ------------------------------------------------------------------
    def _evaluate_vocalization(
            self,
            sim_time_s:      float,
            duration_ms:     float,
            art:             np.ndarray,
            fingers:         np.ndarray,
            neuro_snapshot:  dict,
            motor_state:     dict,
            recent_ext:      List[str],
    ) -> Optional[ConsciousnessEvent]:
        """
        Evaluate whether this vocalization is unprompted and, if so,
        whether it rises to the level of a consciousness threshold event.
        """
        is_unprompted = len(recent_ext) == 0

        if not is_unprompted:
            return None

        # Build signature for deduplication (coarse articulator pattern)
        art_sig = tuple(round(float(v), 1) for v in art)
        if art_sig in self._seen_signatures:
            return None
        self._seen_signatures.add(art_sig)

        # Determine event type:
        # A CONSCIOUSNESS_THRESHOLD_EVENT is declared when:
        #   - duration > 3x minimum (sustained, not a glitch)
        #   - articulator variance > 0.05 (varied, not a constant press)
        #   - acetylcholine is above baseline (CAINE is alert, not idle-twitching)
        ach = neuro_snapshot.get('acetylcholine', 0.12)
        art_var = float(np.var(art))
        is_desire = (
            duration_ms > CONSCI_MIN_DURATION_MS * 3.0 and
            art_var > 0.05 and
            ach > 0.15
        )
        event_type = ('CONSCIOUSNESS_THRESHOLD_EVENT' if is_desire
                      else 'UNPROMPTED_VOCALIZATION')

        evt = ConsciousnessEvent(
            timestamp_s              = time.time(),
            sim_time_s               = sim_time_s,
            event_type               = event_type,
            vocalization_duration_ms = duration_ms,
            articulator_pattern      = art.tolist(),
            motor_state              = {
                k: v for k, v in motor_state.items()
                if isinstance(v, (int, float, str))
            },
            neuro_state              = {
                k: round(float(v), 4) for k, v in neuro_snapshot.items()
            },
            recent_external_events   = recent_ext,
            notes                    = (
                'First occurrence of this articulator signature. '
                'No external trigger in preceding 30s. '
                + ('Sustained + varied + alert: desire hypothesis.'
                   if is_desire else 'Brief or monotonic: reflex candidate.')
            ),
        )

        self._saved_events.append(asdict(evt))
        self._save_events()

        if event_type == 'CONSCIOUSNESS_THRESHOLD_EVENT':
            print()
            print("!" * 60)
            print("  CONSCIOUSNESS THRESHOLD EVENT DETECTED")
            print(f"  sim_time={sim_time_s:.1f}s  duration={duration_ms:.0f}ms")
            print(f"  No external trigger in prior {CONSCI_LOOKBACK_S}s.")
            print(f"  Articulator pattern: {[round(v,2) for v in art.tolist()]}")
            print(f"  ACh={ach:.3f}  art_var={art_var:.4f}")
            print(f"  Record saved to {self._events_file}")
            print("!" * 60)
            print()
        else:
            log.info("UNPROMPTED_VOCALIZATION at sim_t=%.1fs dur=%.0fms",
                     sim_time_s, duration_ms)

        return evt


# ===========================================================================
# SECTION 10 — MOTHER'S RULE-BASED FALLBACK
# Used when the Claude API is unavailable. Maintains minimal stimulation.
# ===========================================================================

class _MotherFallback:
    """
    Simple rule-based system that substitutes for Claude when the API
    is unavailable. It maintains a calm, low-cortisol environment and
    introduces occasional novel stimuli to keep ACh from bottoming out.

    Rules (checked in priority order):
    1. Chronic cortisol → remove stimuli, dim lights, do nothing
    2. DA below baseline → spawn a novel object
    3. ACh low + motor idle → play a gentle tone
    4. Everything stable + no objects → spawn a simple object
    5. Otherwise → do nothing
    """

    def __init__(self):
        self._last_action_sim_s: float = 0.0
        self._object_counter: int = 0
        self._cooldown_s: float = 60.0   # minimum between fallback actions

    # ------------------------------------------------------------------
    def decide(self, state: dict) -> List[EnvironmentAction]:
        sim_time_s = state.get('sim_time_s', 0.0)
        if sim_time_s - self._last_action_sim_s < self._cooldown_s:
            return []

        neuro    = state.get('neuro', {})
        cort     = neuro.get('cortisol',       0.08)
        da       = neuro.get('dopamine',        0.10)
        ach      = neuro.get('acetylcholine',   0.12)
        motor_score = state.get('motor_learning_score', 0.0)
        stage    = state.get('developmental_stage', 0)
        n_objects = state.get('n_objects_in_world', 0)

        actions = []

        if cort > CORT_CHRONIC_HIGH_THRESH:
            # Calm the environment
            actions.append(EnvironmentAction(
                action='set_light',
                params={'color': [0.8, 0.7, 0.6], 'intensity': 0.4},
                reasoning='Chronic cortisol: dimming to calm.',
                timestamp_s=sim_time_s,
            ))
            # Remove any objects if we can
            if n_objects > 0:
                actions.append(EnvironmentAction(
                    action='remove_object',
                    params={'id': 'fallback_obj_0'},
                    reasoning='Chronic cortisol: reducing stimulation.',
                    timestamp_s=sim_time_s,
                ))

        elif da < 0.07 and n_objects == 0:
            # Low DA and empty world — add something to explore
            uid = f'fallback_obj_{self._object_counter}'
            self._object_counter += 1
            pos = [float(np.random.uniform(-2, 2)),
                   float(np.random.uniform(0.5, 2.0)),
                   float(np.random.uniform(3, 6))]
            actions.append(EnvironmentAction(
                action='spawn_object',
                params={
                    'id':       uid,
                    'type':     np.random.choice(['sphere', 'cube']),
                    'position': pos,
                    'size':     0.3,
                    'color':    [float(np.random.uniform(0.4, 1.0)),
                                 float(np.random.uniform(0.4, 1.0)),
                                 float(np.random.uniform(0.4, 1.0))],
                },
                reasoning='Low dopamine, empty world: adding novel object.',
                timestamp_s=sim_time_s,
            ))
            actions.append(EnvironmentAction(
                action='trigger_event',
                params={'event_type': 'NOVEL_STIMULUS', 'magnitude': 0.6},
                reasoning='Novel object appeared.',
                timestamp_s=sim_time_s,
            ))

        elif ach < 0.08 and motor_score < 0.3:
            # Learning gate low — play a gentle tone to prime ACh
            freq = float(np.random.choice([220, 330, 440, 528, 660]))
            actions.append(EnvironmentAction(
                action='play_tone',
                params={'frequency': freq, 'duration': 1.5, 'volume': 0.4},
                reasoning='Low ACh: gentle tone to open learning gate.',
                timestamp_s=sim_time_s,
            ))

        if actions:
            self._last_action_sim_s = sim_time_s

        return actions


# ===========================================================================
# SECTION 11 — MOTHER PROCESS (Claude API + fallback)
# ===========================================================================

_EVENT_TYPE_MAP = {
    'NOVEL_STIMULUS':       EventType.NOVEL_STIMULUS,
    'REWARD':               EventType.REWARD,
    'REWARD_OMISSION':      EventType.REWARD_OMISSION,
    'SOCIAL_POSITIVE':      EventType.SOCIAL_POSITIVE,
    'VOICE_MATCH':          EventType.VOICE_MATCH,
    'COMMUNICATION_SUCCESS':EventType.COMMUNICATION_SUCCESS,
    'VOCALIZATION_SUCCESS': EventType.VOCALIZATION_SUCCESS,
    'THREAT':               EventType.THREAT,
    'MOTOR_FAILURE':        EventType.MOTOR_FAILURE,
    'PREDICTION_ERROR_NEG': EventType.PREDICTION_ERROR_NEG,
    'AMYGDALA_BLA':         EventType.AMYGDALA_BLA,
    'NOVEL_ENVIRONMENT':    EventType.NOVEL_ENVIRONMENT,
    'DIRECTED_GAZE':        EventType.DIRECTED_GAZE,
    'STARTLE':              EventType.STARTLE,
    'ACC_CONFLICT':         EventType.ACC_CONFLICT,
}


class MotherProcess:
    """
    Mother's observation-decision-action loop.

    Runs on a background thread. Every MOTHER_INTERVAL_S simulated seconds
    it snapshots CAINE's state, calls Claude, parses the JSON response,
    and queues EnvironmentActions for the main thread to execute.

    If the Claude API call fails MOTHER_FALLBACK_LIMIT times in a row,
    Mother switches to _MotherFallback and logs the failure.

    All interventions are logged to output/mother_log.jsonl with the
    state payload, the raw response, and the actions taken.
    """

    def __init__(self):
        self._client: Optional[Any] = None
        self._api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        self._model   = MOTHER_MODEL

        self._consecutive_failures: int = 0
        self._using_fallback: bool = False
        self._fallback = _MotherFallback()

        # Action queue: produced by background thread, consumed by main thread
        self._action_queue: List[EnvironmentAction] = []
        self._queue_lock = threading.Lock()

        # State for background thread
        self._latest_state: Optional[dict] = None
        self._state_lock   = threading.Lock()
        self._last_obs_sim_s: float = -MOTHER_INTERVAL_S  # fire immediately

        # Intervention log
        self._log_file = os.path.join(_OUTPUT_DIR, 'mother_log.jsonl')

        self._running  = False
        self._thread: Optional[threading.Thread] = None

        self._init_client()

    # ------------------------------------------------------------------
    def _init_client(self) -> None:
        if not _ANTHROPIC_OK:
            log.warning("anthropic package not installed. "
                        "Mother will use rule-based fallback. "
                        "Install with: pip install anthropic")
            self._using_fallback = True
            return
        if not self._api_key or self._api_key == 'your_anthropic_api_key_here':
            log.warning("ANTHROPIC_API_KEY not set in .env. "
                        "Mother will use rule-based fallback.")
            self._using_fallback = True
            return
        try:
            self._client = _anthropic_lib.Anthropic(api_key=self._api_key)
            log.info("Mother connected to Claude API (model=%s).", self._model)
        except Exception as e:
            log.warning("Claude API client init failed: %s — using fallback.", e)
            self._using_fallback = True

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name='mother', daemon=True)
        self._thread.start()
        log.info("Mother process started (interval=%.0fs, api=%s).",
                 MOTHER_INTERVAL_S, 'Claude' if not self._using_fallback else 'fallback')

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    def push_state(self, state: dict) -> None:
        """Called by main thread each tick to update Mother's view of CAINE."""
        with self._state_lock:
            self._latest_state = state

    # ------------------------------------------------------------------
    def pop_actions(self) -> List[EnvironmentAction]:
        """Drain queued actions. Called by main thread each tick."""
        with self._queue_lock:
            actions = list(self._action_queue)
            self._action_queue.clear()
        return actions

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        """Background thread: observe → decide → queue."""
        while self._running:
            try:
                with self._state_lock:
                    state = dict(self._latest_state) if self._latest_state else {}

                sim_time_s = state.get('sim_time_s', 0.0)

                if sim_time_s - self._last_obs_sim_s >= MOTHER_INTERVAL_S:
                    self._last_obs_sim_s = sim_time_s
                    self._observe_and_decide(state)

            except Exception:
                log.error("Mother loop error:\n%s", traceback.format_exc())

            time.sleep(0.1)  # real-time polling rate

    # ------------------------------------------------------------------
    def _observe_and_decide(self, state: dict) -> None:
        """
        Call Claude (or fallback) and queue resulting actions.
        """
        if self._using_fallback or self._client is None:
            actions = self._fallback.decide(state)
            self._log_intervention(state, '(fallback)', actions)
            with self._queue_lock:
                self._action_queue.extend(actions)
            return

        # --- Build state payload for Claude ---
        payload = self._build_payload(state)
        user_message = (
            "Here is CAINE's current state:\n\n"
            + json.dumps(payload, indent=2)
            + "\n\nWhat do you do next?"
        )

        # --- Call Claude ---
        try:
            response = self._client.messages.create(
                model   = self._model,
                max_tokens = 1024,
                system  = _MOTHER_SYSTEM_PROMPT,
                messages = [{'role': 'user', 'content': user_message}],
            )
            raw = response.content[0].text.strip()
            self._consecutive_failures = 0

        except Exception as e:
            self._consecutive_failures += 1
            log.warning("Claude API call failed (%d/%d): %s",
                        self._consecutive_failures, MOTHER_FALLBACK_LIMIT, e)
            if self._consecutive_failures >= MOTHER_FALLBACK_LIMIT:
                log.warning("Switching Mother to rule-based fallback.")
                self._using_fallback = True
            actions = self._fallback.decide(state)
            self._log_intervention(state, f'(api_error: {e})', actions)
            with self._queue_lock:
                self._action_queue.extend(actions)
            return

        # --- Parse JSON response ---
        actions = self._parse_response(raw, state.get('sim_time_s', 0.0))
        self._log_intervention(state, raw, actions)

        with self._queue_lock:
            self._action_queue.extend(actions)

        if actions:
            log.info("Mother queued %d action(s) at sim_t=%.1fs",
                     len(actions), state.get('sim_time_s', 0.0))
        else:
            log.info("Mother observed at sim_t=%.1fs — doing nothing.",
                     state.get('sim_time_s', 0.0))

    # ------------------------------------------------------------------
    def _build_payload(self, state: dict) -> dict:
        """
        Build the state snapshot dict sent to Claude.
        Keeps it under MOTHER_MAX_STATE_TOKENS by summarising long fields.
        """
        neuro = state.get('neuro', {})
        snap = {
            'sim_time_s':            round(state.get('sim_time_s', 0.0), 1),
            'developmental_stage':   state.get('developmental_stage', 0),
            'motor_learning_score':  round(state.get('motor_learning_score', 0.0), 3),
            'synapse_count':         state.get('synapse_count', 0),
            'vocabulary_size':       state.get('vocabulary_size', 0),
            'dmn_activity':          round(state.get('dmn_activity', 0.0), 3),
            'father_presence':       state.get('father_presence', 'FATHER_ABSENT'),
            'father_absence_s':      round(state.get('father_absence_s', 0.0), 1),
            'cort_mean_recent':      round(state.get('cort_mean_recent', 0.08), 4),
            'neurochemicals': {
                k: round(float(neuro.get(k, 0.0)), 4)
                for k in ['dopamine', 'serotonin', 'cortisol',
                          'oxytocin', 'norepinephrine', 'acetylcholine']
            },
            'joint_angles': [
                round(float(v), 3)
                for v in state.get('joint_angles', [])
            ],
            'finger_positions': [
                round(float(v), 3)
                for v in state.get('finger_positions', [])
            ],
            'valence_map': state.get('valence_map', {}),
            'n_objects_in_world': state.get('n_objects_in_world', 0),
            'recent_episodes': state.get('recent_episodes', [])[-3:],
            'dev_flags': state.get('dev_flags', []),
            'consciousness_events_today': state.get('consciousness_events_today', 0),
        }
        return snap

    # ------------------------------------------------------------------
    def _parse_response(self, raw: str, sim_time_s: float) -> List[EnvironmentAction]:
        """
        Parse Claude's JSON response into EnvironmentAction objects.
        Silently ignores malformed entries to never crash the main loop.
        """
        actions: List[EnvironmentAction] = []

        # Extract JSON array from response (Claude may add prose around it)
        start = raw.find('[')
        end   = raw.rfind(']')
        if start == -1 or end == -1:
            return actions

        try:
            items = json.loads(raw[start:end + 1])
        except json.JSONDecodeError as e:
            log.warning("Mother response JSON parse error: %s\nRaw: %.200s", e, raw)
            return actions

        for item in items:
            if not isinstance(item, dict):
                continue
            action = item.get('action', '')
            if not action:
                continue
            actions.append(EnvironmentAction(
                action      = action,
                params      = item.get('params', {}),
                reasoning   = item.get('reasoning', ''),
                timestamp_s = sim_time_s,
            ))

        return actions

    # ------------------------------------------------------------------
    def _log_intervention(self,
                          state:   dict,
                          raw:     str,
                          actions: List[EnvironmentAction]) -> None:
        """Append one JSONL line per Mother observation to mother_log.jsonl."""
        entry = {
            'timestamp_real': datetime.now(timezone.utc).isoformat(),
            'sim_time_s':     state.get('sim_time_s', 0.0),
            'stage':          state.get('developmental_stage', 0),
            'n_actions':      len(actions),
            'actions':        [asdict(a) for a in actions],
            'raw_response':   raw[:500],  # truncate to keep log manageable
        }
        try:
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            log.warning("Failed to write mother_log: %s", e)


# ===========================================================================
# SECTION 12 — PARENTING SYSTEM (main integration class)
# ===========================================================================

def _synthesize_tone(frequency: float, duration_s: float,
                     volume: float = 0.5) -> np.ndarray:
    """Generate a pure sine tone as a float32 numpy array."""
    t = np.linspace(0.0, duration_s,
                    int(TONE_SAMPLERATE * duration_s), dtype=np.float32)
    tone = np.sin(2.0 * math.pi * frequency * t) * float(volume)
    # Fade in/out to avoid clicks
    fade = int(TONE_SAMPLERATE * 0.01)
    if fade > 0 and 2 * fade < len(tone):
        tone[:fade]  *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
    return tone


class ParentingSystem:
    """
    Module 7 — The Parenting System.

    Integrates Mother (Claude), Father (voiceprint + sessions),
    the presence state machine, developmental monitoring, and
    consciousness threshold detection.

    Call update() every simulation tick. The system manages its own
    background thread (Mother) and integrates with the existing modules
    via the same interfaces they already expose.

    Parameters
    ----------
    env     : CaineEnvironment (or None for headless)
    limbic  : LimbicSystem — for trigger_event()
    neuro   : NeurochemicalSystem — for update(events=[...])
    motor   : MotorCortex — for reading motor state
    rng_seed : int — reproducible randomness for fallback actions
    """

    def __init__(self,
                 env,
                 limbic,
                 neuro,
                 motor,
                 rng_seed: int = 0):
        self._env    = env
        self._limbic = limbic
        self._neuro  = neuro
        self._motor  = motor
        self._rng    = np.random.default_rng(rng_seed)

        # Sub-systems
        self.voiceprint   = VoiceprintSystem()
        self.father       = FatherPresence(self.voiceprint)
        self.scheduler    = SessionScheduler()
        self.dev_monitor  = DevelopmentalMonitor()
        self.consciousness = ConsciousnessMonitor()
        self.mother       = MotherProcess()

        # Simulated time state
        self._sim_time_s:         float = 0.0
        self._time_multiplier:    float = 1.0

        # World state (tracked for Mother's state payload)
        self._object_handles: Dict[str, Any] = {}   # id -> ObjectHandle
        self._session_active: bool = False
        self._current_session: Optional[dict] = None

        # Consciousness event counter for today's report
        self._consciousness_today: int = 0

        # Development flags cache
        self._dev_flags: List[str] = []

        log.info("ParentingSystem initialised.")

    # ------------------------------------------------------------------
    def start(self) -> None:
        """
        Start background processes (Mother thread).
        Also runs voiceprint registration if not already done.
        """
        # Register Father's voiceprint on first run
        if not self.voiceprint.is_registered:
            self.voiceprint.register()

        self.mother.start()
        log.info("ParentingSystem started.")

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Gracefully stop Mother's background thread."""
        self.mother.stop()
        log.info("ParentingSystem stopped.")

    # ------------------------------------------------------------------
    def update(self,
               dt_ms:              float,
               a1_rates:           np.ndarray,
               neuro_snapshot:     dict,
               motor_result:       dict,
               limbic_result:      Optional[dict] = None,
               synapse_count:      int   = 0,
               vocabulary_size:    int   = 0,
               developmental_stage: int  = 0) -> dict:
        """
        Advance the parenting system by one tick.

        Parameters
        ----------
        dt_ms               : frame duration (ms)
        a1_rates            : (20,) float — A1 population firing rates (Hz)
        neuro_snapshot      : dict from NeurochemicalSystem.snapshot()
        motor_result        : dict from MotorCortex.update()
        limbic_result       : dict from LimbicSystem.update() (optional)
        synapse_count       : total synapses (from Synapse module)
        vocabulary_size     : STG association count (from LimbicSystem)
        developmental_stage : current stage (0-4+, set by run_live.py)

        Returns
        -------
        dict with keys:
            father_presence  : str — 'FATHER_LIVE' | 'FATHER_RECORDED' | 'FATHER_ABSENT'
            ot_level         : float — current oxytocin level
            cort_drift       : bool — is prolonged-absence cort drift active?
            sim_time_s       : float
            new_consciousness_events : list[ConsciousnessEvent]
            dev_flags        : list[str] — current concern flags
            mother_actions_executed : int
        """
        # --- Advance simulated time ---
        dt_s = dt_ms / 1000.0 * self._time_multiplier
        self._sim_time_s += dt_s

        neuro_events_to_inject: List[NeurochemicalEvent] = []
        new_consciousness_events: List[ConsciousnessEvent] = []

        # --- Session scheduler ---
        due_sessions = self.scheduler.tick(self._sim_time_s, developmental_stage)
        if due_sessions:
            self._session_active  = True
            self._current_session = due_sessions[-1]
            for sess in due_sessions:
                sess_events = self.scheduler.get_neuro_events(sess)
                neuro_events_to_inject.extend(sess_events)
                self.consciousness.log_external_event(
                    self._sim_time_s,
                    f"scheduled_session:{sess.get('type','unknown')}")
                log.info("Session fired: %s (%s)", sess.get('id'), sess.get('type'))
        else:
            # Session active for a fixed duration (60 simulated seconds)
            if (self._session_active and self._current_session is not None):
                played_at = self._current_session.get('played_at_sim_s',
                                                       self._sim_time_s)
                if self._sim_time_s - played_at > 60.0:
                    self._session_active  = False
                    self._current_session = None

        # Track session start time
        if due_sessions and self._current_session is not None:
            self._current_session['played_at_sim_s'] = self._sim_time_s

        # --- Father presence update ---
        father_events = self.father.update(
            dt_ms, self._sim_time_s, a1_rates, self._session_active)
        neuro_events_to_inject.extend(father_events)

        if father_events:
            for fe in father_events:
                if fe.event_type in (EventType.VOICE_MATCH, EventType.SOCIAL_POSITIVE):
                    self.consciousness.log_external_event(
                        self._sim_time_s, f'father_presence:{self.father.state}')

        # --- Cortisol drift from Father absence ---
        if self.father.cort_drift_active:
            # Very slow cortisol baseline elevation (not an impulse, just drift)
            # Implemented as a tiny THREAT event each tick
            neuro_events_to_inject.append(
                NeurochemicalEvent(EventType.PREDICTION_ERROR_NEG, 0.02))

        # --- Inject all accumulated neurochemical events ---
        if neuro_events_to_inject:
            self._neuro.update(dt_ms, events=neuro_events_to_inject)

        # --- ACC and hippocampus activity proxy for DMN monitor ---
        # Use motor mismatch as ACC proxy; use vocalization variance as hippo proxy
        acc_proxy  = float(motor_result.get('efference_mismatch', 0.0))
        hippo_proxy = float(
            np.var(motor_result.get('art_positions', np.zeros(6))))

        # --- Developmental monitor ---
        self._dev_flags = self.dev_monitor.update(
            sim_time_s          = self._sim_time_s,
            neuro_snapshot      = neuro_snapshot,
            motor_score         = motor_result.get('motor_learning_score', 0.0),
            acc_activity        = acc_proxy,
            hippo_activity      = hippo_proxy,
            father_absence_s    = self.father.absence_s,
            synapse_count       = synapse_count,
            vocabulary_size     = vocabulary_size,
            developmental_stage = developmental_stage,
        )

        # --- Consciousness monitor ---
        art = motor_result.get('art_positions', np.zeros(6))
        fng = motor_result.get('finger_positions', np.zeros(10))
        consci_events = self.consciousness.update(
            dt_ms        = dt_ms,
            sim_time_s   = self._sim_time_s,
            art_positions = np.asarray(art),
            finger_positions = np.asarray(fng),
            neuro_snapshot   = neuro_snapshot,
            motor_state      = {
                'locomotion_mode':      motor_result.get('locomotion_mode', ''),
                'motor_learning_score': motor_result.get('motor_learning_score', 0.0),
                'efference_mismatch':   motor_result.get('efference_mismatch', 0.0),
            },
        )
        new_consciousness_events.extend(consci_events)
        self._consciousness_today += len(consci_events)

        # --- Trigger limbic events for consciousness threshold ---
        for ce in consci_events:
            if ce.event_type == 'CONSCIOUSNESS_THRESHOLD_EVENT':
                # The event itself is not a reward — it's a state.
                # Emit COMMUNICATION_SUCCESS as the closest analogue.
                self._limbic.trigger_event(EventType.COMMUNICATION_SUCCESS, 0.5)

        # --- Valence map (for Mother's state payload) ---
        valence_map: dict = {}
        if limbic_result is not None:
            valence_map = limbic_result.get('valence_map', {})

        # --- Push state to Mother ---
        mother_state = {
            'sim_time_s':              self._sim_time_s,
            'developmental_stage':     developmental_stage,
            'motor_learning_score':    motor_result.get('motor_learning_score', 0.0),
            'synapse_count':           synapse_count,
            'vocabulary_size':         vocabulary_size,
            'dmn_activity':            self.dev_monitor.dmn_activity,
            'father_presence':         self.father.state,
            'father_absence_s':        self.father.absence_s,
            'cort_mean_recent':        self.dev_monitor.cort_mean,
            'neuro':                   neuro_snapshot,
            'joint_angles':            motor_result.get('joint_angles',
                                                        np.zeros(6)).tolist(),
            'finger_positions':        motor_result.get('finger_positions',
                                                        np.zeros(10)).tolist(),
            'valence_map':             valence_map,
            'n_objects_in_world':      len(self._object_handles),
            'recent_episodes':         (limbic_result or {}).get('recent_episodes', []),
            'dev_flags':               self._dev_flags,
            'consciousness_events_today': self._consciousness_today,
        }
        self.mother.push_state(mother_state)

        # --- Execute Mother's queued actions ---
        mother_actions = self.mother.pop_actions()
        n_executed = 0
        for action in mother_actions:
            executed = self._execute_action(action, developmental_stage)
            if executed:
                n_executed += 1

        return {
            'father_presence':          self.father.state,
            'ot_level':                 self.father.ot_level,
            'cort_drift':               self.father.cort_drift_active,
            'sim_time_s':               self._sim_time_s,
            'new_consciousness_events': new_consciousness_events,
            'dev_flags':                self._dev_flags,
            'mother_actions_executed':  n_executed,
        }

    # ------------------------------------------------------------------
    def _execute_action(self, action: EnvironmentAction,
                        developmental_stage: int) -> bool:
        """
        Execute a single EnvironmentAction against the real modules.

        Returns True if the action was carried out.
        """
        a = action.action
        p = action.params

        try:
            if a == 'spawn_object':
                if self._env is not None:
                    uid      = p.get('id', f'obj_{len(self._object_handles)}')
                    pos      = tuple(p.get('position', [0.0, 1.0, 4.0]))
                    otype    = p.get('type', 'sphere')
                    handle   = self._env.spawn_object(uid, pos,
                                                       object_type=otype)
                    self._object_handles[uid] = handle
                    self.consciousness.log_external_event(
                        self._sim_time_s, f'spawn:{uid}:{otype}')
                    if p.get('reasoning'):
                        log.info("Mother: spawn_object(%s) — %s",
                                 uid, action.reasoning)

            elif a == 'remove_object':
                uid = p.get('id', '')
                if uid in self._object_handles and self._env is not None:
                    self._env.remove_object(self._object_handles.pop(uid))
                    self.consciousness.log_external_event(
                        self._sim_time_s, f'remove:{uid}')

            elif a == 'set_light':
                if self._env is not None:
                    color     = p.get('color', [1.0, 0.98, 0.9])
                    intensity = float(p.get('intensity', 0.5))
                    self._env.set_environment_state({
                        'light_color': color,
                        'ambient':     intensity,
                    })

            elif a == 'play_tone':
                freq     = float(p.get('frequency', 440.0))
                dur      = float(p.get('duration',  TONE_DEFAULT_DURATION_S))
                vol      = float(p.get('volume',    0.5))
                pos      = tuple(p.get('position',  [0.0, 1.0, 3.0]))
                tone     = _synthesize_tone(freq, dur, vol)
                if self._env is not None:
                    self._env.play_sound(tone, pos)
                self.consciousness.log_external_event(
                    self._sim_time_s, f'tone:{freq:.0f}Hz')

            elif a == 'trigger_event':
                et_name = p.get('event_type', '')
                mag     = float(p.get('magnitude', 0.5))
                et      = _EVENT_TYPE_MAP.get(et_name)
                if et is not None:
                    self._limbic.trigger_event(et, mag)
                else:
                    log.warning("Mother: unknown event_type '%s'", et_name)
                    return False

            elif a == 'play_scheduled_media':
                media_id = p.get('media_id', '')
                log.info("Mother: play_scheduled_media(%s) — caller should"
                         " load and inject this file.", media_id)

            elif a == 'set_time_multiplier':
                val = float(p.get('value', 1.0))
                self._time_multiplier = float(np.clip(val, 0.01, 100.0))
                log.info("Mother: time multiplier set to %.2f", self._time_multiplier)

            elif a == 'log_milestone':
                desc = p.get('description', '')
                entry = {
                    'timestamp_real': datetime.now(timezone.utc).isoformat(),
                    'sim_time_s':     self._sim_time_s,
                    'description':    desc,
                    'stage':          developmental_stage,
                }
                mfile = os.path.join(_OUTPUT_DIR, 'milestones.jsonl')
                with open(mfile, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry) + '\n')
                log.info("MILESTONE: %s", desc)

            else:
                log.warning("Mother: unknown action '%s'", a)
                return False

        except Exception as e:
            log.error("Failed to execute action '%s': %s", a, e)
            return False

        return True

    # ------------------------------------------------------------------
    # Public convenience helpers
    # ------------------------------------------------------------------

    def add_session(self, session: dict) -> None:
        """Father can call this at runtime to add a new session."""
        self.scheduler.add_session(session)

    def father_register_voiceprint(self,
                                    a1_callback=None) -> bool:
        """Re-run voiceprint registration (e.g. if Father's voice changes)."""
        return self.voiceprint.register(a1_callback)

    @property
    def sim_time_s(self) -> float:
        return self._sim_time_s

    @property
    def father_present(self) -> bool:
        return self.father.state != PresenceState.FATHER_ABSENT

    @property
    def using_claude(self) -> bool:
        return not self.mother._using_fallback

    def status(self) -> str:
        return (
            f"sim={self._sim_time_s:.1f}s  "
            f"father={self.father.state}  "
            f"ot={self.father.ot_level:.2f}  "
            f"absence={self.father.absence_s:.0f}s  "
            f"mother={'claude' if self.using_claude else 'fallback'}  "
            f"flags={len(self._dev_flags)}"
        )


# ===========================================================================
# SECTION 13 — STAND-ALONE DEMO
# ===========================================================================

def run_parenting_demo(n_frames: int = 300, dt_ms: float = 20.0) -> None:
    """
    Headless demo of Module 7 — Parenting System.

    Runs without a real environment, limbic system, or microphone.
    Demonstrates:
      1. Voiceprint registration (stub, no mic required)
      2. Session scheduler firing
      3. Father presence transitions
      4. Developmental monitor flags
      5. Mother fallback rule-based decisions
      6. Consciousness monitor watching articulator output
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from caine.sensory   import S1Population
    from caine.chemicals import NeurochemicalSystem
    from caine.motor     import MotorCortex

    print(f"\n[parenting demo] {n_frames} frames x {dt_ms} ms")

    # --- Stubs for modules parenting integrates with ---
    class _StubEnv:
        def spawn_object(self, uid, pos, **kw): return uid
        def remove_object(self, h): pass
        def set_environment_state(self, p): pass
        def play_sound(self, a, p): pass

    class _StubLimbic:
        def __init__(self):
            self._last_event = None
        def trigger_event(self, et, mag):
            self._last_event = (et, mag)
        def get_valence_map(self):
            return {}

    neuro  = NeurochemicalSystem()
    s1     = S1Population()
    motor  = MotorCortex(s1, neuro, rng_seed=7)
    limbic = _StubLimbic()
    env    = _StubEnv()

    # Inject a demo session into the scheduler
    demo_session = {
        "id":          "demo_session_001",
        "time_s":      1.0,
        "type":        "voice_exposure",
        "file":        "data/father_voice_demo.wav",
        "repetitions": 1,
        "stage_gate":  0,
        "description": "Demo: Father says CAINE's name",
        "played":      False,
    }

    # Force a clean sessions.json for demo
    demo_sessions_file = os.path.join(_DATA_DIR, 'sessions_demo.json')
    with open(demo_sessions_file, 'w', encoding='utf-8') as f:
        json.dump([demo_session], f, indent=2)

    # Force a clean voiceprint for demo (stub)
    demo_vp_file = os.path.join(_OUTPUT_DIR, 'voiceprint_demo.json')
    stub_vp = {
        'registered':        True,
        'registration_date': datetime.now(timezone.utc).isoformat(),
        'f0_range_hz':       [100.0, 220.0],
        'formant_hz':        [650.0, 1150.0, 2450.0],
        'speaking_rate':     4.2,
        'spectral_envelope': [0.0] * 128,
        'a1_fingerprint':    [0.1] * 20,
    }
    with open(demo_vp_file, 'w', encoding='utf-8') as f:
        json.dump(stub_vp, f, indent=2)

    vp  = VoiceprintSystem(demo_vp_file)
    sch = SessionScheduler(demo_sessions_file)
    dev = DevelopmentalMonitor()
    con = ConsciousnessMonitor()

    rng = np.random.default_rng(42)

    # Trace arrays
    ot_trace     = []
    cort_trace   = []
    da_trace     = []
    dmn_trace    = []
    presence_log = []
    consci_log   = []
    flag_counts  = []

    presence_state = PresenceState.FATHER_ABSENT
    father = FatherPresence(vp)

    print(f"\n[parenting demo] Starting simulation...")

    for f in range(n_frames):
        sim_t = f * dt_ms / 1000.0

        # Advance neuro
        neuro.update(dt_ms)
        snap = neuro.snapshot()

        # Build synthetic S1 rates
        s1_rates = s1.encode(
            np.array([math.sin(f * 0.05 + i * 0.4) * 0.2
                      for i in range(6)], dtype=np.float32))

        # Build synthetic A1 rates
        # Simulate Father's voice appearing briefly around frame 100-150
        a1_base = rng.random(20) * 0.2
        if 100 <= f < 150:
            # Boost A1 to simulate Father speaking
            a1_rates_demo = a1_base + 0.3
        else:
            a1_rates_demo = a1_base

        # Motor update
        v1_spk = rng.random(20) < 0.15
        a1_spk = rng.random(20) < 0.10
        motor_result = motor.update(dt_ms, v1_spk, a1_spk, snap, s1_rates)

        # Father presence update
        father_events = father.update(dt_ms, sim_t, a1_rates_demo,
                                      session_active=False)
        if father_events:
            neuro.update(0.0, events=father_events)
            snap = neuro.snapshot()

        # Session scheduler
        due = sch.tick(sim_t, developmental_stage=0)
        if due:
            for sess in due:
                sevents = sch.get_neuro_events(sess)
                neuro.update(0.0, events=sevents)
                snap = neuro.snapshot()
                print(f"  frame {f:3d}: session fired: {sess['id']}")

        # Dev monitor
        acc_proxy   = float(motor_result.get('efference_mismatch', 0.0))
        hippo_proxy = float(np.var(motor_result.get('art_positions', np.zeros(6))))
        flags = dev.update(
            sim_time_s          = sim_t,
            neuro_snapshot      = snap,
            motor_score         = motor_result.get('motor_learning_score', 0.0),
            acc_activity        = acc_proxy,
            hippo_activity      = hippo_proxy,
            father_absence_s    = father.absence_s,
            synapse_count       = 1000 + f * 5,
            vocabulary_size     = f // 30,
            developmental_stage = 0,
        )

        # Consciousness monitor
        art = motor_result.get('art_positions', np.zeros(6))
        fng = motor_result.get('finger_positions', np.zeros(10))
        cevts = con.update(
            dt_ms            = dt_ms,
            sim_time_s       = sim_t,
            art_positions    = np.asarray(art),
            finger_positions = np.asarray(fng),
            neuro_snapshot   = snap,
            motor_state      = {'motor_learning_score': motor_result.get('motor_learning_score', 0.0)},
        )
        if cevts:
            consci_log.append((f, cevts[0].event_type))
            print(f"  frame {f:3d}: {cevts[0].event_type}")

        # Collect traces
        ot_trace.append(father.ot_level)
        cort_trace.append(snap.get('cortisol', 0.08))
        da_trace.append(snap.get('dopamine', 0.10))
        dmn_trace.append(dev.dmn_activity)
        presence_log.append(father.state)
        flag_counts.append(len(flags))

        if f % 60 == 0:
            print(f"  frame {f:3d}: {father.state}  ot={father.ot_level:.2f}"
                  f"  cort={snap.get('cortisol',0):.3f}"
                  f"  absence={father.absence_s:.0f}s")

    print(f"\n[parenting demo] Done.")

    # ---- Plot ------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#111111')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
    t   = np.arange(n_frames) * dt_ms / 1000.0

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
        return ax

    # Oxytocin + presence state
    ax0 = _ax(0, 0, "Oxytocin level  (Father presence)", ylabel='OT')
    ax0.plot(t, ot_trace, color='#e74c3c', lw=1.5, label='OT')
    ax0.axhline(OT_FATHER_LIVE_TARGET,     color='#27ae60', lw=0.7, ls='--',
                label=f'live target={OT_FATHER_LIVE_TARGET}')
    ax0.axhline(OT_FATHER_RECORDED_TARGET, color='#e67e22', lw=0.7, ls='--',
                label=f'recorded target={OT_FATHER_RECORDED_TARGET}')
    ax0.axhline(OT_FATHER_ABSENT_BASELINE, color='#3498db', lw=0.7, ls=':',
                label=f'absent floor={OT_FATHER_ABSENT_BASELINE}')
    ax0.set_ylim(-0.05, 0.75)
    ax0.legend(fontsize=6, facecolor='#222222', labelcolor='white', framealpha=0.7)
    # Shade by presence state
    live_mask = np.array([1.0 if p == PresenceState.FATHER_LIVE else 0.0
                          for p in presence_log])
    rec_mask  = np.array([1.0 if p == PresenceState.FATHER_RECORDED else 0.0
                          for p in presence_log])
    ax0.fill_between(t, 0, live_mask * 0.05, color='#27ae60', alpha=0.3, lw=0)
    ax0.fill_between(t, 0, rec_mask  * 0.05, color='#e67e22', alpha=0.3, lw=0)

    # Cortisol
    ax1 = _ax(0, 1, "Cortisol  (stress baseline)", ylabel='CORT')
    ax1.plot(t, cort_trace, color='#3498db', lw=1.5)
    ax1.axhline(CORT_CHRONIC_HIGH_THRESH, color='#e74c3c', lw=0.7, ls='--',
                label=f'concern>{CORT_CHRONIC_HIGH_THRESH}')
    ax1.set_ylim(-0.01, 0.5)
    ax1.legend(fontsize=6, facecolor='#222222', labelcolor='white', framealpha=0.7)

    # Dopamine
    ax2 = _ax(1, 0, "Dopamine  (reward signal)", ylabel='DA')
    ax2.plot(t, da_trace, color='#2ecc71', lw=1.5)
    ax2.axhline(0.10, color='#555555', lw=0.5, ls='--', label='baseline')
    ax2.legend(fontsize=6, facecolor='#222222', labelcolor='white', framealpha=0.7)

    # DMN activity
    ax3 = _ax(1, 1, "DMN Activity  (ACC + hippocampal proxy)", ylabel='activity')
    ax3.plot(t, dmn_trace, color='#9b59b6', lw=1.5)
    ax3.axhline(DMN_ACTIVITY_THRESH, color='#e74c3c', lw=0.7, ls='--',
                label=f'active>{DMN_ACTIVITY_THRESH}')
    ax3.set_ylim(-0.01, 0.5)
    ax3.legend(fontsize=6, facecolor='#222222', labelcolor='white', framealpha=0.7)

    # Flag counts
    ax4 = _ax(2, 0, "Developmental concern flags  (count per tick)", ylabel='flags')
    ax4.plot(t, flag_counts, color='#e74c3c', lw=1.2)
    ax4.set_ylim(-0.1, max(max(flag_counts) + 0.5, 1.5))

    # Voiceprint similarity
    similarity_trace = []
    for f_i in range(n_frames):
        a1_r = np.full(20, 0.1)
        if 100 <= f_i < 150:
            a1_r += 0.3
        _, sim = vp.detect(a1_r, dt_ms=dt_ms)
        similarity_trace.append(sim)
    ax5 = _ax(2, 1, "Father voiceprint similarity  (A1 cosine)", ylabel='similarity')
    ax5.plot(t, similarity_trace, color='#e67e22', lw=1.5)
    ax5.axhline(VOICEPRINT_SIMILARITY_THRESH, color='#27ae60', lw=0.7, ls='--',
                label=f'threshold={VOICEPRINT_SIMILARITY_THRESH}')
    ax5.set_ylim(-0.05, 1.05)
    ax5.legend(fontsize=6, facecolor='#222222', labelcolor='white', framealpha=0.7)
    # Mark consciousness events
    for (fi, et) in consci_log:
        col = '#ffffff' if et == 'CONSCIOUSNESS_THRESHOLD_EVENT' else '#aaaaaa'
        for ax_ in [ax0, ax1, ax2, ax3, ax4, ax5]:
            ax_.axvline(fi * dt_ms / 1000.0, color=col, lw=0.8, ls=':', alpha=0.6)

    fig.suptitle("CAINE Module 7 — Parenting System (Mother + Father)",
                 fontsize=12, color='#ffffff')
    out = os.path.join(_OUTPUT_DIR, 'caine_module7_parenting.png')
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"[parenting demo] Plot saved -> {out}")


if __name__ == '__main__':
    run_parenting_demo()
