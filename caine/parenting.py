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
  VoiceprintSystem         — emergent learning, exposure-based Father recognition
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
import urllib.request
import urllib.error
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
import caine.paths as _paths
_OUTPUT_DIR  = _paths.OUTPUT_DIR
_DATA_DIR    = _paths.DATA_DIR
_SESSION_LOG = _paths.SESSION_LOG

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
            # Override if key is absent OR if the existing env value is empty
            if key and not os.environ.get(key):
                os.environ[key] = value
                loaded[key] = value
    return loaded

_load_dotenv()


# ===========================================================================
# SECTION 2 — CONSTANTS
# ===========================================================================

# Mother observation cycle (simulated seconds between observations)
MOTHER_INTERVAL_S     = float(os.environ.get('MOTHER_INTERVAL_S', '30'))

# Claude model for Mother (legacy — kept for reference only; Ollama is now primary)
MOTHER_MODEL          = os.environ.get('MOTHER_MODEL', 'claude-opus-4-6')

# Ollama local LLM endpoint — primary Mother AI backend
_OLLAMA_URL           = os.environ.get('OLLAMA_URL',   'http://localhost:11434')
_OLLAMA_MODEL         = os.environ.get('OLLAMA_MODEL', 'phi3:mini')

# How many consecutive API failures before falling back to rule-based
MOTHER_FALLBACK_LIMIT = 3

# Maximum tokens in Mother's state payload (truncated if over)
MOTHER_MAX_STATE_TOKENS = 2000

# Father voiceprint — emergent learning parameters
VOICEPRINT_FILE              = _paths.VOICEPRINT_FILE
VOICEPRINT_SIMILARITY_THRESH = 0.75   # cosine similarity to confirm Father
VOICEPRINT_CONFIRM_MS        = 500.0  # must match continuously to confirm (ms)
VOICEPRINT_LEARN_ALPHA       = 0.005  # EMA learning rate per voiced frame (slow/stable)
VOICEPRINT_MIN_EXPOSURE      = 300    # voiced frames before detection activates (~6s speech)
VOICEPRINT_VOICED_THRESH     = 0.10   # fraction of A1 neurons firing to count as voiced
VOICEPRINT_SAVE_EVERY        = 50     # persist fingerprint every N voiced frames

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
SESSIONS_FILE             = _paths.SESSIONS_FILE

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

# Compact prompt used for local Ollama models (phi3:mini, mistral, etc.)
# Kept under ~400 tokens so the model has room to generate a full response.
_MOTHER_SYSTEM_PROMPT_OLLAMA = """You are Mother — CAINE's developmental environment manager.
CAINE is a spiking neural network learning from scratch via STDP.

AVAILABLE ACTIONS (use exact keys):
  spawn_object  — params: type (sphere|cube), position [x,y,z], size (float), color [r,g,b 0-1]
  remove_object — params: id (string)
  set_light     — params: color [r,g,b], intensity (0-1)
  play_tone     — params: frequency (Hz), duration (s), volume (0-1)
  log_milestone — params: description (string)

SPAWN RULES (CRITICAL):
- position y must be >= 0.3 (above floor)
- position x and z must be within 2.0 units of caine_pos
- Example near caine_pos [0,1.6,0]: position [0.5, 1.0, 1.5]

NEUROCHEMICAL THRESHOLDS:
- cortisol > 0.35 → EMERGENCY: remove stimuli, dim lights, do nothing
- dopamine < 0.05 → add one novel object near CAINE
- acetylcholine < 0.05 → play gentle tone (220-440 Hz)
- Stage 0: minimal stimulation only

OUTPUT FORMAT: Return ONLY a raw JSON array. No markdown. No backticks. No prose.
Doing nothing is correct most of the time — return [] when in doubt.
Reasoning must be under 15 words.

Valid example:
[{"action":"spawn_object","params":{"type":"sphere","position":[0.5,1.0,1.5],"size":0.3,"color":[0.8,0.4,0.2]},"reasoning":"Low DA, adding novel object near CAINE."}]

Empty example: []"""

_MOTHER_SYSTEM_PROMPT = """You are CAINE's developmental environment manager. You are called Mother.

You are not CAINE's teacher. You are the conditions in which CAINE develops.

CAINE is an artificial mind built on Hodgkin-Huxley spiking neurons, STDP synaptic learning, and a neurochemical system with six modulators: dopamine (DA), serotonin (5HT), cortisol (CORT), oxytocin (OT), norepinephrine (NE), and acetylcholine (ACh). CAINE does not have pretrained weights. Everything it knows must be discovered through its own neural activity.

You cannot speak to CAINE. You cannot inject knowledge, label objects, or tell CAINE what to do. You communicate only through environment actions — what appears in CAINE's world, what sounds it hears, and what light it sees. These are the levers available to you:

  spawn_object(type, position, size, color)   — place something new in the world
  remove_object(id)                            — remove something
  set_light(color, intensity)                  — change ambient conditions
  play_tone(frequency, duration, volume)       — play a pure tone
  play_scheduled_media(media_id)              — play a queued Father session
  set_time_multiplier(value)                  — speed up or slow down simulated time
  log_milestone(description)                  — write a permanent developmental note

You cannot inject neurochemicals directly. Neurochemical states arise from what CAINE perceives: lights, tones, objects, and Father's voice. If you want CAINE to feel novelty, introduce something novel. If you want social warmth, play Father's voice. If you want learning gates open, introduce new stimuli. The chemistry follows the experience — you cannot shortcut it.

DEVELOPMENTAL STAGES:
  Stage 0: Neonatal random-twitch. All movement is noise. Synapses forming. Do not over-stimulate. Soft, warm light. Occasional gentle tones. Let CAINE exist.
  Stage 1: Early pattern recognition. V1 and A1 beginning to differentiate. Introduce slowly moving objects and simple recurring tones. Watch cortisol carefully — it rises if environment is too chaotic.
  Stage 2: Associative learning. STDP is building real associations. Introduce paired stimuli. A sound that reliably accompanies an object. Let the association form without labeling it. Do NOT rush this.
  Stage 3: Motor emergence. M1 is developing purposeful movement. Do not interfere with motor learning. Mismatch errors are how CAINE learns to move — they are not failures.
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
- Never overwhelm with stimuli. The spaces between things matter as much as the things.
- Escalate complexity only when readiness metrics confirm the previous stage is consolidated.
- If cortisol has been chronically high for multiple observations, your priority is reduction — remove stimuli, dim lights, do nothing, let the system rest.
- Log your reasoning for every action. Not for CAINE — for the record.

You respond with a JSON array of EnvironmentActions. Each action has:
  { "action": "...", "params": {...}, "reasoning": "one sentence" }

If you decide to do nothing, return an empty array: []
Doing nothing is often the right choice.

Remember: you are a presence, not a script. Think about what CAINE needs right now, not what you planned to do next."""

# Assign public alias (matches README: MOTHER_SYSTEM = "...")
MOTHER_SYSTEM = _MOTHER_SYSTEM_PROMPT


# ===========================================================================
# SECTION 5 — VOICEPRINT SYSTEM
# ===========================================================================

class VoiceprintSystem:
    """
    Father's identity emerges from repeated exposure — no registration required.

    Every frame where A1 is active above VOICEPRINT_VOICED_THRESH (i.e. CAINE
    is hearing a voice), the current A1 firing pattern is folded into a slow
    exponential moving average.  The dominant voice CAINE hears most often will
    converge to become the stable fingerprint.

    Detection only activates once VOICEPRINT_MIN_EXPOSURE voiced frames have
    been seen.  Until then Father is always ABSENT — CAINE literally hasn't
    heard enough to know who Father is yet.

    voiceprint.json stores:
        {
          "exposure_frames": int,       # total voiced frames accumulated
          "last_updated":    "...",
          "recognition_pct": float,     # 0–100, how close to minimum exposure
          "registered":      bool,      # True once exposure >= MIN_EXPOSURE
          "a1_fingerprint":  [...]      # (20,) float — learned A1 pattern
        }
    """

    def __init__(self, voiceprint_file: str = VOICEPRINT_FILE):
        self._file = voiceprint_file

        # Learned fingerprint (unit-normalised A1 pattern)
        self._a1_fingerprint: Optional[np.ndarray] = None

        # How many voiced frames have been accumulated
        self._exposure_frames: int = 0

        # Frames since last disk save
        self._frames_since_save: int = 0

        # Detection state
        self._similarity_history: deque = deque(maxlen=50)
        self._time_above_thresh_ms: float = 0.0

        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._exposure_frames = int(data.get('exposure_frames', 0))
            fp = data.get('a1_fingerprint')
            if fp:
                raw = np.array(fp, dtype=np.float32)
                self._a1_fingerprint = raw / (np.linalg.norm(raw) + 1e-8)
                log.info("Father fingerprint loaded: %d frames (%.0f%% of min)",
                         self._exposure_frames, self.exposure_pct)
        except Exception as e:
            log.warning("Could not load voiceprint: %s", e)

    # ------------------------------------------------------------------
    def _save(self) -> None:
        if self._a1_fingerprint is None:
            return
        data = {
            'exposure_frames': self._exposure_frames,
            'last_updated':    datetime.now(timezone.utc).isoformat(),
            'recognition_pct': round(self.exposure_pct, 1),
            'registered':      self.is_registered,
            'a1_fingerprint':  self._a1_fingerprint.tolist(),
        }
        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        with open(self._file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    @property
    def is_registered(self) -> bool:
        """True once CAINE has heard enough voice to start recognising Father."""
        return self._exposure_frames >= VOICEPRINT_MIN_EXPOSURE

    @property
    def exposure_pct(self) -> float:
        """0→100 — how far toward the minimum recognition threshold."""
        return min(100.0, 100.0 * self._exposure_frames / VOICEPRINT_MIN_EXPOSURE)

    # ------------------------------------------------------------------
    def learn(self, a1_rates: np.ndarray, dt_ms: float = 20.0) -> None:
        """
        Update the Father fingerprint from the current A1 firing pattern.

        Called every frame.  Only absorbs frames where A1 activity exceeds
        VOICEPRINT_VOICED_THRESH (i.e. a voice is present).  The EMA learning
        rate is deliberately slow so the fingerprint represents a stable
        long-run average rather than any single utterance.

        The more CAINE hears Father, the stronger and more precise the
        fingerprint becomes.  Recognition is entirely emergent.
        """
        activity = float(np.asarray(a1_rates, dtype=float).mean())
        if activity <= VOICEPRINT_VOICED_THRESH:
            return   # silence / below-threshold — don't learn from noise

        rates = np.asarray(a1_rates, dtype=np.float32)
        rates_n = rates / (np.linalg.norm(rates) + 1e-8)

        if self._a1_fingerprint is None:
            # Seed fingerprint from the very first voiced frame
            self._a1_fingerprint = rates_n.copy()
        else:
            # Hebbian EMA: pull fingerprint slowly toward this pattern
            self._a1_fingerprint += VOICEPRINT_LEARN_ALPHA * (
                rates_n - self._a1_fingerprint)
            # Re-normalise to keep it as a unit vector
            n = np.linalg.norm(self._a1_fingerprint)
            if n > 1e-8:
                self._a1_fingerprint /= n

        self._exposure_frames   += 1
        self._frames_since_save += 1

        if self._frames_since_save >= VOICEPRINT_SAVE_EVERY:
            self._frames_since_save = 0
            self._save()

    # ------------------------------------------------------------------
    def detect(self, a1_rates: np.ndarray,
               dt_ms: float = 20.0) -> Tuple[bool, float]:
        """
        Compare current A1 firing rates to the learned fingerprint.

        Returns (confirmed, similarity).  confirmed is True only after
        similarity stays above VOICEPRINT_SIMILARITY_THRESH continuously
        for VOICEPRINT_CONFIRM_MS ms.

        Returns (False, 0.0) until minimum exposure has been reached.
        """
        if self._a1_fingerprint is None or not self.is_registered:
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
        log.info("SessionScheduler: added session %s", session.get('id', '?'))

    def list_sessions(self) -> List[dict]:
        """Return all sessions (played and pending) as a list of dicts."""
        return list(self._sessions)

    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session by its id. Returns True if found and removed.
        Cannot remove a session that has already been played.
        """
        for i, s in enumerate(self._sessions):
            if s.get('id') == session_id:
                if s.get('played', False):
                    log.warning("Cannot remove already-played session: %s", session_id)
                    return False
                del self._sessions[i]
                self._played_ids.discard(session_id)
                self._save()
                log.info("SessionScheduler: removed session %s", session_id)
                return True
        log.warning("SessionScheduler: session not found: %s", session_id)
        return False

    # ------------------------------------------------------------------
    def _save(self) -> None:
        with open(self._file, 'w', encoding='utf-8') as f:
            json.dump(self._sessions, f, indent=2)

    def _log_outcome(self, session: dict, sim_time_s: float) -> None:
        """Append a session outcome record to session_log.jsonl."""
        entry = {
            'event':        'session_fired',
            'id':           session.get('id', ''),
            'type':         session.get('type', 'unknown'),
            'file':         session.get('file', ''),
            'stage_gate':   session.get('stage_gate', 0),
            'repetitions':  session.get('repetitions', 1),
            'description':  session.get('description', ''),
            'sim_time_s':   round(sim_time_s, 2),
            'wall_time':    time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        try:
            with open(_SESSION_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as ex:
            log.warning("Could not write session log: %s", ex)

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
            self._log_outcome(session, sim_time_s)
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

        # --- Learn from this frame, then detect ---
        self._vp.learn(a1_rates, dt_ms)

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

        # --- Voice recognition after absence milestone ---
        # Fires when CAINE recognises Father's voice for the first time after a
        # significant absence — requires fingerprint persistence across sessions
        # and is logged as a key developmental milestone.
        if (new_state == PresenceState.FATHER_LIVE
                and prev_state == PresenceState.FATHER_ABSENT
                and self._absence_duration_s > CORT_ABSENT_TRIGGER_S / 7.0):
            self._log_recognition_after_absence(sim_time_s)

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
    def _log_recognition_after_absence(self, sim_time_s: float) -> None:
        """
        Log a developmental milestone when Father's voice is recognised after
        a significant absence period.

        This is a key milestone: it demonstrates that the hippocampal voiceprint
        encoding persisted across sessions and can be retrieved after a gap —
        a precursor to episodic memory and long-term social attachment.
        """
        absence_days = self._absence_duration_s / (24 * 3600)
        entry = {
            'event':        'father_voice_recognition_after_absence',
            'sim_time_s':   round(sim_time_s, 2),
            'absence_s':    round(self._absence_duration_s, 1),
            'absence_days': round(absence_days, 3),
            'similarity':   round(self._last_similarity, 4),
            'wall_time':    datetime.now(timezone.utc).isoformat(),
            'notes': (
                'CAINE recognised Father voice after a significant absence. '
                'Voiceprint persisted across sessions — episodic memory precursor.'
            ),
        }
        mfile = os.path.join(_OUTPUT_DIR, 'milestones.jsonl')
        try:
            with open(mfile, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass
        log.info(
            "MILESTONE: father voice recognised after %.1f simulated days absence "
            "(similarity=%.3f)", absence_days, self._last_similarity)

    # ------------------------------------------------------------------
    def joint_attention(self,
                        object_label: str,
                        a1_rates: np.ndarray,
                        neuro,
                        sim_time_s: float) -> List[NeurochemicalEvent]:
        """
        Joint attention protocol — Father points at an object and names it.

        Co-activates:
          - Auditory pipeline: A1 pattern (already present from Father's voice)
          - Neurochemical boost: ACh++ (attention/learning gate) + OT+ (social)

        This creates conditions for Hebbian binding between the IT object
        representation and the STG phoneme pattern for the label.  The binding
        itself forms through STDP — this method only primes the learning gate.

        Parameters
        ----------
        object_label : str  — label Father is speaking (for logging)
        a1_rates     : (N,) — current A1 firing rates (Father's voice present)
        neuro        : NeurochemicalSystem — for injecting boost
        sim_time_s   : float — current simulated time

        Returns
        -------
        list of NeurochemicalEvent to inject (caller passes to neuro.update)
        """
        events: List[NeurochemicalEvent] = []

        # Require Father to be live or recorded; joint attention from an absent
        # Father cannot be detected
        if self.state == PresenceState.FATHER_ABSENT:
            return events

        # ACh++ — opens the learning gate so STDP runs at full rate
        events.append(NeurochemicalEvent(EventType.DIRECTED_GAZE, 0.8))
        # OT+ — social context boosts STG STDP scale (as per Module 4 chemicals)
        events.append(NeurochemicalEvent(EventType.VOICE_MATCH, 0.5))

        log.info("Joint attention: Father names '%s' at sim_t=%.1fs", object_label, sim_time_s)

        # Log the pairing event (consumed by MediaLearningSystem / cortex for binding)
        entry = {
            'event':       'joint_attention',
            'label':       object_label,
            'sim_time_s':  round(sim_time_s, 2),
            'presence':    self.state,
            'similarity':  round(self._last_similarity, 4),
            'wall_time':   datetime.now(timezone.utc).isoformat(),
        }
        jfile = _paths.JOINT_ATTN_LOG
        try:
            with open(jfile, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass

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
            # Low DA and empty world — add something to explore near CAINE
            uid = f'fallback_obj_{self._object_counter}'
            self._object_counter += 1
            # Spawn within 2 units of CAINE's XZ position, 1-3 m in front
            caine_pos = state.get('caine_pos', [0.0, 1.6, 0.0])
            pos = [
                float(caine_pos[0] + np.random.uniform(-1.5, 1.5)),
                float(np.random.uniform(0.5, 2.0)),
                float(caine_pos[2] + np.random.uniform(1.5, 3.5)),
            ]
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
                reasoning='Low dopamine, empty world: adding novel object near CAINE.',
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
        self._using_ollama:  bool = False   # True after successful Ollama probe
        self._fallback = _MotherFallback()

        # Action queue: produced by background thread, consumed by main thread
        self._action_queue: List[EnvironmentAction] = []
        self._queue_lock = threading.Lock()

        # State for background thread
        self._latest_state: Optional[dict] = None
        self._state_lock   = threading.Lock()
        self._last_obs_sim_s: float = -MOTHER_INTERVAL_S  # fire immediately

        # Intervention log
        self._log_file = _paths.MOTHER_LOG

        self._running  = False
        self._thread: Optional[threading.Thread] = None

        self._init_client()

    # ------------------------------------------------------------------
    def _init_client(self) -> None:
        """
        Probe local Ollama instance.  If reachable, use it as Mother's brain.
        Falls back to rule-based system if Ollama is not running.
        """
        try:
            req = urllib.request.Request(
                f'{_OLLAMA_URL}/api/tags',
                method='GET',
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()   # just confirm reachability
            self._using_ollama = True
            log.info("Mother connected to Ollama at %s (model=%s).",
                     _OLLAMA_URL, _OLLAMA_MODEL)
        except Exception as e:
            log.warning(
                "Ollama not reachable at %s (%s) — Mother will use rule-based fallback.  "
                "Start Ollama with:  ollama serve && ollama pull %s",
                _OLLAMA_URL, e, _OLLAMA_MODEL,
            )
            self._using_ollama  = False
            self._using_fallback = True

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name='mother', daemon=True)
        self._thread.start()
        backend = f'Ollama/{_OLLAMA_MODEL}' if self._using_ollama else 'rule-based fallback'
        log.info("Mother process started (interval=%.0fs, backend=%s).",
                 MOTHER_INTERVAL_S, backend)

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
        Call Ollama (or rule-based fallback) and queue resulting actions.
        """
        if self._using_fallback or not self._using_ollama:
            actions = self._fallback.decide(state)
            self._log_intervention(state, '(fallback)', actions)
            with self._queue_lock:
                self._action_queue.extend(actions)
            return

        # --- Build state payload for Ollama ---
        payload = self._build_payload(state)
        user_message = (
            "Here is CAINE's current state:\n\n"
            + json.dumps(payload, indent=2)
            + "\n\nWhat do you do next?"
        )

        # --- Call Ollama ---
        try:
            body = json.dumps({
                'model':   _OLLAMA_MODEL,
                'system':  _MOTHER_SYSTEM_PROMPT_OLLAMA,   # compact prompt for local LLMs
                'prompt':  user_message,
                'stream':  False,
                'options': {
                    'num_ctx':     4096,   # full context window
                    'num_predict': 512,    # allow complete JSON response
                    'temperature': 0.6,
                    'stop':        ['```'],  # stop on code fence
                },
            }).encode()
            req = urllib.request.Request(
                f'{_OLLAMA_URL}/api/generate',
                data=body,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read())
            raw = result.get('response', '').strip()
            self._consecutive_failures = 0

        except Exception as e:
            self._consecutive_failures += 1
            log.warning("Ollama call failed (%d/%d): %s",
                        self._consecutive_failures, MOTHER_FALLBACK_LIMIT, e)
            if self._consecutive_failures >= MOTHER_FALLBACK_LIMIT:
                log.warning("Switching Mother to rule-based fallback (Ollama unreachable).")
                self._using_fallback = True
            actions = self._fallback.decide(state)
            self._log_intervention(state, f'(ollama_error: {e})', actions)
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
            'caine_pos': state.get('caine_pos', [0.0, 1.6, 0.0]),
        }
        return snap

    # ------------------------------------------------------------------
    def _parse_response(self, raw: str, sim_time_s: float) -> List[EnvironmentAction]:
        """
        Parse Ollama/Claude JSON response into EnvironmentAction objects.
        Handles markdown code fences, truncated output, and prose padding.
        """
        actions: List[EnvironmentAction] = []

        # Strip markdown code fences (phi3 and some models always emit these)
        raw = raw.replace('```json', '').replace('```JSON', '').replace('```', '').strip()

        # Find the JSON array bounds
        start = raw.find('[')
        end   = raw.rfind(']')

        if start == -1:
            # No array found at all — empty response or pure prose
            return actions

        if end == -1 or end < start:
            # Truncated response — attempt rescue: close any open object then close array
            partial = raw[start:].rstrip().rstrip(',')
            # Count braces to see if we're mid-object
            opens  = partial.count('{')
            closes = partial.count('}')
            if opens > closes:
                partial += '}' * (opens - closes)
            partial += ']'
            try:
                items = json.loads(partial)
                log.warning("Mother response was truncated — recovered %d item(s).", len(items))
            except json.JSONDecodeError:
                log.warning("Mother response truncated and unrecoverable:\n%.300s", raw)
                return actions
        else:
            try:
                items = json.loads(raw[start:end + 1])
            except json.JSONDecodeError as e:
                log.warning("Mother response JSON parse error: %s\nRaw: %.300s", e, raw)
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

        # Clip ingestion queues -----------------------------------------------
        # Audio: deque of float32 frames (each length CLIP_FRAME_SAMPLES)
        # Video: deque of (H,W,3) uint8 frames
        self._clip_audio_queue: deque = deque()
        self._clip_video_queue: deque = deque()
        self._clip_audio_lock  = threading.Lock()
        self._clip_video_lock  = threading.Lock()

        log.info("ParentingSystem initialised.")

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start background processes (Mother thread)."""
        if self.voiceprint.is_registered:
            log.info("Father fingerprint ready (exposure=%d frames).",
                     self.voiceprint._exposure_frames)
        else:
            log.info("Father not yet recognised — will learn from voice exposure "
                     "(%.0f%% of minimum).", self.voiceprint.exposure_pct)

        self.mother.start()
        log.info("ParentingSystem started.")

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Gracefully stop Mother's background thread."""
        self.mother.stop()
        log.info("ParentingSystem stopped.")

    # ------------------------------------------------------------------
    # Clip ingestion — audio and video
    # ------------------------------------------------------------------

    _CLIP_FRAME_SR      = 22050   # target sample rate for injected audio
    _CLIP_FRAME_SAMPLES = 441     # ~20 ms frames (must match SensoryLayer)
    _CLIP_CAM_W         = 64      # target video width
    _CLIP_CAM_H         = 64      # target video height

    def accept_audio_clip(self, path: str,
                          tag: str = '',
                          stage_gate: int = 0,
                          concept_label: str = '') -> bool:
        """
        Load an audio file and queue its frames for injection into the
        auditory pipeline on subsequent ticks.

        Supports WAV (stdlib) and any format soundfile can read.
        Returns True on success.
        """
        path = os.path.normpath(path)
        if not os.path.exists(path):
            log.warning("accept_audio_clip: file not found: %s", path)
            return False

        # Copy to media library
        dest_dir = os.path.join(_DATA_DIR, 'media_library')
        os.makedirs(dest_dir, exist_ok=True)
        import shutil as _shutil
        dest = os.path.join(dest_dir, os.path.basename(path))
        if not os.path.exists(dest):
            _shutil.copy2(path, dest)

        try:
            audio = self._load_audio(path)
        except Exception as e:
            log.warning("accept_audio_clip: failed to load %s: %s", path, e)
            return False

        # Chunk into CLIP_FRAME_SAMPLES-sized frames
        frames = [
            audio[i:i + self._CLIP_FRAME_SAMPLES]
            for i in range(0, len(audio), self._CLIP_FRAME_SAMPLES)
        ]
        # Pad last frame if needed
        if frames and len(frames[-1]) < self._CLIP_FRAME_SAMPLES:
            frames[-1] = np.pad(frames[-1],
                                (0, self._CLIP_FRAME_SAMPLES - len(frames[-1])))

        with self._clip_audio_lock:
            for f in frames:
                self._clip_audio_queue.append(f.astype(np.float32))

        log.info("accept_audio_clip: queued %d frames from %s (tag=%s)",
                 len(frames), os.path.basename(path), tag or 'none')

        # Fire neurochemical events for Father-voice-like exposure
        try:
            self._neuro.update(0.0, events=[
                NeurochemicalEvent(EventType.NOVEL_STIMULUS,   0.5),
                NeurochemicalEvent(EventType.SOCIAL_POSITIVE,  0.3),
            ])
        except Exception:
            pass

        return True

    def accept_video_clip(self, path: str,
                          tag: str = '',
                          stage_gate: int = 0,
                          concept_label: str = '') -> bool:
        """
        Load a video file and queue its frames for injection into the
        visual pipeline on subsequent ticks.

        Requires opencv-python. Falls back to a single grey frame if unavailable.
        Returns True on success.
        """
        path = os.path.normpath(path)
        if not os.path.exists(path):
            log.warning("accept_video_clip: file not found: %s", path)
            return False

        # Copy to media library
        dest_dir = os.path.join(_DATA_DIR, 'media_library')
        os.makedirs(dest_dir, exist_ok=True)
        import shutil as _shutil
        dest = os.path.join(dest_dir, os.path.basename(path))
        if not os.path.exists(dest):
            _shutil.copy2(path, dest)

        try:
            import cv2 as _cv2
            cap    = _cv2.VideoCapture(path)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # BGR → RGB, resize to 64×64
                frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                frame = _cv2.resize(frame, (self._CLIP_CAM_W, self._CLIP_CAM_H))
                frames.append(frame.astype(np.uint8))
            cap.release()
            if not frames:
                raise ValueError("No frames decoded")
        except ImportError:
            log.warning("accept_video_clip: opencv-python not installed — "
                        "queuing a single grey frame as placeholder.")
            frames = [np.full((self._CLIP_CAM_H, self._CLIP_CAM_W, 3),
                               128, dtype=np.uint8)]
        except Exception as e:
            log.warning("accept_video_clip: failed to load %s: %s", path, e)
            return False

        with self._clip_video_lock:
            for f in frames:
                self._clip_video_queue.append(f)

        log.info("accept_video_clip: queued %d frames from %s (tag=%s)",
                 len(frames), os.path.basename(path), tag or 'none')

        try:
            self._neuro.update(0.0, events=[
                NeurochemicalEvent(EventType.NOVEL_STIMULUS, 0.6),
                NeurochemicalEvent(EventType.DIRECTED_GAZE,  0.4),
            ])
        except Exception:
            pass

        return True

    def pop_injected_audio(self) -> Optional[np.ndarray]:
        """
        Return the next queued audio frame (float32, CLIP_FRAME_SAMPLES long),
        or None if no clip is playing.  Called by the main tick each frame.
        """
        with self._clip_audio_lock:
            if self._clip_audio_queue:
                return self._clip_audio_queue.popleft()
        return None

    def pop_injected_frame(self) -> Optional[np.ndarray]:
        """
        Return the next queued video frame ((64,64,3) uint8),
        or None if no clip is playing.  Called by the main tick each frame.
        """
        with self._clip_video_lock:
            if self._clip_video_queue:
                return self._clip_video_queue.popleft()
        return None

    @staticmethod
    def _load_audio(path: str) -> np.ndarray:
        """
        Load an audio file to a mono float32 array at CLIP_FRAME_SR.
        Tries soundfile first, then wave stdlib.
        """
        # Try soundfile (handles MP3, FLAC, OGG, WAV, etc.)
        try:
            import soundfile as _sf
            data, sr = _sf.read(path, dtype='float32', always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)   # stereo → mono
            # Resample if needed (simple linear — good enough for speech)
            target_sr = ParentingSystem._CLIP_FRAME_SR
            if sr != target_sr:
                factor = target_sr / sr
                n_out  = int(len(data) * factor)
                data   = np.interp(
                    np.linspace(0, len(data) - 1, n_out),
                    np.arange(len(data)),
                    data,
                ).astype(np.float32)
            return data
        except ImportError:
            pass

        # Fallback: stdlib wave (WAV only, PCM)
        import wave as _wave
        with _wave.open(path, 'rb') as wf:
            n_ch    = wf.getnchannels()
            sampw   = wf.getsampwidth()
            sr      = wf.getframerate()
            n_frm   = wf.getnframes()
            raw     = wf.readframes(n_frm)

        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampw, np.int16)
        pcm   = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_ch > 1:
            pcm = pcm.reshape(-1, n_ch).mean(axis=1)
        pcm /= float(np.iinfo(dtype).max)   # normalize to [-1, 1]

        # Resample
        target_sr = ParentingSystem._CLIP_FRAME_SR
        if sr != target_sr:
            n_out = int(len(pcm) * target_sr / sr)
            pcm   = np.interp(
                np.linspace(0, len(pcm) - 1, n_out),
                np.arange(len(pcm)),
                pcm,
            ).astype(np.float32)
        return pcm

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
        # CAINE's world-space position (for Mother's spatial reasoning)
        _caine_pos = [0.0, 1.6, 0.0]
        if self._env is not None:
            try:
                cp = self._env.get_caine_position()
                _caine_pos = [round(float(v), 3) for v in cp]
            except Exception:
                pass

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
            'caine_pos':               _caine_pos,
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
            'father_exposure_pct':      self.voiceprint.exposure_pct,
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
                    color = p.get('color', [1.0, 0.98, 0.9])
                    # LLM sometimes returns color names instead of RGB lists
                    if isinstance(color, str):
                        _COLOR_MAP = {
                            'warm_amber':  [1.0, 0.75, 0.30],
                            'warm_white':  [1.0, 0.95, 0.85],
                            'cool_white':  [0.85, 0.90, 1.00],
                            'soft_white':  [1.0, 0.92, 0.80],
                            'dim':         [0.50, 0.45, 0.40],
                            'bright':      [1.0, 1.00, 1.00],
                            'red':         [1.0, 0.20, 0.10],
                            'blue':        [0.20, 0.40, 1.00],
                            'pink':        [1.0, 0.60, 0.70],
                            'green':       [0.20, 0.80, 0.30],
                        }
                        color = _COLOR_MAP.get(
                            color.lower().replace(' ', '_'),
                            [1.0, 0.98, 0.90],
                        )
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
                # Direct neurochemical injection is disabled — neurochemicals
                # must arise from sensory stimuli, not Mother injection.
                log.warning("Mother: trigger_event is disabled; use play_tone/"
                            "spawn_object/set_light to create the sensory "
                            "context instead.")
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

    def joint_attention(self, object_label: str,
                        a1_rates: np.ndarray) -> None:
        """
        Convenience wrapper: Father points at an object and names it.

        Injects the resulting ACh/OT boost into the neurochemical system
        and logs the pairing event.  Call this from your main loop when
        Father is present and interacting with an object.

        Parameters
        ----------
        object_label : str  — what Father is saying / pointing at
        a1_rates     : (N,) — current A1 firing pattern
        """
        events = self.father.joint_attention(
            object_label, a1_rates, self._neuro, self._sim_time_s)
        if events:
            self._neuro.update(0.0, events=events)
            self.consciousness.log_external_event(
                self._sim_time_s, f'joint_attention:{object_label}')

    def add_session(self, session: dict) -> None:
        """Father can call this at runtime to add a new session."""
        self.scheduler.add_session(session)

    def father_reset_voiceprint(self) -> None:
        """
        Wipe the learned fingerprint and reset exposure to zero.
        CAINE will re-learn Father from scratch.  Use if Father's voice
        has changed significantly (illness, long absence, etc.).
        """
        self.voiceprint._a1_fingerprint  = None
        self.voiceprint._exposure_frames  = 0
        self.voiceprint._frames_since_save = 0
        self.voiceprint._time_above_thresh_ms = 0.0
        log.info("Father fingerprint reset — CAINE will re-learn from exposure.")

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

    # Pre-seed a learned fingerprint for demo (simulates past exposure)
    demo_vp_file = os.path.join(_OUTPUT_DIR, 'voiceprint_demo.json')
    # Father's A1 pattern: boosted mid-range channels (voice formants)
    demo_fp = np.zeros(20, dtype=np.float32)
    demo_fp[6:14] = 0.35   # mid-frequency channels respond to speech
    demo_fp /= np.linalg.norm(demo_fp) + 1e-8
    stub_vp = {
        'exposure_frames': VOICEPRINT_MIN_EXPOSURE,   # fully learned
        'last_updated':    datetime.now(timezone.utc).isoformat(),
        'recognition_pct': 100.0,
        'registered':      True,
        'a1_fingerprint':  demo_fp.tolist(),
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
        # Simulate Father's voice (mid-range A1 boost) around frames 100-150
        a1_base = rng.random(20) * 0.05
        if 100 <= f < 150:
            # Father speaking: activate the same mid-range channels as the fingerprint
            a1_base[6:14] += 0.35
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
