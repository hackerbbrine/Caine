"""
CAINE Visualization Layer -- Module 10
=======================================
WebSocket server at ws://localhost:7734.

Architecture
------------
    CAINE runtime  ->  VisualizationServer.tick()
                            |
                      JSON frame dict
                            |
                   asyncio background thread
                            |
                    ws://localhost:7734
                            |
                   Electron / browser client
                            v
                  Three.js / D3.js panels

7 Panels (outgoing JSON keys)
------------------------------
  panel1_brain     -- 3-D neuron spheres, synapse lines, AMPA/NMDA/GABA colours
  panel2_vision    -- Camera feed (base64 PNG), V1 edge overlay, IT heatmap, ACh spotlight
  panel3_auditory  -- Waveform, FFT/Mel bands, A1 tonotopy, STG cluster label + confidence
  panel4_neuro     -- 6-chem live history (60 s), event log
  panel5_dev       -- Stage badge, simulated age, vocab, synapse counts, DMN, WM, self-model
  panel6_internal  -- WM contents, inferred emotion, motor intention (labelled INFERRED)
  panel7_controls  -- Simulation state (is_paused, time_scale); reflects Mission Control

Incoming control messages (Panel 7 / Mission Control)
------------------------------------------------------
  {"action": "pause"}
  {"action": "resume"}
  {"action": "stop"}
  {"action": "set_time_scale",          "value": 2.0}
  {"action": "cortisol_flush"}
  {"action": "forced_rest",             "duration_s": 300.0}
  {"action": "stage_rollback"}
  {"action": "media_upload",            "filename": "...", "stage": 1, "data_b64": "..."}
  {"action": "schedule_father_session", "time_s": 3600, "type": "voice_exposure",
                                        "file": "...", "stage_gate": 0,
                                        "description": "..."}
  {"action": "mother_override",         "approve": true}

    Retrieve pending commands each tick with ``viz.pop_commands()``.

Wiring all systems
------------------
  Attach any combination of live module instances via ``attach()``.
  Every field degrades gracefully to zero / stub data when the module
  is not provided -- the server still runs and sends valid frames.

Dependencies
------------
    pip install websockets

Optional:
    pip install Pillow            (faster PNG encoding; falls back to matplotlib)

Usage
-----
    from caine.visualization import VisualizationServer

    viz = VisualizationServer()
    viz.attach(
        neuro         = neuro_system,    # NeurochemicalSystem  (chemicals.py)
        v1            = v1_pop,          # V1Population         (cortex.py)
        a1            = a1_pop,          # A1Population         (cortex.py)
        it            = it_pop,          # ITPopulation         (cortex.py)
        stg           = stg_pop,         # STGPopulation        (cortex.py)
        pfc           = pfc_pop,         # PFCPopulation        (cortex.py)
        ag            = ag_pop,          # AngularGyrusPopulation (cortex.py)
        dmn           = dmn_monitor,     # DMNMonitor           (cortex.py)
        hippocampus   = hippo,           # Hippocampus          (limbic.py)
        sensory       = sensory_layer,   # SensoryLayer         (sensory.py)
        env           = caine_env,       # CaineEnvironment     (environment.py)
        stage_manager = stage_mgr,       # StageManager         (environment.py)
        avatar        = avatar_ctrl,     # AvatarController     (avatar.py)
        media         = media_sys,       # MediaLearningSystem  (media.py)
    )
    viz.start()

    for tick in range(n_ticks):
        result = sensory_layer.update(...)
        viz.tick(sim_time_s=tick * dt_s,
                 dt_ms=20.0,
                 sensory_result=result)

        for cmd in viz.pop_commands():
            if cmd['action'] == 'pause':
                sim.pause()

    viz.stop()
"""

import asyncio
import base64
import io
import json
import os
import sys as _sys
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import caine.paths as _paths
_OUTPUT_DIR = _paths.OUTPUT_DIR

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import websockets
    _WS_OK = True
except ImportError:
    _WS_OK = False
    websockets = None  # type: ignore

try:
    from PIL import Image as _PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_PORT_DEFAULT    = 7734
_FRAME_INTERVAL  = 0.10    # max WebSocket send rate: 10 Hz
_HISTORY_S       = 60.0    # neurochemical history window (seconds)
_MAX_HISTORY_PTS = 600     # ring-buffer length (one point every 100 ms at 10 Hz)
_MAX_EVENT_LOG   = 50      # neurochemical events kept in frame
_MAX_SYNAPSES    = 200     # synapse lines sent per frame
_MAX_COMMANDS    = 256     # incoming command ring buffer depth

_NEURO_CHEMICALS = [
    'dopamine', 'serotonin', 'cortisol',
    'oxytocin', 'norepinephrine', 'acetylcholine',
]

# HTML colour strings used by the Electron / Three.js renderer
_CHEM_COLORS = {
    'dopamine':       '#4169e1',
    'serotonin':      '#3cb371',
    'cortisol':       '#ff6347',
    'oxytocin':       '#da70d6',
    'norepinephrine': '#ff8c00',
    'acetylcholine':  '#4682b4',
}

# Synapse-type colours (Panel 1)
_SYNTYPE_COLORS = {'AMPA': '#ff4040', 'NMDA': '#4040ff', 'GABA': '#40c040'}

# Named emotional states: (DA, 5HT, CORT, OT, NE, ACh) normalised to [0,1]
_EMOTIONAL_STATES: Dict[str, Tuple[float, ...]] = {
    'content':    (0.50, 0.60, 0.10, 0.45, 0.30, 0.40),
    'curious':    (0.60, 0.40, 0.15, 0.35, 0.65, 0.80),
    'anxious':    (0.25, 0.20, 0.72, 0.15, 0.85, 0.40),
    'joyful':     (0.80, 0.70, 0.08, 0.75, 0.50, 0.55),
    'distressed': (0.15, 0.10, 0.90, 0.10, 0.92, 0.35),
    'calm':       (0.45, 0.80, 0.08, 0.55, 0.20, 0.30),
    'engaged':    (0.70, 0.50, 0.12, 0.50, 0.72, 0.90),
    'fatigued':   (0.30, 0.40, 0.40, 0.35, 0.20, 0.25),
    'bonded':     (0.55, 0.65, 0.05, 0.90, 0.40, 0.50),
    'startled':   (0.35, 0.25, 0.55, 0.20, 0.95, 0.70),
}

# STG cluster -> phoneme-like label (24 clusters)
_STG_LABELS = [
    'sil', 'PP', 'FF', 'TH', 'DD', 'kk', 'CH', 'SS',
    'nn',  'RR', 'aa', 'E',  'ih', 'oh', 'ou', 'uu',
    'mm',  'vv', 'zz', 'll', 'yy', 'ww', 'hh', 'ng',
]

# ---------------------------------------------------------------------------
# Anatomical 3D layout (brain-space units, ~333 neuron positions)
# ---------------------------------------------------------------------------

# (cx, cy, cz, scatter_radius)
_REGION_LAYOUT: Dict[str, Tuple[float, float, float, float]] = {
    'V1':   ( 0.0,  0.0, -6.0, 1.5),  # primary visual  -- occipital pole
    'V2':   ( 0.5,  0.2, -5.5, 1.0),  # secondary visual
    'V4':   ( 1.0, -0.5, -5.0, 1.0),  # colour / form
    'MT':   ( 2.0, -0.5, -4.5, 1.0),  # motion
    'IT':   ( 2.5, -1.0, -3.5, 1.2),  # inferotemporal
    'A1L':  (-3.5,  0.0, -1.5, 1.0),  # primary auditory -- left
    'A1R':  ( 3.5,  0.0, -1.5, 1.0),  # primary auditory -- right
    'A2L':  (-3.5,  0.5, -0.8, 0.8),  # belt auditory -- left
    'A2R':  ( 3.5,  0.5, -0.8, 0.8),  # belt auditory -- right
    'STGL': (-4.0,  1.0,  0.0, 1.0),  # superior temporal -- left
    'STGR': ( 4.0,  1.0,  0.0, 1.0),  # superior temporal -- right
    'S1':   ( 0.0,  3.0,  1.5, 2.0),  # somatosensory
    'M1':   ( 0.0,  3.0,  3.0, 2.0),  # primary motor
    'PFC':  ( 0.0,  2.0,  5.5, 2.0),  # lateral PFC
    'mPFC': ( 0.0,  2.5,  5.0, 0.8),  # medial PFC
    'PCC':  ( 0.0,  1.0,  2.0, 0.8),  # posterior cingulate
    'CA3':  (-1.5, -2.5,  0.0, 0.8),  # hippocampus CA3
    'CA1':  (-1.5, -3.0,  0.2, 0.8),  # hippocampus CA1
    'BLA':  (-2.0, -2.5, -1.5, 0.8),  # basolateral amygdala
    'ACC':  ( 0.0,  1.5,  4.0, 1.2),  # anterior cingulate
    'AG':   ( 3.5,  0.5,  1.5, 1.0),  # angular gyrus
}

_REGION_N: Dict[str, int] = {
    'V1': 20, 'V2': 12, 'V4': 12, 'MT': 12, 'IT': 24,
    'A1L': 10, 'A1R': 10, 'A2L': 8, 'A2R': 8,
    'STGL': 12, 'STGR': 12,
    'S1': 32, 'M1': 32, 'PFC': 32, 'mPFC': 8, 'PCC': 8,
    'CA3': 15, 'CA1': 10, 'BLA': 10, 'ACC': 20, 'AG': 16,
}

_REGION_SYNTYPE: Dict[str, str] = {
    'V1': 'AMPA', 'V2': 'AMPA', 'V4': 'AMPA', 'MT': 'AMPA',
    'IT': 'NMDA', 'A1L': 'AMPA', 'A1R': 'AMPA',
    'A2L': 'AMPA', 'A2R': 'AMPA',
    'STGL': 'NMDA', 'STGR': 'NMDA',
    'S1': 'AMPA', 'M1': 'AMPA',
    'PFC': 'NMDA', 'mPFC': 'NMDA', 'PCC': 'NMDA',
    'CA3': 'AMPA', 'CA1': 'AMPA', 'BLA': 'GABA',
    'ACC': 'AMPA', 'AG': 'NMDA',
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _encode_image_b64(arr: np.ndarray) -> str:
    """Encode a (H, W, 3) uint8 RGB array as a base64 PNG string."""
    if arr is None or arr.size == 0:
        # 1-pixel black fallback
        arr = np.zeros((1, 1, 3), dtype=np.uint8)

    if _PIL_OK:
        img = _PILImage.fromarray(arr.astype(np.uint8), 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('ascii')

    # Fallback: matplotlib imsave
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.imsave(buf, arr.astype(np.uint8), format='png')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')
    except Exception:
        # Last resort: raw bytes
        return base64.b64encode(arr.astype(np.uint8).tobytes()).decode('ascii')


def _sim_age_human(sim_time_s: float) -> str:
    """Convert simulation seconds to a human-readable age string."""
    if sim_time_s < 3600:
        m = int(sim_time_s // 60)
        s = int(sim_time_s % 60)
        return f'{m}m {s}s'
    if sim_time_s < 86400:
        h = int(sim_time_s // 3600)
        m = int((sim_time_s % 3600) // 60)
        return f'{h}h {m}m'
    days = sim_time_s / 86400.0
    if days < 30:
        return f'{days:.1f}d'
    months = days / 30.0
    return f'{months:.1f}mo'


def _safe_tolist(arr) -> list:
    """Convert numpy array to JSON-serialisable Python list safely."""
    if arr is None:
        return []
    try:
        return [float(x) for x in np.asarray(arr).ravel()]
    except Exception:
        return []


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Neuron layout builder (deterministic, called once on attach)
# ---------------------------------------------------------------------------

def _angle_to_hex(angle_deg: float) -> str:
    """Map a V1 preferred orientation angle [0,180) to an RGB hex colour."""
    import colorsys
    hue = (angle_deg % 180.0) / 180.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def _build_neuron_layout() -> List[dict]:
    """
    Build a deterministic 3-D layout for ~1000 neurons across all cortical
    regions.  Each real tracked neuron has ``background: False``.  The
    remaining ~677 are GABA interneurons (``background: True``) scattered
    within the same regional volumes to fill the spec's ~1000 target.

    Each dict: {id, region, pos: [x,y,z], type, background}
    """
    rng = np.random.RandomState(42)
    neurons: List[dict] = []

    for region, (cx, cy, cz, scatter) in _REGION_LAYOUT.items():
        n_real = _REGION_N.get(region, 8)
        n_bg   = max(2, n_real * 2)          # ~2x background interneurons
        syn_type = _REGION_SYNTYPE.get(region, 'AMPA')
        centre   = np.array([cx, cy, cz])

        # Real (tracked) neurons
        pos_real = rng.randn(n_real, 3) * scatter + centre
        for i in range(n_real):
            neurons.append({
                'id':         f'{region}_{i}',
                'region':     region,
                'pos':        [round(float(pos_real[i, 0]), 3),
                               round(float(pos_real[i, 1]), 3),
                               round(float(pos_real[i, 2]), 3)],
                'type':       syn_type,
                'background': False,
            })

        # Background interneurons (GABA, slightly wider scatter)
        pos_bg = rng.randn(n_bg, 3) * (scatter * 1.3) + centre
        for i in range(n_bg):
            neurons.append({
                'id':         f'{region}_bg_{i}',
                'region':     region,
                'pos':        [round(float(pos_bg[i, 0]), 3),
                               round(float(pos_bg[i, 1]), 3),
                               round(float(pos_bg[i, 2]), 3)],
                'type':       'GABA',
                'background': True,
            })

    return neurons


# Pre-build layout once at import time
_NEURON_LAYOUT: List[dict] = _build_neuron_layout()

# Map region+index -> flat neuron index for fast lookup
_NEURON_ID_MAP: Dict[str, int] = {
    n['id']: i for i, n in enumerate(_NEURON_LAYOUT)
}


# ---------------------------------------------------------------------------
# VisualizationServer
# ---------------------------------------------------------------------------

class VisualizationServer:
    """
    WebSocket visualization server for CAINE (Module 10).

    Connects to all CAINE subsystems via ``attach()`` and broadcasts a
    structured JSON frame at up to 10 Hz to every connected client.

    The server runs its asyncio event loop in a daemon background thread
    so it never blocks the simulation loop.
    """

    PORT = _PORT_DEFAULT

    def __init__(self, port: int = _PORT_DEFAULT):
        self._port = port

        # ---- Module references (set via attach) ----
        self._neuro         = None   # NeurochemicalSystem
        self._v1            = None   # V1Population
        self._a1            = None   # A1Population
        self._it            = None   # ITPopulation
        self._stg           = None   # STGPopulation
        self._pfc           = None   # PFCPopulation
        self._ag            = None   # AngularGyrusPopulation
        self._dmn           = None   # DMNMonitor
        self._hippocampus   = None   # Hippocampus (limbic.py)
        self._sensory       = None   # SensoryLayer
        self._env           = None   # CaineEnvironment
        self._stage_manager = None   # StageManager
        self._avatar        = None   # AvatarController
        self._media         = None   # MediaLearningSystem
        self._motor         = None   # MotorCortex (motor.py)   -- Panel 6
        self._parenting     = None   # ParentingSystem (parenting.py) -- Panel 7

        # ---- Neurochemical history ring buffers ----
        # {chemical: deque of (t_s, value)}
        self._neuro_hist: Dict[str, deque] = {
            c: deque(maxlen=_MAX_HISTORY_PTS) for c in _NEURO_CHEMICALS
        }
        self._neuro_hist_t: deque = deque(maxlen=_MAX_HISTORY_PTS)

        # ---- Neurochemical event log ----
        self._event_log: deque = deque(maxlen=_MAX_EVENT_LOG)
        self._event_log_cursor: int = 0   # index into neuro.event_log already seen

        # ---- Sensory cache (updated from tick sensory_result) ----
        self._last_mel:     np.ndarray = np.zeros(128, dtype='f4')
        self._last_waveform: np.ndarray = np.zeros(512, dtype='f4')
        self._last_camera:  Optional[np.ndarray] = None

        # ---- Session synapse stats ----
        self._synapses_created: int = 0
        self._synapses_pruned:  int = 0
        self._last_it_weights:  Optional[np.ndarray] = None

        # ---- Synapse pruning / sprouting tracking (Panel 1) ----
        # Maps (from_id, to_id) -> last known weight; stale entries auto-expire
        self._prev_syn_weights: Dict[Tuple[str, str], float] = {}
        # Sets hold transient flags that clear every _SYN_FLAG_TTL frames
        self._pruned_syn_ids:   set = set()
        self._sprouting_syn_ids: set = set()
        self._syn_flag_tick:    int = 0
        self._SYN_FLAG_TTL      = 5   # frames before flags are cleared

        # ---- Scheduled session calendar (Panel 7) ----
        self._scheduled_sessions: List[dict] = []

        # ---- Spike decay cache for Panel 1 ----
        # {neuron_id: spike_level [0..1] -- decays to 0}
        self._spike_levels: Dict[str, float] = {}

        # ---- Mission Control state ----
        self.is_paused:  bool  = False
        self.time_scale: float = 1.0
        self._command_queue: deque = deque(maxlen=_MAX_COMMANDS)

        # ---- WebSocket server internals ----
        self._clients:     Set = set()
        self._loop:        Optional[asyncio.AbstractEventLoop] = None
        self._thread:      Optional[threading.Thread]          = None
        self._ws_server                                        = None
        self._running:     bool = False
        self._last_send_wall: float = 0.0

        # Simulated time (updated each tick)
        self._sim_time_s: float = 0.0

        print(f'[viz] VisualizationServer initialised (port {self._port}).')

    # ------------------------------------------------------------------
    # Wiring API
    # ------------------------------------------------------------------

    def attach(self, **modules) -> None:
        """
        Attach any combination of live module instances.

        Keyword argument names:
            neuro, v1, a1, it, stg, pfc, ag, dmn, hippocampus,
            sensory, env, stage_manager, avatar, media,
            motor, parenting
        """
        mapping = {
            'neuro':         '_neuro',
            'v1':            '_v1',
            'a1':            '_a1',
            'it':            '_it',
            'stg':           '_stg',
            'pfc':           '_pfc',
            'ag':            '_ag',
            'dmn':           '_dmn',
            'hippocampus':   '_hippocampus',
            'sensory':       '_sensory',
            'env':           '_env',
            'stage_manager': '_stage_manager',
            'avatar':        '_avatar',
            'media':         '_media',
            'motor':         '_motor',
            'parenting':     '_parenting',
        }
        for key, attr in mapping.items():
            if key in modules:
                setattr(self, attr, modules[key])
                print(f'[viz] Attached {key} -> {type(modules[key]).__name__}')

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the asyncio WebSocket server in a background daemon thread."""
        if not _WS_OK:
            print('[viz] WARNING: websockets package not installed -- '
                  'running in log-only mode (no WebSocket output).')
        if self._running:
            print('[viz] Already running.')
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_thread, daemon=True, name='caine-viz-ws')
        self._thread.start()
        print(f'[viz] WebSocket server starting on ws://localhost:{self._port}')

    def stop(self) -> None:
        """Shut down the WebSocket server."""
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        print('[viz] WebSocket server stopped.')

    # ------------------------------------------------------------------
    # Main tick (called from simulation loop)
    # ------------------------------------------------------------------

    def tick(self, sim_time_s: float = 0.0, dt_ms: float = 20.0,
             sensory_result: Optional[dict] = None,
             cortex_state:   Optional[dict] = None) -> None:
        """
        Advance the visualization server by one simulation tick.

        Parameters
        ----------
        sim_time_s      : current simulation time in seconds
        dt_ms           : simulation timestep in milliseconds
        sensory_result  : dict returned by SensoryLayer.update() this tick
                          (keys: mel_energy, orient_energy, a1_rates, dog, ...)
        cortex_state    : dict passed to StageManager.tick()
                          (keys: dmn_correlation, pfc_wm_span_ms, ...)
        """
        self._sim_time_s = sim_time_s

        # Cache sensory data
        if sensory_result is not None:
            mel = sensory_result.get('mel_energy')
            if mel is not None:
                self._last_mel = np.asarray(mel, dtype='f4')
            raw = sensory_result.get('audio_frame')
            if raw is not None:
                n = min(512, len(raw))
                self._last_waveform[:] = 0
                self._last_waveform[:n] = np.asarray(raw[:n], dtype='f4')

        # Pull camera frame from environment
        if self._env is not None:
            try:
                self._last_camera = self._env.get_camera_feed()
            except Exception:
                pass

        # Update neurochemical history + new event log entries
        self._update_neuro_history(sim_time_s)

        # Track IT weight changes for synapse creation / pruning stats
        self._update_synapse_stats()

        # Clear transient pruned/sprouting flags every TTL frames
        self._syn_flag_tick += 1
        if self._syn_flag_tick % self._SYN_FLAG_TTL == 0:
            self._pruned_syn_ids.clear()
            self._sprouting_syn_ids.clear()

        # Decay spike levels (τ = 200 ms -> decay per tick = exp(-dt/200))
        decay = float(np.exp(-dt_ms / 200.0))
        for nid in list(self._spike_levels.keys()):
            self._spike_levels[nid] *= decay
            if self._spike_levels[nid] < 0.01:
                del self._spike_levels[nid]

        # Inject fresh spike events from live populations
        self._update_spike_levels()

        # Throttle: only send if enough wall-time has passed
        if not self._running:
            return
        wall_now = time.time()
        if wall_now - self._last_send_wall < _FRAME_INTERVAL:
            return
        self._last_send_wall = wall_now

        if not self._clients or self._loop is None:
            return

        try:
            frame      = self._build_frame(sim_time_s, cortex_state or {})
            frame_json = json.dumps(frame)
            asyncio.run_coroutine_threadsafe(
                self._broadcast(frame_json), self._loop)
        except Exception as exc:
            print(f'[viz] tick error: {exc}')

    # ------------------------------------------------------------------
    # Command queue (Panel 7 -> simulation loop)
    # ------------------------------------------------------------------

    def pop_commands(self) -> List[dict]:
        """Return and clear all pending control commands from clients."""
        cmds = list(self._command_queue)
        self._command_queue.clear()
        return cmds

    # ------------------------------------------------------------------
    # Frame builder
    # ------------------------------------------------------------------

    def _build_frame(self, sim_time_s: float,
                     cortex_state: dict) -> dict:
        return {
            't':               round(sim_time_s, 3),
            'panel1_brain':    self._build_panel1_brain(),
            'panel2_vision':   self._build_panel2_vision(),
            'panel3_auditory': self._build_panel3_auditory(),
            'panel4_neuro':    self._build_panel4_neuro(),
            'panel5_dev':      self._build_panel5_dev(sim_time_s, cortex_state),
            'panel6_internal': self._build_panel6_internal(),
            'panel7_controls': self._build_panel7_controls(),
            'panel8_env':      self._build_panel8_env(),
        }

    # ------------------------------------------------------------------
    # Panel 1 -- 3-D Brain
    # ------------------------------------------------------------------

    def _build_panel1_brain(self) -> dict:
        """
        Neuron spheres with spike intensity, synapse lines with weight opacity.

        Neurons: full _NEURON_LAYOUT list with current spike level injected.
        Synapses: up to MAX_SYNAPSES sampled from IT.W_hebb and STG.W_in/W_recur.
        """
        # Build neuron list
        neurons = []
        for nd in _NEURON_LAYOUT:
            spike = round(self._spike_levels.get(nd['id'], 0.0), 4)
            neurons.append({
                'id':         nd['id'],
                'region':     nd['region'],
                'pos':        nd['pos'],
                'type':       nd['type'],
                'spike':      spike,
                'color':      _SYNTYPE_COLORS.get(nd['type'], '#aaaaaa'),
                'background': nd.get('background', False),
            })

        synapses = self._sample_synapses()

        return {
            'neurons':           neurons,
            'synapses':          synapses,
            'total_neuron_count': len(neurons),
        }

    def _sample_synapses(self) -> List[dict]:
        """
        Sample up to MAX_SYNAPSES synapse lines from live weight matrices.

        Each synapse carries:
          pruned    -- True if weight dropped below noise floor this window
                       (frontend: fade line out / show ghost)
          sprouting -- True if weight grew from near-zero this window
                       (frontend: render as faint dotted line)

        Sources:
          - IT.W_hebb  (24 x 36)  : V2/V4/MT -> IT
          - STG.W_in   (24 x 16)  : A2L/A2R  -> STGL/STGR
          - STG.W_recur(24 x 24)  : STGL/STGR recurrent
        """
        rng   = np.random.RandomState(int(time.time() * 10) % 100000)
        synaps: List[dict] = []

        def _classify(key: Tuple[str, str], w: float) -> Tuple[bool, bool]:
            """Return (pruned, sprouting) based on weight history."""
            prev = self._prev_syn_weights.get(key)
            self._prev_syn_weights[key] = w
            if prev is None:
                return False, False
            pruned    = prev > 0.01  and w < 0.005
            sprouting = prev < 0.005 and w > 0.01
            if pruned:
                self._pruned_syn_ids.add(key)
            if sprouting:
                self._sprouting_syn_ids.add(key)
            return key in self._pruned_syn_ids, key in self._sprouting_syn_ids

        # ---- IT.W_hebb ----
        if self._it is not None:
            W = getattr(self._it, 'W_hebb', None)
            if W is not None:
                n_it = min(W.shape[0], _REGION_N['IT'])
                v2_n, v4_n, mt_n = _REGION_N['V2'], _REGION_N['V4'], _REGION_N['MT']
                src_regions = (
                    [('V2',  i) for i in range(v2_n)]
                    + [('V4', i) for i in range(v4_n)]
                    + [('MT', i) for i in range(mt_n)]
                )
                n_cols  = min(W.shape[1], len(src_regions))
                indices = rng.choice(n_it * n_cols,
                                     size=min(_MAX_SYNAPSES // 2, n_it * n_cols),
                                     replace=False)
                for idx in indices:
                    r, c = divmod(int(idx), n_cols)
                    if c >= len(src_regions):
                        continue
                    src_reg, src_i = src_regions[c]
                    w   = float(W[r, c]) if c < W.shape[1] else 0.0
                    key = (f'{src_reg}_{src_i}', f'IT_{r}')
                    pruned, sprouting = _classify(key, w)
                    # Include pruned synapses (weight=0) so frontend can fade them
                    if w < 0.005 and not pruned:
                        continue
                    synaps.append({
                        'from_id':   key[0],
                        'to_id':     key[1],
                        'weight':    round(w, 4),
                        'type':      'NMDA',
                        'active':    w > 0.1,
                        'pruned':    pruned,
                        'sprouting': sprouting,
                    })
                    if len(synaps) >= _MAX_SYNAPSES:
                        break

        # ---- STG.W_in ----
        if self._stg is not None and len(synaps) < _MAX_SYNAPSES:
            W_in = getattr(self._stg, 'W_in', None)
            if W_in is not None:
                n_a2  = min(W_in.shape[1], _REGION_N['A2L'] + _REGION_N['A2R'])
                n_stg = min(W_in.shape[0], _REGION_N['STGL'] + _REGION_N['STGR'])
                indices = rng.choice(n_stg * n_a2,
                                     size=min(50, n_stg * n_a2),
                                     replace=False)
                for idx in indices:
                    r, c = divmod(int(idx), n_a2)
                    if r >= W_in.shape[0] or c >= W_in.shape[1]:
                        continue
                    w         = float(W_in[r, c])
                    stg_side  = 'STGL' if r < _REGION_N['STGL'] else 'STGR'
                    a2_side   = 'A2L'  if c < _REGION_N['A2L']  else 'A2R'
                    key       = (f'{a2_side}_{c % _REGION_N["A2L"]}',
                                 f'{stg_side}_{r % _REGION_N["STGL"]}')
                    pruned, sprouting = _classify(key, w)
                    if w < 0.005 and not pruned:
                        continue
                    synaps.append({
                        'from_id':   key[0],
                        'to_id':     key[1],
                        'weight':    round(w, 4),
                        'type':      'AMPA',
                        'active':    w > 0.1,
                        'pruned':    pruned,
                        'sprouting': sprouting,
                    })
                    if len(synaps) >= _MAX_SYNAPSES:
                        break

        return synaps

    # ------------------------------------------------------------------
    # Panel 2 -- Vision Feed
    # ------------------------------------------------------------------

    def _build_panel2_vision(self) -> dict:
        """
        camera_feed_b64 : base64 PNG of CAINE's 64x64 camera
        v1_orientations  : 4-column orientation energies (from V1.pref_orient groups)
        it_heatmap       : 24 normalised IT firing rates
        ach_attention    : float [0,1] -- acetylcholine-weighted attention
        """
        # Camera feed
        cam_b64 = _encode_image_b64(self._last_camera)

        # V1 edge orientation map (group 20 neurons into 4 orientation columns)
        v1_orients = [0.0, 0.0, 0.0, 0.0]
        if self._v1 is not None:
            rates = getattr(self._v1, 'rate_est', None)
            if rates is not None and len(rates) >= 20:
                n = len(rates)
                col_sz = max(1, n // 4)
                for k in range(4):
                    v1_orients[k] = round(
                        float(rates[k * col_sz:(k + 1) * col_sz].mean()), 4)

        # IT heatmap
        it_heatmap = [0.0] * _REGION_N['IT']
        if self._it is not None:
            rates = getattr(self._it, 'rate_est', None)
            if rates is not None:
                mx = float(rates.max()) or 1.0
                it_heatmap = [round(float(r) / mx, 4) for r in rates[:24]]

        # ACh level
        ach = 0.0
        if self._neuro is not None:
            try:
                ach = _safe_float(self._neuro.acetylcholine)
            except Exception:
                pass

        # V1 orientation columns -- per-neuron color coding (hue = angle)
        # v1_columns: [{angle_deg, color_hex, rate, spike}]  -- one entry per neuron
        v1_columns = []
        if self._v1 is not None:
            rates  = getattr(self._v1, 'rate_est',   None)
            angles = getattr(self._v1, 'pref_orient', None)
            if rates is not None and angles is not None:
                n = min(len(rates), len(angles))
                mx = float(rates[:n].max()) or 1.0
                for i in range(n):
                    ang = float(angles[i]) % 180.0
                    v1_columns.append({
                        'angle_deg': round(ang, 1),
                        'color':     _angle_to_hex(ang),
                        'rate':      round(float(rates[i]) / mx, 4),
                        'spike':     round(self._spike_levels.get(f'V1_{i}', 0.0), 4),
                    })

        # ACh attention spotlight -- 8x8 spatial map
        # Rows = V1 rate buckets, scaled by ACh; each cell = mean activation
        # of V1 neurons whose pref_orient maps to that cell's angle quadrant.
        ach_map = [0.0] * 64   # 8x8 flattened, row-major
        if self._v1 is not None and v1_columns:
            import colorsys as _cs
            n_cells = 8
            buckets = [0.0] * (n_cells * n_cells)
            counts  = [0]   * (n_cells * n_cells)
            for i, col in enumerate(v1_columns):
                # Map angle [0,180) to x [0,7]; rate to y [0,7]
                xi = int(col['angle_deg'] / 180.0 * n_cells) % n_cells
                yi = int(col['rate'] * n_cells) % n_cells
                idx = yi * n_cells + xi
                buckets[idx] += col['rate']
                counts[idx]  += 1
            mx_b = max(buckets) or 1.0
            ach_map = [round(ach * (b / mx_b), 4) for b in buckets]

        # V1 DoG response (orientation selectivity overlay)
        dog_flat: list = []
        if self._sensory is not None:
            dog = getattr(self._sensory, '_last_dog', None)
            if dog is not None:
                dog_ds = np.asarray(dog)
                if dog_ds.ndim == 2 and dog_ds.size > 0:
                    h, w = dog_ds.shape
                    th, tw = min(h, 16), min(w, 16)
                    step_h, step_w = max(1, h // th), max(1, w // tw)
                    dog_flat = _safe_tolist(dog_ds[::step_h, ::step_w][:th, :tw])

        return {
            'camera_feed_b64':   cam_b64,
            'v1_orientations':   v1_orients,      # legacy 4-bucket summary
            'v1_columns':        v1_columns,       # per-neuron {angle, color, rate, spike}
            'it_heatmap':        it_heatmap,
            'ach_attention':     round(ach, 4),    # scalar (legacy)
            'ach_attention_map': ach_map,          # 8x8 spatial spotlight (64 floats)
            'dog_overlay':       dog_flat,
        }

    # ------------------------------------------------------------------
    # Panel 3 -- Auditory Feed
    # ------------------------------------------------------------------

    def _build_panel3_auditory(self) -> dict:
        """
        waveform      : 512 samples (last audio frame seen)
        mel_bands     : 128 Mel filterbank energies
        a1_activation : 20 A1 firing rates (tonotopic bar)
        stg_label     : highest-confidence STG cluster phoneme label
        stg_confidence: normalised confidence [0,1]
        """
        # Waveform
        waveform = _safe_tolist(self._last_waveform)

        # Mel bands
        mel_bands = _safe_tolist(self._last_mel)

        # A1 activation + preferred frequencies for tonotopic axis labelling
        a1_act      = [0.0] * 20
        a1_pref_freq = [0.0] * 20
        if self._a1 is not None:
            rates = getattr(self._a1, 'rate_est', None)
            freqs = getattr(self._a1, 'pref_freq', None)
            if rates is not None:
                mx = float(rates.max()) or 1.0
                a1_act = [round(float(r) / mx, 4) for r in rates[:20]]
            if freqs is not None:
                a1_pref_freq = [round(float(f), 1) for f in freqs[:20]]

        # STG label -- highest-rate cluster
        stg_label = 'sil'
        stg_conf  = 0.0
        if self._stg is not None:
            rates = getattr(self._stg, 'rate_est', None)
            if rates is not None and len(rates) > 0:
                mx     = float(rates.max())
                winner = int(np.argmax(rates))
                stg_label = _STG_LABELS[winner % len(_STG_LABELS)]
                stg_conf  = round(mx / (mx + 50.0), 4)   # soft sigmoid vs 50 Hz

        return {
            'waveform':        waveform,
            'mel_bands':       mel_bands,
            'a1_activation':   a1_act,
            'a1_pref_freq':    a1_pref_freq,   # Hz per neuron -- tonotopic axis labels
            'stg_label':       stg_label,
            'stg_confidence':  stg_conf,
        }

    # ------------------------------------------------------------------
    # Panel 4 -- Neurochemical Dashboard
    # ------------------------------------------------------------------

    def _build_panel4_neuro(self) -> dict:
        """
        current  : {chemical: float} -- live concentrations
        history  : {chemical: [float...], t: [float...]} -- last 60 s
        events   : [{t_ms, chemical, delta, trigger}...] -- event log
        colors   : {chemical: html_color}
        """
        current: Dict[str, float] = {}
        if self._neuro is not None:
            try:
                snap = self._neuro.snapshot()
                current = {k: round(float(v), 5) for k, v in snap.items()
                           if k in _NEURO_CHEMICALS}
            except Exception:
                pass
        # Fallback zeros
        for c in _NEURO_CHEMICALS:
            current.setdefault(c, 0.0)

        # History
        hist_t  = list(self._neuro_hist_t)
        history = {c: list(self._neuro_hist[c]) for c in _NEURO_CHEMICALS}
        history['t'] = hist_t

        # Event log
        events = list(self._event_log)

        return {
            'current': current,
            'history': history,
            'events':  events,
            'colors':  _CHEM_COLORS,
        }

    # ------------------------------------------------------------------
    # Panel 5 -- Developmental Metrics
    # ------------------------------------------------------------------

    def _build_panel5_dev(self, sim_time_s: float, cortex_state: dict) -> dict:
        """
        Stage badge, simulated age, vocabulary, synapse counts,
        DMN correlation, PFC WM span, self-model confidence,
        stage exit conditions checklist.
        """
        stage      = 0
        stage_name = 'Stage 0 -- The Void'
        conditions: dict = {}
        conds_met  = 0
        conds_tot  = 0

        if self._stage_manager is not None:
            try:
                stage      = int(self._stage_manager.stage)
                stage_name = self._stage_manager.stage_name
                conditions = self._stage_manager.last_conditions
                conds_met  = sum(1 for v in conditions.values() if v)
                conds_tot  = len(conditions)
            except Exception:
                pass

        # Vocabulary: confirmed STG Hebbian bindings above threshold
        vocab = 0
        if self._it is not None:
            W = getattr(self._it, 'W_hebb', None)
            if W is not None:
                vocab = int((W.max(axis=1) > 0.3).sum())

        # Synapse count estimate: sum of weight matrix sizes
        syn_count = self._estimate_synapse_count()

        # DMN correlation
        dmn_corr = _safe_float(cortex_state.get('dmn_correlation', 0.0))
        if dmn_corr == 0.0 and self._dmn is not None:
            try:
                if self._dmn.corr_log:
                    dmn_corr = round(float(self._dmn.corr_log[-1][1]), 4)
            except Exception:
                pass

        # PFC WM span
        pfc_wm_ms = _safe_float(cortex_state.get('pfc_wm_span_ms', 0.0))
        if pfc_wm_ms == 0.0 and self._pfc is not None:
            try:
                myelin = _safe_float(getattr(self._pfc, 'myelination_factor', 0.0))
                # Rough estimate: myelination_factor 0->1 maps to 200ms->5000ms
                pfc_wm_ms = round(200.0 + myelin * 4800.0, 1)
            except Exception:
                pass

        # Self-model confidence: mPFC population stability (std of rates)
        self_conf = 0.0
        if self._pfc is not None:
            try:
                mpfc = np.asarray(self._pfc.mPFC_rates)
                # Stability: inverse of coefficient of variation, clipped [0,1]
                if mpfc.mean() > 1e-6:
                    self_conf = round(
                        float(np.clip(1.0 - mpfc.std() / (mpfc.mean() + 1e-6),
                                      0.0, 1.0)), 4)
            except Exception:
                pass

        # Hippocampal episode count
        n_episodes = 0
        if self._hippocampus is not None:
            try:
                n_episodes = _safe_int(self._hippocampus.n_episodes)
            except Exception:
                pass

        # PFC myelination factor
        myelin_f = 0.0
        if self._pfc is not None:
            myelin_f = _safe_float(getattr(self._pfc, 'myelination_factor', 0.0))

        return {
            'stage':                    stage,
            'stage_name':               stage_name,
            'sim_age_s':                round(sim_time_s, 1),
            'sim_age_human':            _sim_age_human(sim_time_s),
            'vocabulary_size':          vocab,
            'synapse_count':            syn_count,
            'synapses_created_session': self._synapses_created,
            'synapses_pruned_session':  self._synapses_pruned,
            'dmn_correlation':          round(dmn_corr, 4),
            'dmn_active':               bool(getattr(self._dmn, 'dmn_active', False)),
            'dmn_emerged':              bool(getattr(self._dmn, 'dmn_emerged', False)),
            'pfc_wm_span_ms':           round(pfc_wm_ms, 1),
            'pfc_myelination':          round(myelin_f, 4),
            'self_model_confidence':    self_conf,
            'n_episodes':               n_episodes,
            'stage_conditions':         {k: bool(v) for k, v in conditions.items()},
            'conditions_met':           conds_met,
            'conditions_total':         conds_tot,
        }

    # ------------------------------------------------------------------
    # Panel 6 -- Internal State (INFERRED)
    # ------------------------------------------------------------------

    def _build_panel6_internal(self) -> dict:
        """
        Best-effort inference of CAINE's internal state.
        All fields are labelled INFERRED -- not ground truth.
        """
        # Working memory contents: top-N IT + STG activations
        wm_contents: List[str] = []
        if self._it is not None:
            rates = getattr(self._it, 'rate_est', None)
            if rates is not None and len(rates) > 0:
                top_it = int(np.argmax(rates))
                wm_contents.append(f'IT_concept_{top_it}')
        if self._stg is not None:
            rates = getattr(self._stg, 'rate_est', None)
            if rates is not None and len(rates) > 0:
                top_stg = int(np.argmax(rates))
                label   = _STG_LABELS[top_stg % len(_STG_LABELS)]
                wm_contents.append(f'phoneme_{label}')

        # Emotional state -- nearest neighbor to named state table
        emo_label, emo_conf = self._infer_emotional_state()

        # Motor intention: M1 population vector direction
        # Primary source: MotorCortex.m1_pop.rate_est (60 neurons, 6 joint columns)
        # Compute direction as the weighted centroid of 6 joint column unit vectors.
        # Column layout (from motor.py): cols 0-5 = avatar joints
        #   0=spine, 1=shoulder_L, 2=shoulder_R, 3=elbow_L, 4=elbow_R, 5=leg_R
        # Unit vectors arranged in a rough 3-D kinematic basis.
        _JOINT_DIRS = np.array([
            [ 0.0,  1.0,  0.0],   # spine: upward
            [-1.0,  0.5,  0.0],   # shoulder_L: left-up
            [ 1.0,  0.5,  0.0],   # shoulder_R: right-up
            [-1.0, -0.5,  0.3],   # elbow_L: left-down-forward
            [ 1.0, -0.5,  0.3],   # elbow_R: right-down-forward
            [ 0.0, -1.0,  0.0],   # leg_R: downward
        ], dtype='f4')

        motor       = [0.0, 0.0, 0.0]
        joint_angles = [0.0] * 6

        if self._motor is not None:
            try:
                m1_pop = getattr(self._motor, 'm1_pop', None)
                if m1_pop is not None:
                    rates = getattr(m1_pop, 'rate_est', None)
                    if rates is not None and len(rates) >= 6:
                        # Average rate per joint column (10 neurons each for cols 0-5)
                        col_rates = np.array([
                            float(rates[c * 10: (c + 1) * 10].mean())
                            for c in range(6)
                        ], dtype='f4')
                        col_rates /= (col_rates.max() + 1e-6)
                        direction = (col_rates[:, None] * _JOINT_DIRS).sum(axis=0)
                        norm = float(np.linalg.norm(direction))
                        if norm > 1e-6:
                            direction /= norm
                        motor = [round(float(d), 4) for d in direction]

                ja = getattr(self._motor, 'joint_angles', None)
                if ja is not None:
                    joint_angles = [round(float(a), 4) for a in ja[:6]]
            except Exception:
                pass

        # Fallback: avatar gaze when motor cortex not attached
        if motor == [0.0, 0.0, 0.0] and self._avatar is not None:
            last = getattr(self._avatar, '_last_state', {})
            if last:
                gaze  = last.get('gaze_target', (0.0, 1.0, 3.0))
                motor = [round(float(g), 4) for g in gaze[:3]]

        return {
            'wm_contents':          wm_contents or ['<empty>'],
            'emotional_state':      emo_label,
            'emotional_confidence': round(emo_conf, 4),
            'motor_intention':      motor,         # 3-D unit vector
            'joint_angles':         joint_angles,  # 6 joint angles (radians)
            'note':                 'INFERRED -- not ground truth',
        }

    # ------------------------------------------------------------------
    # Panel 7 -- Mission Control
    # ------------------------------------------------------------------

    def _build_panel7_controls(self) -> dict:
        # Mother's pending action (first item in parenting._action_queue if present)
        mother_pending = None
        if self._parenting is not None:
            try:
                q = getattr(self._parenting, '_action_queue', [])
                if q:
                    first = q[0]
                    mother_pending = str(getattr(first, 'action_type',
                                                 getattr(first, 'action', str(first))))
            except Exception:
                pass

        return {
            'is_paused':          self.is_paused,
            'time_scale':         self.time_scale,
            'scheduled_sessions': list(self._scheduled_sessions),
            'mother_pending':     mother_pending,  # None or action label string
        }

    # ------------------------------------------------------------------
    # Panel 8 -- 3-D Environment
    # ------------------------------------------------------------------

    def _build_panel8_env(self) -> dict:
        """
        Observer camera feed (base64 PNG) + world state snapshot.

        observer_b64 : base64-encoded PNG of the 800×600 observer camera.
                       Empty string when env is not attached or render fails.
        objects      : list of {id, pos:[x,y,z], type, color:[r,g,b]}
        light        : {dir:[x,y,z], color:[r,g,b], ambient:float}
        stage        : int
        m1_suppressed, father_voice_onset, pfc_myelination : bool flags
        """
        obs_b64   = ''
        objects   = []
        light     = {'dir': [0.6, 1.0, 0.4], 'color': [1.0, 0.98, 0.9], 'ambient': 0.15}
        stage     = 0
        env_flags = {}

        if self._env is not None:
            # Observer camera
            try:
                frame = self._env.get_observer_feed()
                if frame is not None and frame.size > 0:
                    obs_b64 = _encode_image_b64(frame)
            except Exception:
                pass

            # World objects
            try:
                with self._env._lock:
                    snap = dict(self._env._objects)
                for uid, (pos, otype, color) in snap.items():
                    objects.append({
                        'id':    uid,
                        'pos':   [round(float(p), 3) for p in pos],
                        'type':  otype,
                        'color': [round(float(c), 3) for c in color[:3]]
                                  if hasattr(color, '__len__') else [0.8, 0.8, 0.8],
                    })
            except Exception:
                pass

            # Light state
            try:
                light = {
                    'dir':     [round(float(v), 3) for v in self._env._light_dir],
                    'color':   [round(float(v), 3) for v in self._env._light_color],
                    'ambient': round(float(self._env._ambient), 3),
                }
            except Exception:
                pass

            # Stage and flags
            try:
                stage = int(self._env._dev_stage)
                env_flags = {
                    'm1_suppressed':      bool(getattr(self._env, 'm1_suppressed', True)),
                    'father_voice_onset': bool(getattr(self._env, 'father_voice_onset', False)),
                    'pfc_myelination':    bool(getattr(self._env, 'pfc_myelination', False)),
                    'avatar_physics':     bool(getattr(self._env, 'avatar_physics', False)),
                }
            except Exception:
                pass

        # CAINE position
        caine_pos = [0.0, 1.6, 0.0]
        if self._env is not None:
            try:
                cp = self._env.get_caine_position()
                caine_pos = [round(float(v), 3) for v in cp]
            except Exception:
                pass

        # Avatar pose for Three.js skeleton
        avatar_pose = {
            'head_tilt':   0.0,
            'spine_curl':  0.0,
            'hat_tilt':    0.0,
            'eye_width':   1.0,
            'gaze':        [0.0, 0.0, 1.0],
        }
        if self._avatar is not None:
            try:
                pt = getattr(self._avatar, 'current_pose', None)
                if pt is not None:
                    avatar_pose = {
                        'head_tilt':  float(getattr(pt, 'head_tilt',  0.0)),
                        'spine_curl': float(getattr(pt, 'spine_curl', 0.0)),
                        'hat_tilt':   float(getattr(pt, 'hat_tilt',   0.0)),
                        'eye_width':  float(getattr(pt, 'eye_width',  1.0)),
                        'gaze':       list(getattr(pt, 'gaze_target',
                                                   [0.0, 0.0, 1.0]))[:3],
                    }
            except Exception:
                pass

        return {
            'observer_b64': obs_b64,
            'objects':      objects,
            'light':        light,
            'stage':        stage,
            'flags':        env_flags,
            'caine_pos':    caine_pos,
            'avatar_pose':  avatar_pose,
        }

    # ------------------------------------------------------------------
    # Neurochemical history update
    # ------------------------------------------------------------------

    def _update_neuro_history(self, sim_time_s: float) -> None:
        """Push current concentrations into ring buffers; drain new events."""
        if self._neuro is None:
            return

        try:
            snap = self._neuro.snapshot()
        except Exception:
            return

        t_key = round(sim_time_s, 2)
        self._neuro_hist_t.append(t_key)
        for c in _NEURO_CHEMICALS:
            self._neuro_hist[c].append(round(float(snap.get(c, 0.0)), 5))

        # Drain new entries from neuro.event_log
        ev_log = getattr(self._neuro, 'event_log', [])
        while self._event_log_cursor < len(ev_log):
            entry = ev_log[self._event_log_cursor]
            # entry format: (time_ms, EventType, magnitude)
            try:
                t_ms    = _safe_float(entry[0])
                ev_type = str(entry[1])
                mag     = _safe_float(entry[2])
                # Extract chemical from event type name (best-effort)
                chem = _event_type_to_chemical(ev_type)
                self._event_log.append({
                    't_ms':     round(t_ms, 1),
                    'chemical': chem,
                    'delta':    round(mag, 4),
                    'trigger':  ev_type,
                })
            except Exception:
                pass
            self._event_log_cursor += 1

    # ------------------------------------------------------------------
    # Spike level update
    # ------------------------------------------------------------------

    def _update_spike_levels(self) -> None:
        """Inject fresh spike bursts from live population rate_est arrays."""
        def _inject(pop, region_name: str, n_cap: int) -> None:
            if pop is None:
                return
            rates = getattr(pop, 'rate_est', None)
            if rates is None:
                return
            n = min(len(rates), n_cap)
            mx = float(rates[:n].max()) if n > 0 else 0.0
            if mx < 0.5:
                return   # below noise floor
            for i in range(n):
                level = float(rates[i]) / (mx + 1e-6)
                if level > 0.05:
                    nid = f'{region_name}_{i}'
                    self._spike_levels[nid] = max(
                        self._spike_levels.get(nid, 0.0), level)

        _inject(self._v1,  'V1',  _REGION_N['V1'])
        # A1: first half -> A1L, second half -> A1R
        if self._a1 is not None:
            rates = getattr(self._a1, 'rate_est', None)
            if rates is not None:
                n = len(rates)
                half = n // 2
                pop_l = type('_P', (), {'rate_est': rates[:half]})()
                pop_r = type('_P', (), {'rate_est': rates[half:]})()
                _inject(pop_l, 'A1L', _REGION_N['A1L'])
                _inject(pop_r, 'A1R', _REGION_N['A1R'])
        _inject(self._it,  'IT',   _REGION_N['IT'])
        # STG: first half -> STGL, second half -> STGR
        if self._stg is not None:
            rates = getattr(self._stg, 'rate_est', None)
            if rates is not None:
                n = len(rates)
                half = n // 2
                pop_l = type('_P', (), {'rate_est': rates[:half]})()
                pop_r = type('_P', (), {'rate_est': rates[half:]})()
                _inject(pop_l, 'STGL', _REGION_N['STGL'])
                _inject(pop_r, 'STGR', _REGION_N['STGR'])
        _inject(self._pfc, 'PFC',  _REGION_N['PFC'])
        if self._pfc is not None:
            _inject(type('_P', (), {'rate_est': getattr(
                self._pfc, 'mPFC_rates', np.zeros(8))})(),
                'mPFC', _REGION_N['mPFC'])
            _inject(type('_P', (), {'rate_est': getattr(
                self._pfc, 'PCC_rates', np.zeros(8))})(),
                'PCC', _REGION_N['PCC'])
        if self._ag is not None:
            _inject(type('_P', (), {'rate_est': getattr(
                self._ag, 'ag_rates', np.zeros(16))})(),
                'AG', _REGION_N['AG'])
        if self._hippocampus is not None:
            ca3 = getattr(self._hippocampus, 'ca3', None)
            ca1 = getattr(self._hippocampus, 'ca1', None)
            _inject(ca3, 'CA3', _REGION_N['CA3'])
            _inject(ca1, 'CA1', _REGION_N['CA1'])
        # M1 from MotorCortex (60 neurons -> first 32 mapped to 'M1' region)
        if self._motor is not None:
            m1_pop = getattr(self._motor, 'm1_pop', None)
            if m1_pop is not None:
                _inject(m1_pop, 'M1', _REGION_N['M1'])

    # ------------------------------------------------------------------
    # Synapse stats
    # ------------------------------------------------------------------

    def _update_synapse_stats(self) -> None:
        """Track IT.W_hebb changes to count created / pruned synapses."""
        if self._it is None:
            return
        W = getattr(self._it, 'W_hebb', None)
        if W is None:
            return
        if self._last_it_weights is None:
            self._last_it_weights = W.copy()
            return
        diff = W - self._last_it_weights
        self._synapses_created += int((diff > 0.01).sum())
        self._synapses_pruned  += int((diff < -0.05).sum())
        self._last_it_weights   = W.copy()

    def _estimate_synapse_count(self) -> int:
        """Sum of all known weight matrix elements above noise floor."""
        total = 0
        for pop, attr in [
            (self._it,  'W_hebb'),
            (self._stg, 'W_in'),
            (self._stg, 'W_recur'),
        ]:
            if pop is None:
                continue
            W = getattr(pop, attr, None)
            if W is not None:
                total += int((np.asarray(W) > 0.01).sum())
        return total

    # ------------------------------------------------------------------
    # Emotional state inference
    # ------------------------------------------------------------------

    def _infer_emotional_state(self) -> Tuple[str, float]:
        """
        Nearest-neighbor lookup in _EMOTIONAL_STATES.

        Returns (label, confidence) where confidence = 1 - normalised L2.
        """
        if self._neuro is None:
            return 'unknown', 0.0

        try:
            snap = self._neuro.snapshot()
            vec  = np.array([snap.get(c, 0.0) for c in _NEURO_CHEMICALS],
                            dtype='f4')
        except Exception:
            return 'unknown', 0.0

        best_label = 'calm'
        best_dist  = float('inf')

        for label, proto in _EMOTIONAL_STATES.items():
            d = float(np.linalg.norm(vec - np.array(proto, dtype='f4')))
            if d < best_dist:
                best_dist  = d
                best_label = label

        # Confidence: map [0, sqrt(6)] -> [1, 0]
        max_d = float(np.sqrt(len(_NEURO_CHEMICALS)))
        conf  = float(np.clip(1.0 - best_dist / max_d, 0.0, 1.0))
        return best_label, conf

    # ------------------------------------------------------------------
    # Incoming control message handler
    # ------------------------------------------------------------------

    def _handle_control_message(self, msg: dict) -> None:
        """
        Process a control message from a connected Electron / browser client.

        Recognised actions:
          pause, resume, stop, set_time_scale, cortisol_flush,
          forced_rest, stage_rollback, media_upload,
          schedule_father_session, mother_override
        """
        action = msg.get('action', '')

        if action == 'pause':
            self.is_paused = True

        elif action == 'resume':
            self.is_paused = False

        elif action == 'set_time_scale':
            self.time_scale = max(0.0, _safe_float(msg.get('value', 1.0)))

        elif action == 'cortisol_flush':
            # Reset cortisol to baseline directly
            if self._neuro is not None:
                try:
                    chem = self._neuro._chemicals.get('cortisol')
                    if chem is not None:
                        chem.concentration = chem.baseline
                except Exception:
                    pass

        elif action == 'media_upload':
            # Save uploaded file to output/uploads/ with stage metadata sidecar
            fname    = str(msg.get('filename', 'upload.bin'))
            stage    = _safe_int(msg.get('stage', -1))
            data_b64 = msg.get('data_b64', '')
            if data_b64:
                try:
                    up_dir = _paths.UPLOADS_DIR
                    with open(os.path.join(up_dir, fname), 'wb') as f:
                        f.write(base64.b64decode(data_b64))
                    # Write metadata sidecar (.meta.json)
                    meta = {
                        'filename':    fname,
                        'stage':       stage,
                        'uploaded_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'description': str(msg.get('description', '')),
                    }
                    with open(os.path.join(up_dir, fname + '.meta.json'),
                              'w', encoding='utf-8') as mf:
                        json.dump(meta, mf, indent=2)
                    print(f'[viz] Media uploaded: {fname}  stage={stage}')
                except Exception as exc:
                    print(f'[viz] media_upload error: {exc}')

        elif action == 'schedule_father_session':
            # Add a Father session to the scheduler (SessionScheduler in parenting.py)
            session = {
                'id':          str(msg.get('id', f'viz_{int(time.time())}')),
                'time_s':      _safe_float(msg.get('time_s', 0.0)),
                'type':        str(msg.get('type', 'voice_exposure')),
                'file':        str(msg.get('file', '')),
                'repetitions': _safe_int(msg.get('repetitions', 1)),
                'stage_gate':  _safe_int(msg.get('stage_gate', 0)),
                'description': str(msg.get('description', '')),
                'played':      False,
            }
            self._scheduled_sessions.append(session)
            # Forward to live ParentingSystem.scheduler if attached
            if self._parenting is not None:
                try:
                    self._parenting.add_session(session)
                except Exception as exc:
                    print(f'[viz] schedule_father_session parenting error: {exc}')
            print(f'[viz] Scheduled session: {session["description"] or session["id"]} '
                  f'at t={session["time_s"]:.0f}s')

        elif action == 'mother_override':
            # approve=True: let Mother proceed (no-op -- normal flow continues)
            # approve=False: cancel Mother's next planned action by clearing queue
            approve = bool(msg.get('approve', True))
            if not approve and self._parenting is not None:
                try:
                    q = getattr(self._parenting, '_action_queue', None)
                    if q is not None:
                        q.clear()
                        print('[viz] Mother override: next action cancelled.')
                except Exception as exc:
                    print(f'[viz] mother_override error: {exc}')
            elif approve:
                print('[viz] Mother override: approved (proceeding normally).')

        # ----------------------------------------------------------------
        # Environment direct controls (Panel 8)
        # ----------------------------------------------------------------
        elif action == 'env_spawn':
            if self._env is not None:
                try:
                    uid   = str(msg.get('id', f'ui_{int(time.time())}'))
                    pos   = tuple(float(v) for v in msg.get('pos', [0, 0.5, 5]))
                    otype = str(msg.get('object_type', 'sphere'))
                    self._env.spawn_object(uid, pos, object_type=otype)
                    print(f'[viz] env_spawn: {uid} at {pos}')
                except Exception as exc:
                    print(f'[viz] env_spawn error: {exc}')

        elif action == 'env_remove':
            if self._env is not None:
                try:
                    uid = str(msg.get('id', ''))
                    handle = self._env._handles.get(uid)
                    if handle is not None:
                        self._env.remove_object(handle)
                        print(f'[viz] env_remove: {uid}')
                    else:
                        print(f'[viz] env_remove: id not found: {uid}')
                except Exception as exc:
                    print(f'[viz] env_remove error: {exc}')

        elif action == 'env_set_light':
            if self._env is not None:
                try:
                    color     = tuple(float(v) for v in msg.get('color', [1.0, 1.0, 1.0]))
                    intensity = float(msg.get('intensity', 0.15))
                    self._env.set_environment_state({
                        'light_color': color,
                        'ambient':     intensity,
                    })
                    print(f'[viz] env_set_light: color={color} ambient={intensity}')
                except Exception as exc:
                    print(f'[viz] env_set_light error: {exc}')

        elif action == 'env_play_tone':
            if self._env is not None:
                try:
                    import math
                    freq = float(msg.get('frequency', 440.0))
                    dur  = float(msg.get('duration',  1.0))
                    vol  = float(msg.get('volume',    0.5))
                    sr   = 16000
                    n    = int(sr * dur)
                    t    = np.linspace(0.0, dur, n, dtype=np.float32)
                    tone = np.sin(2.0 * math.pi * freq * t) * vol
                    fade = max(1, int(sr * 0.01))
                    if 2 * fade < len(tone):
                        tone[:fade]  *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                        tone[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
                    self._env.play_sound(tone, (0.0, 1.0, 3.0))
                    print(f'[viz] env_play_tone: {freq:.0f}Hz {dur:.1f}s vol={vol:.2f}')
                except Exception as exc:
                    print(f'[viz] env_play_tone error: {exc}')

        elif action == 'env_schedule_event':
            # Schedule a future env action by sim time
            # {action:'env_schedule_event', at_sim_s:float, event:{action:..., ...}}
            try:
                at_s  = float(msg.get('at_sim_s', 0.0))
                event = dict(msg.get('event', {}))
                self._scheduled_sessions.append({
                    'id':          f'evt_{int(time.time())}',
                    'time_s':      at_s,
                    'type':        'env_event',
                    'description': event.get('action', '?'),
                    'event':       event,
                    'played':      False,
                })
                print(f'[viz] env_schedule_event at t={at_s:.0f}s: {event.get("action","?")}')
            except Exception as exc:
                print(f'[viz] env_schedule_event error: {exc}')

        elif action == 'env_set_camera':
            if self._env is not None:
                try:
                    eye    = msg.get('eye',    [8.0, 6.0, -8.0])
                    target = msg.get('target', [0.0, 0.5,  0.0])
                    self._env.set_environment_state({
                        'observer_eye':    tuple(float(v) for v in eye),
                        'observer_target': tuple(float(v) for v in target),
                    })
                except Exception as exc:
                    print(f'[viz] env_set_camera error: {exc}')

        # Forward all commands to the simulation loop via pop_commands()
        self._command_queue.append(msg)

    # ------------------------------------------------------------------
    # asyncio internals
    # ------------------------------------------------------------------

    def _run_thread(self) -> None:
        """Entry point for the background daemon thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        if _WS_OK:
            loop.run_until_complete(self._run_server())
        else:
            # No websockets -- just spin and keep the loop alive so
            # tick() can schedule coroutines against it in the future.
            loop.run_until_complete(self._noop_loop())

    async def _run_server(self) -> None:
        """Coroutine: start the WebSocket server and serve until stopped."""
        try:
            srv = await websockets.serve(
                self._ws_handler,
                '0.0.0.0',
                self._port,
            )
            self._ws_server = srv
            print(f'[viz] WebSocket server listening on port {self._port}.')
            while self._running:
                await asyncio.sleep(0.1)
            srv.close()
            await srv.wait_closed()
        except Exception as exc:
            print(f'[viz] WebSocket server error: {exc}')

    async def _noop_loop(self) -> None:
        """Idle coroutine when websockets is not installed."""
        while self._running:
            await asyncio.sleep(1.0)

    async def _ws_handler(self, websocket, path=None) -> None:
        """Handle a single WebSocket client connection."""
        self._clients.add(websocket)
        print(f'[viz] Client connected ({len(self._clients)} total).')
        try:
            # Send current static neuron layout once on connect
            layout_msg = json.dumps({
                'type':          'neuron_layout',
                'neuron_layout': _NEURON_LAYOUT,
            })
            await websocket.send(layout_msg)

            # Listen for incoming control messages
            async for raw_msg in websocket:
                try:
                    msg = json.loads(raw_msg)
                    self._handle_control_message(msg)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            print(f'[viz] Client disconnected ({len(self._clients)} remaining).')

    async def _broadcast(self, frame_json: str) -> None:
        """Send a JSON frame to all connected clients concurrently."""
        if not self._clients:
            return
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(frame_json)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Helper: map EventType string -> chemical name
# ---------------------------------------------------------------------------

def _event_type_to_chemical(ev_type: str) -> str:
    """
    Best-effort mapping of a NeurochemicalEvent EventType name to a
    chemical string for the event log display.
    """
    ev = ev_type.upper()
    if 'DOMIN' in ev or 'REWARD' in ev or 'CORRECT' in ev:
        return 'dopamine'
    if 'SEROTON' in ev or 'CALM' in ev or 'REST' in ev:
        return 'serotonin'
    if 'CORTISOL' in ev or 'STRESS' in ev or 'THREAT' in ev or 'ABSENT' in ev:
        return 'cortisol'
    if 'OXYTOC' in ev or 'BOND' in ev or 'TOUCH' in ev or 'VOICE' in ev:
        return 'oxytocin'
    if 'NOREPIN' in ev or 'ALERT' in ev or 'STARTLE' in ev or 'NE' in ev:
        return 'norepinephrine'
    if 'ACETYL' in ev or 'ACH' in ev or 'ATTEND' in ev or 'GAZE' in ev:
        return 'acetylcholine'
    return 'dopamine'   # fallback


# ---------------------------------------------------------------------------
# Stand-alone demo / smoke-test
# ---------------------------------------------------------------------------

def run_visualization_demo(n_ticks: int = 50) -> None:
    """
    Headless smoke-test that:
    1. Instantiates VisualizationServer
    2. Creates stub stand-ins for every CAINE module
    3. Runs n_ticks of tick() and verifies all 7 panel builders return
       valid dicts with expected keys
    4. Tests _infer_emotional_state(), emotional nearest-neighbour table
    5. Tests pop_commands() after injecting a synthetic control message
    6. Saves a panel summary to output/visualization_demo.json
    """
    import json as _json

    print()
    print('=' * 60)
    print('  Module 10 -- VisualizationServer smoke-test')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Stub module classes
    # ------------------------------------------------------------------

    rng = np.random.RandomState(0)

    class StubNeuro:
        def snapshot(self):
            return {c: float(rng.uniform(0.1, 0.8)) for c in _NEURO_CHEMICALS}
        dopamine      = 0.45
        serotonin     = 0.55
        cortisol      = 0.12
        oxytocin      = 0.30
        norepinephrine= 0.40
        acetylcholine = 0.60
        event_log = [
            (0.0,    'VOICE_MATCH',   0.3),
            (100.0,  'REWARD',        0.5),
            (250.0,  'CORTISOL_RISE', 0.2),
        ]
        class _chemicals:
            @staticmethod
            def get(name, default=None):
                class _P:
                    concentration = 0.1
                    baseline      = 0.1
                return _P()
        _chemicals = {'cortisol': type('P', (), {'concentration': 0.12, 'baseline': 0.08})()}

    class StubPop:
        def __init__(self, n):
            self.n        = n
            self.rate_est = rng.uniform(0, 100, n).astype('f4')
            self.W_hebb   = rng.uniform(0, 0.5, (n, 36)).astype('f4')
            self.W_in     = rng.uniform(0, 0.3, (n, 16)).astype('f4')
            self.W_recur  = rng.uniform(0, 0.2, (n, n)).astype('f4')
            self.pref_orient = np.linspace(0, 180, n)
            self.pref_freq   = np.logspace(np.log10(80), np.log10(8000), n)
            self.mPFC_rates  = rng.uniform(0, 50, 8).astype('f4')
            self.PCC_rates   = rng.uniform(0, 50, 8).astype('f4')
            self.myelination_factor = 0.25
        def ag_rates(self):
            return rng.uniform(0, 50, 16).astype('f4')
        ag_rates = property(lambda self: rng.uniform(0, 50, 16).astype('f4'))

    class StubDMN:
        dmn_active  = False
        dmn_emerged = False
        corr_log    = [(0.0, 0.12), (100.0, 0.18)]

    class StubHippo:
        n_episodes = 14
        ca3 = StubPop(15)
        ca1 = StubPop(10)

    class StubEnv:
        def get_camera_feed(self):
            return rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        m1_suppressed      = True
        father_voice_onset = False
        pfc_myelination    = False

    class StubSensory:
        _last_dog = rng.uniform(-1, 1, (64, 64)).astype('f4')

    class StubStageManager:
        from caine.environment import DevStage, _STAGE_NAMES
        stage      = DevStage.VOID
        stage_name = _STAGE_NAMES[DevStage.VOID]
        def last_conditions(self):
            return {'v1_orientation': False, 'a1_tonotopy': False,
                    'no_chronic_cort': True, 'min_runtime_48h': False}
        last_conditions = property(lambda self: {
            'v1_orientation': False, 'a1_tonotopy': False,
            'no_chronic_cort': True, 'min_runtime_48h': False})

    class StubAvatar:
        _last_state = {'gaze_target': (0.1, 1.2, 3.0)}

    class StubMotorPop:
        def __init__(self):
            self.rate_est = rng.uniform(0, 80, 60).astype('f4')

    class StubMotor:
        m1_pop      = StubMotorPop()
        joint_angles = np.array([0.1, -0.2, 0.15, -0.05, 0.3, -0.1], dtype='f4')

    class StubParenting:
        class _scheduler:
            @staticmethod
            def add_session(s): pass
        scheduler     = _scheduler()
        _action_queue = []   # empty initially

        def add_session(self, s):
            self._scheduler.add_session(s)

    # ------------------------------------------------------------------
    # Instantiate + attach
    # ------------------------------------------------------------------
    viz = VisualizationServer(port=_PORT_DEFAULT)

    stub_neuro     = StubNeuro()
    stub_v1        = StubPop(20)
    stub_a1        = StubPop(20)
    stub_it        = StubPop(24)
    stub_stg       = StubPop(24)
    stub_pfc       = StubPop(32)
    stub_ag        = StubPop(16)
    stub_dmn       = StubDMN()
    stub_hippo     = StubHippo()
    stub_env       = StubEnv()
    stub_sen       = StubSensory()
    stub_stage     = StubStageManager()
    stub_avatar    = StubAvatar()
    stub_motor     = StubMotor()
    stub_parenting = StubParenting()

    viz.attach(
        neuro         = stub_neuro,
        v1            = stub_v1,
        a1            = stub_a1,
        it            = stub_it,
        stg           = stub_stg,
        pfc           = stub_pfc,
        ag            = stub_ag,
        dmn           = stub_dmn,
        hippocampus   = stub_hippo,
        sensory       = stub_sen,
        env           = stub_env,
        stage_manager = stub_stage,
        avatar        = stub_avatar,
        motor         = stub_motor,
        parenting     = stub_parenting,
    )

    # Do NOT call viz.start() in headless test (no network required)

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------
    dt_ms = 20.0
    for tick in range(n_ticks):
        sim_time_s = tick * dt_ms / 1000.0
        # Vary neuro state slightly each tick
        stub_neuro.dopamine = 0.3 + 0.4 * float(np.sin(sim_time_s * 2))
        viz.tick(sim_time_s=sim_time_s, dt_ms=dt_ms)

    # ------------------------------------------------------------------
    # Verify all panels
    # ------------------------------------------------------------------
    frame = viz._build_frame(sim_time_s=n_ticks * dt_ms / 1000.0,
                             cortex_state={
                                 'dmn_correlation': 0.22,
                                 'pfc_wm_span_ms':  800.0,
                             })

    assert 't' in frame,               "Frame missing 't'"
    assert 'panel1_brain' in frame,    "Frame missing panel1_brain"
    assert 'panel2_vision' in frame,   "Frame missing panel2_vision"
    assert 'panel3_auditory' in frame, "Frame missing panel3_auditory"
    assert 'panel4_neuro' in frame,    "Frame missing panel4_neuro"
    assert 'panel5_dev' in frame,      "Frame missing panel5_dev"
    assert 'panel6_internal' in frame, "Frame missing panel6_internal"
    assert 'panel7_controls' in frame, "Frame missing panel7_controls"

    # -- Gap 1: ~1000 neurons --
    p1 = frame['panel1_brain']
    assert len(p1['neurons']) == len(_NEURON_LAYOUT), \
        f"Expected {len(_NEURON_LAYOUT)} neurons, got {len(p1['neurons'])}"
    real_n = sum(1 for n in p1['neurons'] if not n.get('background', False))
    bg_n   = sum(1 for n in p1['neurons'] if     n.get('background', False))
    assert len(p1['neurons']) >= 900, \
        f"Expected ~1000 neurons, got {len(p1['neurons'])}"
    assert bg_n > 0, "No background interneurons in panel1_brain"
    # -- Gap 2: pruned/sprouting flags present on synapses --
    if p1['synapses']:
        assert 'pruned'    in p1['synapses'][0], "Synapse missing 'pruned' flag"
        assert 'sprouting' in p1['synapses'][0], "Synapse missing 'sprouting' flag"
    print(f"  Panel 1: {len(p1['neurons'])} neurons "
          f"({real_n} real + {bg_n} bg), "
          f"{len(p1['synapses'])} synapses  OK")

    # -- Gap 3 & 4: V1 color columns + 2D ACh map --
    p2 = frame['panel2_vision']
    assert 'camera_feed_b64' in p2 and len(p2['camera_feed_b64']) > 10, \
        "Panel 2 camera_feed_b64 missing/empty"
    assert len(p2['v1_orientations']) == 4,    "Panel 2 v1_orientations wrong length"
    assert len(p2['it_heatmap'])      == 24,   "Panel 2 it_heatmap wrong length"
    assert 'v1_columns' in p2 and len(p2['v1_columns']) == 20, \
        f"Panel 2 v1_columns missing or wrong length: {len(p2.get('v1_columns', []))}"
    assert all('color' in c and c['color'].startswith('#')
               for c in p2['v1_columns']), "Panel 2 v1_columns colors invalid"
    assert 'ach_attention_map' in p2 and len(p2['ach_attention_map']) == 64, \
        "Panel 2 ach_attention_map missing or wrong size"
    print(f"  Panel 2: camera OK, {len(p2['v1_columns'])} V1 columns colored, "
          f"8x8 ACh map, ach={p2['ach_attention']}  OK")

    # -- Gap 5: a1_pref_freq --
    p3 = frame['panel3_auditory']
    assert len(p3['waveform'])      == 512, "Panel 3 waveform wrong length"
    assert len(p3['mel_bands'])     == 128, "Panel 3 mel_bands wrong length"
    assert len(p3['a1_activation']) == 20,  "Panel 3 a1_activation wrong length"
    assert 'a1_pref_freq' in p3 and len(p3['a1_pref_freq']) == 20, \
        "Panel 3 a1_pref_freq missing or wrong length"
    assert p3['a1_pref_freq'][0] > 0, "Panel 3 a1_pref_freq values invalid"
    assert p3['stg_label'] in _STG_LABELS, "Panel 3 stg_label invalid"
    print(f"  Panel 3: stg={p3['stg_label']} conf={p3['stg_confidence']}, "
          f"a1_pref_freq[0]={p3['a1_pref_freq'][0]:.0f}Hz  OK")

    p4 = frame['panel4_neuro']
    assert all(c in p4['current'] for c in _NEURO_CHEMICALS), \
        "Panel 4 current missing chemicals"
    assert len(p4['events']) <= _MAX_EVENT_LOG, "Panel 4 event log overflow"
    print(f"  Panel 4: {len(p4['events'])} events, "
          f"DA={p4['current']['dopamine']:.3f}  OK")

    p5 = frame['panel5_dev']
    assert 'stage' in p5 and 'sim_age_human' in p5, "Panel 5 missing fields"
    assert 'dmn_correlation' in p5 and 'pfc_wm_span_ms' in p5, \
        "Panel 5 missing DMN/WM fields"
    print(f"  Panel 5: stage={p5['stage']}, age={p5['sim_age_human']}, "
          f"vocab={p5['vocabulary_size']}, syns={p5['synapse_count']}  OK")

    # -- Gap 6: motor intention from M1 --
    p6 = frame['panel6_internal']
    assert 'emotional_state' in p6,  "Panel 6 missing emotional_state"
    assert 'wm_contents'     in p6,  "Panel 6 missing wm_contents"
    assert 'motor_intention' in p6 and len(p6['motor_intention']) == 3, \
        "Panel 6 motor_intention missing or wrong length"
    assert 'joint_angles' in p6 and len(p6['joint_angles']) == 6, \
        "Panel 6 joint_angles missing or wrong length"
    assert p6['note'] == 'INFERRED -- not ground truth', "Panel 6 note wrong"
    print(f"  Panel 6: emotion={p6['emotional_state']} "
          f"(conf={p6['emotional_confidence']:.2f}), "
          f"motor={[round(v,2) for v in p6['motor_intention']]}, "
          f"joints={[round(a,2) for a in p6['joint_angles']]}  OK")

    # -- Gaps 7, 8, 9: Panel 7 scheduled sessions + mother override --
    p7 = frame['panel7_controls']
    assert 'is_paused'          in p7, "Panel 7 missing is_paused"
    assert 'time_scale'         in p7, "Panel 7 missing time_scale"
    assert 'scheduled_sessions' in p7, "Panel 7 missing scheduled_sessions"
    assert 'mother_pending'     in p7, "Panel 7 missing mother_pending"
    print(f"  Panel 7: is_paused={p7['is_paused']}, "
          f"time_scale={p7['time_scale']}, "
          f"sessions={len(p7['scheduled_sessions'])}, "
          f"mother_pending={p7['mother_pending']}  OK")

    # ------------------------------------------------------------------
    # Control message tests (all 9 gaps)
    # ------------------------------------------------------------------
    viz._handle_control_message({'action': 'pause'})
    assert viz.is_paused is True, "pause command not applied"
    viz._handle_control_message({'action': 'set_time_scale', 'value': 3.0})
    assert viz.time_scale == 3.0, "set_time_scale not applied"
    viz._handle_control_message({'action': 'resume'})
    assert viz.is_paused is False, "resume command not applied"
    viz._handle_control_message({'action': 'cortisol_flush'})

    # Gap 7: schedule_father_session
    viz._handle_control_message({
        'action': 'schedule_father_session',
        'time_s': 7200.0, 'type': 'voice_exposure',
        'file': 'data/father_1.wav', 'stage_gate': 0,
        'description': 'Father says CAINE hello',
    })
    assert len(viz._scheduled_sessions) == 1, \
        f"Expected 1 scheduled session, got {len(viz._scheduled_sessions)}"
    p7b = viz._build_panel7_controls()
    assert len(p7b['scheduled_sessions']) == 1, "Scheduled session not in panel7"

    # Gap 8: media_upload with stage tag
    viz._handle_control_message({
        'action': 'media_upload', 'filename': 'test_ball.jpg',
        'stage': 1, 'data_b64': base64.b64encode(b'fake_image').decode(),
    })

    # Gap 9: mother_override cancel
    stub_parenting._action_queue = ['pending_action_stub']
    viz._handle_control_message({'action': 'mother_override', 'approve': False})
    assert stub_parenting._action_queue == [], \
        "mother_override cancel should clear action_queue"
    viz._handle_control_message({'action': 'mother_override', 'approve': True})

    cmds = viz.pop_commands()
    assert len(cmds) >= 7, f"Expected >= 7 commands, got {len(cmds)}"
    print(f"  Mission Control: {len(cmds)} commands processed, "
          f"1 session scheduled, media upload OK, mother override OK  OK")

    # ------------------------------------------------------------------
    # Emotional state inference spot-checks
    # ------------------------------------------------------------------
    # Inject a 'joyful' vector
    stub_neuro._override = (0.80, 0.70, 0.08, 0.75, 0.50, 0.55)
    _orig_snap = stub_neuro.snapshot

    def _joy_snap():
        d = {c: v for c, v in zip(_NEURO_CHEMICALS, stub_neuro._override)}
        return d
    stub_neuro.snapshot = _joy_snap
    label, conf = viz._infer_emotional_state()
    assert label == 'joyful', f"Expected 'joyful', got '{label}'"
    print(f"  Emotional inference: 'joyful' vector -> {label} "
          f"(conf={conf:.2f})  OK")
    stub_neuro.snapshot = _orig_snap

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    assert _sim_age_human(0)       == '0m 0s'
    assert _sim_age_human(3600)    == '1h 0m'
    assert '1.0mo' in _sim_age_human(30 * 86400) or 'mo' in _sim_age_human(30 * 86400)
    print(f"  _sim_age_human: 0->{_sim_age_human(0)}, "
          f"3600->{_sim_age_human(3600)}, "
          f"30d->{_sim_age_human(30 * 86400)}  OK")

    # Neuron layout sanity: real neurons match _REGION_N; total ~1000
    real_layout = sum(1 for n in _NEURON_LAYOUT if not n.get('background', False))
    assert real_layout == sum(_REGION_N.values()), \
        f"Real neuron count mismatch: {real_layout} vs {sum(_REGION_N.values())}"
    assert len(_NEURON_LAYOUT) >= 900, \
        f"Total neuron count {len(_NEURON_LAYOUT)} below ~1000 target"
    print(f"  Neuron layout: {len(_NEURON_LAYOUT)} total ({real_layout} real + "
          f"{len(_NEURON_LAYOUT)-real_layout} bg) across "
          f"{len(_REGION_LAYOUT)} regions  OK")

    # ------------------------------------------------------------------
    # Save summary frame to disk
    # ------------------------------------------------------------------
    summary = {
        'module':     'Module 10 -- Visualization Layer',
        'frame_keys': list(frame.keys()),
        'panel5_dev': p5,
        'panel6_internal': p6,
        'panel7_controls': p7,
        'n_neurons':  len(_NEURON_LAYOUT),
        'n_chemicals': len(_NEURO_CHEMICALS),
        'ws_port':    viz.PORT,
        'ws_ok':      _WS_OK,
    }
    out = os.path.join(_OUTPUT_DIR, 'visualization_demo.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to {out}")

    print()
    print('[viz] Module 10 smoke-test PASSED.')
    print(f'      WebSocket server: ws://localhost:{viz.PORT}')
    print(f'      websockets package available: {_WS_OK}')
    print(f'      PIL/Pillow available: {_PIL_OK}')
    print()


if __name__ == '__main__':
    run_visualization_demo()
