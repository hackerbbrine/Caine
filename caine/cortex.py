"""
CAINE - Module 4: Cortical Architecture
=========================================
Two cortical regions built from populations of Hodgkin-Huxley neurons,
connected by a myelinated white matter tract with axonal transmission delay.

Regions:
  V1 (Primary Visual Cortex)
    - Population of HH neurons organized into 4 orientation columns
    - Columns tuned to 0 deg, 45 deg, 90 deg, 135 deg (Hubel & Wiesel 1962)
    - Each neuron's drive = cos^2(preferred - stimulus) -- standard orientation tuning
    - ON/OFF-surround receptive fields approximated as a tuning-curve input
      (full DoG convolution deferred to when real pixel input is available)

  A1 (Primary Auditory Cortex)
    - Population of HH neurons arranged tonotopically
    - Preferred frequencies log-spaced from 200 Hz to 6000 Hz (cochlear scale)
    - Tuning curve: Gaussian in log-frequency space (sigma = 0.5 octaves)
    - Onset detector: extra kick on stimulus onset, then adapts

  White Matter Tract (V1 -> A1)
    - Topographic connection: V1 neuron i drives A1 neuron j with Gaussian
      weight profile centred on matching index positions
    - Axonal conduction delay (default 5 ms)
    - AMPA conductance kinetics (tau = 5 ms, same as Module 2)
    - STDP weight updates gated and scaled by Module 3 neurochemicals

  Neurochemical Modulation (Module 3)
    - NE global_gain() scales the effective input current to both populations
    - ACh learning_gate() gates STDP in the white matter tract
    - DA stdp_scale() scales STDP amplitude
    - Cortisol accelerates white matter health decay

Stimulus scenario (500 ms):
  0   - 50  ms : silence (baseline)
  50  - 200 ms : 45 deg grating -> V1 col-1 fires; WM drives A1 subthreshold
  150 ms       : NOVEL_STIMULUS event -> NE/DA/ACh spike
  200 - 350 ms : 90 deg grating + 2 kHz tone
                 V1 col-2 fires; A1 2kHz neurons fire from audio;
                 WM adds convergent drive (V1 col-2 -> A1 mid-high neurons)
  300 ms       : DIRECTED_GAZE event -> ACh boost -> opens learning gate
  350 - 500 ms : both stimuli off, observe decay

Single file. Builds on Modules 1, 2, 3.
No ML frameworks -- only numpy.
"""

import os
import sys as _sys
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from collections import deque

# All plots go to output/ at the project root
_OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)

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

# --- Population sizes ---
N_V1 = 20        # 5 neurons per orientation column (4 columns)
N_A1 = 20        # tonotopic neurons

# --- V1 orientation columns ---
V1_ORIENTATIONS = [0.0, 45.0, 90.0, 135.0]   # degrees
N_PER_COL       = N_V1 // len(V1_ORIENTATIONS)  # 5

# --- A1 frequency range ---
A1_F_LOW        = 200.0    # Hz
A1_F_HIGH       = 6000.0   # Hz
A1_SIGMA_OCT    = 0.5      # tuning width in octaves

# --- Stimulus drive levels ---
I_BASE          = 3.0      # uA/cm^2 -- constant subthreshold tonic drive
I_STIM_V1       = 8.5      # uA/cm^2 -- peak drive to a perfectly tuned V1 neuron
I_STIM_A1       = 8.5      # uA/cm^2 -- peak drive to a perfectly tuned A1 neuron

# --- White matter tract ---
WM_DELAY_MS     = 5.0      # axonal conduction delay (ms)
WM_INIT_WEIGHT  = 0.18     # initial weight; moderate -- WM alone is subthreshold
WM_CONN_WIDTH   = 0.35     # topographic spread (fraction of population width)
WM_G_PEAK       = 0.5      # AMPA peak conductance per spike (mS/cm^2)
                            # lower than local synapses; long-range tracts are sparser

# --- Spike detection ---
SPIKE_THR       = 0.0      # mV
REFRACTORY_MS   = 2.0

# --- Simulation ---
DT_MS           = 0.05     # 50 us -- accurate enough for HH, fast enough for 40 neurons
DURATION_MS     = 500.0


# ===========================================================================
# SECTION 2 -- V1 POPULATION
# ===========================================================================

class V1Population:
    """
    Primary Visual Cortex -- 20 HH neurons in 4 orientation columns.

    Orientation selectivity is modelled as a cosine-squared tuning curve,
    the standard minimal model of simple-cell selectivity (Hubel & Wiesel).
    All neurons share the same HH biophysics; their identity comes purely
    from which orientation they are tuned to.

    State is stored as numpy arrays for vectorised Euler integration.
    """

    def __init__(self, n_neurons: int = N_V1):
        assert n_neurons % 4 == 0
        self.n        = n_neurons
        n_col         = n_neurons // 4

        # Preferred orientation for each neuron (degrees)
        self.pref_orient = np.array(
            [ang for ang in V1_ORIENTATIONS for _ in range(n_col)],
            dtype=float
        )
        # Column index for each neuron (used for colour coding in plots)
        self.column = np.array(
            [c for c in range(4) for _ in range(n_col)], dtype=int
        )

        # HH state (vectorised)
        m0, h0, n0 = gate_steady_state(V_rest)
        self.V      = np.full(n_neurons, V_rest)
        self.m      = np.full(n_neurons, m0)
        self.h      = np.full(n_neurons, h0)
        self.n_gate = np.full(n_neurons, n0)

        # Spike infrastructure
        self.last_spike  = np.full(n_neurons, -np.inf)
        self.spike_times = [[] for _ in range(n_neurons)]
        self._V_prev     = self.V.copy()

    # ------------------------------------------------------------------
    def compute_drive(self, stim_angle_deg: float,
                      I_max: float, gain_mod: float = 1.0) -> np.ndarray:
        """
        Input current for each neuron given a visual stimulus orientation.

        Tuning: I = I_BASE + I_max * cos^2(pref - stim) * gain_mod
        Neurons whose preferred angle matches the stimulus get I_BASE + I_max.
        Neurons at 90 deg to the stimulus get only I_BASE (no response).
        """
        delta_rad = np.radians(self.pref_orient - stim_angle_deg)
        tuning    = np.cos(delta_rad) ** 2          # [0, 1] per neuron
        return I_BASE + I_max * tuning * gain_mod

    # ------------------------------------------------------------------
    def step(self, dt: float, I_ext: np.ndarray,
             I_syn: np.ndarray | None = None) -> None:
        """
        Vectorised Euler step for all V1 neurons simultaneously.

        I_ext : shape (N,) external drive (from stimulus + neuromodulation)
        I_syn : shape (N,) synaptic inward current (positive = depolarising)
        """
        I_total = I_ext.copy()
        if I_syn is not None:
            I_total += I_syn

        # Ionic currents (outward-positive HH convention)
        I_Na = gNa * self.m**3 * self.h * (self.V - ENa)
        I_K  = gK  * self.n_gate**4     * (self.V - EK)
        I_L  = gL                       * (self.V - EL)

        # Euler integration
        self.V      += dt * (I_total - I_Na - I_K - I_L) / Cm
        self.m      += dt * (alpha_m(self.V) * (1 - self.m)      - beta_m(self.V) * self.m)
        self.h      += dt * (alpha_h(self.V) * (1 - self.h)      - beta_h(self.V) * self.h)
        self.n_gate += dt * (alpha_n(self.V) * (1 - self.n_gate) - beta_n(self.V) * self.n_gate)

    # ------------------------------------------------------------------
    def detect_spikes(self, t: float) -> np.ndarray:
        """
        Vectorised spike detection with refractory period.
        Returns bool array fired[i] = True if neuron i fired this step.
        """
        crossed     = (self._V_prev < SPIKE_THR) & (self.V >= SPIKE_THR)
        not_refrac  = (t - self.last_spike) > REFRACTORY_MS
        fired       = crossed & not_refrac

        self.last_spike[fired] = t
        for i in np.where(fired)[0]:
            self.spike_times[i].append(float(t))

        self._V_prev = self.V.copy()
        return fired


# ===========================================================================
# SECTION 3 -- A1 POPULATION
# ===========================================================================

class A1Population:
    """
    Primary Auditory Cortex -- 20 HH neurons arranged tonotopically.

    Neuron positions map logarithmically to frequency (mimicking the
    basilar membrane and cochlear nucleus). Tuning is Gaussian in
    log-frequency space, consistent with single-unit recordings in A1.

    An onset-transient is added when a new stimulus begins: a brief
    extra-current burst decaying with tau=10ms, mimicking onset-detector
    populations (the README mentions onset/offset detector populations).
    """

    def __init__(self, n_neurons: int = N_A1):
        self.n = n_neurons

        # Preferred frequencies: log-spaced across the speech-relevant range
        self.pref_freq = np.logspace(
            np.log10(A1_F_LOW), np.log10(A1_F_HIGH), n_neurons
        )

        # HH state
        m0, h0, n0 = gate_steady_state(V_rest)
        self.V      = np.full(n_neurons, V_rest)
        self.m      = np.full(n_neurons, m0)
        self.h      = np.full(n_neurons, h0)
        self.n_gate = np.full(n_neurons, n0)

        # Spike infrastructure
        self.last_spike  = np.full(n_neurons, -np.inf)
        self.spike_times = [[] for _ in range(n_neurons)]
        self._V_prev     = self.V.copy()

        # Onset transient state (per-neuron decaying extra current)
        self._onset_current = np.zeros(n_neurons)
        self._onset_tau_ms  = 10.0     # onset decay time constant
        self._prev_stim_hz  = 0.0      # track when stimulus changes

    # ------------------------------------------------------------------
    def compute_drive(self, stim_freq_hz: float,
                      I_max: float, gain_mod: float = 1.0) -> np.ndarray:
        """
        Input current for each A1 neuron given an auditory stimulus frequency.

        Tuning: Gaussian in log-frequency space
          response = exp( -(log2(f_pref/f_stim))^2 / (2*sigma^2) )

        onset_transient: if stimulus frequency just changed, adds a brief
        extra depolarising current to the responding neurons.
        """
        if stim_freq_hz <= 0:
            # Silent: no stimulus, onset current still decays
            return np.full(self.n, I_BASE) + self._onset_current

        octave_dist = np.log2(self.pref_freq / stim_freq_hz)
        tuning      = np.exp(-octave_dist**2 / (2 * A1_SIGMA_OCT**2))

        # Check for stimulus onset (frequency changed from 0 or different value)
        if abs(stim_freq_hz - self._prev_stim_hz) > 1.0:
            # Inject onset transient proportional to tuning
            self._onset_current = 4.0 * tuning * gain_mod
            self._prev_stim_hz  = stim_freq_hz

        return I_BASE + I_max * tuning * gain_mod + self._onset_current

    # ------------------------------------------------------------------
    def decay_onset(self, dt: float) -> None:
        """Exponential decay of onset transient current (called every step)."""
        self._onset_current *= np.exp(-dt / self._onset_tau_ms)

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

        self.V      += dt * (I_total - I_Na - I_K - I_L) / Cm
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


# ===========================================================================
# SECTION 4 -- WHITE MATTER TRACT
# ===========================================================================

class WhiteMatterTract:
    """
    Long-range axonal projection from one cortical region to another.

    Models:
      - Topographic Gaussian connectivity (nearby neurons connect more strongly)
      - AMPA conductance kinetics with exponential decay (tau = 5 ms)
      - Axonal conduction delay implemented via a deque-based spike buffer
        (same approach as Module 2's Synapse)
      - Nearest-neighbour STDP, amplitude gated and scaled by Module 3
        neurochemical outputs (ACh learning gate, DA+ACh stdp_scale)
      - Mean weight tracking for plotting

    Convention: presynaptic region is V1, postsynaptic is A1.
    """

    def __init__(self, n_pre: int, n_post: int,
                 delay_ms: float  = WM_DELAY_MS,
                 init_weight: float = WM_INIT_WEIGHT,
                 conn_width: float  = WM_CONN_WIDTH):

        self.n_pre   = n_pre
        self.n_post  = n_post
        self.delay_ms = delay_ms

        # --- Topographic weight matrix: shape (n_pre, n_post) ---
        # Neuron positions normalised to [0, 1]; Gaussian profile in position space.
        pre_pos  = np.linspace(0, 1, n_pre)
        post_pos = np.linspace(0, 1, n_post)
        dist = np.abs(pre_pos[:, None] - post_pos[None, :])   # (n_pre, n_post)
        self.weights = init_weight * np.exp(-dist**2 / (2 * conn_width**2))

        # --- AMPA conductances: (n_pre, n_post) ---
        self.g_ampa = np.zeros((n_pre, n_post))

        # --- Axonal spike delay buffer: (arrival_time_ms, pre_neuron_index) ---
        self.delay_buffer: deque = deque()

        # --- STDP spike timing ---
        self.last_pre_spike  = np.full(n_pre,  -np.inf)
        self.last_post_spike = np.full(n_post, -np.inf)

        # --- Neurochemical modulation state (written by caller each step) ---
        self.neuro_stdp_scale = 1.0   # from Module 3: DA + ACh + OT
        self.neuro_gate       = 1.0   # from Module 3: ACh learning gate

        # --- Logging ---
        self.mean_weight_log: list = []

    # ------------------------------------------------------------------
    def on_pre_spikes(self, t: float, fired: np.ndarray) -> None:
        """
        Queue delayed spike arrivals for each V1 neuron that fired.
        Also record timing for STDP depression check at delivery time.
        """
        for i in np.where(fired)[0]:
            self.delay_buffer.append((t + self.delay_ms, int(i)))
            self.last_pre_spike[int(i)] = t

    # ------------------------------------------------------------------
    def deliver_and_stdp_depress(self, t: float) -> None:
        """
        Deliver pending spikes whose axonal delay has elapsed.

        For each delivered spike from pre-neuron i:
          1. Activate AMPA conductance: g_ampa[i,:] += weights[i,:] * G_PEAK
          2. STDP depression check: if A1 fired recently (post before pre),
             reduce weights[i,:] -- neuromodulated by ACh gate and DA scale.
        """
        while self.delay_buffer and self.delay_buffer[0][0] <= t:
            _, pre_i = self.delay_buffer.popleft()

            # Activate conductance
            self.g_ampa[pre_i, :] += self.weights[pre_i, :] * WM_G_PEAK

            # STDP: acausal (post before pre) -> depression
            dt_stdp = t - self.last_post_spike        # positive when post was earlier
            valid   = (dt_stdp > 0) & (dt_stdp < 300)
            dw = np.where(
                valid,
                -STDP_A_MINUS * np.exp(-dt_stdp / STDP_TAU_MINUS)
                * self.neuro_stdp_scale * self.neuro_gate,
                0.0
            )
            self.weights[pre_i, :] = np.clip(self.weights[pre_i, :] + dw, 0.0, 1.0)

    # ------------------------------------------------------------------
    def on_post_spikes(self, t: float, fired: np.ndarray) -> None:
        """
        Handle A1 spike events.
        STDP: causal (pre before post) -> potentiation.
        """
        for j in np.where(fired)[0]:
            self.last_post_spike[int(j)] = t

            dt_stdp = t - self.last_pre_spike         # positive when pre was earlier
            valid   = (dt_stdp > 0) & (dt_stdp < 300)
            dw = np.where(
                valid,
                STDP_A_PLUS * np.exp(-dt_stdp / STDP_TAU_PLUS)
                * self.neuro_stdp_scale * self.neuro_gate,
                0.0
            )
            self.weights[:, int(j)] = np.clip(self.weights[:, int(j)] + dw, 0.0, 1.0)

    # ------------------------------------------------------------------
    def get_synaptic_currents(self, V_post: np.ndarray) -> np.ndarray:
        """
        Compute inward synaptic current for each A1 neuron.

        I_syn[j] = -sum_i(g_ampa[i,j]) * (V_post[j] - E_rev)
        Negative of the outward current convention -> positive = depolarising.
        Added directly to I_total in the A1 step.
        """
        g_total       = self.g_ampa.sum(axis=0)           # shape (n_post,)
        I_syn_outward = g_total * (V_post - AMPA_E_REV)   # outward-positive
        return -I_syn_outward                              # inward, positive = depolarising

    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        """
        Exponential decay of all AMPA conductances.
        Log mean weight for plotting.
        """
        self.g_ampa *= np.exp(-dt / AMPA_TAU)
        self.mean_weight_log.append(float(self.weights.mean()))


# ===========================================================================
# SECTION 5 -- STIMULUS SCHEDULE
# ===========================================================================

def _build_stimulus_schedule(duration_ms: float, dt_ms: float) -> tuple:
    """
    Returns (stim_v1, stim_a1, neuro_events) where:

    stim_v1[i]    = (angle_deg, I_max)  -- visual stimulus at timestep i
                    angle_deg=-1 means no visual input
    stim_a1[i]    = (freq_hz, I_max)    -- audio stimulus at timestep i
                    freq_hz=0 means silence
    neuro_events  = dict{timestep_index: [NeurochemicalEvent, ...]}
    """
    N = int(duration_ms / dt_ms) + 1

    stim_v1  = [(-1.0, 0.0)] * N    # default: no visual stimulus
    stim_a1  = [(0.0,  0.0)] * N    # default: silence

    def _ms_to_idx(t_ms):
        return int(round(t_ms / dt_ms))

    # --- Phase 1: 50-200 ms: 45-degree grating (V1 only) ---
    for i in range(_ms_to_idx(50), _ms_to_idx(200)):
        stim_v1[i] = (45.0, I_STIM_V1)

    # --- Phase 2: 200-350 ms: 90-degree grating + 2 kHz tone ---
    for i in range(_ms_to_idx(200), _ms_to_idx(350)):
        stim_v1[i] = (90.0, I_STIM_V1)
        stim_a1[i] = (2000.0, I_STIM_A1)

    # --- Phase 3: 350-500 ms: silence (no stimulus) ---
    # (already default)

    # Neurochemical events
    neuro_events = {}

    def _add_event(t_ms, event_type, magnitude=1.0):
        idx = _ms_to_idx(t_ms)
        neuro_events.setdefault(idx, []).append(
            NeurochemicalEvent(event_type, magnitude)
        )

    # Novel stimulus when first visual input arrives
    _add_event(50.0,  EventType.NOVEL_STIMULUS,    1.0)
    _add_event(50.0,  EventType.NOVEL_ENVIRONMENT, 0.7)
    # Stronger NE/ACh burst when audio+visual together (multimodal novel event)
    _add_event(200.0, EventType.NOVEL_STIMULUS,    0.8)
    _add_event(200.0, EventType.STARTLE,           0.5)   # sudden multimodal onset
    # Directed gaze / attention event -- opens ACh learning gate fully
    _add_event(300.0, EventType.DIRECTED_GAZE,     1.0)
    # Reward after sustained multimodal activity
    _add_event(320.0, EventType.REWARD,            0.7)

    return stim_v1, stim_a1, neuro_events


# ===========================================================================
# SECTION 6 -- MAIN SIMULATION
# ===========================================================================

def run_cortical_simulation(
    duration_ms: float = DURATION_MS,
    dt_ms:       float = DT_MS,
) -> dict:
    """
    Simulate V1 + A1 populations with white matter connectivity and
    neurochemical modulation.

    Returns a results dict with all traces needed for plotting.
    """

    N = int(duration_ms / dt_ms) + 1
    t_array = np.linspace(0.0, duration_ms, N)

    # --- Instantiate populations ---
    v1  = V1Population(N_V1)
    a1  = A1Population(N_A1)
    wm  = WhiteMatterTract(N_V1, N_A1)
    neuro = NeurochemicalSystem()

    # --- Build stimulus ---
    stim_v1, stim_a1, neuro_sched = _build_stimulus_schedule(duration_ms, dt_ms)

    # --- Storage for traces ---
    # Voltage traces: pick 2 representative neurons per region for plotting
    # V1: one neuron from col-1 (45 deg, index 5) and col-2 (90 deg, index 10)
    # A1: one low-freq (index 2) and one mid-freq near 2kHz (index 13)
    v1_trace_idx  = [5, 10]
    a1_trace_idx  = [2, 13]
    V_v1_sample   = {i: np.empty(N) for i in v1_trace_idx}
    V_a1_sample   = {i: np.empty(N) for i in a1_trace_idx}

    # Population firing rate traces (smoothed later)
    v1_pop_fired  = np.zeros(N, dtype=int)   # total spikes across all V1 neurons
    a1_pop_fired  = np.zeros(N, dtype=int)

    # Neurochemical traces
    chem_names    = ['dopamine', 'serotonin', 'norepinephrine', 'acetylcholine', 'cortisol']
    chem_traces   = {name: np.empty(N) for name in chem_names}
    modulation    = {'stdp_scale': np.empty(N), 'learning_gate': np.empty(N),
                     'global_gain': np.empty(N)}

    # White matter mean weight
    wm_weight_trace = np.empty(N)

    # --- Console header ---
    print("=" * 70)
    print("CAINE - Module 4: Cortical Architecture")
    print("=" * 70)
    print(f"  V1 neurons  : {N_V1}  ({N_PER_COL} per orientation column)")
    print(f"  A1 neurons  : {N_A1}  (tonotopic {A1_F_LOW:.0f}-{A1_F_HIGH:.0f} Hz)")
    print(f"  WM delay    : {WM_DELAY_MS} ms  |  WM init weight: {WM_INIT_WEIGHT}")
    print(f"  Duration    : {duration_ms} ms  |  dt: {dt_ms} ms  |  steps: {N}")
    print("-" * 70)

    # -----------------------------------------------------------------------
    # 6.3  Main integration loop
    # -----------------------------------------------------------------------
    for i, t in enumerate(t_array):

        # --- Neurochemical events this step ---
        events_now = neuro_sched.get(i)
        if events_now:
            for ev in events_now:
                print(f"[EVENT] t={t:7.1f} ms  {ev.event_type.name:<28} "
                      f"mag={ev.magnitude:.2f}")
        neuro.update(dt_ms, events=events_now, current_time=t)

        # --- Read modulation values ---
        gain_mod  = neuro.global_gain()     # NE-driven excitability boost
        stdp_s    = neuro.stdp_scale()      # DA + ACh + OT combined
        ach_gate  = neuro.learning_gate()   # ACh threshold gate

        # Push modulation into white matter tract
        wm.neuro_stdp_scale = stdp_s
        wm.neuro_gate       = ach_gate

        # --- Visual stimulus ---
        stim_angle, I_v1_max = stim_v1[i]
        if stim_angle >= 0:
            I_v1 = v1.compute_drive(stim_angle, I_v1_max, gain_mod=gain_mod)
        else:
            I_v1 = np.full(N_V1, I_BASE * gain_mod)

        # --- Auditory stimulus ---
        stim_freq, I_a1_max = stim_a1[i]
        I_a1 = a1.compute_drive(stim_freq, I_a1_max, gain_mod=gain_mod)
        a1.decay_onset(dt_ms)

        # --- White matter synaptic current into A1 ---
        I_wm = wm.get_synaptic_currents(a1.V)

        # --- Advance neurons ---
        v1.step(dt_ms, I_v1)
        a1.step(dt_ms, I_a1, I_syn=I_wm)

        # --- Spike detection ---
        v1_fired = v1.detect_spikes(t)
        a1_fired = a1.detect_spikes(t)

        # --- White matter: queue pre-spikes, deliver arrivals, handle post-spikes ---
        wm.on_pre_spikes(t, v1_fired)
        wm.deliver_and_stdp_depress(t)
        wm.on_post_spikes(t, a1_fired)
        wm.update(dt_ms)

        # --- Record ---
        for ni in v1_trace_idx:
            V_v1_sample[ni][i] = v1.V[ni]
        for ni in a1_trace_idx:
            V_a1_sample[ni][i] = a1.V[ni]

        v1_pop_fired[i] = int(v1_fired.sum())
        a1_pop_fired[i] = int(a1_fired.sum())

        snap = neuro.snapshot()
        for name in chem_names:
            chem_traces[name][i] = snap[name]
        modulation['stdp_scale'][i]   = stdp_s
        modulation['learning_gate'][i] = ach_gate
        modulation['global_gain'][i]   = gain_mod

        wm_weight_trace[i] = wm.weights.mean()

    # --- Summary ---
    v1_total = sum(len(st) for st in v1.spike_times)
    a1_total = sum(len(st) for st in a1.spike_times)
    print("-" * 70)
    print(f"[CAINE] V1 total spikes : {v1_total}")
    print(f"[CAINE] A1 total spikes : {a1_total}")
    print(f"[CAINE] WM mean weight  : init={WM_INIT_WEIGHT:.4f}  "
          f"final={wm.weights.mean():.4f}  "
          f"delta={wm.weights.mean() - WM_INIT_WEIGHT:+.4f}")
    print(f"[CAINE] Peak NE gain    : {modulation['global_gain'].max():.3f}")
    print(f"[CAINE] Peak ACh gate   : {modulation['learning_gate'].max():.3f}")
    print("=" * 70)

    return {
        't':              t_array,
        'v1':             v1,
        'a1':             a1,
        'wm':             wm,
        'V_v1_sample':    V_v1_sample,
        'V_a1_sample':    V_a1_sample,
        'v1_pop_fired':   v1_pop_fired,
        'a1_pop_fired':   a1_pop_fired,
        'chem_traces':    chem_traces,
        'modulation':     modulation,
        'wm_weight_trace': wm_weight_trace,
        'stim_v1':        stim_v1,
        'stim_a1':        stim_a1,
    }


# ===========================================================================
# SECTION 7 -- HELPERS
# ===========================================================================

def _smooth_rate(fired_array: np.ndarray, t_array: np.ndarray,
                 win_ms: float = 10.0, dt_ms: float = DT_MS) -> np.ndarray:
    """
    Convert per-step spike counts into a smoothed population firing rate (Hz).
    Uses a rectangular sliding window of width win_ms.
    Rate = (spikes in window) / (window_duration * n_neurons)
    """
    win_steps = max(1, int(win_ms / dt_ms))
    kernel    = np.ones(win_steps) / (win_ms * 1e-3)  # normalise to Hz per neuron
    return np.convolve(fired_array, kernel, mode='same')


def _stimulus_shade(ax, stim_list, t_array, color, alpha=0.08):
    """Shade timesteps where a stimulus is active."""
    active = np.array([s[0] >= 0 if len(s) == 2 and isinstance(s[0], float)
                       else s[0] > 0
                       for s in stim_list], dtype=bool)
    # Find contiguous ON blocks and fill them
    in_block = False
    t0 = 0.0
    for i, a in enumerate(active):
        if a and not in_block:
            t0 = t_array[i]
            in_block = True
        elif not a and in_block:
            ax.axvspan(t0, t_array[i], color=color, alpha=alpha, linewidth=0)
            in_block = False
    if in_block:
        ax.axvspan(t0, t_array[-1], color=color, alpha=alpha, linewidth=0)


# ===========================================================================
# SECTION 8 -- PLOTTING
# ===========================================================================

def plot_cortex(results: dict) -> None:
    """
    Two output figures:

    Figure 1 -- Activity overview (6 panels):
      Row 1: V1 raster (neurons coloured by orientation column)
             + shaded stimulus periods
      Row 2: A1 raster (neurons coloured by preferred frequency)
      Row 3: Sample voltage traces (V1 col-1 neuron vs V1 col-2 neuron)
      Row 4: Sample voltage traces (A1 low-freq vs A1 2-kHz neuron)
      Row 5: Population firing rate (V1 blue, A1 red, smoothed over 10ms)
      Row 6: White matter mean weight over time

    Figure 2 -- Neurochemical state (4 panels):
      Dopamine | Norepinephrine | Acetylcholine | Learning gate + STDP scale
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

    col_colors  = ['#4477AA', '#EE6677', '#228833', '#CCBB44']  # 4 orientation column colours
    freq_cmap   = plt.cm.plasma

    # -----------------------------------------------------------------------
    # Figure 1: Activity
    # -----------------------------------------------------------------------
    fig1 = plt.figure(figsize=(15, 13))
    fig1.suptitle(
        "CAINE - Module 4: Cortical Architecture\n"
        f"V1 ({N_V1} neurons, 4 orientation cols)  +  "
        f"A1 ({N_A1} neurons, tonotopic {A1_F_LOW:.0f}-{A1_F_HIGH:.0f} Hz)  "
        f"via {WM_DELAY_MS:.0f}ms white-matter tract",
        fontsize=11, fontweight='bold'
    )

    gs = gridspec.GridSpec(6, 1, hspace=0.55, top=0.91, bottom=0.06)
    axes = [fig1.add_subplot(gs[r]) for r in range(6)]

    # -- V1 raster --
    ax = axes[0]
    for ni in range(N_V1):
        col = col_colors[v1.column[ni]]
        for st in v1.spike_times[ni]:
            ax.plot(st, ni, '|', color=col, markersize=5, markeredgewidth=1.0)
    _stimulus_shade(ax, stim_v1, t, 'cornflowerblue', alpha=0.10)
    # Legend: one marker per column
    for c_idx, (ang, col) in enumerate(zip(V1_ORIENTATIONS, col_colors)):
        ax.plot([], [], '|', color=col, markersize=8,
                label=f'{ang:.0f} deg  (neurons {c_idx*N_PER_COL}-{(c_idx+1)*N_PER_COL-1})')
    ax.set_yticks([0, 4, 9, 14, 19])
    ax.set_ylim(-0.5, N_V1 - 0.5)
    ax.set_ylabel('Neuron')
    ax.set_title('V1 Raster  (shading = visual stimulus period)', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.18)

    # -- A1 raster --
    ax = axes[1]
    norm = mcolors.LogNorm(vmin=A1_F_LOW, vmax=A1_F_HIGH)
    for ni in range(N_A1):
        col = freq_cmap(norm(a1.pref_freq[ni]))
        for st in a1.spike_times[ni]:
            ax.plot(st, ni, '|', color=col, markersize=5, markeredgewidth=1.0)
    _stimulus_shade(ax, stim_a1, t, 'salmon', alpha=0.12)
    # Frequency axis label on right
    ax2r = ax.twinx()
    ax2r.set_ylim(-0.5, N_A1 - 0.5)
    freq_ticks = [0, 5, 10, 15, 19]
    ax2r.set_yticks(freq_ticks)
    ax2r.set_yticklabels([f'{a1.pref_freq[i]:.0f}Hz' for i in freq_ticks], fontsize=7)
    ax2r.set_ylabel('Preferred frequency', fontsize=8)
    ax.set_ylim(-0.5, N_A1 - 0.5)
    ax.set_ylabel('Neuron')
    ax.set_title('A1 Raster  (shading = auditory stimulus period, colour = preferred freq)', fontsize=9)
    ax.grid(True, alpha=0.18)

    # -- Sample V1 voltage traces --
    ax = axes[2]
    label_map = {5: 'V1 neuron 5 (45 deg col)', 10: 'V1 neuron 10 (90 deg col)'}
    colors_v1 = ['#4477AA', '#228833']
    for k, (ni, col) in enumerate(zip([5, 10], colors_v1)):
        ax.plot(t, V_v1_sample[ni], color=col, linewidth=0.6,
                label=label_map[ni], alpha=0.9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylim(-90, 65)
    ax.set_ylabel('V (mV)')
    ax.set_title('Sample V1 voltage traces', fontsize=9)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.18)

    # -- Sample A1 voltage traces --
    ax = axes[3]
    label_map_a1 = {2: f'A1 neuron 2 ({a1.pref_freq[2]:.0f} Hz)',
                    13: f'A1 neuron 13 ({a1.pref_freq[13]:.0f} Hz, near 2kHz)'}
    colors_a1 = ['#9933CC', '#CC4400']
    for k, (ni, col) in enumerate(zip([2, 13], colors_a1)):
        ax.plot(t, V_a1_sample[ni], color=col, linewidth=0.6,
                label=label_map_a1[ni], alpha=0.9)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylim(-90, 65)
    ax.set_ylabel('V (mV)')
    ax.set_title('Sample A1 voltage traces', fontsize=9)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.18)

    # -- Population firing rate --
    ax = axes[4]
    rate_v1 = _smooth_rate(results['v1_pop_fired'], t, win_ms=10.0)
    rate_a1 = _smooth_rate(results['a1_pop_fired'], t, win_ms=10.0)
    ax.plot(t, rate_v1, color='steelblue', linewidth=1.1, label='V1 population rate')
    ax.plot(t, rate_a1, color='tomato',    linewidth=1.1, label='A1 population rate')
    ax.set_ylabel('Spikes/s')
    ax.set_title('Population firing rate  (smoothed 10 ms window)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.18)

    # -- White matter mean weight --
    ax = axes[5]
    ax.plot(t, wm_w, color='darkorchid', linewidth=1.1, label='WM mean weight')
    ax.axhline(WM_INIT_WEIGHT, color='darkorchid', linewidth=0.7,
               linestyle=':', alpha=0.5, label=f'Initial ({WM_INIT_WEIGHT:.3f})')
    ax.set_ylim(max(0, wm_w.min() - 0.005), wm_w.max() + 0.005)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Time (ms)')
    ax.set_title('White matter mean synaptic weight  (STDP-modulated by neurochemicals)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.18)

    # Shared x limits
    for ax in axes:
        ax.set_xlim(0, DURATION_MS)

    plt.savefig(os.path.join(_OUTPUT_DIR, 'caine_module4_cortex_activity.png'), dpi=150, bbox_inches='tight')
    print('[CAINE] Plot saved -> output/caine_module4_cortex_activity.png')

    # -----------------------------------------------------------------------
    # Figure 2: Neurochemicals + modulation
    # -----------------------------------------------------------------------
    fig2, ax2s = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    fig2.suptitle(
        "CAINE - Module 4: Neurochemical Modulation During Cortical Activity",
        fontsize=11, fontweight='bold'
    )

    neuro_layout = [
        ('dopamine',       'Dopamine (DA)',        'royalblue'),
        ('norepinephrine', 'Norepinephrine (NE)',  'darkorange'),
        ('acetylcholine',  'Acetylcholine (ACh)',  'steelblue'),
        ('cortisol',       'Cortisol (CORT)',      'tomato'),
    ]
    for ax, (name, label, color) in zip(ax2s[:4], neuro_layout):
        bl = chem[name][0]
        ax.plot(t, chem[name], color=color, linewidth=1.0, label=label)
        ax.axhline(bl, color=color, linewidth=0.6, linestyle=':', alpha=0.5,
                   label=f'baseline ({bl:.2f})')
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel('Concentration')
        ax.set_title(label, fontsize=9)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.20)
        # Mark event times
        for ev_idx in results.get('neuro_sched_times', []):
            ax.axvline(ev_idx, color='gray', linewidth=0.5, alpha=0.3)

    # Overlay STDP scale and learning gate on ACh panel (panel 2, index 2)
    ax_twin = ax2s[2].twinx()
    ax_twin.plot(t, mod['learning_gate'], color='navy', linewidth=0.8,
                 linestyle='--', alpha=0.7, label='Learning gate')
    ax_twin.plot(t, mod['stdp_scale'],    color='purple', linewidth=0.8,
                 linestyle='-.', alpha=0.7, label='STDP scale')
    ax_twin.set_ylim(-0.05, 3.1)
    ax_twin.set_ylabel('Gate / Scale', fontsize=8)
    ax_twin.legend(loc='upper left', fontsize=7)

    ax2s[-1].set_xlabel('Time (ms)')
    for ax in ax2s:
        ax.set_xlim(0, DURATION_MS)

    plt.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, 'caine_module4_neurochemicals.png'), dpi=150, bbox_inches='tight')
    print('[CAINE] Plot saved -> output/caine_module4_neurochemicals.png')

    plt.show()


# ===========================================================================
# SECTION 9 -- ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    results = run_cortical_simulation(
        duration_ms = DURATION_MS,
        dt_ms       = DT_MS,
    )

    # Pass neuro schedule times for event-line overlay in plot
    # (build from stim schedule timestamps)
    event_times_ms = [50.0, 200.0, 300.0, 320.0]
    results['neuro_sched_times'] = event_times_ms

    plot_cortex(results)
