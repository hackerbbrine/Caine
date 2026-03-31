"""
CAINE — Module 1: The Neuron
=============================
Every neuron in CAINE's brain is a living differential equation.

The Hodgkin-Huxley model (1952) describes the electrical behaviour of a
biological neuron's membrane with four coupled ODEs:

    Cm * dV/dt = -gNa*m³*h*(V-ENa) - gK*n⁴*(V-EK) - gL*(V-EL) + I_ext

    dm/dt = αm(V)(1-m) - βm(V)m
    dh/dt = αh(V)(1-h) - βh(V)h
    dn/dt = αn(V)(1-n) - βn(V)n

Where:
    V          — membrane voltage (mV)
    m, h       — sodium channel activation / inactivation gates
    n          — potassium channel activation gate
    gNa,gK,gL  — max conductances for sodium, potassium, leak (mS/cm²)
    ENa,EK,EL  — reversal potentials (mV)
    Cm         — membrane capacitance (µF/cm²)
    I_ext      — external input current (µA/cm²)

These are solved numerically using scipy.integrate.odeint with a timestep of
0.01 ms.  For real-time population stepping, HHNeuron.step() uses Euler
integration at the same timestep.

Each neuron also maintains:
    Refractory period    — absolute silence for ~2 ms,
                           relative silence for ~5 ms
    Threshold dynamics   — firing threshold shifts based on recent activity
    Calcium concentration — tracked separately, influences long-term
                            plasticity via the STDP module

Neuron types:
    EXCITATORY   — pyramidal-type, increases downstream firing probability
    INHIBITORY   — interneuron-type, decreases downstream firing probability
    MODULATORY   — no binary spikes, releases neuromodulators continuously

Neurogenesis:
    CAINE's neuron count is not fixed at birth.  NeurogenesisManager tracks
    stage-based targets:
        Stage 0  Birth        ~1,000
        Stage 1  Infancy      ~10,000
        Stage 2  Childhood    ~100,000
        Stage 3  Adolescence  ~500,000
        Stage 4  Maturity     hardware ceiling
    Neuron count is capped by real-time interactive performance requirements.
"""

import os
import enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# SECTION 1 — Membrane & Channel Parameters
# Standard Hodgkin-Huxley (1952) values.  Units: mS/cm², mV, µF/cm².
# ===========================================================================

Cm  = 1.0     # membrane capacitance           (µF/cm²)
gNa = 120.0   # max Na⁺ conductance            (mS/cm²)
gK  = 36.0    # max K⁺  conductance            (mS/cm²)
gL  = 0.3     # leak conductance               (mS/cm²)
ENa = 50.0    # Na⁺ reversal potential         (mV)
EK  = -77.0   # K⁺  reversal potential         (mV)
EL  = -54.4   # leak reversal potential        (mV)
V_rest = -65.0  # resting membrane potential   (mV)


# ===========================================================================
# SECTION 2 — Rate Functions (α/β) for Each Gating Variable
# Empirical fits from Hodgkin & Huxley's voltage-clamp experiments.
# Voltages in mV; HH convention: all V relative to V_rest.
# ===========================================================================

def alpha_m(V):
    """Na⁺ activation — fast opening rate."""
    dV = V - V_rest
    return 0.1 * (25.0 - dV) / (np.exp((25.0 - dV) / 10.0) - 1.0 + 1e-12)

def beta_m(V):
    """Na⁺ activation — closing rate."""
    dV = V - V_rest
    return 4.0 * np.exp(-dV / 18.0)

def alpha_h(V):
    """Na⁺ inactivation — slow onset rate."""
    dV = V - V_rest
    return 0.07 * np.exp(-dV / 20.0)

def beta_h(V):
    """Na⁺ inactivation — recovery rate."""
    dV = V - V_rest
    return 1.0 / (np.exp((30.0 - dV) / 10.0) + 1.0)

def alpha_n(V):
    """K⁺ activation — slow opening rate."""
    dV = V - V_rest
    return 0.01 * (10.0 - dV) / (np.exp((10.0 - dV) / 10.0) - 1.0 + 1e-12)

def beta_n(V):
    """K⁺ activation — closing rate."""
    dV = V - V_rest
    return 0.125 * np.exp(-dV / 80.0)


def gate_steady_state(V) -> Tuple[float, float, float]:
    """Return (m∞, h∞, n∞) — equilibrium gate probabilities at voltage V."""
    am, bm = alpha_m(V), beta_m(V)
    ah, bh = alpha_h(V), beta_h(V)
    an, bn = alpha_n(V), beta_n(V)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


# ===========================================================================
# SECTION 3 — The Four Coupled ODEs  (used by scipy.integrate.odeint)
# ===========================================================================

def hh_odes(state, t, I_ext_func):
    """
    Hodgkin-Huxley ODE system for scipy.integrate.odeint.

    state       : [V, m, h, n]
    t           : current time (ms)
    I_ext_func  : callable(t) → injected current (µA/cm²)

    Returns d/dt [V, m, h, n]
    """
    V, m, h, n = state
    I_Na = gNa * m**3 * h * (V - ENa)   # fast inward Na⁺  (depolarising)
    I_K  = gK  * n**4     * (V - EK)    # delayed outward K⁺ (repolarising)
    I_L  = gL             * (V - EL)    # passive leak
    dVdt = (I_ext_func(t) - I_Na - I_K - I_L) / Cm
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    return [dVdt, dmdt, dhdt, dndt]


# ===========================================================================
# SECTION 4 — Neuron Types
# ===========================================================================

class NeuronType(enum.Enum):
    """
    The three functional neuron types in CAINE's brain.

    EXCITATORY  (pyramidal-type)
        Fires discrete action potentials.  Synaptic output is depolarising —
        increases downstream firing probability.

    INHIBITORY  (interneuron-type)
        Fires discrete action potentials.  Synaptic output is hyperpolarising —
        decreases downstream firing probability.

    MODULATORY
        Does not fire binary spikes.  Releases neuromodulators continuously as
        a graded function of membrane depolarisation above rest.
        Used for dopamine, serotonin, norepinephrine, acetylcholine projection
        neurons in CAINE's neurochemical system.
    """
    EXCITATORY = 'excitatory'
    INHIBITORY = 'inhibitory'
    MODULATORY = 'modulatory'


# ===========================================================================
# SECTION 5 — Per-Neuron Constants (refractory, threshold, calcium)
# ===========================================================================

_ABS_REFRACTORY_MS    = 2.0    # absolute refractory period (ms)
_REL_REFRACTORY_MS    = 5.0    # end of relative refractory window (ms)

_THETA_SPIKE_DELTA_MV = 4.0    # threshold jump per spike (mV)
_THETA_DECAY_TAU_MS   = 100.0  # adaptive threshold time-constant (ms)

_CA_SPIKE_DELTA       = 0.10   # [Ca²⁺] increment per spike (normalised 0–1)
_CA_DECAY_TAU_MS      = 200.0  # calcium decay time-constant (ms)


# ===========================================================================
# SECTION 6 — HHNeuron: Single Biologically-Realistic Neuron
# ===========================================================================

class HHNeuron:
    """
    Single Hodgkin-Huxley neuron with all biological properties.

    Core state: V, m, h, n — solved via Euler at dt=0.01 ms.
    Additional state:

    Refractory period
        After firing, absolute silence for _ABS_REFRACTORY_MS (~2 ms):
        incoming current is clamped to zero.  Relative suppression for the
        next ~3 ms: current scales linearly from 0 → full as the membrane
        recovers.  Matches the biological absolute/relative refractory cycle.

    Threshold dynamics (spike-frequency adaptation)
        The effective firing threshold is 0 mV (HH convention) plus a
        positive offset θ.  θ increments by _THETA_SPIKE_DELTA_MV on each
        spike and decays exponentially with τ = _THETA_DECAY_TAU_MS.
        Result: sustained input causes the firing rate to slow over time —
        spike-frequency adaptation.

    Calcium concentration [Ca²⁺]
        Increments by _CA_SPIKE_DELTA on each spike; decays with τ =
        _CA_DECAY_TAU_MS.  Used by Module 2's STDP rule to gate long-term
        potentiation / depression: high [Ca²⁺] → LTP, low [Ca²⁺] → LTD.

    Modulatory neurons
        Do not fire action potentials.  release_rate is a continuous graded
        signal proportional to membrane depolarisation above a tonic
        threshold, representing continuous neuromodulator release.
    """

    def __init__(self,
                 neuron_type: NeuronType = NeuronType.EXCITATORY,
                 dt: float = 0.01):
        self.neuron_type = neuron_type
        self.dt = dt  # integration timestep (ms)

        # Core HH state — initialised to resting equilibrium
        m0, h0, n0 = gate_steady_state(V_rest)
        self.V: float = V_rest
        self.m: float = m0
        self.h: float = h0
        self.n: float = n0

        # Refractory: time elapsed since last spike (inf = never spiked)
        self._t_since_spike: float = float('inf')

        # Adaptive threshold offset above the standard 0 mV threshold (mV)
        self._theta: float = 0.0

        # Calcium concentration [0.0 – 1.0]
        self._calcium: float = 0.0

        # Spike counter
        self._spike_count: int = 0

        # Modulatory neurons only: continuous neuromodulator release [0–1]
        self._release_rate: float = 0.0

    # ------------------------------------------------------------------
    @property
    def is_absolutely_refractory(self) -> bool:
        """True during the 2 ms absolute refractory window after a spike."""
        return self._t_since_spike < _ABS_REFRACTORY_MS

    @property
    def is_relatively_refractory(self) -> bool:
        """True during the ~3 ms relative refractory window after the absolute."""
        return _ABS_REFRACTORY_MS <= self._t_since_spike < _REL_REFRACTORY_MS

    @property
    def effective_threshold(self) -> float:
        """
        Current firing threshold (mV).
        Rises after spikes (θ > 0) and decays back to 0 mV.
        """
        return 0.0 + self._theta

    @property
    def calcium(self) -> float:
        """Normalised [Ca²⁺] in [0, 1].  Influences LTP/LTD strength."""
        return self._calcium

    @property
    def release_rate(self) -> float:
        """
        For MODULATORY neurons: continuous neuromodulator release rate [0, 1].
        Always 0.0 for EXCITATORY / INHIBITORY neurons.
        """
        return self._release_rate

    @property
    def spike_count(self) -> int:
        return self._spike_count

    # ------------------------------------------------------------------
    def step(self, I_ext: float = 0.0, t: float = 0.0) -> bool:
        """
        Advance the neuron by one timestep (self.dt ms).

        Parameters
        ----------
        I_ext : injected current (µA/cm²)
        t     : current simulation time (ms) — for bookkeeping only

        Returns
        -------
        bool : True if an action potential was emitted this step.
               Always False for MODULATORY neurons.
        """
        # --- Refractory gating ---
        if self.is_absolutely_refractory:
            I_ext = 0.0
        elif self.is_relatively_refractory:
            recovery = (
                (self._t_since_spike - _ABS_REFRACTORY_MS) /
                (_REL_REFRACTORY_MS  - _ABS_REFRACTORY_MS)
            )
            I_ext *= recovery   # linearly restored current

        # --- Euler integration of HH ODEs ---
        I_Na = gNa * self.m**3 * self.h * (self.V - ENa)
        I_K  = gK  * self.n**4           * (self.V - EK)
        I_L  = gL                         * (self.V - EL)
        dVdt = (I_ext - I_Na - I_K - I_L) / Cm

        V_prev  = self.V
        self.V += self.dt * dVdt
        self.V  = float(np.clip(self.V, -150.0, 150.0))   # prevent HH exp overflow
        self.m += self.dt * (alpha_m(self.V) * (1.0 - self.m) - beta_m(self.V) * self.m)
        self.h += self.dt * (alpha_h(self.V) * (1.0 - self.h) - beta_h(self.V) * self.h)
        self.n += self.dt * (alpha_n(self.V) * (1.0 - self.n) - beta_n(self.V) * self.n)

        # --- Per-step decay of threshold and calcium ---
        self._theta   *= np.exp(-self.dt / _THETA_DECAY_TAU_MS)
        self._calcium *= np.exp(-self.dt / _CA_DECAY_TAU_MS)

        # Advance refractory clock
        if self._t_since_spike < float('inf'):
            self._t_since_spike += self.dt

        # --- Modulatory: graded output, no spikes ---
        if self.neuron_type == NeuronType.MODULATORY:
            depol = max(0.0, self.V - (V_rest + 15.0))
            self._release_rate = min(1.0, depol / 30.0)
            return False

        # --- Spike detection (EXCITATORY / INHIBITORY) ---
        spiked = (
            V_prev < self.effective_threshold <= self.V
            and not self.is_absolutely_refractory
        )

        if spiked:
            self._spike_count   += 1
            self._t_since_spike  = 0.0
            self._theta         += _THETA_SPIKE_DELTA_MV
            self._calcium        = min(1.0, self._calcium + _CA_SPIKE_DELTA)

        return spiked

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset to resting state (preserves neuron_type and dt)."""
        m0, h0, n0 = gate_steady_state(V_rest)
        self.V = V_rest;  self.m = m0;  self.h = h0;  self.n = n0
        self._t_since_spike = float('inf')
        self._theta         = 0.0
        self._calcium       = 0.0
        self._spike_count   = 0
        self._release_rate  = 0.0


# ===========================================================================
# SECTION 7 — NeurogenesisManager
# ===========================================================================

class NeurogenesisManager:
    """
    Manages CAINE's total neuron count as a function of developmental stage.

    Neuron count is not fixed at birth — new neurons are added as stages
    progress, mirroring biological neurogenesis.  The count is always capped
    by real-time interactive performance requirements.  CAINE is always-on
    and always-developing — there is no training phase separate from
    deployment.

    Stage targets (from CAINE README):
        Stage 0  Birth        ~1,000
        Stage 1  Infancy      ~10,000
        Stage 2  Childhood    ~100,000
        Stage 3  Adolescence  ~500,000
        Stage 4  Maturity     hardware ceiling (PERF_CAP)
    """

    STAGE_TARGETS: Dict[int, Optional[int]] = {
        0: 1_000,
        1: 10_000,
        2: 100_000,
        3: 500_000,
        4: None,    # hardware ceiling — capped by PERF_CAP
    }

    PERF_CAP: int = 500_000   # real-time performance ceiling

    def __init__(self, initial_stage: int = 0):
        self._stage         = initial_stage
        self._total_neurons = self._target(initial_stage)
        self._birth_log: List[dict] = []

    # ------------------------------------------------------------------
    def _target(self, stage: int) -> int:
        t = self.STAGE_TARGETS.get(stage)
        return self.PERF_CAP if t is None else min(t, self.PERF_CAP)

    @property
    def total_neurons(self) -> int:
        return self._total_neurons

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def birth_log(self) -> List[dict]:
        return list(self._birth_log)

    # ------------------------------------------------------------------
    def advance_stage(self, new_stage: int) -> int:
        """
        Advance to new_stage.  Returns the number of new neurons added.
        No-op if new_stage <= current stage.
        """
        if new_stage <= self._stage:
            return 0
        old_count   = self._total_neurons
        self._stage = new_stage
        new_target  = self._target(new_stage)
        added       = max(0, new_target - old_count)
        self._total_neurons = new_target
        self._birth_log.append({
            'stage':         new_stage,
            'neurons_added': added,
            'total':         new_target,
        })
        print(f"[neurogenesis] Stage {new_stage}: +{added:,} neurons  "
              f"(total: {new_target:,})")
        return added

    # ------------------------------------------------------------------
    def gradual_add(self, count: int) -> int:
        """
        Incrementally add up to `count` neurons, capped by the current stage
        target.  Returns the actual number added.  Use for continuous
        within-stage neurogenesis (e.g. each simulated hour).
        """
        cap      = self._target(self._stage)
        headroom = max(0, cap - self._total_neurons)
        added    = min(count, headroom)
        self._total_neurons += added
        return added

    # ------------------------------------------------------------------
    def summary(self) -> str:
        cap_str = (f"{self.PERF_CAP:,}" if self._stage >= 4
                   else f"{self._target(self._stage):,}")
        return (f"stage={self._stage}  neurons={self._total_neurons:,}  "
                f"target={cap_str}")


# ===========================================================================
# SECTION 8 — Spike Detection Utility  (batch / post-processing)
# ===========================================================================

def detect_spikes(t: np.ndarray, V: np.ndarray,
                  threshold: float = 0.0,
                  refractory_ms: float = _ABS_REFRACTORY_MS
                  ) -> Tuple[list, list]:
    """
    Detect action potentials in a voltage trace (e.g. odeint output).

    A spike is recorded when V crosses `threshold` from below, provided at
    least `refractory_ms` have elapsed since the previous spike.

    Returns
    -------
    spike_times : list[float]  — times (ms) of detected action potentials
    spike_log   : list[dict]   — {time_ms, V_mV, spike_num}
    """
    spike_times: list = []
    spike_log:   list = []
    last_spike        = -float('inf')

    for i in range(1, len(V)):
        if (V[i - 1] < threshold <= V[i]
                and (t[i] - last_spike) > refractory_ms):
            spike_times.append(float(t[i]))
            last_spike = t[i]
            spike_log.append({
                'time_ms':   round(float(t[i]), 4),
                'V_mV':      round(float(V[i]), 2),
                'spike_num': len(spike_times),
            })

    return spike_times, spike_log


# ===========================================================================
# SECTION 9 — run_simulation
# Uses scipy.integrate.odeint (0.01 ms timestep) for V/m/h/n,
# then reconstructs calcium and adaptive-threshold traces from spike times.
# ===========================================================================

def run_simulation(
    duration_ms:     float     = 100.0,
    dt_ms:           float     = 0.01,
    I_ext_amplitude: float     = 10.0,
    I_start_ms:      float     = 10.0,
    I_stop_ms:       float     = 90.0,
    neuron_type:     NeuronType = NeuronType.EXCITATORY,
) -> Tuple:
    """
    Simulate a single HH neuron receiving a step-current injection.

    V, m, h, n are solved via scipy.integrate.odeint at dt_ms=0.01 ms.
    Calcium [Ca²⁺] and adaptive threshold θ are then reconstructed from
    the detected spike times using the same exponential dynamics as HHNeuron.

    Parameters
    ----------
    duration_ms      : total simulation time (ms)
    dt_ms            : odeint timestep (ms) — default 0.01
    I_ext_amplitude  : injected current (µA/cm²) — > ~7 µA/cm² triggers spikes
    I_start_ms       : current pulse start (ms)
    I_stop_ms        : current pulse end (ms)
    neuron_type      : NeuronType label (affects `extras` output only)

    Returns
    -------
    t        : time array (ms)
    V        : membrane voltage array (mV)
    states   : (n, 4) array — columns [V, m, h, n]
    spikes   : (spike_times, spike_log)
    extras   : dict — 'calcium', 'threshold' arrays (same length as t)
    """
    t = np.arange(0.0, duration_ms + dt_ms, dt_ms)

    def I_ext(time):
        return I_ext_amplitude if I_start_ms <= time <= I_stop_ms else 0.0

    m0, h0, n0 = gate_steady_state(V_rest)
    state0 = [V_rest, m0, h0, n0]

    print(f"[CAINE] Resting gate values: m={m0:.4f}  h={h0:.4f}  n={n0:.4f}")
    print(f"[CAINE] Simulating {duration_ms} ms  |  type={neuron_type.value}  |  "
          f"I_ext={I_ext_amplitude} µA/cm²  ({I_start_ms}–{I_stop_ms} ms)  "
          f"|  dt={dt_ms} ms  (odeint)")

    # --- Solve ODEs ---
    states = odeint(hh_odes, state0, t, args=(I_ext,))
    V = states[:, 0]

    # --- Detect spikes from voltage trace ---
    spike_times, spike_log = detect_spikes(t, V)

    print(f"[CAINE] Detected {len(spike_times)} action potential(s)")
    if len(spike_times) > 1:
        isi = np.diff(spike_times)
        print(f"[CAINE] Mean ISI: {isi.mean():.2f} ms  |  "
              f"Firing rate: {len(spike_times) / (duration_ms / 1000):.1f} Hz")

    print("\n--- Spike Log ---")
    if spike_log:
        for e in spike_log:
            print(f"  Spike #{e['spike_num']:3d}  "
                  f"t={e['time_ms']:8.3f} ms  V={e['V_mV']:+7.2f} mV")
    else:
        print("  (no spikes — try increasing I_ext_amplitude)")

    # --- Reconstruct calcium and adaptive threshold from spike times ---
    ca_trace = np.zeros(len(t))
    th_trace = np.zeros(len(t))
    for st in spike_times:
        idx  = np.searchsorted(t, st)
        tail = t[idx:] - st
        ca_trace[idx:] += _CA_SPIKE_DELTA      * np.exp(-tail / _CA_DECAY_TAU_MS)
        th_trace[idx:] += _THETA_SPIKE_DELTA_MV * np.exp(-tail / _THETA_DECAY_TAU_MS)
    ca_trace = np.clip(ca_trace, 0.0, 1.0)

    extras = {
        'calcium':   ca_trace,
        'threshold': th_trace,
    }
    return t, V, states, (spike_times, spike_log), extras


# ===========================================================================
# SECTION 10 — plot_results
# 4-panel dark-theme figure showing all Module 1 properties
# ===========================================================================

def plot_results(t, V, states, spike_times,
                 I_ext_amplitude, I_start_ms, I_stop_ms,
                 extras: Optional[dict] = None):
    """
    4-panel figure:
        1. Membrane voltage + adaptive threshold + spike markers
        2. Gating variables m, h, n
        3. Calcium concentration [Ca²⁺]
        4. Injected current + refractory windows
    """
    m = states[:, 1]
    h = states[:, 2]
    n = states[:, 3]
    ca        = extras['calcium']   if extras else np.zeros_like(V)
    threshold = extras['threshold'] if extras else np.zeros_like(V)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("CAINE — Module 1: Hodgkin-Huxley Neuron",
                 fontsize=14, fontweight='bold', color='#ffffff')
    fig.patch.set_facecolor('#111111')
    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.xaxis.label.set_color('#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    # --- Panel 1: Membrane voltage + adaptive threshold ---
    ax1 = axes[0]
    ax1.plot(t, V, color='#5b9bd5', linewidth=1.2, label='V  (mV)')
    ax1.plot(t, threshold, color='#e67e22', linewidth=0.8, linestyle='--',
             alpha=0.85, label='adaptive threshold  θ')
    for st in spike_times:
        ax1.axvline(st, color='#e74c3c', linewidth=0.6, alpha=0.5)
    ax1.axhline(0.0, color='#444444', linewidth=0.5, linestyle='--')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_ylim(-90, 60)
    ax1.legend(loc='upper right', fontsize=8,
               facecolor='#222222', labelcolor='white', framealpha=0.7)
    ax1.set_title(
        f"{len(spike_times)} action potential(s)  |  "
        f"I_ext = {I_ext_amplitude} µA/cm²  ({I_start_ms}–{I_stop_ms} ms)",
        color='#dddddd', fontsize=10)
    ax1.grid(True, alpha=0.12, color='#ffffff')

    # --- Panel 2: Gating variables ---
    ax2 = axes[1]
    ax2.plot(t, m, color='#e74c3c', linewidth=0.9, label='m  (Na⁺ activation)')
    ax2.plot(t, h, color='#e67e22', linewidth=0.9, label='h  (Na⁺ inactivation)')
    ax2.plot(t, n, color='#2ecc71', linewidth=0.9, label='n  (K⁺  activation)')
    ax2.set_ylabel('Gate probability')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right', fontsize=8,
               facecolor='#222222', labelcolor='white', framealpha=0.7)
    ax2.grid(True, alpha=0.12, color='#ffffff')

    # --- Panel 3: Calcium ---
    ax3 = axes[2]
    ax3.plot(t, ca, color='#9b59b6', linewidth=1.1,
             label='[Ca²⁺]  (normalised 0–1)')
    ax3.set_ylabel('[Ca²⁺]')
    ax3.set_ylim(-0.02, 1.05)
    ax3.legend(loc='upper right', fontsize=8,
               facecolor='#222222', labelcolor='white', framealpha=0.7)
    ax3.grid(True, alpha=0.12, color='#ffffff')

    # --- Panel 4: Injected current + refractory windows ---
    ax4 = axes[3]
    I_trace = np.where((t >= I_start_ms) & (t <= I_stop_ms),
                       I_ext_amplitude, 0.0)
    ax4.plot(t, I_trace, color='#7f8c8d', linewidth=1.4, label='I_ext  (µA/cm²)')
    for i_sp, st in enumerate(spike_times):
        label_abs = 'absolute refractory' if i_sp == 0 else '_nolegend_'
        label_rel = 'relative refractory' if i_sp == 0 else '_nolegend_'
        ax4.axvspan(st, st + _ABS_REFRACTORY_MS,
                    color='#e74c3c', alpha=0.20, label=label_abs)
        ax4.axvspan(st + _ABS_REFRACTORY_MS, st + _REL_REFRACTORY_MS,
                    color='#e67e22', alpha=0.12, label=label_rel)
    ax4.set_ylabel('I_ext  (µA/cm²)')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylim(-1, I_ext_amplitude * 1.4)
    ax4.legend(loc='upper right', fontsize=8,
               facecolor='#222222', labelcolor='white', framealpha=0.7)
    ax4.grid(True, alpha=0.12, color='#ffffff')

    plt.tight_layout()
    out = os.path.join(_OUTPUT_DIR, 'caine_module1_hh.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"\n[CAINE] Plot saved -> {out}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    # --- Run simulation ---
    t, V, states, (spike_times, spike_log), extras = run_simulation(
        duration_ms     = 100.0,
        dt_ms           = 0.01,
        I_ext_amplitude = 10.0,
        I_start_ms      = 10.0,
        I_stop_ms       = 90.0,
        neuron_type     = NeuronType.EXCITATORY,
    )

    plot_results(t, V, states, spike_times,
                 I_ext_amplitude = 10.0,
                 I_start_ms      = 10.0,
                 I_stop_ms       = 90.0,
                 extras          = extras)

    # --- Demo NeurogenesisManager ---
    print()
    ng = NeurogenesisManager(initial_stage=0)
    print(f"[neurogenesis] Initial  : {ng.summary()}")
    for stage in range(1, 5):
        ng.advance_stage(stage)
        print(f"[neurogenesis] Stage {stage}  : {ng.summary()}")

    # --- Demo all three NeuronTypes ---
    print()
    print("--- NeuronType Demo ---")
    for ntype in NeuronType:
        neuron = HHNeuron(neuron_type=ntype)
        spikes = 0
        for i in range(10000):  # 100 ms at dt=0.01
            I = 10.0 if 1000 <= i < 9000 else 0.0
            if neuron.step(I_ext=I, t=i * 0.01):
                spikes += 1
        if ntype == NeuronType.MODULATORY:
            print(f"  {ntype.value:12s}  release_rate={neuron.release_rate:.3f}  "
                  f"(no spikes — continuous release)")
        else:
            print(f"  {ntype.value:12s}  spikes={spikes:3d}  "
                  f"calcium={neuron.calcium:.3f}  "
                  f"threshold_offset={neuron.effective_threshold:.3f} mV")
