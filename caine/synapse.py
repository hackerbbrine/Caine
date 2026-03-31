"""
CAINE — Module 2: Synaptic Connection and Plasticity
======================================================
Synapses are not simple weight multipliers.  Each synapse is modelled with:

    class Synapse:
        weight: float           # connection strength [0.0 - 1.0]
        delay: float            # axonal transmission delay (ms)
        neurotransmitter: str   # AMPA / NMDA / GABA-A / GABA-B
        last_pre_spike: float   # timestamp of last presynaptic spike
        last_post_spike: float  # timestamp of last postsynaptic spike
        eligibility: float      # eligibility trace for delayed reward
        health: float           # pruning health value [0.0 - 1.0]

Neurotransmitter type determines the time course of postsynaptic current:
    AMPA   — fast excitation,  decays in ~5 ms
    NMDA   — slow excitation,  voltage-dependent, decays in ~100 ms,
              critical for plasticity (coincidence detector via Mg²⁺ block)
    GABA-A — fast inhibition,  decays in ~10 ms
    GABA-B — slow inhibition,  decays in ~200 ms

Spike-Timing Dependent Plasticity (STDP):
    Pre fires BEFORE post (causal):    ΔW = +A+ * exp(-Δt / τ+)   [potentiation]
    Pre fires AFTER  post (acausal):   ΔW = -A- * exp(-Δt / τ-)   [depression]

Where τ+, τ- ≈ 20 ms and A+, A- are learning rate constants modulated by
current neurochemical levels (acetylcholine gates all STDP; dopamine scales
A+ in reward-relevant regions).

Synaptic Pruning:
    health(t) = health(t-1) * decay_rate + activity_bonus * recent_firing_rate

When health drops below the prune threshold the synapse is permanently
deleted.  New synaptic sprouts can form between spatially nearby neurons
with low probability each timestep, mimicking axonal sprouting.  Pruning
rates are modulated by neurochemical state — high cortisol accelerates
pruning (smaller decay_rate), high serotonin slows it (larger decay_rate).
"""

import os
import sys as _sys
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

_OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)

from caine.neuron import (
    Cm, gNa, gK, gL, ENa, EK, EL, V_rest,
    alpha_m, beta_m,
    alpha_h, beta_h,
    alpha_n, beta_n,
    gate_steady_state,
)


# ===========================================================================
# SECTION 1 — STDP Parameters
# Standard asymmetric STDP window (Song, Miller & Abbott 2000)
# ===========================================================================

STDP_A_PLUS    = 0.01    # potentiation amplitude  (pre → post, causal)
STDP_A_MINUS   = 0.012   # depression  amplitude   (post → pre, acausal)
STDP_TAU_PLUS  = 20.0    # potentiation time constant (ms)
STDP_TAU_MINUS = 20.0    # depression  time constant  (ms)


# ===========================================================================
# SECTION 2 — Neurotransmitter Kinetics
# Per-type: tau (decay ms), e_rev (reversal potential mV), g_peak (mS/cm²)
# ===========================================================================

NT_PARAMS = {
    'AMPA':   {'tau': 5.0,   'e_rev':  0.0,  'g_peak': 0.8},
    'NMDA':   {'tau': 100.0, 'e_rev':  0.0,  'g_peak': 0.6},
    'GABA-A': {'tau': 10.0,  'e_rev': -70.0, 'g_peak': 0.8},
    'GABA-B': {'tau': 200.0, 'e_rev': -90.0, 'g_peak': 0.3},
}

# Backward-compat named constants used by cortex.py
AMPA_TAU   = NT_PARAMS['AMPA']['tau']
AMPA_E_REV = NT_PARAMS['AMPA']['e_rev']
AMPA_G_PEAK = NT_PARAMS['AMPA']['g_peak']

# NMDA Mg²⁺ block parameters (Jahr & Stevens 1990)
NMDA_MG_CONC = 1.0    # extracellular [Mg²⁺] concentration (mM)


# ===========================================================================
# SECTION 3 — Synaptic Health / Pruning Parameters
# ===========================================================================

HEALTH_TAU             = 200.0  # baseline decay time constant (ms)
HEALTH_ACTIVITY_BONUS  = 0.05   # health boost per spike (pre or post)
HEALTH_PRUNE_THRESHOLD = 0.30   # synapse dies if health drops below this
HEALTH_INIT            = 1.0    # starting health (fully healthy)


# ===========================================================================
# SECTION 4 — NMDA Mg²⁺ Block
# ===========================================================================

def nmda_mg_block(V_post: float) -> float:
    """
    Voltage-dependent Mg²⁺ block of NMDA receptors.

    At hyperpolarised potentials Mg²⁺ physically blocks the channel —
    preventing current flow even when glutamate is bound.  Depolarisation
    relieves the block, making NMDA a coincidence detector: both the
    presynaptic (glutamate) and postsynaptic (depolarisation) conditions
    must be met simultaneously.

    block(V) = 1 / (1 + [Mg²⁺] * exp(-0.062 * V) / 3.57)

    Returns a value in [0, 1]:  0 = fully blocked, 1 = fully open.
    """
    return 1.0 / (1.0 + NMDA_MG_CONC * np.exp(-0.062 * V_post) / 3.57)


# ===========================================================================
# SECTION 5 — Synapse Class
# ===========================================================================

class Synapse:
    """
    Single chemical synapse connecting two Hodgkin-Huxley neurons.

    Fields match the Module 2 README spec exactly:
        weight           connection strength [0.0 – 1.0]
        delay            axonal transmission delay (ms)
        neurotransmitter 'AMPA' | 'NMDA' | 'GABA-A' | 'GABA-B'
        last_pre_spike   timestamp of last presynaptic spike (ms)
        last_post_spike  timestamp of last postsynaptic spike (ms)
        eligibility      eligibility trace for delayed reward modulation
        health           pruning health value [0.0 – 1.0]

    Neurochemical modulation (set externally by NeurochemicalSystem):
        neuro_health_mod  multiplier on HEALTH_TAU
                           >1 = slower pruning (serotonin), <1 = faster (cortisol)
        neuro_stdp_scale  multiplier on A+/A-
                           gated by acetylcholine; boosted by dopamine
    """

    def __init__(self,
                 weight: float = 0.5,
                 delay: float = 1.0,
                 neurotransmitter: str = 'AMPA'):
        if neurotransmitter not in NT_PARAMS:
            raise ValueError(
                f"Unknown neurotransmitter '{neurotransmitter}'. "
                f"Must be one of: {list(NT_PARAMS)}")

        self.weight           = float(np.clip(weight, 0.0, 1.0))
        self.delay            = float(delay)          # axonal delay (ms)
        self.neurotransmitter = neurotransmitter
        self.last_pre_spike   = -float('inf')
        self.last_post_spike  = -float('inf')
        self.eligibility      = 0.0    # eligibility trace — updated each spike
        self.health           = HEALTH_INIT
        self.pruned           = False

        # Active conductance (mS/cm²) — single value regardless of NT type
        self._g = 0.0

        # Neurochemical modulation — initialised to neutral; updated externally
        # by NeurochemicalSystem.modulate_synapse()
        self.neuro_health_mod = 1.0   # tau multiplier (1.0 = baseline)
        self.neuro_stdp_scale = 1.0   # A+/A- multiplier (1.0 = baseline)

        # Real-time weight-change log: (time_ms, ΔW, new_weight, reason)
        self.weight_log: list = []

    # ------------------------------------------------------------------
    # Backward-compatibility alias
    # ------------------------------------------------------------------

    @property
    def delay_ms(self) -> float:
        """Alias for self.delay — preserved for legacy callers."""
        return self.delay

    @property
    def g_ampa(self) -> float:
        """Legacy accessor — returns active conductance when NT is AMPA."""
        return self._g if self.neurotransmitter == 'AMPA' else 0.0

    @g_ampa.setter
    def g_ampa(self, value: float) -> None:
        if self.neurotransmitter == 'AMPA':
            self._g = value

    # ------------------------------------------------------------------
    # 5.1  Conductance kinetics
    # ------------------------------------------------------------------

    def update_conductance(self, dt: float) -> None:
        """
        Exponential decay of open-channel conductance.

        Each neurotransmitter type decays with its own time constant:
            AMPA   τ =   5 ms   (fast excitation)
            NMDA   τ = 100 ms   (slow excitation, coincidence detector)
            GABA-A τ =  10 ms   (fast inhibition)
            GABA-B τ = 200 ms   (slow inhibition)
        """
        tau = NT_PARAMS[self.neurotransmitter]['tau']
        self._g *= np.exp(-dt / tau)

    # ------------------------------------------------------------------
    # 5.2  Synaptic current
    # ------------------------------------------------------------------

    def synaptic_current(self, V_post: float) -> float:
        """
        Net synaptic current at the postsynaptic membrane (µA/cm²).

        Uses the HH outward-positive convention:
            I_syn = g * (V_post - E_rev)

        At rest V_post ≈ -65 mV:
            AMPA / NMDA  E_rev =   0 mV  →  I_syn < 0  (inward, depolarising)
            GABA-A       E_rev = -70 mV  →  I_syn > 0  (outward, hyperpolarising)
            GABA-B       E_rev = -90 mV  →  I_syn > 0  (outward, hyperpolarising)

        NMDA current is additionally scaled by the Mg²⁺ block factor:
            I_NMDA = g * block(V_post) * (V_post - E_rev)
        """
        if self.pruned:
            return 0.0
        e_rev = NT_PARAMS[self.neurotransmitter]['e_rev']
        if self.neurotransmitter == 'NMDA':
            return self._g * nmda_mg_block(V_post) * (V_post - e_rev)
        return self._g * (V_post - e_rev)

    # ------------------------------------------------------------------
    # 5.3  Pre-spike arrival handler
    # ------------------------------------------------------------------

    def on_pre_spike(self, t: float) -> None:
        """
        Called when a presynaptic spike arrives (after axonal delay).

        1. Opens conductance (quantal neurotransmitter release).
        2. Acausal STDP: if post fired recently BEFORE this pre spike,
           apply depression (ΔW ∝ -A- * exp(-Δt / τ-)).
        3. Update eligibility trace.
        4. Health bonus (activity keeps synapse alive).
        """
        if self.pruned:
            return

        # --- Neurotransmitter release ---
        g_peak = NT_PARAMS[self.neurotransmitter]['g_peak']
        self._g += self.weight * g_peak

        # --- STDP: acausal (post BEFORE pre) → depression ---
        if self.last_post_spike > -float('inf'):
            delta_t = t - self.last_post_spike
            if delta_t > 0:
                a_minus = STDP_A_MINUS * self.neuro_stdp_scale
                dw = -a_minus * np.exp(-delta_t / STDP_TAU_MINUS)
                self._apply_weight_change(dw, t, reason='depression  (post->pre)')

        # Eligibility trace: decays between spikes (see Module 4)
        self.eligibility += 1.0 - self.eligibility

        self.last_pre_spike = t
        self._update_health(dt=0.0, spike_occurred=True)

    # ------------------------------------------------------------------
    # 5.4  Post-spike detection handler
    # ------------------------------------------------------------------

    def on_post_spike(self, t: float) -> None:
        """
        Called when the postsynaptic neuron fires.

        1. Causal STDP: if pre fired recently BEFORE this post spike,
           apply potentiation (ΔW ∝ +A+ * exp(-Δt / τ+)).
        2. Update eligibility trace.
        3. Health bonus.
        """
        if self.pruned:
            return

        # --- STDP: causal (pre BEFORE post) → potentiation ---
        if self.last_pre_spike > -float('inf'):
            delta_t = t - self.last_pre_spike
            if delta_t > 0:
                a_plus = STDP_A_PLUS * self.neuro_stdp_scale
                dw = a_plus * np.exp(-delta_t / STDP_TAU_PLUS)
                self._apply_weight_change(dw, t, reason='potentiation (pre->post)')

        self.last_post_spike = t
        self._update_health(dt=0.0, spike_occurred=True)

    # ------------------------------------------------------------------
    # 5.5  Continuous health decay
    # ------------------------------------------------------------------

    def update_health(self, dt: float) -> None:
        """
        Continuous health decay — called every simulation step.

        Implements the pruning formula from the README:
            health(t) = health(t-1) * decay_rate + activity_bonus * firing_rate

        decay_rate is derived from an effective tau modulated by neurochemicals:
            tau_eff = HEALTH_TAU * neuro_health_mod

        High cortisol  → neuro_health_mod < 1 → smaller tau → faster decay
        High serotonin → neuro_health_mod > 1 → larger  tau → slower decay
        """
        self._update_health(dt=dt, spike_occurred=False)

    def _update_health(self, dt: float, spike_occurred: bool) -> None:
        # Continuous exponential decay modulated by neurochemical state
        if dt > 0.0:
            effective_tau = HEALTH_TAU * max(0.1, self.neuro_health_mod)
            self.health  *= np.exp(-dt / effective_tau)

        # Spike activity replenishes health
        if spike_occurred:
            self.health = min(self.health + HEALTH_ACTIVITY_BONUS, 1.0)

        # Pruning: permanent deletion when health falls below threshold
        if self.health < HEALTH_PRUNE_THRESHOLD and not self.pruned:
            self.pruned = True
            self.weight = 0.0
            self._g     = 0.0
            print(f"[SYNAPSE] *** PRUNED ***  health={self.health:.4f} "
                  f"< threshold={HEALTH_PRUNE_THRESHOLD}")

    # ------------------------------------------------------------------
    # 5.6  Weight update with logging
    # ------------------------------------------------------------------

    def _apply_weight_change(self, delta_w: float, t: float, reason: str) -> None:
        old_w       = self.weight
        self.weight = float(np.clip(self.weight + delta_w, 0.0, 1.0))
        actual_dw   = self.weight - old_w
        self.weight_log.append((t, actual_dw, self.weight, reason))
        print(f"[STDP]  t={t:8.3f} ms  |  dW={actual_dw:+.6f}  |  "
              f"W={self.weight:.6f}  |  {reason}")


# ===========================================================================
# SECTION 6 — Synaptic Sprouting
# ===========================================================================

class SynapticSprouting:
    """
    Models the formation of new synaptic connections between nearby neurons.

    Each timestep, with probability SPROUT_PROBABILITY * dt_ms, a new synapse
    sprouts between two candidate neurons.  New sprouts are fragile — they
    start with low weight and health just above the prune threshold.  Most
    will be eliminated quickly unless they become active.

    Sprouting rate is:
        - Increased by norepinephrine (high arousal → more plasticity)
        - Decreased by cortisol (chronic stress suppresses new connections)

    This mirrors the biology: enriched environments produce more synaptic
    density; chronic stress reduces it.
    """

    SPROUT_PROBABILITY = 1e-5   # baseline: per candidate pair per ms
    SPROUT_INIT_WEIGHT = 0.05   # weak initial weight
    SPROUT_INIT_HEALTH = 0.35   # just above prune threshold — fragile by design

    def __init__(self,
                 neurotransmitter: str = 'AMPA',
                 rng: np.random.Generator = None):
        if neurotransmitter not in NT_PARAMS:
            raise ValueError(f"Unknown neurotransmitter '{neurotransmitter}'")
        self.neurotransmitter = neurotransmitter
        self._rng             = rng or np.random.default_rng()
        self.total_sprouted   = 0

    def tick(self,
             dt_ms: float,
             n_candidates: int = 100,
             norepinephrine: float = 0.0,
             cortisol: float = 0.0) -> list:
        """
        Compute new synaptic sprouts for this timestep.

        Parameters
        ----------
        dt_ms          : frame duration (ms)
        n_candidates   : number of potential new connections to test
        norepinephrine : current NE level [0–1] — increases sprouting rate
        cortisol       : current cortisol level [0–1] — reduces sprouting rate

        Returns
        -------
        list of new Synapse objects (caller attaches them to specific neurons)
        """
        rate = self.SPROUT_PROBABILITY * dt_ms
        rate *= (1.0 + norepinephrine * 2.0)
        rate *= max(0.1, 1.0 - cortisol * 0.8)
        rate  = min(rate, 1.0)

        n_new = int(self._rng.binomial(n_candidates, rate))
        new_synapses = []
        for _ in range(n_new):
            s = Synapse(
                weight           = self.SPROUT_INIT_WEIGHT,
                delay            = float(self._rng.uniform(0.5, 5.0)),
                neurotransmitter = self.neurotransmitter,
            )
            s.health = self.SPROUT_INIT_HEALTH
            new_synapses.append(s)

        self.total_sprouted += n_new
        return new_synapses


# ===========================================================================
# SECTION 7 — Euler Integrator (step-by-step HH for synapse simulation)
# ===========================================================================

def hh_step_euler(V, m, h, n, I_total, dt):
    """
    Advance one Hodgkin-Huxley neuron by a single Euler step.

    I_total : net injected current (µA/cm²), inward-positive.
              Includes both external drive and synaptic current with sign
              adjusted so that excitatory input depolarises.
    Returns (V_new, m_new, h_new, n_new).
    """
    I_Na = gNa * m**3 * h * (V - ENa)
    I_K  = gK  * n**4     * (V - EK)
    I_L  = gL             * (V - EL)
    dV   = (I_total - I_Na - I_K - I_L) / Cm
    dm   = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dh   = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dn   = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    return (V + dt * dV, m + dt * dm, h + dt * dh, n + dt * dn)


# ===========================================================================
# SECTION 8 — Spike Tracker
# ===========================================================================

class SpikeTracker:
    """
    Real-time threshold-crossing detector with absolute refractory period.
    Used during step-by-step simulation so STDP fires immediately on spike.
    """

    def __init__(self, threshold: float = 0.0, refractory_ms: float = 2.0):
        self.threshold     = threshold
        self.refractory_ms = refractory_ms
        self.last_spike    = -float('inf')
        self.spike_times   = []
        self._V_prev       = None

    def check(self, t: float, V: float) -> bool:
        fired = False
        if self._V_prev is not None:
            crossed    = self._V_prev < self.threshold <= V
            not_refrac = (t - self.last_spike) > self.refractory_ms
            if crossed and not_refrac:
                self.spike_times.append(t)
                self.last_spike = t
                fired = True
        self._V_prev = V
        return fired


# ===========================================================================
# SECTION 9 — Main Simulation
# ===========================================================================

def run_synapse_simulation(
    duration_ms:       float = 300.0,
    dt_ms:             float = 0.01,
    I_pre_amplitude:   float = 10.0,
    I_pre_start_ms:    float = 10.0,
    I_pre_stop_ms:     float = 280.0,
    I_post_amplitude:  float = 6.5,
    I_post_start_ms:   float = 10.0,
    I_post_stop_ms:    float = 280.0,
    init_weight:       float = 0.5,
    synaptic_delay_ms: float = 1.0,
    neurotransmitter:  str   = 'AMPA',
):
    """
    Simulate two HH neurons connected by a single synapse.

    The presynaptic neuron fires reliably under strong drive.
    The postsynaptic neuron receives sub-threshold drive and relies on
    synaptic input to reach spike threshold — so pre almost always fires
    before post, producing net STDP potentiation.

    Returns
    -------
    t_array, V_pre_trace, V_post_trace, weight_trace, health_trace,
    pre_spikes, post_spikes, synapse
    """
    t_array = np.arange(0.0, duration_ms + dt_ms, dt_ms)
    N       = len(t_array)

    m0, h0, n0 = gate_steady_state(V_rest)
    Vp, mp, hp, np_ = V_rest, m0, h0, n0
    Vq, mq, hq, nq  = V_rest, m0, h0, n0

    V_pre_trace  = np.empty(N)
    V_post_trace = np.empty(N)
    weight_trace = np.empty(N)
    health_trace = np.empty(N)

    synapse = Synapse(weight=init_weight,
                      delay=synaptic_delay_ms,
                      neurotransmitter=neurotransmitter)

    pre_tracker  = SpikeTracker(threshold=0.0, refractory_ms=2.0)
    post_tracker = SpikeTracker(threshold=0.0, refractory_ms=2.0)
    delay_buffer: deque = deque()

    nt_info = NT_PARAMS[neurotransmitter]
    print("=" * 65)
    print("CAINE - Module 2: Synapse + STDP")
    print("=" * 65)
    print(f"  Neurotransmitter : {neurotransmitter}  "
          f"(tau={nt_info['tau']} ms, E_rev={nt_info['e_rev']} mV)")
    print(f"  Duration         : {duration_ms} ms   dt={dt_ms} ms")
    print(f"  Initial weight   : {init_weight:.3f}")
    print(f"  Axonal delay     : {synaptic_delay_ms} ms")
    print(f"  STDP: A+={STDP_A_PLUS}  A-={STDP_A_MINUS}  "
          f"tau+={STDP_TAU_PLUS} ms  tau-={STDP_TAU_MINUS} ms")
    print(f"  Health: tau={HEALTH_TAU} ms  "
          f"prune_threshold={HEALTH_PRUNE_THRESHOLD}")
    print("-" * 65)

    for i, t in enumerate(t_array):
        I_ext_pre  = I_pre_amplitude  if I_pre_start_ms  <= t <= I_pre_stop_ms  else 0.0
        I_ext_post = I_post_amplitude if I_post_start_ms <= t <= I_post_stop_ms else 0.0

        # Synaptic current drives post neuron
        # Excitatory (AMPA/NMDA): I_syn < 0 at rest → subtract to depolarise
        # Inhibitory (GABA):      I_syn > 0 at rest → subtract to hyperpolarise
        I_syn        = synapse.synaptic_current(Vq)
        I_total_post = I_ext_post - I_syn

        Vp, mp, hp, np_ = hh_step_euler(Vp, mp, hp, np_, I_ext_pre,   dt_ms)
        Vq, mq, hq, nq  = hh_step_euler(Vq, mq, hq, nq,  I_total_post, dt_ms)

        pre_fired  = pre_tracker.check(t, Vp)
        post_fired = post_tracker.check(t, Vq)

        if pre_fired:
            delay_buffer.append(t + synapse.delay)

        while delay_buffer and delay_buffer[0] <= t:
            delay_buffer.popleft()
            synapse.on_pre_spike(t)

        if post_fired:
            synapse.on_post_spike(t)

        synapse.update_conductance(dt_ms)
        synapse.update_health(dt_ms)

        V_pre_trace[i]  = Vp
        V_post_trace[i] = Vq
        weight_trace[i] = synapse.weight
        health_trace[i] = synapse.health

    print("-" * 65)
    print(f"[CAINE] Pre  spikes  : {len(pre_tracker.spike_times)}")
    print(f"[CAINE] Post spikes  : {len(post_tracker.spike_times)}")
    print(f"[CAINE] STDP events  : {len(synapse.weight_log)}")
    print(f"[CAINE] Final weight : {synapse.weight:.6f}  "
          f"(init={init_weight:.3f}  delta={synapse.weight - init_weight:+.6f})")
    print(f"[CAINE] Final health : {synapse.health:.6f}  "
          f"status={'PRUNED' if synapse.pruned else 'alive'}")
    print("=" * 65)

    return (t_array, V_pre_trace, V_post_trace, weight_trace, health_trace,
            pre_tracker.spike_times, post_tracker.spike_times, synapse)


# ===========================================================================
# SECTION 10 — Plotting
# ===========================================================================

def plot_module2(t, V_pre, V_post, weight_trace, health_trace,
                 pre_spikes, post_spikes, synapse,
                 init_weight: float = 0.5):
    """
    4-panel dark-theme figure:
        1. Presynaptic voltage
        2. Postsynaptic voltage
        3. Synaptic weight + STDP event markers
        4. Synaptic health + prune threshold
    """
    nt   = synapse.neurotransmitter
    info = NT_PARAMS[nt]

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor('#111111')
    fig.suptitle(
        f"CAINE — Module 2: Synapse + STDP\n"
        f"{nt}  tau={info['tau']} ms  E_rev={info['e_rev']} mV  |  "
        f"A+={STDP_A_PLUS}  A-={STDP_A_MINUS}  tau+/-={STDP_TAU_PLUS} ms  |  "
        f"init_w={init_weight:.2f} -> final_w={weight_trace[-1]:.4f}",
        fontsize=10, fontweight='bold', color='#ffffff',
    )

    gs  = gridspec.GridSpec(4, 1, hspace=0.50, top=0.91, bottom=0.07)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for i, ax in enumerate(axes):
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.xaxis.label.set_color('#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        if i < 3:
            plt.setp(ax.get_xticklabels(), visible=False)

    ax1, ax2, ax3, ax4 = axes

    # --- Panel 1: Presynaptic voltage ---
    ax1.plot(t, V_pre, color='#5b9bd5', linewidth=0.7,
             label=f'Pre ({len(pre_spikes)} spikes)')
    for st in pre_spikes:
        ax1.axvline(st, color='#5b9bd5', linewidth=0.4, alpha=0.3)
    ax1.set_ylabel('V (mV)')
    ax1.set_ylim(-90, 65)
    ax1.set_title('Presynaptic neuron', fontsize=9, color='#dddddd')
    ax1.legend(loc='upper right', fontsize=8,
               facecolor='#222222', labelcolor='white', framealpha=0.7)
    ax1.grid(True, alpha=0.12, color='#ffffff')

    # --- Panel 2: Postsynaptic voltage ---
    ax2.plot(t, V_post, color='#e74c3c', linewidth=0.7,
             label=f'Post ({len(post_spikes)} spikes)')
    for st in post_spikes:
        ax2.axvline(st, color='#e74c3c', linewidth=0.4, alpha=0.3)
    ax2.set_ylabel('V (mV)')
    ax2.set_ylim(-90, 65)
    ax2.set_title(f'Postsynaptic neuron  —  sub-threshold drive + {nt} input',
                  fontsize=9, color='#dddddd')
    ax2.legend(loc='upper right', fontsize=8,
               facecolor='#222222', labelcolor='white', framealpha=0.7)
    ax2.grid(True, alpha=0.12, color='#ffffff')

    # --- Panel 3: Synaptic weight ---
    ax3.plot(t, weight_trace, color='#9b59b6', linewidth=1.2,
             label='Weight', zorder=3)
    ax3.axhline(init_weight, color='#9b59b6', linewidth=0.7,
                linestyle=':', alpha=0.5, label=f'Initial ({init_weight})')
    from matplotlib.lines import Line2D
    for (t_ev, dw, _, reason) in synapse.weight_log:
        col = '#2ecc71' if 'potentiation' in reason else '#e74c3c'
        ax3.axvline(t_ev, color=col, linewidth=0.8, alpha=0.5, zorder=2)
    legend_extra = [
        Line2D([0], [0], color='#2ecc71', lw=1.2, label='Potentiation (pre->post)'),
        Line2D([0], [0], color='#e74c3c', lw=1.2, label='Depression  (post->pre)'),
    ]
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles + legend_extra, labels + [e.get_label() for e in legend_extra],
               loc='upper right', fontsize=7, ncol=2,
               facecolor='#222222', labelcolor='white', framealpha=0.7)
    ax3.set_ylabel('Weight')
    ax3.set_ylim(-0.02, 1.05)
    ax3.set_title(f'Synaptic weight  ({len(synapse.weight_log)} STDP events)',
                  fontsize=9, color='#dddddd')
    ax3.grid(True, alpha=0.12, color='#ffffff')

    # --- Panel 4: Synaptic health ---
    ax4.plot(t, health_trace, color='#f39c12', linewidth=1.2,
             label='Health')
    ax4.axhline(HEALTH_PRUNE_THRESHOLD, color='#e74c3c', linewidth=1.0,
                linestyle='--', alpha=0.8,
                label=f'Prune threshold ({HEALTH_PRUNE_THRESHOLD})')
    if synapse.pruned:
        ax4.text(0.5, 0.5, 'PRUNED', transform=ax4.transAxes,
                 color='#e74c3c', fontsize=20, alpha=0.4,
                 ha='center', va='center', fontweight='bold')
    ax4.set_ylabel('Health')
    ax4.set_ylim(-0.02, 1.10)
    ax4.set_xlabel('Time (ms)')
    ax4.set_title('Synaptic health  (cortisol accelerates, serotonin slows)',
                  fontsize=9, color='#dddddd')
    ax4.legend(loc='upper right', fontsize=8,
               facecolor='#222222', labelcolor='white', framealpha=0.7)
    ax4.grid(True, alpha=0.12, color='#ffffff')

    out = os.path.join(_OUTPUT_DIR, 'caine_module2_synapse.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f'\n[CAINE] Plot saved -> {out}')


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    # --- Primary demo: AMPA synapse ---
    (t, V_pre, V_post, weight_trace, health_trace,
     pre_spikes, post_spikes, synapse) = run_synapse_simulation(
        duration_ms       = 300.0,
        dt_ms             = 0.01,
        I_pre_amplitude   = 10.0,
        I_pre_start_ms    = 10.0,
        I_pre_stop_ms     = 280.0,
        I_post_amplitude  = 6.5,
        I_post_start_ms   = 10.0,
        I_post_stop_ms    = 280.0,
        init_weight       = 0.5,
        synaptic_delay_ms = 1.0,
        neurotransmitter  = 'AMPA',
    )
    plot_module2(t, V_pre, V_post, weight_trace, health_trace,
                 pre_spikes, post_spikes, synapse, init_weight=0.5)

    # --- NMDA Mg²⁺ block demo ---
    print("\n--- NMDA Mg-block at various voltages ---")
    for V in [-80, -65, -50, -30, 0, 20]:
        print(f"  V={V:+4d} mV  ->  Mg-block={nmda_mg_block(float(V)):.3f}")

    # --- Neurotransmitter kinetics demo ---
    print("\n--- All 4 neurotransmitter types ---")
    for nt, p in NT_PARAMS.items():
        s = Synapse(weight=0.5, delay=1.0, neurotransmitter=nt)
        s._g = 0.5
        I = s.synaptic_current(V_post=V_rest)
        print(f"  {nt:6s}  tau={p['tau']:5.0f} ms  "
              f"E_rev={p['e_rev']:+5.1f} mV  "
              f"I_syn(V_rest)={I:+.4f} uA/cm2")

    # --- Synaptic sprouting demo ---
    print("\n--- Synaptic sprouting (100 candidates, 100 ms, NE=0.5) ---")
    sprout = SynapticSprouting(neurotransmitter='AMPA')
    new_syns = sprout.tick(dt_ms=100.0, n_candidates=100,
                           norepinephrine=0.5, cortisol=0.0)
    print(f"  Sprouted {len(new_syns)} new synapses  "
          f"(weight={SPROUT_PROBABILITY:.1e} baseline)")
    if new_syns:
        s0 = new_syns[0]
        print(f"  Example: weight={s0.weight:.3f}  "
              f"health={s0.health:.3f}  delay={s0.delay:.2f} ms")

    # Expose as named constant for SynapticSprouting callers
    SPROUT_PROBABILITY = SynapticSprouting.SPROUT_PROBABILITY
