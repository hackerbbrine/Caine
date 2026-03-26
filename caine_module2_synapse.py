"""
CAINE — Module 2: Synaptic Connection and Plasticity
=====================================================
Two Hodgkin-Huxley neurons (Module 1) connected by a single AMPA synapse.

Implements:
  - AMPA synapse with fast conductance kinetics (tau ~5ms)
  - Spike-Timing Dependent Plasticity (STDP):
      Pre fires BEFORE post  →  potentiation  (ΔW = +A+ * exp(-Δt/τ+))
      Post fires BEFORE pre  →  depression    (ΔW = -A- * exp(-Δt/τ-))
      A+=0.01, A-=0.012, τ+=τ-=20ms
  - Synaptic health that decays over time; synapse is pruned when it falls
    below a threshold (mimicking biological synaptic elimination)
  - Real-time weight-change logging to stdout
  - 4-panel plot: pre voltage | post voltage | synapse weight | synapse health

Built on top of Module 1 (caine_module1_hh_neuron.py).
No ML frameworks — only numpy and scipy.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# ---------------------------------------------------------------------------
# Re-use all Module 1 constants and gate-rate functions directly.
# This module builds ON TOP of Module 1 — we import its primitives instead
# of copy-pasting them.
# ---------------------------------------------------------------------------
from caine_module1_hh_neuron import (
    Cm, gNa, gK, gL, ENa, EK, EL, V_rest,
    alpha_m, beta_m,
    alpha_h, beta_h,
    alpha_n, beta_n,
    gate_steady_state,
)


# ===========================================================================
# SECTION 1 — STDP PARAMETERS
# Standard symmetric STDP window (Song, Miller & Abbott 2000)
# ===========================================================================

STDP_A_PLUS    = 0.01   # potentiation amplitude  (pre → post, causal)
STDP_A_MINUS   = 0.012  # depression  amplitude   (post → pre, acausal)
STDP_TAU_PLUS  = 20.0   # potentiation time constant (ms)
STDP_TAU_MINUS = 20.0   # depression  time constant  (ms)


# ===========================================================================
# SECTION 2 — AMPA SYNAPSE PARAMETERS
# AMPA receptors: fast, excitatory, no voltage dependence (unlike NMDA)
# ===========================================================================

AMPA_TAU   = 5.0   # conductance decay time constant (ms) — fast excitation
AMPA_E_REV = 0.0   # reversal potential (mV) — mixed cation channel

# Peak conductance scaling factor.  The actual conductance injected on a
# spike is:  g_ampa += weight * AMPA_G_PEAK  (units: mS/cm²)
AMPA_G_PEAK = 0.8


# ===========================================================================
# SECTION 3 — SYNAPTIC HEALTH / PRUNING PARAMETERS
# Health is a continuous value in [0,1].  It decays exponentially when the
# synapse is inactive and recovers slightly with each spike.
# Below HEALTH_PRUNE_THRESHOLD the synapse is permanently deleted.
# ===========================================================================

HEALTH_TAU             = 200.0  # decay time constant (ms)
HEALTH_ACTIVITY_BONUS  = 0.05   # health boost per spike (pre or post)
HEALTH_PRUNE_THRESHOLD = 0.30   # synapse dies if health drops below this
HEALTH_INIT            = 1.0    # starting health (fully healthy)


# ===========================================================================
# SECTION 4 — SYNAPSE CLASS
# ===========================================================================

class Synapse:
    """
    Single chemical synapse connecting two Hodgkin-Huxley neurons.

    State variables (matching the README spec for Module 2):
      weight           : connection strength [0.0 – 1.0]
      delay_ms         : axonal transmission delay in milliseconds
      neurotransmitter : 'AMPA' | 'NMDA' | 'GABA-A' | 'GABA-B'
      last_pre_spike   : timestamp of last presynaptic spike (ms)
      last_post_spike  : timestamp of last postsynaptic spike (ms)
      eligibility      : eligibility trace (placeholder for future reward modulation)
      health           : pruning health value [0.0 – 1.0]
      g_ampa           : running AMPA conductance (mS/cm²)
      pruned           : True once health < threshold (permanent deletion)
    """

    def __init__(self, weight: float = 0.5,
                 delay_ms: float = 1.0,
                 neurotransmitter: str = 'AMPA'):
        self.weight           = float(np.clip(weight, 0.0, 1.0))
        self.delay_ms         = float(delay_ms)
        self.neurotransmitter = neurotransmitter
        self.last_pre_spike   = -np.inf   # sentinel: no spike recorded yet
        self.last_post_spike  = -np.inf
        self.eligibility      = 0.0       # reserved for Module 4 neuromodulation
        self.health           = HEALTH_INIT
        self.g_ampa           = 0.0       # conductance resting at zero
        self.pruned           = False

        # Real-time weight-change log: list of (time_ms, ΔW, new_weight, reason)
        self.weight_log: list = []

    # ------------------------------------------------------------------
    # 4.1  Conductance kinetics — called every timestep
    # ------------------------------------------------------------------

    def update_conductance(self, dt: float) -> None:
        """
        Exponential decay of open AMPA conductance.

        dg/dt = -g / tau_ampa
        Solution: g(t+dt) = g(t) * exp(-dt / tau_ampa)

        This recreates the time course of real AMPA receptor de-activation
        after a glutamate pulse: fast onset (modelled by on_pre_spike) and
        an ~5ms exponential tail.
        """
        if self.neurotransmitter == 'AMPA':
            self.g_ampa *= np.exp(-dt / AMPA_TAU)

    # ------------------------------------------------------------------
    # 4.2  Synaptic current — called every timestep to drive post neuron
    # ------------------------------------------------------------------

    def synaptic_current(self, V_post: float) -> float:
        """
        AMPA synaptic current using the HH outward-positive convention.

        I_syn = g_ampa * (V_post - E_rev)

        Sign interpretation (consistent with Module 1's HH formulation):
          - At rest V_post ≈ -65 mV, E_rev = 0 mV
          - I_syn = g * (-65 - 0) < 0   →  inward (depolarising) ✓
          - When subtracted in the ODE (dV = (I_ext - ΣI_ion - I_syn)/Cm),
            the negative I_syn reduces the total outward current → raises V.

        Returns 0 if the synapse has been pruned.
        """
        if self.pruned:
            return 0.0
        return self.g_ampa * (V_post - AMPA_E_REV)

    # ------------------------------------------------------------------
    # 4.3  Pre-spike arrival handler
    # ------------------------------------------------------------------

    def on_pre_spike(self, t: float) -> None:
        """
        Called when a presynaptic spike arrives (after axonal delay).

        Actions:
          1. Activate AMPA conductance (quantal release of neurotransmitter).
          2. STDP depression check: if post fired recently before pre, weaken.
          3. Record pre spike time for future post-spike STDP lookups.
          4. Health bonus: activity keeps the synapse alive.
        """
        if self.pruned:
            return

        # --- Neurotransmitter release: bump AMPA conductance ---
        self.g_ampa += self.weight * AMPA_G_PEAK

        # --- STDP: acausal (post BEFORE pre) → depression ---
        # Δt = t_pre - t_post_last  (positive: post fired first)
        if self.last_post_spike > -np.inf:
            delta_t = t - self.last_post_spike
            if delta_t > 0:
                dw = -STDP_A_MINUS * np.exp(-delta_t / STDP_TAU_MINUS)
                self._apply_weight_change(dw, t, reason='depression  (post->pre)')

        # Record timing for future post-spike STDP
        self.last_pre_spike = t

        # Activity bonus prevents premature pruning
        self._update_health(dt=0.0, spike_occurred=True)

    # ------------------------------------------------------------------
    # 4.4  Post-spike detection handler
    # ------------------------------------------------------------------

    def on_post_spike(self, t: float) -> None:
        """
        Called when the postsynaptic neuron fires.

        Actions:
          1. STDP potentiation check: if pre fired recently before post, strengthen.
          2. Record post spike time.
          3. Health bonus.
        """
        if self.pruned:
            return

        # --- STDP: causal (pre BEFORE post) → potentiation ---
        # Δt = t_post - t_pre_last  (positive: pre fired first)
        if self.last_pre_spike > -np.inf:
            delta_t = t - self.last_pre_spike
            if delta_t > 0:
                dw = STDP_A_PLUS * np.exp(-delta_t / STDP_TAU_PLUS)
                self._apply_weight_change(dw, t, reason='potentiation (pre->post)')

        self.last_post_spike = t
        self._update_health(dt=0.0, spike_occurred=True)

    # ------------------------------------------------------------------
    # 4.5  Continuous health decay — called every timestep
    # ------------------------------------------------------------------

    def update_health(self, dt: float) -> None:
        """
        Slow exponential decay of synaptic health; no activity → eventual pruning.
        This is called every simulation step regardless of spike activity.
        """
        self._update_health(dt=dt, spike_occurred=False)

    def _update_health(self, dt: float, spike_occurred: bool) -> None:
        """
        Combined health update:
          health(t+dt) = health(t) * exp(-dt / tau_health)   [continuous decay]
          if spike: health += activity_bonus                  [discrete boost]

        Pruning check: if health drops below threshold, synapse is eliminated.
        """
        # Exponential decay (skipped when dt=0, e.g. during spike events)
        if dt > 0.0:
            self.health *= np.exp(-dt / HEALTH_TAU)

        # Spike activity replenishes health slightly
        if spike_occurred:
            self.health = min(self.health + HEALTH_ACTIVITY_BONUS, 1.0)

        # --- Pruning: permanent deletion below threshold ---
        if self.health < HEALTH_PRUNE_THRESHOLD and not self.pruned:
            self.pruned = True
            self.weight = 0.0
            self.g_ampa = 0.0
            print(f"[SYNAPSE] *** PRUNED ***  health={self.health:.4f} "
                  f"< threshold={HEALTH_PRUNE_THRESHOLD}")

    # ------------------------------------------------------------------
    # 4.6  Weight update with logging
    # ------------------------------------------------------------------

    def _apply_weight_change(self, delta_w: float, t: float, reason: str) -> None:
        """
        Clip weight to [0, 1], record the change, and log it to stdout.
        Clipping implements the hard bounds on synaptic strength that
        prevent runaway potentiation or complete silencing beyond zero.
        """
        old_w        = self.weight
        self.weight  = float(np.clip(self.weight + delta_w, 0.0, 1.0))
        actual_delta = self.weight - old_w

        # Append to in-memory log for later plotting
        self.weight_log.append((t, actual_delta, self.weight, reason))

        # Real-time stdout log
        print(f"[STDP]  t={t:8.3f} ms  |  dW={actual_delta:+.6f}  |  "
              f"W={self.weight:.6f}  |  {reason}")


# ===========================================================================
# SECTION 5 — STEP-BY-STEP EULER INTEGRATOR FOR HH NEURONS
#
# Module 1 used scipy.odeint over the whole time-vector, which is perfect for
# a single neuron with a fixed stimulus.  Here we need to inject a synaptic
# current that changes moment-to-moment based on the OTHER neuron's spikes,
# so we advance one Euler step at a time.
#
# Euler is less accurate than RK4/odeint, but with dt=0.01 ms (matching
# Module 1's default) the error is well within acceptable bounds for
# qualitative neuroscience demonstrations.
# ===========================================================================

def hh_step_euler(V: float, m: float, h: float, n: float,
                  I_total: float, dt: float):
    """
    Advance one Hodgkin-Huxley neuron by a single Euler step.

    Parameters
    ----------
    V, m, h, n : current state (voltage mV, gate probabilities)
    I_total    : net injected current at this step (µA/cm²)
                 POSITIVE = depolarising (inward current convention).
                 Should include both external drive and synaptic inward current.
    dt         : timestep (ms)

    Returns
    -------
    (V_new, m_new, h_new, n_new)

    The ODE used (identical to Module 1's hh_odes):
      Cm * dV/dt = I_total - I_Na - I_K - I_L
      dx/dt = α_x(V)*(1-x) - β_x(V)*x   for x in {m, h, n}
    """
    # Ionic currents (outward-positive convention)
    I_Na = gNa * m**3 * h * (V - ENa)
    I_K  = gK  * n**4     * (V - EK)
    I_L  = gL             * (V - EL)

    # Derivatives
    dV = (I_total - I_Na - I_K - I_L) / Cm
    dm = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dh = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dn = alpha_n(V) * (1.0 - n) - beta_n(V) * n

    return (V + dt * dV,
            m + dt * dm,
            h + dt * dh,
            n + dt * dn)


# ===========================================================================
# SECTION 6 — SPIKE TRACKER
# Detects threshold crossings with refractory-period enforcement during
# step-by-step simulation (Module 1 did this in post-processing; here we
# need real-time detection so we can fire STDP immediately).
# ===========================================================================

class SpikeTracker:
    """
    Rolling spike detector for step-by-step HH integration.

    Detects upward threshold crossings while enforcing an absolute
    refractory period (default 2 ms, matching Module 1's detect_spikes).
    """

    def __init__(self, threshold: float = 0.0, refractory_ms: float = 2.0):
        self.threshold     = threshold
        self.refractory_ms = refractory_ms
        self.last_spike    = -np.inf
        self.spike_times   = []
        self._V_prev       = None     # voltage at previous timestep

    def check(self, t: float, V: float) -> bool:
        """
        Returns True (and records the time) if a spike is detected at
        the current timestep.

        Spike criterion: V crossed threshold upward AND refractory period over.
        """
        fired = False
        if self._V_prev is not None:
            crossed    = (self._V_prev < self.threshold <= V)
            not_refrac = (t - self.last_spike) > self.refractory_ms
            if crossed and not_refrac:
                self.spike_times.append(t)
                self.last_spike = t
                fired = True
        self._V_prev = V
        return fired


# ===========================================================================
# SECTION 7 — MAIN SIMULATION
# ===========================================================================

def run_synapse_simulation(
    duration_ms:       float = 300.0,
    dt_ms:             float = 0.01,
    # --- Presynaptic neuron: strong drive → fires reliably ---
    I_pre_amplitude:   float = 10.0,   # µA/cm²  (well above HH threshold ~7)
    I_pre_start_ms:    float = 10.0,
    I_pre_stop_ms:     float = 280.0,
    # --- Postsynaptic neuron: sub-threshold drive → needs synapse to fire ---
    I_post_amplitude:  float = 6.5,    # µA/cm²  (below firing threshold)
    I_post_start_ms:   float = 10.0,
    I_post_stop_ms:    float = 280.0,
    # --- Synapse ---
    init_weight:       float = 0.5,
    synaptic_delay_ms: float = 1.0,
):
    """
    Simulate two HH neurons connected by a single AMPA synapse.

    The presynaptic neuron fires reliably under strong drive.
    The postsynaptic neuron receives sub-threshold drive and relies on
    synaptic input to reach spike threshold — so pre almost always fires
    before post, producing net STDP potentiation.

    A spike-delay buffer implements the axonal transmission delay.

    Returns
    -------
    t_array       : time vector (ms)
    V_pre_trace   : presynaptic voltage over time (mV)
    V_post_trace  : postsynaptic voltage over time (mV)
    weight_trace  : synaptic weight over time
    health_trace  : synaptic health over time
    pre_spikes    : list of presynaptic spike times (ms)
    post_spikes   : list of postsynaptic spike times (ms)
    synapse       : the Synapse object (contains weight_log etc.)
    """

    # -----------------------------------------------------------------------
    # 7.1  Initialise time, neurons, and synapse
    # -----------------------------------------------------------------------
    t_array = np.arange(0.0, duration_ms + dt_ms, dt_ms)
    N       = len(t_array)

    m0, h0, n0 = gate_steady_state(V_rest)

    # Presynaptic neuron state [V, m, h, n]
    Vp, mp, hp, np_ = V_rest, m0, h0, n0

    # Postsynaptic neuron state
    Vq, mq, hq, nq = V_rest, m0, h0, n0

    # Pre-allocate output arrays
    V_pre_trace  = np.empty(N)
    V_post_trace = np.empty(N)
    weight_trace = np.empty(N)
    health_trace = np.empty(N)

    # Synapse
    synapse = Synapse(weight=init_weight,
                      delay_ms=synaptic_delay_ms,
                      neurotransmitter='AMPA')

    # Spike trackers
    pre_tracker  = SpikeTracker(threshold=0.0, refractory_ms=2.0)
    post_tracker = SpikeTracker(threshold=0.0, refractory_ms=2.0)

    # Axonal delay buffer: each entry is the arrival time of a pending spike.
    # When that time is reached, the spike is delivered to the synapse.
    delay_buffer: deque = deque()

    # -----------------------------------------------------------------------
    # 7.2  Console header
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("CAINE - Module 2: Synapse + STDP")
    print("=" * 65)

    print(f"  Duration      : {duration_ms} ms   dt={dt_ms} ms")
    print(f"  Neurotransmitter : {synapse.neurotransmitter}")
    print(f"  AMPA tau      : {AMPA_TAU} ms   g_peak={AMPA_G_PEAK}")
    print(f"  Initial weight: {init_weight:.3f}")
    print(f"  Axonal delay  : {synaptic_delay_ms} ms")
    print(f"  STDP  A+={STDP_A_PLUS}  A-={STDP_A_MINUS}  "
          f"tau+={STDP_TAU_PLUS}ms  tau-={STDP_TAU_MINUS}ms")
    print(f"  Health tau    : {HEALTH_TAU} ms   "
          f"prune threshold={HEALTH_PRUNE_THRESHOLD}")
    print("-" * 65)

    # -----------------------------------------------------------------------
    # 7.3  Main loop — advance one Euler step at a time
    # -----------------------------------------------------------------------
    for i, t in enumerate(t_array):

        # --- External drive (step currents) ---
        I_ext_pre  = (I_pre_amplitude
                      if I_pre_start_ms <= t <= I_pre_stop_ms  else 0.0)
        I_ext_post = (I_post_amplitude
                      if I_post_start_ms <= t <= I_post_stop_ms else 0.0)

        # --- Compute synaptic current into post neuron ---
        # I_syn follows HH outward-positive convention:
        #   I_syn = g_ampa * (V_post - E_rev)
        # At rest V_post < E_rev = 0 mV → I_syn < 0 → inward → depolarising.
        # We subtract I_syn from the total so it acts like an inward current:
        #   dV/dt = (I_total - I_Na - I_K - I_L - I_syn) / Cm
        #         = ((I_ext - I_syn) - ...) / Cm
        I_syn        = synapse.synaptic_current(Vq)
        I_total_post = I_ext_post - I_syn    # net "inward" drive for post neuron

        # --- Advance both neurons by one Euler step ---
        Vp, mp, hp, np_ = hh_step_euler(Vp, mp, hp, np_, I_ext_pre,   dt_ms)
        Vq, mq, hq, nq  = hh_step_euler(Vq, mq, hq, nq,  I_total_post, dt_ms)

        # --- Detect spikes (real-time, for immediate STDP) ---
        pre_fired  = pre_tracker.check(t, Vp)
        post_fired = post_tracker.check(t, Vq)

        # --- Handle presynaptic spike: queue delayed delivery ---
        if pre_fired:
            arrival_time = t + synapse.delay_ms
            delay_buffer.append(arrival_time)

        # --- Deliver any queued spikes whose delay has elapsed ---
        while delay_buffer and delay_buffer[0] <= t:
            delay_buffer.popleft()
            synapse.on_pre_spike(t)   # triggers AMPA + STDP depression check

        # --- Handle postsynaptic spike: immediate STDP potentiation check ---
        if post_fired:
            synapse.on_post_spike(t)

        # --- AMPA conductance decay (every step) ---
        synapse.update_conductance(dt_ms)

        # --- Synaptic health decay (every step, continuous) ---
        synapse.update_health(dt_ms)

        # --- Store traces ---
        V_pre_trace[i]  = Vp
        V_post_trace[i] = Vq
        weight_trace[i] = synapse.weight
        health_trace[i] = synapse.health

    # -----------------------------------------------------------------------
    # 7.4  Summary
    # -----------------------------------------------------------------------
    print("-" * 65)
    print(f"[CAINE] Pre  spikes  : {len(pre_tracker.spike_times)}")
    print(f"[CAINE] Post spikes  : {len(post_tracker.spike_times)}")
    print(f"[CAINE] STDP events  : {len(synapse.weight_log)}")
    print(f"[CAINE] Final weight : {synapse.weight:.6f}  "
          f"(init={init_weight:.3f}  "
          f"delta={synapse.weight - init_weight:+.6f})")
    print(f"[CAINE] Final health : {synapse.health:.6f}")
    if synapse.pruned:
        print(f"[CAINE] Synapse status: PRUNED")
    else:
        print(f"[CAINE] Synapse status: alive")
    print("=" * 65)

    return (t_array, V_pre_trace, V_post_trace, weight_trace, health_trace,
            pre_tracker.spike_times, post_tracker.spike_times, synapse)


# ===========================================================================
# SECTION 8 — PLOTTING
# ===========================================================================

def plot_module2(t, V_pre, V_post, weight_trace, health_trace,
                 pre_spikes, post_spikes, synapse,
                 init_weight: float = 0.5):
    """
    4-panel figure:
      Panel 1 — Presynaptic voltage trace (blue)
      Panel 2 — Postsynaptic voltage trace (red)
      Panel 3 — Synapse weight over time with STDP event markers
                (green lines = potentiation, red lines = depression)
      Panel 4 — Synaptic health over time with pruning-threshold marker
    """
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        "CAINE — Module 2: Synapse + STDP\n"
        f"AMPA  |  τ={AMPA_TAU}ms  |  "
        f"A+={STDP_A_PLUS}  A-={STDP_A_MINUS}  tau+/-={STDP_TAU_PLUS}ms  |  "
        f"init_w={init_weight:.2f}  →  final_w={weight_trace[-1]:.4f}",
        fontsize=11, fontweight='bold'
    )

    gs = gridspec.GridSpec(4, 1, hspace=0.50, top=0.91, bottom=0.07)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # ---- Panel 1: Presynaptic voltage ----------------------------------------
    ax1.plot(t, V_pre, color='steelblue', linewidth=0.7,
             label=f'Presynaptic  ({len(pre_spikes)} spikes)')
    # Mark spike times with faint vertical lines
    for st in pre_spikes:
        ax1.axvline(st, color='steelblue', linewidth=0.5, alpha=0.35)
    ax1.set_ylabel('V (mV)')
    ax1.set_title('Presynaptic neuron (neuron 1)', fontsize=9)
    ax1.set_ylim(-90, 65)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.22)

    # ---- Panel 2: Postsynaptic voltage ----------------------------------------
    ax2.plot(t, V_post, color='tomato', linewidth=0.7,
             label=f'Postsynaptic  ({len(post_spikes)} spikes)')
    for st in post_spikes:
        ax2.axvline(st, color='tomato', linewidth=0.5, alpha=0.35)
    ax2.set_ylabel('V (mV)')
    ax2.set_title('Postsynaptic neuron (neuron 2)  — sub-threshold drive + AMPA input',
                  fontsize=9)
    ax2.set_ylim(-90, 65)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.22)

    # ---- Panel 3: Synaptic weight ----------------------------------------
    ax3.plot(t, weight_trace, color='darkorchid', linewidth=1.2,
             label='Synaptic weight', zorder=3)
    ax3.axhline(init_weight, color='darkorchid', linewidth=0.8,
                linestyle=':', alpha=0.5, label=f'Initial weight ({init_weight})')

    # Overlay STDP event markers (coloured vertical lines per event)
    for (t_ev, dw, w_after, reason) in synapse.weight_log:
        col = 'limegreen' if 'potentiation' in reason else 'crimson'
        ax3.axvline(t_ev, color=col, linewidth=0.9, alpha=0.55, zorder=2)

    # Custom legend entries for event colours
    from matplotlib.lines import Line2D
    legend_extra = [
        Line2D([0], [0], color='limegreen', linewidth=1.2, label='Potentiation (pre→post)'),
        Line2D([0], [0], color='crimson',   linewidth=1.2, label='Depression  (post→pre)'),
    ]
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles + legend_extra, labels + [e.get_label() for e in legend_extra],
               loc='upper right', fontsize=7.5, ncol=2)

    ax3.set_ylabel('Weight')
    ax3.set_ylim(-0.02, 1.05)
    ax3.set_title(f'Synaptic weight  ({len(synapse.weight_log)} STDP events)', fontsize=9)
    ax3.grid(True, alpha=0.22)

    # ---- Panel 4: Synaptic health ----------------------------------------
    ax4.plot(t, health_trace, color='goldenrod', linewidth=1.2,
             label='Synaptic health')
    ax4.axhline(HEALTH_PRUNE_THRESHOLD, color='crimson', linewidth=1.0,
                linestyle='--', alpha=0.8,
                label=f'Prune threshold ({HEALTH_PRUNE_THRESHOLD})')
    if synapse.pruned:
        ax4.text(0.5, 0.5, 'PRUNED', transform=ax4.transAxes,
                 color='crimson', fontsize=20, alpha=0.4,
                 ha='center', va='center', fontweight='bold')
    ax4.set_ylabel('Health')
    ax4.set_ylim(-0.02, 1.10)
    ax4.set_xlabel('Time (ms)')
    ax4.set_title('Synaptic health  (exponential decay, activity bonus per spike)', fontsize=9)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.22)

    # Shared x-axis label on Panel 4 only
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    out_path = 'caine_module2_synapse.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n[CAINE] Plot saved -> {out_path}')
    plt.show()


# ===========================================================================
# SECTION 9 — ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    # Run the full simulation
    (t, V_pre, V_post, weight_trace, health_trace,
     pre_spikes, post_spikes, synapse) = run_synapse_simulation(
        duration_ms       = 300.0,
        dt_ms             = 0.01,
        # Pre: strong drive (fires reliably, ~70 Hz)
        I_pre_amplitude   = 10.0,
        I_pre_start_ms    = 10.0,
        I_pre_stop_ms     = 280.0,
        # Post: sub-threshold drive (needs synaptic input to fire)
        I_post_amplitude  = 6.5,
        I_post_start_ms   = 10.0,
        I_post_stop_ms    = 280.0,
        # Synapse starts at moderate weight; should potentiate (pre fires first)
        init_weight       = 0.5,
        synaptic_delay_ms = 1.0,
    )

    plot_module2(t, V_pre, V_post, weight_trace, health_trace,
                 pre_spikes, post_spikes, synapse,
                 init_weight=0.5)
