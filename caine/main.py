"""
CAINE - Combined Demo: All Systems Running Together
====================================================
Demonstrates Modules 1-3 operating simultaneously:

  Module 1 (neuron)    -- two HH neurons (pre and post)
  Module 2 (synapse)   -- AMPA synapse with STDP connecting them
  Module 3 (chemicals) -- NeurochemicalSystem modulating STDP in real time

Scenario (300 ms):
  0   -  20 ms : both neurons at rest (baseline drive only)
  20  - 280 ms : pre neuron driven above threshold (fires reliably)
                 post neuron driven below threshold (needs synapse to fire)
  100 ms       : NOVEL_STIMULUS  -> DA + NE + ACh spike
                   NE global_gain  boosts effective post-neuron drive
                   ACh learning_gate opens -> STDP amplitude rises
  160 ms       : REWARD          -> DA spike -> further potentiation boost
  220 ms       : DIRECTED_GAZE   -> ACh spike -> sustains learning gate

The combined plot (5 panels) shows:
  1. Pre-synaptic voltage trace
  2. Post-synaptic voltage trace
  3. Synapse weight over time (STDP events as coloured markers)
  4. DA, NE, ACh concentrations
  5. STDP effective scale factor (DA + ACh + OT combined)

All output written to output/.
"""

import os
import sys as _sys
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from collections import deque

# ---------------------------------------------------------------------------
# Package imports -- all three modules through the caine package
# ---------------------------------------------------------------------------
from caine import OUTPUT_DIR
from caine.neuron import (
    Cm, gNa, gK, gL, ENa, EK, EL, V_rest,
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n,
    gate_steady_state,
)
from caine.synapse import (
    Synapse, SpikeTracker, hh_step_euler,
    STDP_A_PLUS, STDP_A_MINUS, STDP_TAU_PLUS, STDP_TAU_MINUS,
    AMPA_TAU, AMPA_E_REV, AMPA_G_PEAK,
    HEALTH_TAU,
)
from caine.chemicals import (
    NeurochemicalSystem, NeurochemicalEvent, EventType,
)


# ===========================================================================
# SECTION 1 -- SIMULATION PARAMETERS
# ===========================================================================

DURATION_MS      = 300.0
DT_MS            = 0.01

I_PRE_AMP        = 10.0   # uA/cm^2 -- pre fires reliably (~70 Hz)
I_PRE_START      = 20.0
I_PRE_STOP       = 280.0

I_POST_AMP       = 6.5    # uA/cm^2 -- sub-threshold; needs synapse to fire
I_POST_START     = 20.0
I_POST_STOP      = 280.0

INIT_WEIGHT      = 0.5
SYNAPTIC_DELAY   = 1.0    # ms


# Neurochemical event schedule: (time_ms, EventType, magnitude)
_NEURO_SCHEDULE = [
    (100.0, EventType.NOVEL_STIMULUS,  1.0),   # DA + NE + ACh
    (160.0, EventType.REWARD,          1.0),   # DA boost
    (220.0, EventType.DIRECTED_GAZE,   1.0),   # ACh sustain
]


# ===========================================================================
# SECTION 2 -- COMBINED SIMULATION
# ===========================================================================

def run_combined(
    duration_ms: float = DURATION_MS,
    dt_ms:       float = DT_MS,
) -> dict:
    """
    Run all three modules simultaneously.

    The neurochemical system updates every step.  Its modulation outputs
    are applied to the synapse before each STDP event:
      - synapse.neuro_stdp_scale  <- neuro.stdp_scale()
      - synapse.neuro_health_mod  <- neuro.health_decay_mod()
      - synapse.neuro_gate        <- neuro.learning_gate()

    The synapse's STDP events use these scale factors so that, e.g.,
    a dopamine spike at t=100ms directly amplifies weight changes that
    occur in the following ~800ms window (DA tau = 800ms).
    """

    t_array = np.arange(0.0, duration_ms + dt_ms, dt_ms)
    N       = len(t_array)

    # --- Initialise neurons ---
    m0, h0, n0 = gate_steady_state(V_rest)
    Vp, mp, hp, np_ = V_rest, m0, h0, n0   # pre
    Vq, mq, hq, nq  = V_rest, m0, h0, n0   # post

    # --- Initialise synapse and neurochemical system ---
    synapse = Synapse(weight=INIT_WEIGHT, delay_ms=SYNAPTIC_DELAY,
                      neurotransmitter='AMPA')
    neuro   = NeurochemicalSystem()

    # --- Spike trackers ---
    pre_tracker  = SpikeTracker(threshold=0.0, refractory_ms=2.0)
    post_tracker = SpikeTracker(threshold=0.0, refractory_ms=2.0)
    delay_buffer: deque = deque()

    # Build event lookup: step index -> list of NeurochemicalEvent
    event_map: dict = {}
    for t_ev, ev_type, mag in _NEURO_SCHEDULE:
        idx = int(round(t_ev / dt_ms))
        event_map.setdefault(idx, []).append(NeurochemicalEvent(ev_type, mag))

    # --- Storage ---
    V_pre_trace  = np.empty(N)
    V_post_trace = np.empty(N)
    weight_trace = np.empty(N)
    health_trace = np.empty(N)

    chem_names = ['dopamine', 'norepinephrine', 'acetylcholine']
    chem_traces = {n: np.empty(N) for n in chem_names}
    stdp_scale_trace = np.empty(N)

    # --- Console header ---
    print("=" * 65)
    print("CAINE - Combined Demo: Modules 1 + 2 + 3")
    print("=" * 65)
    print(f"  Duration : {duration_ms} ms  |  dt: {dt_ms} ms")
    print(f"  Synapse  : {synapse.neurotransmitter}  "
          f"init_weight={INIT_WEIGHT}  delay={SYNAPTIC_DELAY}ms")
    print(f"  STDP     : A+={STDP_A_PLUS}  A-={STDP_A_MINUS}  "
          f"tau+={STDP_TAU_PLUS}ms  tau-={STDP_TAU_MINUS}ms")
    print(f"  Events   : {len(_NEURO_SCHEDULE)} neurochemical events scheduled")
    print("-" * 65)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for i, t in enumerate(t_array):

        # --- Neurochemical events ---
        events_now = event_map.get(i)
        if events_now:
            for ev in events_now:
                print(f"[EVENT] t={t:7.1f} ms  {ev.event_type.name:<26}"
                      f"mag={ev.magnitude:.2f}")
        neuro.update(dt_ms, events=events_now, current_time=t)

        # --- Push neurochemical modulation into the synapse ---
        # This is the live coupling between Module 3 and Module 2.
        # Every STDP weight change this step will be scaled by these values.
        neuro.modulate_synapse(synapse)

        # --- External drives ---
        I_ext_pre  = (I_PRE_AMP
                      if I_PRE_START <= t <= I_PRE_STOP else 0.0)
        I_ext_post = (I_POST_AMP
                      if I_POST_START <= t <= I_POST_STOP else 0.0)

        # NE global_gain boosts the post neuron's effective excitability.
        # Scaling I_ext_post approximates lowered firing threshold under NE.
        I_ext_post_mod = I_ext_post * neuro.global_gain()

        # --- Synaptic current (outward-positive; subtracted in ODE) ---
        I_syn        = synapse.synaptic_current(Vq)
        I_total_post = I_ext_post_mod - I_syn

        # --- Euler step both neurons ---
        Vp, mp, hp, np_ = hh_step_euler(Vp, mp, hp, np_, I_ext_pre,   dt_ms)
        Vq, mq, hq, nq  = hh_step_euler(Vq, mq, hq, nq,  I_total_post, dt_ms)

        # --- Spike detection ---
        pre_fired  = pre_tracker.check(t, Vp)
        post_fired = post_tracker.check(t, Vq)

        # --- Synapse update (Module 2 logic, neurochemically modulated) ---
        if pre_fired:
            delay_buffer.append(t + synapse.delay_ms)
        while delay_buffer and delay_buffer[0] <= t:
            delay_buffer.popleft()
            synapse.on_pre_spike(t)
        if post_fired:
            synapse.on_post_spike(t)

        synapse.update_conductance(dt_ms)
        synapse.update_health(dt_ms)

        # --- Record ---
        V_pre_trace[i]      = Vp
        V_post_trace[i]     = Vq
        weight_trace[i]     = synapse.weight
        health_trace[i]     = synapse.health
        stdp_scale_trace[i] = synapse.neuro_stdp_scale

        snap = neuro.snapshot()
        for name in chem_names:
            chem_traces[name][i] = snap[name]

    # --- Summary ---
    print("-" * 65)
    print(f"[CAINE] Pre  spikes   : {len(pre_tracker.spike_times)}")
    print(f"[CAINE] Post spikes   : {len(post_tracker.spike_times)}")
    print(f"[CAINE] STDP events   : {len(synapse.weight_log)}")
    print(f"[CAINE] Final weight  : {synapse.weight:.6f}  "
          f"(init={INIT_WEIGHT:.3f}  delta={synapse.weight-INIT_WEIGHT:+.6f})")
    print(f"[CAINE] Final health  : {synapse.health:.6f}")
    print(f"[CAINE] Peak DA       : {chem_traces['dopamine'].max():.4f}")
    print(f"[CAINE] Peak NE       : {chem_traces['norepinephrine'].max():.4f}")
    print(f"[CAINE] Peak ACh      : {chem_traces['acetylcholine'].max():.4f}")
    print(f"[CAINE] Peak STDP x   : {stdp_scale_trace.max():.4f}")
    print("=" * 65)

    return {
        't':               t_array,
        'V_pre':           V_pre_trace,
        'V_post':          V_post_trace,
        'weight':          weight_trace,
        'health':          health_trace,
        'chem_traces':     chem_traces,
        'stdp_scale':      stdp_scale_trace,
        'synapse':         synapse,
        'pre_spikes':      pre_tracker.spike_times,
        'post_spikes':     post_tracker.spike_times,
    }


# ===========================================================================
# SECTION 3 -- PLOTTING
# ===========================================================================

def plot_combined(results: dict) -> None:
    """
    5-panel figure showing all three modules interacting:
      1. Pre-synaptic voltage trace
      2. Post-synaptic voltage trace
      3. Synapse weight (green = potentiation events, red = depression events)
      4. Neurochemical concentrations: DA, NE, ACh
      5. STDP effective scale factor (driven by neurochemicals)
    """
    t         = results['t']
    synapse   = results['synapse']
    chem      = results['chem_traces']

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(
        "CAINE - Combined Demo: HH Neurons + AMPA Synapse + Neurochemical Modulation\n"
        f"Pre drive={I_PRE_AMP} uA/cm2  |  Post drive={I_POST_AMP} uA/cm2  |  "
        f"init_w={INIT_WEIGHT:.2f}  ->  final_w={results['weight'][-1]:.4f}  "
        f"({len(results['pre_spikes'])} pre / {len(results['post_spikes'])} post spikes)",
        fontsize=10, fontweight='bold'
    )

    gs   = gridspec.GridSpec(5, 1, hspace=0.50, top=0.89, bottom=0.06)
    axes = [fig.add_subplot(gs[r]) for r in range(5)]

    # Event time lines (shared across panels)
    event_times = [ev[0] for ev in _NEURO_SCHEDULE]
    event_labels = {
        100.0: 'NOVEL_STIMULUS',
        160.0: 'REWARD',
        220.0: 'DIRECTED_GAZE',
    }

    def _mark_events(ax):
        for t_ev in event_times:
            ax.axvline(t_ev, color='dimgray', linewidth=0.7,
                       linestyle='--', alpha=0.45)

    # -- Panel 1: Pre voltage --
    ax = axes[0]
    ax.plot(t, results['V_pre'], color='steelblue', linewidth=0.6,
            label=f"Pre  ({len(results['pre_spikes'])} spikes)")
    for st in results['pre_spikes']:
        ax.axvline(st, color='steelblue', linewidth=0.4, alpha=0.3)
    _mark_events(ax)
    ax.set_ylim(-90, 65)
    ax.set_ylabel('V (mV)')
    ax.set_title('Presynaptic neuron (neuron 1)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # -- Panel 2: Post voltage --
    ax = axes[1]
    ax.plot(t, results['V_post'], color='tomato', linewidth=0.6,
            label=f"Post ({len(results['post_spikes'])} spikes)")
    for st in results['post_spikes']:
        ax.axvline(st, color='tomato', linewidth=0.4, alpha=0.3)
    _mark_events(ax)
    ax.set_ylim(-90, 65)
    ax.set_ylabel('V (mV)')
    ax.set_title('Postsynaptic neuron (neuron 2) -- sub-threshold drive + AMPA + NE gain', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # -- Panel 3: Synapse weight --
    ax = axes[2]
    ax.plot(t, results['weight'], color='darkorchid', linewidth=1.1,
            label='Synaptic weight', zorder=3)
    ax.axhline(INIT_WEIGHT, color='darkorchid', linewidth=0.7,
               linestyle=':', alpha=0.5, label=f'Initial ({INIT_WEIGHT})')
    for (t_ev, dw, w_after, reason) in synapse.weight_log:
        col = 'limegreen' if 'potentiation' in reason else 'crimson'
        ax.axvline(t_ev, color=col, linewidth=0.9, alpha=0.55, zorder=2)
    _mark_events(ax)
    legend_extra = [
        Line2D([0], [0], color='limegreen', lw=1.2, label='Potentiation (pre->post)'),
        Line2D([0], [0], color='crimson',   lw=1.2, label='Depression  (post->pre)'),
        Line2D([0], [0], color='dimgray',   lw=0.7, linestyle='--', label='Neuro event'),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + legend_extra, l + [e.get_label() for e in legend_extra],
              loc='upper left', fontsize=7, ncol=2)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Weight')
    ax.set_title(f'Synapse weight  ({len(synapse.weight_log)} STDP events, '
                 f'modulated by DA + ACh)', fontsize=9)
    ax.grid(True, alpha=0.2)

    # -- Panel 4: Neurochemicals --
    ax = axes[3]
    chem_plot = [
        ('dopamine',       'Dopamine (DA)',       'royalblue'),
        ('norepinephrine', 'Norepinephrine (NE)', 'darkorange'),
        ('acetylcholine',  'Acetylcholine (ACh)', 'steelblue'),
    ]
    for name, label, color in chem_plot:
        ax.plot(t, chem[name], color=color, linewidth=1.0, label=label)
    _mark_events(ax)
    # Annotate events
    for t_ev, label in event_labels.items():
        ax.text(t_ev + 2, 0.92, label, fontsize=6.5, color='dimgray',
                rotation=90, va='top', ha='left')
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel('Concentration')
    ax.set_title('Neurochemical concentrations  (Module 3 -> Module 2 coupling)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # -- Panel 5: STDP scale factor --
    ax = axes[4]
    ax.plot(t, results['stdp_scale'], color='purple', linewidth=1.0,
            label='STDP effective scale (DA + ACh + OT)')
    ax.axhline(1.0, color='gray', linewidth=0.7, linestyle='--', alpha=0.6,
               label='Baseline (1.0)')
    _mark_events(ax)
    ax.set_ylim(0.0, 3.2)
    ax.set_ylabel('Scale')
    ax.set_xlabel('Time (ms)')
    ax.set_title('STDP scale factor  (1.0 = baseline; rises with DA/ACh events)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Shared x limits
    for ax in axes:
        ax.set_xlim(0, DURATION_MS)
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    out_path = os.path.join(OUTPUT_DIR, 'caine_combined_demo.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n[CAINE] Plot saved -> output/caine_combined_demo.png')
    plt.show()


# ===========================================================================
# SECTION 4 -- ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    results = run_combined(
        duration_ms = DURATION_MS,
        dt_ms       = DT_MS,
    )
    plot_combined(results)
