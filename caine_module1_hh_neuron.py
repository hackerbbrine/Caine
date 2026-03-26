"""
CAINE - Module 1: Hodgkin-Huxley Neuron
========================================
A biologically-accurate spiking neuron model based on the 1952 Hodgkin-Huxley
equations. This models a single patch of squid giant axon membrane.

The four coupled ODEs describe:
  1. Membrane voltage (V)
  2. Na+ activation gate (m)
  3. Na+ inactivation gate (h)
  4. K+ activation gate (n)
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Membrane & channel parameters (standard HH values, units: mS/cm², mV, µF/cm²)
# ---------------------------------------------------------------------------
Cm  = 1.0    # membrane capacitance (µF/cm²)
gNa = 120.0  # max Na+ conductance (mS/cm²)
gK  = 36.0   # max K+ conductance  (mS/cm²)
gL  = 0.3    # leak conductance    (mS/cm²)
ENa = 50.0   # Na+ reversal potential (mV)
EK  = -77.0  # K+  reversal potential (mV)
EL  = -54.4  # leak reversal potential (mV)

# Resting membrane potential (mV) — HH use V=0 as rest, we shift to biological units
V_rest = -65.0


# ---------------------------------------------------------------------------
# Rate functions (alpha/beta) for each gating variable
# These are empirical fits from Hodgkin & Huxley's voltage-clamp experiments.
# All voltages are in mV relative to resting potential (HH convention: V_shift).
# ---------------------------------------------------------------------------

def alpha_m(V):
    """Na+ activation: fast opening rate."""
    dV = V - V_rest
    # Avoid division by zero at dV = 25
    return 0.1 * (25.0 - dV) / (np.exp((25.0 - dV) / 10.0) - 1.0 + 1e-12)

def beta_m(V):
    """Na+ activation: closing rate."""
    dV = V - V_rest
    return 4.0 * np.exp(-dV / 18.0)

def alpha_h(V):
    """Na+ inactivation: slow inactivation onset rate."""
    dV = V - V_rest
    return 0.07 * np.exp(-dV / 20.0)

def beta_h(V):
    """Na+ inactivation: recovery rate."""
    dV = V - V_rest
    return 1.0 / (np.exp((30.0 - dV) / 10.0) + 1.0)

def alpha_n(V):
    """K+ activation: slow opening rate."""
    dV = V - V_rest
    return 0.01 * (10.0 - dV) / (np.exp((10.0 - dV) / 10.0) - 1.0 + 1e-12)

def beta_n(V):
    """K+ activation: closing rate."""
    dV = V - V_rest
    return 0.125 * np.exp(-dV / 80.0)


# ---------------------------------------------------------------------------
# Steady-state gate values at rest (used to initialise the simulation)
# ---------------------------------------------------------------------------

def gate_steady_state(V):
    """Return (m∞, h∞, n∞) — equilibrium gate probabilities at voltage V."""
    am, bm = alpha_m(V), beta_m(V)
    ah, bh = alpha_h(V), beta_h(V)
    an, bn = alpha_n(V), beta_n(V)
    m_inf = am / (am + bm)
    h_inf = ah / (ah + bh)
    n_inf = an / (an + bn)
    return m_inf, h_inf, n_inf


# ---------------------------------------------------------------------------
# The four coupled ODEs  — the heart of the model
# ---------------------------------------------------------------------------

def hh_odes(state, t, I_ext_func):
    """
    Hodgkin-Huxley system of ODEs.

    state : [V, m, h, n]
    t     : current time (ms)
    I_ext_func : callable(t) → injected current (µA/cm²)

    Returns d/dt [V, m, h, n]
    """
    V, m, h, n = state

    # --- Ionic currents (Ohm's law: I = g * (V - E_rev)) ---
    I_Na = gNa * m**3 * h * (V - ENa)   # fast inward Na+ (depolarising)
    I_K  = gK  * n**4     * (V - EK)    # delayed outward K+ (repolarising)
    I_L  = gL             * (V - EL)    # passive leak

    # --- Membrane voltage ODE: Cm * dV/dt = I_ext - sum(ionic currents) ---
    dVdt = (I_ext_func(t) - I_Na - I_K - I_L) / Cm

    # --- Gate variable ODEs: dx/dt = α(V)*(1-x) - β(V)*x ---
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n

    return [dVdt, dmdt, dhdt, dndt]


# ---------------------------------------------------------------------------
# Spike detection + refractory period tracking
# ---------------------------------------------------------------------------

def detect_spikes(t, V, threshold=0.0, refractory_ms=2.0):
    """
    Find action potential peaks in voltage trace.

    A spike is counted when V crosses `threshold` (mV) from below, provided
    at least `refractory_ms` have elapsed since the last spike.

    Returns
    -------
    spike_times : list of float  — times (ms) of detected spikes
    spike_log   : list of dict   — timestamped log entries
    """
    spike_times = []
    spike_log   = []
    last_spike  = -np.inf

    for i in range(1, len(V)):
        crossed_threshold = V[i - 1] < threshold <= V[i]
        past_refractory   = (t[i] - last_spike) > refractory_ms

        if crossed_threshold and past_refractory:
            spike_times.append(t[i])
            last_spike = t[i]
            spike_log.append({
                "time_ms"   : round(float(t[i]), 4),
                "V_mV"      : round(float(V[i]), 2),
                "spike_num" : len(spike_times),
            })

    return spike_times, spike_log


# ---------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------

def run_simulation(
    duration_ms: float = 100.0,
    dt_ms: float = 0.01,
    I_ext_amplitude: float = 10.0,   # µA/cm²  (>~7 triggers spikes)
    I_start_ms: float = 10.0,
    I_stop_ms: float = 90.0,
):
    """
    Simulate a single HH neuron receiving a step-current injection.

    Parameters
    ----------
    duration_ms       : total simulation time
    dt_ms             : time step
    I_ext_amplitude   : injected current strength (µA/cm²)
    I_start_ms        : when the current pulse starts
    I_stop_ms         : when the current pulse ends

    Returns
    -------
    t        : time array (ms)
    V        : membrane voltage array (mV)
    states   : full state matrix [V, m, h, n] over time
    spikes   : (spike_times, spike_log)
    """
    # Time vector
    t = np.arange(0.0, duration_ms + dt_ms, dt_ms)

    # Step-current stimulus
    def I_ext(time):
        return I_ext_amplitude if I_start_ms <= time <= I_stop_ms else 0.0

    # Initial conditions: resting state
    m0, h0, n0 = gate_steady_state(V_rest)
    state0 = [V_rest, m0, h0, n0]

    print(f"[CAINE] Resting gate values → m={m0:.4f}, h={h0:.4f}, n={n0:.4f}")
    print(f"[CAINE] Simulating {duration_ms} ms | I_ext={I_ext_amplitude} µA/cm² "
          f"({I_start_ms}–{I_stop_ms} ms) ...")

    # Integrate the ODEs
    states = odeint(hh_odes, state0, t, args=(I_ext,))

    V = states[:, 0]   # membrane voltage
    m = states[:, 1]   # Na+ activation
    h = states[:, 2]   # Na+ inactivation
    n = states[:, 3]   # K+  activation

    # Detect spikes
    spike_times, spike_log = detect_spikes(t, V)

    print(f"[CAINE] Detected {len(spike_times)} action potential(s)")
    if spike_times:
        isi = np.diff(spike_times)
        print(f"[CAINE] Mean ISI: {isi.mean():.2f} ms | "
              f"Firing rate: {len(spike_times) / (duration_ms / 1000):.1f} Hz"
              if len(isi) > 0 else "[CAINE] Single spike — no ISI")

    # Print spike log
    print("\n--- Spike Log ---")
    if spike_log:
        for entry in spike_log:
            print(f"  Spike #{entry['spike_num']:3d}  t={entry['time_ms']:8.3f} ms  "
                  f"V={entry['V_mV']:+7.2f} mV")
    else:
        print("  (no spikes detected — try increasing I_ext_amplitude)")

    return t, V, states, (spike_times, spike_log)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(t, V, states, spike_times, I_ext_amplitude, I_start_ms, I_stop_ms):
    """Render voltage trace + gating variables + stimulus."""
    m = states[:, 1]
    h = states[:, 2]
    n = states[:, 3]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("CAINE — Module 1: Hodgkin-Huxley Neuron", fontsize=14, fontweight="bold")

    # -- Panel 1: Membrane voltage --
    ax1 = axes[0]
    ax1.plot(t, V, color="steelblue", linewidth=1.2, label="V (mV)")
    for st in spike_times:
        ax1.axvline(st, color="red", linewidth=0.7, alpha=0.5)
    ax1.axhline(0.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax1.set_ylabel("Membrane\nvoltage (mV)")
    ax1.set_ylim(-90, 60)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"{len(spike_times)} action potentials  |  "
                  f"I_ext = {I_ext_amplitude} µA/cm²  ({I_start_ms}–{I_stop_ms} ms)")
    ax1.grid(True, alpha=0.3)

    # -- Panel 2: Gating variables --
    ax2 = axes[1]
    ax2.plot(t, m, color="tomato",      linewidth=1.0, label="m  (Na⁺ activation)")
    ax2.plot(t, h, color="darkorange",  linewidth=1.0, label="h  (Na⁺ inactivation)")
    ax2.plot(t, n, color="seagreen",    linewidth=1.0, label="n  (K⁺ activation)")
    ax2.set_ylabel("Gate\nprobability")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # -- Panel 3: Injected current --
    ax3 = axes[2]
    I_trace = np.where((t >= I_start_ms) & (t <= I_stop_ms), I_ext_amplitude, 0.0)
    ax3.plot(t, I_trace, color="slateblue", linewidth=1.5, label="I_ext (µA/cm²)")
    ax3.set_ylabel("Injected\ncurrent (µA/cm²)")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylim(-1, I_ext_amplitude * 1.4)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("caine_hh_trace.png", dpi=150, bbox_inches="tight")
    print("\n[CAINE] Plot saved → caine_hh_trace.png")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t, V, states, (spike_times, spike_log) = run_simulation(
        duration_ms      = 100.0,
        dt_ms            = 0.01,
        I_ext_amplitude  = 10.0,   # µA/cm²
        I_start_ms       = 10.0,
        I_stop_ms        = 90.0,
    )

    plot_results(t, V, states, spike_times,
                 I_ext_amplitude=10.0,
                 I_start_ms=10.0,
                 I_stop_ms=90.0)
