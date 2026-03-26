"""
CAINE - Module 3: Neurochemical System
=======================================
Simulates CAINE's neuromodulatory system as six coupled differential
equations. These are NOT scalar reward signals -- they are dynamic
chemical concentrations that rise on biological events and decay
exponentially toward individual baselines, directly modulating neuron
firing thresholds and STDP learning rates across the whole brain.

Chemicals modelled:
  dopamine        (DA)   -- reward prediction error, approach/exploration
  serotonin       (5HT)  -- social reinforcement, synaptic stability
  cortisol        (CORT) -- stress, accelerated pruning, memory impairment
  oxytocin        (OT)   -- social bonding, STG STDP scaling
  norepinephrine  (NE)   -- arousal/attention, global gain increase
  acetylcholine   (ACh)  -- learning gate (STDP near-zero when ACh is low)

Core ODE (per chemical C with baseline B and time-constant tau):
  dC/dt = sum(release_i * event_i) - (1/tau) * (C - B)

Solved exactly each timestep:
  C(t+dt) = B + (C(t) - B) * exp(-dt/tau) + sum(pulse_releases)

Modulation outputs (consumed by Modules 1 & 2):
  stdp_scale()          -- multiplier on STDP A+/A-  (DA + ACh + OT)
  learning_gate()       -- ACh threshold gate; STDP -> ~0 when gate < 0.1
  health_decay_mod()    -- synapse health tau modifier (5HT slows, CORT speeds)
  global_gain()         -- neuron threshold shift (NE raises sensitivity)
  memory_gate()         -- hippocampal encoding gate (suppressed by CORT)

Integrates cleanly with Module 2's Synapse via modulate_synapse().

No ML frameworks -- only numpy.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

# All plots go to output/ at the project root
_OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# SECTION 1 -- EVENT TYPES
# Every biological event that triggers a chemical release is represented
# as a member of this enum. The NeurochemicalSystem maps each event to
# a set of (chemical, release_amount) pairs.
# ===========================================================================

class EventType(Enum):
    # --- Reward / prediction ---
    NOVEL_STIMULUS            = auto()  # DA +, NE +, ACh +
    REWARD                    = auto()  # DA ++ (better than predicted)
    REWARD_OMISSION           = auto()  # DA -- (predicted reward absent)

    # --- Social / communication ---
    SOCIAL_POSITIVE           = auto()  # OT +, 5HT +
    VOICE_MATCH               = auto()  # OT ++ (parent voice recognised)
    COMMUNICATION_SUCCESS     = auto()  # 5HT +, OT +
    VOCALIZATION_SUCCESS      = auto()  # 5HT +

    # --- Stress / threat ---
    THREAT                    = auto()  # CORT ++
    MOTOR_FAILURE             = auto()  # CORT +
    PREDICTION_ERROR_NEG      = auto()  # CORT + (negative surprise)
    AMYGDALA_BLA              = auto()  # CORT ++ (direct amygdala activation)

    # --- Attention / learning readiness ---
    NOVEL_ENVIRONMENT         = auto()  # ACh ++
    DIRECTED_GAZE             = auto()  # ACh +

    # --- Arousal ---
    STARTLE                   = auto()  # NE ++
    ACC_CONFLICT              = auto()  # NE + (anterior cingulate conflict)


# ===========================================================================
# SECTION 2 -- EVENT DATACLASS
# A timestamped event injected into the system each update cycle.
# Multiple events can arrive in a single update call.
# ===========================================================================

@dataclass
class NeurochemicalEvent:
    """
    A single biological event that triggers neuromodulator release.

    Parameters
    ----------
    event_type : EventType
    magnitude  : float in [0.0, 1.0] -- scales the release amount.
                 1.0 = full canonical release; 0.5 = half-intensity.
    """
    event_type: EventType
    magnitude:  float = 1.0


# ===========================================================================
# SECTION 3 -- CHEMICAL PROFILE
# Internal dataclass holding per-chemical parameters and state.
# ===========================================================================

@dataclass
class ChemicalProfile:
    """
    State and kinetics for one neuromodulator.

    concentration : current level [0.0 - 1.0]
    baseline      : resting level that concentration decays toward
    tau_ms        : decay time constant (ms) -- governs how long a pulse lasts
    name          : human-readable label
    color         : plot colour string
    """
    name:          str
    baseline:      float
    tau_ms:        float
    color:         str
    concentration: float = field(init=False)

    def __post_init__(self):
        self.concentration = self.baseline


# ===========================================================================
# SECTION 4 -- RELEASE TABLE
# Maps each EventType to {chemical_attribute: release_amount}.
# release_amount can be negative (e.g. DA suppression on reward omission).
# Amounts are added as impulse pulses on top of the decaying concentration.
# ===========================================================================

# Attribute names match NeurochemicalSystem fields exactly.
_RELEASE_TABLE: dict = {
    EventType.NOVEL_STIMULUS:        {'dopamine': 0.30, 'norepinephrine': 0.25, 'acetylcholine': 0.35},
    EventType.REWARD:                {'dopamine': 0.50},
    EventType.REWARD_OMISSION:       {'dopamine': -0.30},   # dips below baseline
    EventType.SOCIAL_POSITIVE:       {'oxytocin':  0.40, 'serotonin': 0.25},
    EventType.VOICE_MATCH:           {'oxytocin':  0.50},
    EventType.COMMUNICATION_SUCCESS: {'serotonin': 0.35, 'oxytocin': 0.25},
    EventType.VOCALIZATION_SUCCESS:  {'serotonin': 0.25},
    EventType.THREAT:                {'cortisol':  0.55},
    EventType.MOTOR_FAILURE:         {'cortisol':  0.35},
    EventType.PREDICTION_ERROR_NEG:  {'cortisol':  0.30},
    EventType.AMYGDALA_BLA:          {'cortisol':  0.60},
    EventType.NOVEL_ENVIRONMENT:     {'acetylcholine': 0.50},
    EventType.DIRECTED_GAZE:         {'acetylcholine': 0.30},
    EventType.STARTLE:               {'norepinephrine': 0.55, 'acetylcholine': 0.20},
    EventType.ACC_CONFLICT:          {'norepinephrine': 0.35},
}

# NE co-releases ACh -- when norepinephrine is released, a fraction spills
# into acetylcholine (README: "boosts acetylcholine co-release").
_NE_ACH_COSPILL = 0.20   # fraction of NE release that also boosts ACh


# ===========================================================================
# SECTION 5 -- NEUROCHEMICAL SYSTEM
# ===========================================================================

class NeurochemicalSystem:
    """
    CAINE's neuromodulatory system.

    Six coupled ODEs, one per chemical. Each update step:
      1. Decay all concentrations exponentially toward their baselines.
      2. Process incoming events -- add instantaneous release pulses.
      3. Apply NE -> ACh co-spill.
      4. Clip all concentrations to [0.0, 1.0].

    Modulation methods return scalar multipliers/modifiers that
    Modules 1 and 2 consume to adjust their dynamics.
    """

    def __init__(self):
        # Six chemical profiles -- parameters grounded in biological literature.
        # tau values are intentionally varied: fast phasic signals (DA, ACh)
        # vs. slow tonic modulators (CORT, OT).
        self._chemicals: dict[str, ChemicalProfile] = {
            'dopamine':       ChemicalProfile('Dopamine',       baseline=0.10, tau_ms=800.0,   color='royalblue'),
            'serotonin':      ChemicalProfile('Serotonin',      baseline=0.10, tau_ms=3000.0,  color='mediumseagreen'),
            'cortisol':       ChemicalProfile('Cortisol',       baseline=0.08, tau_ms=8000.0,  color='tomato'),
            'oxytocin':       ChemicalProfile('Oxytocin',       baseline=0.08, tau_ms=4000.0,  color='orchid'),
            'norepinephrine': ChemicalProfile('Norepinephrine', baseline=0.10, tau_ms=1200.0,  color='darkorange'),
            'acetylcholine':  ChemicalProfile('Acetylcholine',  baseline=0.12, tau_ms=300.0,   color='steelblue'),
        }

        # Log of all events for post-hoc analysis: (time_ms, EventType, magnitude)
        self.event_log: List[tuple] = []

    # ------------------------------------------------------------------
    # 5.1  Property accessors -- give clean attribute-style access
    # ------------------------------------------------------------------

    @property
    def dopamine(self)       -> float: return self._chemicals['dopamine'].concentration
    @property
    def serotonin(self)      -> float: return self._chemicals['serotonin'].concentration
    @property
    def cortisol(self)       -> float: return self._chemicals['cortisol'].concentration
    @property
    def oxytocin(self)       -> float: return self._chemicals['oxytocin'].concentration
    @property
    def norepinephrine(self) -> float: return self._chemicals['norepinephrine'].concentration
    @property
    def acetylcholine(self)  -> float: return self._chemicals['acetylcholine'].concentration

    # ------------------------------------------------------------------
    # 5.2  Core update -- advance one timestep
    # ------------------------------------------------------------------

    def update(self, dt: float, events: Optional[List[NeurochemicalEvent]] = None,
               current_time: float = 0.0) -> None:
        """
        Advance the neurochemical system by dt milliseconds.

        Parameters
        ----------
        dt           : timestep in ms
        events       : list of NeurochemicalEvent objects occurring this step
        current_time : simulation time (ms), used only for logging
        """
        # --- Step 1: Exponential decay toward each chemical's baseline ---
        # Exact solution of dC/dt = -(1/tau)(C - B):
        #   C(t+dt) = B + (C(t) - B) * exp(-dt/tau)
        for chem in self._chemicals.values():
            chem.concentration = (
                chem.baseline
                + (chem.concentration - chem.baseline) * np.exp(-dt / chem.tau_ms)
            )

        # --- Step 2: Process event pulses ---
        if events:
            # Accumulate NE releases this step for co-spill calculation
            ne_release_total = 0.0

            for event in events:
                self.event_log.append((current_time, event.event_type, event.magnitude))
                releases = _RELEASE_TABLE.get(event.event_type, {})
                for chem_name, amount in releases.items():
                    self._chemicals[chem_name].concentration += amount * event.magnitude
                    if chem_name == 'norepinephrine':
                        ne_release_total += amount * event.magnitude

            # --- Step 3: NE -> ACh co-spill ---
            # Norepinephrine release co-activates basal forebrain cholinergic
            # neurons, flagging high-stakes moments as priority for consolidation.
            if ne_release_total > 0:
                self._chemicals['acetylcholine'].concentration += (
                    ne_release_total * _NE_ACH_COSPILL
                )

        # --- Step 4: Clip all to [0.0, 1.0] ---
        for chem in self._chemicals.values():
            chem.concentration = float(np.clip(chem.concentration, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 5.3  Modulation outputs
    # These are the values Module 1 (neurons) and Module 2 (synapses) read.
    # ------------------------------------------------------------------

    def stdp_scale(self) -> float:
        """
        Combined STDP amplitude multiplier.

        Driven by dopamine (reward learning), acetylcholine (attention gate),
        and oxytocin (social region booster).

        At baseline levels this returns ~1.0 (no change).
        High DA + ACh -> strong potentiation; low ACh -> near-zero STDP.

        Formula:
          scale = learning_gate * (1 + DA_gain*(DA - DA_base) + OT_gain*(OT - OT_base))
        """
        da_base  = self._chemicals['dopamine'].baseline
        ot_base  = self._chemicals['oxytocin'].baseline

        da_contribution = 1.5 * (self.dopamine       - da_base)   # DA scales A+
        ot_contribution = 0.8 * (self.oxytocin        - ot_base)  # OT boosts social STDP

        raw_scale = 1.0 + da_contribution + ot_contribution
        return float(np.clip(raw_scale * self.learning_gate(), 0.0, 3.0))

    def learning_gate(self) -> float:
        """
        ACh-dependent learning gate.

        Without acetylcholine above a minimum threshold, STDP is suppressed
        to near-zero -- CAINE must be attending to learn.

        Gate is a sigmoid centred at the ACh threshold (~0.15):
          gate(ACh) = sigmoid( (ACh - threshold) * steepness )
        Returns a value in [0.0, 1.0].
        """
        ach_threshold = 0.15
        steepness     = 20.0
        return float(1.0 / (1.0 + np.exp(-steepness * (self.acetylcholine - ach_threshold))))

    def health_decay_mod(self) -> float:
        """
        Modifier on synaptic health decay time-constant (tau_health).

        Serotonin SLOWS decay (stabilises connections):
          tau_eff = tau_base * (1 + 5HT_gain * serotonin)

        Cortisol SPEEDS decay (accelerates pruning):
          tau_eff /= (1 + CORT_gain * cortisol)

        Returns a positive multiplier on tau_health.
        Values > 1.0 -> slower pruning (5HT dominant).
        Values < 1.0 -> faster pruning (CORT dominant).
        """
        serotonin_stabilise = 1.0 + 2.0 * self.serotonin          # max ~3x slower
        cortisol_destabilise = 1.0 + 3.0 * self.cortisol          # max ~4x faster
        return float(np.clip(serotonin_stabilise / cortisol_destabilise, 0.1, 5.0))

    def global_gain(self) -> float:
        """
        Norepinephrine-driven global gain multiplier on neuron excitability.

        NE lowers effective firing threshold, making all neurons more
        responsive. Modelled as a multiplicative gain on input current.

        gain = 1.0 + NE_gain * (norepinephrine - NE_baseline)
        At baseline NE this is 1.0 (no change).
        """
        ne_base = self._chemicals['norepinephrine'].baseline
        gain    = 1.0 + 2.0 * (self.norepinephrine - ne_base)
        return float(np.clip(gain, 0.5, 3.0))

    def memory_gate(self) -> float:
        """
        Hippocampal CA1 encoding gate.

        Sustained high cortisol suppresses new memory formation.
        Gate is 1.0 (full encoding) at low cortisol; approaches 0 at high cortisol.

        gate = 1 - cortisol_gain * cortisol
        """
        return float(np.clip(1.0 - 1.2 * self.cortisol, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 5.4  Integration with Module 2: modulate a Synapse object in-place
    # ------------------------------------------------------------------

    def modulate_synapse(self, synapse) -> None:
        """
        Apply current neurochemical state to a Module 2 Synapse object.

        Adjusts:
          - STDP amplitude (via stdp_scale) -- stored as synapse.neuro_stdp_scale
          - Health decay tau multiplier     -- stored as synapse.neuro_health_mod
          - Learning gate value             -- stored as synapse.neuro_gate

        The Synapse class doesn't know about neurochemicals directly; the
        caller should multiply STDP_A_PLUS/MINUS by synapse.neuro_stdp_scale
        and multiply HEALTH_TAU by synapse.neuro_health_mod before each update.

        If the synapse doesn't have these attributes yet, they are created here.
        """
        synapse.neuro_stdp_scale = self.stdp_scale()
        synapse.neuro_health_mod = self.health_decay_mod()
        synapse.neuro_gate       = self.learning_gate()

    # ------------------------------------------------------------------
    # 5.5  State snapshot for logging
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a dict of all current concentrations (for logging/recording)."""
        return {name: chem.concentration
                for name, chem in self._chemicals.items()}

    def print_state(self, t: float = 0.0) -> None:
        """Print a formatted one-line state summary to stdout."""
        s = self.snapshot()
        print(
            f"[NEURO] t={t:8.1f}ms  "
            f"DA={s['dopamine']:.3f}  "
            f"5HT={s['serotonin']:.3f}  "
            f"CORT={s['cortisol']:.3f}  "
            f"OT={s['oxytocin']:.3f}  "
            f"NE={s['norepinephrine']:.3f}  "
            f"ACh={s['acetylcholine']:.3f}  "
            f"| gate={self.learning_gate():.3f}  "
            f"stdp_x={self.stdp_scale():.3f}  "
            f"health_mod={self.health_decay_mod():.3f}"
        )


# ===========================================================================
# SECTION 6 -- SIMULATION RUNNER
# Drives the system through a scripted sequence of biological events to
# demonstrate realistic chemical dynamics before building higher modules.
# ===========================================================================

def run_neurochemical_simulation(
    duration_ms: float = 15000.0,
    dt_ms:       float = 1.0,        # 1ms steps are fine for slow chemistry
) -> tuple:
    """
    Run a scripted neurochemical scenario over `duration_ms` milliseconds.

    The event schedule mirrors CAINE's anticipated early developmental
    experience: curiosity, reward, social contact, stress, recovery.

    Returns
    -------
    t_array  : time vector (ms)
    traces   : dict of chemical name -> concentration array
    mods     : dict of modulation output name -> value array
    system   : the NeurochemicalSystem after simulation
    schedule : list of (time_ms, EventType, magnitude) tuples
    """

    # -----------------------------------------------------------------------
    # 6.1  Event schedule (time_ms, EventType, magnitude)
    # -----------------------------------------------------------------------
    schedule = [
        # --- Early exploration ---
        (1000.0,  EventType.NOVEL_STIMULUS,        1.0),   # first novel stimulus
        (1000.0,  EventType.NOVEL_ENVIRONMENT,     0.8),   # new place
        (2200.0,  EventType.DIRECTED_GAZE,         0.9),   # notices something
        (3000.0,  EventType.REWARD,                1.0),   # positive outcome
        (3500.0,  EventType.NOVEL_STIMULUS,        0.6),   # another novelty
        # --- Social encounter ---
        (4500.0,  EventType.VOICE_MATCH,           1.0),   # parent voice recognised
        (4500.0,  EventType.SOCIAL_POSITIVE,       0.9),
        (5200.0,  EventType.COMMUNICATION_SUCCESS, 1.0),   # successful exchange
        (5800.0,  EventType.VOCALIZATION_SUCCESS,  0.8),
        # --- Stress episode ---
        (7000.0,  EventType.STARTLE,               1.0),   # sudden noise
        (7200.0,  EventType.THREAT,                0.8),   # perceived danger
        (7800.0,  EventType.MOTOR_FAILURE,         0.7),   # stumbles
        (8000.0,  EventType.AMYGDALA_BLA,          0.6),   # amygdala peaks
        # --- Recovery + reward omission ---
        (9500.0,  EventType.ACC_CONFLICT,          0.7),   # uncertainty
        (10000.0, EventType.REWARD_OMISSION,       1.0),   # expected reward absent
        (10500.0, EventType.PREDICTION_ERROR_NEG,  0.8),
        # --- Re-engagement ---
        (12000.0, EventType.NOVEL_STIMULUS,        0.7),
        (12500.0, EventType.DIRECTED_GAZE,         1.0),
        (13000.0, EventType.SOCIAL_POSITIVE,       0.9),
        (13500.0, EventType.COMMUNICATION_SUCCESS, 1.0),
        (14000.0, EventType.REWARD,                0.9),
    ]

    # Build a fast lookup: time_ms -> [events at that ms]
    event_map: dict = {}
    for t_ev, ev_type, mag in schedule:
        t_key = round(t_ev / dt_ms) * dt_ms   # snap to nearest dt
        event_map.setdefault(t_key, []).append(NeurochemicalEvent(ev_type, mag))

    # -----------------------------------------------------------------------
    # 6.2  Initialise
    # -----------------------------------------------------------------------
    system  = NeurochemicalSystem()
    t_array = np.arange(0.0, duration_ms + dt_ms, dt_ms)
    N       = len(t_array)

    chem_names = ['dopamine', 'serotonin', 'cortisol',
                  'oxytocin', 'norepinephrine', 'acetylcholine']
    mod_names  = ['stdp_scale', 'learning_gate',
                  'health_decay_mod', 'global_gain', 'memory_gate']

    traces = {name: np.empty(N) for name in chem_names}
    mods   = {name: np.empty(N) for name in mod_names}

    print("=" * 70)
    print("CAINE - Module 3: Neurochemical System")
    print("=" * 70)
    print(f"  Duration  : {duration_ms:.0f} ms  |  dt={dt_ms} ms")
    print(f"  Events    : {len(schedule)} scripted events")
    print(f"  Chemicals : {', '.join(chem_names)}")
    print("-" * 70)

    # -----------------------------------------------------------------------
    # 6.3  Integration loop
    # -----------------------------------------------------------------------
    for i, t in enumerate(t_array):
        events_now = event_map.get(t, None)

        # Log events to stdout
        if events_now:
            for ev in events_now:
                print(f"[EVENT] t={t:8.1f} ms  {ev.event_type.name:<26}  "
                      f"magnitude={ev.magnitude:.2f}")

        system.update(dt_ms, events=events_now, current_time=t)

        # Record traces
        snap = system.snapshot()
        for name in chem_names:
            traces[name][i] = snap[name]

        # Record modulation outputs
        mods['stdp_scale'][i]       = system.stdp_scale()
        mods['learning_gate'][i]    = system.learning_gate()
        mods['health_decay_mod'][i] = system.health_decay_mod()
        mods['global_gain'][i]      = system.global_gain()
        mods['memory_gate'][i]      = system.memory_gate()

    # -----------------------------------------------------------------------
    # 6.4  Summary
    # -----------------------------------------------------------------------
    print("-" * 70)
    system.print_state(t=duration_ms)
    print(f"[NEURO] Peak dopamine       : {traces['dopamine'].max():.4f}")
    print(f"[NEURO] Peak cortisol       : {traces['cortisol'].max():.4f}")
    print(f"[NEURO] Peak oxytocin       : {traces['oxytocin'].max():.4f}")
    print(f"[NEURO] Min learning gate   : {mods['learning_gate'].min():.4f}")
    print(f"[NEURO] Min health decay mod: {mods['health_decay_mod'].min():.4f}")
    print("=" * 70)

    return t_array, traces, mods, system, schedule


# ===========================================================================
# SECTION 7 -- PLOTTING
# ===========================================================================

def plot_neurochemical(t, traces, mods, schedule):
    """
    Two-figure output:

    Figure 1 -- Chemical traces (3 rows x 2 cols):
      Row 1: Dopamine | Serotonin
      Row 2: Cortisol | Oxytocin
      Row 3: Norepinephrine | Acetylcholine
      Event timings overlaid as labelled vertical lines.

    Figure 2 -- Derived modulation outputs (5 panels):
      STDP scale | Learning gate | Health decay mod | Global gain | Memory gate
    """

    # Convert schedule to seconds for readable x-axis ticks
    t_s = t / 1000.0

    chem_layout = [
        ('dopamine',       'Dopamine (DA)',        'royalblue'),
        ('serotonin',      'Serotonin (5HT)',      'mediumseagreen'),
        ('cortisol',       'Cortisol (CORT)',      'tomato'),
        ('oxytocin',       'Oxytocin (OT)',        'orchid'),
        ('norepinephrine', 'Norepinephrine (NE)',  'darkorange'),
        ('acetylcholine',  'Acetylcholine (ACh)',  'steelblue'),
    ]

    # -----------------------------------------------------------------
    # Figure 1: Chemical concentrations
    # -----------------------------------------------------------------
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig1.suptitle("CAINE - Module 3: Neurochemical Concentrations",
                  fontsize=12, fontweight='bold')

    ax_flat = axes1.flatten()
    for idx, (name, label, color) in enumerate(chem_layout):
        ax = ax_flat[idx]
        baseline = traces[name][0]   # first value is baseline
        ax.plot(t_s, traces[name], color=color, linewidth=1.0, label=label)
        ax.axhline(baseline, color=color, linewidth=0.7, linestyle=':', alpha=0.5,
                   label=f'baseline ({baseline:.2f})')

        # Mark event times with subtle vertical lines, coloured by chemical relevance
        for (t_ev, ev_type, mag) in schedule:
            ax.axvline(t_ev / 1000.0, color='gray', linewidth=0.5, alpha=0.25)

        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel('Concentration')
        ax.set_title(label, fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.2)

    for ax in axes1[2]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    fig1.savefig(os.path.join(_OUTPUT_DIR, 'caine_module3_chemicals.png'), dpi=150, bbox_inches='tight')
    print('[NEURO] Plot saved -> output/caine_module3_chemicals.png')

    # -----------------------------------------------------------------
    # Figure 2: Modulation outputs
    # -----------------------------------------------------------------
    mod_layout = [
        ('stdp_scale',       'STDP Scale Factor',
         '(1.0=baseline)  DA + ACh + OT combined', 'purple', [0.0, 3.0]),
        ('learning_gate',    'Learning Gate (ACh)',
         'STDP suppressed below ~0.5', 'steelblue', [-0.05, 1.05]),
        ('health_decay_mod', 'Health Decay Modifier',
         '>1 = slower pruning (5HT), <1 = faster pruning (CORT)', 'goldenrod', [0.0, 5.0]),
        ('global_gain',      'Global Gain (NE)',
         'Multiplier on neuron excitability', 'darkorange', [0.4, 3.1]),
        ('memory_gate',      'Memory Gate (CA1 encoding)',
         'Suppressed by high cortisol', 'tomato', [-0.05, 1.05]),
    ]

    fig2, axes2 = plt.subplots(len(mod_layout), 1, figsize=(14, 11),
                               sharex=True)
    fig2.suptitle("CAINE - Module 3: Neurochemical Modulation Outputs",
                  fontsize=12, fontweight='bold')

    for ax, (key, title, subtitle, color, ylim) in zip(axes2, mod_layout):
        ax.plot(t_s, mods[key], color=color, linewidth=1.1)
        ax.axhline(1.0, color='gray', linewidth=0.7, linestyle='--', alpha=0.6)

        # Shade stressed region (cortisol dominant) in faint red
        if key == 'health_decay_mod':
            ax.fill_between(t_s, 1.0, mods[key],
                            where=(mods[key] < 1.0),
                            alpha=0.15, color='tomato', label='faster pruning')
            ax.fill_between(t_s, 1.0, mods[key],
                            where=(mods[key] >= 1.0),
                            alpha=0.15, color='mediumseagreen', label='slower pruning')
            ax.legend(loc='upper right', fontsize=7)

        # Mark event lines
        for (t_ev, ev_type, mag) in schedule:
            ax.axvline(t_ev / 1000.0, color='gray', linewidth=0.5, alpha=0.25)

        ax.set_ylim(ylim)
        ax.set_title(f'{title}  --  {subtitle}', fontsize=8.5)
        ax.set_ylabel(title.split(' ')[0])
        ax.grid(True, alpha=0.2)

    axes2[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fig2.savefig(os.path.join(_OUTPUT_DIR, 'caine_module3_modulation.png'), dpi=150, bbox_inches='tight')
    print('[NEURO] Plot saved -> output/caine_module3_modulation.png')

    plt.show()


# ===========================================================================
# SECTION 8 -- ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    t, traces, mods, system, schedule = run_neurochemical_simulation(
        duration_ms = 15000.0,
        dt_ms       = 1.0,
    )

    plot_neurochemical(t, traces, mods, schedule)
