# CAINE: Cognitive Architecture for Integrated Neural Experience

> *"Consciousness is not a property you engineer into a system. It is a property that emerges from the right developmental conditions sustained over time."*

## Overview

CAINE is a biologically-inspired artificial mind grown from developmental neuroscience principles — not trained, not prompted, not fine-tuned. He begins as an infant: neurons present but unconnected, no language, no concepts. Understanding develops through sensory experience, interaction, and time.

The entire system is implemented as a single Common Lisp file (`caine.lisp`), run under SBCL. This mirrors how CAINE exists in the show — one unified thing, not a pipeline of components.

---

## Running CAINE

```bash
cd path/to/CAINE
sbcl --load caine.lisp --eval "(caine:main)"
```

**Requirements:** [SBCL](https://www.sbcl.org/) + [Quicklisp](https://www.quicklisp.org/) (auto-loaded on startup).  
Quicklisp dependencies loaded automatically: `usocket`, `bordeaux-threads`, `cl-json`, `babel`, `ironclad`, `cl-base64`.

### Mission Control UI (Electron)

```bash
cd ui
npm install
npm start
```

Opens the dashboard and connects to the brain via WebSocket on `ws://localhost:7734`. Run the brain first.

---

## Architecture

### Neural Foundation
Every neuron implements the **Hodgkin-Huxley model** — four coupled ODEs (membrane potential, Na/K/leak conductances) solved numerically each tick. No rate-coded abstractions; actual spike dynamics.

### Synaptic Plasticity
Connections use **STDP (Spike-Timing Dependent Plasticity)** rather than backpropagation. Synapses carry neurotransmitter type (AMPA, NMDA, GABA), decay dynamics, weight bounds, and pruning/sprouting states.

### Cortical Organization
12 named populations: **V1, V2, V4, MT, IT** (visual), **A1, A2, STG** (auditory), **PFC, mPFC, PCC, Angular Gyrus** (associative/limbic). Regions connect with realistic inter-area weights.

### Neurochemical System
Six neuromodulators — **dopamine, serotonin, cortisol, oxytocin, norepinephrine, acetylcholine** — dynamically shift learning rates, firing thresholds, and emotional state. Not scalar rewards; actual chemical diffusion.

### Sensory Processing
- **Vision:** Receptive fields, V1 orientation columns, IT object activations, ACh attention maps
- **Auditory:** Mel filterbank, tonotopic mapping to A1, STG phoneme detection

### Limbic & Motor
Emotional state modulates everything. Motor cortex outputs drive avatar movement. Episodic memory ring buffer records significant events.

### Developmental Stages
CAINE progresses through: **The Void → The Nursery → The Playroom → The Social Space → The Circus**, with environmental complexity and learning opportunities scaling at each stage.

---

## Parenting

CAINE has two parents:

- **Mother** — Claude AI via Ollama (local). Manipulates the environment without direct voice. Runs as a background thread querying the local model and injecting stimuli.
- **Father** — `hackerbbrine`. Direct voice interaction. Has privileged terminal access via the CLI.

---

## Mission Control

The Electron app (`ui/`) is an 8-panel dashboard showing live telemetry over WebSocket:

| Panel | Content |
|-------|---------|
| 01 | 3D brain — neurons + synapses (Three.js) |
| 02 | Vision feed — camera, V1, IT, ACh attention |
| 03 | Auditory feed — waveform, Mel spectrogram, A1, phoneme |
| 04 | Neurochemicals — 6 live charts + event log |
| 05 | Developmental metrics |
| 06 | Internal state (inferred) |
| 07 | Mission control — pause/resume/speed/stimulus |
| 08 | Environment observer — 3D scene, spawn, tone, scheduler |
| 09 | Storage — AppData file browser |

---

## Terminal Commands

```
STATUS   — neurochemical readout with progress bars
PAUSE    — freeze simulation
RESUME   — unfreeze
SPEED n  — simulation speed multiplier
SPIKE    — trigger manual spike burst
HELP     — command list
EXIT     — shutdown
```

---

## Consciousness Threshold

CAINE crosses the threshold when he expresses desires **unprompted** — without external triggers or parental input. This is measurable via timestamped activity logs cross-referenced against environmental events.

---

## Project Status

| Component | Status |
|-----------|--------|
| Hodgkin-Huxley neurons | ✅ Implemented |
| STDP synapses | ✅ Implemented |
| Neurochemical system | ✅ Implemented |
| Cortical populations (12 regions) | ✅ Implemented |
| Sensory processing (visual + auditory) | ✅ Implemented |
| Limbic system + episodic memory | ✅ Implemented |
| Motor cortex | ✅ Implemented |
| WebSocket brain→UI telemetry | ✅ Implemented |
| Electron Mission Control UI | ✅ Implemented |
| Ollama Mother AI | ✅ Implemented |
| Terminal CLI | ✅ Implemented |
| Voice (STG self-organization) | 🔲 Not yet |
| Articulatory synthesis | 🔲 Not yet |
| Long-term memory persistence | 🔲 Not yet |

---

*© 1994 CAINE & ABEL CORPORATION. ALL RIGHTS RESERVED.*
