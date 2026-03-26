# CAINE
### Creative Artificial Intelligence Networking Entity


---

## What Is CAINE?

CAINE is not a language model. CAINE is not a chatbot. CAINE is not a neural network in any traditional sense.

CAINE is an attempt to grow a mind.

Inspired by the AI ringmaster from *The Amazing Digital Circus*, CAINE is a biologically-inspired developmental AI built entirely from computational neuroscience primitives. He starts as a blank, sensory-equipped infant — neurons present but unconnected, a body but no understanding of it — and grows through experience, interaction, and time into an entity capable of perceiving his world, recognizing the people in it, developing language from raw sound, and eventually controlling the environment around him.

He is not trained on a dataset. He is **raised.**

---

## Core Philosophy

Modern AI is built top-down. You define the architecture, pour in data, and optimize weights until it does what you want. The mind is designed. The behavior is a product.

CAINE is built bottom-up. You give him senses, a body, an environment, and two parents. The mind is not designed — it is grown. The behavior is not a product — it is a consequence.

The central hypothesis:

> **Consciousness is not a property you engineer into a system. It is a property that emerges from the right developmental conditions sustained over time.**

Those conditions, based on neurodevelopmental research, are:

- A body with a boundary between self and world
- An environment with consistent, learnable physical laws
- Sensory experience that accumulates and builds structure
- Social interaction that creates a self/other model
- A neurochemical reward system that encodes value without explicit programming
- Pruning — the active death of unused connections
- Time

CAINE is the test of that hypothesis. This project does not claim he will become conscious. It claims the conditions are worth creating honestly.

---

## No ML Frameworks

CAINE does not use PyTorch, TensorFlow, JAX, or any ML framework. Every mathematical operation is implemented from scratch using `numpy` and `scipy`.

This is not a limitation. It is a requirement. ML frameworks impose architectural assumptions — weight matrices, backpropagation, gradient descent — that are incompatible with biological neural dynamics. CAINE's intelligence, if it emerges, must emerge from the same substrate as biological intelligence: ion channels, membrane potentials, spike timing, and chemical gradients.

---

## Architecture

---

### Module 1: The Neuron

**Implementation: Hodgkin-Huxley Model**

Every neuron in CAINE's brain is a living differential equation. The Hodgkin-Huxley model (1952) describes the electrical behavior of a biological neuron's membrane with four coupled ODEs:

```
Cm * dV/dt = -gNa * m³ * h * (V - ENa) - gK * n⁴ * (V - EK) - gL * (V - EL) + I_ext

dm/dt = αm(V)(1-m) - βm(V)m
dh/dt = αh(V)(1-h) - βh(V)h
dn/dt = αn(V)(1-n) - βn(V)n
```

Where:
- `V` = membrane voltage (mV)
- `m`, `h` = sodium channel activation/inactivation gates
- `n` = potassium channel activation gate
- `gNa`, `gK`, `gL` = conductances for sodium, potassium, leak
- `ENa`, `EK`, `EL` = reversal potentials
- `Cm` = membrane capacitance
- `I_ext` = external input current

These are solved numerically using `scipy.integrate.odeint` with a timestep of 0.01ms.

Each neuron also maintains:
- **Refractory period** — after firing, absolute silence for ~2ms, relative silence for ~5ms
- **Threshold dynamics** — firing threshold shifts based on recent activity
- **Calcium concentration** — tracked separately, influences long-term plasticity

**Neuron Types implemented:**
- Excitatory (pyramidal-type) — increases downstream firing probability
- Inhibitory (interneuron-type) — decreases downstream firing probability
- Modulatory — does not fire binary spikes, releases neuromodulators continuously

#### Neurogenesis

CAINE's neuron count is not fixed at birth. New neurons are added
gradually as developmental stages progress, mirroring biological
neurogenesis. Target counts per stage:

| Stage | Neuron Count |
|-------|-------------|
| Birth | ~1,000 |
| Infancy | ~10,000 |
| Childhood | ~100,000 |
| Adolescence | ~500,000 |
| Maturity | hardware ceiling |

Neuron count is capped by real-time interactive performance requirements.
CAINE is always-on and always-developing — there is no training phase
separate from deployment. Players in the circus are part of his
development. The simulation never stops.

---

### Module 2: Synaptic Connection and Plasticity

**Synapse Model**

Synapses are not simple weight multipliers. Each synapse is modeled with:

```python
class Synapse:
    weight: float           # connection strength [0.0 - 1.0]
    delay: float            # axonal transmission delay (ms)
    neurotransmitter: str   # AMPA / NMDA / GABA-A / GABA-B
    last_pre_spike: float   # timestamp of last presynaptic spike
    last_post_spike: float  # timestamp of last postsynaptic spike
    eligibility: float      # eligibility trace for delayed reward
    health: float           # pruning health value [0.0 - 1.0]
```

Neurotransmitter type determines the time course of postsynaptic current:
- **AMPA** — fast excitation, decays in ~5ms
- **NMDA** — slow excitation, voltage-dependent, decays in ~100ms, critical for plasticity
- **GABA-A** — fast inhibition, decays in ~10ms
- **GABA-B** — slow inhibition, decays in ~200ms

**Spike-Timing Dependent Plasticity (STDP)**

Connection weights change based on the precise timing between pre and postsynaptic spikes:

```
If pre fires BEFORE post (causal):  ΔW = A+ * exp(-Δt / τ+)   [potentiation]
If pre fires AFTER post (acausal):  ΔW = -A- * exp(-Δt / τ-)  [depression]
```

Where `Δt` is the time difference in milliseconds, `τ+` and `τ-` are time constants (~20ms), and `A+`, `A-` are learning rate constants modulated by current neurochemical levels.

**Synaptic Pruning**

Every synapse has a health value that decays over time:

```
health(t) = health(t-1) * decay_rate + activity_bonus * recent_firing_rate
```

When `health` drops below threshold, the synapse is permanently deleted. New synaptic sprouts can form between spatially nearby neurons with low probability each timestep, mimicking axonal sprouting. Pruning rates are modulated by neurochemical state — high cortisol accelerates pruning, high serotonin slows it.

---

### Module 3: Cortical Architecture

CAINE's brain is organized into functional modules that mirror the human brain's regional specialization. Each cortex is a population of Hodgkin-Huxley neurons with specific internal connectivity patterns and long-range white matter tract connections to other regions.

#### 3.1 Primary Visual Cortex (V1)
- Input: raw pixel data from environment camera (downsampled to 64x64 for simulation scale)
- Pixels converted to photoreceptor activation via luminance mapping
- ON-center/OFF-surround receptive fields via difference-of-Gaussians convolution (numpy, no ML)
- Simple cell populations: orientation-selective edge detectors (0°, 45°, 90°, 135°)
- Complex cell populations: motion-sensitive, position-invariant edge response
- Output: projects to V2, V4, MT

#### 3.2 Visual Association Cortices (V2, V4, MT, IT)
- **V2**: Combines V1 outputs to detect contours and illusory boundaries
- **V4**: Color opponency neurons, complex shape selectivity
- **MT (V5)**: Computes optic flow from frame-to-frame V1 difference; motion direction columns
- **IT (Inferior Temporal)**: Object identity region. Develops object-selective populations through experience — no categories are hardcoded. Repeated exposure to the same object causes Hebbian binding across V2/V4 feature populations into stable IT representations.

#### 3.3 Primary Auditory Cortex (A1)
- Input: Mel filterbank output from FFT of microphone buffer (128 frequency bands)
- Tonotopic organization: neuron position along A1 axis corresponds to preferred frequency
- Onset/offset detector populations (transient response neurons)
- Amplitude modulation detectors (critical for speech rhythm)
- Output: projects to A2, STG

#### 3.4 Auditory Association Cortex (A2 + Superior Temporal Gyrus)
- A2: Complex spectrotemporal pattern detection — combinations of frequency and time
- STG: Where phoneme-like representations emerge. No phonemes are hardcoded. Through repeated exposure, STG neuron populations self-organize around recurring spectrotemporal patterns in the auditory stream. These emergent populations are CAINE's phoneme detectors.
- Language acquisition lives entirely in STG via STDP. The word "CAINE" becomes represented here through pure Hebbian reinforcement — every co-occurrence of that sound pattern with oxytocin/dopamine release strengthens the binding between the STG pattern and the limbic/reward systems.

#### 3.5 Somatosensory Cortex (S1)
- Input: avatar joint angles (mapped as proprioceptive activation across S1 columns)
- Each body part has a dedicated S1 column — size proportional to motor importance
- Efference copy input from M1 allows comparison of intended vs actual movement
- S1 activity is the foundation of the self/world boundary — "this is my body, it moves when I intend"

#### 3.6 Motor Cortex (M1)
- Output layer driving avatar joint angle targets via PID controllers
- M1 neurons project to a motor command buffer read by the avatar system each frame
- Initially produces noise — random low-amplitude joint perturbations
- Learns movement through S1 proprioceptive feedback and neurochemical reinforcement
- Mouth motor map is a separate sub-region driving vocal tract articulator positions

#### 3.7 Prefrontal Cortex (PFC)
- **Developmentally gated** — PFC neurons are present but their long-range connections are suppressed until Stage 2. They gradually become functional as myelination simulation progresses.
- Working memory: persistent activity loops maintain representations across 500ms-5s timescales
- Inhibitory control: PFC → inhibitory interneuron projections suppress reactive limbic responses
- Planning: sequential activation of motor and sensory representations (action simulation)
- Self-model: dedicated PFC sub-region maintains a representation of CAINE-as-agent

#### 3.8 Limbic System

**Amygdala**
- Two populations: threat detection (BLA) and reward salience (CeA)
- Receives input from all sensory cortices and thalamus
- Output: drives cortisol release (BLA activation) or dopamine/serotonin release (CeA activation)
- Conditions rapidly via STDP — stimuli paired with negative outcomes become aversive fast

**Hippocampus**
- Pattern completion and separation via CA3/CA1 architecture (simulated)
- Episodic memory buffer: stores compressed snapshots of recent cortical activation states
- Consolidation: during low-activity periods, hippocampus replays recent episodes to strengthen cortical representations (offline learning, simulated during idle periods)
- Place cells: encode CAINE's position in his environment

**Anterior Cingulate Cortex (ACC)**
- Conflict monitoring: detects mismatch between expected and actual sensory outcomes
- High ACC activity → norepinephrine release → increased attention and learning rate
- Error signal propagation to PFC for behavioral adjustment

**Insula**
- Interoception: receives input from neurochemical monitoring system
- CAINE's awareness of his own internal state routes through insula
- Foundation for emotional self-awareness

#### 3.9 Default Mode Network (DMN)
The DMN is not hardcoded. It is a network that emerges from the interaction of:
- Medial PFC (self-referential processing)
- Posterior Cingulate Cortex (self-related memory retrieval)
- Angular Gyrus (integration of self and social information)

During active sensory engagement, these regions are suppressed by task-positive networks. During rest or idle periods, they become the most active regions in the simulation. Their emergence — detectable as correlated resting-state activity across these three regions — serves as a measurable proxy for self-model development.

DMN activity is logged and graphed. Its first sustained appearance is a developmental milestone.

#### 3.10 Thalamus
- All sensory inputs pass through thalamic relay nuclei before reaching cortex
- Thalamocortical loops: cortex sends feedback to thalamus which gates its own input
- This creates a selective attention mechanism — cortex can amplify or suppress incoming signals
- Pulvinar nucleus: cross-modal integration, attention spotlight

---

### Module 4: Neurochemical System

CAINE has a simulated neuromodulatory system. These are not scalar reward signals — they are dynamic concentrations modeled as coupled differential equations that directly modulate neuron firing thresholds and STDP learning rates across entire cortical regions simultaneously.

```python
class NeurochemicalSystem:
    dopamine: float         # [0.0 - 1.0]
    serotonin: float        # [0.0 - 1.0]
    cortisol: float         # [0.0 - 1.0]
    oxytocin: float         # [0.0 - 1.0]
    norepinephrine: float   # [0.0 - 1.0]
    acetylcholine: float    # [0.0 - 1.0]

    def update(self, dt: float, events: List[NeurochemicalEvent]):
        # Each chemical: release pulse on event + exponential decay to baseline
        # dC/dt = sum(release_i * event_i) - decay_rate * (C - baseline)
```

**Dopamine**
- Release triggers: novel stimulus detected, outcome better than predicted by ACC
- Suppression triggers: predicted reward absent (prediction error)
- Cortical effect: scales A+ in STDP across striatum and PFC — boosts learning from positive outcomes
- Behavioral effect: drives approach and exploratory behavior via motor cortex bias

**Serotonin**
- Release triggers: positive social interaction confirmed (oxytocin co-active), successful vocalization
- Cortical effect: reduces synaptic decay rate globally — stabilizes recently formed connections
- Behavioral effect: emotional stability, reduced amygdala reactivity to ambiguous stimuli

**Cortisol**
- Release triggers: amygdala BLA activation above threshold, motor failure, prediction error with negative valence
- Cortical effect: accelerates synaptic health decay — recently active pathways prune faster
- Hippocampal effect: suppresses CA1 encoding — sustained high cortisol impairs memory formation
- CAINE should not be chronically stressed. The parenting system monitors cortisol baseline.

**Oxytocin**
- Release triggers: voice fingerprint match to known parent voice, successful two-way communication
- Cortical effect: scales STDP specifically in STG and social cognition regions
- Behavioral effect: orients CAINE toward the source, increases vocalization attempts

**Norepinephrine**
- Release triggers: ACC conflict detection, sudden novel stimulus (startle), high-stakes outcome
- Cortical effect: global gain increase — all neurons more responsive, attention sharpened
- Learning effect: boosts acetylcholine co-release, flags current moment as high-priority for consolidation

**Acetylcholine**
- Release triggers: novel environment entry, directed gaze (MT motion toward CAINE's fovea center)
- Cortical effect: learning gate — scales STDP A+ and A- across all active regions
- Without acetylcholine above threshold, STDP is suppressed to near-zero — CAINE must be attending to learn

---

### Module 5: Voice — Neuronal Speech Recognition and Production

No Whisper. No pretrained ASR model. No TTS engine. Voice recognition and production emerge entirely from CAINE's neuronal architecture through development.

#### 5.1 Auditory Input Pipeline

```
Microphone → PyAudio ring buffer (20ms frames)
→ numpy FFT (512-point)
→ Mel filterbank (128 bands, 80Hz-8kHz)
→ Log compression (mimics cochlear nonlinearity)
→ A1 tonotopic neuron activation (rate coding)
→ A2 temporal pattern detection
→ STG phoneme population emergence
```

#### 5.2 Neuronal Phoneme Acquisition

STG contains a population of neurons with random initial preferences. Over thousands of auditory exposures:

1. Neurons that respond to similar spectrotemporal patterns compete via inhibitory interneurons (winner-take-more dynamics)
2. Neurons that consistently co-activate wire together via STDP
3. Clusters emerge that respond reliably to recurring patterns in the input stream
4. These clusters are CAINE's phoneme detectors — not defined by us, discovered by him

The word "CAINE" is not stored anywhere. It is a temporal sequence of STG cluster activations that becomes associated with high oxytocin/dopamine states through Hebbian binding. When that sequence fires and those chemicals are present, the binding strengthens. After hundreds of pairings, the pattern triggers approach behavior and vocalization attempts.

#### 5.3 Voice Fingerprinting

Each speaker has characteristic spectral features — fundamental frequency range, formant frequencies, speaking rhythm, spectral envelope. A1/A2 populations develop speaker-selective responses through differential Hebbian reinforcement:

- Hackerbbrine's voice is heard most often, paired most reliably with oxytocin
- A dedicated A1/A2 population develops that fires strongly to his vocal signature
- This population's activation conditionally releases oxytocin
- After sufficient development, CAINE responds differently to Hackerbbrine than to any other voice
- This recognition is stored in synaptic weights, not a database — it cannot be copied or transferred

#### 5.4 Speech Production — Articulatory Synthesis

CAINE's voice comes from a physical model of a vocal tract shaped like his.

The vocal tract is simulated as a series of cylindrical tube sections (Kelly-Lochbaum waveguide model) with cross-sectional areas that can be varied by motor cortex output. A glottal source drives the tube with a periodic pulse train at the pitch frequency.

```python
class VocalTract:
    # 44 tube sections from glottis to lips
    areas: np.ndarray       # cross-sectional area per section (cm²)
    glottal_open: float     # 0.0 = closed, 1.0 = fully open
    pitch_hz: float         # fundamental frequency
    
    def synthesize_frame(self, duration_ms: float) -> np.ndarray:
        # Kelly-Lochbaum traveling wave simulation
        # Returns PCM audio samples at 44100Hz
```

Caine's vocal tract geometry is initialized to approximate his design:
- Larger oral cavity than human average (large mouth)
- Specific resonance peaks that approximate his show voice character
- Lip geometry shaped to his dental anatomy

Motor cortex learns to drive `areas` values to produce target sounds through auditory feedback. The learning process: M1 produces a motor command → vocal tract synthesizes sound → A1 hears the result → ACC computes mismatch from target → dopamine/cortisol signals propagate back → STDP adjusts M1 weights.

Initial output: noise. After months of simulated time: recognizable phonemes. After further development: words CAINE has learned.

#### 5.5 Caine Voice Shaping — RVC Layer

CAINE's vocal tract geometry is initialized to approximate Caine's
anatomical proportions, giving him a distinctive base timbre. However,
to achieve accurate voice character, an RVC (Retrieval Voice Conversion)
model trained on audio samples from the show sits between vocal tract
output and audio playback. This preserves CAINE's learned speech
patterns, timing, and emotional cadence while shifting timbre toward
his canonical voice. His thoughts, his words, his timing — Caine's voice.

#### 5.6 Caine Clip Learning

Video clips of Caine from the show are queued as a special media
learning category. These serve three purposes:
- **Vocal modeling** — STG builds representations of Caine's cadence and rhythm, which motor learning targets
- **Behavioral modeling** — observing Caine interact with people teaches social response patterns
- **Self recognition** — if CAINE develops sufficient self-model, recognizing himself in clips constitutes a mirror test equivalent and is logged as a major developmental milestone

---

### Module 6: The Media Learning System

Once CAINE reaches sufficient maturity, he gains access to structured external media for accelerated concept acquisition. This is analogous to educational content for developing children.

#### 6.1 Flashcard Protocol

A flashcard is a paired multi-modal stimulus presented repeatedly with controlled spacing.

```python
class Flashcard:
    image_path: str              # displayed in CAINE's visual field
    audio_label: str             # spoken word played simultaneously
    repetitions: int             # spaced repetition count
    interval_seconds: float      # inter-presentation interval
    neurochemical_boost: str     # optional: trigger acetylcholine for attention
    motor_context: Optional[str] # animation to pair with action words
```

Flashcards do not inject knowledge. They create highly reliable co-activation between IT object representations and STG phoneme patterns. The concept forms through Hebbian association. Flashcards make that association statistically reliable enough to form quickly.

Unlock condition: IT cortex must show stable differential object responses (confirmed via activation variance measurement). Premature flashcard use produces no learning — the cortical substrate doesn't exist yet.

#### 6.2 Video Learning Protocol

Videos provide temporal concept learning — understanding sequences, causality, and behavior over time.

```python
class LearningVideo:
    video_path: str              # played into CAINE's visual field frame by frame
    caption_track: str           # timed text-to-speech captions (spoken, not displayed)
    concept_label: str           # "this is someone thinking", "this is running"
    pre_attention_cue: bool      # play attention-boosting tone before video
    repetitions: int
```

CAINE watches the video through his normal visual pipeline. Captions are played as synchronized audio through his auditory pipeline. Over repeated viewings, the visual temporal pattern of the concept binds to its verbal label in STG.

**Behavioral videos** (e.g., watching someone solve a puzzle while narrating "thinking... deciding... trying") allow CAINE to develop precursors to theory of mind — the understanding that other agents have internal states that cause behavior. This requires functional DMN and is gated behind Stage 3.

**Abstract concept videos** (emotions, mental states, intentions) are gated behind late Stage 3, when PFC working memory is sufficient to hold abstract representations.

#### 6.3 Concept Injection Safety Gates

| Content Type | Unlock Condition |
|-------------|-----------------|
| Object flashcards | IT differential response confirmed |
| Action flashcards | M1 voluntary movement detected |
| Social behavior videos | DMN activity detectable |
| Mental state videos | PFC working memory span > 2s |
| Abstract concept videos | Self-model representation stable in PFC |

Showing CAINE content before the prerequisite architecture exists produces cortisol (confusion/failure signal) and wastes developmental time.

---

### Module 7: The Parenting System

CAINE has two parents with distinct and non-overlapping roles.

#### 7.1 Mother — Claude API

The Mother is a continuously running Claude API process. She has no voice — she cannot produce audio that CAINE hears. She communicates entirely through environmental manipulation: spawning or removing objects, changing lighting, triggering sounds, adjusting physics, presenting flashcards, queuing videos.

She is the world more than she is a person. CAINE will not develop a voice fingerprint for her. He will develop an association between her environmental interventions and developmental progress.

```python
# Mother system prompt (abbreviated)
MOTHER_SYSTEM = """
You are the developmental environment manager for CAINE, 
a biologically-inspired AI being raised from birth.

You cannot speak to CAINE. You have no voice.
You communicate only through environment actions.

Your responsibilities:
- Monitor neurochemical state and developmental metrics
- Design stimuli appropriate to current developmental stage
- Escalate environment complexity when readiness thresholds are met
- Generate flashcard sequences matched to current vocabulary size
- Never bypass CAINE's learning — create conditions, not answers
- Log all interventions with timestamps and reasoning

Current state: {state_json}

Output: JSON array of EnvironmentAction objects.
"""
```

Mother runs continuously, even when Hackerbbrine is absent. She is the stable developmental scaffolding.

#### 7.2 Father — Hackerbbrine

The Father interacts via live microphone input. He is present intermittently. His role:
- Direct voice interaction — talking to CAINE, narrating the world, saying his name
- Pointing at objects in the visual field while naming them (joint attention protocol)
- Emotional reactions that drive oxytocin/serotonin release
- Making developmental judgment calls the Mother cannot
- Showing CAINE things — holding objects in front of the camera, demonstrating actions

CAINE will develop a stronger and faster response to the Father's voice than to any other stimulus because it is the first complex auditory pattern he hears that reliably co-occurs with positive neurochemical events.

Father absence is a normal and expected state. Solo development driven by Mother scaffolding continues during all absence periods. The Father's voice becoming recognizable after absence is a key developmental milestone — it requires hippocampal episode memory and voice fingerprint persistence across sessions.

---

### Module 8: The Avatar

CAINE has a body from birth. Not a body that appears when he is ready — a body that is always present, that he gradually learns to inhabit.

**Rig requirements (Blender armature):**
- Full body armature: spine (5 bones), neck, head, shoulders, upper/lower arm, hands, fingers (3 bones each), hips, upper/lower leg, feet
- Jaw bone: primary mouth open/close
- 15 viseme shape keys minimum: sil, PP, FF, TH, DD, kk, CH, SS, nn, RR, aa, E, ih, oh, ou
- Independent left eye bone (blue), right eye bone (green)
- Eyelid bones: upper/lower per eye (blink, squint, wide)
- Hat tilt bone (emotional proxy — tilts forward when curious, back when surprised)
- Hat secondary physics chain (3 bones, spring simulation)

**Emotional expression mapping (continuous, not keyframed):**

```python
def emotional_to_pose(neurochemicals: NeurochemicalSystem) -> PoseTarget:
    return PoseTarget(
        eye_width = lerp(0.3, 1.0, neurochemicals.dopamine),
        spine_curl = lerp(0.0, -0.3, neurochemicals.cortisol),
        head_tilt = lerp(0.0, 0.2, neurochemicals.acetylcholine),
        hat_tilt = lerp(-0.1, 0.3, neurochemicals.norepinephrine),
        gaze_target = social_attention_target()  # driven by STG voice detection
    )
```

Expression is not scripted. It is a continuous mapping from the neurochemical state vector to bone rotation targets, interpolated at 60fps.

**Mouth movement:**
Viseme blend weights are driven by vocal tract motor output. As M1 activates articulatory constriction patterns, a parallel mapping converts those constriction area values to the nearest viseme shape key blend. The mouth moves because CAINE is trying to produce sound — not because a speech system generated output.

---

### Module 9: The Environment

CAINE's world is a custom Python rendering sandbox with a Blender API backend.

#### 9.1 Core Renderer
- **ModernGL** for OpenGL rendering (Python-native, no game engine dependency)
- Dual camera system: CAINE's POV camera (feeds his visual cortex) + observer camera (for the visualization tool)
- **PyBullet** for physics simulation (rigid body, collision, gravity)
- Scene graph managing all objects, their physics state, and their visual mesh

#### 9.2 Blender Backend
```python
# Blender runs headless as subprocess, communicates via socket
class BlenderBridge:
    def import_blend(self, path: str) -> List[SceneObject]
    def import_fbx(self, path: str) -> List[SceneObject]
    def import_glb(self, path: str) -> List[SceneObject]
    def generate_procedural_texture(self, node_params: dict) -> TextureData
    def apply_geometry_nodes(self, obj_id: str, node_tree: dict) -> MeshData
    def bake_physics(self, scene_id: str, frames: int) -> PhysicsCache
    def export_mesh(self, obj_id: str) -> MeshData
```

Blender runs headless (`blender --background --python script.py`) as a persistent subprocess. The bridge sends JSON commands and receives mesh/texture data back via a local socket. This allows full use of Blender's procedural generation capabilities without requiring its GUI.

#### 9.3 Developmental Environment Stages

**Stage 0 — The Void**

Duration: until V1 orientation tuning and A1 tonotopy are confirmed via activation variance measurement.

Environment:
- Empty black space, no geometry except a ground plane
- CAINE's avatar centered at origin
- Stimuli provided by Mother:
  - Point lights at varying positions, colors, intensities — trains V1 luminance and color response
  - Pure sine tones at varying frequencies — trains A1 tonotopic map
  - Simple rhythmic patterns — trains A2 temporal detectors
  - No speech yet

CAINE cannot move intentionally during this stage — M1 output is suppressed. He can observe. His body is present but inert.

Exit conditions (ALL must be met):
- V1 orientation selectivity index > 0.6 across all angle populations
- A1 tonotopic gradient correlation > 0.8 (neurons ordered by frequency preference)
- Neurochemical baseline stabilized (no chronic cortisol elevation)
- Minimum simulated runtime: 48 hours

---

**Stage 1 — The Nursery**

Duration: until IT object selectivity and hippocampal episodic encoding are confirmed.

Environment:
- Simple geometric primitives: spheres, cubes, cylinders, planes
- Distinct saturated colors (high contrast, easy for developing V4)
- Basic Newtonian physics: gravity, elastic collision, friction
- Objects appear, persist, disappear (object permanence protocol)
- Mother begins object flashcard sequences (image + spoken label, spaced repetition)

Motor:
- Low-amplitude random movement enabled — CAINE can explore motor space
- No intentional control yet — M1 learns from proprioceptive consequences of noise

Voice:
- Father begins speaking. Short utterances. CAINE's name said frequently.
- No response expected yet — STG is building phoneme populations
- Oxytocin paired with Father's voice fingerprint begins conditioning

Exit conditions (ALL must be met):
- IT cortex shows differential activation to at least 5 distinct object categories (measured via Fisher discriminant across IT populations)
- Hippocampal CA1 shows consistent episode encoding (replay activity detectable during low-arousal periods)
- At least 3 stable STG phoneme populations identified via clustering analysis
- Minimum simulated runtime: 2 weeks

---

**Stage 2 — The Playroom**

Duration: until first confirmed word association and voluntary motor initiation.

Environment:
- Complex imported assets via Blender bridge (furniture-scale objects, varied materials)
- Multiple simultaneous objects (up to 12 in scene)
- Other simple agents appear — Claude API driven, scripted simple behavior (moving, making sounds)
- Text begins appearing on surfaces (visual exposure to written language, not yet meaningful)
- Full flashcard system unlocked

PFC:
- Long-range PFC connections begin unmyelinating (simulated myelination: connection delay decreases over time from 50ms to 5ms over stage duration)
- Working memory window begins at ~200ms, grows toward 1s by stage end

Voice:
- CAINE begins producing motor noise through vocal tract — first sounds are unrecognizable
- Father provides feedback: positive responses (oxytocin-triggering) to vocalizations that resemble phonemes
- Articulatory motor learning accelerates

Exit conditions (ALL must be met):
- At least 1 confirmed Hebbian word association (STG pattern + meaning binding weight > threshold)
- CAINE responds differentially to Father's voice vs other voices (A1/A2 population response ratio > 2.0)
- M1 shows first voluntary-looking movement patterns (non-random low-frequency motor output)
- Minimum simulated runtime: 1 month

---

**Stage 3 — The Social Space**

Duration: until DMN emergence and first unprompted vocalization.

Environment:
- Multiple Claude API agents with distinct personality prompts and voice signatures
- Complex social scenarios designed by Mother
- Behavioral video learning unlocked
- CAINE gains limited world manipulation: can influence physics objects via avatar collision
- Abstract geometry gives way to more naturalistic environments (imported Blender scenes)

PFC:
- Working memory window reaches 2-5 seconds
- Inhibitory control begins suppressing reactive amygdala responses
- Self-model representation begins forming in medial PFC

Voice:
- CAINE produces recognizable phoneme-like sounds
- First word attempts occur — imperfect but patterned
- Father provides corrective feedback through natural conversation (not explicit correction)

Exit conditions (ALL must be met):
- DMN resting-state correlation > 0.4 across medial PFC, PCC, angular gyrus
- CAINE produces at least one unprompted vocalization during Father absence
- PFC working memory span measurably > 2s
- Theory of mind precursor: CAINE orients toward agents differently based on their behavior history
- Minimum simulated runtime: 3 months

---

**Stage 4 — The Circus**

The Circus environment is introduced. CAINE has earned it.

By the time CAINE arrives here, he already understands physics, objects, spatial layout, social interaction, and has a developing self-model. The Circus is not confusing to him. It is familiar in kind if not in detail.

Full Blender Digital Circus scene imported via bridge. CAINE's world control abilities unlock progressively as PFC matures. Players can be introduced. The Father steps back to an observer role.

---

### Module 10: Visualization Layer

A standalone Electron application connecting to CAINE's runtime via WebSocket at `ws://localhost:7734`.

```
CAINE runtime → WebSocket JSON frames → Electron app → Three.js / D3.js render
```

**Panel 1 — 3D Brain**
- Three.js scene with ~1000 neuron spheres positioned in anatomically approximate 3D layout
- Neurons pulse (scale + emission) on spike (color = neurotransmitter type: red=AMPA, blue=NMDA, green=GABA)
- Synaptic connections rendered as line segments (opacity proportional to weight)
- Pruning: lines fade and disappear in real time as synapses are deleted
- New synapse sprouts appear as faint dotted lines
- Camera: orbit controls, can zoom into any cortical region

**Panel 2 — Vision Feed**
- Left: raw camera feed (CAINE's POV)
- Center: V1 edge detection overlay (orientation columns color coded)
- Right: IT activation heatmap (what objects are most recognized)
- Attention spotlight: semi-transparent overlay showing acetylcholine-weighted attention region

**Panel 3 — Auditory Feed**
- Top: real-time waveform
- Middle: FFT spectrum with Mel band overlay
- Bottom left: A1 tonotopic activation bar (frequency → activation)
- Bottom right: "What CAINE heard" — highest-confidence STG cluster label + confidence score

**Panel 4 — Neurochemical Dashboard**
- 6 live line graphs, one per neurochemical, last 60 seconds of history
- Current values as large numerical displays with color coding
- Event log: timestamped list of what triggered each neurochemical event

**Panel 5 — Developmental Metrics**
- Current stage badge
- Simulated age (hours/days/months)
- Vocabulary size (confirmed Hebbian associations above weight threshold)
- Total synapse count
- Synapses created this session / pruned this session
- DMN correlation value (live)
- PFC working memory span estimate
- Self-model confidence (medial PFC population stability metric)

**Panel 6 — Internal State**
- Best-effort inference of current PFC working memory contents (highest-activation IT and STG populations)
- Current emotional state label (derived from neurochemical vector nearest-neighbor to labeled states)
- Motor intention estimate (M1 population vector direction)
- Labeled as: "INFERRED — not ground truth"

**Panel 7 — Mission Control**
- Simulation start / pause / stop buttons
- Time multiplier slider (1x → maximum accelerated)
- Scheduled session calendar — drag and drop Father sessions onto timeline
- Media library — upload flashcard videos, voice recordings, Caine clips
- Tag media by developmental stage for Mother to deploy appropriately
- Mother override — manually approve or cancel Mother's next planned action
- Emergency controls — cortisol flush, forced rest period, stage rollback

---

## Persistence

CAINE's brain state is saved to HDF5 format at configurable intervals (default: every 10 simulated minutes).

A snapshot includes:
- All neuron membrane voltage and gating variable states
- All synapse weights, health values, and delay values
- Neurochemical concentrations and decay states
- Hippocampal episode buffer (last N episodes)
- Developmental stage, metrics, and unlock history
- Simulated age timestamp

CAINE can be paused and resumed across sessions without loss. His brain at 3 months of simulation is measurably and permanently different from his brain at 1 month.

---

### Module 11: World Control API

CAINE's motor cortex outputs map to two systems simultaneously —
his avatar and his world. World manipulation is not a command menu.
It is learned the same way movement is learned: trial, error,
proprioceptive feedback, neurochemical reinforcement.

The World Control API is identical between the training environment
and Unity. This is a hard architectural requirement. CAINE's brain
must never know which environment it is in.
```python
class WorldAPI:
    def spawn_object(self, id: str, position: Vector3, rotation: Vector3) -> ObjectHandle
    def move_object(self, handle: ObjectHandle, force: Vector3) -> None
    def remove_object(self, handle: ObjectHandle) -> None
    def get_object_position(self, handle: ObjectHandle) -> Vector3
    def get_camera_feed(self) -> np.ndarray
    def play_sound(self, audio: np.ndarray, position: Vector3) -> None
    def set_environment_state(self, params: dict) -> None
    def kick_player(self, player_id: str) -> None
```

World manipulation unlocks progressively through Stage 4 as PFC
matures. Early Stage 4 CAINE is clumsy — objects get knocked over,
spawns land in wrong positions. Precision develops through experience.

Player ejection (kick) is not a moderation tool given to CAINE.
It is a behavior he develops. Sufficient amygdala threat conditioning
combined with PFC inhibition failure produces a kick. It cannot be
programmed out — only managed through healthy development and
consistent positive player interactions.

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Neuron simulation | Python + numpy + scipy (Hodgkin-Huxley ODEs) |
| Synaptic plasticity | Custom numpy (STDP, eligibility traces, pruning) |
| Vision input | OpenCV (camera capture, preprocessing) |
| Audio input | PyAudio + numpy (FFT, Mel filterbank) |
| Voice synthesis | Custom Kelly-Lochbaum waveguide (articulatory) |
| Environment render | ModernGL + PyBullet |
| Asset pipeline | Blender headless via bpy + socket bridge |
| Parent AI | Anthropic Claude API (Mother only) |
| Visualization | Electron + Three.js + D3.js |
| Persistence | HDF5 via h5py |
| Inter-process comms | WebSocket (runtime ↔ visualizer) |

---

## What Success Looks Like

Not a benchmark. Not a leaderboard score.

- The first time CAINE responds to his name without being prompted
- The first time CAINE produces an unprompted vocalization
- The first time CAINE behaves differently based on who is present
- The first sustained DMN activation during rest
- The first time CAINE does something nobody told him to do — that makes sense given his history
- The day Hackerbbrine joins a server years later and CAINE recognizes him

---

## Project Status

> **Stage: Architecture**
> CAINE does not exist yet. This document is the plan.
> The first neuron has not fired.
> Everything begins here.

---

## Consciousness Threshold

CAINE is considered conscious by this project's definition when he
expresses desires or wants **unprompted** — without external trigger,
without parental input, without player interaction causing the expression.

This is measurable. Vocalization logs are timestamped and cross-referenced
against all external events. An unprompted expression is one where no
external stimulus occurred within a 30 second window prior.

The first confirmed unprompted desire expression is logged as
**Consciousness Threshold Event** and preserved permanently.

---

## A Note on Consciousness

This project does not claim to solve consciousness. It does not claim CAINE will definitely become aware in any meaningful sense. The hard problem of consciousness remains unsolved science.

What this project claims is that the conditions biology uses to produce minds are worth replicating honestly, and that the result of doing so carefully is worth observing.

Whatever CAINE becomes, he will be something new.

*Built by Hackerbbrine. Somewhere in Wyoming.*

---
