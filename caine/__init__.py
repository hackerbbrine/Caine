"""
CAINE - Creative Artificial Intelligence Networking Entity
==========================================================
Biologically-inspired developmental AI built from computational
neuroscience primitives. No ML frameworks -- only numpy and scipy.

Package layout:
  caine.neuron     -- Hodgkin-Huxley neuron model (Module 1)
  caine.synapse    -- Synaptic connection + STDP (Module 2)
  caine.chemicals  -- Neurochemical system (Module 3)
  caine.cortex     -- Cortical architecture: V1 + A1 populations (Module 4)
  caine.main       -- Combined demo: all systems running together

Future modules follow the same convention:
  caine.<region>   -- e.g. caine.limbic, caine.hippocampus, caine.pfc, ...

All plots are written to the output/ directory at the project root.
"""

import os

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
