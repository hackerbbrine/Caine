"""
CAINE Live Simulation
======================
Run this from the project root:

    python run_live.py

Starts the full sensory pipeline with a real-time matplotlib display.
Press Ctrl-C to stop.

Flags (edit at the top of this file or pass as env vars)
---------------------------------------------------------
  USE_MIC      = True   — use microphone; falls back to generated tones if unavailable
  FRAME_MS     = 20.0   — sensory tick duration in milliseconds
  UPDATE_EVERY = 1      — redraw display every N ticks (increase to speed up sim)
  INJECT_EVENTS= True   — fire random neurochemical events during the run
"""

import sys
import os
import time
import signal

# Force line-buffered stdout so status lines appear immediately even when
# the process is launched from Claude Code or another parent process.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Ensure the project root is on sys.path so 'caine' package is importable
# regardless of which directory the user launches from.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

# ---- User-adjustable settings ----------------------------------------
USE_MIC      = True    # set False to always use generated tones
FRAME_MS     = 20.0    # sensory frame duration (ms)
UPDATE_EVERY = 1       # display refresh: 1 = every frame
INJECT_EVENTS = True   # randomly inject neurochemical events
# ----------------------------------------------------------------------

from caine.cortex    import V1Population, A1Population
from caine.chemicals import NeurochemicalSystem, NeurochemicalEvent, EventType
from caine.sensory   import SensoryLayer, SAMPLE_RATE, FRAME_SAMPLES
from caine.limbic    import LimbicSystem
from caine.environment import CaineEnvironment
from caine.visualizer  import LiveVisualizer

# Shutdown flag (set by Ctrl-C handler)
_running = True

def _sigint(sig, frame):
    global _running
    _running = False
    print("\n[run_live] Stopping...")

signal.signal(signal.SIGINT, _sigint)


def _make_synthetic_frame(tick: int, rng) -> np.ndarray:
    """
    Generate a synthetic 64x64 RGB frame when no real environment is running.
    Alternates between a bar grating and a circle to give V1 something to respond to.
    """
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    if tick % 60 < 30:
        # Horizontal grating
        for row in range(0, 64, 8):
            img[row:row+4, :, :] = 200
    else:
        # Circle
        y, x = np.ogrid[-32:32, -32:32]
        mask = x**2 + y**2 < 20**2
        img[mask] = [220, 180, 60]
    noise = rng.integers(0, 20, (64, 64, 3), dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _make_synthetic_audio(tick: int) -> np.ndarray:
    """Sweep a sine tone through A1's frequency range over ~3 seconds."""
    t_s     = tick * FRAME_MS / 1000.0
    freq    = 200.0 * (6000.0 / 200.0) ** ((t_s % 3.0) / 3.0)  # log sweep 200→6000 Hz
    t_audio = np.arange(FRAME_SAMPLES) / SAMPLE_RATE
    return np.sin(2 * np.pi * freq * t_audio).astype(np.float32)


def _maybe_inject_event(tick: int, neuro: NeurochemicalSystem):
    """Fire a random neurochemical event every ~2 seconds."""
    if tick % 100 == 50:
        event = np.random.choice([
            EventType.NOVEL_STIMULUS,
            EventType.REWARD,
            EventType.DIRECTED_GAZE,
            EventType.STARTLE,
        ])
        neuro.update(FRAME_MS, events=[NeurochemicalEvent(event, 0.8)])
    else:
        neuro.update(FRAME_MS)


def main():
    global _running

    print("=" * 55, flush=True)
    print("  CAINE Live Simulation", flush=True)
    print("=" * 55, flush=True)
    print(f"  Sensory frame : {FRAME_MS} ms", flush=True)
    print(f"  Display update: every {UPDATE_EVERY} frame(s)", flush=True)
    print(f"  Microphone    : {'on' if USE_MIC else 'off (generated tones)'}", flush=True)
    print("  Press Ctrl-C to stop.", flush=True)
    print(flush=True)

    rng = np.random.default_rng()

    # ---- Build all modules -------------------------------------------
    v1     = V1Population(n_neurons=20)
    a1     = A1Population(n_neurons=20)
    neuro  = NeurochemicalSystem()
    sense  = SensoryLayer(v1, a1, neuro, use_mic=USE_MIC)
    limbic = LimbicSystem(v1, a1, neuro)
    viz    = LiveVisualizer(update_every=UPDATE_EVERY)
    # No valence tags — associations emerge from STDP only.
    # Mother can later call limbic.trigger_event(EventType.REWARD, ...) or
    # limbic.trigger_event(EventType.THREAT, ...) while an object is in view
    # to shape what CAINE learns to approach or avoid.

    # Try to start the environment (stubs silently if no ModernGL/PyBullet)
    env = CaineEnvironment()
    env.start()

    # Spawn a couple of objects so the camera has something to look at
    try:
        sphere = env.spawn_object('ball_1', (0.0, 1.5, 5.0), object_type='sphere')
        cube   = env.spawn_object('box_1',  (2.0, 0.5, 5.0), object_type='cube')
    except Exception:
        sphere = cube = None

    # ---- Simulation loop ---------------------------------------------
    tick = 0
    t_wall_start = time.perf_counter()

    while _running:
        tick += 1
        t_tick_start = time.perf_counter()

        # -- Advance environment --
        env.step()

        # -- Get camera frame --
        cam_frame = env.get_camera_feed()
        if cam_frame.max() == 0:
            # Stub renderer returned a black frame — use synthetic
            cam_frame = _make_synthetic_frame(tick, rng)

        # -- Get audio --
        audio_frame = _make_synthetic_audio(tick)  # real mic is handled inside sense
        # If using mic, sense.update will read it; otherwise we inject synthetic
        injected = None if USE_MIC else audio_frame

        # -- Neurochemical events --
        if INJECT_EVENTS:
            _maybe_inject_event(tick, neuro)
        else:
            neuro.update(FRAME_MS)

        # Gentle push to sphere every 3 seconds
        if sphere is not None and tick % 150 == 0:
            env.move_object(sphere, (0.0, 5.0, 0.0))

        # -- Active stimulus: alternate every 3 seconds so each object gets
        #    a chance to accumulate STDP associations -----------------------
        period = int(3000 / FRAME_MS)   # ticks per 3-second window
        phase  = tick % (2 * period)
        if phase < period:
            limbic.set_active_stimulus('ball_1')
        else:
            limbic.set_active_stimulus('box_1')

        # -- Run sensory pipeline --
        joint_angles = np.array([
            0.4 * np.sin(2 * np.pi * 0.3 * tick * FRAME_MS / 1000.0 + i * 1.0)
            for i in range(6)
        ], dtype=np.float32)

        result = sense.update(cam_frame, joint_angles,
                              dt_ms=FRAME_MS,
                              injected_audio=injected)

        # -- Run limbic system --
        limbic_result = limbic.update(
            FRAME_MS,
            result['v1_spikes'],
            result['a1_spikes'],
            result['neuro_snapshot'],
        )

        # -- Update live display --
        viz.update(result,
                   frame_rgb=cam_frame,
                   audio_frame=audio_frame,
                   limbic_result=limbic_result)

        # -- Timing: pace the loop to ~FRAME_MS wall-clock time --
        elapsed = time.perf_counter() - t_tick_start
        target  = FRAME_MS / 1000.0
        sleep   = target - elapsed
        if sleep > 0:
            time.sleep(sleep)

        # Periodic console status
        if tick % 50 == 0:
            t_elapsed = time.perf_counter() - t_wall_start
            fps  = tick / t_elapsed
            nv1  = result['v1_spikes'].sum()
            na1  = result['a1_spikes'].sum()
            snap = result['neuro_snapshot']
            print(f"  t={t_elapsed:6.1f}s  tick={tick:5d}  fps={fps:5.1f}"
                  f"  V1={nv1:2d}  A1={na1:2d}"
                  f"  DA={snap.get('dopamine',0):.3f}  NE={snap.get('norepinephrine',0):.3f}"
                  f"  {limbic.status()}",
                  flush=True)

    # ---- Cleanup -------------------------------------------------------
    env.stop()
    viz.close()
    print("[run_live] Done.")


if __name__ == '__main__':
    main()
