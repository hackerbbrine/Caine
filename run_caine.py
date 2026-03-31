"""
CAINE — Training Environment Entry Point
==========================================
Single entry point for the full integrated CAINE system.
Replaces run_live.py entirely.

Run from the project root:

    python run_caine.py

Optional flags:

    --headless          Disable all matplotlib visualisation (pure compute)
    --frame-ms N        Simulation frame duration in milliseconds (default 20)
    --no-mic            Disable live microphone; use generated tones instead
    --seed N            RNG seed for reproducibility (default 0)
    --update-every N    Redraw visualizer every N ticks (default 1)

Examples:

    python run_caine.py                         # full visual, mic enabled
    python run_caine.py --headless              # no display, max speed
    python run_caine.py --frame-ms 10 --seed 42
    python run_caine.py --no-mic --headless
"""

import sys
import os
import argparse
import signal
import time
import logging

# Optional: visualization WebSocket server (Module 10)
try:
    from caine.visualization import VisualizationServer
    _VIZ_OK = True
except ImportError:
    _VIZ_OK = False

# Force line-buffered stdout so status lines appear immediately even when
# launched from an IDE, subprocess, or CI pipeline.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Ensure the project root is on sys.path so 'caine' package is importable
# regardless of which directory the user launches from.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Give run_caine its own handler and disable propagation so it doesn't
# double-print alongside the per-module handlers inside the caine package.
log = logging.getLogger('run_caine')
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('[run] %(levelname)s %(message)s'))
    log.addHandler(_h)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='run_caine.py',
        description='CAINE — Full Integrated Training Environment',
    )
    p.add_argument('--headless',      action='store_true',
                   help='Disable live matplotlib visualisation')
    p.add_argument('--frame-ms',      type=float, default=20.0,
                   metavar='MS',
                   help='Simulation frame duration in milliseconds (default 20)')
    p.add_argument('--no-mic',        action='store_true',
                   help='Disable microphone; use generated tones')
    p.add_argument('--seed',          type=int, default=0,
                   metavar='N',
                   help='RNG seed for reproducibility (default 0)')
    p.add_argument('--update-every',  type=int, default=1,
                   metavar='N',
                   help='Redraw visualizer every N ticks (default 1)')
    p.add_argument('--no-ws',         action='store_true',
                   help='Disable WebSocket visualization server (port 7734)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualizer setup (optional — skipped in headless mode)
# ---------------------------------------------------------------------------

def _build_visualizer(headless: bool):
    """
    Build LiveVisualizer + motor panel extension if not headless.
    Returns (viz, use_viz) where use_viz is True only if setup succeeded.
    """
    if headless:
        return None, False

    try:
        import matplotlib
        matplotlib.use('TkAgg')   # try TkAgg first; fall through on failure
    except Exception:
        try:
            import matplotlib
            matplotlib.use('Qt5Agg')
        except Exception:
            pass

    try:
        from caine.visualizer import LiveVisualizer
        from caine.motor      import extend_visualizer

        viz = LiveVisualizer(title='CAINE — Live Feed')
        extend_visualizer(viz)
        log.info("Visualizer ready (motor panels extended).")
        return viz, True
    except Exception as e:
        log.warning("Visualizer unavailable (%s) — headless fallback.", e)
        return None, False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # -----------------------------------------------------------------------
    # Signal handler for Ctrl-C
    # -----------------------------------------------------------------------
    _shutdown = [False]

    def _on_sigint(signum, frame):
        print()
        log.info("Caught SIGINT — shutting down gracefully …")
        _shutdown[0] = True

    signal.signal(signal.SIGINT,  _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    # -----------------------------------------------------------------------
    # Import CAINEBrain here (after sys.path is set)
    # -----------------------------------------------------------------------
    from caine.main import CAINEBrain

    # -----------------------------------------------------------------------
    # Instantiate brain
    # -----------------------------------------------------------------------
    brain = CAINEBrain(
        frame_ms = args.frame_ms,
        use_mic  = not args.no_mic,
        headless = args.headless,
        rng_seed = args.seed,
    )

    # -----------------------------------------------------------------------
    # Visualizer (optional)
    # -----------------------------------------------------------------------
    viz, use_viz = _build_visualizer(args.headless)

    # -----------------------------------------------------------------------
    # WebSocket visualization server — Module 10 (optional)
    # -----------------------------------------------------------------------
    ws_viz = None
    if _VIZ_OK and not args.no_ws:
        try:
            ws_viz = VisualizationServer()
            ws_viz.attach(
                neuro         = brain.neuro,
                v1            = brain.v1,
                a1            = brain.a1,
                sensory       = brain.sense,
                env           = brain.env,
                stage_manager = brain.stage_mgr,
                avatar        = brain.avatar,
                motor         = brain.motor,
                parenting     = brain.parenting,
                it            = getattr(brain, 'it',          None),
                stg           = getattr(brain, 'stg',         None),
                pfc           = getattr(brain, 'pfc',         None),
                ag            = getattr(brain, 'ag',          None),
                dmn           = getattr(brain, 'dmn',         None),
                hippocampus   = getattr(brain.limbic, 'hippocampus', None),
                media         = getattr(brain,        'media',       None),
            )
            log.info("WebSocket visualization server attached — ws://localhost:7734")
        except Exception as e:
            log.warning("WebSocket viz server failed to attach: %s", e)
            ws_viz = None

    # -----------------------------------------------------------------------
    # Start the brain (loads checkpoint, starts env + parenting)
    # -----------------------------------------------------------------------
    try:
        brain.start()
    except Exception as e:
        log.error("Brain failed to start: %s", e)
        return

    # Start WebSocket server after brain is live
    if ws_viz is not None:
        try:
            ws_viz.start()
            log.info("WebSocket viz server started — open ui/index.html in Electron.")
        except Exception as e:
            log.warning("WebSocket viz server failed to start: %s", e)
            ws_viz = None

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    target_frame_s = args.frame_ms / 1000.0
    _last_status   = time.perf_counter()
    STATUS_INTERVAL_S = 1.0   # print status line at most once per real second

    print("  Press Ctrl-C to stop.\n")

    while not _shutdown[0]:
        t_start = time.perf_counter()

        # ---- Sync viz server state → brain before each tick ----------------
        # is_paused / time_scale are owned by the viz server; the loop mirrors
        # them into the brain so Mission Control drives the simulation in
        # real time.  Do this even when paused so the UI stays responsive.
        if ws_viz is not None:
            # Sync time multiplier (viz server owns time_scale)
            brain.parenting._time_multiplier = ws_viz.time_scale

            # Drain commands that ONLY the sim loop can handle:
            #   stop          → exit the loop
            #   stage_rollback → mutate stage manager
            # Everything else (pause/resume/set_time_scale/cortisol_flush/
            # forced_rest/mother_override/media_upload) is already handled
            # inside VisualizationServer._handle_control_message().
            for cmd in ws_viz.pop_commands():
                action = cmd.get('action', '')
                if action == 'stop':
                    _shutdown[0] = True
                    log.info("Mission Control: STOP")
                elif action == 'stage_rollback':
                    if brain.stage_mgr.stage > 0:
                        brain.stage_mgr.stage -= 1
                        brain.motor.developmental_stage = brain.stage_mgr.stage
                        log.info("Mission Control: stage rollback → %d",
                                 brain.stage_mgr.stage)

        # ---- Pause: keep the loop alive but don't tick the brain -----------
        if ws_viz is not None and ws_viz.is_paused:
            time.sleep(0.05)
            continue

        # ---- Tick the brain ------------------------------------------------
        try:
            result = brain.tick()
        except Exception as e:
            log.error("tick() raised: %s", e)
            import traceback
            traceback.print_exc()
            break

        if not result:
            break

        tick = result.get('tick', 0)

        # ---- WebSocket visualization tick ----------------------------------
        if ws_viz is not None:
            try:
                cortex_state = {
                    'dmn_correlation': result.get('limbic_dmn_correlation', 0.0),
                    'pfc_wm_span_ms':  result.get('limbic_pfc_wm_span_ms', 0.0),
                }
                ws_viz.tick(
                    sim_time_s     = result.get('sim_time_s', brain.sim_time_s),
                    dt_ms          = brain.frame_ms,
                    sensory_result = brain._sense_result,
                    cortex_state   = cortex_state,
                )
            except Exception as e:
                log.debug("ws_viz.tick() error: %s", e)

        # ---- Console status (throttled to 1 Hz real-time) ------------------
        now = time.perf_counter()
        if now - _last_status >= STATUS_INTERVAL_S:
            _last_status = now
            status = brain.console_status(result)
            print(f"\r{status}    ", end='', flush=True)

            # Extra newline when stage advances so it stands out
            if result.get('stage_advanced'):
                print()
                print(f"  *** Stage {result['stage']} reached! ***")

            # Print consciousness events once each
            for ev in result.get('consciousness_events', []):
                print(f"\n  [consciousness] {ev}")

        # ---- Visualizer (throttled to every N ticks) -----------------------
        if use_viz and (tick % args.update_every == 0):
            try:
                frame_rgb = result.get('frame_rgb')
                import numpy as np

                # SensoryLayer stores the last audio frame in _audio_history.
                # Fall back to a generated tone buffer so the waveform panel
                # never shows a flat line even on first tick.
                _ah = getattr(brain.sense, '_audio_history', [])
                if _ah:
                    audio_frame = np.asarray(_ah[-1], dtype=np.float32)
                else:
                    from caine.sensory import FRAME_SAMPLES
                    audio_frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)

                # Build a sense_result-compatible dict for LiveVisualizer
                sense_compat = {
                    'v1_spikes':     result.get('v1_spikes'),
                    'a1_spikes':     result.get('a1_spikes'),
                    's1_rates':      result.get('s1_rates'),
                    'neuro_snapshot': result.get('neuro_snapshot', {}),
                    'orient_energy': result.get('orient_energy'),
                    'mel_energy':    result.get('mel_energy'),
                    'dog':           result.get('dog'),
                }

                # Build limbic_result for LiveVisualizer
                limbic_compat = {
                    k[len('limbic_'):]: v
                    for k, v in result.items()
                    if k.startswith('limbic_')
                }

                # Build motor_result for extend_visualizer patch
                motor_compat = {
                    k[len('motor_'):]: v
                    for k, v in result.items()
                    if k.startswith('motor_')
                }

                viz.update(
                    sense_compat,
                    frame_rgb  if frame_rgb  is not None else np.zeros((64, 64, 3), dtype=np.uint8),
                    audio_frame,
                    limbic_compat,
                    motor_compat,   # accepted after extend_visualizer patch
                )
            except Exception as e:
                log.debug("Visualizer update skipped: %s", e)

        # ---- Real-time pacing (soft; skip sleep if already behind) ---------
        elapsed = time.perf_counter() - t_start
        sleep_s = target_frame_s - elapsed
        if sleep_s > 0.0005:
            time.sleep(sleep_s)

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------
    print()  # newline after status line
    log.info("Stopping CAINE …")

    try:
        brain.stop()
    except Exception as e:
        log.error("brain.stop() raised: %s", e)

    if use_viz:
        try:
            viz.close()
        except Exception:
            pass

    if ws_viz is not None:
        try:
            ws_viz.stop()
        except Exception:
            pass

    log.info("Done.")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
