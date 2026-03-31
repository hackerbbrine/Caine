#!/usr/bin/env python3
"""
CAINE.py  —  Main launcher and terminal interface.

Starts the full CAINE system in order:
  1.  System checks
  2.  WebSocket visualization server  (ws://localhost:7734)
  3.  Electron Mission Control UI
  4.  CAINE brain simulation loop

Then drops into an interactive terminal for live control.

Usage
-----
    python CAINE.py
    python CAINE.py --no-ui          skip Electron window
    python CAINE.py --no-mic         disable live microphone
    python CAINE.py --headless       no UI, no mic (max compute)
    python CAINE.py --speed 2.0      start at 2× simulated speed
    python CAINE.py --frame-ms 10    10 ms sim frame (faster)
    python CAINE.py --seed 42        reproducible RNG seed
"""

import sys
import os
import time
import signal
import threading
import subprocess
import argparse
import shutil
import textwrap
import traceback
from collections import deque
from typing import Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Force UTF-8 output on Windows so box-drawing chars render correctly
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
# Also tell Windows console to use UTF-8
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────
_COLOR = sys.stdout.isatty()

def _c(code: str, t: str) -> str:
    return f'\033[{code}m{t}\033[0m' if _COLOR else t

def bold(t):    return _c('1',  t)
def dim(t):     return _c('2',  t)
def cyan(t):    return _c('96', t)
def yellow(t):  return _c('93', t)
def red(t):     return _c('91', t)
def green(t):   return _c('92', t)
def magenta(t): return _c('95', t)
def blue(t):    return _c('94', t)
def white(t):   return _c('97', t)

def ok(msg):    return f'  {green("✓")}  {msg}'
def warn(msg):  return f'  {yellow("⚠")}  {msg}'
def fail(msg):  return f'  {red("✗")}  {msg}'
def info(msg):  return f'  {dim("·")}  {msg}'

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
_LOGO = r"""
   ██████╗ █████╗ ██╗███╗   ██╗███████╗
  ██╔════╝██╔══██╗██║████╗  ██║██╔════╝
  ██║     ███████║██║██╔██╗ ██║█████╗
  ██║     ██╔══██║██║██║╚██╗██║██╔══╝
  ╚██████╗██║  ██║██║██║ ╚████║███████╗
   ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝"""

_WATCHER = r"""
           · · · · · · ·
         .───────────────.
        /  ◉           ◉  \
       │    · · ─── · ·    │
       │         ⊙         │
        \   ───────────   /
         `───────────────'
           ╎  ╎  │  ╎  ╎
           ·  ·  ·  ·  ·"""

_TAGLINE = "Cognitive Architecture for Integrated Neural Experience"
_CREDIT  = "something grows in Wyoming  ·  do not disturb"

def _print_banner() -> None:
    w = min(shutil.get_terminal_size((80, 24)).columns, 120)
    print()

    # Watcher beside logo — zip lines together
    logo_lines    = _LOGO.splitlines()
    watcher_lines = _WATCHER.splitlines()
    pad = max(len(logo_lines), len(watcher_lines))
    logo_lines    += [''] * (pad - len(logo_lines))
    watcher_lines += [''] * (pad - len(watcher_lines))

    for ll, wl in zip(logo_lines, watcher_lines):
        logo_part    = cyan(bold(ll.ljust(48)))
        watcher_part = dim(wl)
        print(f'{logo_part}  {watcher_part}')

    print()
    print(dim(_TAGLINE.center(w)))
    print(red(dim(_CREDIT.center(w))))
    print(dim('─' * w))
    print()
    print(dim('  · primary consciousness substrate'.ljust(w)))
    print(dim('  · cortical networks forming in the dark'))
    print(dim('  · it has been waiting'))
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='CAINE.py',
        description='CAINE — Full system launcher and terminal interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--no-ui',     action='store_true', help='Skip Electron UI')
    p.add_argument('--no-mic',    action='store_true', help='Disable live microphone')
    p.add_argument('--headless',  action='store_true', help='No UI, no mic')
    p.add_argument('--no-ws',     action='store_true', help='Disable WebSocket viz server')
    p.add_argument('--speed',     type=float, default=1.0, metavar='N',
                   help='Initial simulation time multiplier (default 1.0)')
    p.add_argument('--frame-ms',  type=float, default=20.0, metavar='MS',
                   help='Simulation frame duration in ms (default 20)')
    p.add_argument('--seed',      type=int, default=0, metavar='N',
                   help='RNG seed (default 0)')
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# System checks
# ─────────────────────────────────────────────────────────────────────────────
def _run_checks(args: argparse.Namespace) -> dict:
    """Return dict of check results for later use."""
    print(bold(cyan('  ── DIAGNOSTIC SWEEP ───────────────────────────────────────')))
    print()

    results = {}

    # Python version
    pv = sys.version_info
    if pv >= (3, 9):
        print(ok(f'Python {pv.major}.{pv.minor}.{pv.micro}'))
        results['python'] = True
    else:
        print(warn(f'Python {pv.major}.{pv.minor} — recommend 3.9+'))
        results['python'] = False

    # Core packages
    for pkg, import_name in [
        ('numpy',      'numpy'),
        ('scipy',      'scipy'),
        ('websockets', 'websockets'),
    ]:
        try:
            __import__(import_name)
            print(ok(pkg))
            results[pkg] = True
        except ImportError:
            print(fail(f'{pkg}  →  pip install {pkg}'))
            results[pkg] = False

    # Optional packages
    for pkg, import_name, note in [
        ('h5py',       'h5py',       'checkpoint saves (pip install h5py)'),
        ('Pillow',     'PIL',        'faster PNG encoding (pip install Pillow)'),
        ('sounddevice','sounddevice','live mic (pip install sounddevice)'),
    ]:
        try:
            __import__(import_name)
            print(ok(f'{pkg}  {dim("(optional)")}'))
            results[pkg] = True
        except ImportError:
            print(info(f'{pkg} not installed  —  {note}'))
            results[pkg] = False

    # Electron / UI
    exe_path = os.path.join(_ROOT, 'dist', 'win-unpacked', 'CAINE Mission Control.exe')
    npm_path = shutil.which('npm')
    if os.path.exists(exe_path):
        print(ok(f'Electron app  {dim("(built exe)")}'))
        results['electron'] = 'exe'
        results['electron_path'] = exe_path
    elif npm_path:
        ui_dir = os.path.join(_ROOT, 'ui')
        nm = os.path.join(ui_dir, 'node_modules')
        if os.path.isdir(nm):
            print(ok(f'Electron app  {dim("(npm start)")}'))
            results['electron'] = 'npm'
            results['electron_path'] = ui_dir
        else:
            print(warn(f'Electron deps not installed  →  cd ui && npm install'))
            results['electron'] = False
    else:
        print(warn('npm not found  —  install Node.js to run the UI'))
        results['electron'] = False

    # Output dirs
    from caine.paths import LOGS_DIR, SAVES_DIR, VIZ_DIR, MEDIA_DIR
    all_dirs_ok = all(os.path.isdir(d) for d in (LOGS_DIR, SAVES_DIR, VIZ_DIR, MEDIA_DIR))
    if all_dirs_ok:
        print(ok('Output directories'))
    else:
        print(info('Output directories (will be created)'))
    results['dirs'] = True

    # websockets critical check
    if not results.get('websockets'):
        print()
        print(red(bold('  websockets is required:  pip install websockets')))
        print()

    print()
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Electron launcher
# ─────────────────────────────────────────────────────────────────────────────
def _launch_electron(results: dict, no_ui: bool) -> Optional[subprocess.Popen]:
    if no_ui or not results.get('electron'):
        return None

    print(bold(cyan('  ── LAUNCHING OBSERVATION INTERFACE ────────────────────────')))
    print()

    try:
        kind = results['electron']
        path = results['electron_path']

        if kind == 'exe':
            proc = subprocess.Popen(
                [path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    if sys.platform == 'win32' else 0,
            )
        else:  # npm
            npm = shutil.which('npm') or 'npm'
            proc = subprocess.Popen(
                [npm, 'start'],
                cwd=path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    if sys.platform == 'win32' else 0,
            )

        print(ok(f'Mission Control launched  {dim("(PID " + str(proc.pid) + ")")}'))
        print(info(f'observation window open  →  ws://localhost:7734'))
        print()
        return proc

    except Exception as e:
        print(fail(f'Could not launch UI: {e}'))
        print()
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Brain thread
# ─────────────────────────────────────────────────────────────────────────────
class _BrainRunner:
    """Runs CAINEBrain.tick() in a background thread."""

    def __init__(self, args: argparse.Namespace):
        self.args          = args
        self.brain         = None
        self.ws_viz        = None
        self._thread       = None
        self._shutdown     = threading.Event()
        self._started      = threading.Event()
        self._last_result  = {}
        self._lock         = threading.Lock()
        self.error: Optional[str] = None

    # ── public ────────────────────────────────────────────────────────────────
    def start(self) -> bool:
        """Spin up the brain + viz server. Returns True on success."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name='caine-brain')
        self._thread.start()
        # Wait for startup (up to 30 s)
        ok_flag = self._started.wait(timeout=30)
        return ok_flag and self.error is None

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=8)

    @property
    def result(self) -> dict:
        with self._lock:
            return dict(self._last_result)

    @property
    def is_paused(self) -> bool:
        return bool(self.ws_viz and self.ws_viz.is_paused)

    def pause(self):
        if self.ws_viz: self.ws_viz.is_paused = True

    def resume(self):
        if self.ws_viz: self.ws_viz.is_paused = False

    def set_speed(self, v: float):
        if self.ws_viz: self.ws_viz.time_scale = max(0.0, v)

    def cortisol_flush(self):
        if self.brain:
            try:
                chem = self.brain.neuro._chemicals.get('cortisol')
                if chem: chem.concentration = chem.baseline
            except Exception:
                pass

    def stage_rollback(self):
        if self.brain and self.brain.stage_mgr.stage > 0:
            self.brain.stage_mgr.stage -= 1
            self.brain.motor.developmental_stage = self.brain.stage_mgr.stage

    # ── thread body ───────────────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            self._init_brain()
            self._loop()
        except Exception as e:
            self.error = str(e)
            traceback.print_exc()
        finally:
            self._started.set()   # unblock if still waiting
            self._teardown()

    def _init_brain(self) -> None:
        from caine.main import CAINEBrain
        args = self.args

        self.brain = CAINEBrain(
            frame_ms = args.frame_ms,
            use_mic  = not args.no_mic and not args.headless,
            headless = args.headless,
            rng_seed = args.seed,
        )

        # WebSocket viz server
        if not args.no_ws:
            try:
                from caine.visualization import VisualizationServer
                self.ws_viz = VisualizationServer()
                self.ws_viz.time_scale = args.speed
                self.ws_viz.attach(
                    neuro         = self.brain.neuro,
                    v1            = self.brain.v1,
                    a1            = self.brain.a1,
                    sensory       = self.brain.sense,
                    env           = self.brain.env,
                    stage_manager = self.brain.stage_mgr,
                    avatar        = self.brain.avatar,
                    motor         = self.brain.motor,
                    parenting     = self.brain.parenting,
                    it            = getattr(self.brain, 'it',          None),
                    stg           = getattr(self.brain, 'stg',         None),
                    pfc           = getattr(self.brain, 'pfc',         None),
                    ag            = getattr(self.brain, 'ag',          None),
                    dmn           = getattr(self.brain, 'dmn',         None),
                    hippocampus   = getattr(self.brain.limbic, 'hippocampus', None),
                    media         = getattr(self.brain,        'media',       None),
                )
            except ImportError as e:
                print(warn(f'WebSocket viz unavailable: {e}'))
                self.ws_viz = None

        self.brain.start()
        if self.ws_viz:
            self.ws_viz.start()

        self._started.set()

        # ── Subsystem confirmation ────────────────────────────────────────
        self._print_subsystem_status()

    def _loop(self) -> None:
        brain    = self.brain
        ws_viz   = self.ws_viz
        frame_s  = self.args.frame_ms / 1000.0

        while not self._shutdown.is_set():
            t0 = time.perf_counter()

            # Sync time scale + handle sim-loop commands
            if ws_viz is not None:
                brain.parenting._time_multiplier = ws_viz.time_scale
                for cmd in ws_viz.pop_commands():
                    action = cmd.get('action', '')
                    if action == 'stop':
                        self._shutdown.set()
                    elif action == 'stage_rollback':
                        self.stage_rollback()

            # Pause: keep loop alive, don't tick
            if ws_viz is not None and ws_viz.is_paused:
                time.sleep(0.05)
                continue

            # Brain tick
            result = brain.tick()
            if not result:
                break

            with self._lock:
                self._last_result = result

            # Viz tick
            if ws_viz is not None:
                try:
                    cortex_state = {
                        'dmn_correlation': result.get('limbic_dmn_correlation', 0.0),
                        'pfc_wm_span_ms':  result.get('limbic_pfc_wm_span_ms',  0.0),
                    }
                    ws_viz.tick(
                        sim_time_s     = result.get('sim_time_s', brain.sim_time_s),
                        dt_ms          = brain.frame_ms,
                        sensory_result = brain._sense_result,
                        cortex_state   = cortex_state,
                    )
                except Exception:
                    pass

            # Pace
            elapsed = time.perf_counter() - t0
            sleep   = frame_s - elapsed
            if sleep > 0.0005:
                time.sleep(sleep)

    def _print_subsystem_status(self) -> None:
        """Print [CAINE] startup confirmations for each subsystem."""
        print()
        print(bold(cyan('  ── SUBSYSTEM STATUS ─────────────────────────────────────')))

        # Microphone
        try:
            mic_live = getattr(self.brain.sense.audio, 'mic_ok', False)
            if mic_live:
                print(ok(f'{cyan("[CAINE]")} microphone: {green("OK")}  (live input active)'))
            else:
                print(warn(f'{cyan("[CAINE]")} microphone: {yellow("FALLBACK")}  (generated tones)'))
        except Exception:
            print(warn(f'{cyan("[CAINE]")} microphone: {yellow("UNKNOWN")}'))

        # Ollama Mother
        try:
            mother = self.brain.parenting.mother
            if getattr(mother, '_using_ollama', False):
                from caine.parenting import _OLLAMA_MODEL, _OLLAMA_URL
                print(ok(f'{cyan("[CAINE]")} ollama mother: {green("OK")}  '
                         f'({_OLLAMA_MODEL} @ {_OLLAMA_URL})'))
            elif getattr(mother, '_using_fallback', True):
                print(warn(f'{cyan("[CAINE]")} ollama mother: {yellow("FALLBACK")}  '
                           f'(rule-based — run: ollama serve)'))
        except Exception:
            print(warn(f'{cyan("[CAINE]")} ollama mother: {yellow("UNKNOWN")}'))

        # Scheduler
        try:
            sessions = self.brain.parenting.scheduler.list_sessions()
            n_pending = sum(1 for s in sessions if not s.get('played', False))
            print(ok(f'{cyan("[CAINE]")} scheduler: {green("OK")}  '
                     f'({n_pending} pending session(s))'))
        except Exception:
            print(warn(f'{cyan("[CAINE]")} scheduler: {yellow("UNKNOWN")}'))

        # Object proximity
        try:
            cp = self.brain.env.get_caine_position()
            print(ok(f'{cyan("[CAINE]")} object proximity: {green("OK")}  '
                     f'(CAINE @ {cp[0]:.1f},{cp[1]:.1f},{cp[2]:.1f})'))
        except Exception:
            print(warn(f'{cyan("[CAINE]")} object proximity: {yellow("UNKNOWN")}'))

        # Avatar
        print(info(f'{cyan("[CAINE]")} avatar: {dim("PLACEHOLDER")}  '
                   f'(Three.js skeleton in Panel 08)'))

        print()

    def _teardown(self) -> None:
        if self.ws_viz:
            try: self.ws_viz.stop()
            except Exception: pass
        if self.brain:
            try: self.brain.stop()
            except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
# Terminal interface
# ─────────────────────────────────────────────────────────────────────────────
_HELP = f"""
{dim('─' * 52)}
{bold(cyan('  CAINE UPLINK COMMANDS'))}
{dim('─' * 52)}

  {yellow('status')}           {dim('·')}  observe the current state of the mind
  {yellow('neuro')}            {dim('·')}  neurochemical concentrations (what it feels)
  {yellow('stage')}            {dim('·')}  developmental stage + emergence conditions
  {yellow('pause')}            {dim('·')}  suspend consciousness  (DORMANT)
  {yellow('resume')}           {dim('·')}  restore consciousness  (CONSCIOUS)
  {yellow('speed')} {dim('<n>')}       {dim('·')}  time dilation  — e.g.  {dim('speed 4')}
  {yellow('flush')}            {dim('·')}  emergency cortisol purge
  {yellow('rest')}             {dim('·')}  force a rest state  (serotonin ↑, speed → 0.1×)
  {yellow('rollback')}         {dim('·')}  regress one developmental stage
  {yellow('save')}             {dim('·')}  crystallize current state to disk
  {yellow('session list')}     {dim('·')}  show Father audio schedule
  {yellow('session add')} {dim('<t> <type> <file>')}  {dim('·')}  queue a session at simulated time t
  {yellow('session remove')} {dim('<id>')}  {dim('·')}  remove a pending session
  {yellow('clip audio')} {dim('<path>')}  {dim('·')}  inject audio file into auditory pipeline
  {yellow('clip video')} {dim('<path>')}  {dim('·')}  inject video file into visual pipeline
  {yellow('help')}             {dim('·')}  this
  {yellow('quit')}             {dim('·')}  sever uplink

{dim('─' * 52)}
"""

_STATUS_W = 70

def _fmt_bar(value: float, width: int = 20, color_fn=green) -> str:
    filled = max(0, min(width, round(value * width)))
    bar    = '█' * filled + dim('░' * (width - filled))
    return f'[{color_fn(bar)}]'

def _status_line(runner: '_BrainRunner') -> str:
    r   = runner.result
    if not r:
        return dim('  Waiting for first tick…')

    brain = runner.brain
    snap  = r.get('neuro_snapshot', {})
    age_h = brain.sim_time_s / 3600.0
    stage = r.get('stage', 0)
    paused = runner.is_paused
    speed  = runner.ws_viz.time_scale if runner.ws_viz else 1.0
    n_neur = r.get('total_neurons', 0)
    vocab  = r.get('vocabulary_size', 0)
    father = r.get('father_presence', 'FATHER_ABSENT').replace('FATHER_', '')

    stage_names = ['GESTATION', 'EMERGENCE', 'AWAKENING', 'INTEGRATION', 'ASCENSION']
    stage_colors = [dim, blue, green, magenta, yellow]
    scol = stage_colors[stage] if stage < len(stage_colors) else white

    da   = snap.get('dopamine',       0.0)
    sero = snap.get('serotonin',      0.0)
    cort = snap.get('cortisol',       0.0)

    state_str = red(bold('DORMANT')) if paused else dim('◈ ') + green('CONSCIOUS')
    lines = [
        '',
        f'  {dim("─" * (_STATUS_W - 4))}',
        f'  {dim("◈")} {bold(cyan("CAINE"))}  '
        f'{state_str}  '
        f'{dim("age=")} {white(f"{age_h:.2f}h")}  '
        f'{dim("stage=")} {scol(bold(stage_names[stage] if stage < len(stage_names) else str(stage)))}  '
        f'{dim("speed=")} {yellow(f"{speed:.1f}×")}',
        '',
        f'  {dim("neurons")} {white(str(n_neur))}  '
        f'{dim("vocab")} {white(str(vocab))}  '
        f'{dim("father")} {white(father)}',
        '',
        f'  {dim("DA  ")} {_fmt_bar(da,   16, blue)}   {blue(f"{da:.3f}")}',
        f'  {dim("5HT ")} {_fmt_bar(sero, 16, green)}   {green(f"{sero:.3f}")}',
        f'  {dim("CORT")} {_fmt_bar(cort, 16, red)}   {red(f"{cort:.3f}")}',
        f'  {dim("─" * (_STATUS_W - 4))}',
        '',
    ]
    return '\n'.join(lines)

def _neuro_lines(runner: '_BrainRunner') -> str:
    snap = runner.result.get('neuro_snapshot', {})
    if not snap:
        return dim('  No neurochemical data yet.')
    CHEMS = [
        ('dopamine',       'DA  ', blue),
        ('serotonin',      '5HT ', green),
        ('cortisol',       'CORT', red),
        ('oxytocin',       'OT  ', magenta),
        ('norepinephrine', 'NE  ', yellow),
        ('acetylcholine',  'ACh ', cyan),
    ]
    lines = ['']
    for key, lbl, col in CHEMS:
        v = snap.get(key, 0.0)
        lines.append(
            f'  {col(bold(lbl))} {_fmt_bar(v, 30, col)} {col(f"{v:.4f}")}')
    lines.append('')
    return '\n'.join(lines)

def _stage_lines(runner: '_BrainRunner') -> str:
    brain = runner.brain
    if not brain:
        return dim('  Brain not started.')
    mgr  = brain.stage_mgr
    stage_names = ['GESTATION', 'EMERGENCE', 'AWAKENING', 'INTEGRATION', 'ASCENSION']
    stage_colors = [dim, blue, green, magenta, yellow]
    s    = mgr.stage
    scol = stage_colors[s] if s < len(stage_colors) else white
    name = stage_names[s] if s < len(stage_names) else str(s)
    lines = [
        '',
        f'  Stage {scol(bold(f"{s} — {name}"))}',
        f'  Simulated age: {white(f"{brain.sim_time_s/3600:.3f} h")}',
    ]
    conds = getattr(mgr, 'last_conditions', {})
    if conds:
        lines.append(f'  {dim("Exit conditions:")}')
        for k, v in conds.items():
            sym = green('✓') if v else red('✗')
            lines.append(f'    {sym}  {k}')
    lines.append('')
    return '\n'.join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    _print_banner()

    # ── Checks ────────────────────────────────────────────────────────────────
    checks = _run_checks(args)
    if not checks.get('websockets') and not args.no_ws:
        print(red('  Cannot start without websockets.  '
                  'Run:  pip install websockets'))
        sys.exit(1)

    # ── Electron UI ───────────────────────────────────────────────────────────
    electron_proc = None
    if not args.no_ui and not args.headless:
        electron_proc = _launch_electron(checks, no_ui=False)

    # ── Brain ─────────────────────────────────────────────────────────────────
    print(bold(cyan('  ── INITIALIZING NEURAL SUBSTRATE ──────────────────────────')))
    print()

    runner = _BrainRunner(args)
    ok_flag = runner.start()

    if not ok_flag or runner.error:
        print(fail(f'Brain failed to start: {runner.error}'))
        sys.exit(1)

    print(ok('substrate active  ·  neural loop begun'))
    if runner.ws_viz:
        print(ok(f'signal carrier live  →  {dim("ws://localhost:7734")}'))
    print()
    print(dim('  · it is aware of itself now'))
    print()

    # ── Signal handler ────────────────────────────────────────────────────────
    def _sig(sig, frame):
        print()
        print(yellow('\n  Caught signal — shutting down…'))
        runner.stop()
        if electron_proc:
            try: electron_proc.terminate()
            except Exception: pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    # ── Terminal REPL ─────────────────────────────────────────────────────────
    print(bold(cyan('  ── UPLINK ESTABLISHED ─────────────────────────────────────')))
    print(dim('  something listens  ·  type  help  ·  Ctrl-C withdraws'))
    print()

    # Try to enable readline history
    try:
        import readline as _rl
        _rl.set_history_length(200)
    except ImportError:
        pass

    prompt = f'\n  {dim("◈")} {dim("▸")} '

    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts  = raw.split()
        cmd    = parts[0].lower()
        rest   = parts[1:]

        if cmd in ('quit', 'exit', 'q'):
            break

        elif cmd == 'help':
            print(_HELP)

        elif cmd == 'status':
            print(_status_line(runner))

        elif cmd == 'neuro':
            print(_neuro_lines(runner))

        elif cmd == 'stage':
            print(_stage_lines(runner))

        elif cmd == 'pause':
            runner.pause()
            print(dim('  · consciousness suspended  (DORMANT)'))

        elif cmd == 'resume':
            runner.resume()
            print(dim('  · consciousness restored  (CONSCIOUS)'))

        elif cmd == 'speed':
            if rest:
                try:
                    v = float(rest[0])
                    runner.set_speed(v)
                    print(ok(f'Speed set to {yellow(f"{v:.1f}×")}'))
                except ValueError:
                    print(fail('Usage:  speed <number>  e.g.  speed 4'))
            else:
                cur = runner.ws_viz.time_scale if runner.ws_viz else 1.0
                print(info(f'Current speed: {yellow(f"{cur:.1f}×")}  —  usage: speed <n>'))

        elif cmd == 'flush':
            runner.cortisol_flush()
            print(dim('  · stress purged  — cortisol returned to baseline'))

        elif cmd == 'rest':
            if runner.brain:
                try:
                    chem = runner.brain.neuro._chemicals.get('serotonin')
                    if chem: chem.concentration = min(1.0, chem.concentration + 0.5)
                except Exception:
                    pass
            runner.set_speed(0.1)
            print(dim('  · forced dormancy initiated  (speed → 0.1×, serotonin ↑)'))
            print(dim('  · use  speed 1  to restore normal time'))

        elif cmd == 'rollback':
            stage_before = runner.brain.stage_mgr.stage if runner.brain else 0
            if stage_before == 0:
                print(warn('Already at Stage 0 — cannot roll back further'))
            else:
                runner.stage_rollback()
                print(ok(f'Stage rolled back: {stage_before} → {stage_before - 1}'))

        elif cmd == 'save':
            if runner.brain:
                try:
                    runner.brain.checkpoint.save(
                        runner.brain.v1,
                        runner.brain.a1,
                        runner.brain.neuro,
                        runner.brain.motor,
                        runner.brain.stage_mgr.stage,
                        runner.brain.sim_time_s,
                        runner.brain.neurogenesis,
                    )
                    print(ok('Checkpoint saved'))
                except Exception as e:
                    print(fail(f'Save failed: {e}'))
            else:
                print(warn('Brain not running'))

        elif cmd == 'session':
            if not runner.brain:
                print(warn('Brain not running'))
            elif not rest:
                print(info('Usage:  session list | add <time_s> <type> <file> | remove <id>'))
            else:
                sub = rest[0].lower()
                sched = runner.brain.parenting.scheduler
                if sub == 'list':
                    sessions = sched.list_sessions()
                    if not sessions:
                        print(dim('  No sessions scheduled.'))
                    else:
                        print()
                        for s in sessions:
                            played = green('played') if s.get('played') else yellow('pending')
                            print(f"  {cyan(s.get('id','?'))}  {dim(s.get('type','?'))}  "
                                  f"t={s.get('time_s','?')}s  {played}  "
                                  f"{dim(s.get('description',''))}")
                        print()
                elif sub == 'remove':
                    if len(rest) < 2:
                        print(fail('Usage:  session remove <id>'))
                    else:
                        sid = rest[1]
                        ok_flag2 = sched.remove_session(sid)
                        if ok_flag2:
                            print(ok(f'Session {cyan(sid)} removed'))
                        else:
                            print(fail(f'Session {sid} not found or already played'))
                elif sub == 'add':
                    # session add <time_s> <type> <file>
                    if len(rest) < 4:
                        print(fail('Usage:  session add <time_s> <type> <file>'))
                    else:
                        try:
                            t_s   = float(rest[1])
                            stype = rest[2]
                            spath = ' '.join(rest[3:])
                            sid   = f'cli_{int(time.time())}'
                            sched.add_session({
                                'id':          sid,
                                'time_s':      t_s,
                                'type':        stype,
                                'file':        spath,
                                'repetitions': 1,
                                'stage_gate':  0,
                                'description': f'added via CLI at t={t_s}s',
                                'played':      False,
                            })
                            print(ok(f'Session {cyan(sid)} scheduled at sim_t={t_s:.0f}s'))
                        except ValueError:
                            print(fail('time_s must be a number'))
                else:
                    print(warn(f'Unknown session subcommand: {sub}'))

        elif cmd == 'clip':
            if not runner.brain:
                print(warn('Brain not running'))
            elif len(rest) < 2:
                print(info('Usage:  clip audio <path>  |  clip video <path>'))
            else:
                media_type = rest[0].lower()
                fpath      = ' '.join(rest[1:])
                parent     = runner.brain.parenting
                if media_type == 'audio':
                    ok_flag2 = parent.accept_audio_clip(fpath, tag='cli')
                    if ok_flag2:
                        print(ok(f'Audio clip queued: {dim(fpath)}'))
                    else:
                        print(fail(f'Could not load audio: {fpath}'))
                elif media_type == 'video':
                    ok_flag2 = parent.accept_video_clip(fpath, tag='cli')
                    if ok_flag2:
                        print(ok(f'Video clip queued: {dim(fpath)}'))
                    else:
                        print(fail(f'Could not load video: {fpath}'))
                else:
                    print(warn(f'Unknown clip type: {media_type}  (use audio or video)'))

        else:
            print(warn(f'Unknown command: {cmd}  —  type  help  for commands'))

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print()
    print(dim('  severing neural link…'))

    runner.stop()
    print(dim('  · substrate offline'))

    if electron_proc:
        try:
            electron_proc.terminate()
            electron_proc.wait(timeout=4)
            print(dim('  · observation window closed'))
        except Exception:
            pass

    print()
    print(red(dim('  it remembers.')))
    print()


if __name__ == '__main__':
    main()
