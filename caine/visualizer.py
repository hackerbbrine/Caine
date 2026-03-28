"""
CAINE Live Visualizer
======================
Renders a real-time matplotlib window that updates every simulation frame.

Layout (one figure, 3×3 grid)
------------------------------
  [0,0] CAINE camera (64x64, raw)   [0,1] DoG response        [0,2] Mel spectrum (bar)
  [1,0] Audio waveform               [1,1] V1 spike raster      [1,2] A1 spike raster
  [2,0] S1 firing rates (bar)        [2,1] Neurochemicals       [2,2] STDP scale / gain

Call `LiveVisualizer.update(result, frame_rgb, audio_frame)` each tick.
The window refreshes without blocking the simulation loop.
"""

import os
import sys as _sys
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
# Use TkAgg (or Qt5Agg) for an interactive window on Windows.
# Fall back to Agg silently so imports never crash headless environments.
_BACKENDS = ['TkAgg', 'Qt5Agg', 'WXAgg', 'Agg']
for _b in _BACKENDS:
    try:
        matplotlib.use(_b)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# History window (frames)
HISTORY = 80

# Neurochemical display colours — keys match chemicals.py snapshot() output
_CHEM_COLORS = {
    'dopamine':       '#e74c3c',
    'serotonin':      '#9b59b6',
    'cortisol':       '#3498db',
    'oxytocin':       '#e67e22',
    'norepinephrine': '#2ecc71',
    'acetylcholine':  '#1abc9c',
}
# Short labels for the legend
_CHEM_LABELS = {
    'dopamine': 'DA', 'serotonin': '5HT', 'cortisol': 'CORT',
    'oxytocin': 'OT', 'norepinephrine': 'NE', 'acetylcholine': 'ACh',
}


class LiveVisualizer:
    """
    Real-time matplotlib display for the CAINE sensory + limbic pipeline.

    Usage
    -----
        viz = LiveVisualizer()
        while running:
            result       = sense.update(frame, joints, dt_ms=20)
            limbic_result = limbic.update(dt_ms, v1_spikes, a1_spikes, snap)
            viz.update(result, frame_rgb=frame, audio_frame=audio,
                       limbic_result=limbic_result)
    """

    def __init__(self, title: str = "CAINE — Live Sensory Feed",
                 update_every: int = 1):
        """
        Parameters
        ----------
        title        : window title
        update_every : only redraw every N simulation frames (1 = every frame)
        """
        self._update_every = update_every
        self._frame_count  = 0

        # Rolling history buffers — sensory
        self._v1_history    = deque(maxlen=HISTORY)
        self._a1_history    = deque(maxlen=HISTORY)
        self._s1_history    = deque(maxlen=HISTORY)
        self._neuro_history = deque(maxlen=HISTORY)
        self._stdp_history  = deque(maxlen=HISTORY)
        self._gain_history  = deque(maxlen=HISTORY)

        # Rolling history buffers — limbic
        self._bla_history    = deque(maxlen=HISTORY)
        self._cea_history    = deque(maxlen=HISTORY)
        self._ca1_history    = deque(maxlen=HISTORY)
        self._acc_history    = deque(maxlen=HISTORY)
        self._felt_v_history = deque(maxlen=HISTORY)   # felt_valence scalar
        self._felt_a_history = deque(maxlen=HISTORY)   # felt_arousal scalar

        self._fig = None
        self._axes = {}
        self._artists = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, result: dict, frame_rgb: np.ndarray,
               audio_frame: np.ndarray,
               limbic_result: dict = None) -> None:
        """
        Refresh all panels with the latest simulation data.

        Parameters
        ----------
        result        : dict returned by SensoryLayer.update()
        frame_rgb     : (H,W,3) uint8 — raw camera frame
        audio_frame   : (N,) float32 — most recent audio buffer
        limbic_result : dict returned by LimbicSystem.update() (optional)
        """
        self._frame_count += 1

        # Accumulate sensory histories every tick regardless of draw frequency
        self._v1_history.append(result['v1_spikes'].astype(float))
        self._a1_history.append(result['a1_spikes'].astype(float))
        self._s1_history.append(result['s1_rates'])
        self._neuro_history.append(result['neuro_snapshot'])
        self._stdp_history.append(result['neuro_snapshot'].get('dopamine', 0.1))
        self._gain_history.append(result['neuro_snapshot'].get('norepinephrine', 0.1))

        # Accumulate limbic histories (zeros when limbic not running)
        if limbic_result is not None:
            self._bla_history.append(limbic_result['bla_spikes'].astype(float))
            self._cea_history.append(limbic_result['cea_spikes'].astype(float))
            self._ca1_history.append(limbic_result['ca1_spikes'].astype(float))
            self._acc_history.append(limbic_result['acc_spikes'].astype(float))
            self._felt_v_history.append(float(limbic_result.get('felt_valence', 0.0)))
            self._felt_a_history.append(float(limbic_result.get('felt_arousal', 0.0)))

        if self._frame_count % self._update_every != 0:
            return

        if not self._initialized:
            self._build_figure()

        self._draw_camera(frame_rgb)
        self._draw_dog(result['dog'])
        self._draw_mel(result['mel_energy'])
        self._draw_audio(audio_frame)
        self._draw_raster(self._axes['v1'], self._v1_history, 'V1', 'hot')
        self._draw_raster(self._axes['a1'], self._a1_history, 'A1', 'Blues')
        self._draw_s1(result['s1_rates'])
        self._draw_neuro()
        self._draw_gain()

        if limbic_result is not None:
            self._draw_stacked_raster(
                self._axes['bla'], 'bla_img',
                self._bla_history, self._cea_history)
            self._draw_stacked_raster(
                self._axes['ca1'], 'ca1_img',
                self._ca1_history, self._acc_history)
            self._draw_felt()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        plt.ion()
        self._fig = plt.figure(figsize=(15, 12), num="CAINE Live")
        self._fig.patch.set_facecolor('#111111')
        plt.get_current_fig_manager().set_window_title("CAINE — Live Sensory Feed")

        # bottom=0.27 leaves clear space for the two motor rows that
        # extend_visualizer (motor.py) injects at y=[0.01, 0.22].
        gs = gridspec.GridSpec(4, 3, figure=self._fig,
                               hspace=0.55, wspace=0.35,
                               left=0.06, right=0.97,
                               top=0.96, bottom=0.27)

        def _ax(r, c, label):
            ax = self._fig.add_subplot(gs[r, c])
            ax.set_facecolor('#1a1a1a')
            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.tick_params(colors='#aaaaaa', labelsize=7)
            ax.yaxis.label.set_color('#aaaaaa')
            ax.xaxis.label.set_color('#aaaaaa')
            ax.title.set_color('#dddddd')
            self._axes[label] = ax
            return ax

        # Row 0 — vision + mel
        ax_cam = _ax(0, 0, 'cam')
        ax_dog = _ax(0, 1, 'dog')
        ax_mel = _ax(0, 2, 'mel')

        # Row 1 — audio + rasters
        ax_wav = _ax(1, 0, 'wav')
        ax_v1  = _ax(1, 1, 'v1')
        ax_a1  = _ax(1, 2, 'a1')

        # Row 2 — S1 + neuro + gain
        ax_s1    = _ax(2, 0, 's1')
        ax_neuro = _ax(2, 1, 'neuro')
        ax_gain  = _ax(2, 2, 'gain')

        # Row 3 — Limbic: BLA/CeA rasters | CA1/ACC rasters | Felt state
        ax_bla  = _ax(3, 0, 'bla')
        ax_ca1  = _ax(3, 1, 'ca1')
        ax_felt = _ax(3, 2, 'felt')

        # ---- Static labels / layout ----
        ax_cam.set_title("Camera (CAINE POV)", fontsize=9)
        ax_cam.axis('off')

        ax_dog.set_title("DoG (ON/OFF)", fontsize=9)
        ax_dog.axis('off')

        ax_mel.set_title("Mel Spectrum", fontsize=9)
        ax_mel.set_xlabel("band", fontsize=7)
        ax_mel.set_ylabel("energy", fontsize=7)
        ax_mel.set_xlim(0, 128)
        ax_mel.set_ylim(0, 1.05)

        ax_wav.set_title("Audio Waveform", fontsize=9)
        ax_wav.set_xlabel("ms", fontsize=7)
        ax_wav.set_ylabel("amp", fontsize=7)
        ax_wav.set_ylim(-1.2, 1.2)

        ax_v1.set_title("V1 Spikes (raster)", fontsize=9)
        ax_v1.set_xlabel("time (frames)", fontsize=7)
        ax_v1.set_ylabel("neuron", fontsize=7)

        ax_a1.set_title("A1 Spikes (raster)", fontsize=9)
        ax_a1.set_xlabel("time (frames)", fontsize=7)
        ax_a1.set_ylabel("neuron", fontsize=7)

        ax_s1.set_title("S1 Firing Rates", fontsize=9)
        ax_s1.set_xlabel("neuron", fontsize=7)
        ax_s1.set_ylabel("rate", fontsize=7)
        ax_s1.set_xlim(0, 20)
        ax_s1.set_ylim(0, 1.05)

        ax_neuro.set_title("Neurochemicals", fontsize=9)
        ax_neuro.set_xlabel("frame", fontsize=7)
        ax_neuro.set_ylabel("level", fontsize=7)
        ax_neuro.set_ylim(0, 0.15)   # start tight around baseline (~0.10); expands on events

        ax_gain.set_title("DA / NE (gain proxy)", fontsize=9)
        ax_gain.set_xlabel("frame", fontsize=7)
        ax_gain.set_ylabel("level", fontsize=7)
        ax_gain.set_ylim(0, 0.15)

        # Row 3 — limbic labels
        ax_bla.set_title("BLA (top) / CeA (bot) rasters", fontsize=9)
        ax_bla.set_xlabel("time (frames)", fontsize=7)
        ax_bla.set_ylabel("neuron", fontsize=7)

        ax_ca1.set_title("CA1 (top) / ACC (bot) rasters", fontsize=9)
        ax_ca1.set_xlabel("time (frames)", fontsize=7)
        ax_ca1.set_ylabel("neuron", fontsize=7)

        ax_felt.set_title("Felt Valence / Arousal", fontsize=9)
        ax_felt.set_xlabel("frame", fontsize=7)
        ax_felt.set_ylabel("level", fontsize=7)
        ax_felt.set_ylim(-1.1, 1.1)

        # ---- Pre-create artists that will be updated in-place ----

        # Camera image
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        self._artists['cam'] = ax_cam.imshow(blank, interpolation='nearest')

        # DoG image — seismic keeps the diverging red/blue meaning but maps
        # exactly-zero to black (dark background) instead of white.
        self._artists['dog'] = ax_dog.imshow(
            np.zeros((64, 64)), cmap='seismic', vmin=-1, vmax=1,
            interpolation='nearest')

        # Mel bars
        x_mel = np.arange(128)
        self._artists['mel'] = ax_mel.bar(
            x_mel, np.zeros(128), width=1.0, color='mediumpurple', linewidth=0)

        # Audio waveform
        self._artists['wav'], = ax_wav.plot([], [], lw=0.8, color='#00ccff')

        # V1 / A1 raster heatmaps
        self._artists['v1_img'] = ax_v1.imshow(
            np.zeros((20, 1)), aspect='auto', cmap='hot',
            vmin=0, vmax=1, interpolation='nearest')
        self._artists['a1_img'] = ax_a1.imshow(
            np.zeros((20, 1)), aspect='auto', cmap='Blues',
            vmin=0, vmax=1, interpolation='nearest')

        # S1 bars
        self._artists['s1'] = ax_s1.bar(
            np.arange(20), np.zeros(20), width=0.8, color='#f39c12', linewidth=0)

        # Neurochemical lines (one per chemical)
        self._artists['neuro_lines'] = {}
        chem_names = list(_CHEM_COLORS.keys())
        for name in chem_names:
            line, = ax_neuro.plot([], [], lw=1.5,
                                  color=_CHEM_COLORS[name],
                                  label=_CHEM_LABELS.get(name, name))
            self._artists['neuro_lines'][name] = line
        ax_neuro.legend(fontsize=6, loc='upper right',
                        facecolor='#222222', labelcolor='white',
                        ncol=2, framealpha=0.7)

        # DA / NE gain lines
        self._artists['da_line'], = ax_gain.plot(
            [], [], lw=1.5, color='#e74c3c', label='DA')
        self._artists['ne_line'], = ax_gain.plot(
            [], [], lw=1.5, color='#2ecc71', label='NE')
        ax_gain.legend(fontsize=6, loc='upper right',
                       facecolor='#222222', labelcolor='white',
                       framealpha=0.7)

        # BLA / CeA stacked raster (BLA top half, CeA bottom half — concatenated)
        self._artists['bla_img'] = ax_bla.imshow(
            np.zeros((40, 1)), aspect='auto', cmap='hot',
            vmin=0, vmax=1, interpolation='nearest')

        # CA1 / ACC stacked raster — 'plasma' maps 0→near-black and 1→bright
        # yellow, so an inactive raster is dark (not the blinding cyan of 'cool').
        self._artists['ca1_img'] = ax_ca1.imshow(
            np.zeros((40, 1)), aspect='auto', cmap='plasma',
            vmin=0, vmax=1, interpolation='nearest')

        # Felt valence and arousal lines
        self._artists['felt_v'], = ax_felt.plot(
            [], [], lw=1.5, color='#f1c40f', label='valence')
        self._artists['felt_a'], = ax_felt.plot(
            [], [], lw=1.5, color='#e67e22', label='arousal')
        ax_felt.axhline(0, color='#444444', lw=0.8, linestyle='--')
        ax_felt.legend(fontsize=6, loc='upper right',
                       facecolor='#222222', labelcolor='white',
                       framealpha=0.7)

        self._fig.canvas.draw()
        plt.pause(0.001)
        self._initialized = True

    # ------------------------------------------------------------------
    # Per-panel draw helpers
    # ------------------------------------------------------------------

    def _draw_camera(self, frame_rgb):
        self._artists['cam'].set_data(frame_rgb)

    def _draw_dog(self, dog):
        self._artists['dog'].set_data(dog)

    def _draw_mel(self, mel_norm):
        for bar, h in zip(self._artists['mel'], mel_norm):
            bar.set_height(float(h))

    def _draw_audio(self, audio):
        from caine.sensory import SAMPLE_RATE
        n = len(audio)
        t_ms = np.linspace(0, n / SAMPLE_RATE * 1000, n)
        self._artists['wav'].set_data(t_ms, audio)
        self._axes['wav'].set_xlim(0, t_ms[-1])

    def _draw_raster(self, ax, history, key, cmap):
        img_key = key.lower() + '_img'
        if not history:
            return
        mat = np.array(history).T  # (neurons, time)
        self._artists[img_key].set_data(mat)
        self._artists[img_key].set_extent(
            [0, mat.shape[1], mat.shape[0], 0])
        ax.set_xlim(0, mat.shape[1])
        ax.set_ylim(mat.shape[0], 0)

    def _draw_s1(self, rates):
        for bar, r in zip(self._artists['s1'], rates):
            bar.set_height(float(r))

    def _draw_neuro(self):
        if not self._neuro_history:
            return
        times = np.arange(len(self._neuro_history))
        all_vals = []
        for name, line in self._artists['neuro_lines'].items():
            vals = [snap.get(name, 0) for snap in self._neuro_history]
            line.set_data(times, vals)
            all_vals.extend(vals)
        ax = self._axes['neuro']
        ax.set_xlim(0, max(1, len(times) - 1))
        # Auto-scale y: always show at least 0–0.15, expand to fit peaks
        peak = max(all_vals) if all_vals else 0.15
        ax.set_ylim(0, max(0.15, peak * 1.15))

    def _draw_gain(self):
        if not self._neuro_history:
            return
        times = np.arange(len(self._neuro_history))
        da_vals = [snap.get('dopamine', 0) for snap in self._neuro_history]
        ne_vals = [snap.get('norepinephrine', 0) for snap in self._neuro_history]
        self._artists['da_line'].set_data(times, da_vals)
        self._artists['ne_line'].set_data(times, ne_vals)
        ax = self._axes['gain']
        ax.set_xlim(0, max(1, len(times) - 1))
        peak = max(max(da_vals), max(ne_vals)) if da_vals else 0.15
        ax.set_ylim(0, max(0.15, peak * 1.15))

    def _draw_stacked_raster(self, ax, img_key, hist_top, hist_bot):
        """Stack two population histories vertically into one raster image."""
        if not hist_top and not hist_bot:
            return
        rows_top = np.array(hist_top) if hist_top else np.zeros((1, 20))
        rows_bot = np.array(hist_bot) if hist_bot else np.zeros((1, 20))
        # Ensure same number of time steps
        n_t = max(rows_top.shape[0], rows_bot.shape[0])
        def _pad_t(mat, n):
            if mat.shape[0] < n:
                mat = np.vstack([np.zeros((n - mat.shape[0], mat.shape[1])), mat])
            return mat
        rows_top = _pad_t(rows_top, n_t)
        rows_bot = _pad_t(rows_bot, n_t)
        # (time, n_top) and (time, n_bot) → stack neurons → (n_top+n_bot, time)
        mat = np.vstack([rows_top.T, rows_bot.T])
        self._artists[img_key].set_data(mat)
        self._artists[img_key].set_extent([0, mat.shape[1], mat.shape[0], 0])
        ax.set_xlim(0, mat.shape[1])
        ax.set_ylim(mat.shape[0], 0)

    def _draw_felt(self):
        """Draw felt valence and arousal time series."""
        if not self._felt_v_history:
            return
        times = np.arange(len(self._felt_v_history))
        self._artists['felt_v'].set_data(times, list(self._felt_v_history))
        self._artists['felt_a'].set_data(times, list(self._felt_a_history))
        ax = self._axes['felt']
        ax.set_xlim(0, max(1, len(times) - 1))
