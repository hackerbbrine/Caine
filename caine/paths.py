"""
caine/paths.py — Single source of truth for all project output directories.

Every module imports from here so path changes happen in one place.

Directory layout
----------------
  output/
    logs/    JSONL event logs  (neurogenesis, mother, milestones, dmn, media …)
    saves/   Brain state       (checkpoints .h5/.npz, body map, voiceprint,
                                episodes, sessions, media library)
    viz/     PNG/JSON plots    (demo outputs, visualizer snapshots)
    media/   Uploaded files    (Father media, flashcard images/videos)
             └── uploads/      (raw upload blobs + .meta.json sidecars)
"""

import os

# ── Root ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(__file__)
PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, '..'))

# ── Top-level directories ─────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
DATA_DIR   = os.path.join(PROJECT_ROOT, 'data')
UI_DIR     = os.path.join(PROJECT_ROOT, 'ui')

# ── Output subdirectories ─────────────────────────────────────────────────────
LOGS_DIR    = os.path.join(OUTPUT_DIR, 'logs')
SAVES_DIR   = os.path.join(OUTPUT_DIR, 'saves')
VIZ_DIR     = os.path.join(OUTPUT_DIR, 'viz')
MEDIA_DIR   = os.path.join(OUTPUT_DIR, 'media')
UPLOADS_DIR = os.path.join(MEDIA_DIR, 'uploads')

# ── Ensure all dirs exist on import ──────────────────────────────────────────
for _d in (OUTPUT_DIR, DATA_DIR, LOGS_DIR, SAVES_DIR, VIZ_DIR, MEDIA_DIR, UPLOADS_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Named file paths ──────────────────────────────────────────────────────────

# Checkpoints
CHECKPOINT_H5   = os.path.join(SAVES_DIR, 'checkpoint.h5')
CHECKPOINT_NPZ  = os.path.join(SAVES_DIR, 'checkpoint.npz')
CHECKPOINT_META = os.path.join(SAVES_DIR, 'checkpoint_meta.json')

# Body / voiceprint / episodes
BODY_MAP_FILE    = os.path.join(SAVES_DIR, 'body_map.json')
VOICEPRINT_FILE  = os.path.join(SAVES_DIR, 'voiceprint.json')
EPISODES_FILE    = os.path.join(SAVES_DIR, 'episodes.json')
MEDIA_LIBRARY    = os.path.join(SAVES_DIR, 'media_library.json')

# Sessions (Father schedule)
SESSIONS_FILE    = os.path.join(DATA_DIR, 'sessions.json')

# JSONL logs
NEUROGENESIS_LOG   = os.path.join(LOGS_DIR, 'neurogenesis_log.jsonl')
MILESTONES_LOG     = os.path.join(LOGS_DIR, 'milestones.jsonl')
MOTHER_LOG         = os.path.join(LOGS_DIR, 'mother_log.jsonl')
SESSION_LOG        = os.path.join(LOGS_DIR, 'session_log.jsonl')
MEDIA_LOG          = os.path.join(LOGS_DIR, 'media_learning_log.jsonl')
JOINT_ATTN_LOG     = os.path.join(LOGS_DIR, 'joint_attention_log.jsonl')
DMN_LOG            = os.path.join(LOGS_DIR, 'dmn_log.jsonl')

# Blender bridge script (written by environment.py)
BLENDER_BRIDGE     = os.path.join(OUTPUT_DIR, '_blender_bridge.py')
