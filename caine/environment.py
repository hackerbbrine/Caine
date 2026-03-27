"""
CAINE Training Environment — Module E
======================================
A minimal OpenGL sandbox implemented with ModernGL + PyBullet.

This is the digital void CAINE is born into: an infinite black space with a
ground plane, one adjustable ambient light, two cameras, and basic rigid-body
physics.  The public API surface matches the WorldAPI defined in the README so
the same interface can later be mirrored in Unity.

Dependencies
------------
    pip install moderngl moderngl-window PyBullet numpy

Usage
-----
    from caine.environment import CaineEnvironment

    env = CaineEnvironment()
    env.start()                         # open window + physics thread

    handle = env.spawn_object('sphere', (0, 1, 0))
    for _ in range(300):
        feed = env.get_camera_feed()    # 64x64 RGB numpy array
        env.step()                      # advance physics + render one frame
    env.stop()
"""

import math
import os
import sys as _sys
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import struct
import threading
import time
from collections import namedtuple
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully so the rest of CAINE still imports
# even when ModernGL / PyBullet are not installed.
# ---------------------------------------------------------------------------
try:
    import moderngl
    import moderngl_window as mglw
    from moderngl_window.geometry import quad_fs
    _MODERNGL_OK = True
except ImportError:
    _MODERNGL_OK = False

try:
    import pybullet as pb
    import pybullet_data
    _PYBULLET_OK = True
except ImportError:
    _PYBULLET_OK = False

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Vector3 = Tuple[float, float, float]

ObjectHandle = namedtuple('ObjectHandle', ['uid', 'body_id', 'object_type'])

# ---------------------------------------------------------------------------
# GLSL shaders
# ---------------------------------------------------------------------------

# ---- Scene vertex shader ---------------------------------------------------
_SCENE_VERT = """
#version 330

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_normal;
out vec3 v_world_pos;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos    = world_pos.xyz;
    v_normal       = mat3(transpose(inverse(u_model))) * in_normal;
    gl_Position    = u_proj * u_view * world_pos;
}
"""

# ---- Scene fragment shader -------------------------------------------------
_SCENE_FRAG = """
#version 330

uniform vec3  u_light_dir;      // normalised, world space
uniform vec3  u_light_color;
uniform float u_ambient;

in vec3 v_normal;
in vec3 v_world_pos;

out vec4 f_color;

void main() {
    vec3 n        = normalize(v_normal);
    float diffuse = max(dot(n, u_light_dir), 0.0);
    vec3  color   = u_light_color * (u_ambient + diffuse * (1.0 - u_ambient));
    f_color = vec4(color, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Geometry helpers (pure-numpy mesh builders)
# ---------------------------------------------------------------------------

def _sphere_mesh(radius: float = 0.5, stacks: int = 16, slices: int = 16):
    """Return (vertices_f32, normals_f32, indices_u32) for a UV sphere."""
    verts, norms, idxs = [], [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            theta = 2 * math.pi * j / slices
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            verts.extend([x * radius, y * radius, z * radius])
            norms.extend([x, y, z])
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            idxs.extend([a, a + slices + 1, a + 1,
                          a + 1, a + slices + 1, a + slices + 2])
    return (np.array(verts, dtype='f4'),
            np.array(norms, dtype='f4'),
            np.array(idxs, dtype='u4'))


def _box_mesh(half: float = 0.5):
    """Return (vertices_f32, normals_f32, indices_u32) for an axis-aligned box."""
    h = half
    # 6 faces, 4 verts each
    faces = [
        # +Y top
        ([ h, h,-h], [ h, h, h], [-h, h, h], [-h, h,-h], [0, 1, 0]),
        # -Y bottom
        ([-h,-h,-h], [-h,-h, h], [ h,-h, h], [ h,-h,-h], [0,-1, 0]),
        # +X right
        ([ h,-h,-h], [ h,-h, h], [ h, h, h], [ h, h,-h], [1, 0, 0]),
        # -X left
        ([-h,-h, h], [-h,-h,-h], [-h, h,-h], [-h, h, h], [-1, 0, 0]),
        # +Z front
        ([-h,-h, h], [ h,-h, h], [ h, h, h], [-h, h, h], [0, 0, 1]),
        # -Z back
        ([ h,-h,-h], [-h,-h,-h], [-h, h,-h], [ h, h,-h], [0, 0,-1]),
    ]
    verts, norms, idxs, base = [], [], [], 0
    for f in faces:
        *corners, normal = f
        for c in corners:
            verts.extend(c)
            norms.extend(normal)
        idxs.extend([base, base+1, base+2, base, base+2, base+3])
        base += 4
    return (np.array(verts, dtype='f4'),
            np.array(norms, dtype='f4'),
            np.array(idxs, dtype='u4'))


def _ground_mesh(size: float = 50.0):
    """Flat quad in XZ plane."""
    h = size
    verts = np.array([
        -h, 0,  h,
         h, 0,  h,
         h, 0, -h,
        -h, 0, -h,
    ], dtype='f4')
    norms = np.tile([0, 1, 0], 4).astype('f4')
    idxs  = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
    return verts, norms, idxs


# ---------------------------------------------------------------------------
# Camera math (pure numpy, no GLM dependency)
# ---------------------------------------------------------------------------

def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype='f4')
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = -1.0
    m[3, 2] = (2 * far * near) / (near - far)
    return m


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye;  f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype='f4')
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[3, 0] = -np.dot(r, eye)
    m[3, 1] = -np.dot(u, eye)
    m[3, 2] =  np.dot(f, eye)
    return m.T


def _translation(tx, ty, tz) -> np.ndarray:
    m = np.eye(4, dtype='f4')
    m[3, 0] = tx; m[3, 1] = ty; m[3, 2] = tz
    return m.T


# ---------------------------------------------------------------------------
# Stub renderer — used when ModernGL is not installed
# ---------------------------------------------------------------------------

class _StubRenderer:
    """Returns black frames so CAINE can still tick without a display."""

    def __init__(self, cam_size):
        self._size = cam_size

    def render_to_array(self, objects, light_dir, light_color, ambient,
                        cam_eye, cam_target):
        return np.zeros((*self._size, 3), dtype=np.uint8)

    def present(self):
        pass

    def destroy(self):
        pass


# ---------------------------------------------------------------------------
# ModernGL renderer
# ---------------------------------------------------------------------------

class _ModernGLRenderer:
    """Manages the OpenGL context, framebuffers, and mesh VAOs."""

    def __init__(self, caine_cam_size=(64, 64), observer_win_size=(800, 600)):
        if not _MODERNGL_OK:
            raise RuntimeError("ModernGL not installed — use _StubRenderer instead.")

        # Stand-alone context (off-screen capable, headless-friendly)
        self.ctx = moderngl.create_standalone_context()

        self.caine_size    = caine_cam_size    # (W, H)
        self.observer_size = observer_win_size  # (W, H)

        # CAINE framebuffer: 64x64 RGBA + depth
        self.caine_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture(caine_cam_size, 4)],
            depth_attachment=self.ctx.depth_renderbuffer(caine_cam_size),
        )

        # Observer framebuffer (larger, optional display)
        self.observer_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture(observer_win_size, 4)],
            depth_attachment=self.ctx.depth_renderbuffer(observer_win_size),
        )

        # Compile shader program
        self.prog = self.ctx.program(
            vertex_shader=_SCENE_VERT,
            fragment_shader=_SCENE_FRAG,
        )

        # Build static meshes
        self._sphere_vao  = self._make_vao(*_sphere_mesh())
        self._box_vao     = self._make_vao(*_box_mesh())
        self._ground_vao  = self._make_vao(*_ground_mesh())

        # Ground model matrix (identity — sits at y=0)
        self._ground_model = np.eye(4, dtype='f4')

    # ------------------------------------------------------------------

    def _make_vao(self, verts, norms, idxs):
        vbo_pos  = self.ctx.buffer(verts.tobytes())
        vbo_norm = self.ctx.buffer(norms.tobytes())
        ibo      = self.ctx.buffer(idxs.tobytes())
        vao = self.ctx.vertex_array(
            self.prog,
            [(vbo_pos,  '3f', 'in_position'),
             (vbo_norm, '3f', 'in_normal')],
            ibo,
        )
        vao._index_count = len(idxs)
        return vao

    # ------------------------------------------------------------------

    def _set_uniforms(self, proj, view, model, light_dir, light_color, ambient):
        self.prog['u_proj'].write(proj.astype('f4').tobytes())
        self.prog['u_view'].write(view.astype('f4').tobytes())
        self.prog['u_model'].write(model.astype('f4').tobytes())
        self.prog['u_light_dir'].value = tuple(light_dir)
        self.prog['u_light_color'].value = tuple(light_color)
        self.prog['u_ambient'].value = float(ambient)

    def _draw_scene(self, fbo, objects, light_dir, light_color, ambient,
                    cam_eye, cam_target, fov=60.0):
        w, h = fbo.size
        fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)  # black void
        self.ctx.enable(moderngl.DEPTH_TEST)

        proj = _perspective(fov, w / h, 0.1, 500.0)
        view = _look_at(np.array(cam_eye, dtype='f4'),
                        np.array(cam_target, dtype='f4'),
                        np.array([0, 1, 0], dtype='f4'))

        # Draw ground (dark grey tint to distinguish from void)
        ground_color = light_color * 0.3
        self._set_uniforms(proj, view, self._ground_model,
                           light_dir, ground_color, ambient)
        self._ground_vao.render()

        # Draw objects
        for handle, (pos, otype, color) in objects.items():
            model = _translation(*pos)
            self._set_uniforms(proj, view, model, light_dir, color, ambient)
            vao = self._sphere_vao if otype == 'sphere' else self._box_vao
            vao.render()

    # ------------------------------------------------------------------

    def render_to_array(self, objects, light_dir, light_color, ambient,
                        cam_eye, cam_target):
        """Render CAINE's 64x64 view and return it as an RGB uint8 array."""
        self._draw_scene(self.caine_fbo, objects, light_dir, light_color,
                         ambient, cam_eye, cam_target)
        raw = self.caine_fbo.color_attachments[0].read()
        img = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.caine_size[1], self.caine_size[0], 4)
        return img[:, :, :3]  # drop alpha

    def render_observer(self, objects, light_dir, light_color, ambient,
                        cam_eye, cam_target):
        """Render the larger observer view."""
        self._draw_scene(self.observer_fbo, objects, light_dir, light_color,
                         ambient, cam_eye, cam_target)
        raw = self.observer_fbo.color_attachments[0].read()
        img = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.observer_size[1], self.observer_size[0], 4)
        return img[:, :, :3]

    def present(self):
        """No-op for off-screen context; subclass to blit to a window."""
        pass

    def destroy(self):
        self.ctx.release()


# ---------------------------------------------------------------------------
# Physics backend
# ---------------------------------------------------------------------------

class _PhysicsWorld:
    """Thin wrapper around PyBullet."""

    def __init__(self):
        if not _PYBULLET_OK:
            raise RuntimeError("PyBullet not installed.")
        self._client = pb.connect(pb.DIRECT)  # headless
        pb.setGravity(0, -9.81, 0, physicsClientId=self._client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                    physicsClientId=self._client)
        # Ground plane
        self._plane_id = pb.loadURDF(
            'plane.urdf', physicsClientId=self._client)

    def spawn(self, shape: str, position: Vector3) -> int:
        """Create a rigid body. Returns the PyBullet body ID."""
        col_id = pb.createCollisionShape(
            pb.GEOM_SPHERE if shape == 'sphere' else pb.GEOM_BOX,
            radius=0.5 if shape == 'sphere' else 1.0,
            halfExtents=[0.5, 0.5, 0.5],
            physicsClientId=self._client,
        )
        vis_id = pb.createVisualShape(
            pb.GEOM_SPHERE if shape == 'sphere' else pb.GEOM_BOX,
            radius=0.5 if shape == 'sphere' else 1.0,
            halfExtents=[0.5, 0.5, 0.5],
            physicsClientId=self._client,
        )
        body_id = pb.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=list(position),
            physicsClientId=self._client,
        )
        return body_id

    def remove(self, body_id: int):
        pb.removeBody(body_id, physicsClientId=self._client)

    def apply_force(self, body_id: int, force: Vector3):
        pb.applyExternalForce(
            body_id, -1, list(force), [0, 0, 0],
            pb.WORLD_FRAME, physicsClientId=self._client)

    def get_position(self, body_id: int) -> Vector3:
        pos, _ = pb.getBasePositionAndOrientation(
            body_id, physicsClientId=self._client)
        return tuple(pos)

    def step(self, dt: float = 1.0 / 60.0):
        pb.stepSimulation(physicsClientId=self._client)

    def disconnect(self):
        pb.disconnect(self._client)


class _StubPhysics:
    """Fake physics when PyBullet is not installed."""
    _next_id = 0
    _bodies: Dict[int, list] = {}

    def spawn(self, shape, position):
        bid = _StubPhysics._next_id
        _StubPhysics._next_id += 1
        _StubPhysics._bodies[bid] = list(position)
        return bid

    def remove(self, body_id):
        _StubPhysics._bodies.pop(body_id, None)

    def apply_force(self, body_id, force):
        pass

    def get_position(self, body_id):
        return tuple(_StubPhysics._bodies.get(body_id, [0, 0, 0]))

    def step(self, dt=1/60):
        pass

    def disconnect(self):
        pass


# ---------------------------------------------------------------------------
# WorldAPI colours palette (deterministic, visually distinct)
# ---------------------------------------------------------------------------
_OBJECT_COLORS = [
    np.array([1.0, 0.3, 0.3], dtype='f4'),  # red
    np.array([0.3, 0.8, 0.3], dtype='f4'),  # green
    np.array([0.3, 0.5, 1.0], dtype='f4'),  # blue
    np.array([1.0, 0.9, 0.2], dtype='f4'),  # yellow
    np.array([0.9, 0.4, 1.0], dtype='f4'),  # purple
    np.array([0.2, 0.9, 0.9], dtype='f4'),  # cyan
]


# ---------------------------------------------------------------------------
# CaineEnvironment — the main public class / WorldAPI implementation
# ---------------------------------------------------------------------------

class CaineEnvironment:
    """
    CAINE's training environment.

    Implements the WorldAPI from the README:
        spawn_object, move_object, remove_object, get_object_position,
        get_camera_feed, play_sound, set_environment_state, kick_player

    Additional helpers:
        step()             — advance physics + render one frame
        start() / stop()   — lifecycle management
        get_observer_feed() — larger observer camera array
    """

    # Default camera positions
    _CAINE_EYE    = (0.0, 1.6, 0.0)   # ~eye height, facing +Z
    _CAINE_TARGET = (0.0, 1.6, 10.0)
    _OBS_EYE      = (8.0, 6.0, -8.0)
    _OBS_TARGET   = (0.0, 0.5,  0.0)

    def __init__(self,
                 caine_cam_size: Tuple[int, int] = (64, 64),
                 observer_win_size: Tuple[int, int] = (800, 600),
                 target_fps: float = 60.0):

        self._cam_size    = caine_cam_size
        self._obs_size    = observer_win_size
        self._target_fps  = target_fps
        self._dt          = 1.0 / target_fps

        # World state
        self._objects: Dict[str, Tuple[Vector3, str, np.ndarray]] = {}
        # uid -> (position, object_type, color)

        self._handles: Dict[str, ObjectHandle] = {}   # uid -> ObjectHandle
        self._color_idx = 0
        self._running   = False
        self._lock      = threading.Lock()

        # Light state
        self._light_dir   = np.array([0.6, 1.0, 0.4], dtype='f4')
        self._light_dir  /= np.linalg.norm(self._light_dir)
        self._light_color = np.array([1.0, 0.98, 0.9], dtype='f4')
        self._ambient     = 0.15

        # Latest rendered frame (CAINE's POV)
        self._latest_frame: Optional[np.ndarray] = None

        # Audio queue: list of (array, position) pairs waiting to be "heard"
        self._audio_queue = []

        # Initialise renderer
        if _MODERNGL_OK:
            try:
                self._renderer = _ModernGLRenderer(caine_cam_size, observer_win_size)
            except Exception as e:
                print(f"[environment] ModernGL init failed ({e}), using stub renderer.")
                self._renderer = _StubRenderer(caine_cam_size)
        else:
            print("[environment] ModernGL not found — using stub renderer.")
            self._renderer = _StubRenderer(caine_cam_size)

        # Initialise physics
        if _PYBULLET_OK:
            try:
                self._physics = _PhysicsWorld()
            except Exception as e:
                print(f"[environment] PyBullet init failed ({e}), using stub physics.")
                self._physics = _StubPhysics()
        else:
            print("[environment] PyBullet not found — using stub physics.")
            self._physics = _StubPhysics()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Mark the environment as running.  Call step() each frame tick."""
        self._running = True
        print("[environment] CAINE training environment started.")
        print(f"  Camera feed : {self._cam_size[0]}x{self._cam_size[1]} RGB")
        print(f"  Target FPS  : {self._target_fps}")
        print(f"  Renderer    : {type(self._renderer).__name__}")
        print(f"  Physics     : {type(self._physics).__name__}")

    def stop(self):
        """Shut down the environment and release GPU/physics resources."""
        self._running = False
        self._renderer.destroy()
        self._physics.disconnect()
        print("[environment] CAINE training environment stopped.")

    # ------------------------------------------------------------------
    # WorldAPI — core interface
    # ------------------------------------------------------------------

    def spawn_object(self, id: str, position: Vector3,
                     rotation: Vector3 = (0.0, 0.0, 0.0),
                     object_type: str = 'sphere') -> ObjectHandle:
        """
        Spawn a physics object at the given world-space position.

        Parameters
        ----------
        id          : unique string identifier for this object
        position    : (x, y, z) world coordinates
        rotation    : (rx, ry, rz) Euler angles in degrees (currently ignored)
        object_type : 'sphere' or 'cube'

        Returns
        -------
        ObjectHandle namedtuple — pass to move_object / remove_object
        """
        if object_type not in ('sphere', 'cube'):
            raise ValueError(f"object_type must be 'sphere' or 'cube', got {object_type!r}")

        color = _OBJECT_COLORS[self._color_idx % len(_OBJECT_COLORS)].copy()
        self._color_idx += 1

        body_id = self._physics.spawn(object_type, position)
        handle  = ObjectHandle(uid=id, body_id=body_id, object_type=object_type)

        with self._lock:
            self._handles[id] = handle
            self._objects[id] = (tuple(position), object_type, color)

        return handle

    def move_object(self, handle: ObjectHandle, force: Vector3) -> None:
        """Apply an impulse force to an object (Newtons, world frame)."""
        self._physics.apply_force(handle.body_id, force)

    def remove_object(self, handle: ObjectHandle) -> None:
        """Remove an object from the world."""
        self._physics.remove(handle.body_id)
        with self._lock:
            self._handles.pop(handle.uid, None)
            self._objects.pop(handle.uid, None)

    def get_object_position(self, handle: ObjectHandle) -> Vector3:
        """Return the current (x, y, z) position of an object."""
        pos = self._physics.get_position(handle.body_id)
        # Keep render state in sync
        with self._lock:
            if handle.uid in self._objects:
                _, otype, color = self._objects[handle.uid]
                self._objects[handle.uid] = (pos, otype, color)
        return pos

    def get_camera_feed(self) -> np.ndarray:
        """
        Return CAINE's current camera feed as a (H, W, 3) uint8 numpy array.
        Call step() first to ensure the frame is up to date.
        """
        if self._latest_frame is None:
            # Return black frame before first render
            return np.zeros((self._cam_size[1], self._cam_size[0], 3), dtype=np.uint8)
        return self._latest_frame.copy()

    def play_sound(self, audio: np.ndarray, position: Vector3) -> None:
        """
        Queue a spatialized audio event.

        audio    : 1-D float32 numpy array (normalised −1…1), any sample rate
        position : (x, y, z) source position in world space

        The sensory layer (Module 5) polls this queue to inject audio into A1.
        """
        self._audio_queue.append({'audio': audio.copy(), 'position': position})

    def set_environment_state(self, params: dict) -> None:
        """
        Update global environment parameters.

        Recognised keys
        ---------------
        'light_direction' : (x, y, z) float — normalised on write
        'light_color'     : (r, g, b) float in [0, 1]
        'ambient'         : float in [0, 1]
        'caine_eye'       : (x, y, z) — override CAINE's camera position
        'caine_target'    : (x, y, z) — override CAINE's gaze target
        'observer_eye'    : (x, y, z) — override observer position
        'observer_target' : (x, y, z) — override observer gaze target
        """
        if 'light_direction' in params:
            d = np.array(params['light_direction'], dtype='f4')
            self._light_dir = d / np.linalg.norm(d)
        if 'light_color' in params:
            self._light_color = np.array(params['light_color'], dtype='f4')
        if 'ambient' in params:
            self._ambient = float(params['ambient'])
        if 'caine_eye' in params:
            self._CAINE_EYE = tuple(params['caine_eye'])
        if 'caine_target' in params:
            self._CAINE_TARGET = tuple(params['caine_target'])
        if 'observer_eye' in params:
            self._OBS_EYE = tuple(params['observer_eye'])
        if 'observer_target' in params:
            self._OBS_TARGET = tuple(params['observer_target'])

    def kick_player(self, player_id: str) -> None:
        """
        Remove a player/agent from the environment.

        In single-agent CAINE this is a no-op, but the method exists to keep
        the API surface compatible with the multi-agent / Unity mirror spec.
        """
        print(f"[environment] kick_player({player_id!r}) — no-op in single-agent mode.")

    # ------------------------------------------------------------------
    # Frame tick
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        Advance the simulation by one frame (1/fps seconds).

        1. Sync physics positions into the render object table.
        2. Step PyBullet physics.
        3. Render CAINE's 64x64 camera.
        """
        if not self._running:
            return

        # Sync positions from physics engine
        with self._lock:
            for uid, handle in self._handles.items():
                pos = self._physics.get_position(handle.body_id)
                _, otype, color = self._objects[uid]
                self._objects[uid] = (pos, otype, color)

            render_snapshot = dict(self._objects)

        # Step physics
        self._physics.step(self._dt)

        # Render CAINE's feed
        self._latest_frame = self._renderer.render_to_array(
            render_snapshot,
            self._light_dir,
            self._light_color,
            self._ambient,
            self._CAINE_EYE,
            self._CAINE_TARGET,
        )

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    def get_observer_feed(self) -> np.ndarray:
        """
        Render and return the observer (third-person) camera as a numpy array.
        Larger than CAINE's feed — intended for human monitoring.
        """
        if not hasattr(self._renderer, 'render_observer'):
            return np.zeros((self._obs_size[1], self._obs_size[0], 3), dtype=np.uint8)

        with self._lock:
            snapshot = dict(self._objects)

        return self._renderer.render_observer(
            snapshot,
            self._light_dir,
            self._light_color,
            self._ambient,
            self._OBS_EYE,
            self._OBS_TARGET,
        )

    def pop_audio_queue(self):
        """Return and clear pending audio events (for Module 5 to consume)."""
        q = list(self._audio_queue)
        self._audio_queue.clear()
        return q

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

def run_environment_demo(n_frames: int = 120):
    """
    Headless smoke-test: spawn a sphere and a cube, step 120 frames, verify
    that get_camera_feed() returns a (64, 64, 3) uint8 array each frame.
    """
    import os
    _OUTPUT_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    env = CaineEnvironment()
    env.start()

    sphere = env.spawn_object('ball_1', (0.0, 2.0, 5.0), object_type='sphere')
    cube   = env.spawn_object('box_1',  (2.0, 1.0, 5.0), object_type='cube')

    # Inject environment tone via play_sound
    t = np.linspace(0, 0.5, 22050 // 2, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440.0 * t)
    env.play_sound(tone, position=(0.0, 1.0, 5.0))

    frames = []
    for i in range(n_frames):
        env.step()
        feed = env.get_camera_feed()
        frames.append(feed)

        # Apply gentle force every 30 frames to see the sphere move
        if i % 30 == 0:
            env.move_object(sphere, (0.5, 3.0, 0.0))

    env.stop()

    # Save a contact sheet of 9 evenly-spaced frames as a PNG
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        indices = np.linspace(0, n_frames - 1, 9, dtype=int)
        for ax, idx in zip(axes.flat, indices):
            ax.imshow(frames[idx])
            ax.set_title(f"frame {idx}")
            ax.axis('off')
        fig.suptitle("CAINE Environment — Camera Feed Demo", fontsize=14)
        plt.tight_layout()
        out = os.path.join(_OUTPUT_DIR, 'caine_environment_demo.png')
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"[environment] Contact sheet saved to {out}")
    except ImportError:
        print("[environment] matplotlib not available — skipping contact sheet.")

    print(f"[environment] Demo complete. Rendered {n_frames} frames.")
    print(f"  Feed shape : {frames[0].shape}, dtype={frames[0].dtype}")
    return frames


if __name__ == '__main__':
    run_environment_demo()
