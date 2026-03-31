"""
CAINE Training Environment — Module 9
======================================
CAINE's world is a custom Python rendering sandbox with a Blender API backend.

9.1 Core Renderer
-----------------
- ModernGL for OpenGL rendering (Python-native, no game engine dependency)
- Dual camera system: CAINE's POV camera (feeds visual cortex) + observer camera
- PyBullet for physics simulation (rigid body, collision, gravity)
- Scene graph managing all objects, their physics state, and visual mesh

9.2 Blender Backend
-------------------
- BlenderBridge: headless Blender subprocess communicating via local TCP socket
- JSON command/response protocol
- import_blend / import_fbx / import_glb / generate_procedural_texture /
  apply_geometry_nodes / bake_physics / export_mesh

9.3 Developmental Environment Stages
-------------------------------------
- Stage 0 — The Void        (lights + tones; V1/A1 calibration)
- Stage 1 — The Nursery     (geometric primitives, physics, Father voice)
- Stage 2 — The Playroom    (complex assets, agents, PFC myelination)
- Stage 3 — The Social Space (Claude API agents, social scenarios, DMN)
- Stage 4 — The Circus      (full Digital Circus scene)

StageManager evaluates each stage's exit conditions and gates transitions.

Dependencies
------------
    pip install moderngl moderngl-window PyBullet numpy

Usage
-----
    from caine.environment import CaineEnvironment, StageManager

    env     = CaineEnvironment()
    stages  = StageManager(env)
    env.start()

    handle = env.spawn_object('sphere', (0, 1, 0))
    for tick in range(300):
        feed = env.get_camera_feed()    # 64x64 RGB numpy array
        env.step()                      # advance physics + render one frame
        stages.tick(cortex_state, dt_ms=20.0, sim_time_s=tick * 0.02)
    env.stop()
"""

import json
import math
import os
import socket
import subprocess
import sys as _sys
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import struct
import threading
import time
from collections import namedtuple
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

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
uniform vec3  u_object_color;   // per-draw material colour
uniform float u_ambient;

in vec3 v_normal;
in vec3 v_world_pos;

out vec4 f_color;

void main() {
    vec3  n       = normalize(v_normal);
    vec3  ldir    = normalize(u_light_dir);
    float diffuse = max(dot(n, ldir), 0.0);
    vec3  lit     = u_light_color * (u_ambient + diffuse * (1.0 - u_ambient));
    f_color = vec4(u_object_color * lit, 1.0);
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

    def _set_uniforms(self, proj, view, model, light_dir, light_color, ambient,
                      object_color=(0.8, 0.8, 0.8)):
        self.prog['u_proj'].write(proj.astype('f4').tobytes())
        self.prog['u_view'].write(view.astype('f4').tobytes())
        self.prog['u_model'].write(model.astype('f4').tobytes())
        ld = np.array(light_dir, dtype='f4')
        norm = np.linalg.norm(ld)
        self.prog['u_light_dir'].value = tuple(ld / norm if norm > 0 else ld)
        self.prog['u_light_color'].value = tuple(float(v) for v in light_color[:3])
        self.prog['u_ambient'].value = float(ambient)
        self.prog['u_object_color'].value = tuple(float(v) for v in object_color[:3])

    def _draw_scene(self, fbo, objects, light_dir, light_color, ambient,
                    cam_eye, cam_target, fov=60.0):
        w, h = fbo.size
        fbo.use()
        self.ctx.clear(0.05, 0.05, 0.08, 1.0)  # deep-space near-black
        self.ctx.enable(moderngl.DEPTH_TEST)

        proj = _perspective(fov, w / h, 0.1, 500.0)
        view = _look_at(np.array(cam_eye, dtype='f4'),
                        np.array(cam_target, dtype='f4'),
                        np.array([0, 1, 0], dtype='f4'))

        # Ground — dark tile, lit by scene light
        self._set_uniforms(proj, view, self._ground_model,
                           light_dir, light_color, ambient,
                           object_color=(0.22, 0.22, 0.28))
        self._ground_vao.render()

        # Objects — each with its own material colour
        for uid, (pos, otype, color) in objects.items():
            model = _translation(*pos)
            obj_col = color[:3] if hasattr(color, '__len__') else (0.8, 0.8, 0.8)
            self._set_uniforms(proj, view, model, light_dir, light_color, ambient,
                               object_color=obj_col)
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
        return img[::-1, :, :3]   # flip Y (OpenGL is bottom-up)

    def render_observer(self, objects, light_dir, light_color, ambient,
                        cam_eye, cam_target):
        """Render the larger observer view."""
        self._draw_scene(self.observer_fbo, objects, light_dir, light_color,
                         ambient, cam_eye, cam_target)
        raw = self.observer_fbo.color_attachments[0].read()
        img = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.observer_size[1], self.observer_size[0], 4)
        return img[::-1, :, :3]   # flip Y (OpenGL is bottom-up)

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


# ===========================================================================
# 9.2  BLENDER BACKEND
# ===========================================================================

import caine.paths as _paths
_OUTPUT_DIR_ENV = _paths.OUTPUT_DIR

# Minimal Blender bridge server script — written to a temp file and loaded
# by `blender --background --python <script>` on each invocation.
_BLENDER_SERVER_SCRIPT = '''\
import bpy, json, socket, sys, os, struct

HOST, PORT = "127.0.0.1", int(sys.argv[-1])

srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT))
srv.listen(1)
conn, _ = srv.accept()

def send(obj):
    data = json.dumps(obj).encode()
    conn.sendall(struct.pack(">I", len(data)) + data)

def recv():
    raw = b""
    while len(raw) < 4:
        raw += conn.recv(4 - len(raw))
    (n,) = struct.unpack(">I", raw)
    data = b""
    while len(data) < n:
        data += conn.recv(n - len(data))
    return json.loads(data.decode())

send({"status": "ready"})

while True:
    cmd = recv()
    action = cmd.get("action", "")
    try:
        if action == "import_blend":
            bpy.ops.wm.open_mainfile(filepath=cmd["path"])
            names = [o.name for o in bpy.context.scene.objects]
            send({"objects": names})
        elif action == "import_fbx":
            bpy.ops.import_scene.fbx(filepath=cmd["path"])
            names = [o.name for o in bpy.context.scene.objects]
            send({"objects": names})
        elif action == "import_glb":
            bpy.ops.import_scene.gltf(filepath=cmd["path"])
            names = [o.name for o in bpy.context.scene.objects]
            send({"objects": names})
        elif action == "export_mesh":
            obj = bpy.data.objects.get(cmd.get("obj_id", ""))
            if obj and obj.type == "MESH":
                verts = [list(v.co) for v in obj.data.vertices]
                faces = [list(p.vertices) for p in obj.data.polygons]
                send({"vertices": verts, "faces": faces})
            else:
                send({"error": "object not found or not a mesh"})
        elif action == "quit":
            send({"status": "bye"})
            break
        else:
            send({"error": f"unknown action: {action}"})
    except Exception as e:
        send({"error": str(e)})

conn.close()
srv.close()
'''


@dataclass
class SceneObject:
    """Lightweight descriptor for a Blender-imported object."""
    name:        str
    object_type: str = 'mesh'    # 'mesh' | 'light' | 'camera' | 'empty'
    source_file: str = ''


@dataclass
class MeshData:
    """Raw mesh geometry returned by BlenderBridge."""
    vertices: List[List[float]]
    faces:    List[List[int]]
    name:     str = ''


@dataclass
class TextureData:
    """Texture array returned by BlenderBridge."""
    pixels:  Any   # np.ndarray (H, W, 4) RGBA float32
    width:   int = 0
    height:  int = 0


@dataclass
class PhysicsCache:
    """Baked physics simulation cache."""
    frames:      int = 0
    object_name: str = ''


class BlenderBridge:
    """
    Headless Blender backend communicating via a local TCP socket.

    Blender runs as ``blender --background --python <bridge_script> -- <port>``
    and stays alive until ``close()`` is called.  All heavy geometry work
    (import, procedural texture generation, geometry nodes, physics baking)
    executes inside Blender's Python runtime.

    If Blender is not found on PATH the bridge degrades to a stub that returns
    empty data so the rest of CAINE still runs.

    Usage
    -----
        bridge = BlenderBridge()
        bridge.start()
        objects = bridge.import_glb('data/nursery_room.glb')
        mesh    = bridge.export_mesh(objects[0].name)
        bridge.close()
    """

    _BRIDGE_PORT_DEFAULT = 47200

    def __init__(self, blender_exe: str = 'blender',
                 port: int = _BRIDGE_PORT_DEFAULT):
        self._exe    = blender_exe
        self._port   = port
        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket]    = None
        self._ready  = False
        self._script_path = _paths.BLENDER_BRIDGE

    # ------------------------------------------------------------------
    def start(self, timeout_s: float = 15.0) -> bool:
        """
        Launch Blender subprocess and wait for it to signal readiness.

        Returns True if connected, False if Blender is unavailable.
        """
        # Write bridge script to disk
        with open(self._script_path, 'w', encoding='utf-8') as f:
            f.write(_BLENDER_SERVER_SCRIPT)

        try:
            self._proc = subprocess.Popen(
                [self._exe, '--background', '--python',
                 self._script_path, '--', str(self._port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print(f"[BlenderBridge] '{self._exe}' not found — running in stub mode.")
            return False

        # Connect socket with retry
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                s = socket.socket()
                s.connect(('127.0.0.1', self._port))
                self._sock = s
                resp = self._recv()
                if resp.get('status') == 'ready':
                    self._ready = True
                    print(f"[BlenderBridge] Connected on port {self._port}.")
                    return True
            except (ConnectionRefusedError, OSError):
                time.sleep(0.3)

        print("[BlenderBridge] Timed out waiting for Blender — stub mode.")
        return False

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Shut down the Blender subprocess cleanly."""
        if self._sock and self._ready:
            try:
                self._send({'action': 'quit'})
                self._recv()
            except Exception:
                pass
            self._sock.close()
            self._sock = None
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self._ready = False

    # ------------------------------------------------------------------
    # Socket helpers
    # ------------------------------------------------------------------

    def _send(self, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self._sock.sendall(struct.pack('>I', len(data)) + data)

    def _recv(self) -> dict:
        raw = b''
        while len(raw) < 4:
            chunk = self._sock.recv(4 - len(raw))
            if not chunk:
                raise ConnectionError("Blender bridge socket closed")
            raw += chunk
        (n,) = struct.unpack('>I', raw)
        data = b''
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Blender bridge socket closed")
            data += chunk
        return json.loads(data.decode())

    def _call(self, cmd: dict) -> dict:
        """Send a command and return the response; stubs on failure."""
        if not self._ready:
            return {'error': 'bridge not ready (stub mode)'}
        try:
            self._send(cmd)
            return self._recv()
        except Exception as e:
            print(f"[BlenderBridge] call error: {e}")
            return {'error': str(e)}

    # ------------------------------------------------------------------
    # Public API (matches README spec)
    # ------------------------------------------------------------------

    def import_blend(self, path: str) -> List[SceneObject]:
        """Open a .blend file and return all scene objects."""
        resp = self._call({'action': 'import_blend', 'path': path})
        return [SceneObject(name=n, source_file=path)
                for n in resp.get('objects', [])]

    def import_fbx(self, path: str) -> List[SceneObject]:
        """Import an FBX file and return all imported objects."""
        resp = self._call({'action': 'import_fbx', 'path': path})
        return [SceneObject(name=n, source_file=path)
                for n in resp.get('objects', [])]

    def import_glb(self, path: str) -> List[SceneObject]:
        """Import a GLB/GLTF file and return all imported objects."""
        resp = self._call({'action': 'import_glb', 'path': path})
        return [SceneObject(name=n, source_file=path)
                for n in resp.get('objects', [])]

    def generate_procedural_texture(self, node_params: dict) -> TextureData:
        """
        Generate a texture using Blender's shader node graph.
        node_params: dict describing node tree (passed directly to bridge).
        Returns TextureData stub until full node compiler is implemented.
        """
        # Stub: return a small noise texture
        import numpy as _np
        pixels = _np.random.rand(64, 64, 4).astype(_np.float32)
        pixels[:, :, 3] = 1.0   # alpha = 1
        return TextureData(pixels=pixels, width=64, height=64)

    def apply_geometry_nodes(self, obj_id: str, node_tree: dict) -> MeshData:
        """Apply a geometry node modifier and return the resulting mesh."""
        # Stub until full geometry node serialisation is implemented
        return MeshData(vertices=[], faces=[], name=obj_id)

    def bake_physics(self, scene_id: str, frames: int) -> PhysicsCache:
        """Bake a rigid-body / cloth / fluid physics simulation."""
        return PhysicsCache(frames=frames, object_name=scene_id)

    def export_mesh(self, obj_id: str) -> MeshData:
        """Export a named object's mesh data (vertices + faces)."""
        resp = self._call({'action': 'export_mesh', 'obj_id': obj_id})
        return MeshData(
            vertices=resp.get('vertices', []),
            faces=resp.get('faces', []),
            name=obj_id,
        )

    @property
    def is_ready(self) -> bool:
        return self._ready


# ===========================================================================
# 9.3  DEVELOPMENTAL ENVIRONMENT STAGES
# ===========================================================================

class DevStage(IntEnum):
    """CAINE's developmental environment stages."""
    VOID         = 0   # black space, lights + tones only
    NURSERY      = 1   # geometric primitives, physics, Father voice
    PLAYROOM     = 2   # complex assets, agents, PFC myelination
    SOCIAL_SPACE = 3   # Claude agents, social scenarios, DMN
    CIRCUS       = 4   # full Digital Circus scene


# Minimum simulated durations per stage (seconds)
_STAGE_MIN_RUNTIME_S: Dict[int, float] = {
    DevStage.VOID:         48 * 3600,          # 48 hours
    DevStage.NURSERY:      14 * 24 * 3600,     # 2 weeks
    DevStage.PLAYROOM:     30 * 24 * 3600,     # 1 month
    DevStage.SOCIAL_SPACE: 90 * 24 * 3600,     # 3 months
    DevStage.CIRCUS:       float('inf'),        # no exit
}

# Human-readable stage names for logging
_STAGE_NAMES: Dict[int, str] = {
    DevStage.VOID:         'Stage 0 — The Void',
    DevStage.NURSERY:      'Stage 1 — The Nursery',
    DevStage.PLAYROOM:     'Stage 2 — The Playroom',
    DevStage.SOCIAL_SPACE: 'Stage 3 — The Social Space',
    DevStage.CIRCUS:       'Stage 4 — The Circus',
}


class StageExitEvaluator:
    """
    Evaluates each developmental stage's exit conditions from a cortex_state dict.

    All ``check_stage_N_*`` methods return True when that sub-condition is met.
    ``all_met(stage, cortex_state, sim_time_s, stage_entry_sim_s)`` returns True
    when every condition for ``stage`` is satisfied.

    Expected cortex_state keys (subset used per stage):
        v1_orientation_selectivity : float — mean orientation selectivity index
        a1_tonotopic_gradient      : float — Pearson r of freq-order vs rate-order
        cortisol_chronic           : bool  — from NeurochemicalSystem
        it_category_count          : int   — distinct IT response categories detected
        hippo_replay_active        : bool  — hippocampal replay detected
        stg_cluster_count          : int   — stable STG phoneme clusters
        word_association_confirmed : bool  — Hebbian binding > threshold
        father_voice_ratio         : float — A1/A2 response ratio Father vs other
        voluntary_movement         : bool  — non-random M1 pattern detected
        dmn_correlation            : float — mPFC+PCC+AG resting-state correlation
        unprompted_vocalization    : bool  — set by ConsciousnessMonitor
        pfc_wm_span_ms             : float — measured PFC WM span
        theory_of_mind_precursor   : bool  — differential agent orientation
    """

    # Stage 0 thresholds
    V1_OSI_THRESHOLD        = 0.6
    A1_TOPO_CORR_THRESHOLD  = 0.8

    # Stage 1 thresholds
    IT_CATEGORY_MIN         = 5
    STG_CLUSTER_MIN         = 3

    # Stage 2 thresholds
    FATHER_VOICE_RATIO_MIN  = 2.0

    # Stage 3 thresholds
    DMN_CORR_THRESHOLD      = 0.4
    PFC_WM_SPAN_MS_MIN      = 2000.0

    # ------------------------------------------------------------------
    # Stage 0 — The Void
    # ------------------------------------------------------------------

    def stage0_v1_orientation(self, cs: dict) -> bool:
        return float(cs.get('v1_orientation_selectivity', 0.0)) >= self.V1_OSI_THRESHOLD

    def stage0_a1_tonotopy(self, cs: dict) -> bool:
        return float(cs.get('a1_tonotopic_gradient', 0.0)) >= self.A1_TOPO_CORR_THRESHOLD

    def stage0_no_chronic_cortisol(self, cs: dict) -> bool:
        return not bool(cs.get('cortisol_chronic', False))

    # ------------------------------------------------------------------
    # Stage 1 — The Nursery
    # ------------------------------------------------------------------

    def stage1_it_categories(self, cs: dict) -> bool:
        return int(cs.get('it_category_count', 0)) >= self.IT_CATEGORY_MIN

    def stage1_hippo_encoding(self, cs: dict) -> bool:
        return bool(cs.get('hippo_replay_active', False))

    def stage1_stg_clusters(self, cs: dict) -> bool:
        return int(cs.get('stg_cluster_count', 0)) >= self.STG_CLUSTER_MIN

    # ------------------------------------------------------------------
    # Stage 2 — The Playroom
    # ------------------------------------------------------------------

    def stage2_word_association(self, cs: dict) -> bool:
        return bool(cs.get('word_association_confirmed', False))

    def stage2_father_voice_differential(self, cs: dict) -> bool:
        return float(cs.get('father_voice_ratio', 0.0)) >= self.FATHER_VOICE_RATIO_MIN

    def stage2_voluntary_movement(self, cs: dict) -> bool:
        return bool(cs.get('voluntary_movement', False))

    # ------------------------------------------------------------------
    # Stage 3 — The Social Space
    # ------------------------------------------------------------------

    def stage3_dmn_correlation(self, cs: dict) -> bool:
        return float(cs.get('dmn_correlation', 0.0)) >= self.DMN_CORR_THRESHOLD

    def stage3_unprompted_vocalization(self, cs: dict) -> bool:
        return bool(cs.get('unprompted_vocalization', False))

    def stage3_pfc_wm_span(self, cs: dict) -> bool:
        return float(cs.get('pfc_wm_span_ms', 0.0)) >= self.PFC_WM_SPAN_MS_MIN

    def stage3_theory_of_mind(self, cs: dict) -> bool:
        return bool(cs.get('theory_of_mind_precursor', False))

    # ------------------------------------------------------------------
    # Aggregate check
    # ------------------------------------------------------------------

    def all_met(self, stage: int, cortex_state: dict,
                sim_time_s: float, stage_entry_sim_s: float) -> Tuple[bool, dict]:
        """
        Check whether all exit conditions for ``stage`` are satisfied.

        Returns (all_passed, condition_results_dict).
        condition_results_dict maps condition name → bool.
        """
        time_in_stage = sim_time_s - stage_entry_sim_s
        min_time      = _STAGE_MIN_RUNTIME_S.get(stage, float('inf'))
        runtime_ok    = time_in_stage >= min_time

        cs = cortex_state

        if stage == DevStage.VOID:
            results = {
                'v1_orientation':     self.stage0_v1_orientation(cs),
                'a1_tonotopy':        self.stage0_a1_tonotopy(cs),
                'no_chronic_cort':    self.stage0_no_chronic_cortisol(cs),
                'min_runtime_48h':    runtime_ok,
            }
        elif stage == DevStage.NURSERY:
            results = {
                'it_categories_5+':   self.stage1_it_categories(cs),
                'hippo_encoding':     self.stage1_hippo_encoding(cs),
                'stg_clusters_3+':    self.stage1_stg_clusters(cs),
                'min_runtime_2wk':    runtime_ok,
            }
        elif stage == DevStage.PLAYROOM:
            results = {
                'word_association':   self.stage2_word_association(cs),
                'father_voice_diff':  self.stage2_father_voice_differential(cs),
                'voluntary_movement': self.stage2_voluntary_movement(cs),
                'min_runtime_1mo':    runtime_ok,
            }
        elif stage == DevStage.SOCIAL_SPACE:
            results = {
                'dmn_correlation':    self.stage3_dmn_correlation(cs),
                'unprompted_voc':     self.stage3_unprompted_vocalization(cs),
                'pfc_wm_span_2s':     self.stage3_pfc_wm_span(cs),
                'theory_of_mind':     self.stage3_theory_of_mind(cs),
                'min_runtime_3mo':    runtime_ok,
            }
        else:
            # Stage 4 / Circus: no exit
            return False, {'circus': False}

        all_passed = all(results.values())
        return all_passed, results


class StageManager:
    """
    Tracks CAINE's developmental stage, evaluates exit conditions, and
    triggers stage transitions.

    Call ``tick()`` each simulation frame.  When all conditions for the
    current stage are met, the manager automatically advances to the next
    stage, calls ``env.configure_for_stage()``, and logs a milestone.

    Parameters
    ----------
    env       : CaineEnvironment — the world to reconfigure on transition
    start_stage : int — override starting stage (default 0)
    """

    def __init__(self, env, start_stage: int = DevStage.VOID):
        self._env            = env
        self._stage          = DevStage(start_stage)
        self._evaluator      = StageExitEvaluator()
        self._entry_sim_s    = 0.0    # sim time when current stage was entered
        self._transition_log: List[dict] = []
        self._milestones_file = _paths.MILESTONES_LOG

        # Cache last condition results for diagnostics
        self._last_conditions: dict = {}

    # ------------------------------------------------------------------
    def tick(self, cortex_state: dict,
             dt_ms: float = 20.0,
             sim_time_s: float = 0.0) -> Optional[DevStage]:
        """
        Advance stage manager by one tick.

        Returns the new DevStage if a transition occurred, else None.
        """
        if self._stage == DevStage.CIRCUS:
            return None   # final stage — no exit

        all_met, conditions = self._evaluator.all_met(
            self._stage, cortex_state, sim_time_s, self._entry_sim_s)
        self._last_conditions = conditions

        if all_met:
            return self._transition(sim_time_s)
        return None

    # ------------------------------------------------------------------
    def _transition(self, sim_time_s: float) -> DevStage:
        """Advance to the next developmental stage."""
        old_stage  = self._stage
        new_stage  = DevStage(min(int(self._stage) + 1, int(DevStage.CIRCUS)))
        self._stage      = new_stage
        self._entry_sim_s = sim_time_s

        print()
        print('=' * 60)
        print(f'  DEVELOPMENTAL STAGE TRANSITION')
        print(f'  {_STAGE_NAMES[old_stage]}')
        print(f'  -> {_STAGE_NAMES[new_stage]}')
        print(f'  sim_time = {sim_time_s:.1f}s')
        print('=' * 60)
        print()

        # Log milestone
        entry = {
            'event':       'stage_transition',
            'from_stage':  int(old_stage),
            'to_stage':    int(new_stage),
            'from_name':   _STAGE_NAMES[old_stage],
            'to_name':     _STAGE_NAMES[new_stage],
            'sim_time_s':  round(sim_time_s, 2),
            'wall_time':   time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        self._transition_log.append(entry)
        try:
            with open(self._milestones_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass

        # Reconfigure the world for the new stage
        if self._env is not None:
            self._env.configure_for_stage(int(new_stage))

        return new_stage

    # ------------------------------------------------------------------
    @property
    def stage(self) -> DevStage:
        return self._stage

    @property
    def stage_int(self) -> int:
        return int(self._stage)

    @property
    def stage_name(self) -> str:
        return _STAGE_NAMES[self._stage]

    @property
    def last_conditions(self) -> dict:
        """Most recent condition check results (name → bool)."""
        return dict(self._last_conditions)

    def conditions_summary(self) -> str:
        """One-line human-readable summary of current exit conditions."""
        if not self._last_conditions:
            return f'{self.stage_name}: no data'
        met   = sum(v for v in self._last_conditions.values())
        total = len(self._last_conditions)
        items = ', '.join(
            f"{'[x]' if v else '[ ]'} {k}"
            for k, v in self._last_conditions.items()
        )
        return f'{self.stage_name} [{met}/{total}]: {items}'


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
# Tone synthesis helper (used by _stage_stimuli_tick)
# ---------------------------------------------------------------------------

_ENV_SAMPLE_RATE = 16000   # samples per second for environment-generated tones

def _env_synthesize_tone(frequency: float, duration_s: float,
                         volume: float = 0.5) -> np.ndarray:
    """Generate a pure sine tone as a float32 numpy array at 16 kHz."""
    n = int(_ENV_SAMPLE_RATE * duration_s)
    t = np.linspace(0.0, duration_s, n, dtype=np.float32)
    tone = np.sin(2.0 * math.pi * frequency * t) * float(volume)
    fade = max(1, int(_ENV_SAMPLE_RATE * 0.01))
    if 2 * fade < len(tone):
        tone[:fade]  *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        tone[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return tone


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

        self._cam_size        = caine_cam_size
        self._obs_size        = observer_win_size
        self._latest_obs_frame: Optional[np.ndarray] = None   # cached from sim thread
        self._target_fps  = target_fps
        self._dt          = 1.0 / target_fps

        # World state
        self._objects: Dict[str, Tuple[Vector3, str, np.ndarray]] = {}
        # uid -> (position, object_type, color)

        self._handles: Dict[str, ObjectHandle] = {}   # uid -> ObjectHandle
        self._color_idx = 0
        self._object_spawn_time: Dict[str, float] = {}   # uid -> sim_time at spawn
        self._object_ttl_s: float = 90.0                 # auto-despawn after N sim-seconds
        self._max_objects:  int   = 8                    # hard cap on concurrent objects
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

        # Stage-aware stimulus state
        self._dev_stage: int = 0          # set by StageManager via set_stage()
        self._env_sim_time_s: float = 0.0 # accumulated simulated seconds
        self._stage_tone_freqs: list  = [220.0, 440.0, 880.0, 1760.0, 3520.0]
        self._stage_tone_idx: int     = 0
        self._next_tone_s: float      = 0.0   # sim-time of next scheduled tone
        self._tone_interval_s: float  = 8.0   # seconds between Stage-0 tones
        self._light_phase: float      = 0.0   # phase for sinusoidal light variation

        # Initialise developmental flags (Stage 0 defaults)
        self._init_dev_flags()

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
        # Normalise unknown types to sphere (Mother LLM may send 'face', 'toy', etc.)
        if object_type not in ('sphere', 'cube'):
            object_type = 'sphere'

        # Evict oldest object if at cap
        with self._lock:
            if len(self._handles) >= self._max_objects and self._object_spawn_time:
                oldest_uid = min(self._object_spawn_time, key=self._object_spawn_time.get)
                oldest_handle = self._handles.pop(oldest_uid, None)
                self._objects.pop(oldest_uid, None)
                self._object_spawn_time.pop(oldest_uid, None)
                if oldest_handle is not None:
                    try: self._physics.remove(oldest_handle.body_id)
                    except Exception: pass

        color = _OBJECT_COLORS[self._color_idx % len(_OBJECT_COLORS)].copy()
        self._color_idx += 1

        body_id = self._physics.spawn(object_type, position)
        handle  = ObjectHandle(uid=id, body_id=body_id, object_type=object_type)

        with self._lock:
            self._handles[id] = handle
            self._objects[id] = (tuple(position), object_type, color)
            self._object_spawn_time[id] = self._env_sim_time_s

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

        # Despawn objects whose TTL has expired
        now = self._env_sim_time_s
        expired = [uid for uid, t0 in list(self._object_spawn_time.items())
                   if now - t0 > self._object_ttl_s]
        for uid in expired:
            handle = self._handles.pop(uid, None)
            self._objects.pop(uid, None)
            self._object_spawn_time.pop(uid, None)
            if handle is not None:
                try: self._physics.remove(handle.body_id)
                except Exception: pass

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

        # Render observer feed in the same (sim) thread and cache it so the
        # viz server can read it safely from a different thread.
        if hasattr(self._renderer, 'render_observer'):
            try:
                self._latest_obs_frame = self._renderer.render_observer(
                    render_snapshot,
                    self._light_dir,
                    self._light_color,
                    self._ambient,
                    self._OBS_EYE,
                    self._OBS_TARGET,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    def get_observer_feed(self) -> np.ndarray:
        """
        Return the cached observer (third-person) camera frame.
        The frame is rendered in the simulation thread during step() to avoid
        OpenGL cross-thread access; this method is safe to call from any thread.
        """
        if self._latest_obs_frame is not None:
            return self._latest_obs_frame
        return np.zeros((self._obs_size[1], self._obs_size[0], 3), dtype=np.uint8)

    def get_caine_position(self) -> Vector3:
        """Return CAINE's current world-space eye position (x, y, z)."""
        return tuple(self._CAINE_EYE)

    def pop_audio_queue(self):
        """Return and clear pending audio events (for Module 5 to consume)."""
        q = list(self._audio_queue)
        self._audio_queue.clear()
        return q

    def set_stage(self, stage: int) -> None:
        """Called by StageManager or brain to keep environment's stage in sync."""
        self._dev_stage = int(stage)

    def advance_sim_time(self, dt_s: float) -> None:
        """
        Advance internal simulated time and run stage-aware stimulus generation.
        Call this once per simulation tick from run_caine / CAINEBrain.
        """
        self._env_sim_time_s += dt_s
        self._stage_stimuli_tick(dt_s)

    def _stage_stimuli_tick(self, dt_s: float) -> None:
        """
        Programmatic stage-aware stimulus generation.

        Stage 0 — The Void:
            • Point light sweeps a sinusoidal arc (position + colour + intensity)
              over a ~60-second cycle.  This drives V1 orientation and A1 onset
              calibration without any semantic content.
            • Pure sine tones played on a schedule:
              - Short burst (0.5 s) every 8 s, cycling through 5 octave frequencies
              - Rhythm pair: two tones 250 ms apart every 30 s (A2 temporal pattern)

        Stage 1 — The Nursery:
            • Warm directional key light rotates slowly (300 s period) to produce
              slow luminance gradients (IT object-permanence calibration).
            • Tone schedule reduced to one gentle tone every 15 s.
        """
        t = self._env_sim_time_s
        self._light_phase += dt_s

        if self._dev_stage == 0:
            # ----------------------------------------------------------
            # Stage 0: The Void — static near-black room.
            # No visual stimuli. Only auditory tones for A1 calibration.
            # ----------------------------------------------------------
            self.set_environment_state({
                'light_direction': (0.3, 1.0, 0.4),
                'light_color':     (0.6, 0.65, 1.0),   # cold blue starlight
                'ambient':         0.03,                  # near-black
            })

            # Tone schedule: cycling octave tones
            if t >= self._next_tone_s:
                freq = self._stage_tone_freqs[
                    self._stage_tone_idx % len(self._stage_tone_freqs)]
                self._stage_tone_idx += 1
                tone = _env_synthesize_tone(freq, 0.5, 0.35)
                self.play_sound(tone, (0.0, 1.0, 3.0))
                self._next_tone_s = t + self._tone_interval_s

                # Rhythm pair: every 5th tone, play a second burst 250 ms later
                if self._stage_tone_idx % 5 == 0:
                    tone2 = _env_synthesize_tone(freq * 1.5, 0.4, 0.25)
                    self.play_sound(tone2, (0.5, 1.0, 3.0))

        elif self._dev_stage == 1:
            # ----------------------------------------------------------
            # Stage 1: slow light rotation + gentle tone every 15 s
            # ----------------------------------------------------------
            import math
            period = 300.0   # 5-minute rotation
            phi = 2.0 * math.pi * (self._light_phase % period) / period
            az = 0.4 + 0.3 * math.sin(phi)
            self.set_environment_state({
                'light_direction': (az, 1.0, 0.3),
                'light_color':     (1.0, 0.95, 0.85),
                'ambient':         0.20,
            })
            if t >= self._next_tone_s:
                freq = self._stage_tone_freqs[
                    self._stage_tone_idx % len(self._stage_tone_freqs)]
                self._stage_tone_idx += 1
                tone = _env_synthesize_tone(freq, 0.6, 0.3)
                self.play_sound(tone, (0.0, 1.0, 3.0))
                self._next_tone_s = t + 15.0

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Stage configuration (called by StageManager on transition)
    # ------------------------------------------------------------------

    def configure_for_stage(self, stage: int) -> None:
        """
        Reconfigure the world for the given developmental stage.

        Called automatically by StageManager._transition().  Safe to call
        manually during testing.

        Stage 0 — The Void
            Black void, point lights at varying positions/colors/intensities.
            Pure sine tones scheduled.  No speech.  M1 output flag set to
            suppressed.  All objects cleared.

        Stage 1 — The Nursery
            Clear scene.  Spawn a small set of geometric primitives with
            distinct saturated colours.  Enable full Newtonian physics.
            White ambient lighting with warm directional key light.
            Set m1_suppressed flag OFF (low-amplitude random movement
            enabled).  Request Father voice onset via parenting flag.

        Stage 2 — The Playroom
            Spawn up to 12 varied primitive objects.  Queue a BlenderBridge
            import if bridge is ready.  Enable PFC myelination flag.
            Brighter, more complex lighting.

        Stage 3 — The Social Space
            Clear scene.  Import a naturalistic Blender environment if
            available.  Enable behavioral video learning flag and DMN
            monitoring.  Agent slots set to multi-agent.

        Stage 4 — The Circus
            Import full Digital Circus Blender scene via bridge.
            Unlock all world manipulation.  Observer camera positioned
            for theatrical view.
        """
        stage = int(stage)

        # ---- Stage 0 — The Void ----------------------------------------
        if stage == DevStage.VOID:
            # Remove all objects
            for uid in list(self._handles.keys()):
                self.remove_object(self._handles[uid])

            # Pure black void: near-zero ambient, no directional colour
            self.set_environment_state({
                'light_direction': (0.0, 1.0, 0.0),
                'light_color':     (0.05, 0.05, 0.10),
                'ambient':         0.02,
            })

            # Developmental flags
            self.m1_suppressed       = True
            self.father_voice_onset  = False
            self.pfc_myelination     = False
            self.avatar_physics      = False
            self.multi_agent         = False
            self._dev_stage          = 0
            self._stage_tone_freqs   = [220.0, 440.0, 880.0, 1760.0, 3520.0]
            self._stage_tone_idx     = 0
            self._next_tone_s        = self._env_sim_time_s + 2.0   # first tone after 2s
            self._tone_interval_s    = 8.0
            self._light_phase        = 0.0   # reset light animation

            print("[environment] Stage 0 — Void configured: black space, lights+tones.")

        # ---- Stage 1 — The Nursery -------------------------------------
        elif stage == DevStage.NURSERY:
            # Clear scene
            for uid in list(self._handles.keys()):
                self.remove_object(self._handles[uid])

            # Warm nursery lighting
            self.set_environment_state({
                'light_direction': (0.4, 1.0, 0.3),
                'light_color':     (1.0, 0.95, 0.85),
                'ambient':         0.20,
            })

            # Spawn 6 distinct geometric primitives (alternating sphere/cube)
            nursery_positions = [
                (-2.0, 0.5, 6.0), (0.0, 0.5, 5.5), (2.0, 0.5, 6.0),
                (-1.5, 0.5, 4.0), (1.5, 0.5, 4.0), (0.0, 0.5, 3.5),
            ]
            for i, pos in enumerate(nursery_positions):
                otype = 'sphere' if i % 2 == 0 else 'cube'
                self.spawn_object(f'nursery_{i}', pos, object_type=otype)

            # Developmental flags
            self.m1_suppressed       = False   # low-amplitude random movement OK
            self.father_voice_onset  = True    # Father begins speaking
            self.pfc_myelination     = False
            self.avatar_physics      = False
            self.multi_agent         = False
            self._dev_stage          = 1
            self._stage_tone_freqs   = [261.6, 329.6, 392.0, 523.3, 659.3]  # C4-E5 (musical)
            self._stage_tone_idx     = 0
            self._next_tone_s        = self._env_sim_time_s + 5.0
            self._light_phase        = 0.0

            print("[environment] Stage 1 — Nursery configured: 6 primitives, Father voice ON.")

        # ---- Stage 2 — The Playroom ------------------------------------
        elif stage == DevStage.PLAYROOM:
            # Clear scene
            for uid in list(self._handles.keys()):
                self.remove_object(self._handles[uid])

            # Bright playroom lighting
            self.set_environment_state({
                'light_direction': (0.3, 1.0, 0.5),
                'light_color':     (1.0, 1.0, 0.95),
                'ambient':         0.30,
            })

            # Spawn up to 12 objects
            playroom_positions = [
                (-3.0, 0.5,  7.0), (-1.5, 0.5,  7.5), (0.0, 0.5,  6.0),
                ( 1.5, 0.5,  7.0), ( 3.0, 0.5,  7.5), (-2.5, 0.5, 5.0),
                (-0.5, 0.5,  5.5), ( 0.5, 0.5,  4.5), ( 2.5, 0.5, 5.0),
                (-1.0, 0.5,  3.5), ( 1.0, 0.5,  3.0), ( 0.0, 0.5, 8.0),
            ]
            for i, pos in enumerate(playroom_positions):
                otype = 'sphere' if i % 3 != 1 else 'cube'
                self.spawn_object(f'play_{i}', pos, object_type=otype)

            # Try BlenderBridge import if available
            if hasattr(self, 'blender_bridge') and self.blender_bridge.is_ready:
                assets = self.blender_bridge.import_glb('data/playroom_assets.glb')
                print(f"[environment] Stage 2 — Blender assets loaded: "
                      f"{len(assets)} objects.")

            # Developmental flags
            self.m1_suppressed      = False
            self.father_voice_onset = True
            self.pfc_myelination    = True    # PFC myelination simulation begins
            self.avatar_physics     = False
            self.multi_agent        = False   # simple scripted agents only

            print("[environment] Stage 2 — Playroom configured: 12 objects, "
                  "PFC myelination ON.")

        # ---- Stage 3 — The Social Space --------------------------------
        elif stage == DevStage.SOCIAL_SPACE:
            # Clear scene
            for uid in list(self._handles.keys()):
                self.remove_object(self._handles[uid])

            # Naturalistic warm lighting
            self.set_environment_state({
                'light_direction': (0.2, 0.9, 0.4),
                'light_color':     (1.0, 0.97, 0.90),
                'ambient':         0.25,
            })

            # Import naturalistic Blender environment if bridge ready
            if hasattr(self, 'blender_bridge') and self.blender_bridge.is_ready:
                scene_objs = self.blender_bridge.import_blend(
                    'data/social_space.blend')
                print(f"[environment] Stage 3 — Blender social scene: "
                      f"{len(scene_objs)} objects.")
            else:
                # Fallback: a few primitives to anchor the space
                for i, pos in enumerate([(0, 0.5, 5), (-2, 0.5, 5), (2, 0.5, 5)]):
                    self.spawn_object(f'social_{i}', pos, object_type='sphere')

            # Developmental flags
            self.m1_suppressed      = False
            self.father_voice_onset = True
            self.pfc_myelination    = True
            self.avatar_physics     = True    # avatar can collide with physics objects
            self.multi_agent        = True    # Claude API multi-agent enabled

            print("[environment] Stage 3 — Social Space configured: "
                  "avatar physics ON, multi-agent ON.")

        # ---- Stage 4 — The Circus --------------------------------------
        elif stage == DevStage.CIRCUS:
            # Clear scene
            for uid in list(self._handles.keys()):
                self.remove_object(self._handles[uid])

            # Circus: colourful dramatic lighting
            self.set_environment_state({
                'light_direction':  (0.5, 0.8, 0.3),
                'light_color':      (1.0, 0.92, 0.80),
                'ambient':          0.18,
                'observer_eye':     (15.0, 10.0, -15.0),
                'observer_target':  (0.0, 2.0, 0.0),
            })

            # Import full Digital Circus Blender scene via bridge
            if hasattr(self, 'blender_bridge') and self.blender_bridge.is_ready:
                circus_objs = self.blender_bridge.import_blend(
                    'data/digital_circus.blend')
                print(f"[environment] Stage 4 — Digital Circus loaded: "
                      f"{len(circus_objs)} objects.")
            else:
                print("[environment] Stage 4 — BlenderBridge not ready; "
                      "Circus scene pending Blender install.")

            # Developmental flags — all unlocked
            self.m1_suppressed      = False
            self.father_voice_onset = True
            self.pfc_myelination    = True
            self.avatar_physics     = True
            self.multi_agent        = True

            print("[environment] Stage 4 — The Circus configured: all systems unlocked.")

        else:
            print(f"[environment] configure_for_stage({stage}): unknown stage, ignoring.")

    # ------------------------------------------------------------------
    # Developmental flag accessors (set to defaults on __init__)
    # ------------------------------------------------------------------

    def _init_dev_flags(self) -> None:
        """Initialise developmental flags to Stage 0 defaults."""
        self.m1_suppressed      = True
        self.father_voice_onset = False
        self.pfc_myelination    = False
        self.avatar_physics     = False
        self.multi_agent        = False
        self._stage_tone_freqs  = [220.0, 440.0, 880.0, 1760.0, 3520.0]
        self._stage_tone_idx    = 0
        self.blender_bridge: Optional[BlenderBridge] = None

    def attach_blender_bridge(self, bridge: 'BlenderBridge') -> None:
        """Attach a running BlenderBridge for use during Stage 2+."""
        self.blender_bridge = bridge


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------

def run_environment_demo(n_frames: int = 120):
    """
    Headless smoke-test covering:
    - CaineEnvironment lifecycle (start / step / stop)
    - spawn_object / move_object / get_camera_feed
    - play_sound / pop_audio_queue
    - BlenderBridge (stub mode if blender not installed)
    - DevStage / StageExitEvaluator / StageManager
    - configure_for_stage() for stages 0-4
    - Contact sheet PNG saved to output/
    """
    _OUTPUT_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Core environment
    # ------------------------------------------------------------------
    env = CaineEnvironment()
    env.start()

    sphere = env.spawn_object('ball_1', (0.0, 2.0, 5.0), object_type='sphere')
    cube   = env.spawn_object('box_1',  (2.0, 1.0, 5.0), object_type='cube')

    # Inject a 440 Hz tone
    t    = np.linspace(0, 0.5, 22050 // 2, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440.0 * t)
    env.play_sound(tone, position=(0.0, 1.0, 5.0))
    audio_events = env.pop_audio_queue()
    print(f"[demo] Audio queue: {len(audio_events)} event(s) popped.")

    frames = []
    for i in range(n_frames):
        env.step()
        frames.append(env.get_camera_feed())
        if i % 30 == 0:
            env.move_object(sphere, (0.5, 3.0, 0.0))

    assert frames[0].shape == (64, 64, 3), f"Unexpected frame shape: {frames[0].shape}"
    assert frames[0].dtype == np.uint8,    f"Unexpected dtype: {frames[0].dtype}"
    print(f"[demo] {n_frames} frames rendered — shape={frames[0].shape}, "
          f"dtype={frames[0].dtype}  OK")

    # ------------------------------------------------------------------
    # 2. BlenderBridge smoke-test (stub mode)
    # ------------------------------------------------------------------
    bridge = BlenderBridge()
    started = bridge.start(timeout_s=2.0)   # will fail gracefully if no blender
    print(f"[demo] BlenderBridge.start() -> {started} "
          f"(stub mode expected in CI)")
    objs = bridge.import_glb('data/test.glb')
    tex  = bridge.generate_procedural_texture({'type': 'noise'})
    mesh = bridge.apply_geometry_nodes('Cube', {})
    pc   = bridge.bake_physics('Scene', 24)
    print(f"[demo] BlenderBridge stubs: import_glb={objs}, "
          f"texture shape={tex.pixels.shape}, "
          f"mesh verts={len(mesh.vertices)}, "
          f"physics frames={pc.frames}")
    bridge.close()

    # Attach bridge to env for configure_for_stage calls
    env.attach_blender_bridge(bridge)

    # ------------------------------------------------------------------
    # 3. configure_for_stage() round-trip for all 5 stages
    # ------------------------------------------------------------------
    print()
    print("[demo] configure_for_stage() round-trip:")
    for s in range(5):
        env.configure_for_stage(s)
        print(f"  Stage {s}: m1_suppressed={env.m1_suppressed}, "
              f"father_voice={env.father_voice_onset}, "
              f"pfc_myelin={env.pfc_myelination}, "
              f"avatar_phys={env.avatar_physics}, "
              f"multi_agent={env.multi_agent}")

    # ------------------------------------------------------------------
    # 4. StageManager with synthetic cortex_state
    # ------------------------------------------------------------------
    print()
    print("[demo] StageManager tick tests:")

    # Reset to Stage 0
    env.configure_for_stage(0)
    stages = StageManager(env, start_stage=DevStage.VOID)

    # Cortex state that does NOT meet Stage 0 exit conditions
    cs_not_met = {
        'v1_orientation_selectivity': 0.3,
        'a1_tonotopic_gradient':      0.5,
        'cortisol_chronic':           True,
    }
    result = stages.tick(cs_not_met, dt_ms=20.0, sim_time_s=100.0)
    assert result is None, "Expected no transition with failing conditions"
    print(f"  Tick (conditions not met): transition={result}  "
          f"summary={stages.conditions_summary()}")

    # Cortex state that DOES meet Stage 0 exit conditions + min time
    cs_met_stage0 = {
        'v1_orientation_selectivity': 0.75,
        'a1_tonotopic_gradient':      0.90,
        'cortisol_chronic':           False,
    }
    fake_sim_s = _STAGE_MIN_RUNTIME_S[DevStage.VOID] + 10.0  # past 48 h
    result = stages.tick(cs_met_stage0, dt_ms=20.0, sim_time_s=fake_sim_s)
    assert result == DevStage.NURSERY, \
        f"Expected transition to NURSERY, got {result}"
    print(f"  Tick (Stage 0 all met): transition={result} ({stages.stage_name})  OK")

    # Verify configure_for_stage(1) was invoked: m1_suppressed should be False
    assert env.m1_suppressed is False, \
        "configure_for_stage(1) should set m1_suppressed=False"
    print(f"  m1_suppressed after Nursery transition: {env.m1_suppressed}  OK")

    # One more tick at Stage 1 — conditions not met yet
    result = stages.tick({'it_category_count': 2}, dt_ms=20.0,
                         sim_time_s=fake_sim_s + 100.0)
    assert result is None
    print(f"  Stage 1 tick (conditions partial): transition={result}  OK")

    # ------------------------------------------------------------------
    # 5. StageExitEvaluator direct checks
    # ------------------------------------------------------------------
    print()
    print("[demo] StageExitEvaluator direct checks:")
    ev = StageExitEvaluator()
    assert ev.stage0_v1_orientation({'v1_orientation_selectivity': 0.65}) is True
    assert ev.stage0_v1_orientation({'v1_orientation_selectivity': 0.55}) is False
    assert ev.stage1_it_categories({'it_category_count': 5}) is True
    assert ev.stage1_it_categories({'it_category_count': 4}) is False
    assert ev.stage2_father_voice_differential({'father_voice_ratio': 2.1}) is True
    assert ev.stage3_dmn_correlation({'dmn_correlation': 0.5}) is True
    all_met, conds = ev.all_met(DevStage.VOID,
                                {'v1_orientation_selectivity': 0.9,
                                 'a1_tonotopic_gradient': 0.95,
                                 'cortisol_chronic': False},
                                sim_time_s=_STAGE_MIN_RUNTIME_S[DevStage.VOID] + 1,
                                stage_entry_sim_s=0.0)
    assert all_met is True, f"Expected all_met=True, got conds={conds}"
    print(f"  all_met(Stage0, passing state) = {all_met}  OK")

    # ------------------------------------------------------------------
    # 6. Contact sheet
    # ------------------------------------------------------------------
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
        fig.suptitle("CAINE Environment — Module 9 Demo", fontsize=14)
        plt.tight_layout()
        out = os.path.join(_OUTPUT_DIR, 'caine_environment_demo.png')
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"[demo] Contact sheet saved to {out}")
    except ImportError:
        print("[demo] matplotlib not available — skipping contact sheet.")

    env.stop()
    print()
    print("[demo] Module 9 smoke-test PASSED.")
    return frames


if __name__ == '__main__':
    run_environment_demo()
