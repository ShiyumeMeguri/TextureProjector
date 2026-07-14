"""
GPU depth capture for TextureProjector.

Renders the scene depth into an offscreen buffer using the exact same
view/projection matrices as the color capture and the UV projection,
which makes depth-based generation pixel-aligned by construction.

Performance: the depth buffer is consumed through the buffer protocol
straight into numpy (no Python list round-trip).
"""

import os
import numpy as np

import bpy
import gpu
from gpu_extras.batch import batch_for_shader


class DepthRenderError(Exception):
    pass


def render_depth_map(context, view_matrix, projection_matrix,
                     width: int, height: int, filepath: str,
                     invert: bool = True) -> str:
    """Render normalized scene depth to a PNG at filepath.

    Convention (invert=True): near = white, far = black, background = black.
    Must be called from the main thread (GPU access).
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise DepthRenderError(f"Invalid depth resolution {width}x{height}")

    offscreen = gpu.types.GPUOffScreen(width, height)
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)

            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(view_matrix)
                gpu.matrix.load_projection_matrix(projection_matrix)

                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                depsgraph = context.evaluated_depsgraph_get()

                for instance in depsgraph.object_instances:
                    if not instance.show_self:
                        continue
                    obj = instance.object
                    if obj.type != 'MESH':
                        continue

                    mesh = obj.evaluated_get(depsgraph).to_mesh()
                    if not mesh or not len(mesh.vertices) or not len(mesh.loop_triangles):
                        if mesh:
                            obj.to_mesh_clear()
                        continue

                    mesh.transform(instance.matrix_world)
                    mesh.calc_loop_triangles()

                    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
                    mesh.vertices.foreach_get("co", verts)
                    mesh.loop_triangles.foreach_get("vertices", tris)

                    batch = batch_for_shader(
                        shader, 'TRIS',
                        {"pos": verts.reshape(-1, 3)},
                        indices=tris.reshape(-1, 3),
                    )
                    batch.draw(shader)
                    obj.to_mesh_clear()

            depth_buf = fb.read_depth(0, 0, width, height)
            try:
                depth = np.asarray(depth_buf, dtype=np.float32).reshape(height, width)
            except (TypeError, ValueError):
                depth = np.array(depth_buf.to_list(), dtype=np.float32).reshape(height, width)

            gpu.state.depth_test_set('NONE')
    finally:
        offscreen.free()

    # Normalize: geometry pixels are depth < 1.0; background stays black.
    geometry = depth < 1.0
    if invert:
        depth = 1.0 - depth
    if geometry.any():
        vals = depth[geometry]
        d_min, d_max = float(vals.min()), float(vals.max())
        if d_max > d_min:
            depth = np.where(geometry, (depth - d_min) / (d_max - d_min), 0.0)
        else:
            depth = np.where(geometry, 1.0, 0.0)
    np.clip(depth, 0.0, 1.0, out=depth)

    _save_grayscale(depth, width, height, filepath)
    return filepath


def _save_grayscale(values: np.ndarray, width: int, height: int, filepath: str):
    """Write a HxW float array as a grayscale PNG via a scratch Blender image."""
    pixels = np.empty((height, width, 4), dtype=np.float32)
    pixels[..., 0] = values
    pixels[..., 1] = values
    pixels[..., 2] = values
    pixels[..., 3] = 1.0
    _save_rgba(pixels, width, height, filepath)


def _save_rgba(pixels: np.ndarray, width: int, height: int, filepath: str):
    """Write a HxWx4 float array as a PNG via a scratch Blender image."""
    img = bpy.data.images.new("gemini_gpu_scratch", width=width, height=height,
                              alpha=True, float_buffer=False)
    try:
        img.pixels.foreach_set(
            np.ascontiguousarray(pixels, dtype=np.float32).ravel())
        img.filepath_raw = filepath
        img.file_format = 'PNG'
        img.save()
    finally:
        bpy.data.images.remove(img)


class DepthRenderer:
    """Thin compatibility wrapper kept for external callers."""

    def render_depth_gpu(self, context, width=None, height=None,
                         view_matrix=None, projection_matrix=None,
                         invert=True, filepath=None) -> str:
        if view_matrix is None or projection_matrix is None:
            rv3d = context.space_data.region_3d
            view_matrix = rv3d.view_matrix.copy()
            projection_matrix = rv3d.window_matrix.copy()
            width = width or context.region.width
            height = height or context.region.height

        if not filepath:
            import tempfile
            filepath = os.path.join(
                tempfile.mkdtemp(prefix="gemini_depth_"), "gpu_depth.png")

        return render_depth_map(context, view_matrix, projection_matrix,
                                width, height, filepath, invert=invert)
