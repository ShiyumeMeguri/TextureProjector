import bpy
import numpy as np
import os
import tempfile
import threading
import time
import gpu
from gpu_extras.batch import batch_for_shader
from typing import Optional, Tuple, List

class DepthRenderError(Exception):
    """Custom exception for depth rendering errors"""
    pass

class DepthRenderer:
    """Handles depth map rendering and normalization"""
    
    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []  # I track temporary directories too
    
    def cleanup_temp_files(self):
        """I clean up all temporary files and directories."""
        print("Starting cleanup of temporary files...")
        
        files_cleaned = 0
        for filepath in self.temp_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    files_cleaned += 1
                    print(f"Removed temp file: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Could not remove temp file {filepath}: {e}")
        
        dirs_cleaned = 0
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                    dirs_cleaned += 1
                    print(f"Removed temp directory: {os.path.basename(temp_dir)}")
            except Exception as e:
                print(f"Could not remove temp directory {temp_dir}: {e}")
        
        self.temp_files.clear()
        self.temp_dirs.clear()
        
        print(f"Cleanup completed: {files_cleaned} files, {dirs_cleaned} directories removed")
    
    
    def validate_scene(self, scene, require_camera=True) -> None:
        """Validate scene is ready for rendering"""

        if require_camera and not scene.camera:
            raise DepthRenderError("No active camera found. Please add a camera to the scene.")
        

        visible_objects = [obj for obj in scene.objects if obj.visible_get()]
        if len(visible_objects) == 0:
            raise DepthRenderError("No visible objects found. Please add some objects to the scene.")
        

        if scene.render.engine not in ['CYCLES', 'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT']:
            print(f"Warning: Render engine {scene.render.engine} may not support depth pass properly")
    
    
    def render_depth_gpu(self, context, width=None, height=None, view_matrix=None, projection_matrix=None, invert=True) -> str:
        """
        Unified GPU-based depth rendering. 
        Works for both viewport and camera by passing appropriate matrices.
        """
        import bmesh
        import mathutils
        
        scene = context.scene

        self.validate_scene(scene, require_camera=False)
        

        if view_matrix is None or projection_matrix is None:
            rv3d = context.space_data.region_3d
            view_matrix = rv3d.view_matrix.copy()
            projection_matrix = rv3d.window_matrix.copy()
            width = width or context.region.width
            height = height or context.region.height
        
        width = int(width)
        height = int(height)
        

        temp_dir = tempfile.mkdtemp(prefix="gemini_depth_gpu_")
        depth_path = os.path.join(temp_dir, "gpu_depth.png")
        self.temp_files.append(depth_path)
        self.temp_dirs.append(temp_dir)
        
        def _execute():
            offscreen = gpu.types.GPUOffScreen(width, height)
            
            with offscreen.bind():
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1)
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.depth_mask_set(True)
                
                with gpu.matrix.push_pop():
                    gpu.matrix.load_matrix(view_matrix)
                    gpu.matrix.load_projection_matrix(projection_matrix)
                    
                    # I use the builtin UNIFORM_COLOR shader for efficient depth drawing
                    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                    
                    depsgraph = context.evaluated_depsgraph_get()
                    for instance in depsgraph.object_instances:
                        if not instance.show_self:
                            continue
                            
                        obj = instance.object
                        if obj.type != 'MESH':
                            continue
                            
                        mesh = obj.evaluated_get(depsgraph).to_mesh()
                        if not mesh or len(mesh.vertices) == 0 or len(mesh.loop_triangles) == 0:
                            if mesh: obj.to_mesh_clear()
                            continue
                            
                        mesh.transform(instance.matrix_world)
                        mesh.calc_loop_triangles()
                        
                        vertices = np.empty((len(mesh.vertices), 3), 'f')
                        indices = np.empty((len(mesh.loop_triangles), 3), 'i')
                        
                        mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))
                        mesh.loop_triangles.foreach_get("vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))
                        
                        batch = batch_for_shader(
                            shader, 'TRIS',
                            {"pos": vertices},
                            indices=indices,
                        )
                        batch.draw(shader)
                        obj.to_mesh_clear()
                
                depth_data = np.array(fb.read_depth(0, 0, width, height).to_list())
                
                # I invert the depth if requested to align with common AI models (1=near, 0=far)
                if invert:
                    depth_data = 1.0 - depth_data
                

                masked_depth = np.ma.masked_equal(depth_data, 0, copy=False)
                if masked_depth.count() > 0:
                    d_min, d_max = masked_depth.min(), depth_data.max()
                    if d_max > d_min:
                        depth_data = (depth_data - d_min) / (d_max - d_min)
                
                depth_data = np.clip(depth_data, 0, 1)
                

                img = bpy.data.images.new("temp_gpu_depth", width=width, height=height)
                pixels = np.zeros((width * height, 4), dtype=np.float32)
                pixels[:, 0] = depth_data.ravel()
                pixels[:, 1] = depth_data.ravel()
                pixels[:, 2] = depth_data.ravel()
                pixels[:, 3] = 1.0
                img.pixels.foreach_set(pixels.ravel())
                img.filepath_raw = depth_path
                img.file_format = 'PNG'
                img.save()
                
                bpy.data.images.remove(img)
            
            offscreen.free()
            gpu.state.depth_test_set('NONE')

        # I since this involves GPU operations, it MUST run in the main thread
        # I operators usually run in the main thread, so this is safe for synchronous call
        _execute()
        
        return depth_path


    def render_depth_viewport(self, context, width=None, height=None, invert=True) -> str:
        """I use render_depth_gpu instead for consistency."""
        return self.render_depth_gpu(context, width, height, invert=invert)



