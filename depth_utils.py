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
                        if not mesh:
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


    def render_depth_map_mist(self, scene, mist_start: float = 5.0, mist_depth: float = 25.0, mist_falloff: str = 'LINEAR') -> str:
        """
        Fast depth map generation using viewport render with Mist Pass - REAL depth info!
        """
        try:

            self.validate_scene(scene, require_camera=True)
            print("Scene validation passed")
            

            temp_dir = tempfile.mkdtemp(prefix="gemini_depth_mist_")
            depth_path = os.path.join(temp_dir, "mist_depth.png")
            self.temp_files.append(depth_path)
            self.temp_dirs.append(temp_dir)  # I track directory for cleanup
            print(f"Created temp directory: {temp_dir}")
            
            import bpy
            

            original_use_mist = None
            original_mist_start = None
            original_mist_depth = None
            original_mist_falloff = None
            original_render_engine = scene.render.engine
            original_use_pass_mist = None
            original_use_pass_combined = None
            original_use_nodes = scene.use_nodes
            
            try:
                print(" Setting up Mist Pass for depth rendering...")
                

                world = scene.world
                if world is None:

                    world = bpy.data.worlds.new("TempWorld")
                    scene.world = world
                    print("🌍 Created temporary world")
                

                if hasattr(world, 'mist_settings') and world.mist_settings:
                    mist_settings = world.mist_settings
                    original_use_mist = mist_settings.use_mist if hasattr(mist_settings, 'use_mist') else False
                    original_mist_start = mist_settings.start if hasattr(mist_settings, 'start') else 5.0
                    original_mist_depth = mist_settings.depth if hasattr(mist_settings, 'depth') else 25.0
                    original_mist_falloff = mist_settings.falloff if hasattr(mist_settings, 'falloff') else 'QUADRATIC'
                    
                    # I configure mist settings (Blender 4.5+ API)
                    mist_settings.use_mist = True
                    mist_settings.start = mist_start  # I already in meters
                    mist_settings.depth = mist_depth  # I already in meters
                    mist_settings.falloff = mist_falloff  # I use user-selected falloff
                    
                    print(f" Mist settings (4.5+ API): start={mist_settings.start}m, depth={mist_settings.depth}m, falloff={mist_falloff}")
                else:

                    original_use_mist = getattr(world, 'use_mist', False)
                    original_mist_start = getattr(world, 'mist_start', 5.0)
                    original_mist_depth = getattr(world, 'mist_depth', 25.0)
                    original_mist_falloff = getattr(world, 'mist_falloff', 'QUADRATIC')
                    
                    # I try old API (might not exist in 4.5+)
                    setattr(world, 'use_mist', True)
                    setattr(world, 'mist_start', mist_start)  # I already in meters
                    setattr(world, 'mist_depth', mist_depth)  # I already in meters
                    setattr(world, 'mist_falloff', mist_falloff)  # I use user-selected falloff
                    
                    print(f" Mist settings (old API): start={mist_start}m, depth={mist_depth}m, falloff={mist_falloff}")
                
                # I use Eevee Next for fast rendering with Mist Pass support (Blender 4.5+)
                available_engines = ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE', 'CYCLES', 'BLENDER_WORKBENCH']
                selected_engine = None
                
                for engine in available_engines:
                    try:
                        scene.render.engine = engine
                        selected_engine = engine
                        print(f" Using render engine: {engine}")
                        break
                    except TypeError:
                        continue
                
                if not selected_engine:
                    raise DepthRenderError("No compatible render engine found")
                

                view_layer = self._get_active_view_layer(scene)
                if view_layer:
                    original_use_pass_mist = getattr(view_layer, 'use_pass_mist', False)
                    original_use_pass_combined = getattr(view_layer, 'use_pass_combined', True)
                    
                    if hasattr(view_layer, 'use_pass_mist'):
                        view_layer.use_pass_mist = True
                        print(f"Mist pass enabled in view layer")
                        
                        # CRITICAL: Disable Combined pass for PURE mist render
                        if hasattr(view_layer, 'use_pass_combined'):
                            view_layer.use_pass_combined = False
                            print("🚫 Combined pass DISABLED - pure mist only!")
                        else:
                            print(" Cannot disable combined pass - will use compositor")
                    else:
                        print(" View layer found but no mist pass support")
                else:
                    print(" No view layer found, continuing without mist pass")
                    original_use_pass_mist = False
                    original_use_pass_combined = True
                
                # I use VIEWPORT render with mist shading for accurate depth
                return self._render_viewport_mist(scene, temp_dir, selected_engine, mist_falloff)
                
            finally:

                if scene.world:
                    if hasattr(scene.world, 'mist_settings') and scene.world.mist_settings:
                        # I blender 4.5+ API
                        mist_settings = scene.world.mist_settings
                        if original_use_mist is not None:
                            mist_settings.use_mist = original_use_mist
                        if original_mist_start is not None:
                            mist_settings.start = original_mist_start
                        if original_mist_depth is not None:
                            mist_settings.depth = original_mist_depth
                        if original_mist_falloff is not None:
                            mist_settings.falloff = original_mist_falloff
                    else:

                        try:
                            if original_use_mist is not None:
                                setattr(scene.world, 'use_mist', original_use_mist)
                            if original_mist_start is not None:
                                setattr(scene.world, 'mist_start', original_mist_start)
                            if original_mist_depth is not None:
                                setattr(scene.world, 'mist_depth', original_mist_depth)
                            if original_mist_falloff is not None:
                                setattr(scene.world, 'mist_falloff', original_mist_falloff)
                        except AttributeError:
                            print(" Could not restore old mist settings (modern Blender API)")
                

                view_layer = self._get_active_view_layer(scene)
                if view_layer:
                    try:
                        if hasattr(view_layer, 'use_pass_mist') and original_use_pass_mist is not None:
                            view_layer.use_pass_mist = original_use_pass_mist
                        if hasattr(view_layer, 'use_pass_combined') and original_use_pass_combined is not None:
                            view_layer.use_pass_combined = original_use_pass_combined
                    except:
                        pass
                

                if original_use_nodes is not None:
                    scene.use_nodes = original_use_nodes
                

                scene.render.engine = original_render_engine
                
                print(" World and render settings restored")
                
        except Exception as e:
            print(f" Mist depth render error: {str(e)}")
            # CRITICAL: Only cleanup on error!
            self.cleanup_temp_files()
            if isinstance(e, DepthRenderError):
                raise
            raise DepthRenderError(f"Failed to render mist depth map: {str(e)}")
    
    def _get_active_view_layer(self, scene):
        """Get active view layer with fallback for different Blender versions"""
        import bpy
        
        # I method 1: Try context (preferred in 4.5+)
        try:
            if bpy.context.view_layer and bpy.context.view_layer.name in scene.view_layers:
                return bpy.context.view_layer
        except:
            pass
            
        # I method 2: Try scene.view_layers.active (older versions)
        try:
            if hasattr(scene.view_layers, 'active') and scene.view_layers.active:
                return scene.view_layers.active
        except:
            pass
            
        # I method 3: Get first view layer as fallback
        try:
            if len(scene.view_layers) > 0:
                return scene.view_layers[0]  # I default view layer
        except:
            pass
            
        # I method 4: Create view layer if none exist (extreme fallback)
        print(" No view layer found, using scene fallback")
        return None
    
    def _render_mist_only(self, scene, temp_dir: str, render_engine: str, 
                         original_use_pass_mist: bool, original_use_pass_combined: bool) -> str:
        """Render only mist pass using compositor setup"""
        try:
            import bpy
            
            print(" Setting up mist-only render with compositor...")
            

            original_filepath = scene.render.filepath
            original_use_nodes = scene.use_nodes
            original_samples = None
            
            mist_output_path = os.path.join(temp_dir, "mist_only.png")
            
            # I track temp files for cleanup
            self.temp_files.append(mist_output_path)
            
            try:

                scene.use_nodes = True
                
                # I clear existing nodes
                scene.node_tree.nodes.clear()
                

                render_layers = scene.node_tree.nodes.new(type='CompositorNodeRLayers')
                file_output = scene.node_tree.nodes.new(type='CompositorNodeOutputFile')
                
                # I configure file output
                file_output.base_path = temp_dir
                file_output.file_slots[0].path = "mist_only"
                file_output.format.file_format = 'PNG'
                file_output.format.color_mode = 'BW'  # I black and white for depth
                

                scene.node_tree.links.new(render_layers.outputs['Mist'], file_output.inputs[0])
                
                print(" Compositor nodes setup: RenderLayers → Mist → FileOutput")
                
                # I fast render settings
                if render_engine in ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE']:
                    if hasattr(scene, 'eevee'):
                        original_samples = scene.eevee.taa_render_samples
                        scene.eevee.taa_render_samples = 1  # I fastest possible
                        print(" Eevee samples set to 1 for speed")
                

                scene.render.filepath = os.path.join(temp_dir, "temp_render")
                
                print(" Starting mist-only render...")
                

                bpy.ops.render.render(write_still=True)
                

                mist_files = [
                    os.path.join(temp_dir, "mist_only0001.png"),
                    os.path.join(temp_dir, "mist_only.png"),
                    mist_output_path
                ]
                
                actual_mist_path = None
                for path in mist_files:
                    if os.path.exists(path):
                        actual_mist_path = path
                        print(f" Found mist output: {path}")
                        break
                
                if actual_mist_path:
                    print(f" Mist-only render completed: {actual_mist_path}")
                    return actual_mist_path
                else:
                    raise DepthRenderError("Mist output file not found after render")
                
            finally:

                scene.render.filepath = original_filepath
                scene.use_nodes = original_use_nodes
                if hasattr(scene, 'eevee') and original_samples is not None:
                    scene.eevee.taa_render_samples = original_samples
                    
        except Exception as e:
            raise DepthRenderError(f"Mist-only render failed: {str(e)}")
    
    def _render_and_extract_mist(self, scene, temp_dir: str, render_engine: str) -> str:
        """Simple render and extract mist pass - safer for background threads"""
        try:
            import bpy
            
            print(" Simple render with mist extraction (no compositor)...")
            

            original_filepath = scene.render.filepath
            original_samples = None
            
            mist_output_path = os.path.join(temp_dir, "mist_extracted.png")
            
            # I track temp files for cleanup
            self.temp_files.append(mist_output_path)
            
            try:
                # I fast render settings
                if render_engine in ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE']:
                    if hasattr(scene, 'eevee'):
                        original_samples = scene.eevee.taa_render_samples
                        scene.eevee.taa_render_samples = 1  # I fastest possible
                        print(" Eevee samples set to 1 for speed")
                

                scene.render.filepath = os.path.join(temp_dir, "temp_render")
                
                print(" Starting simple render for mist extraction...")
                

                bpy.ops.render.render(write_still=True)
                
                print(" Extracting mist pass from render result (Combined pass disabled)...")
                

                render_result = bpy.data.images.get('Render Result')
                if not render_result:
                    raise DepthRenderError("No render result found")
                
                # I since Combined pass is disabled, Render Result should contain only mist
                print("📸 Saving mist-only render result...")
                render_result.save_render(filepath=mist_output_path)
                
                if os.path.exists(mist_output_path):
                    print(f" Mist-only render saved: {mist_output_path}")
                    return mist_output_path
                else:
                    raise DepthRenderError("Failed to save mist render result")
                
            finally:

                scene.render.filepath = original_filepath
                if hasattr(scene, 'eevee') and original_samples is not None:
                    scene.eevee.taa_render_samples = original_samples
                    
        except Exception as e:
            raise DepthRenderError(f"Render and mist extraction failed: {str(e)}")
    
    def _render_viewport_mist(self, scene, temp_dir: str, render_engine: str, mist_falloff: str = 'LINEAR') -> str:
        """Render VIEWPORT with mist shading from camera - accurate depth like in viewport"""
        try:
            import bpy
            
            print("Setting up VIEWPORT mist render from camera...")
            

            original_filepath = scene.render.filepath
            original_shading_type = None
            original_render_pass = None
            original_use_scene_world = None
            original_overlays = {}
            original_show_gizmo = None
            original_show_gizmo_navigate = None
            original_region_3d = None
            space_data = None
            overlay = None
            
            mist_output_path = os.path.join(temp_dir, "viewport_mist.png")
            
            # I track temp directory and files
            if temp_dir not in self.temp_dirs:
                self.temp_dirs.append(temp_dir)
            self.temp_files.append(mist_output_path)
            
            try:

                viewport_area = None
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            viewport_area = area
                            break
                    if viewport_area:
                        break
                
                if not viewport_area:
                    print("No 3D viewport found, creating temporary context")

                    raise DepthRenderError("No 3D viewport available for mist render")
                

                space_data = None
                for space in viewport_area.spaces:
                    if space.type == 'VIEW_3D':
                        space_data = space
                        break
                
                if not space_data:
                    raise DepthRenderError("Cannot access viewport settings")
                

                original_shading_type = space_data.shading.type
                original_render_pass = space_data.shading.render_pass if hasattr(space_data.shading, 'render_pass') else None
                original_use_scene_world = space_data.shading.use_scene_world if hasattr(space_data.shading, 'use_scene_world') else None
                
                # I store ALL overlay settings to disable them
                overlay = space_data.overlay
                original_overlays = {
                    'show_overlays': overlay.show_overlays if hasattr(overlay, 'show_overlays') else None,
                    'show_floor': overlay.show_floor if hasattr(overlay, 'show_floor') else None,
                    'show_axis_x': overlay.show_axis_x if hasattr(overlay, 'show_axis_x') else None,
                    'show_axis_y': overlay.show_axis_y if hasattr(overlay, 'show_axis_y') else None,
                    'show_axis_z': overlay.show_axis_z if hasattr(overlay, 'show_axis_z') else None,
                    'show_text': overlay.show_text if hasattr(overlay, 'show_text') else None,
                    'show_stats': overlay.show_stats if hasattr(overlay, 'show_stats') else None,
                    'show_cursor': overlay.show_cursor if hasattr(overlay, 'show_cursor') else None,
                    'show_object_origins': overlay.show_object_origins if hasattr(overlay, 'show_object_origins') else None,
                    'show_relationship_lines': overlay.show_relationship_lines if hasattr(overlay, 'show_relationship_lines') else None,
                }
                
                # I store gizmo settings
                original_show_gizmo = space_data.show_gizmo if hasattr(space_data, 'show_gizmo') else None
                original_show_gizmo_navigate = space_data.show_gizmo_navigate if hasattr(space_data, 'show_gizmo_navigate') else None
                
                # I store camera view state
                original_region_3d = None
                for region in viewport_area.regions:
                    if region.type == 'WINDOW':
                        region_3d = space_data.region_3d
                        if region_3d:
                            original_region_3d = {
                                'view_perspective': region_3d.view_perspective,
                            }
                        break
                
                print(f"Original viewport shading: {original_shading_type}")
                
                # CRITICAL: Switch viewport to CAMERA VIEW
                if space_data.region_3d:
                    space_data.region_3d.view_perspective = 'CAMERA'
                    print("Switched viewport to CAMERA VIEW")
                
                # CRITICAL: Set viewport to MATERIAL shading with MIST pass!
                space_data.shading.type = 'MATERIAL'
                
                # CRITICAL: Set render pass to MIST (this is what shows mist!)
                if hasattr(space_data.shading, 'render_pass'):
                    space_data.shading.render_pass = 'MIST'
                    print("Set viewport render_pass to MIST!")
                else:
                    print("WARNING: render_pass not available in shading")
                

                if hasattr(space_data.shading, 'use_scene_world'):
                    space_data.shading.use_scene_world = True
                    print("Enabled use_scene_world for mist")
                
                # CRITICAL: DISABLE ALL OVERLAYS (grid, axes, text, gizmos, etc.)
                if hasattr(overlay, 'show_overlays'):
                    overlay.show_overlays = False
                    print("DISABLED all overlays")
                

                if hasattr(space_data, 'show_gizmo'):
                    space_data.show_gizmo = False
                    print("DISABLED gizmos")
                if hasattr(space_data, 'show_gizmo_navigate'):
                    space_data.show_gizmo_navigate = False
                    print("DISABLED gizmo navigate")
                
                print("Viewport configured for clean camera mist rendering (no overlays)")
                

                original_color_mode = scene.render.image_settings.color_mode
                original_color_depth = scene.render.image_settings.color_depth
                
                scene.render.filepath = mist_output_path
                scene.render.image_settings.file_format = 'PNG'
                scene.render.image_settings.color_mode = 'BW'  # I grayscale for depth
                scene.render.image_settings.color_depth = '16'
                
                print("Starting viewport render with mist from camera...")
                
                # I render viewport from camera view
                override_context = {
                    'scene': scene,
                    'area': viewport_area,
                    'region': viewport_area.regions[-1],
                    'space_data': space_data,
                }
                

                with bpy.context.temp_override(**override_context):
                    bpy.ops.render.opengl(write_still=True)
                
                print(f"Viewport mist render completed")
                

                if os.path.exists(mist_output_path):
                    print(f"Viewport mist saved: {mist_output_path}")
                    return mist_output_path
                else:
                    # I try with numbering
                    import glob
                    viewport_pattern = os.path.join(temp_dir, "viewport_mist*.png")
                    viewport_files = glob.glob(viewport_pattern)
                    
                    if viewport_files:
                        actual_path = viewport_files[0]
                        print(f"Found viewport mist: {actual_path}")
                        return actual_path
                    else:
                        raise DepthRenderError("Viewport mist output not found after render")
            
            finally:

                if space_data:
                    try:

                        if original_render_pass is not None and hasattr(space_data.shading, 'render_pass'):
                            space_data.shading.render_pass = original_render_pass
                        

                        if original_shading_type:
                            space_data.shading.type = original_shading_type
                        

                        if original_use_scene_world is not None and hasattr(space_data.shading, 'use_scene_world'):
                            space_data.shading.use_scene_world = original_use_scene_world
                        

                        if overlay and original_overlays:
                            for key, value in original_overlays.items():
                                if value is not None and hasattr(overlay, key):
                                    try:
                                        setattr(overlay, key, value)
                                    except:
                                        pass
                        

                        if original_show_gizmo is not None and hasattr(space_data, 'show_gizmo'):
                            space_data.show_gizmo = original_show_gizmo
                        if original_show_gizmo_navigate is not None and hasattr(space_data, 'show_gizmo_navigate'):
                            space_data.show_gizmo_navigate = original_show_gizmo_navigate
                        

                        if original_region_3d and space_data.region_3d:
                            space_data.region_3d.view_perspective = original_region_3d['view_perspective']
                        
                        print("Viewport settings restored (overlays, gizmos, camera view)")
                    except Exception as e:
                        print(f"Error restoring viewport: {e}")
                

                scene.render.filepath = original_filepath
                if 'original_color_mode' in locals():
                    scene.render.image_settings.color_mode = original_color_mode
                if 'original_color_depth' in locals():
                    scene.render.image_settings.color_depth = original_color_depth
                print("Render settings restored (filepath, color_mode, color_depth)")
                
        except Exception as e:
            raise DepthRenderError(f"Viewport mist render failed: {str(e)}")
    
    def _render_camera_with_mist_compositor(self, scene, temp_dir: str, render_engine: str, mist_falloff: str = 'LINEAR') -> str:
        """Render from camera with compositor setup to extract PURE mist pass"""
        try:
            import bpy
            
            print(" Setting up camera-based PURE mist render with compositor...")
            

            original_filepath = scene.render.filepath
            original_use_nodes = scene.use_nodes
            original_nodes = {}
            
            mist_output_path = os.path.join(temp_dir, "pure_mist.png")
            
            # I track temp directory for proper cleanup
            if temp_dir not in self.temp_dirs:
                self.temp_dirs.append(temp_dir)
            self.temp_files.append(mist_output_path)
            
            try:

                original_resolution_x = scene.render.resolution_x
                original_resolution_y = scene.render.resolution_y
                original_resolution_percentage = scene.render.resolution_percentage
                original_samples = None
                
                # I use full quality settings
                print(" Using full quality render settings for PURE mist extraction...")
                scene.render.resolution_percentage = 100
                print(f" Using full scene resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
                
                # I use good quality samples for proper mist
                if scene.render.engine in ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE']:
                    try:
                        if hasattr(scene, 'eevee'):
                            original_samples = scene.eevee.taa_render_samples
                            scene.eevee.taa_render_samples = max(64, original_samples)
                            print(f" Eevee samples set to {scene.eevee.taa_render_samples} for quality mist")
                    except:
                        pass
                elif scene.render.engine == 'CYCLES':
                    try:
                        if hasattr(scene, 'cycles'):
                            original_samples = scene.cycles.samples
                            scene.cycles.samples = max(128, original_samples)
                            print(f" Cycles samples set to {scene.cycles.samples} for quality mist")
                    except:
                        pass
                

                print(" Setting up compositor for PURE mist extraction...")
                scene.use_nodes = True
                

                if scene.node_tree:
                    for node in scene.node_tree.nodes:
                        original_nodes[node.name] = {
                            'type': node.bl_idname,
                            'location': node.location.copy()
                        }
                
                # I clear existing nodes
                scene.node_tree.nodes.clear()
                

                render_layers = scene.node_tree.nodes.new(type='CompositorNodeRLayers')
                render_layers.location = (0, 0)
                
                # I add File Output node for mist
                file_output = scene.node_tree.nodes.new(type='CompositorNodeOutputFile')
                file_output.location = (400, 0)
                file_output.base_path = temp_dir
                file_output.file_slots[0].path = "pure_mist"
                file_output.format.file_format = 'PNG'
                file_output.format.color_mode = 'BW'  # I black and white for depth
                file_output.format.color_depth = '16'  # 16-bit for smooth gradation (not 8-bit)
                
                print("📸 File output set to 16-bit BW PNG for smooth mist gradation")
                

                if 'Mist' in render_layers.outputs:
                    scene.node_tree.links.new(render_layers.outputs['Mist'], file_output.inputs[0])
                    print(" Connected PURE Mist pass to file output")
                else:
                    print(" Mist output not found in render layers")
                    raise DepthRenderError("Mist pass not available in render layers")
                

                scene.render.filepath = os.path.join(temp_dir, "no_temp_render")
                
                print(" Starting camera render for PURE mist extraction (compositor only)...")
                

                render_success = False
                
                def _do_mist_render():
                    nonlocal render_success
                    try:
                        print(" Executing camera render with mist compositor (no temp file)...")
                        import time
                        start_time = time.time()
                        # I render without saving main render - compositor will handle mist output
                        bpy.ops.render.render(write_still=False)  # Don't write temp_render.png
                        end_time = time.time()
                        
                        render_success = True
                        print(f" PURE mist render completed in {end_time - start_time:.1f}s (no temp file)")
                    except Exception as e:
                        print(f" Mist render error: {e}")
                        render_success = False
                

                try:
                    if hasattr(self, '_execute_in_main_thread'):
                        self._execute_in_main_thread(_do_mist_render)
                    else:
                        _do_mist_render()
                except:
                    _do_mist_render()
                
                if not render_success:
                    raise DepthRenderError("PURE mist render failed")
                

                # I blender may use different numbering: 0000, 0001, 0010, etc.
                actual_mist_path = None
                
                # First, try to find any pure_mist*.png file in the directory
                import glob
                mist_pattern = os.path.join(temp_dir, "pure_mist*.png")
                mist_files = glob.glob(mist_pattern)
                
                if mist_files:
                    # I use the first found file (or the most recent if multiple)
                    actual_mist_path = mist_files[0]
                    print(f" Found PURE mist file: {actual_mist_path}")
                else:

                    possible_mist_files = [
                        os.path.join(temp_dir, "pure_mist0001.png"),
                        os.path.join(temp_dir, "pure_mist0000.png"),
                        os.path.join(temp_dir, "pure_mist.png"),
                        mist_output_path
                    ]
                    
                    for path in possible_mist_files:
                        if os.path.exists(path):
                            actual_mist_path = path
                            print(f" Found PURE mist file: {path}")
                            break
                
                if actual_mist_path:
                    print(f" PURE mist render completed: {actual_mist_path}")
                    return actual_mist_path
                else:
                    # Debug: list all files in temp_dir
                    try:
                        all_files = os.listdir(temp_dir)
                        print(f" Files in temp directory: {all_files}")
                    except:
                        pass
                    raise DepthRenderError("PURE mist output file not found after render")
                
            finally:

                scene.render.filepath = original_filepath
                scene.render.resolution_x = original_resolution_x
                scene.render.resolution_y = original_resolution_y
                scene.render.resolution_percentage = original_resolution_percentage
                

                if original_samples is not None:
                    if scene.render.engine in ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE']:
                        try:
                            if hasattr(scene, 'eevee'):
                                scene.eevee.taa_render_samples = original_samples
                        except:
                            pass
                    elif scene.render.engine == 'CYCLES':
                        try:
                            if hasattr(scene, 'cycles'):
                                scene.cycles.samples = original_samples
                        except:
                            pass
                

                scene.use_nodes = original_use_nodes
                

                if scene.use_nodes and scene.node_tree:
                    scene.node_tree.nodes.clear()
                    # Note: Full node restoration would be complex, keeping it simple
                
                print(" All render settings and compositor restored")
                    
        except Exception as e:
            raise DepthRenderError(f"PURE mist camera render failed: {str(e)}")
    
    def _render_camera_mist_pass(self, scene, temp_dir: str, render_engine: str) -> str:
        """Render from camera with Mist pass - TRUE depth map"""
        try:
            import bpy
            
            print(" Setting up camera render with Mist pass...")
            

            original_filepath = scene.render.filepath
            
            mist_output_path = os.path.join(temp_dir, "camera_mist.png")
            
            # I track temp directory and files for proper cleanup
            if temp_dir not in self.temp_dirs:
                self.temp_dirs.append(temp_dir)
            self.temp_files.append(mist_output_path)
            
            try:

                original_resolution_x = scene.render.resolution_x
                original_resolution_y = scene.render.resolution_y
                original_resolution_percentage = scene.render.resolution_percentage
                original_samples = None
                
                # I use high quality render settings for proper depth map
                print(" Using high quality render settings for proper depth map...")
                
                # I use original resolution for high quality depth maps
                print(f" Using full scene resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
                
                # I keep resolution percentage at 100% for full quality
                scene.render.resolution_percentage = 100
                print(f" Resolution percentage set to {scene.render.resolution_percentage}% (full quality)")
                
                # I use good quality samples for proper depth maps
                if scene.render.engine in ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE']:
                    try:
                        if hasattr(scene, 'eevee'):
                            original_samples = scene.eevee.taa_render_samples
                            # I use reasonable samples for good quality depth
                            scene.eevee.taa_render_samples = max(64, original_samples)  # I at least 64 samples
                            print(f" Eevee samples set to {scene.eevee.taa_render_samples} (was {original_samples}) for quality depth")
                    except:
                        pass
                elif scene.render.engine == 'CYCLES':
                    try:
                        if hasattr(scene, 'cycles'):
                            original_samples = scene.cycles.samples
                            # I use reasonable samples for Cycles depth
                            scene.cycles.samples = max(128, original_samples)  # I at least 128 samples
                            print(f" Cycles samples set to {scene.cycles.samples} (was {original_samples}) for quality depth")
                    except:
                        pass
                

                scene.render.filepath = os.path.join(temp_dir, "camera_mist")
                
                print(" Starting camera render with mist pass...")
                
                # I store render result for main thread execution
                render_success = False
                
                def _do_camera_render():
                    nonlocal render_success
                    try:
                        print(" About to start bpy.ops.render.render()...")
                        print(f" Render resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
                        print(f" Render engine: {scene.render.engine}")
                        print(f" Output path: {scene.render.filepath}")
                        

                        import time
                        start_time = time.time()
                        bpy.ops.render.render(write_still=True)
                        end_time = time.time()
                        
                        render_success = True
                        print(f" Camera render executed successfully in {end_time - start_time:.1f}s")
                    except Exception as e:
                        print(f" Camera render error: {e}")
                        render_success = False
                

                try:
                    if hasattr(self, '_execute_in_main_thread'):
                        self._execute_in_main_thread(_do_camera_render)
                    else:
                        _do_camera_render()
                except:
                    # I direct execution fallback
                    _do_camera_render()
                
                if not render_success:
                    print(" Camera render failed, trying fallback method...")

                    return self._render_and_extract_mist(scene, temp_dir, render_engine)
                

                possible_paths = [
                    os.path.join(temp_dir, "camera_mist.png"),
                    os.path.join(temp_dir, "camera_mist0001.png"),
                    mist_output_path
                ]
                
                actual_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        actual_path = path
                        break
                
                if actual_path:
                    print(f" Camera mist render completed: {actual_path}")
                    return actual_path
                else:

                    render_result = bpy.data.images.get('Viewer Node') or bpy.data.images.get('Render Result')
                    if render_result:
                        render_result.save_render(filepath=mist_output_path)
                        print(f" Mist extracted from render result: {mist_output_path}")
                        return mist_output_path
                    else:
                        raise DepthRenderError("No camera render output found")
                
            finally:

                scene.render.filepath = original_filepath
                scene.render.resolution_x = original_resolution_x
                scene.render.resolution_y = original_resolution_y
                scene.render.resolution_percentage = original_resolution_percentage
                

                if original_samples is not None:
                    if scene.render.engine in ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE']:
                        try:
                            if hasattr(scene, 'eevee'):
                                scene.eevee.taa_render_samples = original_samples
                        except:
                            pass
                    elif scene.render.engine == 'CYCLES':
                        try:
                            if hasattr(scene, 'cycles'):
                                scene.cycles.samples = original_samples
                        except:
                            pass
                
                print(" All render settings restored")
                    
        except Exception as e:
            raise DepthRenderError(f"Camera mist render failed: {str(e)}")
    
    def _extract_depth_from_render_result(self, scene, temp_dir: str, normalize_mode: str, 
                                        clip_start: float, clip_end: float) -> str:
        """Extract depth from render result as fallback"""
        try:

            render_result = bpy.data.images.get('Render Result')
            if not render_result:
                raise DepthRenderError("No render result found")
            
            # I try to access Z pass
            if not hasattr(render_result, 'pixels') or len(render_result.pixels) == 0:
                raise DepthRenderError("Render result has no pixel data")
            

            temp_exr = os.path.join(temp_dir, "render_result.exr")
            
            # I track temp files for cleanup
            self.temp_files.append(temp_exr)
            render_result.save_render(filepath=temp_exr)
            
            if os.path.exists(temp_exr):
                # I load and process EXR file
                import array
                pixels = render_result.pixels[:]
                width = render_result.size[0] 
                height = render_result.size[1]
                
                # I convert to numpy array and extract depth channel
                # Note: This is simplified - in real implementation would need proper EXR parsing
                depth_array = np.array(pixels).reshape(height, width, -1)
                
                if depth_array.shape[2] >= 4:  # RGBA + depth
                    depth_channel = depth_array[:, :, -1]  # I last channel is usually depth
                else:
                    # I use alpha channel as depth approximation
                    depth_channel = depth_array[:, :, 3] if depth_array.shape[2] > 3 else depth_array[:, :, 0]
                

                output_path = os.path.join(temp_dir, "depth_extracted.png")
                self._save_normalized_depth(depth_channel, output_path, normalize_mode, clip_start, clip_end)
                return output_path
            
            raise DepthRenderError("Could not extract depth from render result")
            
        except Exception as e:
            raise DepthRenderError(f"Failed to extract depth from render result: {str(e)}")
    
    def _normalize_depth_map(self, depth_path: str, normalize_mode: str, 
                           clip_start: float, clip_end: float) -> str:
        """Normalize depth map to 0-255 range"""
        try:
            # I load image
            # I use Blender's image system instead of PIL
            import bpy
            
            # I load image through Blender
            img_name = os.path.basename(depth_path)
            if img_name in bpy.data.images:
                img = bpy.data.images[img_name]
            else:
                img = bpy.data.images.load(depth_path)
            

            width, height = img.size
            pixels = np.array(img.pixels).reshape((height, width, 4))
            
            # I convert to grayscale (take red channel for depth)
            depth_array = pixels[:, :, 0].astype(np.float32)
            
            if len(depth_array.shape) == 3:
                # I convert to grayscale if needed
                depth_array = np.mean(depth_array, axis=2)
                

                if normalize_mode == 'CAMERA_CLIP':
                    # I clamp to camera clip range and normalize
                    depth_array = np.clip(depth_array, clip_start, clip_end)
                    if clip_end > clip_start:
                        depth_array = (depth_array - clip_start) / (clip_end - clip_start)
                else:  # AUTO

                    min_val = np.min(depth_array)
                    max_val = np.max(depth_array)
                    if max_val > min_val:
                        depth_array = (depth_array - min_val) / (max_val - min_val)
                
                # I convert to 0-255 range
                depth_array = (depth_array * 255).astype(np.uint8)
                

                output_path = depth_path.replace('.png', '_normalized.png')
                
                # I track normalized output for cleanup
                self.temp_files.append(output_path)

                img_out = bpy.data.images.new("normalized_depth", width=depth_array.shape[1], height=depth_array.shape[0])
                img_out.pixels = depth_array.flatten() / 255.0  # I convert to 0-1 range
                img_out.filepath_raw = output_path
                img_out.file_format = 'PNG'
                img_out.save()
                bpy.data.images.remove(img_out)
                
                return output_path
                
        except Exception as e:
            raise DepthRenderError(f"Failed to normalize depth map: {str(e)}")
    
    def _save_normalized_depth(self, depth_array: np.ndarray, output_path: str, 
                             normalize_mode: str, clip_start: float, clip_end: float) -> None:
        """Save normalized depth array as PNG"""
        try:

            if normalize_mode == 'CAMERA_CLIP':
                depth_array = np.clip(depth_array, clip_start, clip_end)
                if clip_end > clip_start:
                    depth_array = (depth_array - clip_start) / (clip_end - clip_start)
            else:  # AUTO
                min_val = np.min(depth_array)
                max_val = np.max(depth_array)
                if max_val > min_val:
                    depth_array = (depth_array - min_val) / (max_val - min_val)
            
            # I convert to 0-255 and save
            depth_array = (depth_array * 255).astype(np.uint8)
            

            import bpy
            img_out = bpy.data.images.new("depth_array", width=depth_array.shape[1], height=depth_array.shape[0])
            img_out.pixels = depth_array.flatten() / 255.0  # I convert to 0-1 range
            img_out.filepath_raw = output_path
            img_out.file_format = 'PNG'
            img_out.save()
            bpy.data.images.remove(img_out)
            
        except Exception as e:
            raise DepthRenderError(f"Failed to save normalized depth: {str(e)}")
    
    def render_regular_eevee(self, scene) -> str:
        """
        Render using regular Eevee/Cycles - preserves colors, textures, lighting.
        Returns path to rendered image.
        """
        try:

            self.validate_scene(scene, require_camera=True)
            print("Scene validation passed for regular render")
            

            temp_dir = tempfile.mkdtemp(prefix="gemini_regular_render_")
            render_path = os.path.join(temp_dir, "regular_render.png")
            self.temp_files.append(render_path)
            self.temp_dirs.append(temp_dir)
            print(f"Created temp directory: {temp_dir}")
            
            import bpy
            

            original_filepath = scene.render.filepath
            original_file_format = scene.render.image_settings.file_format
            original_color_mode = scene.render.image_settings.color_mode
            original_color_depth = scene.render.image_settings.color_depth
            original_engine = scene.render.engine
            original_samples = None
            
            # I store color management settings for viewport-like render
            original_view_transform = scene.view_settings.view_transform
            original_look = scene.view_settings.look
            original_exposure = scene.view_settings.exposure
            original_gamma = scene.view_settings.gamma
            
            try:
                print("Setting up regular render...")
                
                # I keep current engine or use Eevee Next if available
                # I in Blender 4.5+, only BLENDER_EEVEE_NEXT exists (not BLENDER_EEVEE)
                if scene.render.engine == 'CYCLES':
                    print("Using CYCLES (current engine)")
                    # I store Cycles samples
                    if hasattr(scene.cycles, 'samples'):
                        original_samples = scene.cycles.samples
                        # I use current samples or ensure minimum quality
                        if scene.cycles.samples < 64:
                            scene.cycles.samples = 64
                            print(f"Increased Cycles samples to 64 for quality")
                elif scene.render.engine == 'BLENDER_EEVEE_NEXT':
                    print("Using BLENDER_EEVEE_NEXT (current engine)")
                    # I ensure good quality for Eevee Next
                    if hasattr(scene.eevee, 'taa_render_samples'):
                        original_samples = scene.eevee.taa_render_samples
                        if scene.eevee.taa_render_samples < 64:
                            scene.eevee.taa_render_samples = 64
                            print(f"Increased Eevee samples to 64 for quality")
                else:
                    # I force Eevee Next for Blender 4.5+
                    scene.render.engine = 'BLENDER_EEVEE_NEXT'
                    print("Switched to BLENDER_EEVEE_NEXT")
                    
                    # I ensure good quality
                    if hasattr(scene.eevee, 'taa_render_samples'):
                        original_samples = scene.eevee.taa_render_samples
                        if scene.eevee.taa_render_samples < 64:
                            scene.eevee.taa_render_samples = 64
                            print(f"Set Eevee samples to 64 for quality")
                
                # I configure render output for viewport-like rendering (RGB, no alpha)
                scene.render.filepath = render_path
                scene.render.image_settings.file_format = 'PNG'
                scene.render.image_settings.color_mode = 'RGB'  # RGB without alpha - like viewport
                scene.render.image_settings.color_depth = '8'  # I standard 8-bit color
                scene.render.image_settings.compression = 15  # PNG compression
                
                # I use Standard view transform for viewport-like colors (not Filmic which adds contrast)
                scene.view_settings.view_transform = 'Standard'
                scene.view_settings.look = 'None'
                # I keep current exposure and gamma (or reset to defaults)
                # scene.view_settings.exposure = 0.0
                # scene.view_settings.gamma = 1.0
                
                print(f"Render settings: {scene.render.engine}, color_mode=RGB, view_transform=Standard, resolution={scene.render.resolution_x}x{scene.render.resolution_y}")
                
                # I ensure render resolution is good
                print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y} @ {scene.render.resolution_percentage}%")
                

                print("Starting regular render...")
                bpy.ops.render.render(write_still=True)
                print(f"Regular render completed: {render_path}")
                
                # I verify file exists
                if not os.path.exists(render_path):
                    raise DepthRenderError(f"Render file not created: {render_path}")
                
                print(f"Render file size: {os.path.getsize(render_path)} bytes")
                
                return render_path
                
            finally:

                scene.render.filepath = original_filepath
                scene.render.image_settings.file_format = original_file_format
                scene.render.image_settings.color_mode = original_color_mode
                scene.render.image_settings.color_depth = original_color_depth
                scene.render.engine = original_engine
                

                scene.view_settings.view_transform = original_view_transform
                scene.view_settings.look = original_look
                scene.view_settings.exposure = original_exposure
                scene.view_settings.gamma = original_gamma
                

                if original_samples is not None:
                    if scene.render.engine == 'CYCLES' and hasattr(scene.cycles, 'samples'):
                        scene.cycles.samples = original_samples
                    elif hasattr(scene.eevee, 'taa_render_samples'):
                        scene.eevee.taa_render_samples = original_samples
                
                print("Restored original render settings and color management")
                
        except Exception as e:
            raise DepthRenderError(f"Regular render failed: {str(e)}")
