import bpy
import math
from bpy.types import Operator
from bpy.props import IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
import os
import shutil
import tempfile
import gpu
import gpu.texture
from gpu_extras.batch import batch_for_shader
import bmesh
import bpy_extras
from bpy_extras import view3d_utils, object_utils
import mathutils
import numpy as np
import threading
from . import gemini_api
from . import depth_utils
from . import threading_utils
from . import projection_utils

# 官方分辨率硬编码像素对齐
SUPPORTED_RESOLUTIONS = {
    "1:1":   (1024, 1024),
    "2:3":   (832, 1248),
    "3:2":   (1248, 832),
    "3:4":   (864, 1184),
    "4:3":   (1184, 864),
    "4:5":   (896, 1152),
    "5:4":   (1152, 896),
    "9:16":  (768, 1344),
    "16:9":  (1344, 768),
    "21:9":  (1536, 672),
}

def snap_to_supported_resolution(res_x, res_y):
    """
    Find the closest ratio from SUPPORTED_RESOLUTIONS and return a resolution 
    that matches that ratio while maintaining the approximate scale.
    Example: 2048x1999 -> Ratio 1.02 -> 1:1 -> 2048x2048
    """
    if res_y == 0: return (1024, 1024)
    target_ratio = res_x / res_y
    best_ratio_key = "1:1"
    min_diff = float('inf')
    
    for ratio_name, (w, h) in SUPPORTED_RESOLUTIONS.items():
        match_ratio = w / h
        diff = abs(target_ratio - match_ratio)
        if diff < min_diff:
            min_diff = diff
            best_ratio_key = ratio_name
            

    off_w, off_h = SUPPORTED_RESOLUTIONS[best_ratio_key]
    

    # Example: 2048x1999 and target 1:1 (1024x1024) -> scale = max(2048/1024, 1999/1024) = 2.0
    scale = max(res_x / off_w, res_y / off_h)
    
    final_w = int(off_w * scale)
    final_h = int(off_h * scale)
    
    # Pixel-perfect alignment: Ensure even numbers for encoders and avoid floating artifacts
    if final_w % 2 != 0: final_w += 1
    if final_h % 2 != 0: final_h += 1
    
    return (final_w, final_h)

# === DEBUG STORAGE ===
_debug_storage = {
    'step': 0,
    'temp_dir': None,
    'init_img_path': None,
    'depth_path': None,
    'mask_repair_data': None,
    'target_objects_data': [],
    'material_name': "Gemini_Projection_Material",
    'image_node_name': "Gemini_Image_Node",
    'original_state': {}
}
# =====================

# =====================

def get_current_view_state(context):
    """Capture current viewport or camera state for restoration"""
    view_state = {}
    try:
        scene = context.scene
        # 1. Check if we are in camera view
        rv3d = context.region_data
        if rv3d:
            view_state['location'] = rv3d.view_location.copy()
            view_state['rotation'] = rv3d.view_rotation.copy()
            view_state['view_distance'] = rv3d.view_distance
            view_state['is_camera_view'] = (rv3d.view_perspective == 'CAMERA')
            

            if scene.camera:
                view_state['lens'] = scene.camera.data.lens
            else:

                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                view_state['lens'] = space.lens
                                break
        
        print(f"View state captured: CamView={view_state.get('is_camera_view')}")
    except Exception as e:
        print(f"Failed to capture view state: {e}")
    return view_state

def restore_view_state(context, history_item):
    """I restore the viewport or camera state from a history item."""
    try:

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        rv3d = space.region_3d
                        if rv3d:

                            rv3d.view_location = history_item.cam_location
                            rv3d.view_rotation = history_item.cam_rotation
                            rv3d.view_distance = history_item.view_distance
                            

                            if history_item.is_camera_view:
                                rv3d.view_perspective = 'CAMERA'
                            else:
                                rv3d.view_perspective = 'PERSP'
                            

                            space.lens = history_item.cam_lens
                            
                            area.tag_redraw()
                            print(f"View state restored from history")
                            return True
        return False
    except Exception as e:
        print(f"Failed to restore view state: {e}")
        return False

class GEMINI_OT_ai_render(Operator):
    """I handle the AI Render operation."""
    bl_idname = "gemini.ai_render"
    bl_label = "AI Render"
    bl_description = "Render depth map and send to AI for photorealistic conversion"
    bl_options = {'REGISTER'}
    
    current_thread = None
    
    def execute(self, context):
        """Execute AI render operation"""
        scene = context.scene
        props = scene.gemini_render
        
        print("Starting AI render operation...")
        
        try:
            if props.is_rendering:
                print("Previous render still active, resetting...")
                props.is_rendering = False
            
            error = self._validate_inputs(context, props)
            if error:
                print(f"Validation error: {error}")
                self.report({'ERROR'}, error)
                props.status_text = f"Error: {error}"
                return {'CANCELLED'}
            
            if self.current_thread and self.current_thread.is_alive():
                print("Stopping previous background thread...")
                self.current_thread.stop()
                self.current_thread.join(timeout=2.0)
            
            api_key = props.api_key.strip() or gemini_api.get_api_key()
            if not api_key:
                error_msg = "No API key found. Set GEMINI_API_KEY environment variable or enter key in UI."
                print(f"{error_msg}")
                self.report({'ERROR'}, error_msg)
                props.status_text = "No API key"
                return {'CANCELLED'}
            
            print(f"API key found, length: {len(api_key)}")
            print(f"User prompt: '{props.prompt[:50]}...'")
            
            depth_renderer = depth_utils.DepthRenderer()
            api_client = gemini_api.GeminiAPI(api_key, model_name=props.model_name)
            
            props.is_rendering = True
            props.status_text = "Starting AI render..."
            
            print("Starting full background thread...")
            
            cam_data = get_current_view_state(context)
            
            self.current_thread = threading_utils.FullRenderThread(
                context=context,
                depth_renderer=depth_renderer,
                api_client=api_client,
                user_prompt=props.prompt,
                cam_data=cam_data
            )
            
            self.current_thread.start()
            
            self.report({'INFO'}, "AI render started in background - check Console for progress")
            print("Background thread started successfully")
            return {'FINISHED'}
            
        except Exception as e:
            error_msg = f"Failed to start AI render: {str(e)}"
            print(f"Exception: {error_msg}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            
            self.report({'ERROR'}, error_msg)
            props.is_rendering = False
            props.status_text = f"Error: {str(e)}"
            return {'CANCELLED'}
    
    def _validate_inputs(self, context, props) -> str:
        """Validate inputs and return error message if invalid"""

        if not props.prompt.strip():
            return "Prompt cannot be empty"
        
        if len(props.prompt.strip()) < 10:
            return "Prompt too short (minimum 10 characters)"
        

        scene = context.scene
        

        visible_objects = [obj for obj in scene.objects if obj.visible_get() and obj.type == 'MESH']
        if len(visible_objects) == 0:
            return "No visible mesh objects found. Add some objects to the scene."
        
        # Note: clip values validation removed since normalize_depth was removed
        

# Note: _validate_projection and bake_projection moved to projection_utils.py

def capture_viewport_to_file(operator, context, scene, props, space_data, region, v_width, v_height, temp_dir, filename, show_wireframe=False, is_depth=False):
    """Refactored helper for viewport/camera capture.
    
    CAMERA PRIORITY LOGIC:
    - If scene has a camera → Use camera render (bpy.ops.render.render)
    - Otherwise → Fallback to viewport capture (bpy.ops.render.opengl)
    
    This ensures pixel-perfect alignment with world_to_camera_view() UV projection.
    """
    target_path = os.path.join(temp_dir, filename)
    original_shading_type = space_data.shading.type
    rv3d = context.region_data
    is_camera_view = (rv3d.view_perspective == 'CAMERA') if rv3d else False
    has_camera = scene.camera is not None and is_camera_view
    
    try:
        if is_depth:
            # I depth capture logic
            depth_renderer = depth_utils.DepthRenderer()
            
            if has_camera:
                # CAMERA DEPTH: Use unified GPU depth render from camera
                print(" Capturing CAMERA DEPTH (Unified GPU)...")
                view_matrix = scene.camera.matrix_world.inverted()
                # I use context.evaluated_depsgraph_get() for calc_matrix_camera in 4.0+
                projection_matrix = scene.camera.calc_matrix_camera(context.evaluated_depsgraph_get(), x=v_width, y=v_height)
                raw_depth_path = depth_renderer.render_depth_gpu(
                    context, 
                    width=v_width, 
                    height=v_height, 
                    view_matrix=view_matrix, 
                    projection_matrix=projection_matrix,
                    invert=True # I keep consistent with previous logic
                )
            else:
                # VIEWPORT DEPTH: Use unified GPU depth render from viewport
                print(" Capturing VIEWPORT DEPTH (Unified GPU)...")
                raw_depth_path = depth_renderer.render_depth_gpu(
                    context, 
                    width=v_width, 
                    height=v_height,
                    invert=True
                )
            
            if raw_depth_path and os.path.exists(raw_depth_path):
                shutil.copy2(raw_depth_path, target_path)
                return True
            return False
        else:
            # Color/Wireframe capture
            original_render_engine = scene.render.engine
            original_filepath = scene.render.filepath
            
            # UNIFIED COLOR CAPTURE LOGIC (High speed via OpenGL)
            # I this matches other projects efficiency for both Camera and Viewport
            print(f" Using HIGH-SPEED CAPTURE for {'wireframe' if show_wireframe else 'color'} ({'Camera' if has_camera else 'Viewport'})")
            

            if show_wireframe:
                space_data.shading.type = 'WIREFRAME'
            else:
                # I ensure EEVEE + RENDERED for high quality but fast color capture
                if scene.render.engine not in ['BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT']:
                    try:
                        scene.render.engine = 'BLENDER_EEVEE_NEXT'
                    except:
                        scene.render.engine = 'BLENDER_EEVEE'
                space_data.shading.type = 'RENDERED'
            

            # (Resolution already snapped and set in scene.render by caller)
            scene.render.filepath = target_path
            scene.render.image_settings.file_format = 'PNG'
            

            # view_context=True ensures it respects the current camera/viewport state
            bpy.ops.render.opengl(write_still=True, view_context=True)
            

            space_data.shading.type = original_shading_type
            scene.render.engine = original_render_engine
            scene.render.filepath = original_filepath
            
            print(f" Capture saved: {target_path}")
            return True
            
    except Exception as e:
        print(f" Capture helper failed: {e}")
        import traceback
        traceback.print_exc()
        if 'original_shading_type' in locals():
            space_data.shading.type = original_shading_type
        if 'original_render_engine' in locals():
            scene.render.engine = original_render_engine
        return False

class GEMINI_OT_texture_projection(Operator):
    """Deeply integrated AI Texture Projection"""
    bl_idname = "gemini.texture_projection"
    bl_label = "AI Texture Projection"
    bl_description = "Capture viewport, project to mesh, and generate texture using AI"
    bl_options = {'REGISTER', 'UNDO'}

    current_thread = None

    @classmethod
    def poll(cls, context):
        try:
            projection_utils.validate_projection(context)
            return True
        except:
            return False

    def execute(self, context):
        import bmesh # I fix UnboundLocalError
        scene = context.scene
        props = scene.gemini_render
        
        # 1. Validation
        try:
            projection_utils.validate_projection(context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}



        original_active_uvs = {}
        for obj in context.selected_objects:
            if obj.type == 'MESH' and obj.data.uv_layers.active:
                original_active_uvs[obj.name] = obj.data.uv_layers.active.name
                print(f"Target UV for {obj.name}: {original_active_uvs[obj.name]}")


        viewport_area = next((a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if not viewport_area:
            self.report({'ERROR'}, "No 3D Viewport found")
            return {'CANCELLED'}
        region = viewport_area.regions[-1]
        space_data = next(s for s in viewport_area.spaces if s.type == 'VIEW_3D')
        
        # ======================================================================
        # CAMERA PRIORITY: If camera exists, use camera render + camera resolution
        # I this ensures pixel-perfect UV alignment (world_to_camera_view matches render)
        # ======================================================================
        rv3d = context.region_data
        is_camera_view = (rv3d.view_perspective == 'CAMERA') if rv3d else False
        has_camera = scene.camera is not None and is_camera_view
        use_camera_render = has_camera  # I use camera render when camera view is active
        
        if use_camera_render:
            # ======================================================================
            # CAMERA RESOLUTION ENFORCEMENT:
            # I force camera resolution to nearest SUPPORTED_RESOLUTIONS entry
            # ======================================================================
            best_res = snap_to_supported_resolution(scene.render.resolution_x, scene.render.resolution_y)
            print(f" Resolution Snap: {scene.render.resolution_x}x{scene.render.resolution_y} -> {best_res[0]}x{best_res[1]}")
            
            # I application of resolution involves updating 'capture_width/height'
            # and later forcing scene.render.resolution_x/y in the capture block.
            capture_width = best_res[0]
            capture_height = best_res[1]
            print(f" CAMERA MODE: Using snapped resolution {capture_width}x{capture_height}")
            print(f" Camera type: {'ORTHO' if scene.camera.data.type == 'ORTHO' else 'PERSP'}")
        else:
            # Fallback: Use viewport dimensions
            capture_width, capture_height = region.width, region.height
            print(f" VIEWPORT MODE: Using viewport resolution {capture_width}x{capture_height}")
        
        # UV projection dimensions (CRITICAL: must match capture resolution for pixel alignment)
        v_width, v_height = capture_width, capture_height


        render_filepath = scene.render.filepath
        render_format = scene.render.image_settings.file_format
        original_show_overlays = space_data.overlay.show_overlays
        original_shading_type = space_data.shading.type
        
        # I resolution management for consistent capture
        original_res_x = scene.render.resolution_x
        original_res_y = scene.render.resolution_y
        original_res_pct = scene.render.resolution_percentage
        
        # 2. Logic Branching based on Projection Source
        if props.projection_source == 'IMAGE':
            source_image_override = props.projection_image
            if not source_image_override:
                self.report({'ERROR'}, "No image selected for projection")
                return {'CANCELLED'}
            
            source_path = "" # I not used for 'IMAGE'
            sim_path = ""
            mask_repair_data = None # I direct image mode doesn't support mask repair for now
            print(f"🖼 Direct Image Mode active: Using '{source_image_override.name}'")
        else:
            source_image_override = None
            # I temporary directory for workspace
            temp_dir = tempfile.mkdtemp(prefix="gemini_proj_")
            source_path = os.path.join(temp_dir, "captured_input.png")

            # I configure for capture
            space_data.overlay.show_overlays = False
            scene.render.image_settings.file_format = 'PNG'
            

            scene.render.resolution_x = capture_width
            scene.render.resolution_y = capture_height
            scene.render.resolution_percentage = 100
            
            # I mask Repair Mode: Create temp mask mesh WITHOUT affecting original object state
            mask_repair_data = None
            if props.mask_repair_mode:
                try:
                    print("Mask Repair Mode: Creating mask overlay mesh...")
                    

                    obj = context.active_object
                    if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
                        raise Exception("Active object must be a mesh in Edit Mode")
                    
                    obj_name = obj.name
                    print(f"Mask Repair Mode: Active object = '{obj_name}'")
                    

                    bm = bmesh.from_edit_mesh(obj.data)
                    
                    # I count and identify selected faces
                    selected_faces = [f for f in bm.faces if f.select]
                    print(f"Mask Repair Mode: {len(selected_faces)} faces selected")
                    
                    if not selected_faces:
                        raise Exception("No faces selected")
                    
                    # FACE-AWARE TEXTURE TARGETING: Use material of first selected face
                    target_mat_index = selected_faces[0].material_index
                    original_texture = None
                    
                    if target_mat_index < len(obj.material_slots):
                        slot = obj.material_slots[target_mat_index]
                        if slot.material and slot.material.use_nodes:
                            for node in slot.material.node_tree.nodes:
                                if node.type == 'TEX_IMAGE' and node.image:
                                    original_texture = node.image
                                    break
                    

                    if not original_texture:
                        print(" Targeted slot has no image, searching all slots...")
                        for slot in obj.material_slots:
                            if slot.material and slot.material.use_nodes:
                                for node in slot.material.node_tree.nodes:
                                    if node.type == 'TEX_IMAGE' and node.image:
                                        original_texture = node.image
                                        break
                            if original_texture: break
                    
                    if not selected_faces:
                        raise Exception("No faces selected")
                    

                    bm_copy = bm.copy()
                    
                    # I delete unselected faces from the copy
                    faces_to_delete = [f for f in bm_copy.faces if not f.select]
                    bmesh.ops.delete(bm_copy, geom=faces_to_delete, context='FACES')
                    

                    temp_mesh = bpy.data.meshes.new(f"{obj_name}_MaskTemp_Data")
                    bm_copy.to_mesh(temp_mesh)
                    bm_copy.free()
                    

                    temp_obj = bpy.data.objects.new(f"{obj_name}_MaskTemp", temp_mesh)
                    temp_obj.matrix_world = obj.matrix_world.copy()
                    
                    # I link to scene
                    context.collection.objects.link(temp_obj)
                    
                    # ANTI-ZFIGHTING: Add Solidify modifier with offset=0
                    # I this expands the mesh symmetrically in both directions from the original surface,
                    # which is immune to normal direction issues and prevents z-fighting reliably.
                    solidify_mod = temp_obj.modifiers.new(name="Gemini_Solidify", type='SOLIDIFY')
                    solidify_mod.thickness = 0.002  # 2mm total thickness (1mm each direction)
                    solidify_mod.offset = 0  # I expand equally in both directions
                    solidify_mod.use_rim = False  # I no rim faces needed
                    

                    mask_mat_name = "Gemini_Mask_Material_Temp"
                    mask_mat = bpy.data.materials.get(mask_mat_name)
                    if mask_mat:
                        bpy.data.materials.remove(mask_mat)
                    mask_mat = bpy.data.materials.new(name=mask_mat_name)
                    mask_mat.use_nodes = True
                    mask_nodes = mask_mat.node_tree.nodes
                    mask_links = mask_mat.node_tree.links
                    mask_nodes.clear()
                    
                    mask_output = mask_nodes.new("ShaderNodeOutputMaterial")
                    mask_output.location = (300, 0)
                    mask_emit = mask_nodes.new("ShaderNodeEmission")
                    mask_emit.location = (100, 0)
                    mask_emit.inputs['Color'].default_value = (props.mask_color[0], props.mask_color[1], props.mask_color[2], 1.0)
                    mask_emit.inputs['Strength'].default_value = 1.0
                    mask_links.new(mask_emit.outputs['Emission'], mask_output.inputs['Surface'])
                    
                    # I assign material to temp mesh
                    temp_obj.data.materials.clear()
                    temp_obj.data.materials.append(mask_mat)
                    
                    # I store mask repair data
                    mask_repair_data = {
                        'temp_objects': [temp_obj.name],
                        'temp_materials': [mask_mat_name],
                        'original_textures': {obj_name: original_texture.name},
                        'original_object_names': [obj_name],
                    }
                    
                    print(f"Mask Repair Mode: Created temp mesh '{temp_obj.name}' with mask material")
                    print("Mask Repair Mode: Original object state UNCHANGED")
                    
                except Exception as mask_error:
                    print(f" Mask Repair Mode setup error: {mask_error}")
                    import traceback
                    traceback.print_exc()
                    # CRITICAL CHANGE: Abort immediately to prevent falling back to Standard Mode (which would destroy the texture)
                    self.report({'ERROR'}, f"Mask Repair Setup Failed: {mask_error}. Aborting to protect texture.")
                    return {'CANCELLED'}
            
            # PRE-FLIGHT CHECK
            if props.mask_repair_mode and not mask_repair_data:
                 self.report({'ERROR'}, "Mask Repair Mode active but no mask data generated. Aborting.")
                 return {'CANCELLED'}

            try:
                # A. Capture Color (Native Resolution)
                # I forced Resolution Application (for Camera Render)
                if use_camera_render:
                    scene.render.resolution_x = capture_width
                    scene.render.resolution_y = capture_height
                    scene.render.resolution_percentage = 100
                
                print(f"Capturing viewport color ({capture_width}x{capture_height})...")
                props.status_text = "📸 Capturing viewport..."
                
                # D. Determine Capture Requirements & Execute
                # Anti-redundancy with meaningful debug:
                # 1. Capture the INTENDED AI SOURCE
                source_filename = "captured_source.png"
                source_path = os.path.join(temp_dir, source_filename)
                
                if props.projection_source == 'COLOR':
                    print("🎨 Capturing Color Viewport (Intended AI Source)")
                    success = capture_viewport_to_file(self, context, scene, props, space_data, region, v_width, v_height, temp_dir, source_filename)
                else: # DEPTH
                    print(" Capturing Depth Map (Intended AI Source)")
                    success = capture_viewport_to_file(self, context, scene, props, space_data, region, v_width, v_height, temp_dir, source_filename, is_depth=True)

                if not success:
                    raise Exception("Primary viewport capture failed")

                # 2. Capture SIMULATION GRID if needed
                sim_path = ""
                if props.grid_simulation:
                    sim_filename = "grid_simulation.png"
                    sim_path = os.path.join(temp_dir, sim_filename)
                    print("⚒ Capturing Wireframe Grid (Simulation Output)")
                    if not capture_viewport_to_file(self, context, scene, props, space_data, region, v_width, v_height, temp_dir, sim_filename, show_wireframe=True):
                        raise Exception("Grid simulation capture failed")

                # E. Local Debug Output Setup
                if props.debug_mode:
                    blend_path = bpy.data.filepath
                    base_debug_dir = os.path.join(os.path.dirname(blend_path), "textures") if blend_path else os.path.join(temp_dir, "textures")
                    if not os.path.exists(base_debug_dir): os.makedirs(base_debug_dir)
                    

                    shutil.copy2(source_path, os.path.join(base_debug_dir, "input.png"))
                    print(f"🐞 Debug input (AI Intention) saved: {os.path.join(base_debug_dir, 'input.png')}")

            except Exception as capture_error:
                print(f" Capture Error: {capture_error}")
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, f"Capture failed: {str(capture_error)}")
                return {'CANCELLED'}

            finally:

                scene.render.filepath = render_filepath
                scene.render.image_settings.file_format = render_format
                space_data.overlay.show_overlays = original_show_overlays
                space_data.shading.type = original_shading_type
                

                scene.render.resolution_x = original_res_x
                scene.render.resolution_y = original_res_y
                scene.render.resolution_percentage = original_res_pct

        # 3. Setup Projection Logic (Remapping UVs)
        target_objects_data = []
        
        # I use a consistent material name but create fresh if needed
        mat_name = "Gemini_Projection_Material"
        material = bpy.data.materials.get(mat_name)
        if not material:
            material = bpy.data.materials.new(name=mat_name)
            material.use_nodes = True
        
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # CLEAR ALL DEFAULT NODES (Especially the Principled BSDF)
        nodes.clear()
        
        # 1. Create Output Node
        output_node = nodes.new("ShaderNodeOutputMaterial")
        output_node.location = (400, 0)
        
        # 2. Create Emission Node (For shadeless color)
        emit_node = nodes.new("ShaderNodeEmission")
        emit_node.location = (200, 0)
        emit_node.inputs['Strength'].default_value = 1.0
        
        # 3. Create Image Texture Node
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.name = "Gemini_Image_Node"
        image_node.location = (0, 0)
        
        # 4. Create UV Map Node
        uv_map_node = nodes.new("ShaderNodeUVMap")
        uv_map_node.name = "Gemini_UV_Map"
        uv_map_node.uv_map = "Projected UVs"
        uv_map_node.location = (-200, 0)
        
        # I link them up: UV -> Image -> Emission -> Output
        links.new(uv_map_node.outputs['UV'], image_node.inputs['Vector'])
        links.new(image_node.outputs['Color'], emit_node.inputs['Color'])
        links.new(emit_node.outputs['Emission'], output_node.inputs['Surface'])
        
        import bmesh
        from bpy_extras import view3d_utils, object_utils

        processed_count = 0
        # CAMERA PRIORITY: Use camera projection when camera exists (matches camera render used for capture)
        # I this ensures pixel-perfect alignment between capture and UV projection
        # NOTE: use_camera_render is set earlier based on has_camera
        if use_camera_render:
            print(f" UV Projection: Using CAMERA coordinates (matches camera render)")
        else:
            print(f" UV Projection: Using VIEWPORT coordinates (fallback)")
        for obj in context.selected_objects:
            if obj.type != 'MESH': continue
            
            # ABSOLUTE ZERO-LEAK GUARD: If in Repair Mode, strictly skip ORIGINAL objects
            # in the standard projection setup path. Only the temp objects should receive
            # the projection material.
            if props.mask_repair_mode and mask_repair_data:
                if obj.name in mask_repair_data['original_object_names']:
                    print(f"🛡 Repair Mode Isolation: Skipping original object '{obj.name}' setup")
                    
                    # Instead, we setup the corresponding temp object
                    try:
                        idx = mask_repair_data['original_object_names'].index(obj.name)
                        temp_obj_name = mask_repair_data['temp_objects'][idx]
                        temp_obj = bpy.data.objects.get(temp_obj_name)
                        
                        if temp_obj:
                            print(f"Redirecting projection to TEMP object: '{temp_obj.name}'")
                            
                            # 1. Apply Projection Material (it was already cleared/has mask mat, but let's be sure)
                            temp_obj.data.materials.clear()
                            temp_obj.data.materials.append(material)
                            
                            # 2. UV Projection on Temp Obj
                            bpy.ops.object.mode_set(mode='OBJECT')
                            bpy.ops.object.select_all(action='DESELECT')
                            temp_obj.select_set(True)
                            context.view_layer.objects.active = temp_obj
                            bpy.ops.object.mode_set(mode='EDIT')
                            
                            bm_temp = bmesh.from_edit_mesh(temp_obj.data)
                            uv_layer_name = "Projected UVs"
                            uv_layer = bm_temp.loops.layers.uv.get(uv_layer_name) or bm_temp.loops.layers.uv.new(uv_layer_name)
                            
                            for face in bm_temp.faces:
                                face.material_index = 0
                                for loop in face.loops:
                                    world_co = temp_obj.matrix_world @ loop.vert.co
                                    if use_camera_render:
                                        screen_co = object_utils.world_to_camera_view(scene, scene.camera, world_co)
                                        uv = (screen_co[0], screen_co[1])
                                    else:
                                        screen_co = view3d_utils.location_3d_to_region_2d(region, space_data.region_3d, world_co)
                                        uv = (screen_co[0]/v_width, screen_co[1]/v_height) if screen_co else (0,0)
                                    loop[uv_layer].uv = uv
                                    
                            bmesh.update_edit_mesh(temp_obj.data)
                            bm_copy = bm_temp.copy()
                            
                            # I add to data (Source=Temp, Target=Original for baking reference)
                            dest_uv_name = original_active_uvs.get(obj.name, "")
                            if not dest_uv_name and obj.data.uv_layers.active:
                                dest_uv_name = obj.data.uv_layers.active.name

                            target_objects_data.append({
                                'object_name': temp_obj.name, 
                                'original_object_name': obj.name,
                                'bm_copy': bm_copy,
                                'src_uv_name': uv_layer_name,
                                'dest_uv_name': dest_uv_name
                            })
                            processed_count += 1
                            bpy.ops.object.mode_set(mode='OBJECT')
                    except Exception as e:
                        print(f" Repair Mode Setup Error: {e}")
                    
                    continue # IMPORTANT: SKIP standard processing for the original object
                
            # === END MASK REPAIR MODE PATH ===

            # CRITICAL: Skip temp mask objects - they should NEVER be processed for projection/baking in normal flow
            if '_MaskTemp' in obj.name:
                continue
            
            # I robust Material Slot Assignment
            found_slot = -1
            for i, slot in enumerate(obj.material_slots):
                if slot.material == material:
                    found_slot = i
                    break
            
            if found_slot == -1:
                # I add new slot
                obj.data.materials.append(material)
                found_slot = len(obj.data.materials) - 1
            

            bm = None
            if obj.mode != 'EDIT':
                # I need to switch this object to edit mode
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                print(f"Switched {obj.name} to Edit Mode")
            
            if obj.mode == 'EDIT':
                bm = bmesh.from_edit_mesh(obj.data)
            
            if not bm:
                print(f" Skipping {obj.name} - could not access Edit Mode mesh")
                continue
                

            uv_layer_name = "Projected UVs"
            uv_layer = bm.loops.layers.uv.get(uv_layer_name) or bm.loops.layers.uv.new(uv_layer_name)
            
            # I apply to selected faces
            face_updated = False
            for face in bm.faces:
                if face.select:
                    face.material_index = found_slot
                    face_updated = True
                    for loop in face.loops:
                        world_co = obj.matrix_world @ loop.vert.co
                        if use_camera_render:
                            screen_co = object_utils.world_to_camera_view(scene, scene.camera, world_co)
                            uv = (screen_co[0], screen_co[1])
                        else:
                            screen_co = view3d_utils.location_3d_to_region_2d(region, space_data.region_3d, world_co)
                            if screen_co:
                                uv = (screen_co[0] / v_width, screen_co[1] / v_height)
                            else:
                                uv = (0, 0)
                        loop[uv_layer].uv = uv
            
            if face_updated:
                bmesh.update_edit_mesh(obj.data)
                processed_count += 1
                
                if props.projection_bake:
                    bm_copy = bm.copy()
                    bmesh.ops.delete(bm_copy, geom=[f for f in bm_copy.faces if not f.select], context='FACES')
                    target_objects_data.append({
                        'object_name': obj.name,
                        'bm_copy': bm_copy,
                        'src_uv_name': uv_layer_name,
                        'dest_uv_name': original_active_uvs.get(obj.name, obj.data.uv_layers.active.name)
                    })

        if processed_count == 0:
            self.report({'WARNING'}, "No faces were projected. Ensure meshes are in Edit Mode with faces selected.")
            return {'CANCELLED'}


        cam_data = get_current_view_state(context)

        # 4. Start AI Thread
        api_key = props.api_key.strip() or gemini_api.get_api_key()
        api_client = gemini_api.GeminiAPI(api_key, model_name=props.model_name)
        
        print(f" Starting Projection Thread (Accurate Sim Debug) for {processed_count} objects...")
        self.current_thread = threading_utils.ProjectionRenderThread(
            context=context,
            api_client=api_client,
            user_prompt=props.prompt,
            source_path=source_path, # I intended AI Source
            sim_path=sim_path,       # I local grid simulation (if any)
            target_objects_data=target_objects_data,
            image_node_name=image_node.name,
            material_name=material.name,
            do_bake=props.projection_bake,
            bypass_api=props.grid_simulation,
            mask_repair_data=mask_repair_data,
            projection_source=props.projection_source,
            debug_mode=props.debug_mode,
            source_image_override=source_image_override,
            cam_data=cam_data
        )
        self.current_thread.start()
        
        self.report({'INFO'}, f"AI Projection started for {processed_count} objects...")
        return {'FINISHED'}


class GEMINI_OT_stop_render(Operator):
    """Stop current AI render operation"""
    bl_idname = "gemini.stop_render"
    bl_label = "Stop Render"
    bl_description = "Stop the current AI render operation"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        """Execute stop render operation"""
        try:
            print(" Stop render requested...")
            props = context.scene.gemini_render
            
            # I stop background thread if running
            if hasattr(GEMINI_OT_ai_render, 'current_thread') and GEMINI_OT_ai_render.current_thread:
                if GEMINI_OT_ai_render.current_thread.is_alive():
                    print(" Stopping active background thread...")
                    GEMINI_OT_ai_render.current_thread.stop()
                    GEMINI_OT_ai_render.current_thread.join(timeout=3.0)
                    print(" Background thread stopped")
                else:
                    print(" Background thread not active, just resetting UI...")
            else:
                print(" No background thread found, just resetting UI...")
            
            # I always reset UI state
            props.is_rendering = False
            props.status_text = " Cancelled by user"
            
            self.report({'INFO'}, "AI render stopped - UI reset")
            print(" UI state reset to normal")
            return {'FINISHED'}
            
        except Exception as e:
            print(f" Error stopping render: {str(e)}")
            # I force reset UI even if there was an error
            try:
                props = context.scene.gemini_render
                props.is_rendering = False
                props.status_text = " Error stopping - reset manually"
            except:
                pass
            self.report({'ERROR'}, f"Failed to stop render: {str(e)}")
            return {'CANCELLED'}

# I additional utility operators

class GEMINI_OT_open_api_key_url(Operator):
    """Open Google AI Studio URL to get API key"""
    bl_idname = "gemini.open_api_key_url"
    bl_label = "Get API Key"
    bl_description = "Open Google AI Studio to get your Gemini API key"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        """Open API key URL"""
        import webbrowser
        webbrowser.open("https://aistudio.google.com/app/apikey")
        self.report({'INFO'}, "Opened Google AI Studio in browser")
        return {'FINISHED'}

class GEMINI_OT_validate_api_key(Operator):
    """Validate API key"""
    bl_idname = "gemini.validate_api_key"
    bl_label = "Test API Key"
    bl_description = "Test if the API key is valid"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        """Validate API key"""
        try:
            props = context.scene.gemini_render
            api_key = props.api_key.strip() or gemini_api.get_api_key()
            
            if not api_key:
                self.report({'ERROR'}, "No API key provided")
                return {'CANCELLED'}
            
            # TODO: Add actual API validation call

            if api_key.startswith('AIza') and len(api_key) > 35:
                self.report({'INFO'}, "API key format looks valid")
            else:
                self.report({'WARNING'}, "API key format seems invalid (should start with 'AIza')")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to validate API key: {str(e)}")
            return {'CANCELLED'}

class GEMINI_OT_reset_state(Operator):
    """Reset UI state in case of stuck rendering"""
    bl_idname = "gemini.reset_state"
    bl_label = "Reset UI State"
    bl_description = "Force reset render state (use if UI is stuck)"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        """Reset UI state"""
        try:
            print(" Force resetting UI state...")
            props = context.scene.gemini_render
            
            # I force reset all state
            props.is_rendering = False
            props.status_text = " UI state reset"
            
            # I try to stop any running background threads
            if hasattr(GEMINI_OT_ai_render, 'current_thread') and GEMINI_OT_ai_render.current_thread:
                try:
                    GEMINI_OT_ai_render.current_thread.stop()
                    GEMINI_OT_ai_render.current_thread = None
                    print(" Force stopped background thread")
                except:
                    pass
            
            # I redraw UI
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            
            self.report({'INFO'}, "UI state reset - render button should work now")
            print(" UI state force reset completed")
            return {'FINISHED'}
            
        except Exception as e:
            print(f" Error resetting state: {str(e)}")
            self.report({'ERROR'}, f"Failed to reset state: {str(e)}")
            return {'CANCELLED'}

class GEMINI_OT_open_console(Operator):
    """Open Blender console to view logs"""
    bl_idname = "gemini.open_console"
    bl_label = "Open Console"
    bl_description = "Open Blender console to view detailed logs"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        """Open console"""
        try:
            # I toggle console
            bpy.ops.wm.console_toggle()
            self.report({'INFO'}, "Console toggled - look for messages")
            print("📋 Console opened - all logs are prefixed with [GEMINI]")
            print("📋 Look for messages with  emojis for render progress")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'WARNING'}, "Console toggle not available on this platform")
            print(" Console toggle not available - check System Console in Window menu")
            return {'CANCELLED'}


class GEMINI_OT_load_history(Operator):
    """Load render result from history"""
    bl_idname = "gemini.load_history"
    bl_label = "Load History Render"
    bl_description = "Load this render result from history"
    bl_options = {'REGISTER'}

    history_index: IntProperty(
        name="History Index",
        description="Index of history item to load",
        default=0
    )

    def execute(self, context):
        try:
            props = context.scene.gemini_render
            
            print(f"🖼 GEMINI_OT_load_history called with index: {self.history_index}")
            
            if self.history_index < 0 or self.history_index >= len(props.render_history):
                print(f" Invalid history index: {self.history_index}, history length: {len(props.render_history)}")
                self.report({'ERROR'}, "Invalid history index")
                return {'CANCELLED'}
            
            history_item = props.render_history[self.history_index]
            print(f"🖼 Load history item: {history_item.prompt[:30]}...")
            
            # I try to find the AI_Result image in Blender
            image = None
            if history_item.image_name and history_item.image_name in bpy.data.images:
                try:
                    candidate = bpy.data.images[history_item.image_name]
                    if candidate.has_data:
                        image = candidate
                        print(f" Found AI_Result image for loading: {candidate.name}")
                    else:
                        print(f" AI_Result image has no data: {candidate.name}")
                except Exception as e:
                    print(f" Error finding AI_Result image: {e}")
                    pass
            
            if image:
                
                # I load as render result
                from .threading_utils import execute_in_main_thread
                
                def _load_from_history():
                    try:
                        # I use the original AI_Result image directly - no duplication needed
                        print(f"🖼 Using original AI_Result image for loading: {image.name}")
                        duplicate_image = image  # I use original, don't create copy
                        
                        # I open in new window or existing Image Editor (don't replace Render Result)
                        try:
                            # I method 1: Create new window with Image Editor
                            bpy.ops.wm.window_new()
                            new_window = bpy.context.window_manager.windows[-1]
                            
                            for area in new_window.screen.areas:
                                if area.type != 'IMAGE_EDITOR':
                                    area.type = 'IMAGE_EDITOR'
                                    for space in area.spaces:
                                        if space.type == 'IMAGE_EDITOR':
                                            space.image = duplicate_image
                                            area.tag_redraw()
                                            print(f"🖼 Opened AI_Result in new window: {duplicate_image.name}")
                                            return
                                    break
                                    
                        except Exception as e:
                            print(f" Could not create new window: {e}")
                            
                            # I method 2: Use existing Image Editor (but don't replace Render Result)
                            for area in context.screen.areas:
                                if area.type == 'IMAGE_EDITOR':
                                    for space in area.spaces:
                                        if space.type == 'IMAGE_EDITOR':
                                            space.image = duplicate_image
                                            area.tag_redraw()
                                            print(f"🖼 Set AI_Result in existing Image Editor: {duplicate_image.name}")
                                            return
                        
                        print(f" AI_Result image opened: {duplicate_image.name}")
                    
                    except Exception as e:
                        print(f" Error loading history: {e}")
                
                execute_in_main_thread(_load_from_history)
                
                self.report({'INFO'}, f"Opened: {image.name}")
                return {'FINISHED'}
            else:
                self.report({'WARNING'}, "History image not found or has no valid data")
                return {'CANCELLED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load history: {str(e)}")
            return {'CANCELLED'}

class GEMINI_OT_open_history_image(Operator):
    """Open history image in a new window for viewing"""
    bl_idname = "gemini.open_history_image"
    bl_label = "View History Photo"
    bl_description = "Open this render result in the image editor"
    bl_options = {'REGISTER'}

    history_index: IntProperty(
        name="History Index",
        description="Index of history item to view",
        default=0
    )

    def execute(self, context):
        # reuse logic from load_history but focused on just opening the image
        op = bpy.ops.gemini.load_history(history_index=self.history_index)
        return {'FINISHED'}

class GEMINI_OT_delete_history(Operator):
    """Delete render from history"""
    bl_idname = "gemini.delete_history"
    bl_label = "Delete History Render"
    bl_description = "Delete this render from history"
    bl_options = {'REGISTER'}

    history_index: IntProperty(
        name="History Index",
        description="Index of history item to delete",
        default=0
    )

    def execute(self, context):
        try:
            props = context.scene.gemini_render
            
            if self.history_index < 0 or self.history_index >= len(props.render_history):
                self.report({'ERROR'}, "Invalid history index")
                return {'CANCELLED'}
            
            history_item = props.render_history[self.history_index]
            
            # I remove main AI_Result image from Blender if it exists
            if history_item.image_name in bpy.data.images:
                image = bpy.data.images[history_item.image_name]
                bpy.data.images.remove(image)
                print(f"🗑 Removed AI_Result image: {history_item.image_name}")
            
            # I remove style reference thumbnail if it exists
            if history_item.style_reference_thumbnail and history_item.style_reference_thumbnail in bpy.data.images:
                style_thumbnail = bpy.data.images[history_item.style_reference_thumbnail]
                bpy.data.images.remove(style_thumbnail)
                print(f"🗑 Removed style thumbnail: {history_item.style_reference_thumbnail}")
            
            # NO STYLE THUMBNAIL REMOVAL - they don't exist anymore
            
            # I remove from history
            props.render_history.remove(self.history_index)
            
            self.report({'INFO'}, f"Deleted: {history_item.prompt[:30]}...")
            print(f"🗑 History item deleted: {history_item.prompt[:50]}...")
            return {'FINISHED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Failed to delete history: {str(e)}")
            return {'CANCELLED'}


class GEMINI_OT_use_history_prompt(Operator):
    """Use prompt from render history"""
    bl_idname = "gemini.use_history_prompt"
    bl_label = "Use History Prompt"
    bl_description = "Copy prompt from this history item to current prompt field"
    bl_options = {'REGISTER'}

    history_index: IntProperty(
        name="History Index",
        description="Index of history item to use prompt from",
        default=0
    )

    def execute(self, context):
        try:
            props = context.scene.gemini_render
            
            if self.history_index < 0 or self.history_index >= len(props.render_history):
                self.report({'ERROR'}, "Invalid history index")
                return {'CANCELLED'}
            
            history_item = props.render_history[self.history_index]
            
            # I copy prompt to current prompt field
            props.prompt = history_item.prompt
            
            self.report({'INFO'}, f"Prompt copied: {history_item.prompt[:50]}...")
            print(f"📝 Prompt copied from history: {history_item.prompt[:50]}...")
            return {'FINISHED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Failed to use history prompt: {str(e)}")
            return {'CANCELLED'}


class GEMINI_OT_use_history_style(Operator):
    """Use style reference from render history"""
    bl_idname = "gemini.use_history_style"
    bl_label = "Use History Style"
    bl_description = "Copy style reference from this history item to current style reference"
    bl_options = {'REGISTER'}

    history_index: IntProperty(
        name="History Index", 
        description="Index of history item to use style from",
        default=0
    )

    def execute(self, context):
        try:
            props = context.scene.gemini_render
            
            if self.history_index < 0 or self.history_index >= len(props.render_history):
                self.report({'ERROR'}, "Invalid history index")
                return {'CANCELLED'}
            
            history_item = props.render_history[self.history_index]
            
            if not history_item.style_reference_used:
                self.report({'WARNING'}, "This render didn't use a style reference")
                return {'CANCELLED'}
            

            style_image = None
            if history_item.style_reference_name in bpy.data.images:
                style_image = bpy.data.images[history_item.style_reference_name]
            elif history_item.style_reference_thumbnail in bpy.data.images:

                style_image = bpy.data.images[history_item.style_reference_thumbnail]
            
            if not style_image:
                self.report({'ERROR'}, "Style reference image not found")
                return {'CANCELLED'}
            

            props.style_reference_image = style_image
            props.use_style_reference = True
            
            self.report({'INFO'}, f"Style reference set: {style_image.name}")
            print(f"🎨 Style reference copied from history: {style_image.name}")
            return {'FINISHED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Failed to use history style: {str(e)}")
            return {'CANCELLED'}


class GEMINI_OT_history_context_menu(Operator):
    """Show context menu for render history item"""
    bl_idname = "gemini.history_context_menu"
    bl_label = "History Context Menu"
    bl_description = "Show options for this render history item"
    bl_options = {'REGISTER'}

    history_index: IntProperty(
        name="History Index",
        description="Index of history item",
        default=0
    )

    def execute(self, context):
        # I show the compact context menu
        def draw_menu(menu_self, menu_context):
            layout = menu_self.layout
            props = menu_context.scene.gemini_render
            history_idx = menu_context.window_manager.history_menu_index
            
            if history_idx >= len(props.render_history):
                layout.label(text=" Invalid history item")
                return
            
            item = props.render_history[history_idx]
            
            # I compact menu - just the essential actions
            
            # I copy settings
            prompt_op = layout.operator("gemini.use_history_prompt", text="Use Prompt", icon='TEXT')
            prompt_op.history_index = history_idx
            
            # I use style options if available
            if item.style_reference_used:
                style_op = layout.operator("gemini.use_history_style", text="Use Style", icon='IMAGE_DATA')  
                style_op.history_index = history_idx
                
                both_op = layout.operator("gemini.use_history_both", text="Use Both", icon='GHOST_ENABLED')
                both_op.history_index = history_idx
            
            # I use as Projection Source
            layout.separator()
            set_source_op = layout.operator("gemini.set_projection_source", text="Use as Projection Source", icon='ADD')
            set_source_op.history_index = history_idx

            # I delete action
            layout.separator()
            delete_op = layout.operator("gemini.delete_history", text="Delete Item", icon='X')
            delete_op.history_index = history_idx

        # I store history index in context for menu
        context.window_manager.history_menu_index = self.history_index
        context.window_manager.popup_menu(draw_menu, title="History Options", icon='RENDER_RESULT')
        
        return {'FINISHED'}


class GEMINI_OT_use_history_both(Operator):
    """Use both prompt and style from render history"""
    bl_idname = "gemini.use_history_both"
    bl_label = "Use History Prompt + Style"
    bl_description = "Copy both prompt and style reference from this history item"
    bl_options = {'REGISTER'}

    history_index: IntProperty(
        name="History Index",
        description="Index of history item to use",
        default=0
    )

    def execute(self, context):
        try:
            props = context.scene.gemini_render
            
            if self.history_index < 0 or self.history_index >= len(props.render_history):
                self.report({'ERROR'}, "Invalid history index")
                return {'CANCELLED'}
            
            history_item = props.render_history[self.history_index]
            
            # I copy prompt
            props.prompt = history_item.prompt
            
            # I copy style reference if available
            if history_item.style_reference_used:
                style_image = None
                if history_item.style_reference_name in bpy.data.images:
                    style_image = bpy.data.images[history_item.style_reference_name]
                elif history_item.style_reference_thumbnail in bpy.data.images:
                    style_image = bpy.data.images[history_item.style_reference_thumbnail]
                
                if style_image:
                    props.style_reference_image = style_image
                    props.use_style_reference = True
                    self.report({'INFO'}, f" Both copied: prompt + style ({style_image.name})")
                    print(f" Both copied from history: prompt + style ({style_image.name})")
                else:
                    # I just prompt if style missing
                    props.use_style_reference = False
                    self.report({'WARNING'}, f" Style missing, copied prompt only")
                    print(f"📝 Style reference missing, copied prompt only")
            else:
                props.use_style_reference = False
                self.report({'INFO'}, f" Prompt copied (no style was used)")
                print(f"📝 Prompt copied, no style reference was used")
            
            return {'FINISHED'}
                
        except Exception as e:
            self.report({'ERROR'}, f"Failed to use history: {str(e)}")
            return {'CANCELLED'}


class GEMINI_OT_set_projection_source(Operator):
    """Set projection source to this image"""
    bl_idname = "gemini.set_projection_source"
    bl_label = "Use as Projection Source"
    bl_description = "Set this image as the direct source for projection"
    bl_options = {'REGISTER', 'UNDO'}
    
    history_index: IntProperty(name="History Index")
    
    def execute(self, context):
        props = context.scene.gemini_render
        if self.history_index < 0 or self.history_index >= len(props.render_history):
            self.report({'ERROR'}, "Invalid history index")
            return {'CANCELLED'}
            
        item = props.render_history[self.history_index]
        image = bpy.data.images.get(item.image_name)
        
        if not image:
            self.report({'ERROR'}, f"Image '{item.image_name}' not found in Blender")
            return {'CANCELLED'}
            
        props.projection_source = 'IMAGE'
        props.projection_image = image
        

        restore_view_state(context, item)
        
        self.report({'INFO'}, f"Source set to: {item.image_name} (Angle Restored)")
        return {'FINISHED'}


class GEMINI_OT_load_image_as_reference(Operator, ImportHelper):
    """Load image file as style reference"""
    bl_idname = "gemini.load_image_as_reference"
    bl_label = "Load Image as Reference"
    bl_description = "Load an image file to use as style reference"
    bl_options = {'REGISTER'}

    # I file browser properties
    filename_ext = ""
    filter_glob: StringProperty(
        default="*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.tga;*.exr;*.hdr",
        options={'HIDDEN'}
    )
    
    filepath: StringProperty(
        name="File Path",
        description="Filepath used for importing the image file",
        maxlen=1024,
        subtype='FILE_PATH'
    )

    def execute(self, context):
        try:
            props = context.scene.gemini_render
            

            if not self.filepath:
                self.report({'WARNING'}, "No file selected")
                return {'CANCELLED'}
            
            # I load the image into Blender
            try:
                # I load the image
                image = bpy.data.images.load(self.filepath, check_existing=False)
                image_name = os.path.basename(self.filepath)
                

                if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.tga', '.exr', '.hdr')):
                    base_name = os.path.splitext(image_name)[0]
                    image.name = f"Reference_{base_name}"
                
                print(f"🎨 Loaded reference image: {image.name}")
                print(f" Image size: {image.size[0]}x{image.size[1]}")
                
            except Exception as e:
                self.report({'ERROR'}, f"Failed to load image: {str(e)}")
                return {'CANCELLED'}
            
            # I automatically set as reference
            props.style_reference_image = image
            

            if not props.use_style_reference:
                props.use_style_reference = True
                
            self.report({'INFO'}, f"Reference image loaded: {image.name}")
            print(f" Reference image set automatically: {image.name}")
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load reference image: {str(e)}")
            print(f" Reference loading error: {str(e)}")
            return {'CANCELLED'}
    
    def invoke(self, context, event):

        import os
        try:
            if os.name == 'nt':  # Windows
                pictures_folder = os.path.join(os.path.expanduser('~'), 'Pictures')
                if os.path.exists(pictures_folder):
                    self.filepath = pictures_folder
                else:
                    self.filepath = os.path.expanduser('~')
            else:  # Linux/Mac
                self.filepath = os.path.expanduser('~')
        except:
            pass
            
        # I open the file browser
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class GEMINI_OT_load_example_reference(Operator):
    """Load example reference image"""
    bl_idname = "gemini.load_example_reference"
    bl_label = "Load Example Reference"
    bl_description = "Load an example reference image to test style transfer"
    bl_options = {'REGISTER'}

    def execute(self, context):
        try:
            props = context.scene.gemini_render
            

            import bpy
            import bmesh
            from mathutils import Vector
            

            example_name = "Gemini_Example_Reference"
            if example_name in bpy.data.images:
                existing_img = bpy.data.images[example_name]
                props.style_reference_image = existing_img
                if not props.use_style_reference:
                    props.use_style_reference = True
                self.report({'INFO'}, "Example reference reloaded")
                return {'FINISHED'}
            

            width, height = 512, 512
            example_img = bpy.data.images.new(example_name, width, height)
            

            pixels = [0.0] * (width * height * 4)  # RGBA
            
            for y in range(height):
                for x in range(width):
                    index = (y * width + x) * 4
                    

                    r = (x / width) * 0.8 + 0.2  # I red gradient
                    g = (y / height) * 0.6 + 0.3  # I green gradient
                    b = ((x + y) / (width + height)) * 0.9 + 0.1  # I blue mix
                    a = 1.0
                    
                    # I add some noise for texture
                    import random
                    noise = random.random() * 0.1 - 0.05
                    r = max(0, min(1, r + noise))
                    g = max(0, min(1, g + noise))
                    b = max(0, min(1, b + noise))
                    
                    pixels[index] = r
                    pixels[index + 1] = g  
                    pixels[index + 2] = b
                    pixels[index + 3] = a
            

            example_img.pixels = pixels
            

            props.style_reference_image = example_img
            

            if not props.use_style_reference:
                props.use_style_reference = True
            
            self.report({'INFO'}, "Example reference created and loaded")
            print("🎨 Example reference image created")
            return {'FINISHED'}
            
        except Exception as e:
            # Fallback: Show helpful message
            message = (
                "Style reference ideas:\n" +
                "• Find architectural photos online\n" +
                "• Download artwork or paintings\n" +
                "• Use nature photography\n" +
                "• Load via 'Load Photo from Computer' button"
            )
            
            def draw_message(self, context):
                layout = self.layout
                for line in message.split('\n'):
                    layout.label(text=line)
            
            context.window_manager.popup_menu(draw_message, title="Style Reference Examples", icon='IMAGE_DATA')
            self.report({'INFO'}, "Check popup for reference image ideas")
            return {'FINISHED'}

class GEMINI_OT_debug_next(Operator):
    """Manual Step-by-Step Debugging for Texture Projection"""
    bl_idname = "gemini.debug_next"
    bl_label = "Next Debug Step"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import bmesh
        import tempfile
        import os
        
        props = context.scene.gemini_render
        
        print(f"\n [GEMINI DEBUG] Starting Robust Debug Sequence...")
        
        # 1. Immediate Context Validation
        if not context.selected_objects:
             # I try to find active object
             if context.active_object:
                 context.active_object.select_set(True)
             else:
                 self.report({'ERROR'}, "No objects selected! Please select a mesh.")
                 return {'CANCELLED'}

        # 2. Store Active Object Name (Persistence)
        active = context.active_object
        if active:
             _debug_storage['debug_active_object'] = active.name
             
        props = context.scene.gemini_render
        
        # 1. Validation (Synchronous)
        try:
            # I we want to use view_layer.objects.active consistently
            if not context.view_layer.objects.active:
                self.report({'ERROR'}, "No active object selected")
                return {'CANCELLED'}
            
            # I start logic
            if props.debug_step == 0:
                print(" [GEMINI DEBUG] Starting Robust Manual Debug Sequence...")
                # I global cleanup
                _debug_storage.clear()
                _debug_storage['debug_active_object'] = context.active_object.name
                
            # I run EXACTLY ONE STEP
            self.execute_debug_step(context)
            
        except Exception as e:
            self.report({'ERROR'}, f"Debug Execution Error: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

    def execute_debug_step(self, context):
        """Executes a single step of the debug process."""
        import bmesh
        import os
        import tempfile
        import shutil
        from . import projection_utils
        from bpy_extras import view3d_utils, object_utils
        
        props = context.scene.gemini_render
        scene = context.scene
        step = props.debug_step
        
        if not props.debug_mode:
            print(" [DEBUG] Sequence Stopped (Debug Mode Disabled)")
            return
            
        print(f"\n================ [GEMINI DEBUG] Executing Step {step} ================")
        
        try:
            # Re-acquire Active Object safely
            target_name = _debug_storage.get('debug_active_object')
            if target_name:
                obj = bpy.data.objects.get(target_name)
                if obj:
                    context.view_layer.objects.active = obj
                    if not obj.select_get():
                        obj.select_set(True)
            
            # STEP 0: Capture & Init
            if step == 0:
                print("1 [DEBUG] Starting Capture Phase...")
                

                if 'target_objects_data' in _debug_storage: del _debug_storage['target_objects_data']
                if 'mask_repair_data' in _debug_storage: del _debug_storage['mask_repair_data']
                _debug_storage['target_objects_data'] = []
                _debug_storage['mask_repair_data'] = None
                
                # I temp Dir
                temp_dir = tempfile.mkdtemp(prefix="gemini_debug_")
                _debug_storage['temp_dir'] = temp_dir
                init_img_path = os.path.join(temp_dir, "init.png")
                _debug_storage['init_img_path'] = init_img_path
                

                viewport_area = next((a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
                if not viewport_area:
                    print(" No Viewport")
                    return
                    
                print(f"1.1 [DEBUG] Capturing to {init_img_path}")
                

                orig_res_x = scene.render.resolution_x
                orig_res_y = scene.render.resolution_y
                orig_res_pct = scene.render.resolution_percentage
                
                region = viewport_area.regions[-1]
                scene.render.resolution_x = region.width
                scene.render.resolution_y = region.height
                scene.render.resolution_percentage = 100
                scene.render.filepath = init_img_path
                scene.render.image_settings.file_format = 'PNG'
                
                bpy.ops.render.opengl(write_still=True, view_context=True)
                
                # Restore
                scene.render.resolution_x = orig_res_x
                scene.render.resolution_y = orig_res_y
                scene.render.resolution_percentage = orig_res_pct
                
                print("1.2 [DEBUG] Capture Complete")
                props.debug_step += 1
                
            # STEP 1: Temp Object Creation (Repair Mode)
            elif step == 1:
                print("2 [DEBUG] Creating Temp Objects (Mask Repair)...")
                
                if props.mask_repair_mode:
                    obj = context.active_object
                    if not obj or obj.type != 'MESH': 
                         print(" Invalid Object")
                         return
                    
                    obj_name = obj.name
                    print(f"2.1 [DEBUG] Processing Object: {obj_name}")

                    # 1. SETUP BMESH FOR FACE SELECTION
                    if obj.mode != 'EDIT':
                         bpy.ops.object.mode_set(mode='EDIT')
                    
                    bm = bmesh.from_edit_mesh(obj.data)
                    selected_faces = [f for f in bm.faces if f.select]
                    
                    if not selected_faces:
                         print(" No faces selected in Edit Mode")
                         return
                    
                    # 2. FACE-AWARE TEXTURE TARGETING
                    target_mat_index = selected_faces[0].material_index
                    original_texture = None
                    if target_mat_index < len(obj.material_slots):
                        slot = obj.material_slots[target_mat_index]
                        if slot.material and slot.material.use_nodes:
                            for node in slot.material.node_tree.nodes:
                                if node.type == 'TEX_IMAGE' and node.image:
                                    original_texture = node.image
                                    break
                    
                    tex_name = original_texture.name if original_texture else "None"
                    print(f"2.2 [DEBUG] Target Texture (Material Slot {target_mat_index}): {tex_name}")
                    
                    if not original_texture:
                        print(" [DEBUG] No texture found in targeted slot!")

                    # 3. CREATE TEMP OBJECT (NON-DESTRUCTIVE BMesh copy)
                    bm_copy = bm.copy()
                    faces_to_delete = [f for f in bm_copy.faces if not f.select]
                    bmesh.ops.delete(bm_copy, geom=faces_to_delete, context='FACES')
                    
                    # I new Mesh & Object
                    temp_mesh = bpy.data.meshes.new(f"{obj_name}_DebugMask_Data")
                    bm_copy.to_mesh(temp_mesh)
                    bm_copy.free()
                    
                    temp_obj = bpy.data.objects.new(f"{obj_name}_MaskTemp", temp_mesh)
                    temp_obj.matrix_world = obj.matrix_world.copy()
                    context.collection.objects.link(temp_obj)
                    
                    # 4. Apply Mask Material (Magenta)
                    mask_mat = bpy.data.materials.new(name="Gemini_Debug_Mask")
                    mask_mat.use_nodes = True
                    mask_mat.node_tree.nodes.clear()
                    m_out = mask_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
                    m_emit = mask_mat.node_tree.nodes.new("ShaderNodeEmission")
                    m_emit.inputs['Color'].default_value = (props.mask_color[0], props.mask_color[1], props.mask_color[2], 1.0)
                    mask_mat.node_tree.links.new(m_emit.outputs['Emission'], m_out.inputs['Surface'])
                    temp_obj.data.materials.append(mask_mat)
                    
                    # I store Data
                    _debug_storage['mask_repair_data'] = {
                        'temp_objects': [temp_obj.name],
                        'original_textures': {obj_name: tex_name},
                        'original_object_names': [obj_name]
                    }
                    
                    print(f"2.3 [DEBUG] Temp Object Created & Linked: {temp_obj.name}")
                    
                    # SELECT TEMP for Step 2
                    bpy.ops.object.mode_set(mode='OBJECT')
                    bpy.ops.object.select_all(action='DESELECT')
                    temp_obj.select_set(True)
                    context.view_layer.objects.active = temp_obj
                    
                    props.debug_step += 1
                else:
                    print("2.0 [DEBUG] Not in Repair Mode, skipping to Step 3")
                    props.debug_step = 3 # I skip to Mock API
            
            # STEP 2: Material & UV Setup
            elif step == 2:
                print("3 [DEBUG] Setting up Materials & UVs...")
                mask_repair_data = _debug_storage.get('mask_repair_data')
                
                # I need Viewport Data for UV Projection
                viewport_area = next((a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
                region = viewport_area.regions[-1]
                space_data = next(s for s in viewport_area.spaces if s.type == 'VIEW_3D')
                
                # CAMERA PRIORITY: Use camera projection when camera exists
                # I this matches the camera render used for capture
                has_camera = scene.camera is not None
                use_camera_render = has_camera
                
                if use_camera_render:
                    # I use snapped scene render resolution (matches camera render)
                    v_width, v_height = snap_to_supported_resolution(scene.render.resolution_x, scene.render.resolution_y)
                else:
                    # Fallback: viewport dimensions
                    v_width, v_height = region.width, region.height
                
                # I shared Material (Magenta)
                mat_name = "Gemini_Projection_Material"
                material = bpy.data.materials.get(mat_name)
                if not material:
                    material = bpy.data.materials.new(name=mat_name)
                    material.use_nodes = True
                
                nodes = material.node_tree.nodes
                nodes.clear()
                out = nodes.new("ShaderNodeOutputMaterial")
                emit = nodes.new("ShaderNodeEmission")
                emit.inputs['Color'].default_value = (1.0, 0.0, 1.0, 1.0) # MAGENTA
                material.node_tree.links.new(emit.outputs['Emission'], out.inputs['Surface'])
                
                if mask_repair_data:

                     temp_name = mask_repair_data['temp_objects'][0]
                     temp_obj = bpy.data.objects.get(temp_name)
                     original_name = mask_repair_data['original_object_names'][0]
                     
                     if temp_obj:
                         print(f"3.1 [DEBUG] Applying Material to {temp_obj.name}")
                         temp_obj.data.materials.clear()
                         temp_obj.data.materials.append(material)
                         
                         print("3.2 [DEBUG] Projecting UVs...")
                         context.view_layer.objects.active = temp_obj
                         bpy.ops.object.mode_set(mode='EDIT')
                         
                         bm = bmesh.from_edit_mesh(temp_obj.data)
                         uv_layer = bm.loops.layers.uv.get("Projected UVs") or bm.loops.layers.uv.new("Projected UVs")
                         
                         for face in bm.faces:
                            for loop in face.loops:
                                world_co = temp_obj.matrix_world @ loop.vert.co
                                if use_camera_render:
                                     screen_co = object_utils.world_to_camera_view(scene, scene.camera, world_co)
                                     uv = (screen_co[0], screen_co[1])
                                else:
                                     screen_co = view3d_utils.location_3d_to_region_2d(region, space_data.region_3d, world_co)
                                     uv = (screen_co[0]/v_width, screen_co[1]/v_height) if screen_co else (0,0)
                                loop[uv_layer].uv = uv
                         
                         bmesh.update_edit_mesh(temp_obj.data)
                         
                         # I copy BM explicitly while in valid state
                         bm_copy = bm.copy() 
                         
                         bpy.ops.object.mode_set(mode='OBJECT')
                         
                         _debug_storage['target_objects_data'].append({
                            'object_name': temp_obj.name, 
                            'original_object_name': original_name,
                            'bm_copy': bm_copy,
                            'src_uv_name': "Projected UVs",
                            'dest_uv_name': ""
                         })
                         
                props.debug_step += 1

            # STEP 3: MOCK API (Image Load)
            elif step == 3:
                print("4 [DEBUG] Mock API Load...")
                init_path = _debug_storage.get('init_img_path')
                if not init_path or not os.path.exists(init_path):
                     print(" Image capture failed or missing")
                     return
                     
                img = bpy.data.images.load(init_path)
                img.name = "Gemini_Projection_Result"
                

                mat = bpy.data.materials.get("Gemini_Projection_Material")
                nodes = mat.node_tree.nodes
                nodes.clear()
                
                out = nodes.new("ShaderNodeOutputMaterial")
                emit = nodes.new("ShaderNodeEmission")
                tex_node = nodes.new("ShaderNodeTexImage")
                tex_node.name = "Gemini_Image_Node"
                tex_node.image = img
                uv_map = nodes.new("ShaderNodeUVMap")
                uv_map.uv_map = "Projected UVs"
                
                links = mat.node_tree.links
                links.new(uv_map.outputs['UV'], tex_node.inputs['Vector'])
                links.new(tex_node.outputs['Color'], emit.inputs['Color'])
                links.new(emit.outputs['Emission'], out.inputs['Surface'])
                
                print("4.1 [DEBUG] Material updated with Texture")
                props.debug_step += 1

            # STEP 4: BAKE
            elif step == 4:
                print("5 [DEBUG] BAKING...")
                
                for data in _debug_storage['target_objects_data']:
                    temp_name = data['object_name']
                    original_name = data.get('original_object_name')
                    print(f"5.1 [DEBUG] Baking {temp_name} -> {original_name}")
                    
                    obj = bpy.data.objects.get(temp_name) # I temp Obj
                    
                    mask_data = _debug_storage.get('mask_repair_data')
                    if mask_data and original_name:
                        orig_tex_name = mask_data['original_textures'].get(original_name)
                        orig_tex = bpy.data.images.get(orig_tex_name)
                        
                        if obj and orig_tex:
                            print(f"5.2 [DEBUG] Target Texture: {orig_tex.name}")
                            
                            # I incremental Bake Logic
                            projection_utils.bake(
                                context=context,
                                obj=obj,
                                texture_node_name="Gemini_Image_Node",
                                target_image=orig_tex, # In-Place
                                src_uv_name="Projected UVs",
                                margin=1,
                                use_clear=False
                            )
                            print(" Bake Finished")
                
                props.debug_step += 1
            
            # STEP 5: CLEANUP
            elif step == 5:
                print("6 [DEBUG] Cleaning Up...")
                mask_data = _debug_storage.get('mask_repair_data')
                if mask_data:
                    for t in mask_data['temp_objects']:
                        o = bpy.data.objects.get(t)
                        if o:
                            bpy.data.objects.remove(o, do_unlink=True)
                            print(f"6.1 [DEBUG] Removed {t}")
                    
                    # MODIFICATION: Restore selection to original objects
                    first_obj = None
                    for orig_name in mask_data.get('original_object_names', []):
                        o = bpy.data.objects.get(orig_name)
                        if o:
                            o.select_set(True)
                            if not first_obj:
                                first_obj = o
                    
                    if first_obj:
                        context.view_layer.objects.active = first_obj
                        print(f" [DEBUG] Restored selection to {first_obj.name}")
                
                props.debug_step = 0
                print(" [DEBUG] SEQUENCE COMPLETE ")
                self.report({'INFO'}, "Debug Sequence Finished")
                
        except Exception as e:
            print(f" [DEBUG ERROR] Step {step} Failed: {e}")
            import traceback
            traceback.print_exc()
        
        return None
