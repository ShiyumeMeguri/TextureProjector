import bpy
import threading
import os
from queue import Queue
from typing import Callable, Any
import time
import bmesh
import gpu
import mathutils
import numpy as np
from . import projection_utils

class BlenderThreadManager:
    """Manager for thread-safe operations with Blender"""
    
    def __init__(self):
        self.command_queue = Queue()
        self.timer_registered = False
        
    def execute_in_main_thread(self, func: Callable, *args, **kwargs) -> None:
        """Execute function in main Blender thread via timer"""
        self.command_queue.put((func, args, kwargs))
        
        # I register timer if not already registered
        if not self.timer_registered:
            bpy.app.timers.register(self._process_queue, first_interval=0.01)
            self.timer_registered = True
    
    def _process_queue(self) -> float:
        """Process queued commands in main thread"""
        try:
            while not self.command_queue.empty():
                func, args, kwargs = self.command_queue.get_nowait()
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Error executing queued command: {e}")
                    # I continue processing other commands
                    

            return 0.1
            
        except Exception as e:
            print(f"Error in queue processor: {e}")
            return 0.1
    
    def stop_timer(self) -> None:
        """Stop the timer (call when addon is unregistered)"""
        if self.timer_registered:
            try:
                bpy.app.timers.unregister(self._process_queue)
            except:
                pass
            self.timer_registered = False

# I global thread manager instance
_thread_manager = BlenderThreadManager()

def execute_in_main_thread(func: Callable, *args, **kwargs) -> None:
    """Convenience function to execute in main thread"""
    _thread_manager.execute_in_main_thread(func, *args, **kwargs)

def get_view3d_context():
    """Find a 3D view area and return a context-like dictionary for it"""
    import bpy
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return {
                            'window': window,
                            'screen': window.screen,
                            'area': area,
                            'region': region,
                            'space_data': area.spaces.active,
                            'scene': bpy.context.scene,
                            'view_layer': bpy.context.view_layer,
                        }
    return None

def update_render_status(scene, status_text: str, is_rendering: bool = None) -> None:
    """Update render status in UI (thread-safe)"""
    def _update():
        try:
            print(f"Updating status: {status_text}")
            if hasattr(scene, 'gemini_render'):
                props = scene.gemini_render
                props.status_text = status_text
                if is_rendering is not None:
                    props.is_rendering = is_rendering
                    print(f"Set is_rendering = {is_rendering}")
                
                # I force redraw all areas
                try:
                    import bpy
                    for window in bpy.context.window_manager.windows:
                        for area in window.screen.areas:
                            if area.type == 'VIEW_3D':
                                area.tag_redraw()
                except Exception as redraw_error:
                    print(f"Redraw warning: {redraw_error}")
                    
                print("Status updated successfully")
            else:
                print("Scene has no gemini_render property")
        except Exception as e:
            print(f"Error updating status: {e}")
            import traceback
            print(f"Status update traceback:\n{traceback.format_exc()}")
    
    execute_in_main_thread(_update)



def _load_result_image_sync(image_data: bytes, image_name: str = "AI_Result", user_prompt: str = "", cam_data: dict = None) -> Any:
    """Synchronous version of image loading. MUST be called from the main thread.
    Returns the loaded Image object (either the history one or the Render Result copy)."""
    print(f"📥 _load_result_image_sync starting for {image_name}")
    try:
        import tempfile
        import os
        import datetime
        

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(image_data)
            temp_path = f.name
        
        try:
            # I load image into Blender
            if image_name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[image_name])
            
            img = bpy.data.images.load(temp_path)
            img.name = image_name
            
            # I keep original image for history
            permanent_image_for_history = None
            if user_prompt:
                permanent_name = f"AI_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                img.name = permanent_name
                img.pack()
                permanent_image_for_history = img
                print(f"History image: {permanent_name}")
            

            render_result = bpy.data.images.get('Render Result')
            if render_result:
                bpy.data.images.remove(render_result)
            
            render_result = img.copy()
            render_result.name = 'Render Result'
            

            for area in bpy.context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    for space in area.spaces:
                        if space.type == 'IMAGE_EDITOR':
                            space.image = render_result
                    area.tag_redraw()
            
            # I handle Gallery history entries
            if user_prompt:
                scene = bpy.context.scene
                if hasattr(scene, 'gemini_render'):
                    props = scene.gemini_render
                    item = props.render_history.add()
                    item.prompt = user_prompt
                    item.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    item.image_name = permanent_image_for_history.name if permanent_image_for_history else render_result.name
                    
                    if props.use_style_reference and props.style_reference_image:
                        item.style_reference_used = True
                        item.style_reference_name = props.style_reference_image.name
                    
                    # I store Camera Data if available
                    if cam_data:
                        try:
                            item.cam_location = cam_data.get('location', (0,0,0))
                            item.cam_rotation = cam_data.get('rotation', (1,0,0,0))
                            item.cam_lens = cam_data.get('lens', 50.0)
                            item.view_distance = cam_data.get('view_distance', 10.0)
                            item.is_camera_view = cam_data.get('is_camera_view', False)
                            
                            # Store explicit camera object transform (if available)
                            if 'cam_obj_location' in cam_data:
                                item.cam_obj_location = cam_data['cam_obj_location']
                            if 'cam_obj_rotation' in cam_data:
                                item.cam_obj_rotation = cam_data['cam_obj_rotation']
                                
                            print(f" Camera data stored in history item")
                        except Exception as ce:
                            print(f" Failed to store camera data: {ce}")
                    
                    # I cleanup old history
                    while len(props.render_history) > 10:
                        oldest = props.render_history[0]
                        if oldest.image_name in bpy.data.images:
                            bpy.data.images.remove(bpy.data.images[oldest.image_name])
                        props.render_history.remove(0)
                        
            return permanent_image_for_history or render_result
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f" Error in _load_result_image_sync: {e}")
        import traceback
        traceback.print_exc()
        return None



class ProjectionRenderThread(threading.Thread):
    """Background thread for AI Texture Projection pipeline"""
    
    # 1. 在 __init__ 中添加 reference_path 参数
    def __init__(self, context, api_client, user_prompt, source_path, sim_path, target_objects_data, image_node_name, material_name, do_bake, bypass_api=False, mask_repair_data=None, input_source='COLOR', debug_mode=False, source_image_override=None, cam_data=None, reference_path=None):
        super().__init__(daemon=True)
        self.scene = context.scene
        self.api_client = api_client
        self.user_prompt = user_prompt
        self.source_path = source_path
        self.sim_path = sim_path
        self.target_objects_data = target_objects_data
        self.image_node_name = image_node_name
        self.material_name = material_name
        self.do_bake = do_bake
        self.bypass_api = bypass_api
        self.mask_repair_data = mask_repair_data
        self.input_source = input_source
        self.debug_mode = debug_mode
        self.source_image_override = source_image_override
        self.cam_data = cam_data
        self.reference_path = reference_path # Store it!
        self._stop_event = threading.Event()
        print(f"[GEMINI] Thread init. Reference Path: {self.reference_path}")
    
    def stop(self):
        self._stop_event.set()
        
    def run(self):
        print("ProjectionRenderThread starting...")
        try:
            update_render_status(self.scene, "Sending projection to Gemini...", True)
            
            from . import operators 
            
            projection_prompt = f"{self.user_prompt}"
            props = self.scene.gemini_render
            resolution = int(props.resolution)
            
            # === DEBUG MODE: SAVE ACTUAL INPUTS ===
            if self.debug_mode:
                try:
                    import shutil
                    import tempfile
                    blend_path = bpy.data.filepath
                    base_debug_dir = os.path.join(os.path.dirname(blend_path), "textures") if blend_path else os.path.join(tempfile.gettempdir(), "textures")
                    if not os.path.exists(base_debug_dir):
                        os.makedirs(base_debug_dir)
                        
                    # 1. Save Input (Viewport Capture)
                    shutil.copy2(self.source_path, os.path.join(base_debug_dir, "debug_input_source.png"))
                    print(f"🐞 Debug Input saved: {os.path.join(base_debug_dir, 'debug_input_source.png')}")
                    
                    # 2. Save Reference (If exists) <--- YOUR REQUEST
                    if self.reference_path and os.path.exists(self.reference_path):
                        shutil.copy2(self.reference_path, os.path.join(base_debug_dir, "debug_input_reference.png"))
                        print(f"🐞 Debug Reference saved: {os.path.join(base_debug_dir, 'debug_input_reference.png')}")
                    else:
                        print("🐞 Debug: No reference path found to save.")
                        
                except Exception as de:
                    print(f" Debug input sync failed: {de}")
            # ======================================

            if self.source_image_override:
                print(f"🖼 Direct Image Mode...")
                image_data = None 
                mime_type = "image/png"
            elif self.bypass_api:
                print("🛡 Simulation Mode...")
                with open(self.sim_path, 'rb') as f:
                    image_data = f.read()
                mime_type = "image/png"
            else:
                print(f" Calling AI (Reference: {self.reference_path})...")
                
                is_color = (self.input_source == 'COLOR')
                
                image_data, mime_type = self.api_client.generate_image(
                    depth_image_path=self.source_path,
                    user_prompt=projection_prompt,
                    reference_image_path=self.reference_path,
                    is_color_render=is_color,
                    width=resolution,
                    height=resolution
                )
            
            # DEBUG: Save final output (AI or simulated)
            if self.debug_mode:
                try:
                    blend_path = bpy.data.filepath
                    base_debug_dir = os.path.join(os.path.dirname(blend_path), "textures") if blend_path else os.path.join(tempfile.gettempdir(), "textures")
                    if os.path.exists(base_debug_dir):
                        res_path = os.path.join(base_debug_dir, "debug_output.png")
                        with open(res_path, 'wb') as f:
                            f.write(image_data)
                        print(f"🐞 Debug output saved to: {res_path}")
                except Exception as de:
                    print(f" Debug result save failed: {de}")

            if self._stop_event.is_set():
                return
                
            update_render_status(self.scene, "Processing result...", True)
            
            # 2. Main thread callback to apply and bake
            def _apply_result():
                try:
                    import bpy
                    import os
                    import tempfile
                    
                    # 1. Integrate with Render Gallery
                    if self.source_image_override:
                        res_img = self.source_image_override
                    else:
                        # I synchronous version to get the Image object IMMEDIATELY
                        res_img = _load_result_image_sync(image_data, "Gemini_Projection_Result", self.user_prompt, self.cam_data)
                    
                    if not res_img:
                         print(" Failed to load/retrieve projection result image")
                         return


                    material = bpy.data.materials.get(self.material_name)
                    if material:
                        node = material.node_tree.nodes.get(self.image_node_name)
                        if node:
                            node.image = res_img

                    # 2. Baking logic
                    if self.do_bake:
                        update_render_status(self.scene, "Baking projection (Blender Native)...", True)
                        
                        for data in self.target_objects_data:
                            obj = bpy.data.objects.get(data['object_name'])
                            if not obj: continue
                            
                            # ============================================================
                            # MASK REPAIR MODE
                            # ============================================================
                            if self.mask_repair_data:
                                original_obj_name = data.get('original_object_name')
                                
                                if not original_obj_name:
                                    print(f"🛡 Mask Repair Mode: Skipping non-repair object '{obj.name}'")
                                    continue

                                print(f"Mask Repair Mode: {obj.name} -> {original_obj_name}")
                                
                                original_tex_name = self.mask_repair_data['original_textures'].get(original_obj_name)
                                if not original_tex_name or original_tex_name not in bpy.data.images:
                                    print(f"  Mask Repair: Original texture '{original_tex_name}' not found")
                                    continue
                                
                                original_tex = bpy.data.images[original_tex_name]
                                orig_obj_ref = bpy.data.objects.get(original_obj_name)
                                
                                # Use unified bake function
                                projection_utils.perform_projection_bake(
                                    context=bpy.context,
                                    obj=obj,
                                    texture_node_name=self.image_node_name,
                                    target_image=original_tex,
                                    src_uv_name=data['src_uv_name'],
                                    dest_uv_name=data.get('dest_uv_name', "UVMap"),
                                    is_mask_repair=True,
                                    original_obj=orig_obj_ref
                                )
                                continue
                            
                            # ============================================================
                            # NORMAL MODE
                            # ============================================================
                            if '_MaskTemp' in obj.name:
                                print(f"Skipping temp mask object '{obj.name}'")
                                continue
                            
                            # Create unique baked image
                            baked_name = f"{obj.name}_Baked_AI"
                            if baked_name in bpy.data.images:
                                bpy.data.images.remove(bpy.data.images[baked_name])
                            baked_img = bpy.data.images.new(baked_name, res_img.size[0], res_img.size[1])
                            
                            # Prepare material for baking
                            if not material:
                                material = bpy.data.materials.get(self.material_name)
                                
                            baked_mat_name = f"{material.name}_{obj.name}_Baked"
                            obj_mat = bpy.data.materials.get(baked_mat_name)
                            if not obj_mat:
                                obj_mat = material.copy()
                                obj_mat.name = baked_mat_name
                            
                            # Setup emission shader for pure color bake
                            o_nodes = obj_mat.node_tree.nodes
                            o_links = obj_mat.node_tree.links
                            o_nodes.clear()
                            o_out = o_nodes.new("ShaderNodeOutputMaterial")
                            o_out.location = (400, 0)
                            o_emit = o_nodes.new("ShaderNodeEmission")
                            o_emit.location = (200, 0)
                            o_emit.inputs['Strength'].default_value = 1.0
                            o_tex = o_nodes.new("ShaderNodeTexImage")
                            o_tex.name = self.image_node_name
                            o_tex.location = (0, 0)
                            o_uv = o_nodes.new("ShaderNodeUVMap")
                            o_uv.name = "Gemini_UV_Map"
                            o_uv.uv_map = data.get('dest_uv_name', "UVMap")
                            o_uv.location = (-200, 0)
                            o_links.new(o_uv.outputs['UV'], o_tex.inputs['Vector'])
                            o_links.new(o_tex.outputs['Color'], o_emit.inputs['Color'])
                            o_links.new(o_emit.outputs['Emission'], o_out.inputs['Surface'])
                            
                            # Unique material assignment per object
                            unique_mats_map = {}
                            for m_idx, slot in enumerate(obj.material_slots):
                                if not slot.material: continue
                                mat = slot.material
                                if mat not in unique_mats_map:
                                    new_mat = mat.copy()
                                    new_mat.name = f"{mat.name}_{obj.name}_Baked"
                                    unique_mats_map[mat] = new_mat
                                obj.data.materials[m_idx] = unique_mats_map[mat]
                                if unique_mats_map[mat].use_nodes:
                                    proj_node = unique_mats_map[mat].node_tree.nodes.get(self.image_node_name)
                                    if proj_node:
                                        proj_node.image = res_img

                            # Configure UV Layer for baking
                            if data['dest_uv_name'] in obj.data.uv_layers:
                                obj.data.uv_layers.active = obj.data.uv_layers[data['dest_uv_name']]
                                obj.data.uv_layers[data['dest_uv_name']].active_render = True
                            
                            # Perform bake using unified function
                            margin = getattr(self.scene.gemini_render, "bake_margin", 16)
                            projection_utils.perform_projection_bake(
                                context=bpy.context,
                                obj=obj,
                                texture_node_name=self.image_node_name,
                                target_image=baked_img,
                                src_uv_name=data['src_uv_name'],
                                dest_uv_name=data.get('dest_uv_name', "UVMap"),
                                margin=margin,
                                search_img=res_img
                            )
                            
                            baked_img.pack()
                    
                    update_render_status(self.scene, "Projection completed!", False)
                    
                except Exception as e:
                    print(f"Error applying projection result: {e}")
                    import traceback
                    traceback.print_exc()
                    update_render_status(self.scene, f"Error: {str(e)}", False)
                finally:
                    # I cleanup temp files
                    try:
                        if os.path.exists(self.depth_path): os.unlink(self.depth_path)
                        if os.path.exists(self.init_image_path): os.unlink(self.init_image_path)
                    except: pass
                    for data in self.target_objects_data:
                        try: data['bm_copy'].free()
                        except: pass
                    
                    # I mask Repair Mode: Cleanup temp objects and materials
                    if self.mask_repair_data:
                        print("Mask Repair Mode: Cleaning up temp objects and materials...")
                        
                        # I delete temp mesh objects
                        for temp_obj_name in self.mask_repair_data.get('temp_objects', []):
                            try:
                                temp_obj = bpy.data.objects.get(temp_obj_name)
                                if temp_obj:
                                    # I store mesh data reference to delete later
                                    temp_mesh = temp_obj.data
                                    
                                    # I delete object safely with do_unlink=True
                                    bpy.data.objects.remove(temp_obj, do_unlink=True)
                                    
                                    # I delete mesh data if it exists and has no other users
                                    if temp_mesh and temp_mesh.users == 0:
                                        bpy.data.meshes.remove(temp_mesh)
                                        
                                    print(f"Mask Repair Mode: Deleted temp object '{temp_obj_name}' and its mesh data")
                            except Exception as e:
                                print(f" Mask Repair Mode: Failed to delete temp object '{temp_obj_name}': {e}")
                        
                        # I delete temp materials
                        for temp_mat_name in self.mask_repair_data.get('temp_materials', []):
                            try:
                                temp_mat = bpy.data.materials.get(temp_mat_name)
                                if temp_mat:
                                    bpy.data.materials.remove(temp_mat)
                                    print(f"Mask Repair Mode: Deleted temp material '{temp_mat_name}'")
                            except Exception as e:
                                print(f" Mask Repair Mode: Failed to delete temp material '{temp_mat_name}': {e}")
                        
                        print("Mask Repair Mode: Cleanup complete")
                        
                        # MODIFICATION: Restore selection to original objects
                        try:
                            # 1. Clear selection
                            bpy.ops.object.select_all(action='DESELECT')
                            
                            # 2. Re-select original objects
                            first_obj = None
                            for orig_name in self.mask_repair_data.get('original_object_names', []):
                                o = bpy.data.objects.get(orig_name)
                                if o:
                                    o.select_set(True)
                                    if not first_obj:
                                        first_obj = o
                            
                            # 3. Set active
                            if first_obj:
                                bpy.context.view_layer.objects.active = first_obj
                                print(f" Restored selection to original object: {first_obj.name}")
                                
                                # 4. Optionally restore mode if needed (usually Object mode is safer here)

                        except Exception as e:
                            print(f" Selection restoration failed: {e}")

            execute_in_main_thread(_apply_result)
            
        except Exception as e:
            print(f"Projection thread error: {e}")

def stop_thread_manager():
    """Stop the thread manager (call on addon unregister)"""
    _thread_manager.stop_timer()
