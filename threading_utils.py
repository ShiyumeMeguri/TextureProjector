import bpy
import threading
import os
import re
import tempfile
from queue import Queue
from typing import Callable, Any
import time
import bmesh
import gpu
import mathutils
import numpy as np
from . import projection_utils


def get_output_directory() -> str:
    """Return the external output directory for generated textures."""
    if bpy.data.filepath:
        output_dir = bpy.path.abspath("//Textures/NanoBanana")
    else:
        output_dir = os.path.join(tempfile.gettempdir(), "NanoBanana")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]+', '_', name).strip()
    return sanitized or "image"


def build_output_filepath(base_name: str, extension: str = ".png") -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_filename(base_name)
    return os.path.join(get_output_directory(), f"{safe_name}_{timestamp}{extension}")


def save_blender_image(image, base_name: str = None, file_format: str = 'PNG') -> str:
    """Save a Blender image to disk and keep its filepath pointing there."""
    output_path = build_output_filepath(base_name or image.name, ".png")
    original_format = image.file_format
    try:
        image.filepath_raw = output_path
        image.file_format = file_format
        image.save()
    finally:
        image.filepath_raw = output_path
        image.file_format = original_format
    return output_path

class BlenderThreadManager:
    """Manager for thread-safe operations with Blender"""
    
    def __init__(self):
        self.command_queue = Queue()
        self.timer_registered = False
        
    def execute_in_main_thread(self, func: Callable, *args, **kwargs) -> None:
        """Execute function in main Blender thread via timer"""
        self.command_queue.put((func, args, kwargs))
        
        # Blend file switches can invalidate timers while this flag still says True.
        # Re-check the actual timer registration every time we enqueue work.
        if self.timer_registered and not bpy.app.timers.is_registered(self._process_queue):
            self.timer_registered = False

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
        import datetime
        
        temp_path = build_output_filepath(image_name)
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        # I load image into Blender
        if image_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[image_name])
        
        img = bpy.data.images.load(temp_path)
        img.name = image_name
        img.filepath_raw = temp_path
        
        try:
            _ = img.pixels[0]
            if hasattr(img, 'colorspace_settings'):
                img.colorspace_settings.name = 'sRGB'
        except Exception as pe:
            print(f" Warning: Failed to finalize image: {pe}")

        # I keep original image for history
        permanent_image_for_history = None
        if user_prompt:
            permanent_name = f"AI_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            img.name = permanent_name
            img.use_fake_user = True
            print(f" Marked image as fake user: {permanent_name}")
            permanent_image_for_history = img

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
            
    except Exception as e:
        print(f" Error in _load_result_image_sync: {e}")
        import traceback
        traceback.print_exc()
        return None



class ProjectionRenderThread(threading.Thread):
    """Background thread for AI Texture Projection pipeline"""
    
    # 1. 在 __init__ 中添加 reference_path 参数
    def __init__(self, context, api_client, user_prompt, source_path, sim_path, target_objects_data, image_node_name, material_name, do_bake, bypass_api=False, mask_repair_data=None, input_source='COLOR', debug_mode=False, source_image_override=None, cam_data=None, reference_path=None, capture_width=1024, capture_height=1024, temp_dir=None, resource_registry=None):
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
        self.capture_width = capture_width
        self.capture_height = capture_height
        self._stop_event = threading.Event()
        self.error_message = None
        self._finalize_queued = False
        self._finalized = False
        
        # ABSOLUTE RESOURCE TRACKING
        self.temp_dir = temp_dir
        self.resource_registry = resource_registry or {'temp_objects': [], 'temp_materials': [], 'target_objects_data': []}
        
        print(f"[GEMINI] Thread init. Temp Dir: {self.temp_dir} | Objects: {len(self.resource_registry['temp_objects'])}")
        
        # Track active threads for cancellation
        if not hasattr(BlenderThreadManager, 'active_threads'):
            BlenderThreadManager.active_threads = []
        BlenderThreadManager.active_threads.append(self)
    
    def stop(self):
        self._stop_event.set()

    def _queue_main_thread_finalize(self, image_data=None, apply_result=False):
        if self._finalize_queued:
            return
        self._finalize_queued = True
        if apply_result:
            execute_in_main_thread(self._apply_result_and_finalize, image_data)
        else:
            execute_in_main_thread(self._finalize_main_thread)

    def _set_status_sync(self, status_text: str, is_rendering: bool = None):
        """Update UI status from the main thread only."""
        try:
            if hasattr(self.scene, 'gemini_render'):
                props = self.scene.gemini_render
                props.status_text = status_text
                if is_rendering is not None:
                    props.is_rendering = is_rendering
        except Exception as e:
            print(f"Error updating status synchronously: {e}")

        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        except Exception as redraw_error:
            print(f"Redraw warning: {redraw_error}")

    def _apply_result_and_finalize(self, image_data):
        final_status = None
        try:
            if self._stop_event.is_set():
                final_status = "Projection cancelled"
                return

            if self.source_image_override:
                res_img = self.source_image_override
            else:
                print("📥 Loading result image into Blender...")
                res_img = _load_result_image_sync(image_data, "Gemini_Projection_Result", self.user_prompt, self.cam_data)
                if res_img:
                    print(f"✅ Image loaded successfully: {res_img.name}")

            if not res_img:
                raise RuntimeError("Failed to load projection result image")

            processed_mats = set()
            for data in self.target_objects_data:
                m_name = data.get('material_name', self.material_name)
                n_name = data.get('image_node_name', self.image_node_name)

                if m_name and m_name not in processed_mats:
                    mat = bpy.data.materials.get(m_name)
                    if mat and mat.use_nodes:
                        node = mat.node_tree.nodes.get(n_name)
                        if node:
                            node.image = res_img
                            print(f"Applied result to material: {m_name}")
                    processed_mats.add(m_name)

            if self.do_bake and not self._stop_event.is_set():
                self._set_status_sync("🥧 Baking result...", True)

                for data in self.target_objects_data:
                    if self._stop_event.is_set():
                        final_status = "Projection cancelled"
                        break

                    obj = bpy.data.objects.get(data['object_name'])
                    if not obj:
                        continue

                    original_obj_name = data.get('original_object_name')
                    if original_obj_name and self.mask_repair_data:
                        original_tex_name = self.mask_repair_data['original_textures'].get(original_obj_name)
                        original_tex = bpy.data.images.get(original_tex_name)
                        if original_tex:
                            print(f"Baking Mask Repair: {obj.name} -> {original_tex.name}")
                            projection_utils.perform_projection_bake(
                                context=bpy.context,
                                obj=obj,
                                texture_node_name=data.get('image_node_name', self.image_node_name),
                                target_image=original_tex,
                                src_uv_name=data['src_uv_name'],
                                dest_uv_name=data.get('dest_uv_name', "UVMap"),
                                is_mask_repair=True,
                                original_obj=bpy.data.objects.get(original_obj_name),
                                search_img=res_img
                            )
                    else:
                        if '_MaskTemp' in obj.name:
                            continue

                        safe_hash = abs(hash(obj.name)) % 100000
                        safe_name_prefix = obj.name[:30]
                        baked_name = f"{safe_name_prefix}_{safe_hash}_{int(time.time())}_Baked_AI"

                        if baked_name in bpy.data.images:
                            bpy.data.images.remove(bpy.data.images[baked_name])
                        baked_img = bpy.data.images.new(baked_name, res_img.size[0], res_img.size[1])

                        projection_utils.perform_projection_bake(
                            context=bpy.context,
                            obj=obj,
                            texture_node_name=data.get('image_node_name', self.image_node_name),
                            target_image=baked_img,
                            src_uv_name=data['src_uv_name'],
                            dest_uv_name=data.get('dest_uv_name', "UVMap"),
                            search_img=res_img
                        )
                        baked_path = save_blender_image(baked_img, baked_name)
                        print(f"Saved baked texture: {baked_path}")

            if final_status is None:
                final_status = "Projection cancelled" if self._stop_event.is_set() else "Ready to render"

        except Exception as e:
            print(f"Error applying projection result: {e}")
            self.error_message = str(e)
            final_status = f"Error: {self.error_message}"
        finally:
            self._finalize_main_thread(final_status)

    def _cleanup_scene_sync(self):
        try:
            if bpy.context.edit_object or (bpy.context.object and bpy.context.object.mode != 'OBJECT'):
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        for obj_name in self.resource_registry.get('temp_objects', []):
            try:
                obj = bpy.data.objects.get(obj_name)
                if obj:
                    mesh = obj.data
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if mesh and mesh.users == 0:
                        bpy.data.meshes.remove(mesh)
                    print(f"  ✅ Deleted: {obj_name}")
            except Exception as e:
                print(f"  ❌ Obj Removal Error: {e}")

        for mat_name in self.resource_registry.get('temp_materials', []):
            try:
                mat = bpy.data.materials.get(mat_name)
                if mat:
                    bpy.data.materials.remove(mat)
                print(f"  ✅ Deleted material: {mat_name}")
            except Exception as e:
                print(f"  ❌ Mat Removal Error: {e}")

        try:
            bpy.ops.object.select_all(action='DESELECT')
            for name in self.resource_registry.get('original_selected_objs', []):
                obj = bpy.data.objects.get(name)
                if obj:
                    obj.select_set(True)

            active_name = self.resource_registry.get('original_active_obj')
            if active_name:
                active_obj = bpy.data.objects.get(active_name)
                if active_obj:
                    bpy.context.view_layer.objects.active = active_obj
            print("  ✅ Selection and Activation restored")
        except Exception as e:
            print(f"  ❌ Selection Restoration Error: {e}")

    def _cleanup_non_blender_resources(self):
        import shutil

        for data in self.target_objects_data:
            try:
                if 'bm_copy' in data and data['bm_copy']:
                    data['bm_copy'].free()
            except Exception:
                pass

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                if self.debug_mode:
                    print(f"🐞 DEBUG MODE: Preserving directory {self.temp_dir}")
                elif os.path.basename(self.temp_dir).startswith("gemini_proj_"):
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    print(f"  ✅ Removed directory {self.temp_dir}")
            except Exception as e:
                print(f"  ❌ Directory cleanup error: {e}")

    def _finalize_main_thread(self, status_text=None):
        if self._finalized:
            return
        self._finalized = True

        print(f"[GEMINI] Finalizing thread on main thread: {self.name}")
        self._cleanup_scene_sync()
        self._cleanup_non_blender_resources()

        if hasattr(BlenderThreadManager, 'active_threads') and self in BlenderThreadManager.active_threads:
            BlenderThreadManager.active_threads.remove(self)

        from . import operators
        if operators.GEMINI_OT_texture_projection.current_thread == self:
            operators.GEMINI_OT_texture_projection.current_thread = None

        if status_text is None:
            if self.error_message:
                status_text = f"Error: {self.error_message}"
            elif self._stop_event.is_set():
                status_text = "Projection cancelled"
            else:
                status_text = "Ready to render"

        self._set_status_sync(status_text, False)

    def run(self):
        print("ProjectionRenderThread starting...")
        try:
            update_render_status(self.scene, "Sending projection to Gemini...", True)
            
            projection_prompt = f"{self.user_prompt}"
            
            print(f"🚀 Projection Prompt: '{projection_prompt}'")
            print(f"📸 Source Image: {self.source_path}")
            if self.reference_path:
                print(f"🖼 Reference Image: {self.reference_path}")

            image_data = None
            mime_type = "image/png"

            if self.source_image_override:
                print("🖼 Direct Image Mode...")
            elif self.bypass_api:
                print("🛡 Simulation Mode...")
                with open(self.sim_path, 'rb') as f:
                    image_data = f.read()
            else:
                # AI Call
                if self._stop_event.is_set(): return
                print("🌐 Calling Gemini API...")
                image_data, mime_type = self.api_client.generate_image(
                    depth_image_path=self.source_path,
                    user_prompt=projection_prompt,
                    reference_image_path=self.reference_path,
                    width=self.capture_width,
                    height=self.capture_height,
                    is_color_render=(self.input_source == 'COLOR')
                )
                print(f"✅ Gemini API responded: {len(image_data)} bytes ({mime_type})")
                
                # === DEBUG MODE: SAVE AI RESULT ===
                if self.debug_mode and image_data:
                    try:
                        result_path = os.path.join(self.temp_dir, "debug_output.png")
                        with open(result_path, 'wb') as f:
                            f.write(image_data)
                        print(f"✅ [DEBUG] AI Result saved to: {result_path}")
                    except Exception as e:
                        print(f"❌ [DEBUG] Failed to save AI result: {e}")

            if self._stop_event.is_set():
                return

            self._queue_main_thread_finalize(
                image_data=image_data,
                apply_result=(self.source_image_override is not None or image_data is not None)
            )
            
        except Exception as e:
            print(f"Projection thread error: {e}")
            self.error_message = str(e)
        finally:
            if not self._finalize_queued:
                self._queue_main_thread_finalize()
            

def stop_all_projection_threads():
    """Stop all active projection threads"""
    if hasattr(BlenderThreadManager, 'active_threads'):
        print(f"Stopping {len(BlenderThreadManager.active_threads)} active threads...")
        for thread in list(BlenderThreadManager.active_threads):
            thread.stop()
        BlenderThreadManager.active_threads.clear()
    else:
        print("Stopping 0 active threads (manager not initialized or empty)")

def stop_thread_manager():
    """Stop the thread manager (call on addon unregister)"""
    _thread_manager.stop_timer()

def reset_threading_state():
    """Reset thread/timer state after loading a new blend file."""
    stop_all_projection_threads()
    _thread_manager.stop_timer()
    while not _thread_manager.command_queue.empty():
        try:
            _thread_manager.command_queue.get_nowait()
        except Exception:
            break
