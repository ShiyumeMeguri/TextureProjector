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

def save_reference_image_temp(scene) -> str:
    """Save reference image from scene properties to temporary file"""
    try:
        import bpy
        import tempfile
        import os
        

        props = scene.gemini_render if hasattr(scene, 'gemini_render') else None
        if not props or not props.use_style_reference or not props.style_reference_image:
            return None
            
        reference_image = props.style_reference_image
        print(f"🎨 Saving reference image: {reference_image.name}")
        

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        

        saved_successfully = False
        
        # I method 1: For images without filepath (generated images) - use pixel data
        if not reference_image.filepath:
            print("Saving generated image via pixel data...")
            try:

                pixels = list(reference_image.pixels)
                width, height = reference_image.size
                
                # I convert to PIL Image and save
                try:
                    # I try to use PIL if available
                    from PIL import Image
                    try:
                        import numpy as np
                    except ImportError:
                        raise ImportError("NumPy required for PIL image processing")
                    
                    # I convert pixels to numpy array and reshape
                    pixel_array = np.array(pixels).reshape((height, width, reference_image.channels))
                    
                    # I convert to 0-255 range and uint8
                    if pixel_array.max() <= 1.0:
                        pixel_array = (pixel_array * 255).astype(np.uint8)
                    
                    # I handle different channel counts
                    if reference_image.channels == 4:  # RGBA
                        img = Image.fromarray(pixel_array, 'RGBA')
                    elif reference_image.channels == 3:  # RGB
                        img = Image.fromarray(pixel_array, 'RGB')
                    else:  # Grayscale
                        img = Image.fromarray(pixel_array[:,:,0], 'L')
                    
                    img.save(temp_path, 'PNG')
                    saved_successfully = True
                    print("Saved via PIL")
                    
                except ImportError:
                    print("PIL not available, trying Blender save_render...")

                    original_settings = {
                        'filepath': reference_image.filepath,
                        'file_format': reference_image.file_format
                    }
                    
                    reference_image.filepath_raw = temp_path
                    reference_image.file_format = 'PNG'
                    reference_image.save_render(temp_path)
                    

                    reference_image.filepath = original_settings['filepath']
                    reference_image.file_format = original_settings['file_format']
                    saved_successfully = True
                    print("Saved via Blender save_render")
                    
            except Exception as e:
                print(f"Pixel data method failed: {e}")
        
        # I method 2: For packed images
        elif reference_image.packed_file:
            print("📦 Reference image is packed, extracting...")
            try:
                with open(temp_path, 'wb') as f:
                    f.write(reference_image.packed_file.data)
                saved_successfully = True
                print("Saved from packed data")
            except Exception as e:
                print(f"Packed file method failed: {e}")
        
        # I method 3: For images with filepath
        elif reference_image.filepath:
            print(f" Copying reference from filepath...")
            try:
                import shutil
                abs_path = bpy.path.abspath(reference_image.filepath)
                if os.path.exists(abs_path):
                    shutil.copy2(abs_path, temp_path)
                    saved_successfully = True
                    print(f"Copied from: {abs_path}")
                else:
                    print(f"Reference file not found: {abs_path}")
            except Exception as e:
                print(f"Filepath method failed: {e}")
        

        if saved_successfully and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            print(f"Reference image saved to: {temp_path}")
            return temp_path
        else:
            print("Failed to save reference image")

            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            return None
            
    except Exception as e:
        print(f"Error saving reference image: {e}")
        return None

def load_result_image(image_data: bytes, image_name: str = "AI_Result", user_prompt: str = "", cam_data: dict = None) -> None:
    """Load result image into Blender and save to history (thread-safe)"""
    print(f" load_result_image wrapper called for {image_name}")
    execute_in_main_thread(_load_result_image_sync, image_data, image_name, user_prompt, cam_data)

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

class RenderThread(threading.Thread):
    """Background thread for AI rendering operations (DEPRECATED - use APIThread)"""
    
    def __init__(self, scene, depth_renderer, api_client, user_prompt: str):
        super().__init__(daemon=True)
        self.scene = scene
        self.depth_renderer = depth_renderer
        self.api_client = api_client
        self.user_prompt = user_prompt
        self._stop_event = threading.Event()
        print("RenderThread initialized (DEPRECATED)")
    
    def stop(self):
        """Request thread to stop"""
        print("Stop requested for RenderThread")
        self._stop_event.set()
    
    def run(self):
        """Main thread execution"""
        # I this is deprecated - should not be used
        print("RenderThread is deprecated, use APIThread instead")
        update_render_status(self.scene, "Deprecated render method used", False)

class APIThread(threading.Thread):
    """Background thread for API calls only (render happens in main thread)"""
    
    def __init__(self, scene, api_client, user_prompt: str, depth_path: str, cam_data: dict = None):
        super().__init__(daemon=True)
        self.scene = scene
        self.api_client = api_client
        self.user_prompt = user_prompt
        self.depth_path = depth_path
        self.cam_data = cam_data
        self._stop_event = threading.Event()
        print("APIThread initialized")
    
    def stop(self):
        """Request thread to stop"""
        print("Stop requested for APIThread")
        self._stop_event.set()
    
    def run(self):
        """Main thread execution - API calls only"""
        print(" APIThread starting execution...")
        
        try:
            if self._stop_event.is_set():
                print("Stopped before API call")
                return
            
            # I send to AI
            print("Step 1: Sending to Gemini AI...")
            update_render_status(self.scene, "Sending to AI...", True)
            

            reference_path = save_reference_image_temp(self.scene)
            

            props = self.scene.gemini_render if hasattr(self.scene, 'gemini_render') else None
            resolution = int(props.resolution) if props and hasattr(props, 'resolution') else 1024
            
            try:
                image_data, mime_type = self.api_client.generate_image(self.depth_path, self.user_prompt, reference_path, width=resolution, height=resolution)
                print(f"AI response received, image size: {len(image_data)} bytes")
            finally:

                if reference_path:
                    try:
                        import os
                        os.unlink(reference_path)
                        print(f"Reference temp file cleaned up")
                    except:
                        pass
            
            if self._stop_event.is_set():
                print("Stopped after AI response")
                return
            
            # I load result
            print("Step 2: Loading result into Blender...")
            update_render_status(self.scene, "📥 Loading result...", True)
            print(f"About to call load_result_image with user_prompt: '{self.user_prompt}' (length: {len(self.user_prompt) if self.user_prompt else 0})")
            load_result_image(image_data, "Gemini_AI_Result", self.user_prompt, self.cam_data)
            
            # Success
            print("🎉 AI render completed successfully!")
            update_render_status(self.scene, "AI render completed successfully!", False)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"API thread error: {error_msg}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            
            update_render_status(self.scene, error_msg, False)
            
        finally:
            print("API thread cleanup starting...")
            # I cleanup depth file if needed
            try:
                import os
                if os.path.exists(self.depth_path):
                    os.remove(self.depth_path)
                    print(f"Cleaned up depth file: {self.depth_path}")
            except Exception as cleanup_error:
                print(f"Cleanup warning: {cleanup_error}")
            print("APIThread finished")

class FullRenderThread(threading.Thread):
    """Background thread for full render pipeline with proper context handling"""
    
    def __init__(self, context, depth_renderer, api_client, user_prompt: str, cam_data: dict = None):
        super().__init__(daemon=True)
        # I store only what we need from context
        self.scene = context.scene
        self.view_layer = context.view_layer
        # I store window manager for render operations
        import bpy
        self.window_manager = bpy.context.window_manager
        
        self.depth_renderer = depth_renderer
        self.api_client = api_client
        self.user_prompt = user_prompt
        self.cam_data = cam_data
        self._stop_event = threading.Event()
        print("FullRenderThread initialized")
    
    def stop(self):
        """Request thread to stop"""
        print("Stop requested for FullRenderThread")
        self._stop_event.set()
    
    def run(self):
        """Main thread execution with proper context override"""
        print("FullRenderThread starting execution...")
        
        try:

            print("Step 1: Updating status to 'rendering depth'")
            update_render_status(self.scene, "Rendering depth map...", True)
            
            if self._stop_event.is_set():
                print("Stopped before depth render")
                return
            

            props = self.scene.gemini_render if hasattr(self.scene, 'gemini_render') else None
            render_mode = props.render_mode if props and hasattr(props, 'render_mode') else 'DEPTH'
            

            # ONLY use camera if it exists AND the view was in camera mode
            is_camera_view = self.cam_data.get('is_camera_view', False) if self.cam_data else False
            has_camera = self.scene.camera is not None and is_camera_view
            

            render_result = None
            depth_path = None
            
            if not has_camera:
                print("No camera found, using VIEWPORT FALLBACK...")
                
                def _do_viewport_fallback():
                    nonlocal render_result, depth_path
                    try:
                        ctx = get_view3d_context()
                        if not ctx:
                            raise Exception("No 3D viewport found for camera-less capture")
                        
                        # I use a temporary context override
                        with bpy.context.temp_override(**ctx):
                            if render_mode == 'DEPTH':
                                # I use our new viewport depth method
                                print("Rendering viewport depth (fallback)...")
                                depth_path = self.depth_renderer.render_depth_viewport(bpy.context)
                            else:
                                # I use OpenGL render for color (fallback)
                                print("Rendering viewport color (fallback)...")
                                import tempfile
                                temp_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                                
                                # I store and set render settings
                                orig_path = self.scene.render.filepath
                                orig_format = self.scene.render.image_settings.file_format
                                self.scene.render.filepath = temp_path
                                self.scene.render.image_settings.file_format = 'PNG'
                                

                                bpy.ops.render.opengl(write_still=True, view_context=True)
                                
                                # DEBUG SAVING
                                try:
                                    import os
                                    import shutil
                                    import time
                                    debug_dir = r"E:\Debug"
                                    if os.path.exists(debug_dir):
                                        debug_path = os.path.join(debug_dir, f"color_{int(time.time())}.png")
                                        shutil.copy2(temp_path, debug_path)
                                        print(f"🐞 Debug color saved to: {debug_path}")
                                except Exception as de:
                                    print(f" Debug color save failed: {de}")
                                
                                # Restore
                                self.scene.render.filepath = orig_path
                                self.scene.render.image_settings.file_format = orig_format
                                
                                depth_path = temp_path
                                
                            render_result = "success"
                            print(f"Viewport fallback completed: {depth_path}")
                            
                    except Exception as e:
                        render_result = f"error: {str(e)}"
                        print(f"Viewport fallback error: {str(e)}")
                
                execute_in_main_thread(_do_viewport_fallback)
                
            elif render_mode == 'DEPTH':
                # I depth Map (Mist) Mode
                print("Using DEPTH MAP (Mist) mode...")
                
                mist_start = props.mist_start if props else 5.0
                mist_depth = props.mist_depth if props else 25.0
                mist_falloff = props.mist_falloff if props and hasattr(props, 'mist_falloff') else 'LINEAR'
                
                def _do_safe_mist_render():
                    nonlocal render_result, depth_path
                    try:
                        depth_path = self.depth_renderer.render_depth_map_mist(
                            self.scene, mist_start, mist_depth, mist_falloff
                        )
                        render_result = "success"
                        print(f"Mist depth render completed: {depth_path}")
                        
                    except Exception as e:
                        render_result = f"error: {str(e)}"
                        print(f"Mist render error in main thread: {str(e)}")
                

                print("Executing mist render in main thread for safety...")
                execute_in_main_thread(_do_safe_mist_render)
                
            else:
                # I regular Eevee Render Mode
                print("Using REGULAR RENDER (Eevee) mode...")
                
                def _do_safe_eevee_render():
                    nonlocal render_result, depth_path
                    try:
                        # I use regular render method
                        depth_path = self.depth_renderer.render_regular_eevee(self.scene)
                        render_result = "success"
                        print(f"Regular Eevee render completed: {depth_path}")
                        
                    except Exception as e:
                        render_result = f"error: {str(e)}"
                        print(f"Eevee render error in main thread: {str(e)}")
                

                print("Executing regular Eevee render in main thread...")
                execute_in_main_thread(_do_safe_eevee_render)
            
            # I wait for render completion
            import time
            timeout = 180  # 3 minutes timeout for mist render
            elapsed = 0
            while render_result is None and elapsed < timeout and not self._stop_event.is_set():
                time.sleep(0.1)
                elapsed += 0.1
            
            if self._stop_event.is_set():
                print("Stopped during mist render")
                return
            
            if render_result is None:
                raise Exception("Mist render timeout - took longer than 3 minutes")
            elif render_result.startswith("error:"):
                raise Exception(f"Mist render failed: {render_result[7:]}")
            
            if not depth_path:
                raise Exception("No depth path returned from mist render")
            
            # I continue with AI processing
            print("Step 2: Sending to Gemini AI...")
            update_render_status(self.scene, "Sending to AI...", True)
            

            reference_path = save_reference_image_temp(self.scene)
            
            # I determine if using color render mode
            is_color_render = (render_mode == 'EEVEE')
            

            resolution = int(props.resolution) if props and hasattr(props, 'resolution') else 1024
            print(f"Using resolution: {resolution}x{resolution}")
            
            try:
                image_data, mime_type = self.api_client.generate_image(depth_path, self.user_prompt, reference_path, is_color_render, width=resolution, height=resolution)
                print(f"AI response received, image size: {len(image_data)} bytes")
            finally:

                if reference_path:
                    try:
                        import os
                        os.unlink(reference_path)
                        print(f"Reference temp file cleaned up")
                    except:
                        pass
                        
                # CRITICAL: Clean up depth temp files after API usage
                try:
                    self.depth_renderer.cleanup_temp_files()
                    print("Depth temp files cleaned up after API usage")
                except Exception as cleanup_error:
                    print(f"Depth cleanup warning: {cleanup_error}")
            
            if self._stop_event.is_set():
                print("Stopped after AI response")
                return
            
            # I load result
            print("Step 3: Loading result into Blender...")
            update_render_status(self.scene, "📥 Loading result...", True)
            print(f"About to call load_result_image with user_prompt: '{self.user_prompt}' (length: {len(self.user_prompt) if self.user_prompt else 0})")
            load_result_image(image_data, "Gemini_AI_Result", self.user_prompt, self.cam_data)
            
            # Success
            print("🎉 AI render completed successfully!")
            update_render_status(self.scene, "AI render completed successfully!", False)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Full render thread error: {error_msg}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            
            update_render_status(self.scene, error_msg, False)
            
        finally:
            print("Full render thread cleanup starting...")
            # Note: Depth temp files are now cleaned up after API usage, not here
            print("FullRenderThread finished")
    
    def _render_depth_with_override(self, override_context):
        """Render depth with context override"""
        import bpy
        
        scene = override_context['scene']
        view_layer = override_context['view_layer']
        

        print("Setting up depth render...")
        

        original_filepath = scene.render.filepath
        original_use_nodes = scene.use_nodes
        original_file_format = scene.render.image_settings.file_format
        original_color_mode = scene.render.image_settings.color_mode
        

        import tempfile, os
        temp_dir = tempfile.mkdtemp(prefix="gemini_depth_")
        depth_file_path = os.path.join(temp_dir, "depth")
        
        try:
            # I configure render settings
            scene.render.filepath = depth_file_path
            scene.render.image_settings.file_format = 'PNG'
            scene.render.image_settings.color_mode = 'BW'
            

            scene.use_nodes = True
            tree = scene.node_tree
            tree.nodes.clear()
            

            render_layers = tree.nodes.new(type='CompositorNodeRLayers')
            render_layers.location = (0, 0)
            
            output_node = tree.nodes.new(type='CompositorNodeOutputFile')
            output_node.location = (300, 0)
            output_node.base_path = temp_dir
            output_node.file_slots[0].path = "depth"
            output_node.format.file_format = 'PNG'
            output_node.format.color_mode = 'BW'
            

            view_layer.use_pass_z = True
            

            if 'Depth' in render_layers.outputs:
                tree.links.new(render_layers.outputs['Depth'], output_node.inputs[0])
            elif 'Z' in render_layers.outputs:
                tree.links.new(render_layers.outputs['Z'], output_node.inputs[0])
            else:
                print("No depth pass found, using Image")
                tree.links.new(render_layers.outputs['Image'], output_node.inputs[0])
            
            print("Starting render operation...")
            
            # I use render operator with override
            with bpy.context.temp_override(**override_context):
                bpy.ops.render.render(write_still=True)
            
            print("Render operation completed")
            

            possible_paths = [
                os.path.join(temp_dir, "depth0001.png"),
                os.path.join(temp_dir, "depth.png"),
                depth_file_path + ".png"
            ]
            
            actual_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    actual_path = path
                    break
            
            if not actual_path:
                raise Exception("Depth render file not found")
            

            normalized_path = self.depth_renderer._normalize_depth_map(
                actual_path, self.normalize_mode, self.clip_start, self.clip_end
            )
            
            return normalized_path
            
        finally:

            scene.render.filepath = original_filepath
            scene.use_nodes = original_use_nodes
            scene.render.image_settings.file_format = original_file_format
            scene.render.image_settings.color_mode = original_color_mode

class ProjectionRenderThread(threading.Thread):
    """Background thread for AI Texture Projection pipeline"""
    
    def __init__(self, context, api_client, user_prompt, source_path, sim_path, target_objects_data, image_node_name, material_name, do_bake, bypass_api=False, mask_repair_data=None, input_source='COLOR', debug_mode=False, source_image_override=None, cam_data=None):
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
        self._stop_event = threading.Event()
        print(f"[GEMINI] ProjectionRenderThread initialized (bypass_api={bypass_api}, input={input_source})")
    
    def stop(self):
        self._stop_event.set()
        
    def run(self):
        print("ProjectionRenderThread starting...")
        try:
            update_render_status(self.scene, "Sending projection to Gemini...", True)
            
            from . import operators # I local import to avoid circularity
            
            projection_prompt = f"{self.user_prompt}"
            
            # Resolution
            props = self.scene.gemini_render
            resolution = int(props.resolution)
            
            # DEBUG: Save exact intended AI input
            if self.debug_mode:
                try:
                    import shutil
                    blend_path = bpy.data.filepath
                    base_debug_dir = os.path.join(os.path.dirname(blend_path), "textures") if blend_path else os.path.join(tempfile.gettempdir(), "textures")
                    if os.path.exists(base_debug_dir):
                        shutil.copy2(self.source_path, os.path.join(base_debug_dir, "input.png"))
                        print(f"🐞 Debug input confirmed at: {base_debug_dir}")
                except Exception as de:
                    print(f" Debug input sync failed: {de}")

            if self.source_image_override:
                print(f"🖼 Direct Image Mode: Using '{self.source_image_override.name}' directly...")
                # I we need the image data as bytes for the internal loading logic
                # However, the internal loading logic _load_result_image_sync expects bytes.

                # Let's adjust the _apply_result logic to handle an existing image.
                image_data = None 
                mime_type = "image/png" # Dummy
            elif self.bypass_api:
                print("🛡 Simulation Mode: Bypassing AI call, using local grid capture...")
                # I in simulation mode, use the grid capture from sim_path
                with open(self.sim_path, 'rb') as f:
                    image_data = f.read()
                mime_type = "image/png"
            else:
                print(f" Calling AI to generate texture (Input: {self.input_source})...")
                
                # Determine if input is color (for AI API)
                is_color = (self.input_source == 'COLOR')
                
                # I send ONLY ONE image to API - 1:1 Mapping
                image_data, mime_type = self.api_client.generate_image(
                    depth_image_path=self.source_path,
                    user_prompt=projection_prompt,
                    reference_image_path=None, 
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
                        res_path = os.path.join(base_debug_dir, "output.png")
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
