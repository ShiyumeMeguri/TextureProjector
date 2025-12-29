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
        
        # Register timer if not already registered
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
                    # Continue processing other commands
                    
            # Return interval for next check
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

# Global thread manager instance
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
            print(f"[GEMINI] Updating status: {status_text}")
            if hasattr(scene, 'gemini_render'):
                props = scene.gemini_render
                props.status_text = status_text
                if is_rendering is not None:
                    props.is_rendering = is_rendering
                    print(f"[GEMINI] Set is_rendering = {is_rendering}")
                
                # Force redraw all areas
                try:
                    import bpy
                    for window in bpy.context.window_manager.windows:
                        for area in window.screen.areas:
                            if area.type == 'VIEW_3D':
                                area.tag_redraw()
                except Exception as redraw_error:
                    print(f"[GEMINI] Redraw warning: {redraw_error}")
                    
                print("[GEMINI] Status updated successfully")
            else:
                print("[GEMINI] Scene has no gemini_render property")
        except Exception as e:
            print(f"[GEMINI] Error updating status: {e}")
            import traceback
            print(f"[GEMINI] Status update traceback:\n{traceback.format_exc()}")
    
    execute_in_main_thread(_update)

def save_reference_image_temp(scene) -> str:
    """Save reference image from scene properties to temporary file"""
    try:
        import bpy
        import tempfile
        import os
        
        # Get scene properties
        props = scene.gemini_render if hasattr(scene, 'gemini_render') else None
        if not props or not props.use_style_reference or not props.style_reference_image:
            return None
            
        reference_image = props.style_reference_image
        print(f"🎨 [GEMINI] Saving reference image: {reference_image.name}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Save image using different methods based on image type
        saved_successfully = False
        
        # Method 1: For images without filepath (generated images) - use pixel data
        if not reference_image.filepath:
            print("[GEMINI] Saving generated image via pixel data...")
            try:
                # Get pixel data directly
                pixels = list(reference_image.pixels)
                width, height = reference_image.size
                
                # Convert to PIL Image and save
                try:
                    # Try to use PIL if available
                    from PIL import Image
                    try:
                        import numpy as np
                    except ImportError:
                        raise ImportError("NumPy required for PIL image processing")
                    
                    # Convert pixels to numpy array and reshape
                    pixel_array = np.array(pixels).reshape((height, width, reference_image.channels))
                    
                    # Convert to 0-255 range and uint8
                    if pixel_array.max() <= 1.0:
                        pixel_array = (pixel_array * 255).astype(np.uint8)
                    
                    # Handle different channel counts
                    if reference_image.channels == 4:  # RGBA
                        img = Image.fromarray(pixel_array, 'RGBA')
                    elif reference_image.channels == 3:  # RGB
                        img = Image.fromarray(pixel_array, 'RGB')
                    else:  # Grayscale
                        img = Image.fromarray(pixel_array[:,:,0], 'L')
                    
                    img.save(temp_path, 'PNG')
                    saved_successfully = True
                    print("[GEMINI] Saved via PIL")
                    
                except ImportError:
                    print("[GEMINI] PIL not available, trying Blender save_render...")
                    # Fallback to Blender's save_render
                    original_settings = {
                        'filepath': reference_image.filepath,
                        'file_format': reference_image.file_format
                    }
                    
                    reference_image.filepath_raw = temp_path
                    reference_image.file_format = 'PNG'
                    reference_image.save_render(temp_path)
                    
                    # Restore original settings
                    reference_image.filepath = original_settings['filepath']
                    reference_image.file_format = original_settings['file_format']
                    saved_successfully = True
                    print("[GEMINI] Saved via Blender save_render")
                    
            except Exception as e:
                print(f"[GEMINI] Pixel data method failed: {e}")
        
        # Method 2: For packed images
        elif reference_image.packed_file:
            print("📦 [GEMINI] Reference image is packed, extracting...")
            try:
                with open(temp_path, 'wb') as f:
                    f.write(reference_image.packed_file.data)
                saved_successfully = True
                print("[GEMINI] Saved from packed data")
            except Exception as e:
                print(f"[GEMINI] Packed file method failed: {e}")
        
        # Method 3: For images with filepath
        elif reference_image.filepath:
            print(f"📁 [GEMINI] Copying reference from filepath...")
            try:
                import shutil
                abs_path = bpy.path.abspath(reference_image.filepath)
                if os.path.exists(abs_path):
                    shutil.copy2(abs_path, temp_path)
                    saved_successfully = True
                    print(f"[GEMINI] Copied from: {abs_path}")
                else:
                    print(f"[GEMINI] Reference file not found: {abs_path}")
            except Exception as e:
                print(f"[GEMINI] Filepath method failed: {e}")
        
        # Check if saving was successful
        if saved_successfully and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            print(f"[GEMINI] Reference image saved to: {temp_path}")
            return temp_path
        else:
            print("[GEMINI] Failed to save reference image")
            # Clean up failed temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            return None
            
    except Exception as e:
        print(f"[GEMINI] Error saving reference image: {e}")
        return None

def load_result_image(image_data: bytes, image_name: str = "AI_Result", user_prompt: str = "", cam_data: dict = None) -> None:
    """Load result image into Blender and save to history (thread-safe)"""
    print(f"🚀 [GEMINI] load_result_image wrapper called for {image_name}")
    execute_in_main_thread(_load_result_image_sync, image_data, image_name, user_prompt, cam_data)

def _load_result_image_sync(image_data: bytes, image_name: str = "AI_Result", user_prompt: str = "", cam_data: dict = None) -> Any:
    """Synchronous version of image loading. MUST be called from the main thread.
    Returns the loaded Image object (either the history one or the Render Result copy)."""
    print(f"📥 [GEMINI] _load_result_image_sync starting for {image_name}")
    try:
        import tempfile
        import os
        import datetime
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(image_data)
            temp_path = f.name
        
        try:
            # Load image into Blender
            if image_name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[image_name])
            
            img = bpy.data.images.load(temp_path)
            img.name = image_name
            
            # Keep original image for history
            permanent_image_for_history = None
            if user_prompt:
                permanent_name = f"AI_Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                img.name = permanent_name
                img.pack()
                permanent_image_for_history = img
                print(f"[GEMINI] History image: {permanent_name}")
            
            # Update Render Result
            render_result = bpy.data.images.get('Render Result')
            if render_result:
                bpy.data.images.remove(render_result)
            
            render_result = img.copy()
            render_result.name = 'Render Result'
            
            # Update UI
            for area in bpy.context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    for space in area.spaces:
                        if space.type == 'IMAGE_EDITOR':
                            space.image = render_result
                    area.tag_redraw()
            
            # Handle Gallery history entries
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
                    
                    # Store Camera Data if available
                    if cam_data:
                        try:
                            item.cam_location = cam_data.get('location', (0,0,0))
                            item.cam_rotation = cam_data.get('rotation', (1,0,0,0))
                            item.cam_lens = cam_data.get('lens', 50.0)
                            item.view_distance = cam_data.get('view_distance', 10.0)
                            item.is_camera_view = cam_data.get('is_camera_view', False)
                            print(f"📷 [GEMINI] Camera data stored in history item")
                        except Exception as ce:
                            print(f"⚠️ [GEMINI] Failed to store camera data: {ce}")
                    
                    # Cleanup old history
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
        print(f"❌ [GEMINI] Error in _load_result_image_sync: {e}")
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
        print("[GEMINI] RenderThread initialized (DEPRECATED)")
    
    def stop(self):
        """Request thread to stop"""
        print("[GEMINI] Stop requested for RenderThread")
        self._stop_event.set()
    
    def run(self):
        """Main thread execution"""
        # This is deprecated - should not be used
        print("[GEMINI] RenderThread is deprecated, use APIThread instead")
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
        print("[GEMINI] APIThread initialized")
    
    def stop(self):
        """Request thread to stop"""
        print("[GEMINI] Stop requested for APIThread")
        self._stop_event.set()
    
    def run(self):
        """Main thread execution - API calls only"""
        print("🚀 [GEMINI] APIThread starting execution...")
        
        try:
            if self._stop_event.is_set():
                print("[GEMINI] Stopped before API call")
                return
            
            # Send to AI
            print("[GEMINI] Step 1: Sending to Gemini AI...")
            update_render_status(self.scene, "Sending to AI...", True)
            
            # Check for reference image
            reference_path = save_reference_image_temp(self.scene)
            
            # Get resolution
            props = self.scene.gemini_render if hasattr(self.scene, 'gemini_render') else None
            resolution = int(props.resolution) if props and hasattr(props, 'resolution') else 1024
            
            try:
                image_data, mime_type = self.api_client.generate_image(self.depth_path, self.user_prompt, reference_path, width=resolution, height=resolution)
                print(f"[GEMINI] AI response received, image size: {len(image_data)} bytes")
            finally:
                # Clean up reference temp file
                if reference_path:
                    try:
                        import os
                        os.unlink(reference_path)
                        print(f"[GEMINI] Reference temp file cleaned up")
                    except:
                        pass
            
            if self._stop_event.is_set():
                print("[GEMINI] Stopped after AI response")
                return
            
            # Load result
            print("[GEMINI] Step 2: Loading result into Blender...")
            update_render_status(self.scene, "📥 Loading result...", True)
            print(f"[GEMINI] About to call load_result_image with user_prompt: '{self.user_prompt}' (length: {len(self.user_prompt) if self.user_prompt else 0})")
            load_result_image(image_data, "Gemini_AI_Result", self.user_prompt, self.cam_data)
            
            # Success
            print("🎉 [GEMINI] AI render completed successfully!")
            update_render_status(self.scene, "AI render completed successfully!", False)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[GEMINI] API thread error: {error_msg}")
            print(f"[GEMINI] Exception type: {type(e).__name__}")
            import traceback
            print(f"[GEMINI] Full traceback:\n{traceback.format_exc()}")
            
            update_render_status(self.scene, error_msg, False)
            
        finally:
            print("[GEMINI] API thread cleanup starting...")
            # Cleanup depth file if needed
            try:
                import os
                if os.path.exists(self.depth_path):
                    os.remove(self.depth_path)
                    print(f"[GEMINI] Cleaned up depth file: {self.depth_path}")
            except Exception as cleanup_error:
                print(f"[GEMINI] Cleanup warning: {cleanup_error}")
            print("[GEMINI] APIThread finished")

class FullRenderThread(threading.Thread):
    """Background thread for full render pipeline with proper context handling"""
    
    def __init__(self, context, depth_renderer, api_client, user_prompt: str, cam_data: dict = None):
        super().__init__(daemon=True)
        # Store only what we need from context
        self.scene = context.scene
        self.view_layer = context.view_layer
        # Store window manager for render operations
        import bpy
        self.window_manager = bpy.context.window_manager
        
        self.depth_renderer = depth_renderer
        self.api_client = api_client
        self.user_prompt = user_prompt
        self.cam_data = cam_data
        self._stop_event = threading.Event()
        print("[GEMINI] FullRenderThread initialized")
    
    def stop(self):
        """Request thread to stop"""
        print("[GEMINI] Stop requested for FullRenderThread")
        self._stop_event.set()
    
    def run(self):
        """Main thread execution with proper context override"""
        print("[GEMINI] FullRenderThread starting execution...")
        
        try:
            # Update status
            print("[GEMINI] Step 1: Updating status to 'rendering depth'")
            update_render_status(self.scene, "Rendering depth map...", True)
            
            if self._stop_event.is_set():
                print("[GEMINI] Stopped before depth render")
                return
            
            # Get render mode from scene properties
            props = self.scene.gemini_render if hasattr(self.scene, 'gemini_render') else None
            render_mode = props.render_mode if props and hasattr(props, 'render_mode') else 'DEPTH'
            
            # Check for camera to determine if we need viewport fallback
            # ONLY use camera if it exists AND the view was in camera mode
            is_camera_view = self.cam_data.get('is_camera_view', False) if self.cam_data else False
            has_camera = self.scene.camera is not None and is_camera_view
            
            # Execute render based on mode
            render_result = None
            depth_path = None
            
            if not has_camera:
                print("[GEMINI] No camera found, using VIEWPORT FALLBACK...")
                
                def _do_viewport_fallback():
                    nonlocal render_result, depth_path
                    try:
                        ctx = get_view3d_context()
                        if not ctx:
                            raise Exception("No 3D viewport found for camera-less capture")
                        
                        # Use a temporary context override
                        with bpy.context.temp_override(**ctx):
                            if render_mode == 'DEPTH':
                                # Use our new viewport depth method
                                print("[GEMINI] Rendering viewport depth (fallback)...")
                                depth_path = self.depth_renderer.render_depth_viewport(bpy.context)
                            else:
                                # Use OpenGL render for color (fallback)
                                print("[GEMINI] Rendering viewport color (fallback)...")
                                import tempfile
                                temp_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                                
                                # Store and set render settings
                                orig_path = self.scene.render.filepath
                                orig_format = self.scene.render.image_settings.file_format
                                self.scene.render.filepath = temp_path
                                self.scene.render.image_settings.file_format = 'PNG'
                                
                                # Execute viewport render
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
                                        print(f"🐞 [GEMINI] Debug color saved to: {debug_path}")
                                except Exception as de:
                                    print(f"⚠️ [GEMINI] Debug color save failed: {de}")
                                
                                # Restore
                                self.scene.render.filepath = orig_path
                                self.scene.render.image_settings.file_format = orig_format
                                
                                depth_path = temp_path
                                
                            render_result = "success"
                            print(f"[GEMINI] Viewport fallback completed: {depth_path}")
                            
                    except Exception as e:
                        render_result = f"error: {str(e)}"
                        print(f"[GEMINI] Viewport fallback error: {str(e)}")
                
                execute_in_main_thread(_do_viewport_fallback)
                
            elif render_mode == 'DEPTH':
                # Depth Map (Mist) Mode
                print("[GEMINI] Using DEPTH MAP (Mist) mode...")
                
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
                        print(f"[GEMINI] Mist depth render completed: {depth_path}")
                        
                    except Exception as e:
                        render_result = f"error: {str(e)}"
                        print(f"[GEMINI] Mist render error in main thread: {str(e)}")
                
                # Execute mist render in main thread
                print("[GEMINI] Executing mist render in main thread for safety...")
                execute_in_main_thread(_do_safe_mist_render)
                
            else:
                # Regular Eevee Render Mode
                print("[GEMINI] Using REGULAR RENDER (Eevee) mode...")
                
                def _do_safe_eevee_render():
                    nonlocal render_result, depth_path
                    try:
                        # Use regular render method
                        depth_path = self.depth_renderer.render_regular_eevee(self.scene)
                        render_result = "success"
                        print(f"[GEMINI] Regular Eevee render completed: {depth_path}")
                        
                    except Exception as e:
                        render_result = f"error: {str(e)}"
                        print(f"[GEMINI] Eevee render error in main thread: {str(e)}")
                
                # Execute eevee render in main thread
                print("[GEMINI] Executing regular Eevee render in main thread...")
                execute_in_main_thread(_do_safe_eevee_render)
            
            # Wait for render completion
            import time
            timeout = 180  # 3 minutes timeout for mist render
            elapsed = 0
            while render_result is None and elapsed < timeout and not self._stop_event.is_set():
                time.sleep(0.1)
                elapsed += 0.1
            
            if self._stop_event.is_set():
                print("[GEMINI] Stopped during mist render")
                return
            
            if render_result is None:
                raise Exception("Mist render timeout - took longer than 3 minutes")
            elif render_result.startswith("error:"):
                raise Exception(f"Mist render failed: {render_result[7:]}")
            
            if not depth_path:
                raise Exception("No depth path returned from mist render")
            
            # Continue with AI processing
            print("[GEMINI] Step 2: Sending to Gemini AI...")
            update_render_status(self.scene, "Sending to AI...", True)
            
            # Check for reference image
            reference_path = save_reference_image_temp(self.scene)
            
            # Determine if using color render mode
            is_color_render = (render_mode == 'EEVEE')
            
            # Get resolution
            resolution = int(props.resolution) if props and hasattr(props, 'resolution') else 1024
            print(f"[GEMINI] Using resolution: {resolution}x{resolution}")
            
            try:
                image_data, mime_type = self.api_client.generate_image(depth_path, self.user_prompt, reference_path, is_color_render, width=resolution, height=resolution)
                print(f"[GEMINI] AI response received, image size: {len(image_data)} bytes")
            finally:
                # Clean up reference temp file
                if reference_path:
                    try:
                        import os
                        os.unlink(reference_path)
                        print(f"[GEMINI] Reference temp file cleaned up")
                    except:
                        pass
                        
                # CRITICAL: Clean up depth temp files after API usage
                try:
                    self.depth_renderer.cleanup_temp_files()
                    print("[GEMINI] Depth temp files cleaned up after API usage")
                except Exception as cleanup_error:
                    print(f"[GEMINI] Depth cleanup warning: {cleanup_error}")
            
            if self._stop_event.is_set():
                print("[GEMINI] Stopped after AI response")
                return
            
            # Load result
            print("[GEMINI] Step 3: Loading result into Blender...")
            update_render_status(self.scene, "📥 Loading result...", True)
            print(f"[GEMINI] About to call load_result_image with user_prompt: '{self.user_prompt}' (length: {len(self.user_prompt) if self.user_prompt else 0})")
            load_result_image(image_data, "Gemini_AI_Result", self.user_prompt, self.cam_data)
            
            # Success
            print("🎉 [GEMINI] AI render completed successfully!")
            update_render_status(self.scene, "AI render completed successfully!", False)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[GEMINI] Full render thread error: {error_msg}")
            print(f"[GEMINI] Exception type: {type(e).__name__}")
            import traceback
            print(f"[GEMINI] Full traceback:\n{traceback.format_exc()}")
            
            update_render_status(self.scene, error_msg, False)
            
        finally:
            print("[GEMINI] Full render thread cleanup starting...")
            # Note: Depth temp files are now cleaned up after API usage, not here
            print("[GEMINI] FullRenderThread finished")
    
    def _render_depth_with_override(self, override_context):
        """Render depth with context override"""
        import bpy
        
        scene = override_context['scene']
        view_layer = override_context['view_layer']
        
        # Setup scene
        print("[GEMINI] Setting up depth render...")
        
        # Store original settings
        original_filepath = scene.render.filepath
        original_use_nodes = scene.use_nodes
        original_file_format = scene.render.image_settings.file_format
        original_color_mode = scene.render.image_settings.color_mode
        
        # Create temp directory
        import tempfile, os
        temp_dir = tempfile.mkdtemp(prefix="gemini_depth_")
        depth_file_path = os.path.join(temp_dir, "depth")
        
        try:
            # Configure render settings
            scene.render.filepath = depth_file_path
            scene.render.image_settings.file_format = 'PNG'
            scene.render.image_settings.color_mode = 'BW'
            
            # Setup compositor
            scene.use_nodes = True
            tree = scene.node_tree
            tree.nodes.clear()
            
            # Create nodes
            render_layers = tree.nodes.new(type='CompositorNodeRLayers')
            render_layers.location = (0, 0)
            
            output_node = tree.nodes.new(type='CompositorNodeOutputFile')
            output_node.location = (300, 0)
            output_node.base_path = temp_dir
            output_node.file_slots[0].path = "depth"
            output_node.format.file_format = 'PNG'
            output_node.format.color_mode = 'BW'
            
            # Enable depth pass
            view_layer.use_pass_z = True
            
            # Connect depth output
            if 'Depth' in render_layers.outputs:
                tree.links.new(render_layers.outputs['Depth'], output_node.inputs[0])
            elif 'Z' in render_layers.outputs:
                tree.links.new(render_layers.outputs['Z'], output_node.inputs[0])
            else:
                print("[GEMINI] No depth pass found, using Image")
                tree.links.new(render_layers.outputs['Image'], output_node.inputs[0])
            
            print("[GEMINI] Starting render operation...")
            
            # Use render operator with override
            with bpy.context.temp_override(**override_context):
                bpy.ops.render.render(write_still=True)
            
            print("[GEMINI] Render operation completed")
            
            # Find output file
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
            
            # Normalize depth map
            normalized_path = self.depth_renderer._normalize_depth_map(
                actual_path, self.normalize_mode, self.clip_start, self.clip_end
            )
            
            return normalized_path
            
        finally:
            # Restore settings
            scene.render.filepath = original_filepath
            scene.use_nodes = original_use_nodes
            scene.render.image_settings.file_format = original_file_format
            scene.render.image_settings.color_mode = original_color_mode

class ProjectionRenderThread(threading.Thread):
    """Background thread for AI Texture Projection pipeline"""
    
    def __init__(self, context, api_client, user_prompt, source_path, sim_path, target_objects_data, image_node_name, material_name, do_bake, bypass_api=False, mask_repair_data=None, projection_source='DEPTH', debug_mode=False, source_image_override=None, cam_data=None):
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
        self.mask_repair_data = mask_repair_data  # Contains temp objects, materials, original textures
        self.projection_source = projection_source
        self.debug_mode = debug_mode
        self.source_image_override = source_image_override # Blender Image object
        self.cam_data = cam_data
        self._stop_event = threading.Event()
        print(f"[GEMINI) ProjectionRenderThread initialized (mask_repair={mask_repair_data is not None}, source={projection_source}, override={source_image_override is not None})")
    
    def stop(self):
        self._stop_event.set()
        
    def run(self):
        print("[GEMINI] ProjectionRenderThread starting...")
        try:
            update_render_status(self.scene, "Sending projection to Gemini...", True)
            
            from . import operators # Local import to avoid circularity
            
            projection_prompt = f"Project this into a texture: {self.user_prompt}"
            
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
                        print(f"🐞 [GEMINI] Debug input confirmed at: {base_debug_dir}")
                except Exception as de:
                    print(f"⚠️ [GEMINI] Debug input sync failed: {de}")

            if self.source_image_override:
                print(f"🖼️ [GEMINI] Direct Image Mode: Using '{self.source_image_override.name}' directly...")
                # We need the image data as bytes for the internal loading logic
                # However, the internal loading logic _load_result_image_sync expects bytes.
                # If we already have the image in Blender, we can just use it.
                # Let's adjust the _apply_result logic to handle an existing image.
                image_data = None 
                mime_type = "image/png" # Dummy
            elif self.bypass_api:
                print("🛡️ [GEMINI] Simulation Mode: Bypassing AI call, using local grid capture...")
                # In simulation mode, use the grid capture from sim_path
                with open(self.sim_path, 'rb') as f:
                    image_data = f.read()
                mime_type = "image/png"
            else:
                print(f"🚀 [GEMINI] Calling AI to generate texture (Source: {self.projection_source})...")
                
                # In Phase 5, source_path is always the selected AI source
                is_color = (self.projection_source == 'COLOR')
                
                # Send ONLY ONE image to API - 1:1 Mapping
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
                        print(f"🐞 [GEMINI] Debug output saved to: {res_path}")
                except Exception as de:
                    print(f"⚠️ [GEMINI] Debug result save failed: {de}")

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
                        # Synchronous version to get the Image object IMMEDIATELY
                        res_img = _load_result_image_sync(image_data, "Gemini_Projection_Result", self.user_prompt, self.cam_data)
                    
                    if not res_img:
                         print("❌ [GEMINI] Failed to load/retrieve projection result image")
                         return

                    # Update base material for preview
                    material = bpy.data.materials.get(self.material_name)
                    if material:
                        node = material.node_tree.nodes.get(self.image_node_name)
                        if node:
                            node.image = res_img

                    # 2. Baking logic
                    if self.do_bake:
                        update_render_status(self.scene, "Baking projection (Blender Native)...", True)
                        
                        # AGGRESSIVE POST-BAKE RESTORATION HELPER
                        def _finalize_object_materials(target_obj, target_img, dest_uv_name, search_img=None, node_name=None):
                            if not target_obj or not target_obj.data or not hasattr(target_obj.data, 'materials'):
                                return
                                
                            print(f"🔍 [GEMINI] Aggressively finalizing materials for {target_obj.name} (Target UV: {dest_uv_name})")
                            for slot in target_obj.material_slots:
                                if slot.material and slot.material.use_nodes:
                                    m_nodes = slot.material.node_tree.nodes
                                    m_links = slot.material.node_tree.links
                                    
                                    # Force specific restoration for ANY node matching our criteria
                                    for node in m_nodes:
                                        is_match = False
                                        if node.type == 'TEX_IMAGE':
                                            # Match by direct reference
                                            if node.image and (node.image == target_img or (search_img and node.image == search_img)):
                                                is_match = True
                                            # Match by name
                                            elif node.name == node_name:
                                                is_match = True
                                            # Match by image name match (fuzzy backup)
                                            elif target_img and node.image and node.image.name == target_img.name:
                                                is_match = True
                                        
                                        if is_match:
                                            # 1. Update image to the final baked result
                                            if target_img:
                                                node.image = target_img
                                            
                                            # 2. Fix UV mapping node
                                            uv_node = None
                                            if node.inputs['Vector'].is_linked:
                                                from_node = node.inputs['Vector'].links[0].from_node
                                                if from_node.type == 'UV_MAP':
                                                    uv_node = from_node
                                            
                                            # If no UV Map node linked, create one
                                            if not uv_node:
                                                print(f"🔗 [GEMINI] Linking new UV Map node to {node.name} in {slot.material.name}")
                                                # Clear existing links to vector
                                                for l in node.inputs['Vector'].links:
                                                    m_links.remove(l)
                                                uv_node = m_nodes.new('ShaderNodeUVMap')
                                                m_links.new(uv_node.outputs['UV'], node.inputs['Vector'])
                                            
                                            # 3. FORCE SET THE UV MAP
                                            if uv_node:
                                                uv_node.uv_map = dest_uv_name
                                                print(f"✅ [GEMINI] Material '{slot.material.name}' node '{node.name}' finalized (UV: {dest_uv_name})")

                            # Also restore active render layer on the mesh itself
                            if dest_uv_name in target_obj.data.uv_layers:
                                target_obj.data.uv_layers.active = target_obj.data.uv_layers[dest_uv_name]
                                target_obj.data.uv_layers[dest_uv_name].active_render = True

                        for data in self.target_objects_data:
                            obj = bpy.data.objects.get(data['object_name'])
                            if not obj: continue
                            
                            # ============================================================
                            # STRICT MASK REPAIR GUARD
                            # ============================================================
                            if self.mask_repair_data:
                                # In Repair Mode, we MUST only process items that are explicitly identified as repair sources.
                                # These items MUST have 'original_object_name' set.
                                original_obj_name = data.get('original_object_name')
                                
                                if not original_obj_name:
                                    print(f"🛡️ [GEMINI] Mask Repair Mode Active: Skipping non-repair object '{obj.name}' (Guard against global bake leak)")
                                    continue
                                    
                                # Double check: Ensure we are processing the TEMP object (source), not the original
                                # The Temp object usually has '_MaskTemp' in name, or we check against our list
                                if obj.name not in self.mask_repair_data.get('temp_objects', []):
                                     # Careful: if the name check logic in operators.py was loose, we might have mismatch.
                                     # But generally, rely on original_object_name being present.
                                     pass

                                print(f"[GEMINI] Mask Repair Mode: Processing Temp Source '{obj.name}' -> Original Target '{original_obj_name}'")
                                
                                # Get original texture name
                                original_tex_name = self.mask_repair_data['original_textures'].get(original_obj_name)
                                if not original_tex_name or original_tex_name not in bpy.data.images:
                                    print(f"❌ [GEMINI] Mask Repair Mode: Original texture '{original_tex_name}' not found for {original_obj_name}")
                                    continue
                                
                                original_tex = bpy.data.images[original_tex_name]
                                print(f"[GEMINI] Mask Repair Mode: Target texture = '{original_tex.name}'")
                                
                                # Perform incremental bake - NO CLEAR, NO MARGIN
                                try:
                                    print(f"[GEMINI] Mask Repair Mode: Baking with use_clear=False, margin=0")
                                    projection_utils.bake(
                                        context=bpy.context,
                                        obj=obj, # Temp Object as Source
                                        texture_node_name=self.image_node_name,
                                        target_image=original_tex, # Target Original Texture
                                        src_uv_name=data['src_uv_name'],
                                        margin=1,
                                        use_clear=False
                                    )
                                    print(f"✅ [GEMINI] Mask Repair Mode: Incremental bake completed for {obj.name}")
                                    
                                    # UV FIX: Ensure Original Object materials are restored
                                    orig_obj_ref = bpy.data.objects.get(original_obj_name)
                                    if orig_obj_ref:
                                         _finalize_object_materials(
                                             target_obj=orig_obj_ref, 
                                             target_img=original_tex, 
                                             dest_uv_name=data.get('dest_uv_name', "UVMap"),
                                             node_name=self.image_node_name
                                         )
                                         print(f"✅ [GEMINI] Restored active render UV and materials on original object")

                                except Exception as e:
                                    print(f"❌ [GEMINI] Mask Repair Mode: Bake failed for {obj.name}: {e}")
                                    import traceback
                                    traceback.print_exc()
                                
                                continue # Skip NORMAL MODE for this item
                            
                            # ============================================================
                            # NORMAL MODE (Only runs if mask_repair_data is None)
                            # ============================================================
                            
                            # CRITICAL: Skip temp mask objects if they somehow got here without mask_repair_data (shouldn't happen but safety)
                            if '_MaskTemp' in obj.name:
                                print(f"[GEMINI] Skipping temp mask object '{obj.name}' in normal bake loop")
                                continue
                            
                            # Create baked image
                            baked_name = f"{obj.name}_Baked_AI"
                            if baked_name in bpy.data.images:
                                bpy.data.images.remove(bpy.data.images[baked_name])
                            baked_img = bpy.data.images.new(baked_name, res_img.size[0], res_img.size[1])
                            
                            # Ensure we have a valid material reference
                            if not material:
                                material = bpy.data.materials.get(self.material_name)
                                
                            # Prepare Material for this object
                            baked_mat_name = f"{material.name}_{obj.name}_Baked"
                            obj_mat = bpy.data.materials.get(baked_mat_name)
                            if not obj_mat:
                                obj_mat = material.copy()
                                obj_mat.name = baked_mat_name
                            
                            # FORCE SHADELESS - Completely clear and rebuild
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
                            o_uv.uv_map = data.get('dest_uv_name', "UVMap") # INITIAL FIX
                            o_uv.location = (-200, 0)
                            o_links.new(o_uv.outputs['UV'], o_tex.inputs['Vector'])
                            o_links.new(o_tex.outputs['Color'], o_emit.inputs['Color'])
                            o_links.new(o_emit.outputs['Emission'], o_out.inputs['Surface'])
                            
                            # Unique material assignment
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
                                        print(f"[GEMINI] Prepared unique material {new_mat.name} with AI result")

                            # Configure UV Layer for baking
                            if data['dest_uv_name'] in obj.data.uv_layers:
                                obj.data.uv_layers.active = obj.data.uv_layers[data['dest_uv_name']]
                                obj.data.uv_layers[data['dest_uv_name']].active_render = True
                                print(f"[GEMINI] {obj.name} set {data['dest_uv_name']} as active for baking")
                            
                            # Perform Native Bake
                            try:
                                margin = getattr(self.scene.gemini_render, "bake_margin", 16)
                                projection_utils.bake(
                                    context=bpy.context,
                                    obj=obj,
                                    texture_node_name=self.image_node_name,
                                    target_image=baked_img,
                                    src_uv_name=data['src_uv_name'],
                                    margin=margin
                                )
                            except Exception as e:
                                print(f"❌ [GEMINI] Native bake failed for {obj.name}: {e}")
                                continue
                            
                            # Apply to current object
                            _finalize_object_materials(
                                target_obj=obj, 
                                target_img=baked_img, 
                                dest_uv_name=data.get('dest_uv_name', "UVMap"),
                                search_img=res_img, # MATCH PREVIEW IMAGE
                                node_name=self.image_node_name
                            )
                            
                            baked_img.pack()
                            print(f"✅ [GEMINI] Bake successful for {obj.name}")
                    
                    update_render_status(self.scene, "Projection completed!", False)
                    
                except Exception as e:
                    print(f"[GEMINI] Error applying projection result: {e}")
                    import traceback
                    traceback.print_exc()
                    update_render_status(self.scene, f"Error: {str(e)}", False)
                finally:
                    # Cleanup temp files
                    try:
                        if os.path.exists(self.depth_path): os.unlink(self.depth_path)
                        if os.path.exists(self.init_image_path): os.unlink(self.init_image_path)
                    except: pass
                    for data in self.target_objects_data:
                        try: data['bm_copy'].free()
                        except: pass
                    
                    # Mask Repair Mode: Cleanup temp objects and materials
                    if self.mask_repair_data:
                        print("[GEMINI] Mask Repair Mode: Cleaning up temp objects and materials...")
                        
                        # Delete temp mesh objects
                        for temp_obj_name in self.mask_repair_data.get('temp_objects', []):
                            try:
                                temp_obj = bpy.data.objects.get(temp_obj_name)
                                if temp_obj:
                                    # Store mesh data reference to delete later
                                    temp_mesh = temp_obj.data
                                    
                                    # Delete object safely with do_unlink=True
                                    bpy.data.objects.remove(temp_obj, do_unlink=True)
                                    
                                    # Delete mesh data if it exists and has no other users
                                    if temp_mesh and temp_mesh.users == 0:
                                        bpy.data.meshes.remove(temp_mesh)
                                        
                                    print(f"[GEMINI] Mask Repair Mode: Deleted temp object '{temp_obj_name}' and its mesh data")
                            except Exception as e:
                                print(f"⚠️ [GEMINI] Mask Repair Mode: Failed to delete temp object '{temp_obj_name}': {e}")
                        
                        # Delete temp materials
                        for temp_mat_name in self.mask_repair_data.get('temp_materials', []):
                            try:
                                temp_mat = bpy.data.materials.get(temp_mat_name)
                                if temp_mat:
                                    bpy.data.materials.remove(temp_mat)
                                    print(f"[GEMINI] Mask Repair Mode: Deleted temp material '{temp_mat_name}'")
                            except Exception as e:
                                print(f"⚠️ [GEMINI] Mask Repair Mode: Failed to delete temp material '{temp_mat_name}': {e}")
                        
                        print("[GEMINI] Mask Repair Mode: Cleanup complete")
                        
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
                                print(f"✅ [GEMINI] Restored selection to original object: {first_obj.name}")
                                
                                # 4. Optionally restore mode if needed (usually Object mode is safer here)
                                # For now, just ensure it's selected.
                        except Exception as e:
                            print(f"⚠️ [GEMINI] Selection restoration failed: {e}")

            execute_in_main_thread(_apply_result)
            
        except Exception as e:
            print(f"[GEMINI] Projection thread error: {e}")

def stop_thread_manager():
    """Stop the thread manager (call on addon unregister)"""
    _thread_manager.stop_timer()
