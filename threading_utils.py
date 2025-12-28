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

def load_result_image(image_data: bytes, image_name: str = "AI_Result", user_prompt: str = "") -> None:
    """Load result image into Blender and save to history (thread-safe)"""
    print(f"🚀 [GEMINI] load_result_image wrapper called for {image_name}")
    execute_in_main_thread(_load_result_image_sync, image_data, image_name, user_prompt)

def _load_result_image_sync(image_data: bytes, image_name: str = "AI_Result", user_prompt: str = "") -> Any:
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
    
    def __init__(self, scene, api_client, user_prompt: str, depth_path: str):
        super().__init__(daemon=True)
        self.scene = scene
        self.api_client = api_client
        self.user_prompt = user_prompt
        self.depth_path = depth_path
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
            load_result_image(image_data, "Gemini_AI_Result", self.user_prompt)
            
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
    
    def __init__(self, context, depth_renderer, api_client, user_prompt: str):
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
            has_camera = self.scene.camera is not None
            
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
            load_result_image(image_data, "Gemini_AI_Result", self.user_prompt)
            
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
    
    def __init__(self, context, api_client, user_prompt, depth_path, init_image_path, target_objects_data, image_node_name, material_name, do_bake, bypass_api=False):
        super().__init__(daemon=True)
        self.scene = context.scene
        self.api_client = api_client
        self.user_prompt = user_prompt
        self.depth_path = depth_path
        self.init_image_path = init_image_path
        self.target_objects_data = target_objects_data
        self.image_node_name = image_node_name
        self.material_name = material_name
        self.do_bake = do_bake
        self.bypass_api = bypass_api
        self._stop_event = threading.Event()
        print("[GEMINI] ProjectionRenderThread initialized")
    
    def stop(self):
        self._stop_event.set()
    
    def run(self):
        print("[GEMINI] ProjectionRenderThread starting...")
        try:
            update_render_status(self.scene, "Sending projection to Gemini...", True)
            
            # 1. Call API with depth + init_image (color capture)
            # The Gemini API needs to be told this is a projection task
            # We'll use the depth as control and init_image as the base
            
            # Since GeminiAPI.generate_image takes depth_path and reference_path,
            # we'll use init_image as reference_path but maybe with a custom prompt.
            
            # Wait, does generate_image support init_image?
            # nano-banana-render's gemini_api.py:
            # def generate_image(self, depth_image_path: str, user_prompt: str, reference_image_path: str = None, ...)
            
            # I should probably check gemini_api.py if it supports img2img or just depth-to-img.
            # Viewing gemini_api.py again to be sure.
            
            from . import operators # Local import to avoid circularity
            
            # For now, let's assume we use the regular generate_image
            # In Dream Textures, they used a "projection" prompt prefix.
            projection_prompt = f"Project this into a texture: {self.user_prompt}"
            
            # Resolution - use viewport size or 1024
            props = self.scene.gemini_render
            resolution = int(props.resolution)
            
            if self.bypass_api:
                print("🛡️ [GEMINI] Simulation Mode: Bypassing AI call, using local grid simulation...")
                with open(self.init_image_path, 'rb') as f:
                    image_data = f.read()
                mime_type = "image/png"
            else:
                print(f"🚀 [GEMINI] Calling AI to generate texture...")
                image_data, mime_type = self.api_client.generate_image(
                    depth_image_path=self.depth_path,
                    user_prompt=projection_prompt,
                    reference_image_path=self.init_image_path,
                    is_color_render=True,
                    width=resolution,
                    height=resolution
                )
            
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
                    # 1. Integrate with Render Gallery (Synchronous)
                    # We call the sync version to get the Image object IMMEDIATELY
                    res_img = _load_result_image_sync(image_data, "Gemini_Projection_Result", self.user_prompt)
                    
                    if not res_img:
                         print("❌ [GEMINI] Failed to load projection result image")
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
                        
                        for data in self.target_objects_data:
                            obj = bpy.data.objects.get(data['object_name'])
                            if not obj: continue
                            
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
                            
                            # FORCE SHADELESS - Completely clear and rebuild if it's not our spec
                            # (This handles existing materials that might have BSDF)
                            if True: # Always enforce for now to satisfy "彻底清除干净"
                                o_nodes = obj_mat.node_tree.nodes
                                o_links = obj_mat.node_tree.links
                                o_nodes.clear()
                                # Rebuild shadeless
                                o_out = o_nodes.new("ShaderNodeOutputMaterial")
                                o_out.location = (400, 0)
                                o_emit = o_nodes.new("ShaderNodeEmission")
                                o_emit.location = (200, 0)
                                o_emit.inputs['Strength'].default_value = 1.0
                                o_tex = o_nodes.new("ShaderNodeTexImage")
                                o_tex.name = self.image_node_name # Link back to the node name we expect
                                o_tex.location = (0, 0)
                                o_uv = o_nodes.new("ShaderNodeUVMap")
                                o_uv.name = "Gemini_UV_Map"
                                o_uv.location = (-200, 0)
                                
                                o_links.new(o_uv.outputs['UV'], o_tex.inputs['Vector'])
                                o_links.new(o_tex.outputs['Color'], o_emit.inputs['Color'])
                                o_links.new(o_emit.outputs['Emission'], o_out.inputs['Surface'])
                            
                            # Assign material to object
                            # Prepare a unique material for this specific object
                            # (This ensures we don't accidentally bake two objects into the same material/texture)
                            unique_mats_map = {} # Original mat -> Object unique mat
                            
                            for m_idx, slot in enumerate(obj.material_slots):
                                if not slot.material: continue
                                
                                # If this material was our projection material or is shared, copy it
                                mat = slot.material
                                if mat not in unique_mats_map:
                                    new_mat = mat.copy()
                                    new_mat.name = f"{mat.name}_{obj.name}_Baked"
                                    unique_mats_map[mat] = new_mat
                                
                                # Assign the unique copy to the object
                                obj.data.materials[m_idx] = unique_mats_map[mat]
                                
                                # Ensure the projection node in this UNIQUE material has the AI result
                                if unique_mats_map[mat].use_nodes:
                                    proj_node = unique_mats_map[mat].node_tree.nodes.get(self.image_node_name)
                                    if proj_node:
                                        proj_node.image = res_img
                                        print(f"[GEMINI] Prepared unique material {new_mat.name} with AI result")

                            # 2. Configure UV Layer "Indices" for baking:
                            # The ACTIVE UV layer is where Blender's bake operator outputs the result.
                            # We want to bake INTO the target UV (dest_uv_name).
                            if data['dest_uv_name'] in obj.data.uv_layers:
                                obj.data.uv_layers.active = obj.data.uv_layers[data['dest_uv_name']]
                                print(f"[GEMINI] {obj.name} set {data['dest_uv_name']} as active for baking")
                            
                            # Ensure active render is also set (often used as fallback)
                            if data['dest_uv_name'] in obj.data.uv_layers:
                                obj.data.uv_layers[data['dest_uv_name']].active_render = True
                            
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
                            
                            # 3. POST-BAKE MATERIAL FINALIZATION
                            # Update all unique materials of this object to use the baked image
                            for slot in obj.material_slots:
                                if slot.material and slot.material.use_nodes:
                                    m_nodes = slot.material.node_tree.nodes
                                    m_links = slot.material.node_tree.links
                                    
                                    t_node = m_nodes.get(self.image_node_name)
                                    if not t_node:
                                        continue # Fix for AttributeError
                                        
                                    t_node.image = baked_img
                                    print(f"[GEMINI] Finalized {slot.material.name} with baked image")
                                        
                                    # Restore UV mapping to the destination (the UV map we baked to)
                                    for link in t_node.inputs['Vector'].links:
                                        if link.from_node.type == 'UV_MAP':
                                            link.from_node.uv_map = data['dest_uv_name']
                                            print(f"[GEMINI] Switched {slot.material.name} UV to {data['dest_uv_name']}")
                                            break
                                    else:
                                        # Fallback link creation
                                        uv_n = next((n for n in m_nodes if n.type == 'UV_MAP'), None) or m_nodes.new('ShaderNodeUVMap')
                                        uv_n.uv_map = data['dest_uv_name']
                                        if not t_node.inputs['Vector'].is_linked:
                                            m_links.new(uv_n.outputs['UV'], t_node.inputs['Vector'])
                            
                            # Restore UI active UV
                            if data['dest_uv_name'] in obj.data.uv_layers:
                                obj.data.uv_layers.active = obj.data.uv_layers[data['dest_uv_name']]
                            
                            baked_img.pack()
                            print(f"✅ [GEMINI] Bake successful for {obj.name} -> {baked_name}")
                    
                    update_render_status(self.scene, "Projection completed!", False)
                    
                except Exception as e:
                    print(f"[GEMINI] Error applying projection result: {e}")
                    import traceback
                    traceback.print_exc()
                    update_render_status(self.scene, f"Error: {str(e)}", False)
                finally:
                    # Cleanup
                    try:
                        if os.path.exists(self.depth_path): os.unlink(self.depth_path)
                        if os.path.exists(self.init_image_path): os.unlink(self.init_image_path)
                    except: pass
                    for data in self.target_objects_data:
                        try: data['bm_copy'].free()
                        except: pass

            execute_in_main_thread(_apply_result)
            
        except Exception as e:
            print(f"[GEMINI] Projection thread error: {e}")

def stop_thread_manager():
    """Stop the thread manager (call on addon unregister)"""
    _thread_manager.stop_timer()
