"""
Background pipeline for TextureProjector.

A single main-thread dispatcher (queue + bpy.app.timers) survives .blend
file switches, and one ProjectionRenderThread per job performs the API
call off the main thread, then queues all Blender-data mutations (image
load, material update, bake, cleanup) back onto the main thread.
"""

import os
import re
import shutil
import tempfile
import threading
import time
from queue import Queue, Empty
from typing import Callable

import bpy

from . import projection_utils


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def get_output_directory() -> str:
    """External output directory for generated textures."""
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
    return os.path.join(get_output_directory(),
                        f"{_sanitize_filename(base_name)}_{timestamp}{extension}")


def save_blender_image(image, base_name: str = None, file_format: str = 'PNG') -> str:
    """Save a Blender image to the output dir, keeping its filepath there."""
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


# ---------------------------------------------------------------------------
# Main-thread dispatcher
# ---------------------------------------------------------------------------

class BlenderThreadManager:
    """Queues callables from worker threads onto Blender's main thread."""

    def __init__(self):
        self.command_queue = Queue()
        self.timer_registered = False

    def execute_in_main_thread(self, func: Callable, *args, **kwargs) -> None:
        self.command_queue.put((func, args, kwargs))
        # Blend-file switches can silently invalidate timers; re-check the
        # actual registration state on every enqueue.
        if self.timer_registered and not bpy.app.timers.is_registered(self._process_queue):
            self.timer_registered = False
        if not self.timer_registered:
            bpy.app.timers.register(self._process_queue, first_interval=0.01)
            self.timer_registered = True

    def _process_queue(self) -> float:
        while True:
            try:
                func, args, kwargs = self.command_queue.get_nowait()
            except Empty:
                break
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"[GEMINI] Main-thread task error: {e}")
                import traceback
                traceback.print_exc()
        return 0.1

    def stop_timer(self) -> None:
        if self.timer_registered:
            try:
                bpy.app.timers.unregister(self._process_queue)
            except ValueError:
                pass
            self.timer_registered = False


_thread_manager = BlenderThreadManager()


def execute_in_main_thread(func: Callable, *args, **kwargs) -> None:
    _thread_manager.execute_in_main_thread(func, *args, **kwargs)


def show_error_popup(message: str, title: str = "Texture Projector") -> None:
    """Show an error as a popup at the mouse cursor (thread-safe).

    Far more visible than the status-bar text, and the full message is
    line-wrapped instead of truncated.
    """
    import textwrap
    lines = []
    for paragraph in str(message).splitlines():
        lines.extend(textwrap.wrap(paragraph, width=64) or [""])

    def _show():
        def draw(menu, _context):
            col = menu.layout.column()
            for line in lines[:20]:
                col.label(text=line)
        try:
            bpy.context.window_manager.popup_menu(draw, title=title,
                                                  icon='ERROR')
        except Exception as e:
            print(f"[GEMINI] Could not show error popup: {e}")

    execute_in_main_thread(_show)


def update_render_status(scene, status_text: str, is_rendering: bool = None) -> None:
    """Thread-safe UI status update."""
    def _update():
        try:
            props = getattr(scene, 'gemini_render', None)
            if props is not None:
                props.status_text = status_text
                if is_rendering is not None:
                    props.is_rendering = is_rendering
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        except Exception as e:
            print(f"[GEMINI] Status update error: {e}")
    execute_in_main_thread(_update)


# ---------------------------------------------------------------------------
# Result image loading + history (main thread only)
# ---------------------------------------------------------------------------

def _history_limit() -> int:
    from .gemini_api import get_prefs
    prefs = get_prefs()
    return int(getattr(prefs, 'history_limit', 10) or 10)


def _load_result_image_sync(image_data: bytes, image_name: str = "AI_Result",
                            user_prompt: str = "", cam_data: dict = None):
    """Write image bytes to disk, load into Blender, record history.

    Returns the loaded Image datablock. MUST run on the main thread.
    """
    import datetime
    try:
        temp_path = build_output_filepath(image_name)
        with open(temp_path, 'wb') as f:
            f.write(image_data)

        if image_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[image_name])

        img = bpy.data.images.load(temp_path)
        img.name = image_name
        img.filepath_raw = temp_path
        try:
            if hasattr(img, 'colorspace_settings'):
                img.colorspace_settings.name = 'sRGB'
        except Exception as e:
            print(f"[GEMINI] Colorspace warning: {e}")

        permanent_image = None
        if user_prompt:
            permanent_name = ("AI_Result_"
                              + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            img.name = permanent_name
            img.use_fake_user = True
            permanent_image = img

        # Refresh the "Render Result" style preview image.
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

        if user_prompt:
            scene = bpy.context.scene
            props = getattr(scene, 'gemini_render', None)
            if props is not None:
                item = props.render_history.add()
                item.prompt = user_prompt
                item.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                item.image_name = (permanent_image.name if permanent_image
                                   else render_result.name)

                if props.use_style_reference and props.style_reference_image:
                    item.style_reference_used = True
                    item.style_reference_name = props.style_reference_image.name

                if cam_data:
                    try:
                        item.cam_location = cam_data.get('location', (0, 0, 0))
                        item.cam_rotation = cam_data.get('rotation', (1, 0, 0, 0))
                        item.cam_lens = cam_data.get('lens', 50.0)
                        item.view_distance = cam_data.get('view_distance', 10.0)
                        item.is_camera_view = cam_data.get('is_camera_view', False)
                        if 'cam_obj_location' in cam_data:
                            item.cam_obj_location = cam_data['cam_obj_location']
                        if 'cam_obj_rotation' in cam_data:
                            item.cam_obj_rotation = cam_data['cam_obj_rotation']
                    except Exception as e:
                        print(f"[GEMINI] Failed to store camera data: {e}")

                limit = _history_limit()
                while len(props.render_history) > limit:
                    oldest = props.render_history[0]
                    if oldest.image_name in bpy.data.images:
                        bpy.data.images.remove(bpy.data.images[oldest.image_name])
                    props.render_history.remove(0)

        return permanent_image or render_result
    except Exception as e:
        print(f"[GEMINI] Error loading result image: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Projection job
# ---------------------------------------------------------------------------

class ProjectionRenderThread(threading.Thread):
    """Runs the AI request off-thread, then finalizes on the main thread."""

    active_threads = []

    def __init__(self, context, api_client, user_prompt, source_path, sim_path,
                 target_objects_data, do_bake, bypass_api=False,
                 mask_repair_data=None, input_source='COLOR', debug_mode=False,
                 source_image_override=None, cam_data=None, reference_path=None,
                 capture_width=1024, capture_height=1024, temp_dir=None,
                 resource_registry=None):
        super().__init__(daemon=True)
        self.scene = context.scene
        self.api_client = api_client
        self.user_prompt = user_prompt
        self.source_path = source_path
        self.sim_path = sim_path
        self.target_objects_data = target_objects_data
        self.do_bake = do_bake
        self.bypass_api = bypass_api
        self.mask_repair_data = mask_repair_data
        self.input_source = input_source
        self.debug_mode = debug_mode
        self.source_image_override = source_image_override
        self.cam_data = cam_data
        self.reference_path = reference_path
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.temp_dir = temp_dir
        self.resource_registry = resource_registry or {
            'temp_objects': [], 'temp_materials': [], 'target_objects_data': []}

        self._stop_event = threading.Event()
        self.error_message = None
        self._finalize_queued = False
        self._finalized = False
        # True once a valid result image reached the projection materials;
        # anything less rolls the material mutations back on finalize.
        self._applied = False

        ProjectionRenderThread.active_threads.append(self)

    # -- control -----------------------------------------------------------

    def stop(self):
        self._stop_event.set()

    def _queue_finalize(self, image_data=None, apply_result=False):
        if self._finalize_queued:
            return
        self._finalize_queued = True
        if apply_result:
            execute_in_main_thread(self._apply_result_and_finalize, image_data)
        else:
            execute_in_main_thread(self._finalize_main_thread)

    # -- worker ------------------------------------------------------------

    def run(self):
        try:
            update_render_status(self.scene, "Contacting Gemini...", True)
            image_data = None

            if self.source_image_override:
                pass  # direct image mode: nothing to generate
            elif self.bypass_api:
                with open(self.sim_path, 'rb') as f:
                    image_data = f.read()
            else:
                if self._stop_event.is_set():
                    return
                image_data, mime = self.api_client.generate_image(
                    depth_image_path=self.source_path,
                    user_prompt=self.user_prompt,
                    reference_image_path=self.reference_path,
                    width=self.capture_width,
                    height=self.capture_height,
                    is_color_render=(self.input_source == 'COLOR'),
                )
                print(f"[GEMINI] Received {len(image_data)} bytes ({mime})")

                if self.debug_mode and image_data and self.temp_dir:
                    try:
                        with open(os.path.join(self.temp_dir, "debug_output.png"),
                                  'wb') as f:
                            f.write(image_data)
                    except OSError as e:
                        print(f"[GEMINI] Debug save failed: {e}")

            if self._stop_event.is_set():
                return

            self._queue_finalize(
                image_data=image_data,
                apply_result=(self.source_image_override is not None
                              or image_data is not None))
        except Exception as e:
            print(f"[GEMINI] Projection thread error: {e}")
            self.error_message = str(e)
        finally:
            if not self._finalize_queued:
                self._queue_finalize()

    # -- main-thread finalization -------------------------------------------

    def _apply_result_and_finalize(self, image_data):
        final_status = None
        try:
            if self._stop_event.is_set():
                final_status = "Projection cancelled"
                return

            if self.source_image_override:
                res_img = self.source_image_override
            else:
                res_img = _load_result_image_sync(
                    image_data, "Gemini_Projection_Result",
                    self.user_prompt, self.cam_data)
            if not res_img:
                raise RuntimeError("Failed to load projection result image")
            self._applied = True

            # Feed the result into each object's unique projection material.
            processed = set()
            for data in self.target_objects_data:
                m_name = data.get('material_name')
                n_name = data.get('image_node_name', projection_utils.IMAGE_NODE_NAME)
                if not m_name or m_name in processed:
                    continue
                mat = bpy.data.materials.get(m_name)
                if mat and mat.use_nodes:
                    node = mat.node_tree.nodes.get(n_name)
                    if node:
                        node.image = res_img
                processed.add(m_name)

            if self.do_bake and not self._stop_event.is_set():
                self._set_status_sync("Baking result...", True)
                for data in self.target_objects_data:
                    if self._stop_event.is_set():
                        final_status = "Projection cancelled"
                        break
                    self._bake_entry(data, res_img)

            if final_status is None:
                final_status = ("Projection cancelled" if self._stop_event.is_set()
                                else "Done")
        except Exception as e:
            print(f"[GEMINI] Error applying projection result: {e}")
            import traceback
            traceback.print_exc()
            self.error_message = str(e)
            final_status = f"Error: {self.error_message}"
        finally:
            self._finalize_main_thread(final_status)

    def _bake_entry(self, data, res_img):
        obj = bpy.data.objects.get(data['object_name'])
        if not obj:
            return

        original_name = data.get('original_object_name')
        if original_name and self.mask_repair_data:
            tex_name = self.mask_repair_data['original_textures'].get(original_name)
            original_tex = bpy.data.images.get(tex_name)
            if original_tex:
                projection_utils.perform_projection_bake(
                    context=bpy.context, obj=obj,
                    texture_node_name=data.get('image_node_name',
                                               projection_utils.IMAGE_NODE_NAME),
                    target_image=original_tex,
                    src_uv_name=data['src_uv_name'],
                    dest_uv_name=data.get('dest_uv_name', "UVMap"),
                    is_mask_repair=True,
                    original_obj=bpy.data.objects.get(original_name),
                    search_img=res_img)
            return

        if '_MaskTemp' in obj.name:
            return

        safe_hash = abs(hash(obj.name)) % 100000
        baked_name = f"{obj.name[:30]}_{safe_hash}_{int(time.time())}_Baked_AI"
        if baked_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[baked_name])
        baked_img = bpy.data.images.new(baked_name, res_img.size[0], res_img.size[1])

        ok = projection_utils.perform_projection_bake(
            context=bpy.context, obj=obj,
            texture_node_name=data.get('image_node_name',
                                       projection_utils.IMAGE_NODE_NAME),
            target_image=baked_img,
            src_uv_name=data['src_uv_name'],
            dest_uv_name=data.get('dest_uv_name', "UVMap"),
            search_img=res_img)
        if ok:
            baked_path = save_blender_image(baked_img, baked_name)
            print(f"[GEMINI] Saved baked texture: {baked_path}")

    def _set_status_sync(self, status_text: str, is_rendering: bool = None):
        try:
            props = getattr(self.scene, 'gemini_render', None)
            if props is not None:
                props.status_text = status_text
                if is_rendering is not None:
                    props.is_rendering = is_rendering
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        except Exception as e:
            print(f"[GEMINI] Sync status error: {e}")

    def _cleanup_scene_sync(self):
        try:
            if bpy.context.object and bpy.context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass

        for obj_name in self.resource_registry.get('temp_objects', []):
            try:
                obj = bpy.data.objects.get(obj_name)
                if obj:
                    data = obj.data
                    obj_type = obj.type
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if data and data.users == 0:
                        if obj_type == 'MESH':
                            bpy.data.meshes.remove(data)
                        elif obj_type == 'CAMERA':
                            bpy.data.cameras.remove(data)
            except Exception as e:
                print(f"[GEMINI] Temp object cleanup error: {e}")

        for mat_name in self.resource_registry.get('temp_materials', []):
            try:
                mat = bpy.data.materials.get(mat_name)
                if mat:
                    bpy.data.materials.remove(mat)
            except Exception as e:
                print(f"[GEMINI] Temp material cleanup error: {e}")

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
            if self.resource_registry.get('restore_edit'):
                try:
                    bpy.ops.object.mode_set(mode='EDIT')
                except RuntimeError:
                    pass
        except Exception as e:
            print(f"[GEMINI] Selection restore error: {e}")

    def _cleanup_files(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                if self.debug_mode:
                    print(f"[GEMINI] Debug mode: keeping {self.temp_dir}")
                elif os.path.basename(self.temp_dir).startswith("gemini_proj_"):
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
            except OSError as e:
                print(f"[GEMINI] Temp dir cleanup error: {e}")

    def _finalize_main_thread(self, status_text=None):
        if self._finalized:
            return
        self._finalized = True

        # No valid result ever reached the materials (API error, block or
        # cancel): undo the material assignment so nothing renders black.
        if not self._applied:
            projection_utils.rollback_projection_materials(self.resource_registry)

        self._cleanup_scene_sync()
        self._cleanup_files()

        if self.error_message:
            show_error_popup(self.error_message)

        if self in ProjectionRenderThread.active_threads:
            ProjectionRenderThread.active_threads.remove(self)

        from . import operators
        if operators.GEMINI_OT_texture_projection.current_thread is self:
            operators.GEMINI_OT_texture_projection.current_thread = None

        if status_text is None:
            if self.error_message:
                status_text = f"Error: {self.error_message}"
            elif self._stop_event.is_set():
                status_text = "Projection cancelled"
            else:
                status_text = "Done"
        self._set_status_sync(status_text, False)


# ---------------------------------------------------------------------------
# Global lifecycle
# ---------------------------------------------------------------------------

def stop_all_projection_threads():
    for thread in list(ProjectionRenderThread.active_threads):
        thread.stop()
    ProjectionRenderThread.active_threads.clear()


def stop_thread_manager():
    _thread_manager.stop_timer()


def reset_threading_state():
    """Reset thread/timer state after loading a new .blend file."""
    stop_all_projection_threads()
    _thread_manager.stop_timer()
    while True:
        try:
            _thread_manager.command_queue.get_nowait()
        except Empty:
            break
