"""
Operators for TextureProjector.

The projection operator is one-click: in Object Mode it projects onto all
faces of the selected meshes; in Edit Mode it projects onto the selected
faces only. No manual mode juggling required.
"""

import os

import numpy as np
import bpy
from bpy.types import Operator
from bpy.props import IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import gemini_api
from . import projection_utils
from . import threading_utils


# ---------------------------------------------------------------------------
# View state capture / restore (for the gallery's "return to angle" feature)
# ---------------------------------------------------------------------------

def find_view3d(context):
    """Return (area, region, space, rv3d) of the best 3D viewport."""
    if context.area and context.area.type == 'VIEW_3D':
        space = context.area.spaces.active
        rv3d = space.region_3d
        region = next((r for r in context.area.regions if r.type == 'WINDOW'), None)
        if region and rv3d:
            return context.area, region, space, rv3d

    for area in context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        space = area.spaces.active
        rv3d = space.region_3d
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region and rv3d:
            return area, region, space, rv3d
    return None, None, None, None


def get_current_view_state(context):
    """Capture the current viewport/camera state for later restoration."""
    view_state = {}
    try:
        scene = context.scene
        _, _, space, rv3d = find_view3d(context)
        if rv3d:
            view_state['location'] = rv3d.view_location.copy()
            view_state['rotation'] = rv3d.view_rotation.copy()
            view_state['view_distance'] = rv3d.view_distance
            view_state['is_camera_view'] = (rv3d.view_perspective == 'CAMERA')
            view_state['lens'] = (scene.camera.data.lens if scene.camera
                                  else (space.lens if space else 50.0))
        if view_state.get('is_camera_view') and scene.camera:
            view_state['cam_obj_location'] = scene.camera.location.copy()
            view_state['cam_obj_rotation'] = scene.camera.rotation_euler.copy()
    except Exception as e:
        print(f"[GEMINI] Failed to capture view state: {e}")
    return view_state


def restore_view_state(context, history_item):
    """Restore the viewport or camera state from a history item."""
    try:
        area, _, space, rv3d = find_view3d(context)
        if not rv3d:
            return False

        if history_item.is_camera_view:
            scene = context.scene
            if scene.camera:
                scene.camera.location = history_item.cam_obj_location
                scene.camera.rotation_euler = history_item.cam_obj_rotation
            # Do NOT touch rv3d.view_rotation/location here: that would
            # kick the viewport out of camera view.
            rv3d.view_perspective = 'CAMERA'
            space.lens = history_item.cam_lens
        else:
            rv3d.view_perspective = 'PERSP'
            rv3d.view_location = history_item.cam_location
            rv3d.view_rotation = history_item.cam_rotation
            rv3d.view_distance = history_item.view_distance
            space.lens = history_item.cam_lens

        area.tag_redraw()
        return True
    except Exception as e:
        print(f"[GEMINI] Failed to restore view state: {e}")
        return False


# ---------------------------------------------------------------------------
# Render state guard
# ---------------------------------------------------------------------------

def _save_render_state(scene):
    return {
        'res_x': scene.render.resolution_x,
        'res_y': scene.render.resolution_y,
        'res_pct': scene.render.resolution_percentage,
        'filepath': scene.render.filepath,
        'format': scene.render.image_settings.file_format,
    }


def _restore_render_state(scene, saved):
    scene.render.resolution_x = saved['res_x']
    scene.render.resolution_y = saved['res_y']
    scene.render.resolution_percentage = saved['res_pct']
    scene.render.filepath = saved['filepath']
    scene.render.image_settings.file_format = saved['format']


def _cleanup_registry(registry, debug_mode=False):
    """Synchronous cleanup for failures before the worker thread starts."""
    import shutil
    try:
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError:
        pass

    projection_utils.rollback_projection_materials(registry)

    for name in registry.get('temp_objects', []):
        try:
            obj = bpy.data.objects.get(name)
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
            print(f"[GEMINI] Cleanup object error: {e}")

    for name in registry.get('temp_materials', []):
        try:
            mat = bpy.data.materials.get(name)
            if mat:
                bpy.data.materials.remove(mat)
        except Exception as e:
            print(f"[GEMINI] Cleanup material error: {e}")

    temp_dir = registry.get('temp_dir')
    if temp_dir and os.path.exists(temp_dir) and not debug_mode:
        if os.path.basename(temp_dir).startswith("gemini_proj_"):
            shutil.rmtree(temp_dir, ignore_errors=True)

    try:
        bpy.ops.object.select_all(action='DESELECT')
        for name in registry.get('original_selected_objs', []):
            obj = bpy.data.objects.get(name)
            if obj:
                obj.select_set(True)
        active_name = registry.get('original_active_obj')
        if active_name and active_name in bpy.data.objects:
            bpy.context.view_layer.objects.active = bpy.data.objects[active_name]
        if registry.get('restore_edit'):
            bpy.ops.object.mode_set(mode='EDIT')
    except Exception as e:
        print(f"[GEMINI] Cleanup selection restore error: {e}")


# ---------------------------------------------------------------------------
# Main projection operator
# ---------------------------------------------------------------------------

class GEMINI_OT_texture_projection(Operator):
    """Capture the view, generate a texture with AI and project it onto the selection"""
    bl_idname = "gemini.texture_projection"
    bl_label = "AI Texture Projection"
    bl_description = ("One click: capture the current view, generate a texture "
                      "and project/bake it onto the selected meshes")
    bl_options = {'REGISTER', 'UNDO'}

    current_thread = None

    @classmethod
    def poll(cls, context):
        if any(o.type == 'MESH' for o in context.selected_objects):
            return True
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        scene = context.scene
        props = scene.gemini_render

        cls = GEMINI_OT_texture_projection
        if cls.current_thread and cls.current_thread.is_alive():
            self.report({'WARNING'}, "A projection is already running")
            return {'CANCELLED'}

        if props.projection_source == 'AI' and not gemini_api.get_api_key():
            self.report({'ERROR'},
                        "No API key. Set it in the add-on preferences "
                        "(Edit > Preferences > Add-ons > TextureProjector)")
            return {'CANCELLED'}

        registry = {
            'temp_dir': None,
            'temp_objects': [],
            'temp_materials': [],
            'target_objects_data': [],
            'material_rollback': [],
            'original_active_obj': (context.active_object.name
                                    if context.active_object else None),
            'original_selected_objs': [o.name for o in context.selected_objects],
            'restore_edit': False,
        }
        started = False

        try:
            projection_utils.validate_projection(context)

            area, region, space, rv3d = find_view3d(context)
            if not (area and region and rv3d):
                self.report({'ERROR'}, "No 3D Viewport found")
                return {'CANCELLED'}

            # Capture the view state BEFORE anything mutates the scene.
            cam_data = get_current_view_state(context)

            targets, was_edit = projection_utils.gather_projection_targets(context)
            if was_edit:
                # Sync edit-mode face selection into mesh data.
                bpy.ops.object.mode_set(mode='OBJECT')
                registry['restore_edit'] = True

            face_masks = projection_utils.build_face_masks(targets, was_edit)
            if not face_masks:
                self.report({'ERROR'},
                            "No target faces (in Edit Mode select some faces; "
                            "in Object Mode select mesh objects)")
                return {'CANCELLED'}

            # Resolve bake-destination UV maps before any mesh mutation.
            # Empty option = the object's first UV map (UV0); a missing
            # named map is created on the spot.
            dest_uvs = {
                obj.name: projection_utils.resolve_dest_uv(obj, props.bake_uv_name)
                for obj in targets
            }

            width, height, use_scene_cam = projection_utils.get_capture_dimensions(
                scene, region, rv3d)
            cam_obj, cam_is_temp = projection_utils.resolve_capture_camera(
                context, scene, space, rv3d, width, height, use_scene_cam)
            if cam_is_temp:
                registry['temp_objects'].append(cam_obj.name)

            temp_dir = projection_utils.setup_temporary_workspace(props)
            registry['temp_dir'] = temp_dir

            reference_path = None
            source_path = ""
            sim_path = ""
            bypass_api = False
            source_image_override = None
            mask_repair_data = None

            saved_render = _save_render_state(scene)
            try:
                scene.render.resolution_x = width
                scene.render.resolution_y = height
                scene.render.resolution_percentage = 100
                scene.render.image_settings.file_format = 'PNG'

                # --- Style reference (clean, before mask overlays) ---------
                if props.use_style_reference:
                    if props.use_viewport_as_reference:
                        ref_name = ("debug_reference.png" if props.debug_mode
                                    else "viewport_reference.png")
                        ref_path = os.path.join(temp_dir, ref_name)
                        props.status_text = "Capturing reference..."
                        if projection_utils.capture_color(
                                context, scene, cam_obj, space,
                                width, height, ref_path):
                            reference_path = ref_path
                    elif props.style_reference_image:
                        reference_path = self._save_reference_image(
                            props.style_reference_image, temp_dir, props.debug_mode)

                # --- Mask repair overlays (after reference capture) --------
                if props.mask_repair_mode:
                    mask_repair_data = projection_utils.setup_mask_repair_meshes(
                        context, props, registry, face_masks)
                    if not mask_repair_data:
                        self.report({'ERROR'},
                                    "Mask repair setup failed: the selection "
                                    "needs faces on textured objects")
                        return {'CANCELLED'}
                    context.view_layer.update()

                # --- Source capture ----------------------------------------
                if props.projection_source == 'IMAGE':
                    if not props.projection_image:
                        self.report({'ERROR'}, "No projection image selected")
                        return {'CANCELLED'}
                    source_image_override = props.projection_image
                    bypass_api = True
                else:
                    props.status_text = "Capturing source..."
                    if props.debug_mode:
                        source_name = ("debug_output.png"
                                       if props.projection_source in {'GRID', 'VIEW'}
                                       else "debug_capture.png")
                    else:
                        source_name = "captured_source.png"
                    source_path = os.path.join(temp_dir, source_name)

                    if props.projection_source == 'GRID':
                        ok = projection_utils.capture_wireframe(
                            context, scene, cam_obj, area, region, space, rv3d,
                            width, height, source_path)
                        bypass_api = True
                    elif props.projection_source == 'VIEW':
                        ok = projection_utils.capture_color(
                            context, scene, cam_obj, space, width, height,
                            source_path)
                        bypass_api = True
                    elif props.input_source == 'DEPTH':
                        ok = projection_utils.capture_depth(
                            context, cam_obj, width, height, source_path)
                    else:
                        ok = projection_utils.capture_color(
                            context, scene, cam_obj, space, width, height,
                            source_path)
                    if not ok:
                        raise RuntimeError("View capture failed (see console)")
                    if bypass_api:
                        sim_path = source_path

                # --- UV projection + per-object materials -------------------
                processed = self._project_targets(
                    context, props, registry, face_masks, mask_repair_data,
                    dest_uvs, cam_obj, width, height)
                if processed == 0:
                    self.report({'ERROR'}, "No faces were projected")
                    return {'CANCELLED'}
            finally:
                _restore_render_state(scene, saved_render)

            # --- Launch the background job ------------------------------
            api_client = gemini_api.GeminiAPI(
                gemini_api.get_api_key() or "", props.model_name)

            thread = threading_utils.ProjectionRenderThread(
                context=context,
                api_client=api_client,
                user_prompt=props.prompt,
                source_path=source_path,
                sim_path=sim_path,
                target_objects_data=registry['target_objects_data'],
                do_bake=props.projection_bake,
                bypass_api=bypass_api,
                mask_repair_data=mask_repair_data,
                input_source=props.input_source,
                debug_mode=props.debug_mode,
                source_image_override=source_image_override,
                cam_data=cam_data,
                reference_path=reference_path,
                capture_width=width,
                capture_height=height,
                temp_dir=temp_dir,
                resource_registry=registry,
            )
            cls.current_thread = thread
            props.is_rendering = True
            props.status_text = "Processing..."
            thread.start()
            started = True

            self.report({'INFO'}, f"AI projection started ({processed} objects)")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, str(e))
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        finally:
            if not started:
                _cleanup_registry(registry, props.debug_mode)
                props.is_rendering = False

    # -- helpers ------------------------------------------------------------

    def _save_reference_image(self, ref_img, temp_dir, debug_mode):
        ref_name = "debug_reference.png" if debug_mode else "temp_reference_input.png"
        path = os.path.join(temp_dir, ref_name)
        original_filepath = ref_img.filepath_raw
        original_format = ref_img.file_format
        try:
            ref_img.filepath_raw = path
            ref_img.file_format = 'PNG'
            ref_img.save()
        except Exception as e:
            print(f"[GEMINI] Reference save failed: {e}")
            if ref_img.packed_file:
                try:
                    with open(path, 'wb') as f:
                        f.write(ref_img.packed_file.data)
                except OSError:
                    return None
            else:
                return None
        finally:
            ref_img.filepath_raw = original_filepath
            ref_img.file_format = original_format
        return path if os.path.exists(path) else None

    def _project_targets(self, context, props, registry, face_masks,
                         mask_repair_data, dest_uvs,
                         cam_obj, width, height):
        processed = 0
        for obj_name, face_mask in face_masks.items():
            obj = bpy.data.objects.get(obj_name)
            if not obj:
                continue

            if mask_repair_data and obj_name in mask_repair_data['original_object_names']:
                # Zero-leak guard: originals stay untouched; project onto the
                # overlay temp object instead.
                temp_name = next(
                    (t for t, o in mask_repair_data['temp_to_original'].items()
                     if o == obj_name), None)
                temp_obj = bpy.data.objects.get(temp_name) if temp_name else None
                if not temp_obj:
                    continue

                mat_name = f"Gemini_Projection_Material_{temp_obj.name}"
                material, image_node, _ = projection_utils.setup_projection_material(
                    material_name=mat_name)
                temp_obj.data.materials.clear()
                temp_obj.data.materials.append(material)

                if not projection_utils.project_uvs(
                        temp_obj, context, cam_obj, width, height):
                    continue

                registry['target_objects_data'].append({
                    'object_name': temp_obj.name,
                    'original_object_name': obj_name,
                    'src_uv_name': projection_utils.PROJECTED_UV_NAME,
                    'dest_uv_name': dest_uvs.get(obj_name, ""),
                    'material_name': material.name,
                    'image_node_name': image_node.name,
                })
                processed += 1
                continue

            # Normal path: unique material per object (never share!).
            # Snapshot the material state first so a failed/cancelled run
            # can roll back instead of leaving a black projection material.
            mesh = obj.data
            indices_snapshot = np.empty(len(mesh.polygons), dtype=np.int32)
            mesh.polygons.foreach_get('material_index', indices_snapshot)
            slots_before = len(mesh.materials)

            mat_name = f"Gemini_Projection_Material_{obj.name}"
            material, image_node, _ = projection_utils.setup_projection_material(
                material_name=mat_name)
            projection_utils.assign_material_to_faces(obj, material, face_mask)

            registry['material_rollback'].append({
                'object_name': obj.name,
                'indices': indices_snapshot,
                'added_slot': len(mesh.materials) > slots_before,
                'material_name': material.name,
            })

            if not projection_utils.project_uvs(
                    obj, context, cam_obj, width, height, face_mask):
                continue

            registry['target_objects_data'].append({
                'object_name': obj.name,
                'src_uv_name': projection_utils.PROJECTED_UV_NAME,
                'dest_uv_name': dest_uvs.get(obj_name, ""),
                'material_name': material.name,
                'image_node_name': image_node.name,
            })
            processed += 1
        return processed


# ---------------------------------------------------------------------------
# Control / utility operators
# ---------------------------------------------------------------------------

class GEMINI_OT_stop_render(Operator):
    """Stop the current AI operation"""
    bl_idname = "gemini.stop_render"
    bl_label = "Stop Processing"
    bl_description = "Stop the current AI projection or render operation"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.gemini_render
        cls = GEMINI_OT_texture_projection
        if cls.current_thread and cls.current_thread.is_alive():
            cls.current_thread.stop()
            cls.current_thread = None
        threading_utils.stop_all_projection_threads()
        props.is_rendering = False
        props.status_text = "Cancelled by user"
        self.report({'INFO'}, "AI processing stopped")
        return {'FINISHED'}


class GEMINI_OT_reset_state(Operator):
    """Force-reset the addon state if the UI gets stuck"""
    bl_idname = "gemini.reset_state"
    bl_label = "Reset UI State"
    bl_description = "Force reset render state and stop all threads"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.gemini_render
        threading_utils.stop_all_projection_threads()
        cls = GEMINI_OT_texture_projection
        if cls.current_thread:
            cls.current_thread.stop()
            cls.current_thread = None
        props.is_rendering = False
        props.status_text = "Ready"
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        self.report({'INFO'}, "State reset, all threads stopped")
        return {'FINISHED'}


class GEMINI_OT_open_console(Operator):
    """Toggle the system console to view logs"""
    bl_idname = "gemini.open_console"
    bl_label = "Open Console"
    bl_options = {'REGISTER'}

    def execute(self, context):
        try:
            bpy.ops.wm.console_toggle()
            self.report({'INFO'}, "Console toggled")
            return {'FINISHED'}
        except (AttributeError, RuntimeError):
            self.report({'WARNING'}, "Console toggle not available on this platform")
            return {'CANCELLED'}


class GEMINI_OT_open_api_key_url(Operator):
    """Open Google AI Studio to get an API key"""
    bl_idname = "gemini.open_api_key_url"
    bl_label = "Get API Key"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import webbrowser
        webbrowser.open("https://aistudio.google.com/api-keys/")
        self.report({'INFO'}, "Opened Google AI Studio")
        return {'FINISHED'}


class GEMINI_OT_open_addon_prefs(Operator):
    """Open this add-on's preferences"""
    bl_idname = "gemini.open_addon_prefs"
    bl_label = "Open Preferences"
    bl_options = {'REGISTER'}

    def execute(self, context):
        bpy.ops.screen.userpref_show()
        context.preferences.active_section = 'ADDONS'
        try:
            context.window_manager.addon_search = "TextureProjector"
        except AttributeError:
            pass
        return {'FINISHED'}


class GEMINI_OT_validate_api_key(Operator):
    """Verify the configured API key against the Gemini endpoint"""
    bl_idname = "gemini.validate_api_key"
    bl_label = "Test API Key"
    bl_options = {'REGISTER'}

    def execute(self, context):
        key = gemini_api.get_api_key()
        if not key:
            self.report({'ERROR'}, "No API key configured")
            return {'CANCELLED'}
        ok, message = gemini_api.validate_api_key_online(key)
        self.report({'INFO'} if ok else {'ERROR'}, message)
        return {'FINISHED'} if ok else {'CANCELLED'}


class GEMINI_OT_open_output_dir(Operator):
    """Open the generated-textures output folder"""
    bl_idname = "gemini.open_output_dir"
    bl_label = "Open Output Folder"
    bl_options = {'REGISTER'}

    def execute(self, context):
        path = threading_utils.get_output_directory()
        try:
            if os.name == 'nt':
                os.startfile(path)
            else:
                import subprocess
                opener = 'open' if os.uname().sysname == 'Darwin' else 'xdg-open'
                subprocess.Popen([opener, path])
            self.report({'INFO'}, path)
            return {'FINISHED'}
        except OSError as e:
            self.report({'ERROR'}, f"Cannot open folder: {e}")
            return {'CANCELLED'}


# ---------------------------------------------------------------------------
# History operators
# ---------------------------------------------------------------------------

def _valid_history_item(props, index):
    return 0 <= index < len(props.render_history)


class GEMINI_OT_load_history(Operator):
    """Open this render in an image editor"""
    bl_idname = "gemini.load_history"
    bl_label = "Load History Render"
    bl_options = {'REGISTER'}

    history_index: IntProperty(default=0)

    def execute(self, context):
        props = context.scene.gemini_render
        if not _valid_history_item(props, self.history_index):
            self.report({'ERROR'}, "Invalid history index")
            return {'CANCELLED'}

        item = props.render_history[self.history_index]
        image = bpy.data.images.get(item.image_name)

        if image and not image.has_data:
            # Try to revive: packed pixel access, then disk reload.
            try:
                _ = image.pixels[0]
            except (IndexError, RuntimeError):
                pass
            if not image.has_data and image.filepath \
                    and os.path.exists(bpy.path.abspath(image.filepath)):
                try:
                    image.reload()
                except RuntimeError:
                    pass
            if not image.has_data:
                image = None

        if not image:
            self.report({'WARNING'},
                        f"History image '{item.image_name}' not found or empty")
            return {'CANCELLED'}

        # Prefer an existing image editor; otherwise open a new window.
        for window in context.window_manager.windows:
            for w_area in window.screen.areas:
                if w_area.type == 'IMAGE_EDITOR':
                    for space in w_area.spaces:
                        if space.type == 'IMAGE_EDITOR':
                            space.image = image
                            w_area.tag_redraw()
                            self.report({'INFO'}, f"Opened: {image.name}")
                            return {'FINISHED'}
        try:
            bpy.ops.wm.window_new()
            new_window = context.window_manager.windows[-1]
            w_area = new_window.screen.areas[0]
            w_area.type = 'IMAGE_EDITOR'
            for space in w_area.spaces:
                if space.type == 'IMAGE_EDITOR':
                    space.image = image
            w_area.tag_redraw()
        except RuntimeError as e:
            self.report({'WARNING'}, f"Could not open a window: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Opened: {image.name}")
        return {'FINISHED'}


class GEMINI_OT_open_history_image(Operator):
    """View this render"""
    bl_idname = "gemini.open_history_image"
    bl_label = "View Render"
    bl_options = {'REGISTER'}

    history_index: IntProperty(default=0)

    def execute(self, context):
        return bpy.ops.gemini.load_history(history_index=self.history_index)


class GEMINI_OT_delete_history(Operator):
    """Delete this render from the history"""
    bl_idname = "gemini.delete_history"
    bl_label = "Delete History Render"
    bl_options = {'REGISTER'}

    history_index: IntProperty(default=0)

    def execute(self, context):
        props = context.scene.gemini_render
        if not _valid_history_item(props, self.history_index):
            self.report({'ERROR'}, "Invalid history index")
            return {'CANCELLED'}

        item = props.render_history[self.history_index]
        for name in (item.image_name, item.style_reference_thumbnail):
            if name and name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[name])
        prompt_preview = item.prompt[:30]
        props.render_history.remove(self.history_index)
        self.report({'INFO'}, f"Deleted: {prompt_preview}...")
        return {'FINISHED'}


class GEMINI_OT_use_history_prompt(Operator):
    """Copy this item's prompt into the prompt field"""
    bl_idname = "gemini.use_history_prompt"
    bl_label = "Use History Prompt"
    bl_options = {'REGISTER'}

    history_index: IntProperty(default=0)

    def execute(self, context):
        props = context.scene.gemini_render
        if not _valid_history_item(props, self.history_index):
            self.report({'ERROR'}, "Invalid history index")
            return {'CANCELLED'}
        props.prompt = props.render_history[self.history_index].prompt
        self.report({'INFO'}, "Prompt copied")
        return {'FINISHED'}


class GEMINI_OT_use_history_style(Operator):
    """Reuse this item's style reference"""
    bl_idname = "gemini.use_history_style"
    bl_label = "Use History Style"
    bl_options = {'REGISTER'}

    history_index: IntProperty(default=0)

    def execute(self, context):
        props = context.scene.gemini_render
        if not _valid_history_item(props, self.history_index):
            self.report({'ERROR'}, "Invalid history index")
            return {'CANCELLED'}

        item = props.render_history[self.history_index]
        if not item.style_reference_used:
            self.report({'WARNING'}, "This render did not use a style reference")
            return {'CANCELLED'}

        style_image = (bpy.data.images.get(item.style_reference_name)
                       or bpy.data.images.get(item.style_reference_thumbnail))
        if not style_image:
            self.report({'ERROR'}, "Style reference image not found")
            return {'CANCELLED'}

        props.style_reference_image = style_image
        props.use_style_reference = True
        self.report({'INFO'}, f"Style reference set: {style_image.name}")
        return {'FINISHED'}


class GEMINI_OT_use_history_both(Operator):
    """Reuse this item's prompt and style reference"""
    bl_idname = "gemini.use_history_both"
    bl_label = "Use Prompt + Style"
    bl_options = {'REGISTER'}

    history_index: IntProperty(default=0)

    def execute(self, context):
        props = context.scene.gemini_render
        if not _valid_history_item(props, self.history_index):
            self.report({'ERROR'}, "Invalid history index")
            return {'CANCELLED'}

        item = props.render_history[self.history_index]
        props.prompt = item.prompt

        if item.style_reference_used:
            style_image = (bpy.data.images.get(item.style_reference_name)
                           or bpy.data.images.get(item.style_reference_thumbnail))
            if style_image:
                props.style_reference_image = style_image
                props.use_style_reference = True
                self.report({'INFO'}, "Prompt and style copied")
                return {'FINISHED'}
            props.use_style_reference = False
            self.report({'WARNING'}, "Style missing, copied prompt only")
            return {'FINISHED'}

        props.use_style_reference = False
        self.report({'INFO'}, "Prompt copied (no style was used)")
        return {'FINISHED'}


class GEMINI_OT_set_projection_source(Operator):
    """Use this render as the projection source and restore its view angle"""
    bl_idname = "gemini.set_projection_source"
    bl_label = "Use as Projection Source"
    bl_options = {'REGISTER', 'UNDO'}

    history_index: IntProperty(default=0)

    def execute(self, context):
        props = context.scene.gemini_render
        if not _valid_history_item(props, self.history_index):
            self.report({'ERROR'}, "Invalid history index")
            return {'CANCELLED'}

        item = props.render_history[self.history_index]
        image = bpy.data.images.get(item.image_name)
        if not image:
            self.report({'ERROR'}, f"Image '{item.image_name}' not found")
            return {'CANCELLED'}

        props.projection_source = 'IMAGE'
        props.projection_image = image
        restore_view_state(context, item)
        self.report({'INFO'}, f"Source: {item.image_name} (view restored)")
        return {'FINISHED'}


class GEMINI_OT_history_context_menu(Operator):
    """Options for this history item"""
    bl_idname = "gemini.history_context_menu"
    bl_label = "History Options"
    bl_options = {'REGISTER'}

    history_index: IntProperty(default=0)

    def execute(self, context):
        def draw_menu(menu_self, menu_context):
            layout = menu_self.layout
            props = menu_context.scene.gemini_render
            idx = menu_context.window_manager.history_menu_index
            if idx >= len(props.render_history):
                layout.label(text="Invalid history item")
                return
            item = props.render_history[idx]

            layout.operator("gemini.use_history_prompt",
                            text="Use Prompt", icon='TEXT').history_index = idx
            if item.style_reference_used:
                layout.operator("gemini.use_history_style",
                                text="Use Style", icon='IMAGE_DATA').history_index = idx
                layout.operator("gemini.use_history_both",
                                text="Use Both", icon='DUPLICATE').history_index = idx
            layout.separator()
            layout.operator("gemini.set_projection_source",
                            text="Use as Projection Source",
                            icon='MOD_UVPROJECT').history_index = idx
            layout.separator()
            layout.operator("gemini.delete_history",
                            text="Delete", icon='TRASH').history_index = idx

        context.window_manager.history_menu_index = self.history_index
        context.window_manager.popup_menu(draw_menu, title="History Options",
                                          icon='RENDER_RESULT')
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Image loading operators
# ---------------------------------------------------------------------------

_IMAGE_FILTER = "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.tga;*.exr;*.hdr;*.webp"


class GEMINI_OT_load_image_as_reference(Operator, ImportHelper):
    """Load an image file as the style reference"""
    bl_idname = "gemini.load_image_as_reference"
    bl_label = "Load Reference Image"
    bl_options = {'REGISTER'}

    filename_ext = ""
    filter_glob: StringProperty(default=_IMAGE_FILTER, options={'HIDDEN'})
    filepath: StringProperty(subtype='FILE_PATH', maxlen=1024)

    def execute(self, context):
        props = context.scene.gemini_render
        if not self.filepath:
            self.report({'WARNING'}, "No file selected")
            return {'CANCELLED'}
        try:
            image = bpy.data.images.load(self.filepath, check_existing=True)
        except RuntimeError as e:
            self.report({'ERROR'}, f"Failed to load image: {e}")
            return {'CANCELLED'}

        props.style_reference_image = image
        props.use_style_reference = True
        self.report({'INFO'}, f"Reference loaded: {image.name}")
        return {'FINISHED'}


class GEMINI_OT_load_custom_image(Operator, ImportHelper):
    """Load an image file for direct projection"""
    bl_idname = "gemini.load_custom_image"
    bl_label = "Load Custom Texture"
    bl_options = {'REGISTER'}

    filename_ext = ""
    filter_glob: StringProperty(default=_IMAGE_FILTER, options={'HIDDEN'})
    filepath: StringProperty(subtype='FILE_PATH', maxlen=1024)

    def execute(self, context):
        props = context.scene.gemini_render
        if not self.filepath:
            self.report({'WARNING'}, "No file selected")
            return {'CANCELLED'}
        try:
            image = bpy.data.images.load(self.filepath, check_existing=True)
        except RuntimeError as e:
            self.report({'ERROR'}, f"Failed to load image: {e}")
            return {'CANCELLED'}

        props.projection_image = image
        props.projection_source = 'IMAGE'
        self.report({'INFO'}, f"Custom texture loaded: {image.name}")
        return {'FINISHED'}


class GEMINI_OT_load_example_reference(Operator):
    """Create a procedural example style-reference image"""
    bl_idname = "gemini.load_example_reference"
    bl_label = "Load Example Reference"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.gemini_render
        name = "Gemini_Example_Reference"

        image = bpy.data.images.get(name)
        if image is None:
            width = height = 512
            image = bpy.data.images.new(name, width, height)

            # Vectorized gradient + noise (was a 262k-iteration Python loop).
            ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
            pixels = np.empty((height, width, 4), dtype=np.float32)
            pixels[..., 0] = xs / width * 0.8 + 0.2
            pixels[..., 1] = ys / height * 0.6 + 0.3
            pixels[..., 2] = (xs + ys) / (width + height) * 0.9 + 0.1
            pixels[..., 3] = 1.0
            noise = (np.random.random((height, width, 1)).astype(np.float32)
                     * 0.1 - 0.05)
            pixels[..., :3] = np.clip(pixels[..., :3] + noise, 0.0, 1.0)
            image.pixels.foreach_set(pixels.ravel())

        props.style_reference_image = image
        props.use_style_reference = True
        self.report({'INFO'}, "Example reference loaded")
        return {'FINISHED'}
