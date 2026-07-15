"""
Synchronous pipeline utilities for TextureProjector.

DELIBERATELY single-threaded. Every attempt to run the result pipeline
off the main thread deadlocked Blender sooner or later:
  - blocking bake/render operators from bpy.app.timers  -> intermittent
  - the same with a window temp_override                 -> near-always
  - a modal-operator queue runner                        -> still flaky
The whole generate -> apply -> bake flow therefore runs inline in the
operator's execute(). The UI blocks for the duration of the API call and
the bake — and nothing can deadlock. Do not re-introduce threads here.

(The file name is kept for import stability.)
"""

import os
import re
import shutil
import tempfile
import time

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
# Error display
# ---------------------------------------------------------------------------

def show_error_popup(message: str, title: str = "Texture Projector") -> None:
    """Show an error as a popup at the mouse cursor (main thread only)."""
    import textwrap
    lines = []
    for paragraph in str(message).splitlines():
        lines.extend(textwrap.wrap(paragraph, width=64) or [""])

    def draw(menu, _context):
        col = menu.layout.column()
        for line in lines[:20]:
            col.label(text=line)

    try:
        bpy.context.window_manager.popup_menu(draw, title=title, icon='ERROR')
    except Exception as e:
        print(f"[GEMINI] Could not show error popup: {e}")


# ---------------------------------------------------------------------------
# Result image loading + history
# ---------------------------------------------------------------------------

def _history_limit() -> int:
    from .gemini_api import get_prefs
    prefs = get_prefs()
    return int(getattr(prefs, 'history_limit', 10) or 10)


def load_result_image(image_data: bytes, image_name: str = "AI_Result",
                      user_prompt: str = "", cam_data: dict = None):
    """Write image bytes to disk, load into Blender, record history."""
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
# Applying the result
# ---------------------------------------------------------------------------

def apply_result_to_materials(target_objects_data, res_img) -> None:
    """Feed the result image into each object's unique projection material."""
    processed = set()
    for data in target_objects_data:
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


def bake_targets(context, target_objects_data, mask_repair_data, res_img) -> None:
    """Bake every projected object (normal and mask-repair paths)."""
    for data in target_objects_data:
        obj = bpy.data.objects.get(data['object_name'])
        if not obj:
            continue

        original_name = data.get('original_object_name')
        if original_name and mask_repair_data:
            tex_name = mask_repair_data['original_textures'].get(original_name)
            original_tex = bpy.data.images.get(tex_name)
            if original_tex:
                projection_utils.perform_projection_bake(
                    context=context, obj=obj,
                    texture_node_name=data.get('image_node_name',
                                               projection_utils.IMAGE_NODE_NAME),
                    target_image=original_tex,
                    src_uv_name=data['src_uv_name'],
                    dest_uv_name=data.get('dest_uv_name', "UVMap"),
                    is_mask_repair=True,
                    original_obj=bpy.data.objects.get(original_name),
                    search_img=res_img)
            continue

        if '_MaskTemp' in obj.name:
            continue

        safe_hash = abs(hash(obj.name)) % 100000
        baked_name = f"{obj.name[:30]}_{safe_hash}_{int(time.time())}_Baked_AI"
        if baked_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[baked_name])
        baked_img = bpy.data.images.new(baked_name, res_img.size[0], res_img.size[1])

        ok = projection_utils.perform_projection_bake(
            context=context, obj=obj,
            texture_node_name=data.get('image_node_name',
                                       projection_utils.IMAGE_NODE_NAME),
            target_image=baked_img,
            src_uv_name=data['src_uv_name'],
            dest_uv_name=data.get('dest_uv_name', "UVMap"),
            search_img=res_img)
        if ok:
            baked_path = save_blender_image(baked_img, baked_name)
            print(f"[GEMINI] Saved baked texture: {baked_path}")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def finalize_projection(registry, debug_mode=False, rollback=False) -> None:
    """Tear down everything a projection run created.

    rollback=True additionally undoes the material assignment (used when
    no valid result was applied — prevents black projection materials).
    """
    try:
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError:
        pass

    if rollback:
        projection_utils.rollback_projection_materials(registry)

    for obj_name in registry.get('temp_objects', []):
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

    for mat_name in registry.get('temp_materials', []):
        try:
            mat = bpy.data.materials.get(mat_name)
            if mat:
                bpy.data.materials.remove(mat)
        except Exception as e:
            print(f"[GEMINI] Temp material cleanup error: {e}")

    temp_dir = registry.get('temp_dir')
    if temp_dir and os.path.exists(temp_dir):
        try:
            if debug_mode:
                print(f"[GEMINI] Debug mode: keeping {temp_dir}")
            elif os.path.basename(temp_dir).startswith("gemini_proj_"):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError as e:
            print(f"[GEMINI] Temp dir cleanup error: {e}")

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
        print(f"[GEMINI] Selection restore error: {e}")
