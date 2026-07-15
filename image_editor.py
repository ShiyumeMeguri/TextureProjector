"""
Image Editor integration: AI post-editing of generated (or any) images.

Supports iterative refinement, reference-guided object placement and
sketch-based inpainting. All pixel handling is numpy vectorized via
foreach_get/foreach_set (the old list(image.pixels) path copied every
pixel through Python twice).
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import bpy
from bpy.types import Panel, PropertyGroup, Operator
from bpy.props import (StringProperty, BoolProperty, CollectionProperty,
                       IntProperty, PointerProperty, FloatVectorProperty,
                       EnumProperty)


class EditHistoryItem(PropertyGroup):
    prompt: StringProperty(name="Edit Prompt", default="")
    image_name: StringProperty(name="Image Name", default="")
    timestamp: StringProperty(name="Timestamp", default="")
    has_mask: BoolProperty(name="Has Mask", default=False)


def _update_brush_settings(self, context):
    """Push panel brush settings into the active image-paint brush."""
    try:
        ts = context.tool_settings
        paint = getattr(ts, 'image_paint', None)
        if paint and paint.brush:
            paint.brush.size = self.brush_size
            paint.brush.color = self.brush_color
            if hasattr(ts, 'unified_paint_settings'):
                ts.unified_paint_settings.size = self.brush_size
                ts.unified_paint_settings.color = self.brush_color
    except Exception as e:
        print(f"[GEMINI] Brush update error: {e}")


class ImageEditorProperties(PropertyGroup):
    """Session-scoped editor state (WindowManager, never saved in .blend)."""

    edit_prompt: StringProperty(
        name="Edit Prompt",
        description="Describe what to change in the image",
        default="", maxlen=500)

    edit_history: CollectionProperty(type=EditHistoryItem, name="Edit History")
    history_index: IntProperty(name="History Index", default=-1)

    active_image: StringProperty(name="Active Image", default="")
    original_prompt: StringProperty(name="Original Prompt", default="")

    use_reference_image: BoolProperty(
        name="Reference Image",
        description="Add an object/person from a reference image to the scene",
        default=False)
    reference_image: PointerProperty(
        type=bpy.types.Image, name="Reference",
        description="Image containing the object to add (combine with "
                    "inpainting to control placement)")

    use_inpainting: BoolProperty(
        name="Inpainting",
        description="Paint a rough guide of what the AI should create",
        default=False)

    brush_size: IntProperty(
        name="Brush Size", default=50, min=1, max=500,
        update=_update_brush_settings)
    brush_color: FloatVectorProperty(
        name="Brush Color", subtype='COLOR',
        default=(1.0, 1.0, 1.0), min=0.0, max=1.0,
        update=_update_brush_settings)

    show_history: BoolProperty(name="Show History", default=False)
    is_editing: BoolProperty(name="Is Editing", default=False)

    resolution: EnumProperty(
        name="Resolution",
        description="Output resolution for the edit",
        items=[
            ('AUTO', "Auto (Match Input)", "Keep the original resolution tier"),
            ('1024', "1K", "Force 1K output"),
            ('2048', "2K", "Force 2K output"),
            ('4096', "4K", "Force 4K output"),
        ],
        default='AUTO')

    status_text: StringProperty(name="Status", default="Ready to edit")


# ---------------------------------------------------------------------------
# Pixel helpers (numpy)
# ---------------------------------------------------------------------------

def _read_pixels(image) -> np.ndarray:
    """Image pixels as (H, W, C) float32 via the C-speed foreach path."""
    w, h = image.size
    channels = image.channels
    buf = np.empty(w * h * channels, dtype=np.float32)
    image.pixels.foreach_get(buf)
    return buf.reshape(h, w, channels)


def _save_rgb_png(rgb: np.ndarray, filepath: str) -> bool:
    """Save an (H, W, 3) float array as PNG through a scratch Blender image."""
    h, w = rgb.shape[:2]
    img = bpy.data.images.new("gemini_edit_scratch", width=w, height=h, alpha=True)
    try:
        out = np.ones((h, w, 4), dtype=np.float32)
        out[..., :3] = rgb
        img.pixels.foreach_set(out.ravel())
        img.filepath_raw = filepath
        img.file_format = 'PNG'
        img.save()
        return os.path.exists(filepath)
    except Exception as e:
        print(f"[GEMINI] Failed to save guide: {e}")
        return False
    finally:
        bpy.data.images.remove(img)


def _save_image_to_path(image, filepath: str) -> bool:
    """Save a Blender image datablock to disk without altering it."""
    original_filepath = image.filepath_raw
    original_format = image.file_format
    try:
        image.filepath_raw = filepath
        image.file_format = 'PNG'
        try:
            # save() writes raw data (no view transform darkening).
            image.save()
        except RuntimeError:
            image.save_render(filepath)
        if not os.path.exists(filepath):
            image.save_render(filepath)
        return os.path.exists(filepath)
    except Exception as e:
        print(f"[GEMINI] Image save failed: {e}")
        return False
    finally:
        image.filepath_raw = original_filepath
        image.file_format = original_format


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class BANANA_PT_image_editor_panel(Panel):
    """AI edit panel inside the Image Editor."""
    bl_label = "AI Edit"
    bl_idname = "BANANA_PT_image_editor_panel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Gemini"

    @classmethod
    def poll(cls, context):
        sima = context.space_data
        return sima and sima.image is not None

    def draw(self, context):
        layout = self.layout
        sima = context.space_data
        image = sima.image
        props = context.window_manager.nano_banana_editor

        # Image info.
        box = layout.box()
        row = box.row()
        row.label(text=image.name, icon='IMAGE_DATA')
        box.label(text=f"{image.size[0]} x {image.size[1]}", icon='TEXTURE')

        if image.type == 'RENDER_RESULT':
            box.label(text="Render Result is read-only", icon='INFO')
            row = box.row()
            row.scale_y = 1.2
            row.operator("nano_banana.convert_render_result",
                         text="Convert to Editable", icon='IMAGE_RGB')

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(props, "resolution")

        layout.separator()
        layout.label(text="Edit Instructions", icon='TEXT')
        layout.prop(props, "edit_prompt", text="")

        # Inpainting.
        box = layout.box()
        box.prop(props, "use_inpainting", icon='BRUSH_DATA')
        if props.use_inpainting:
            is_paint_mode = sima.mode == 'PAINT'
            row = box.row()
            row.scale_y = 1.4
            if is_paint_mode:
                row.operator("nano_banana.apply_inpaint",
                             text="Apply Drawing", icon='CHECKMARK')
            else:
                row.operator("nano_banana.switch_to_paint",
                             text="Draw", icon='BRUSH_DATA')
            settings = box.column(align=True)
            settings.enabled = is_paint_mode
            settings.prop(props, "brush_size")
            settings.prop(props, "brush_color", text="")

        # Reference.
        box = layout.box()
        box.prop(props, "use_reference_image", icon='IMAGE_REFERENCE')
        if props.use_reference_image:
            row = box.row(align=True)
            row.prop_search(props, "reference_image", bpy.data, "images",
                            text="", icon='IMAGE_DATA')
            row.operator("nano_banana.load_reference_image", text="",
                         icon='FILEBROWSER')
            if props.reference_image:
                row.operator("nano_banana.unlink_reference_image", text="",
                             icon='X')
                hint = ("Draw where to place it (optional)"
                        if props.use_inpainting
                        else "Describe what/where to add")
                box.label(text=hint, icon='INFO')

        # Primary action.
        layout.separator()
        action = layout.column(align=True)
        action.scale_y = 1.8
        if props.is_editing:
            action.enabled = False
            action.operator("nano_banana.apply_edit", text="Processing...",
                            icon='TIME')
        elif props.edit_prompt.strip() or props.use_reference_image:
            action.operator("nano_banana.apply_edit", text="Apply AI Edit",
                            icon='SHADERFX')
        else:
            action.enabled = False
            action.operator("nano_banana.apply_edit", text="Enter a prompt",
                            icon='INFO')

        # Secondary actions.
        col = layout.column(align=True)
        col.operator("nano_banana.finalize_composite",
                     text="Finalize Composite", icon='NODE_COMPOSITING')
        row = layout.row(align=True)
        row.operator("nano_banana.rerender_image", text="Re-render",
                     icon='FILE_REFRESH')
        row.operator("nano_banana.save_version", text="Save Version",
                     icon='DUPLICATE')

        layout.separator()
        layout.label(text=props.status_text, icon='INFO')

        # History.
        if len(props.edit_history) > 0:
            layout.prop(props, "show_history",
                        text=f"History ({len(props.edit_history)})",
                        toggle=True, icon='TIME')
            if props.show_history:
                box = layout.box()
                for i, item in enumerate(reversed(props.edit_history)):
                    actual_index = len(props.edit_history) - 1 - i
                    row = box.row(align=True)
                    row.scale_y = 0.8
                    col = row.column()
                    col.label(text=f"#{len(props.edit_history) - i}  {item.timestamp}")
                    preview = (item.prompt[:40] + "...") if len(item.prompt) > 40 \
                        else item.prompt
                    col.label(text=preview, icon='TEXT')
                    row.operator("nano_banana.load_history_edit", text="",
                                 icon='LOOP_BACK').history_index = actual_index


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _resolve_edit_resolution(resolution: str, original_size) -> tuple:
    """Map the resolution option to target dimensions (AUTO keeps the tier)."""
    if resolution in {'1024', '2048', '4096'}:
        size = int(resolution)
        return size, size
    max_dim = max(original_size[0], original_size[1], 1)
    if max_dim > 2048:
        return 4096, 4096
    if max_dim > 1024:
        return 2048, 2048
    return 1024, 1024


class NANO_BANANA_OT_apply_edit(Operator):
    """Send the current image to the AI with the edit instructions.

    Fully synchronous by design (see gemini.texture_projection): the UI
    blocks while the API call runs, and nothing can deadlock.
    """
    bl_idname = "nano_banana.apply_edit"
    bl_label = "Apply AI Edit"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.window_manager.nano_banana_editor
        sima = context.space_data
        image = sima.image

        if not image:
            self.report({'ERROR'}, "No image in the editor")
            return {'CANCELLED'}
        if not props.edit_prompt.strip() and not props.use_reference_image:
            self.report({'ERROR'}, "Enter edit instructions or pick a reference")
            return {'CANCELLED'}

        from . import gemini_api
        from . import threading_utils
        api_key = gemini_api.get_api_key()
        if not api_key:
            self.report({'ERROR'},
                        "No API key. Set it in the add-on preferences.")
            return {'CANCELLED'}

        props.is_editing = True
        props.status_text = "Editing..."
        temp_dir = None
        original_image_name = image.name

        try:
            temp_dir = tempfile.mkdtemp(prefix="nano_banana_edit_")
            image_path = os.path.join(temp_dir, "original.png")
            if not _save_image_to_path(image, image_path):
                raise RuntimeError("Could not save the current image to disk")

            reference_path = None
            if props.use_reference_image and props.reference_image:
                reference_path = os.path.join(temp_dir, "reference.png")
                if not _save_image_to_path(props.reference_image, reference_path):
                    reference_path = None

            inpaint_guide_path = None
            if props.use_inpainting:
                inpaint_guide_path = self._extract_inpaint_guide(image, temp_dir)
                if not inpaint_guide_path:
                    self.report({'WARNING'},
                                "No drawing found. Click Draw and paint the area.")
                    return {'CANCELLED'}

            width, height = _resolve_edit_resolution(
                props.resolution, (image.size[0], image.size[1]))

            api_client = gemini_api.GeminiAPI(api_key)
            image_data, mime = api_client.edit_image(
                image_path=image_path,
                edit_prompt=props.edit_prompt,
                mask_path=inpaint_guide_path,
                reference_image_path=reference_path,
                width=width,
                height=height,
            )
            print(f"[GEMINI] Edit complete: {len(image_data)} bytes ({mime})")

            # Load the result and show it in every open Image Editor.
            result_path = threading_utils.build_output_filepath(
                f"{original_image_name}_edit")
            with open(result_path, 'wb') as f:
                f.write(image_data)

            timestamp = datetime.now().strftime("%H%M%S")
            new_image = bpy.data.images.load(result_path, check_existing=False)
            new_image.name = f"{original_image_name}_edit_{timestamp}"
            new_image.filepath_raw = result_path
            if hasattr(new_image, 'colorspace_settings'):
                new_image.colorspace_settings.name = 'sRGB'

            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        for space in area.spaces:
                            if space.type == 'IMAGE_EDITOR':
                                space.image = new_image
                                space.mode = 'VIEW'
                        area.tag_redraw()

            item = props.edit_history.add()
            item.prompt = props.edit_prompt
            item.image_name = original_image_name
            item.timestamp = datetime.now().strftime("%H:%M:%S")
            item.has_mask = bool(inpaint_guide_path)

            props.status_text = "Edit complete"
            self.report({'INFO'}, f"Edit finished: {new_image.name}")
            return {'FINISHED'}

        except Exception as e:
            props.status_text = f"Error: {e}"
            self.report({'ERROR'}, str(e))
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        finally:
            props.is_editing = False
            if temp_dir and os.path.exists(temp_dir) \
                    and os.path.basename(temp_dir).startswith("nano_banana_edit_"):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _extract_inpaint_guide(self, image, temp_dir):
        """Save the user's painted guide (any non-black strokes) as a PNG."""
        try:
            image.update()
            pixels = _read_pixels(image)
            if pixels.shape[2] < 3:
                return None
            rgb = pixels[..., :3]
            painted = int((rgb > 0.05).any(axis=2).sum())
            if painted < 50:
                return None
            guide_path = os.path.join(temp_dir, "inpaint_guide.png")
            if _save_rgb_png(rgb, guide_path):
                return guide_path
            return None
        except Exception as e:
            print(f"[GEMINI] Guide extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None


class NANO_BANANA_OT_finalize_composite(Operator):
    """Unify colors, contrast and lighting across the whole image"""
    bl_idname = "nano_banana.finalize_composite"
    bl_label = "Finalize Composite"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.window_manager.nano_banana_editor
        sima = context.space_data
        if not sima or sima.type != 'IMAGE_EDITOR' or not sima.image:
            self.report({'ERROR'}, "No image to finalize")
            return {'CANCELLED'}
        props.edit_prompt = "[FINALIZE_COMPOSITE]"
        bpy.ops.nano_banana.apply_edit()
        self.report({'INFO'}, "Finalizing composite...")
        return {'FINISHED'}


class NANO_BANANA_OT_rerender_image(Operator):
    """Generate a new variation using the previous edit settings"""
    bl_idname = "nano_banana.rerender_image"
    bl_label = "Re-render Image"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.window_manager.nano_banana_editor
        if not context.space_data or not context.space_data.image:
            self.report({'ERROR'}, "No image in the editor")
            return {'CANCELLED'}
        if len(props.edit_history) == 0:
            self.report({'INFO'}, "No edit history yet — use Apply AI Edit first")
            return {'CANCELLED'}
        props.edit_prompt = props.edit_history[-1].prompt
        bpy.ops.nano_banana.apply_edit()
        self.report({'INFO'}, "Re-rendering with previous settings...")
        return {'FINISHED'}


class NANO_BANANA_OT_save_version(Operator):
    """Save the current image as a new version"""
    bl_idname = "nano_banana.save_version"
    bl_label = "Save Version"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sima = context.space_data
        if not sima or not sima.image:
            self.report({'ERROR'}, "No image in the editor")
            return {'CANCELLED'}
        props = context.window_manager.nano_banana_editor
        new_image = sima.image.copy()
        new_image.name = f"{sima.image.name}_v{len(props.edit_history) + 1}"
        self.report({'INFO'}, f"Saved as {new_image.name}")
        return {'FINISHED'}


class NANO_BANANA_OT_load_history_edit(Operator):
    """Load a previous edit"""
    bl_idname = "nano_banana.load_history_edit"
    bl_label = "Load History Edit"
    bl_options = {'REGISTER'}

    history_index: IntProperty()

    def execute(self, context):
        props = context.window_manager.nano_banana_editor
        if not (0 <= self.history_index < len(props.edit_history)):
            self.report({'ERROR'}, "Invalid history index")
            return {'CANCELLED'}
        item = props.edit_history[self.history_index]
        props.edit_prompt = item.prompt
        if item.image_name in bpy.data.images:
            context.space_data.image = bpy.data.images[item.image_name]
            self.report({'INFO'}, f"Loaded edit from {item.timestamp}")
        else:
            self.report({'WARNING'}, f"Image {item.image_name} not found")
        return {'FINISHED'}


class NANO_BANANA_OT_convert_render_result(Operator):
    """Convert the read-only Render Result into an editable image"""
    bl_idname = "nano_banana.convert_render_result"
    bl_label = "Convert Render Result"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import time
        sima = context.space_data
        if not sima or not sima.image:
            return {'CANCELLED'}
        image = sima.image
        if image.type != 'RENDER_RESULT':
            self.report({'INFO'}, "Image is already editable")
            return {'FINISHED'}

        try:
            temp_path = os.path.join(tempfile.gettempdir(),
                                     f"render_convert_{int(time.time())}.png")
            scene = context.scene
            settings = scene.render.image_settings
            saved = (settings.file_format, settings.color_mode,
                     settings.color_depth)
            settings.file_format = 'PNG'
            settings.color_mode = 'RGBA'
            settings.color_depth = '8'
            try:
                # Bakes the view transform in — matches what the user sees.
                image.save_render(temp_path, scene=scene)
            finally:
                (settings.file_format, settings.color_mode,
                 settings.color_depth) = saved

            new_image = bpy.data.images.load(temp_path)
            new_image.name = f"Editable_Render_{len(bpy.data.images)}"
            new_image.filepath_raw = temp_path
            sima.image = new_image
            self.report({'INFO'}, f"Converted to {new_image.name}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Conversion failed: {e}")
            return {'CANCELLED'}


class NANO_BANANA_OT_switch_to_paint(Operator):
    """Switch the Image Editor into Paint mode with the guide brush"""
    bl_idname = "nano_banana.switch_to_paint"
    bl_label = "Start Painting"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.window_manager.nano_banana_editor
        sima = context.space_data
        if not sima or not sima.image:
            self.report({'ERROR'}, "No image in the editor")
            return {'CANCELLED'}

        if sima.image.type == 'RENDER_RESULT':
            image_before = sima.image
            bpy.ops.nano_banana.convert_render_result()
            if sima.image == image_before:
                return {'CANCELLED'}

        try:
            sima.mode = 'PAINT'
            ts = context.tool_settings
            paint = getattr(ts, 'image_paint', None)
            if paint:
                if not paint.brush and 'Draw' in bpy.data.brushes:
                    paint.brush = bpy.data.brushes['Draw']
                if paint.brush:
                    paint.brush.size = props.brush_size
                    paint.brush.color = props.brush_color
                    paint.brush.strength = 1.0
                    if hasattr(ts, 'unified_paint_settings'):
                        ts.unified_paint_settings.size = props.brush_size
                        ts.unified_paint_settings.color = props.brush_color
            self.report({'INFO'}, "Paint mode: draw your guide")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed: {e}")
            return {'CANCELLED'}


class NANO_BANANA_OT_apply_inpaint(Operator):
    """Keep the drawing on a working copy and back up the original"""
    bl_idname = "nano_banana.apply_inpaint"
    bl_label = "Apply Drawing"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sima = context.space_data
        if not sima or not sima.image:
            return {'CANCELLED'}

        try:
            from . import threading_utils
            current_image = sima.image
            props = context.window_manager.nano_banana_editor
            current_image.update()

            # Back up the painted state for the history.
            history_image = current_image.copy()
            history_image.name = (f"{current_image.name}_history_"
                                  f"{len(props.edit_history)}")
            try:
                threading_utils.save_blender_image(history_image,
                                                   history_image.name)
            except Exception as e:
                print(f"[GEMINI] History image save warning: {e}")

            item = props.edit_history.add()
            item.prompt = "[Inpaint sketch applied]"
            item.image_name = history_image.name
            item.timestamp = datetime.now().strftime("%H:%M:%S")
            item.has_mask = True

            # Working copy carries the drawing forward.
            new_image = current_image.copy()
            new_image.name = f"{current_image.name}_inpaint"
            pixels = _read_pixels(current_image)
            new_image.pixels.foreach_set(pixels.ravel())
            new_image.update()
            try:
                threading_utils.save_blender_image(new_image, new_image.name)
            except Exception as e:
                print(f"[GEMINI] Inpaint copy save warning: {e}")

            sima.image = new_image
            sima.mode = 'VIEW'
            self.report({'INFO'},
                        f"Original backed up; drawing saved as {new_image.name}")
            return {'FINISHED'}
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Failed: {e}")
            return {'CANCELLED'}


class NANO_BANANA_OT_load_reference_image(Operator):
    """Load a reference image without switching the editor view"""
    bl_idname = "nano_banana.load_reference_image"
    bl_label = "Load Reference Image"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_image: BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = context.window_manager.nano_banana_editor
        if not self.filepath:
            self.report({'WARNING'}, "No file selected")
            return {'CANCELLED'}
        try:
            image = bpy.data.images.load(self.filepath, check_existing=True)
        except RuntimeError as e:
            self.report({'ERROR'}, f"Failed to load: {e}")
            return {'CANCELLED'}
        props.reference_image = image
        self.report({'INFO'}, f"Loaded: {image.name}")
        return {'FINISHED'}


class NANO_BANANA_OT_unlink_reference_image(Operator):
    """Remove the reference image"""
    bl_idname = "nano_banana.unlink_reference_image"
    bl_label = "Remove Reference"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.window_manager.nano_banana_editor.reference_image = None
        self.report({'INFO'}, "Reference removed")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    EditHistoryItem,
    ImageEditorProperties,
    BANANA_PT_image_editor_panel,
    NANO_BANANA_OT_apply_edit,
    NANO_BANANA_OT_finalize_composite,
    NANO_BANANA_OT_convert_render_result,
    NANO_BANANA_OT_switch_to_paint,
    NANO_BANANA_OT_apply_inpaint,
    NANO_BANANA_OT_load_reference_image,
    NANO_BANANA_OT_unlink_reference_image,
    NANO_BANANA_OT_rerender_image,
    NANO_BANANA_OT_save_version,
    NANO_BANANA_OT_load_history_edit,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.nano_banana_editor = PointerProperty(
        type=ImageEditorProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.WindowManager, 'nano_banana_editor'):
        del bpy.types.WindowManager.nano_banana_editor
