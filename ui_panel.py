"""
UI for TextureProjector.

Blender-native UX: property split layouts, icon-driven labels, one big
primary action, collapsible sub-panels, and a thumbnail gallery.

History item property names are kept identical to v1 so existing .blend
files keep their gallery data.
"""

import bpy
from bpy.types import Panel, PropertyGroup
from bpy.props import (StringProperty, BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, CollectionProperty, PointerProperty,
                       FloatVectorProperty)

from . import gemini_api
from . import history_previews


class GeminiRenderHistoryItem(PropertyGroup):
    """Single render history entry (names frozen for .blend compatibility)."""

    prompt: StringProperty(name="Prompt", default="")
    timestamp: StringProperty(name="Timestamp", default="")
    image_name: StringProperty(name="Image Name", default="")
    thumbnail_name: StringProperty(name="Thumbnail Name", default="")

    style_reference_used: BoolProperty(name="Style Reference Used", default=False)
    style_reference_name: StringProperty(name="Style Reference Name", default="")
    style_reference_thumbnail: StringProperty(name="Style Reference Thumbnail",
                                              default="")

    is_camera_view: BoolProperty(name="Is Camera View", default=False)

    # Viewport state
    cam_location: FloatVectorProperty(name="View Location", size=3)
    cam_rotation: FloatVectorProperty(name="View Rotation", size=4)  # Quaternion
    cam_lens: FloatProperty(name="Lens", default=50.0)
    view_distance: FloatProperty(name="View Distance", default=10.0)

    # Camera object state (for pixel-consistent camera-view restore)
    cam_obj_location: FloatVectorProperty(name="Camera Obj Location", size=3)
    cam_obj_rotation: FloatVectorProperty(name="Camera Obj Rotation", size=3)  # Euler


class GeminiRenderProperties(PropertyGroup):
    """Scene-level settings for the projector."""

    # Legacy only: old versions stored the key here (inside the .blend).
    # It is no longer shown in the UI; gemini_api.get_api_key() still reads
    # it as a last resort so old files keep working.
    api_key: StringProperty(name="API Key (legacy)", default="",
                            subtype='PASSWORD')

    model_name: EnumProperty(
        name="Model",
        description="Gemini image model",
        items=[
            ('gemini-3.1-flash-image-preview', "Gemini 3.1 Flash (Free)",
             "Default free model, fast and balanced"),
            ('gemini-3-pro-image-preview', "Gemini 3 Pro",
             "Highest quality, slower and more rate-limited"),
            ('gemini-2.5-flash-image', "Gemini 2.5 Flash",
             "Legacy compatibility model"),
        ],
        default='gemini-3.1-flash-image-preview',
    )

    prompt: StringProperty(
        name="Prompt",
        description="Describe the texture you want",
        default="Make this photorealistic with detailed materials and proper lighting",
        maxlen=1000,
    )

    render_history: CollectionProperty(type=GeminiRenderHistoryItem,
                                       name="Render History")
    history_index: IntProperty(name="History Index", default=-1)

    use_style_reference: BoolProperty(
        name="Style Reference",
        description="Guide materials/colors with a reference image",
        default=False)
    style_reference_image: PointerProperty(
        type=bpy.types.Image, name="Reference",
        description="Reference image for style/material/lighting guidance")
    use_viewport_as_reference: BoolProperty(
        name="Use Viewport as Reference",
        description="Capture a clean viewport screenshot as the style reference",
        default=False)

    show_settings: BoolProperty(name="Show Settings", default=False)

    status_text: StringProperty(name="Status", default="Ready",
                                options={'SKIP_SAVE'})
    is_rendering: BoolProperty(name="Is Rendering", default=False,
                               options={'SKIP_SAVE'})

    input_source: EnumProperty(
        name="Capture",
        description="What to capture and send to the AI",
        items=[
            ('COLOR', "Viewport Color", "Send a color capture of the view"),
            ('DEPTH', "Depth Map", "Send a normalized depth capture"),
        ],
        default='COLOR')

    projection_source: EnumProperty(
        name="Source",
        description="Where the projected texture comes from",
        items=[
            ('AI', "AI Generated", "Generate the texture with Gemini", 'SHADERFX', 0),
            ('IMAGE', "Custom Image", "Project a chosen image directly",
             'IMAGE_DATA', 1),
            ('VIEW', "Viewport Capture", "Project the current view directly",
             'RESTRICT_VIEW_OFF', 2),
            ('GRID', "Grid (Test)", "Project a wireframe grid for alignment testing",
             'MESH_GRID', 3),
        ],
        default='AI')

    projection_image: PointerProperty(
        type=bpy.types.Image, name="Image",
        description="Custom image to project onto the mesh")

    projection_bake: BoolProperty(
        name="Bake to Original UVs",
        description="Bake the projection back into the object's own UV layout",
        default=True)

    bake_uv_name: StringProperty(
        name="Bake UV",
        description="UV map to bake into. Leave empty for the first UV map "
                    "(UV0). A named map that does not exist is created",
        default="")

    mask_repair_mode: BoolProperty(
        name="Mask Repair Mode",
        description="Repair only the selected faces: they are masked and the AI "
                    "regenerates just that region of the existing texture",
        default=False)
    mask_color: FloatVectorProperty(
        name="Mask Color", subtype='COLOR', size=4,
        min=0.0, max=1.0, default=(1.0, 0.0, 0.0, 1.0))

    debug_mode: BoolProperty(
        name="Debug Mode",
        description="Keep all intermediate images in a persistent "
                    "'textures/gemini_debug_session' folder next to the .blend",
        default=False)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class BANANA_PT_render_panel(Panel):
    """Main projector panel."""
    bl_label = "Texture Projector"
    bl_idname = "BANANA_PT_render_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gemini"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gemini_render

        # API key onboarding (only blocks the AI source).
        needs_key = props.projection_source == 'AI' and not gemini_api.get_api_key()
        if needs_key:
            box = layout.box()
            box.alert = True
            box.label(text="No API key configured", icon='ERROR')
            row = box.row(align=True)
            row.operator("gemini.open_api_key_url", text="Get Key", icon='URL')
            row.operator("gemini.open_addon_prefs", text="Preferences",
                         icon='PREFERENCES')

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(props, "projection_source")

        if props.projection_source == 'AI':
            col.prop(props, "model_name")
            col.prop(props, "input_source")
            layout.separator()
            layout.label(text="Prompt", icon='TEXT')
            layout.prop(props, "prompt", text="")
        elif props.projection_source == 'IMAGE':
            col.prop(props, "projection_image")
            row = layout.row()
            row.operator("gemini.load_custom_image", text="Load Image",
                         icon='FILEBROWSER')
            if not props.projection_image:
                layout.label(text="Select an image to project", icon='INFO')

        # Primary action.
        layout.separator()
        action = layout.column(align=True)
        action.scale_y = 1.8
        action.enabled = not props.is_rendering
        labels = {
            'AI': "AI Texture Projection",
            'IMAGE': "Project Image",
            'VIEW': "Project Viewport",
            'GRID': "Grid Projection (Test)",
        }
        action.operator("gemini.texture_projection",
                        text=labels[props.projection_source],
                        icon='MOD_UVPROJECT')

        # Contextual hint (one-click semantics).
        if context.mode == 'EDIT_MESH':
            layout.label(text="Projecting selected faces", icon='FACESEL')
        elif any(o.type == 'MESH' for o in context.selected_objects):
            layout.label(text="Projecting all faces (Edit Mode to limit)",
                         icon='OBJECT_DATA')
        else:
            layout.label(text="Select a mesh object", icon='ERROR')

        if not props.is_rendering and props.status_text not in {"", "Ready"}:
            layout.label(text=props.status_text, icon='INFO')


class BANANA_PT_style_panel(Panel):
    """Style reference sub-panel."""
    bl_label = "Style Reference"
    bl_idname = "BANANA_PT_style_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gemini"
    bl_parent_id = "BANANA_PT_render_panel"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.scene.gemini_render.projection_source == 'AI'

    def draw_header(self, context):
        self.layout.prop(context.scene.gemini_render, "use_style_reference",
                         text="")

    def draw(self, context):
        layout = self.layout
        props = context.scene.gemini_render
        layout.enabled = props.use_style_reference

        layout.prop(props, "use_viewport_as_reference")
        if not props.use_viewport_as_reference:
            layout.prop(props, "style_reference_image", text="")
            row = layout.row(align=True)
            row.operator("gemini.load_image_as_reference", text="Load Image",
                         icon='FILEBROWSER')
            row.operator("gemini.load_example_reference", text="",
                         icon='SHADERFX')
        else:
            layout.label(text="A clean viewport capture will be used",
                         icon='RESTRICT_VIEW_OFF')


class BANANA_PT_options_panel(Panel):
    """Projection options sub-panel."""
    bl_label = "Options"
    bl_idname = "BANANA_PT_options_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gemini"
    bl_parent_id = "BANANA_PT_render_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.gemini_render
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(props, "projection_bake")
        sub = layout.column()
        sub.enabled = props.projection_bake
        sub.prop(props, "bake_uv_name", icon='UV')
        layout.prop(props, "mask_repair_mode")
        if props.mask_repair_mode:
            layout.prop(props, "mask_color")

        layout.separator()
        layout.prop(props, "debug_mode")
        row = layout.row(align=True)
        row.operator("gemini.open_output_dir", text="Output Folder",
                     icon='FILE_FOLDER')
        row.operator("gemini.reset_state", text="Reset", icon='FILE_REFRESH')


class BANANA_PT_history_panel(Panel):
    """Thumbnail gallery of previous generations."""
    bl_label = "Projection Gallery"
    bl_idname = "BANANA_PT_history_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gemini"
    bl_parent_id = "BANANA_PT_render_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.gemini_render

        if len(props.render_history) == 0:
            box = layout.box()
            box.label(text="No renders yet", icon='INFO')
            return

        layout.label(text=f"{len(props.render_history)} renders",
                     icon='IMAGE_DATA')

        for i, item in enumerate(reversed(props.render_history)):
            actual_index = len(props.render_history) - 1 - i

            card = layout.box()

            icon_id = history_previews.get_preview_icon_id(
                image_name=item.image_name)
            if icon_id:
                card.template_icon(icon_value=icon_id, scale=5.0)

            header = card.row(align=True)
            header.label(text=f"#{len(props.render_history) - i}  {item.timestamp}",
                         icon='TIME')

            btn_row = card.row(align=True)
            btn_row.scale_y = 1.2
            btn_row.operator("gemini.open_history_image", text="View",
                             icon='ZOOM_IN').history_index = actual_index
            btn_row.operator("gemini.use_history_prompt", text="",
                             icon='TEXT').history_index = actual_index
            btn_row.operator("gemini.set_projection_source", text="",
                             icon='MOD_UVPROJECT').history_index = actual_index
            btn_row.operator("gemini.delete_history", text="",
                             icon='TRASH').history_index = actual_index

            if item.prompt:
                preview = (item.prompt[:70] + "...") if len(item.prompt) > 70 \
                    else item.prompt
                sub = card.row()
                sub.scale_y = 0.7
                sub.label(text=preview, icon='TEXT')
