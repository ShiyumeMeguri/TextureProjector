bl_info = {
    "name": "TextureProjector",
    "blender": (4, 2, 0),
    "category": "3D View",
    "version": (2, 0, 0),
    "author": "ShiyumeMeguri",
    "description": "AI texture projection with Gemini: capture, generate, "
                   "project and bake in one click. No external dependencies.",
    "location": "3D Viewport > N Panel > Gemini",
    "doc_url": "https://github.com/ShiyumeMeguri/TextureProjector",
    "tracker_url": "https://github.com/ShiyumeMeguri/TextureProjector/issues",
}

# Development reload support.
if "bpy" in locals():
    import importlib
    for _mod_name in ("gemini_api", "depth_utils", "projection_utils",
                      "threading_utils", "history_previews", "operators",
                      "ui_panel", "image_editor", "image_edit_thread"):
        if _mod_name in locals():
            importlib.reload(locals()[_mod_name])

import bpy
from bpy.app.handlers import persistent
from bpy.types import AddonPreferences
from bpy.props import StringProperty, BoolProperty, IntProperty

from . import gemini_api
from . import depth_utils
from . import projection_utils
from . import threading_utils
from . import history_previews
from . import operators
from . import ui_panel
from . import image_editor
from . import image_edit_thread


class NanoBananaPreferences(AddonPreferences):
    bl_idname = __name__

    api_key: StringProperty(
        name="API Key",
        description="Google Gemini API key (stored in preferences, never in "
                    ".blend files). The GEMINI_API_KEY environment variable "
                    "takes priority if set",
        default="",
        subtype='PASSWORD',
    )

    use_system_prompts: BoolProperty(
        name="Use Optimized System Prompts",
        description="Wrap requests in specialized system prompts (editable in "
                    "system_prompts.json). Disable to send only your raw prompt",
        default=True,
    )

    history_limit: IntProperty(
        name="Gallery Size",
        description="Maximum number of renders kept in the projection gallery",
        default=10, min=1, max=100,
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="API Configuration", icon='KEYINGSET')
        import os
        if os.environ.get('GEMINI_API_KEY', '').strip():
            box.label(text="Using GEMINI_API_KEY environment variable",
                      icon='CHECKMARK')
        box.prop(self, "api_key")
        row = box.row(align=True)
        row.operator("gemini.open_api_key_url", text="Get a Free Key", icon='URL')
        row.operator("gemini.validate_api_key", text="Test Key",
                     icon='CHECKMARK')

        box = layout.box()
        box.label(text="Behavior", icon='PREFERENCES')
        box.prop(self, "use_system_prompts")
        box.prop(self, "history_limit")

        box = layout.box()
        box.label(text="Tools", icon='TOOL_SETTINGS')
        row = box.row(align=True)
        row.operator("gemini.open_output_dir", text="Output Folder",
                     icon='FILE_FOLDER')
        row.operator("gemini.reset_state", text="Reset State",
                     icon='FILE_REFRESH')
        row.operator("gemini.open_console", text="Console", icon='CONSOLE')


classes = (
    NanoBananaPreferences,
    # Property groups first.
    ui_panel.GeminiRenderHistoryItem,
    ui_panel.GeminiRenderProperties,
    # Operators.
    operators.GEMINI_OT_texture_projection,
    operators.GEMINI_OT_stop_render,
    operators.GEMINI_OT_reset_state,
    operators.GEMINI_OT_open_console,
    operators.GEMINI_OT_open_api_key_url,
    operators.GEMINI_OT_open_addon_prefs,
    operators.GEMINI_OT_validate_api_key,
    operators.GEMINI_OT_open_output_dir,
    operators.GEMINI_OT_load_history,
    operators.GEMINI_OT_open_history_image,
    operators.GEMINI_OT_delete_history,
    operators.GEMINI_OT_use_history_prompt,
    operators.GEMINI_OT_use_history_style,
    operators.GEMINI_OT_use_history_both,
    operators.GEMINI_OT_set_projection_source,
    operators.GEMINI_OT_history_context_menu,
    operators.GEMINI_OT_load_image_as_reference,
    operators.GEMINI_OT_load_custom_image,
    operators.GEMINI_OT_load_example_reference,
    # Panels (parents before children).
    ui_panel.BANANA_PT_render_panel,
    ui_panel.BANANA_PT_style_panel,
    ui_panel.BANANA_PT_options_panel,
    ui_panel.BANANA_PT_history_panel,
)


@persistent
def _reset_runtime_state(dummy1=None, dummy2=None):
    """Reset addon runtime state whenever a .blend file is loaded.

    Fixes the historical 'projection bakes black after switching files'
    lifecycle bug: stale threads/timers from the previous file are dropped.
    """
    threading_utils.reset_threading_state()
    operators.GEMINI_OT_texture_projection.current_thread = None
    for scene in bpy.data.scenes:
        props = getattr(scene, 'gemini_render', None)
        if props is not None:
            props.is_rendering = False
            props.status_text = "Ready"
    print("[GEMINI] Runtime state reset on file load")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    image_editor.register()

    bpy.types.Scene.gemini_render = bpy.props.PointerProperty(
        type=ui_panel.GeminiRenderProperties)
    bpy.types.WindowManager.history_menu_index = bpy.props.IntProperty(
        name="History Menu Index", default=0)

    history_previews.init_previews()

    from bpy.app.handlers import load_post
    if _reset_runtime_state not in load_post:
        load_post.append(_reset_runtime_state)


def unregister():
    from bpy.app.handlers import load_post
    if _reset_runtime_state in load_post:
        load_post.remove(_reset_runtime_state)

    try:
        threading_utils.stop_all_projection_threads()
        threading_utils.stop_thread_manager()
    except Exception as e:
        print(f"[GEMINI] Thread shutdown warning: {e}")

    history_previews.clear_previews()

    image_editor.unregister()

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError as e:
            print(f"[GEMINI] Unregister warning for {cls.__name__}: {e}")

    if hasattr(bpy.types.Scene, 'gemini_render'):
        del bpy.types.Scene.gemini_render
    if hasattr(bpy.types.WindowManager, 'history_menu_index'):
        del bpy.types.WindowManager.history_menu_index


if __name__ == "__main__":
    register()
