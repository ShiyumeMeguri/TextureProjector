bl_info = {
    "name": "TextureProjector",
    "blender": (4, 2, 0),
    "category": "3D View", 
    "version": (1, 0, 0),
    "author": "ShiyumeMeguri",
    "description": "AI-assisted texture projection and repair suite. Seamlessly project AI-generated edits and textures onto 3D meshes with automated UV management.",
    "location": "3D Viewport > N Panel > Gemini",
    "doc_url": "https://github.com/ShiyumeMeguri/TextureProjector",
    "tracker_url": "https://github.com/ShiyumeMeguri/TextureProjector/issues",
}

import bpy
from bpy.types import AddonPreferences
from bpy.props import StringProperty
import sys
import subprocess
import importlib

def ensure_dependencies():
    """Ensure required external libraries are installed"""
    required_libs = [
        ("PIL", "Pillow"),
        ("google.genai", "google-genai"),
        ("requests", "requests")
    ]
    
    for module_name, package_name in required_libs:
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"[NANO BANANA] {package_name} not found. Auto-installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"[NANO BANANA] Successfully installed {package_name}")
                
                # Invalidate caches to ensure newly installed package is found
                importlib.invalidate_caches()
            except Exception as e:
                print(f"[NANO BANANA] Failed to auto-install {package_name}: {e}")

# Ensure dependencies before importing local modules
ensure_dependencies()


# I reload modules for development
if "bpy" in locals():
    import importlib
    if "ui_panel" in locals():
        importlib.reload(ui_panel)
    if "operators" in locals():
        importlib.reload(operators)
    if "depth_utils" in locals():
        importlib.reload(depth_utils)
    if "projection_utils" in locals():
        importlib.reload(projection_utils)
    if "gemini_api" in locals():
        importlib.reload(gemini_api)
    if "threading_utils" in locals():
        importlib.reload(threading_utils)
    from .threading_utils import BlenderThreadManager, ProjectionRenderThread
    if "image_editor" in locals():
        importlib.reload(image_editor)
    if "image_edit_thread" in locals():
        importlib.reload(image_edit_thread)

# I import our modules
from . import ui_panel
from . import operators
from . import depth_utils
from . import projection_utils
from . import gemini_api
from . import threading_utils
from . import image_editor
from . import image_edit_thread

class NanoBananaPreferences(AddonPreferences):
    bl_idname = __name__

    api_key: StringProperty(
        name="API Key",
        description="AI API Key for texture generation and repair",
        default="",
        subtype='PASSWORD',
    )

    use_system_prompts: bpy.props.BoolProperty(
        name="Use Optimized System Prompts",
        description="Enable specialized system prompts for better quality (Depth/Mask adherence). If disabled, only your raw prompt is sent.",
        default=True
    )

    def draw(self, context):
        layout = self.layout
        
        # API Key section
        box = layout.box()
        box.label(text="API Configuration:", icon='KEYFRAME_HLT')
        box.prop(self, "api_key")
        
        # System Prompt Config
        box.label(text="Prompt Engineering:", icon='MODIFIER')
        row = box.row()
        row.prop(self, "use_system_prompts", text="Enable System Prompts")
        if not self.use_system_prompts:
            row.label(text="(Raw Input Only)", icon='FILE_TEXT')
        
        # I debug section
        box = layout.box()
        box.label(text="Debug Tools:", icon='TOOL_SETTINGS')
        
        row = box.row(align=True)

        try:
            if hasattr(bpy.types, 'GEMINI_OT_reset_state'):
                row.operator("gemini.reset_state", text="Reset UI State", icon='FILE_REFRESH')
            if hasattr(bpy.types, 'GEMINI_OT_open_console'):  
                row.operator("gemini.open_console", text="Open Console", icon='CONSOLE')
        except:
            row.label(text="Debug operators not available", icon='INFO')
            
        box.label(text="Note: Debug tools are also available in Blender's Console", icon='INFO')

# I registration - Core classes first
core_classes = (
    NanoBananaPreferences,
    ui_panel.GeminiRenderHistoryItem,
    ui_panel.GeminiRenderProperties,
    ui_panel.BANANA_PT_render_panel,
    ui_panel.BANANA_PT_history_panel,
    operators.GEMINI_OT_load_history,
    operators.GEMINI_OT_delete_history,
    operators.GEMINI_OT_use_history_prompt,
    operators.GEMINI_OT_use_history_style,
    operators.GEMINI_OT_use_history_both,
    operators.GEMINI_OT_history_context_menu,
    operators.GEMINI_OT_open_history_image,
    operators.GEMINI_OT_set_projection_source,
    operators.GEMINI_OT_load_image_as_reference,
    operators.GEMINI_OT_load_custom_image,
    operators.GEMINI_OT_load_example_reference,
    operators.GEMINI_OT_texture_projection,
    operators.GEMINI_OT_stop_render,
    operators.GEMINI_OT_open_api_key_url,
    operators.GEMINI_OT_validate_api_key,
)

# I optional debug classes (register separately to avoid conflicts)
debug_classes = (
    operators.GEMINI_OT_reset_state,
    operators.GEMINI_OT_open_console,
    operators.GEMINI_OT_debug_next,
)

# I all classes combined
classes = core_classes + debug_classes

def register():
    # I register core classes first
    for cls in core_classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as e:
            print(f"Error registering core class {cls}: {e}")
    
    # I try to register debug classes (optional)
    for cls in debug_classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as e:
            print(f"Warning: Could not register debug class {cls}: {e}")
            # I continue without debug classes if they fail
    
    # I register Image Editor module
    try:
        image_editor.register()
        print(" [NANO BANANA] Image Editor panel registered")
    except Exception as e:
        print(f"Warning: Could not register Image Editor: {e}")
    
    # I add properties to scene
    bpy.types.Scene.gemini_render = bpy.props.PointerProperty(type=ui_panel.GeminiRenderProperties)
    
    # I add properties to window manager for context menus
    bpy.types.WindowManager.history_menu_index = bpy.props.IntProperty(
        name="History Menu Index",
        description="Index for history context menu",
        default=0
    )
    
    # Register load handler to reset status
    from bpy.app.handlers import load_post
    if _reset_is_rendering not in load_post:
        load_post.append(_reset_is_rendering)

def _reset_is_rendering(dummy1=None, dummy2=None):
    """Reset rendering status on load to prevent infinite 'Processing' state"""
    import bpy
    for scene in bpy.data.scenes:
        if hasattr(scene, 'gemini_render'):
            scene.gemini_render.is_rendering = False
            scene.gemini_render.status_text = "Ready to render"
    print(" [GEMINI] Reset rendering status on file load")

def unregister():
    # I stop any background threads
    try:
        threading_utils.stop_thread_manager()
    except:
        pass
    
    # I unregister Image Editor module
    try:
        image_editor.unregister()
    except:
        pass
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # I remove properties from scene
    if hasattr(bpy.types.Scene, 'gemini_render'):
        del bpy.types.Scene.gemini_render
    
    # I remove properties from window manager
    if hasattr(bpy.types.WindowManager, 'history_menu_index'):
        del bpy.types.WindowManager.history_menu_index
        
    # Remove load handler
    from bpy.app.handlers import load_post
    if _reset_is_rendering in load_post:
        load_post.remove(_reset_is_rendering)

if __name__ == "__main__":
    register()
