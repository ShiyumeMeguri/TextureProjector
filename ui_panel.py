import bpy
from bpy.types import PropertyGroup, Panel
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty, IntProperty, CollectionProperty, PointerProperty, FloatVectorProperty

class GeminiRenderHistoryItem(PropertyGroup):
    """Single render history entry with visual preview"""
    
    prompt: StringProperty(
        name="Prompt",
        description="Prompt used for this render",
        default=""
    )
    
    timestamp: StringProperty(
        name="Timestamp", 
        description="When this render was created",
        default=""
    )
    
    image_name: StringProperty(
        name="Image Name",
        description="Name of the generated image in Blender",
        default=""
    )
    
    # Visual preview data
    thumbnail_name: StringProperty(
        name="Thumbnail Name",
        description="Name of thumbnail image in bpy.data.images",
        default=""
    )
    
    # Style reference data
    style_reference_used: BoolProperty(
        name="Style Reference Used",
        description="Whether style reference was used for this render",
        default=False
    )
    
    style_reference_name: StringProperty(
        name="Style Reference Name",
        description="Name of style reference image used",
        default=""
    )
    
    style_reference_thumbnail: StringProperty(
        name="Style Reference Thumbnail",
        description="Name of style reference thumbnail in bpy.data.images",
        default=""
    )
    
    is_camera_view: BoolProperty(name="Is Camera View", default=False)

class GeminiRenderProperties(PropertyGroup):
    """Properties for Gemini Render addon stored in scene"""
    
    # Main properties
    api_key: StringProperty(
        name="API Key",
        description="Google Gemini API Key",
        default="",
        subtype='PASSWORD',
        update=lambda self, context: sync_api_key(self, context)
    )
    
    model_name: EnumProperty(
        name="API Model",
        description="Choose Gemini model",
        items=[
            ('gemini-2.5-flash-image', "Gemini 2.5 Flash", "User's preferred model ID"),
            ('gemini-3-pro-image-preview', "Gemini 3 Pro", "Best for images, very high rate limiting"),
        ],
        default='gemini-2.5-flash-image'
    )
    
    prompt: StringProperty(
        name="Prompt", 
        description="Describe how you want the depth map to be transformed",
        default="Make this photorealistic with detailed materials and proper lighting",
        maxlen=1000,
    )
    
    render_history: CollectionProperty(
        type=GeminiRenderHistoryItem,
        name="Render History"
    )
    
    history_index: IntProperty(
        name="History Index",
        default=-1,
    )
    

    
    resolution: EnumProperty(
        name="Resolution",
        description="Choose render resolution",
        items=[
            ('1024', "1k (1024x1024)", "Standard square resolution"),
            ('2048', "2k (2048x2048)", "High resolution"),
            ('4096', "4k (4096x4096)", "Ultra high resolution"),
        ],
        default='1024',
    )
    

    
    # I style Reference Image (optional)
    use_style_reference: BoolProperty(
        name="Use Style Reference",
        description="Use a reference image to maintain style/materials",
        default=False
    )
    
    style_reference_image: PointerProperty(
        type=bpy.types.Image,
        name="Style Reference",
        description="Reference image to maintain similar style/materials/lighting"
    )
    
    # UI state
    show_settings: BoolProperty(
        name="Show Settings",
        description="Show advanced settings",
        default=False,
    )
    
    show_auth: BoolProperty(
        name="Show Authentication",
        description="Show authentication panel",
        default=True,
    )
    
    # Status
    status_text: StringProperty(
        name="Status",
        description="Current operation status",
        default="Ready to render",
    )
    
    is_rendering: BoolProperty(
        name="Is Rendering",
        description="Whether AI render is in progress",
        default=False,
    )
    # Projection settings
    input_source: EnumProperty(
        name="Input Source",
        description="What to capture and send to AI (only used when Projection Source = AI)",
        items=[
            ('COLOR', "Viewport Screenshot", "Capture current viewport color"),
            ('DEPTH', "Depth Map", "Capture depth information"),
        ],
        default='COLOR'
    )
    
    projection_source: EnumProperty(
        name="Projection Source",
        description="Where the projected texture comes from",
        items=[
            ('AI', "AI Generated", "Generate texture using AI from captured input"),
            ('IMAGE', "Custom Texture", "Use a selected image directly"),
            ('VIEW', "Viewport Capture", "Directly project the current viewport screenshot"),
            ('GRID', "Grid", "Use wireframe grid for alignment testing"),
        ],
        default='AI'
    )
    
    projection_image: PointerProperty(
        type=bpy.types.Image,
        name="Projection Image",
        description="Custom image to project onto the mesh"
    )
    
    # Projection settings
    projection_bake: BoolProperty(
        name="Bake to UVs",
        description="Bake the projected texture back to the object's original UV layout",
        default=True,
    )
    
    # Mask Repair Mode settings
    mask_repair_mode: BoolProperty(
        name="Mask Repair Mode",
        description="Use mask-based incremental texture repair - selected faces are masked and AI repairs only that region",
        default=False,
    )
    mask_color: FloatVectorProperty(
        name="Mask Color",
        description="Color used for mask rendering (default: red)",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 0.0, 0.0, 1.0),
    )
    
    # Debug Mode
    debug_mode: BoolProperty(
        name="Debug Mode",
        description="Enable manual step-by-step debugging and export debug images to 'textures' folder",
        default=False
    )
    
    debug_step: IntProperty(
        name="Debug Step",
        default=0
    )

class BANANA_PT_render_panel(Panel):
    """Main Texture Projector Panel"""
    bl_label = "Gemini Texture Projector"
    bl_idname = "BANANA_PT_render_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gemini"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.gemini_render
        
        # Auto-sync API key from preferences if scene key is empty
        if not props.api_key:
            try:
                # Use current package name for metadata
                package_name = __package__ if __package__ else "nano_banana_render"
                addon_prefs = context.preferences.addons.get(package_name)
                if addon_prefs and hasattr(addon_prefs.preferences, 'api_key') and addon_prefs.preferences.api_key:
                    props.api_key = addon_prefs.preferences.api_key
            except:
                pass
        
        # Authentication (collapsible)
        box = layout.box()
        row = box.row(align=True)
        row.prop(props, "show_auth", 
                text="🔑 Authentication" if not props.show_auth else "🔑 Hide Authentication",
                toggle=True, icon='TRIA_DOWN' if props.show_auth else 'TRIA_RIGHT')
        
        if props.show_auth:
            box.prop(props, "api_key", text="")
            
            # Model selection
            box.label(text="Model Selection:", icon='NODE_SEL')
            box.prop(props, "model_name", text="")
            
            if not props.api_key.strip():
                box.label(text="Enter API key", icon='ERROR')
        
        # Settings toggle
        row = layout.row()
        row.prop(props, "show_settings", 
                text="Settings" if not props.show_settings else "Hide Settings",
                toggle=True, icon='PREFERENCES')
        
        if props.show_settings:
            box = layout.box()
            
            # Debug options
            box.label(text="Debug:", icon='TOOL_SETTINGS')
            box.prop(props, "debug_mode", text="Manual Debug Mode")
            
            if props.debug_mode:
                debug_box = box.box()
                step_text = f"Next Debug Step ({props.debug_step})"
                debug_box.operator("gemini.debug_next", text=step_text, icon='PLAY')
                debug_box.label(text=f"Step {props.debug_step}: Check logs", icon='CONSOLE')
        
        # Texture Projection Section
        layout.separator()
        box = layout.box()
        box.label(text="Texture Projection", icon='MOD_UVPROJECT')
        
        # Projection Source (what to project)
        proj_box = box.box()
        proj_box.label(text="Projection Source:", icon='MOD_UVPROJECT')
        proj_box.prop(props, "projection_source", text="")
        
        # Show options based on projection source
        if props.projection_source == 'AI':
            # AI Input type
            input_row = proj_box.row()
            input_row.label(text="Capture:")
            input_row.prop(props, "input_source", text="")
            
            # Prompt
            proj_box.label(text="Prompt:", icon='TEXT')
            proj_box.prop(props, "prompt", text="")
            
            # Style Reference
            style_row = proj_box.row()
            style_row.prop(props, "use_style_reference", text="Use Style Reference")
            if props.use_style_reference:
                proj_box.prop(props, "style_reference_image", text="Reference")
                load_row = proj_box.row()
                load_row.operator("gemini.load_image_as_reference", text="Load Image", icon='FILEBROWSER')
        elif props.projection_source == 'IMAGE':
            proj_box.prop(props, "projection_image", text="Image")
            if not props.projection_image:
                proj_box.label(text="Select an image to project", icon='ERROR')
        
        # Main Project Button
        proj_col = box.column(align=True)
        proj_col.scale_y = 2.0
        
        if props.is_rendering:
            proj_col.enabled = False
            proj_col.operator("gemini.texture_projection", text="Processing...", icon='RENDER_ANIMATION')
        else:
            btn_text = "Project Texture"
            if props.projection_source == 'AI':
                btn_text = "AI Texture Projection"
            elif props.projection_source == 'GRID':
                btn_text = "Grid Projection (Test)"
            proj_col.operator("gemini.texture_projection", text=btn_text, icon='MOD_UVPROJECT')
        
        # Options
        row = box.row()
        row.prop(props, "projection_bake", text="Bake to Original UVs")
        
        # Mask Repair Mode
        row = box.row()
        row.prop(props, "mask_repair_mode", text="Mask Repair Mode")
        if props.mask_repair_mode:
            color_row = box.row()
            color_row.prop(props, "mask_color", text="Mask Color")
        
        # Validation feedback
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            box.label(text="Select a mesh object", icon='ERROR')
        elif obj.mode != 'EDIT':
            box.label(text="Enter Edit Mode to project", icon='EDITMODE_HLT')
        else:
             box.label(text="Select faces to project onto", icon='FACESEL')
        
        # Status
        layout.separator()
        box = layout.box()
        status_icon = 'INFO' if not props.is_rendering else 'TIME'
        box.label(text=props.status_text, icon=status_icon)
        



class BANANA_PT_history_panel(Panel):
    """Visual gallery render history panel"""
    bl_label = "Projection Gallery" 
    bl_idname = "BANANA_PT_history_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gemini"
    bl_parent_id = "BANANA_PT_render_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.gemini_render
        
        if len(props.render_history) == 0:
            box = layout.box()
            box.label(text=" No renders yet", icon='INFO')
            box.label(text="Generate AI renders to see gallery")
            return
        
        # I gallery header
        header_row = layout.row(align=True)
        header_row.label(text=f"Gallery ({len(props.render_history)} renders)", icon='IMAGE_DATA')
        
        layout.separator()
        
        # I gallery grid - newest first
        for i, item in enumerate(reversed(props.render_history)):
            actual_index = len(props.render_history) - 1 - i
            render_number = len(props.render_history) - i  # I numbered from newest
            
            # I compact render card with proper structure
            card = layout.box()
            card.scale_y = 0.9
            
            # I row 1: Date and render number (compact)
            date_row = card.row()
            date_row.scale_y = 0.6
            date_row.label(text=f"# {render_number} • {item.timestamp}", icon='TIME')
            
            # I row 2: Buttons - View button (big) + Gear button (small, right)
            btn_row = card.row(align=True)
            btn_row.scale_y = 1.2
            
            # I view photo button (takes most space)
            view_btn = btn_row.operator("gemini.open_history_image", text=" View Photo", icon='ZOOM_IN')
            view_btn.history_index = actual_index
            
            # I gear button (small, just icon, right side)
            gear_btn = btn_row.operator("gemini.history_context_menu", text="", icon='PREFERENCES', emboss=False)
            gear_btn.history_index = actual_index
            
            # I row 3: Prompt (styled exactly like style reference help text)
            prompt_preview = item.prompt[:70] + "..." if len(item.prompt) > 70 else item.prompt
            help_box = card.box() 
            help_box.scale_y = 0.7
            help_box.label(text=prompt_preview, icon='TEXT')
            
            # I minimal separator between items
            if i < len(props.render_history) - 1:
                layout.separator()






def sync_api_key(self, context):
    """Sync API key between scene properties and addon preferences"""
    try:
        import bpy
        

        package_name = __package__ if __package__ else "nano_banana_render"
        addon_prefs = context.preferences.addons.get(package_name)
        
        if addon_prefs and hasattr(addon_prefs.preferences, 'api_key'):
            # I sync scene -> preferences
            if self.api_key and self.api_key != addon_prefs.preferences.api_key:
                addon_prefs.preferences.api_key = self.api_key
                print(f" API key synced to preferences")
            # I sync preferences -> scene (if scene is empty)
            elif not self.api_key and addon_prefs.preferences.api_key:
                self.api_key = addon_prefs.preferences.api_key
                print(f" API key synced from preferences")
        
    except Exception as e:
        print(f" Failed to sync API key: {e}")
