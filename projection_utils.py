import bpy
import bmesh
import gpu
import gpu.texture
from gpu_extras.batch import batch_for_shader
import mathutils
import numpy as np

def validate_projection(context):
    """Validate if projection can be performed. Ensures selected meshes are in Edit Mode with selected faces."""
    selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
    
    if not selected_meshes:
        raise Exception("No mesh objects selected. Select at least one mesh to project onto.")
    
    import bmesh
    has_selection = False
    for obj in selected_meshes:
        # Check if object is in Edit Mode
        if obj.mode != 'EDIT':
            continue
            
        bm = bmesh.from_edit_mesh(obj.data)
        if not bm:
            continue
            
        for f in bm.faces:
            if f.select:
                has_selection = True
                break
        if has_selection:
            break
    
    if not has_selection:
        raise Exception("No faces selected in Edit Mode. Please enter Edit Mode for your meshes and select projection faces.")
    return None

def bake(context, obj, texture_node_name, target_image, src_uv_name="Projected UVs", margin=16):
    """
    Bake projected texture to destination texture using Blender's native baking system.
    Handles margins (dilation) natively and works across all material slots.
    """
    scene = context.scene
    
    # Store original settings
    original_engine = scene.render.engine
    original_active = context.view_layer.objects.active
    original_mode = obj.mode
    
    # Ensure we are in OBJECT mode for baking
    if obj.mode != 'EDIT':
        if obj.mode != 'OBJECT':
             bpy.ops.object.mode_set(mode='OBJECT')
    else:
        # If in edit mode, toggle out
        bpy.ops.object.mode_set(mode='OBJECT')
    
    try:
        # 1. Setup Cycles for baking
        scene.render.engine = 'CYCLES'
        if hasattr(scene.cycles, "device"):
            scene.cycles.device = 'GPU' if context.preferences.addons.get('cycles') and context.preferences.addons['cycles'].preferences.compute_device_type != 'NONE' else 'CPU'
        
        # Optimize for EMIT bake (only need 1 sample)
        scene.cycles.samples = 1
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        
        # 2. Prepare selection
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        
        # 3. Setup ALL Materials for baking
        # We must iterate over all materials assigned to the object because Blender's
        # bake operator will check every material slot for an active image node.
        mats_data = [] # List of (material, original_links, temp_emit, target_node, temp_uv_node, original_vector_links)
        
        unique_materials = {slot.material for slot in obj.material_slots if slot.material and slot.material.use_nodes}
        
        for mat in unique_materials:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # A. Create/Get target image node
            target_node = nodes.get("Gemini_Bake_Target") or nodes.new('ShaderNodeTexImage')
            target_node.name = "Gemini_Bake_Target"
            target_node.image = target_image
            
            # MANDATORY: Set as active AND selected
            for n in nodes: n.select = False
            target_node.select = True
            nodes.active = target_node
            
            # B. Setup PURE COLOR TRANSFER (Emission Bypass)
            surface_output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            temp_emit = None
            temp_uv_node = None
            original_links = []
            original_vector_links = []
            
            # Only setup bypass if we find the source texture node in this material
            src_node = nodes.get(texture_node_name)
            if surface_output and src_node:
                # 1. Ensure src_node has the AI image (it should already, but let's be safe)
                # src_node.image = ... # We rely on threading_utils to have set this
                
                # 2. Setup Source UV Mapping: Force use of src_uv_name (Projected UVs)
                # Store existing links to Vector input
                for link in src_node.inputs['Vector'].links:
                    original_vector_links.append((link.from_socket, link.to_socket))
                    links.remove(link)
                
                # Create temporary UV Map node set to Projected UVs
                temp_uv_node = nodes.new('ShaderNodeUVMap')
                temp_uv_node.uv_map = src_uv_name
                links.new(temp_uv_node.outputs['UV'], src_node.inputs['Vector'])
                
                # 3. Setup Emission Bypass: Store and remove existing surface links
                for link in surface_output.inputs['Surface'].links:
                    original_links.append((link.from_socket, link.to_socket))
                    links.remove(link)
                
                # Create temp emission
                temp_emit = nodes.new('ShaderNodeEmission')
                temp_emit.inputs['Strength'].default_value = 1.0
                links.new(src_node.outputs['Color'], temp_emit.inputs['Color'])
                links.new(temp_emit.outputs['Emission'], surface_output.inputs['Surface'])
            
            mats_data.append((mat, original_links, temp_emit, target_node, temp_uv_node, original_vector_links))
            print(f"[GEMINI] Prepared material {mat.name} for bake (Source UV: {src_uv_name})")

        if not mats_data:
            raise Exception(f"Object {obj.name} has no valid nodal materials for baking")

        # 4. Perform Bake
        print(f"🔥 [GEMINI] Starting EMIT bake for {obj.name} (Margin: {margin})...")
        scene.render.bake.use_clear = True
        scene.render.bake.margin = margin
        scene.render.bake.target = 'IMAGE_TEXTURES'
        
        # Execute bake
        bpy.ops.object.bake(type='EMIT')
        
        # 5. Cleanup and Restore
        for mat, original_links, temp_emit, target_node, temp_uv_node, original_vector_links in mats_data:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # Remove temp nodes
            if temp_emit:
                nodes.remove(temp_emit)
            if temp_uv_node:
                nodes.remove(temp_uv_node)
            if target_node.name in nodes:
                nodes.remove(target_node)
                
            # Restore original links
            surface_output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if surface_output:
                for from_sock, to_sock in original_links:
                    links.new(from_sock, to_sock)
            
            # Restore original vector links
            src_node = nodes.get(texture_node_name)
            if src_node:
                for from_sock, to_sock in original_vector_links:
                    links.new(from_sock, to_sock)
        
        print(f"✅ [GEMINI] Bake completed for {obj.name}")
        
    except Exception as e:
        print(f"❌ [GEMINI] Bake error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Restore settings
        scene.render.engine = original_engine
        if original_active:
            context.view_layer.objects.active = original_active
        if original_mode != obj.mode:
            try: bpy.ops.object.mode_set(mode=original_mode)
            except: pass

