import bpy
import bmesh
import gpu
import gpu.texture
from gpu_extras.batch import batch_for_shader
import mathutils
import numpy as np


def finalize_object_materials(target_obj, target_img, dest_uv_name, search_img=None, node_name=None):
    """
    Post-bake material finalization: Updates texture nodes to use baked image and correct UV map.
    
    Args:
        target_obj: The object whose materials should be finalized
        target_img: The baked image to set on matching texture nodes
        dest_uv_name: The UV map name to set on UV Map nodes
        search_img: Optional image to also match against (e.g., preview image)
        node_name: Optional specific node name to match
    """
    if not target_obj or not target_obj.data or not hasattr(target_obj.data, 'materials'):
        return
        
    print(f"  Finalizing materials for {target_obj.name} (Target UV: {dest_uv_name})")
    for slot in target_obj.material_slots:
        if slot.material and slot.material.use_nodes:
            m_nodes = slot.material.node_tree.nodes
            m_links = slot.material.node_tree.links
            
            for node in m_nodes:
                is_match = False
                if node.type == 'TEX_IMAGE':
                    # Match by direct reference
                    if node.image and (node.image == target_img or (search_img and node.image == search_img)):
                        is_match = True
                    # Match by name
                    elif node.name == node_name:
                        is_match = True
                    # Match by image name (fuzzy backup)
                    elif target_img and node.image and node.image.name == target_img.name:
                        is_match = True
                
                if is_match:
                    # 1. Update image to the final baked result
                    if target_img:
                        node.image = target_img
                    
                    # 2. Fix UV mapping node
                    uv_node = None
                    if node.inputs['Vector'].is_linked:
                        from_node = node.inputs['Vector'].links[0].from_node
                        if from_node.type == 'UV_MAP':
                            uv_node = from_node
                    
                    if not uv_node:
                        print(f"   Linking new UV Map node to {node.name} in {slot.material.name}")
                        # Clear existing links to vector
                        for l in node.inputs['Vector'].links:
                            m_links.remove(l)
                        uv_node = m_nodes.new('ShaderNodeUVMap')
                        m_links.new(uv_node.outputs['UV'], node.inputs['Vector'])
                    
                    # 3. Set the UV map
                    if uv_node:
                        uv_node.uv_map = dest_uv_name
                        print(f"   Material '{slot.material.name}' node '{node.name}' finalized (UV: {dest_uv_name})")

    # Restore active render layer on the mesh itself
    if dest_uv_name in target_obj.data.uv_layers:
        target_obj.data.uv_layers.active = target_obj.data.uv_layers[dest_uv_name]
        target_obj.data.uv_layers[dest_uv_name].active_render = True


def perform_projection_bake(context, obj, texture_node_name, target_image, src_uv_name, dest_uv_name, 
                            margin=16, use_clear=True, is_mask_repair=False, 
                            original_obj=None, search_img=None):
    """
    Unified projection bake that handles both normal and mask repair modes.
    
    Args:
        context: Blender context
        obj: Object to bake from (source)
        texture_node_name: Name of the image texture node containing the source
        target_image: Image to bake onto
        src_uv_name: Source UV map name (e.g., "Projected UVs")
        dest_uv_name: Destination UV map name (e.g., "UVMap")
        margin: Bake margin in pixels (default 16)
        use_clear: If True, clears target before baking (default True)
        is_mask_repair: If True, uses repair mode settings (default False)
        original_obj: For mask repair mode, the original object to finalize materials on
        search_img: Optional image to also match against during finalization
        
    Returns:
        True if bake succeeded, False otherwise
    """
    try:
        # Adjust settings for mask repair mode
        if is_mask_repair:
            margin = 1
            use_clear = False
            print(f"[BAKE] Mask Repair Mode: {obj.name} -> {target_image.name}")
        else:
            print(f"[BAKE] Normal Mode: {obj.name} -> {target_image.name}")
        
        # Perform the bake
        bake(
            context=context,
            obj=obj,
            texture_node_name=texture_node_name,
            target_image=target_image,
            src_uv_name=src_uv_name,
            margin=margin,
            use_clear=use_clear
        )
        
        # Post-bake material finalization
        finalize_target = original_obj if is_mask_repair and original_obj else obj
        finalize_object_materials(
            target_obj=finalize_target,
            target_img=target_image,
            dest_uv_name=dest_uv_name,
            search_img=search_img,
            node_name=texture_node_name
        )
        
        print(f"✓ Bake completed for {obj.name}")
        return True
        
    except Exception as e:
        print(f"✗ Bake failed for {obj.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_projection(context):
    """Validate if projection can be performed. Ensures selected meshes are in Edit Mode with selected faces."""
    selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
    
    if not selected_meshes:
        raise Exception("No mesh objects selected. Select at least one mesh to project onto.")
    
    import bmesh
    has_selection = False
    for obj in selected_meshes:

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

def bake(context, obj, texture_node_name, target_image, src_uv_name="Projected UVs", margin=16, use_clear=True):
    """
    Bake projected texture to destination texture using Blender's native baking system.
    Handles margins (dilation) natively and works across all material slots.
    
    Args:
        use_clear: If True, clears target image before baking. Set to False for incremental/repair baking.
    """
    scene = context.scene
    

    original_engine = scene.render.engine
    original_active = context.view_layer.objects.active
    original_mode = obj.mode
    original_hide_render = obj.hide_render # I phase 6
    original_dither = scene.render.dither_intensity
    original_view_transform = scene.view_settings.view_transform
    original_use_selected_to_active = scene.render.bake.use_selected_to_active
    
    # I ensure we are in OBJECT mode for baking
    if obj.mode != 'EDIT':
        if obj.mode != 'OBJECT':
             bpy.ops.object.mode_set(mode='OBJECT')
    else:

        bpy.ops.object.mode_set(mode='OBJECT')

    # MANDATORY: Enable render visibility for the duration of bake
    obj.hide_render = False
    
    try:
        # 1. Setup Cycles for baking
        scene.render.engine = 'CYCLES'
        if hasattr(scene.cycles, "device"):
            scene.cycles.device = 'GPU' if context.preferences.addons.get('cycles') and context.preferences.addons['cycles'].preferences.compute_device_type != 'NONE' else 'CPU'
        
        # I optimize for EMIT bake (only need 1 sample)
        scene.cycles.samples = 1
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        
        # QUALITY OPTIMIZATION: Disable dither and force Standard view transform
        scene.render.dither_intensity = 0.0
        scene.view_settings.view_transform = 'Standard'
        
        # 2. Prepare selection
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        
        # 3. Setup ALL Materials for baking
        # I we must iterate over all materials assigned to the object because Blender's
        # bake operator will check every material slot for an active image node.
        mats_data = [] # I list of (material, original_links, temp_emit, target_node, temp_uv_node, original_vector_links)
        
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
            
            # I only setup bypass if we find the source texture node in this material
            src_node = nodes.get(texture_node_name)
            if surface_output and src_node:
                # 1. Ensure src_node has the AI image (it should already, but let's be safe)
                # src_node.image = ... # We rely on threading_utils to have set this
                
                # 2. Setup Source UV Mapping: Force use of src_uv_name (Projected UVs)
                # I store existing links to Vector input
                for link in src_node.inputs['Vector'].links:
                    original_vector_links.append((link.from_socket, link.to_socket))
                    links.remove(link)
                

                temp_uv_node = nodes.new('ShaderNodeUVMap')
                temp_uv_node.uv_map = src_uv_name
                links.new(temp_uv_node.outputs['UV'], src_node.inputs['Vector'])
                
                # 3. Setup Emission Bypass: Store and remove existing surface links
                for link in surface_output.inputs['Surface'].links:
                    original_links.append((link.from_socket, link.to_socket))
                    links.remove(link)
                

                temp_emit = nodes.new('ShaderNodeEmission')
                temp_emit.inputs['Strength'].default_value = 1.0
                links.new(src_node.outputs['Color'], temp_emit.inputs['Color'])
                links.new(temp_emit.outputs['Emission'], surface_output.inputs['Surface'])
            
            mats_data.append((mat, original_links, temp_emit, target_node, temp_uv_node, original_vector_links))
            print(f"Prepared material {mat.name} for bake (Source UV: {src_uv_name})")

        if not mats_data:
            raise Exception(f"Object {obj.name} has no valid nodal materials for baking")

        # 4. Perform Bake
        print(f"🔥 Starting EMIT bake for {obj.name} (Margin: {margin}, use_clear: {use_clear})...")
        scene.render.bake.use_clear = use_clear
        scene.render.bake.margin = margin
        scene.render.bake.target = 'IMAGE_TEXTURES'
        
        scene.render.bake.use_selected_to_active = False
        
        bpy.ops.object.bake(type='EMIT')
        
        # 5. Cleanup and Restore
        for mat, original_links, temp_emit, target_node, temp_uv_node, original_vector_links in mats_data:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # I remove temp nodes
            if temp_emit:
                nodes.remove(temp_emit)
            if temp_uv_node:
                nodes.remove(temp_uv_node)
            if target_node.name in nodes:
                nodes.remove(target_node)
                

            surface_output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if surface_output:
                for from_sock, to_sock in original_links:
                    links.new(from_sock, to_sock)
            

            src_node = nodes.get(texture_node_name)
            if src_node:
                for from_sock, to_sock in original_vector_links:
                    links.new(from_sock, to_sock)
        
        print(f" Bake completed for {obj.name}")
        
    except Exception as e:
        print(f" Bake error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:

        scene.render.engine = original_engine
        if original_active:
            context.view_layer.objects.active = original_active
        if original_mode != obj.mode:
            try: bpy.ops.object.mode_set(mode=original_mode)
            except: pass
        
        # I phase 6 Restore: Render Visibility
        obj.hide_render = original_hide_render
        
        # I quality Restore
        scene.render.dither_intensity = original_dither
        scene.view_settings.view_transform = original_view_transform
        scene.render.bake.use_selected_to_active = original_use_selected_to_active


def setup_projection_material(material_name="Gemini_Projection_Material", use_image=None, emission_color=None):
    """
    Setup the projection material with emission and optional image texture.
    
    Args:
        material_name: Name of material to create/get
        use_image: Optional bpy.types.Image to assign to texture node
        emission_color: Optional (r,g,b,a) for solid color emission (overrides image if active).
                       If None, defaults to Image texture or White 1.0 strength.
                       
    Returns:
        (material, image_node, uv_map_node)
    """
    material = bpy.data.materials.get(material_name)
    if not material:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
    
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    
    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (400, 0)
    
    emit_node = nodes.new("ShaderNodeEmission")
    emit_node.location = (200, 0)
    links.new(emit_node.outputs['Emission'], output_node.inputs['Surface'])
    
    image_node = None
    uv_map_node = None
    
    if emission_color:
        # Solid Color Mode (e.g. Debug Magenta)
        emit_node.inputs['Color'].default_value = emission_color
        emit_node.inputs['Strength'].default_value = 1.0
    else:
        # Texture Mode
        emit_node.inputs['Strength'].default_value = 1.0
        
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.name = "Gemini_Image_Node"
        image_node.location = (0, 0)
        if use_image:
            image_node.image = use_image
            
        uv_map_node = nodes.new("ShaderNodeUVMap")
        uv_map_node.name = "Gemini_UV_Map"
        uv_map_node.uv_map = "Projected UVs"
        uv_map_node.location = (-200, 0)
        
        links.new(uv_map_node.outputs['UV'], image_node.inputs['Vector'])
        links.new(image_node.outputs['Color'], emit_node.inputs['Color'])
        
    return material, image_node, uv_map_node


def project_uvs_on_mesh(bm, obj, scene, camera=None, region=None, rv3d=None, v_width=1024, v_height=1024, uv_layer_name="Projected UVs"):
    """
    Project UVs for selected faces in BMesh based on camera or viewport view.
    
    Args:
        bm: BMesh object (from edit mesh)
        obj: The object being projected onto
        scene: Blender scene
        camera: Camera object (if camera projection)
        region: View port region (if viewport projection)
        rv3d: Region 3D View (if viewport projection)
        v_width: Viewport/Capture width
        v_height: Viewport/Capture height
        uv_layer_name: Name of UV layer to write to
        
    Returns:
        True if faces were updated
    """
    from bpy_extras import view3d_utils, object_utils
    
    uv_layer = bm.loops.layers.uv.get(uv_layer_name) or bm.loops.layers.uv.new(uv_layer_name)
    
    face_updated = False
    for face in bm.faces:
        if face.select:
            face_updated = True
            for loop in face.loops:
                world_co = obj.matrix_world @ loop.vert.co
                
                if camera:
                    screen_co = object_utils.world_to_camera_view(scene, camera, world_co)
                    uv = (screen_co[0], screen_co[1])
                elif region and rv3d:
                    screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, world_co)
                    if screen_co:
                        uv = (screen_co[0] / v_width, screen_co[1] / v_height)
                    else:
                        uv = (0, 0)
                else:
                    uv = (0, 0)
                    
def setup_mask_repair_meshes(context, props, registry):
    """
    Setup temporary mask meshes for selected objects in Edit Mode.
    Returns mask_repair_data dictionary or None.
    """
    try:
        print("Mask Repair Mode: Creating mask overlay meshes for selected objects...")
        
        mask_repair_data = {
            'original_textures': {},
            'original_object_names': [],
            'temp_to_original': {}
        }
        
        # Process all selected mesh objects
        for obj in context.selected_objects:
            if obj.type != 'MESH': continue
            if obj.mode != 'EDIT':
                print(f" Skipping {obj.name} - not in Edit Mode")
                continue
                
            obj_name = obj.name
            print(f"Mask Repair Mode: Processing '{obj_name}'")
            
            import bmesh
            bm = bmesh.from_edit_mesh(obj.data)
            
            # Identify selected faces
            selected_faces = [f for f in bm.faces if f.select]
            if not selected_faces:
                print(f" Mask Repair Mode: No faces selected on '{obj_name}', skipping")
                continue
                
            print(f" Mask Repair Mode: {len(selected_faces)} faces selected on '{obj_name}'")
            
            # FACE-AWARE TEXTURE TARGETING
            target_mat_index = selected_faces[0].material_index
            original_texture = None
            
            if target_mat_index < len(obj.material_slots):
                slot = obj.material_slots[target_mat_index]
                if slot.material and slot.material.use_nodes:
                    for node in slot.material.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            original_texture = node.image
                            break
            
            if not original_texture:
                # Fallback to first found texture
                for slot in obj.material_slots:
                    if slot.material and slot.material.use_nodes:
                        for node in slot.material.node_tree.nodes:
                            if node.type == 'TEX_IMAGE' and node.image:
                                original_texture = node.image
                                break
                    if original_texture: break
            
            if not original_texture:
                print(f" ⚠ Warning: Object '{obj_name}' has no textures. Skipping mask for this object.")
                continue

            bm_copy = bm.copy()
            faces_to_delete = [f for f in bm_copy.faces if not f.select]
            bmesh.ops.delete(bm_copy, geom=faces_to_delete, context='FACES')
            
            temp_mesh = bpy.data.meshes.new(f"{obj_name}_MaskTemp_Data")
            bm_copy.to_mesh(temp_mesh)
            bm_copy.free()
            
            temp_obj = bpy.data.objects.new(f"{obj_name}_MaskTemp", temp_mesh)
            temp_obj.matrix_world = obj.matrix_world.copy()
            
            context.collection.objects.link(temp_obj)
            registry['temp_objects'].append(temp_obj.name)
            
            mask_repair_data['original_textures'][obj_name] = original_texture.name
            mask_repair_data['original_object_names'].append(obj_name)
            mask_repair_data['temp_to_original'][temp_obj.name] = obj_name
            
            # ANTI-ZFIGHTING (Solidify)
            solidify_mod = temp_obj.modifiers.new(name="Gemini_Solidify", type='SOLIDIFY')
            solidify_mod.thickness = 0.002
            solidify_mod.offset = 0
            solidify_mod.use_rim = True
            
            # Setup Pure Color Emission Material for Mask
            mask_mat_name = f"Gemini_Mask_{obj_name}_Material_Temp"
            mask_mat = bpy.data.materials.get(mask_mat_name)
            if mask_mat:
                bpy.data.materials.remove(mask_mat)
            mask_mat = bpy.data.materials.new(name=mask_mat_name)
            mask_mat.use_nodes = True
            mask_nodes = mask_mat.node_tree.nodes
            mask_links = mask_mat.node_tree.links
            mask_nodes.clear()
            
            registry['temp_materials'].append(mask_mat_name)

            mask_output = mask_nodes.new("ShaderNodeOutputMaterial")
            mask_output.location = (300, 0)
            mask_emit = mask_nodes.new("ShaderNodeEmission")
            mask_emit.location = (100, 0)
            mask_emit.inputs['Color'].default_value = (props.mask_color[0], props.mask_color[1], props.mask_color[2], 1.0)
            mask_emit.inputs['Strength'].default_value = 1.0
            mask_links.new(mask_emit.outputs['Emission'], mask_output.inputs['Surface'])
            
            temp_obj.data.materials.clear()
            temp_obj.data.materials.append(mask_mat)
        
        if not mask_repair_data['original_object_names']:
            print(" ⚠ Mask Repair Warning: No valid objects found for mask.")
            return None
            
        print(f"Mask Repair Mode: Created {len(mask_repair_data['original_object_names'])} temp meshes.")
        return mask_repair_data
        
    except Exception as e:
        print(f" ❌ setup_mask_repair_meshes failed: {e}")
        import traceback
        traceback.print_exc()
        return None

